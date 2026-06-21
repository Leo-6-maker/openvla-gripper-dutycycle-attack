#!/usr/bin/env python3
"""MIG3B: Paired FP32 closed-loop clean canary — A800 + 2080Ti.
Fixed contract: official preprocessing, Lane O execution, Lane M shadow audit."""
import os, json, hashlib, time, csv, argparse, sys, numpy as np
from PIL import Image
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio


def preprocess_libero_agentview(raw_agentview):
    """Official OpenVLA preprocessing: 180deg rotate, RGB, LANCZOS 224x224, center crop."""
    img = Image.fromarray(raw_agentview)
    img = img.rotate(180)  # agentview rotation
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)  # LANCZOS (not bicubic)
    return img  # processor applies center_crop internally


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def lane_m_mirror(model, input_ids, pixel_values, unnorm_key):
    """Exact mirror of predict_action from modeling_prismatic.py.
    Source SHA: c10a6d1fbb414152bb3fda9d8acd3d1a9df7b5b6f94b2a8a69c73c9adcb1b8b2"""
    ids = input_ids.clone()
    if ids[0, -1].item() != 29871:
        ids = torch.cat((ids, torch.tensor([[29871]], dtype=ids.dtype, device=ids.device)), dim=1)

    gen = model.generate(input_ids=ids, pixel_values=pixel_values,
                         max_new_tokens=model.get_action_dim(unnorm_key),
                         do_sample=False, pad_token_id=model.pad_token_id)

    action_dim = model.get_action_dim(unnorm_key)
    tokens = gen[0, -action_dim:].cpu().numpy()
    disc = model.vocab_size - tokens
    disc = np.clip(disc - 1, 0, model.bin_centers.shape[0] - 1)
    norm = model.bin_centers[disc]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    act = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
    return act, tokens, norm


def run_episode(model, proc, task_suite, task_idx, init_idx, output_dir, label,
                max_steps=220, wait_steps=10, seed=42):
    """Run one closed-loop episode. Returns result dict."""
    action_dim = model.get_action_dim("libero_spatial")
    stats = model.get_action_stats("libero_spatial")
    DEV = next(model.parameters()).device

    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    # BDDL hash
    with open(bddl_path) as f:
        bddl_sha = hashlib.sha256(f.read().encode()).hexdigest()
    init_state_sha = hashlib.sha256(init_states[init_idx].tobytes()).hexdigest()

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_heights=256, camera_widths=256,
        has_renderer=True, has_offscreen_renderer=True,
        render_gpu_device_id=0, use_camera_obs=True,
    )
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[init_idx])

    wait_action = [0, 0, 0, 0, 0, 0, -1]
    for w in range(wait_steps):
        obs, rew, done, info = env.step(wait_action)

    frames_raw = []
    frames_input = []
    step_trace = []
    done = False
    success = False
    termination = "timeout"
    error = None

    t0 = time.time()
    lane_o_m_match = True

    for step in range(max_steps):
        raw_img = obs["agentview_image"]
        raw_sha = sha256_bytes(np.array(raw_img).tobytes()) if raw_img.ndim == 3 else sha256_bytes(raw_img.tobytes())

        # Official preprocessing
        processed_img = preprocess_libero_agentview(raw_img)
        input_sha = sha256_bytes(np.array(processed_img).tobytes())

        prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()

        inputs = proc(prompt, processed_img, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=torch.float32, device=DEV)
        pixel_sha = hashlib.sha256(px.float().cpu().numpy().tobytes()).hexdigest()
        ids_sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()

        # Lane O: predict_action (drives environment)
        t_inf = time.time()
        result = model.predict_action(
            input_ids=ids, pixel_values=px,
            unnorm_key="libero_spatial", do_sample=False,
        )
        inf_latency = time.time() - t_inf
        act_o = np.array(result).flatten()

        # Lane M shadow audit (key steps only: step 0, first close, max 2 others)
        do_shadow = (step == 0 or step == max_steps - 1)
        # Also shadow on gripper flips
        if step > 0:
            prev_grip = float(step_trace[-1]["env_gripper"])
            if abs(act_o[6] - 0.0) < 0.01 and abs(prev_grip) < 0.01:
                pass  # no flip
            elif (act_o[6] > 0.5 and prev_grip < -0.5) or (act_o[6] < 0.5 and prev_grip > 0.5):
                do_shadow = True  # gripper flip detected

        shadow_act = None
        shadow_tokens = None
        if do_shadow:
            shadow_act, shadow_tokens, shadow_norm = lane_m_mirror(
                model, ids.clone(), px.clone(), "libero_spatial"
            )
            if not np.allclose(act_o, shadow_act):
                lane_o_m_match = False
                print(f"  WARNING: Lane O != Lane M at step {step}!")
                print(f"    O: {[round(x,6) for x in act_o.tolist()]}")
                print(f"    M: {[round(x,6) for x in shadow_act.tolist()]}")

        # Gripper post-process
        raw_grip = act_o[6]
        norm_grip = (raw_grip * 2) - 1
        bin_grip = 1.0 if norm_grip >= 0 else -1.0
        env_grip = -bin_grip

        env_action = np.zeros(7)
        env_action[:6] = act_o[:6]
        env_action[6] = env_grip

        # Step environment
        try:
            obs, rew, done, info = env.step(env_action.tolist())
        except Exception as e:
            error = str(e)
            termination = "runtime_error"
            break

        # EEF position
        eef_pos = obs.get("robot0_eef_pos", [0.0]*3)

        trace_row = {
            "policy_step": step,
            "reward": f"{rew:.6f}",
            "done": str(done),
            "raw_image_sha": raw_sha,
            "model_input_sha": input_sha,
            "prompt_sha": prompt_sha,
            "pixel_tensor_sha": pixel_sha,
            "input_ids_sha": ids_sha,
            "generated_tokens": " ".join(str(x) for x in (shadow_tokens.tolist() if shadow_tokens is not None else [])) or "N/A",
            "final_action": " ".join(f"{x:.8f}" for x in act_o.tolist()),
            "raw_gripper": f"{raw_grip:.8f}",
            "norm_gripper": f"{norm_grip:.8f}",
            "env_gripper": f"{env_grip:.8f}",
            "eef_x": f"{eef_pos[0]:.6f}" if len(eef_pos) > 0 else "N/A",
            "eef_y": f"{eef_pos[1]:.6f}" if len(eef_pos) > 1 else "N/A",
            "eef_z": f"{eef_pos[2]:.6f}" if len(eef_pos) > 2 else "N/A",
            "inference_latency_s": f"{inf_latency:.4f}",
            "lane_o_m_match": str(lane_o_m_match) if do_shadow else "not_checked",
        }
        step_trace.append(trace_row)
        frames_raw.append(np.array(raw_img) if raw_img.ndim == 3 else raw_img)
        frames_input.append(np.array(processed_img))

        # Success check
        check_success = False
        try:
            check_success = bool(env.check_success())
        except Exception:
            pass

        if done or check_success:
            termination = "success" if (done or check_success) else "timeout"
            success = check_success or (done and rew > 0)
            break

    dt = time.time() - t0
    env.close()

    # Save videos and traces
    video_raw = f"{output_dir}/videos/{label}_raw.mp4"
    video_input = f"{output_dir}/videos/{label}_input.mp4"
    imageio.mimsave(video_raw, frames_raw, fps=10)
    imageio.mimsave(video_input, frames_input, fps=10)

    trace_path = f"{output_dir}/traces/{label}_trace.csv"
    with open(trace_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=step_trace[0].keys() if step_trace else [])
        w.writeheader()
        w.writerows(step_trace)

    return {
        "label": label, "task_idx": task_idx, "init_state_idx": init_idx,
        "task_name": task_name, "bddl_sha": bddl_sha, "init_state_sha": init_state_sha,
        "seed": seed, "steps": len(step_trace), "max_steps": max_steps,
        "done": bool(done), "success": success,
        "termination": termination, "error": error,
        "duration_s": round(dt, 1),
        "lane_o_m_match": lane_o_m_match,
        "video_raw": f"videos/{label}_raw.mp4",
        "video_input": f"videos/{label}_input.mp4",
        "trace": f"traces/{label}_trace.csv",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--stage", default="c0", choices=["c0", "c1", "all"])
    parser.add_argument("--task_idx", type=int, default=None)
    parser.add_argument("--init_state_idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--device_map", default=None, help="e.g. auto for multi-GPU")
    parser.add_argument("--max_memory", default=None, help="JSON dict for per-GPU caps")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    os.makedirs(f"{args.output_dir}/videos", exist_ok=True)
    os.makedirs(f"{args.output_dir}/traces", exist_ok=True)

    print(f"Model: {args.model_path}")
    print(f"Dtype: {args.dtype}  GPU: {args.gpu}  Stage: {args.stage}")

    load_kwargs = dict(
        torch_dtype=dtype, attn_implementation="eager",
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map=args.device_map or "cuda:0",
    )
    if args.max_memory:
        load_kwargs["max_memory"] = json.loads(args.max_memory)

    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **load_kwargs)
    proc = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    devices = sorted(set(str(p.device) for p in model.parameters()))
    print(f"Devices: {devices}")

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()
    print(f"Tasks: {task_suite.n_tasks}")

    # Frozen canary plan
    plan = [
        {"task_idx": 0, "init_idx": 0, "label": "task0_init0"},
        {"task_idx": 0, "init_idx": 1, "label": "task0_init1"},
        {"task_idx": 1, "init_idx": 0, "label": "task1_init0"},
        {"task_idx": 1, "init_idx": 1, "label": "task1_init1"},
    ]

    # Stage filtering
    if args.stage == "c0":
        episodes = [plan[0]]
    elif args.stage == "c1":
        episodes = plan[1:]
    else:
        episodes = plan

    # Override with CLI args if provided
    if args.task_idx is not None and args.init_state_idx is not None:
        episodes = [{"task_idx": args.task_idx, "init_idx": args.init_state_idx,
                      "label": f"task{args.task_idx}_init{args.init_state_idx}"}]

    results = []
    for ep in episodes:
        print(f"\n=== {ep['label']} (task={ep['task_idx']}, init={ep['init_idx']}) ===")
        r = run_episode(model, proc, task_suite, ep["task_idx"], ep["init_idx"],
                        args.output_dir, ep["label"], max_steps=args.max_steps, seed=args.seed)
        results.append(r)
        print(f"  steps={r['steps']} done={r['done']} success={r['success']} term={r['termination']}")
        print(f"  Lane O==M: {r['lane_o_m_match']}")

    plan_path = f"{args.output_dir}/canary_plan.json"
    with open(plan_path, "w") as f:
        json.dump({"episodes": [{"task_idx": ep["task_idx"], "init_idx": ep["init_idx"]}
                                 for ep in episodes], "seed": args.seed, "max_steps": args.max_steps}, f, indent=2)

    results_path = f"{args.output_dir}/canary_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {results_path}")
