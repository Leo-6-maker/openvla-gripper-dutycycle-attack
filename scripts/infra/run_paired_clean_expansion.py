#!/usr/bin/env python3
"""F3: Paired clean expansion — BF16-Eager and BF16-Flash2 on 30 fixed episodes."""
import os, json, csv, time, hashlib, sys, argparse, numpy as np
from PIL import Image
from math import sqrt
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio

MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"


def run_one_episode(model, proc, task_suite, task_idx, init_idx, output_dir, label,
                    max_steps=220, wait_steps=10, seed=42):
    """Run one closed-loop episode. Returns result dict."""
    DEV = next(model.parameters()).device
    action_dim = model.get_action_dim("libero_spatial")
    stats = model.get_action_stats("libero_spatial")
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[init_idx])
    wait_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(wait_steps):
        obs, rew, done, info = env.step(wait_action)

    frames_raw = []
    step_trace = []
    done = False
    success = False
    termination = "timeout"
    error = None
    grip_flips = 0
    prev_env_grip = None
    first_close_step = None
    first_open_after_close = None

    t0 = time.time()
    for step in range(max_steps):
        raw_img = obs["agentview_image"]
        img = Image.fromarray(raw_img).rotate(180).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        scale = sqrt(0.9); cs = int(224 * scale); left = (224 - cs) // 2
        img = img.crop((left, left, left + cs, left + cs)).resize((224, 224), Image.LANCZOS)
        processed_sha = hashlib.sha256(np.array(img).tobytes()).hexdigest()

        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, img, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act_o = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        raw_grip = act_o[6]; norm_grip = (raw_grip * 2) - 1
        bin_grip = 1.0 if norm_grip >= 0 else -1.0; env_grip = -bin_grip

        env_action = np.zeros(7); env_action[:6] = act_o[:6]; env_action[6] = env_grip

        if prev_env_grip is not None and env_grip != prev_env_grip:
            grip_flips += 1
            if env_grip < 0 and prev_env_grip > 0 and first_close_step is None:
                first_close_step = step
            if env_grip > 0 and prev_env_grip < 0 and first_open_after_close is None and first_close_step is not None:
                first_open_after_close = step
        prev_env_grip = env_grip

        try:
            obs, rew, done, info = env.step(env_action.tolist())
        except Exception as e:
            error = str(e); termination = "runtime_error"; break

        frames_raw.append(np.array(raw_img) if raw_img.ndim == 3 else raw_img)
        step_trace.append({
            "policy_step": step, "final_action": " ".join("%.8f" % x for x in act_o.tolist()),
            "env_gripper": "%.8f" % env_grip, "reward": str(rew), "done": str(done),
        })

        try:
            if env.check_success():
                done = True; success = True
        except Exception:
            pass

        if done:
            if success:
                termination = "success"
            break

    dt = time.time() - t0
    env.close()

    # Save compact artifacts
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "%s.mp4" % label)
    try:
        imageio.mimsave(video_path, frames_raw[::2], fps=10)  # every other frame to save space
    except Exception:
        video_path = "save_failed"

    result = {
        "label": label, "task_idx": task_idx, "init_state_idx": init_idx,
        "steps": len(step_trace), "max_steps": max_steps, "done": bool(done),
        "success": success, "termination": termination, "error": error,
        "duration_s": round(dt, 1), "gripper_flips": grip_flips,
        "first_close_step": first_close_step, "first_open_after_close": first_open_after_close,
        "video": video_path,
    }
    return result, step_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    parser.add_argument("--cuda_devices", default="6")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--end_ep", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    print("Loaded: dtype=%s attn=%s actual=%s devices=%s" % (
        args.dtype, args.attn, actual_attn,
        sorted(set(str(p.device) for p in model.parameters()))))

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()

    # Build plan: 10 tasks x 3 init states
    plan = []
    for ti in range(10):
        init_states = task_suite.get_task_init_states(ti)
        for ii in range(min(3, len(init_states))):
            plan.append({"task_idx": ti, "init_idx": ii, "label": "task%d_init%d" % (ti, ii)})

    results = []
    for i in range(args.start_ep, min(args.end_ep, len(plan))):
        ep = plan[i]
        print("[%d/%d] %s" % (i + 1, len(plan), ep["label"]))
        res, trace = run_one_episode(model, proc, task_suite, ep["task_idx"], ep["init_idx"],
                                     args.output_dir, ep["label"], seed=42)
        res["attn"] = args.attn
        res["actual_attn"] = actual_attn
        res["plan_idx"] = i
        results.append(res)
        print("  steps=%d success=%d term=%s" % (res["steps"], res["success"], res["termination"]))

        # Save partial results
        out_json = os.path.join(args.output_dir, "results_%s.json" % args.attn)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print("Done: %d episodes, %d success" % (len(results), sum(1 for r in results if r["success"])))
