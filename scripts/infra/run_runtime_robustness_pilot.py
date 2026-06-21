#!/usr/bin/env python3
"""Runtime robustness pilot: runs Clean/VIS/Random on a frozen episode set."""
import os, json, csv, time, hashlib, argparse, math, sys
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
assert "MUJOCO_GL" in os.environ and "CUDA_VISIBLE_DEVICES" in os.environ

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio


def preprocess(img_array):
    img = Image.fromarray(img_array).rotate(180).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    s = math.sqrt(0.9); cs = int(224 * s); L = (224 - cs) // 2
    return img.crop((L, L, L + cs, L + cs)).resize((224, 224), Image.LANCZOS)


def run_episode_noop(model, proc, task_suite, ti, ii, seed, max_steps, wait_steps, condition="clean",
                      random_schedule=None):
    """Run one episode. condition = clean | vis | random. random_schedule is list of (start_step, end_step, target_grip)."""
    DEV = next(model.parameters()).device
    adim = model.get_action_dim("libero_spatial")
    stats = model.get_action_stats("libero_spatial")
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

    task = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[ii])
    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    trace, done, success, error = [], False, False, None
    attack_active = False
    attack_step_count = 0
    attack_segment_count = 0
    t0 = time.time()

    for step in range(max_steps):
        raw = obs["agentview_image"]
        processed = preprocess(raw)
        task_name = task.language if hasattr(task, "language") else task.name
        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten()

        # Gripper post-process
        raw_g = act[6]; norm_g = (raw_g * 2) - 1
        bin_g = 1.0 if norm_g >= 0 else -1.0; env_g = -bin_g

        # === ATTACK LOGIC (no-op for clean, placeholder for vis/random) ===
        orig_env_g = env_g
        if condition != "clean" and random_schedule:
            for seg_start, seg_end, target_grip in random_schedule:
                if seg_start <= step <= seg_end:
                    env_g = target_grip
                    attack_active = True
                    attack_step_count += 1
                    if step == seg_start:
                        attack_segment_count += 1
                    break

        env_act = np.zeros(7); env_act[:6] = act[:6]; env_act[6] = env_g

        try:
            obs, rew, done, info = env.step(env_act.tolist())
        except Exception as e:
            error = str(e); break

        trace.append({
            "step": step, "condition": condition,
            "attack_active": str(attack_active),
            "orig_gripper": "%.8f" % orig_env_g,
            "exec_gripper": "%.8f" % env_g,
            "action": " ".join("%.8f" % x for x in act.tolist()),
            "done": str(done), "reward": str(rew),
        })

        try:
            if env.check_success():
                done, success = True, True
        except Exception:
            pass
        if done:
            break

    env.close()
    dt = time.time() - t0
    return {
        "steps": len(trace), "success": success, "condition": condition,
        "termination": "success" if success else ("error" if error else "timeout"),
        "attack_steps": attack_step_count, "attack_segments": attack_segment_count,
        "duration_s": round(dt, 1), "error": error,
    }, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Pilot plan JSON (episode list)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    parser.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    parser.add_argument("--condition", default="clean", choices=["clean", "noop"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    DTYPE = torch.float32 if args.dtype == "float32" else torch.bfloat16

    with open(args.plan) as f:
        plan = json.load(f)

    episodes = plan["episodes"]
    print("Profile: %s %s | Plan: %d eps | Condition: %s" % (
        args.dtype, args.attn, len(episodes), args.condition))

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True,
        low_cpu_mem_usage=True)
    proc = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    devices = sorted(set(str(p.device) for p in model.parameters()))
    print("Devices: %s  Actual attn: %s  VRAM: %.2f GiB" % (
        devices, actual_attn, torch.cuda.max_memory_allocated() / 1024**3))

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()
    print("Tasks: %d" % task_suite.n_tasks)

    results = []
    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = "%s_%s" % (ep.get("label", "task%d_init%d" % (ti, ii)), args.condition)
        ep_dir = os.path.join(args.output_dir, label)
        done_file = os.path.join(ep_dir, ".done")

        if args.resume and os.path.exists(done_file):
            print("[%d/%d] %s SKIP" % (i + 1, len(episodes), label))
            continue

        os.makedirs(ep_dir, exist_ok=True)
        print("[%d/%d] %s" % (i + 1, len(episodes), label), end=" ", flush=True)

        res, trace = run_episode_noop(model, proc, task_suite, ti, ii, args.seed,
                                      args.max_steps, 10, condition="clean")
        res["label"] = label; res["task_idx"] = ti; res["init_idx"] = ii
        results.append(res)

        json.dump(res, open(os.path.join(ep_dir, "result.json"), "w"), indent=2)
        with open(os.path.join(ep_dir, "trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trace[0].keys())
            w.writeheader(); w.writerows(trace)
        open(done_file, "w").close()
        print("OK steps=%d succ=%d" % (res["steps"], res["success"]))

    succ = sum(1 for r in results if r["success"])
    print("Done: %d/%d (%.1f%%)" % (succ, len(results), 100 * succ / max(1, len(results))))

    json.dump({"total": len(results), "success": succ, "dtype": args.dtype, "attn": args.attn,
               "actual_attn": actual_attn}, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
