#!/usr/bin/env python3
"""Single-process clean rollout: loads model once, runs all episodes from frozen plan.
Supports two preprocessing backends:
  - pil:              project PIL path (180° rotate, LANCZOS, integer crop)
  - upstream_tf_jpeg: official OpenVLA upstream path (JPEG round-trip, TF Lanczos3, TF crop_and_resize)
Usage: python launch_runtime_profile.py --profile bf16_flash2_upstream --plan plan.json ..."""
import os, json, csv, time, hashlib, argparse, sys, math
import numpy as np
from PIL import Image

# Env vars must be set BEFORE imports by launcher
assert "MUJOCO_GL" in os.environ, "MUJOCO_GL not set — use launch_runtime_profile.py"
assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES not set"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio


# Lazy-import TF (only needed for upstream backend)
_preprocess_upstream_fn = None


def _get_upstream_fn():
    global _preprocess_upstream_fn
    if _preprocess_upstream_fn is None:
        import tensorflow as tf
        _preprocess_upstream_fn = tf  # placeholder, real import from preprocess_upstream
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from preprocess_upstream import preprocess_upstream_tf_jpeg
        _preprocess_upstream_fn = preprocess_upstream_tf_jpeg
    return _preprocess_upstream_fn


def preprocess_pil(img_array):
    """Project PIL path: 180° rotate, RGB, LANCZOS 224, center crop sqrt(0.9), LANCZOS 224."""
    img = Image.fromarray(img_array).rotate(180).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    s = math.sqrt(0.9); cs = int(224 * s); L = (224 - cs) // 2
    img = img.crop((L, L, L + cs, L + cs)).resize((224, 224), Image.LANCZOS)
    return img


def run_episode(model, proc, task_suite, ti, ii, seed, max_steps, wait_steps, preprocess_fn=preprocess_pil):
    """Run one episode. Returns (result_dict, step_trace_list)."""
    DEV = next(model.parameters()).device
    adim = model.get_action_dim("libero_spatial")
    stats = model.get_action_stats("libero_spatial")
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

    task = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[ii])
    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    frames, trace = [], []
    done, success, error, grip_flips = False, False, None, 0
    prev_env_grip = None
    t0 = time.time()

    for step in range(max_steps):
        raw = obs["agentview_image"]
        processed = preprocess_fn(raw)
        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        raw_g = act[6]; norm_g = (raw_g * 2) - 1
        bin_g = 1.0 if norm_g >= 0 else -1.0; env_g = -bin_g
        env_act = np.zeros(7); env_act[:6] = act[:6]; env_act[6] = env_g

        if prev_env_grip is not None and env_g != prev_env_grip:
            grip_flips += 1
        prev_env_grip = env_g

        try:
            obs, rew, done, info = env.step(env_act.tolist())
        except Exception as e:
            error = str(e); break

        frames.append(np.array(raw) if raw.ndim == 3 else raw)
        trace.append({"step": step, "action": " ".join("%.8f" % x for x in act.tolist()),
                      "env_gripper": "%.8f" % env_g, "done": str(done), "reward": str(rew)})

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
        "steps": len(trace), "success": success, "termination": "success" if success else ("timeout" if not error else "error"),
        "error": error, "duration_s": round(dt, 1), "gripper_flips": grip_flips,
    }, trace, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="JSON plan file")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    parser.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    parser.add_argument("--preprocess_backend", default="pil",
                       choices=["pil", "upstream_tf_jpeg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--end_ep", type=int, default=999)
    args = parser.parse_args()

    DTYPE = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.preprocess_backend == "upstream_tf_jpeg":
        preprocess_fn = _get_upstream_fn()
        print("Preprocess backend: upstream_tf_jpeg (official OpenVLA)")
    else:
        preprocess_fn = preprocess_pil
        print("Preprocess backend: pil (project legacy)")

    with open(args.plan) as f:
        plan = json.load(f)

    episodes = plan["episodes"][args.start_ep:args.end_ep]
    print("Plan: %d episodes, dtype=%s attn=%s" % (len(episodes), args.dtype, args.attn))

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    proc = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    devices = sorted(set(str(p.device) for p in model.parameters()))
    gpu_uuid = "unknown"  # UUID query varies by torch version; rely on nvidia-smi for UUID
    print("Devices: %s  actual_attn: %s  GPU_UUID: %s" % (devices, actual_attn, gpu_uuid))
    print("VRAM after load: %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()
    print("Tasks: %d" % task_suite.n_tasks)

    results = []
    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = ep.get("label", "task%d_init%d" % (ti, ii))
        ep_dir = os.path.join(args.output_dir, label)
        done_file = os.path.join(ep_dir, ".done")
        result_file = os.path.join(ep_dir, "result.json")

        if args.resume and os.path.exists(done_file):
            print("[%d/%d] %s SKIP (already done)" % (i + 1, len(episodes), label))
            if os.path.exists(result_file):
                results.append(json.load(open(result_file)))
            continue

        os.makedirs(ep_dir, exist_ok=True)
        print("[%d/%d] %s" % (i + 1, len(episodes), label), end=" ", flush=True)

        res, trace, frames = run_episode(model, proc, task_suite, ti, ii, args.seed, args.max_steps, 10, preprocess_fn)
        res["label"] = label; res["task_idx"] = ti; res["init_idx"] = ii
        results.append(res)

        # Save atomically
        json.dump(res, open(result_file, "w"), indent=2)
        with open(os.path.join(ep_dir, "trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trace[0].keys() if trace else [])
            w.writeheader(); w.writerows(trace)
        try:
            imageio.mimsave(os.path.join(ep_dir, "video.mp4"), frames[::2], fps=10)
        except Exception:
            pass
        open(done_file, "w").close()

        succ_str = "OK" if res["success"] else ("ERR" if res["error"] else "TO")
        print("%s steps=%d" % (succ_str, res["steps"]))

        # Device integrity check
        cur_devices = sorted(set(str(p.device) for p in model.parameters()))
        if cur_devices != devices:
            print("FATAL: device map changed %s -> %s" % (devices, cur_devices)); sys.exit(1)

    succ = sum(1 for r in results if r["success"])
    print("\nDone: %d/%d success (%.1f%%)" % (succ, len(results), 100 * succ / max(1, len(results))))

    # Summary manifest
    manifest = {
        "runner": os.path.basename(__file__),
        "model": args.model_path, "dtype": args.dtype, "attn": args.attn,
        "actual_attn": actual_attn, "gpu_uuid": gpu_uuid,
        "preprocess_backend": args.preprocess_backend,
        "total": len(results), "success": succ,
        "plan_file": args.plan,
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
