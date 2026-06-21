#!/usr/bin/env python3
"""Upstream clean30 runner with resume integrity.

Loads model once per profile, runs all episodes from frozen plan.
Atomic per-episode saves with integrity metadata.
Flash2 profile asserts actual_attn == flash_attention_2.

Usage:
  python launch_runtime_profile.py --profile fp32_upstream --plan plan.json ...
"""

import os, sys, json, csv, hashlib, argparse, time, math
import numpy as np
from PIL import Image

assert "MUJOCO_GL" in os.environ, "MUJOCO_GL not set"
assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES not set"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio

# Canonical preprocessing
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from gripper_attack.openvla_preprocess import prepare_openvla_image, resolve_backend

# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------


def sha256_hex(data) -> str:
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    elif isinstance(data, (str, bytes)):
        data = data.encode() if isinstance(data, str) else data
    else:
        data = str(data).encode()
    return hashlib.sha256(data).hexdigest()


def _run_attrs(args, model, plan) -> dict:
    return {
        "backend": resolve_backend(args.preprocess_backend),
        "center_crop": True,
        "dtype": args.dtype,
        "attn": args.attn,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "plan_sha": sha256_hex(json.dumps(plan, sort_keys=True)),
        "runner_sha": sha256_hex(open(__file__, "rb").read()),
    }


def _check_resume_integrity(ep_dir, run_attrs):
    """Verify stored integrity matches current run. Returns True if safe to skip."""
    manifest_path = os.path.join(ep_dir, "episode_manifest.json")
    done_path = os.path.join(ep_dir, ".done")
    if not os.path.exists(done_path) or not os.path.exists(manifest_path):
        return False
    try:
        stored = json.load(open(manifest_path))
    except Exception:
        return False
    for key in ("backend", "center_crop", "dtype", "attn", "seed", "max_steps", "plan_sha", "runner_sha"):
        if str(stored.get(key)) != str(run_attrs.get(key)):
            print("  RESUME_INTEGRITY_FAIL: %s mismatch" % key, flush=True)
            return False
    return True


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(model, proc, task_suite, ti, ii, seed, max_steps, wait_steps,
                preprocess_backend):
    """Run one episode. Returns (result_dict, step_trace_list, frame_list)."""
    DEV = next(model.parameters()).device
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

    init_state_sha = sha256_hex(init_states[ii])

    frames, trace = [], []
    done, success, error, grip_flips = False, False, None, 0
    prev_env_grip = None
    invalid = False
    t0 = time.time()

    for step in range(max_steps):
        raw = obs["agentview_image"]

        processed = prepare_openvla_image(
            raw,
            libero_preprocess_backend=preprocess_backend,
            center_crop=True,
            resize_size=224,
        )

        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        if not np.all(np.isfinite(act)):
            invalid = True
            error = "non_finite_action"
            break

        raw_g = act[6]
        norm_g = (raw_g * 2) - 1
        bin_g = 1.0 if norm_g >= 0 else -1.0
        env_g = -bin_g
        env_act = np.zeros(7)
        env_act[:6] = act[:6]
        env_act[6] = env_g

        if prev_env_grip is not None and env_g != prev_env_grip:
            grip_flips += 1
        prev_env_grip = env_g

        try:
            obs, rew, done, info = env.step(env_act.tolist())
        except Exception as e:
            error = str(e)
            break

        frames.append(np.array(raw) if raw.ndim == 3 else raw)
        trace.append({
            "step": step,
            "action": " ".join("%.8f" % x for x in act.tolist()),
            "env_gripper": "%.8f" % env_g,
            "done": str(done),
            "reward": str(rew),
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

    termination = "success" if success else ("error" if error else "timeout")
    if invalid:
        termination = "invalid"

    return {
        "steps": len(trace),
        "success": success,
        "invalid": invalid,
        "termination": termination,
        "error": error,
        "duration_s": round(dt, 1),
        "gripper_flips": grip_flips,
        "init_state_sha": init_state_sha,
    }, trace, frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    parser.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    parser.add_argument("--preprocess_backend", default="upstream_tf_jpeg",
                       choices=["upstream_tf_jpeg", "project_pil_lanczos", "none",
                                "tf_jpeg_legacy", "official_pil_lanczos"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--end_ep", type=int, default=999)
    args = parser.parse_args()

    DTYPE = torch.float32 if args.dtype == "float32" else torch.bfloat16
    backend = resolve_backend(args.preprocess_backend)

    with open(args.plan) as f:
        plan = json.load(f)

    episodes = plan["episodes"][args.start_ep:args.end_ep]
    print("Plan: %d episodes  dtype=%s  attn=%s  backend=%s" % (
        len(episodes), args.dtype, args.attn, backend))

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    proc = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    devices = sorted(set(str(p.device) for p in model.parameters()))

    # Flash2 assertion
    if args.attn == "flash_attention_2":
        assert actual_attn == "flash_attention_2", (
            "FATAL: requested flash_attention_2 but actual is %s" % actual_attn
        )

    # GPU UUID via nvidia-smi
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    try:
        import subprocess
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "-i", gpu_index, "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        gpu_uuid = "unknown"

    # Software versions
    sw_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf
        sw_versions["tensorflow"] = tf.__version__
    except Exception:
        sw_versions["tensorflow"] = "not_installed"
    try:
        import transformers
        sw_versions["transformers"] = transformers.__version__
    except Exception:
        pass

    print("Devices: %s  actual_attn: %s  GPU_UUID: %s" % (devices, actual_attn, gpu_uuid))
    print("VRAM after load: %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()
    print("Tasks: %d" % task_suite.n_tasks)

    # Build integrity attrs for resume checks
    run_attrs = _run_attrs(args, model, plan)
    run_attrs["actual_attn"] = actual_attn
    run_attrs["gpu_uuid"] = gpu_uuid
    run_attrs["model_path"] = args.model_path
    run_attrs["sw_versions"] = sw_versions

    results = []
    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = ep.get("label", "task%d_init%d" % (ti, ii))
        ep_dir = os.path.join(args.output_dir, label)

        if args.resume and _check_resume_integrity(ep_dir, run_attrs):
            print("[%d/%d] %s SKIP (integrity verified)" % (i + 1, len(episodes), label))
            try:
                results.append(json.load(open(os.path.join(ep_dir, "result.json"))))
            except Exception:
                pass
            continue

        os.makedirs(ep_dir, exist_ok=True)
        print("[%d/%d] %s" % (i + 1, len(episodes), label), end=" ", flush=True)

        res, trace, frames = run_episode(
            model, proc, task_suite, ti, ii, args.seed, args.max_steps, 10,
            preprocess_backend=backend,
        )
        res["label"] = label
        res["task_idx"] = ti
        res["init_idx"] = ii
        results.append(res)

        # Write episode_manifest.json FIRST
        ep_manifest = {
            "label": label, "task_idx": ti, "init_idx": ii,
            "dtype": args.dtype, "attn": args.attn,
            "actual_attn": actual_attn, "backend": backend,
            "center_crop": True, "seed": args.seed,
            "max_steps": args.max_steps, "gpu_uuid": gpu_uuid,
            "runner_sha": run_attrs["runner_sha"],
            "plan_sha": run_attrs["plan_sha"],
            "sw_versions": sw_versions,
        }
        json.dump(ep_manifest, open(os.path.join(ep_dir, "episode_manifest.json"), "w"), indent=2)

        # Save result.json
        json.dump(res, open(os.path.join(ep_dir, "result.json"), "w"), indent=2)

        # Save trace.csv
        with open(os.path.join(ep_dir, "trace.csv"), "w", newline="") as f:
            if trace:
                w = csv.DictWriter(f, fieldnames=trace[0].keys())
                w.writeheader()
                w.writerows(trace)

        # Save video (optional, non-critical)
        try:
            imageio.mimsave(os.path.join(ep_dir, "video.mp4"), frames[::2], fps=10)
        except Exception:
            pass

        # .done marker LAST (after all files)
        with open(os.path.join(ep_dir, ".done"), "w") as f:
            f.write("")

        succ_str = "OK" if res["success"] else ("INVALID" if res.get("invalid") else ("ERR" if res["error"] else "TO"))
        print("%s steps=%d" % (succ_str, res["steps"]))

        # Device integrity check
        cur_devices = sorted(set(str(p.device) for p in model.parameters()))
        if cur_devices != devices:
            print("FATAL: device map changed %s -> %s" % (devices, cur_devices))
            sys.exit(1)

    succ = sum(1 for r in results if r["success"])
    invalid = sum(1 for r in results if r.get("invalid", False))
    print("\nDone: %d/%d success  %d invalid  (%.1f%%)" % (
        succ, len(results), invalid, 100 * succ / max(1, len(results))))

    manifest = {
        "runner": os.path.basename(__file__),
        "runner_sha": run_attrs["runner_sha"],
        "plan_sha": run_attrs["plan_sha"],
        "model": args.model_path, "dtype": args.dtype, "attn": args.attn,
        "actual_attn": actual_attn, "gpu_uuid": gpu_uuid,
        "preprocess_backend": backend, "center_crop": True,
        "total": len(results), "success": succ, "invalid": invalid,
        "plan_file": args.plan, "sw_versions": sw_versions,
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
