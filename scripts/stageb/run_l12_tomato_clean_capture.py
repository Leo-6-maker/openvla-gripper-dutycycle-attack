#!/usr/bin/env python3
"""L12 Phase B: Tomato sauce s0 clean capture-only.

Runs clean OpenVLA inference (NO attack, NO PGD, NO RAND, NO Layer3).
Captures all privileged and deployment-safe fields per step.
Exports: clean_trace.csv, clean_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Server paths ──
LIBERO_ROOT = "/data/liuyu/repos/LIBERO-official-20260525"
REPO_ROOT = "/data/liuyu/repos/l12_clean_window_pipeline"
MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
BENCHMARK_ROOT = "/data/liuyu/repos/LIBERO-official-20260525/libero/libero/benchmark"
OUTPUT_BASE = "/data/liuyu/outputs/l12_clean_capture_20260614"

sys.path.insert(0, LIBERO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, str(Path(REPO_ROOT) / "src"))


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tomato_sauce")
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--output-dir", default=OUTPUT_BASE)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    import libero
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import benchmark, get_libero_path

    # ── Commit & environment info ──
    import subprocess
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"

    run_id = args.run_id or f"{args.task}_s{args.state_id}_seed{args.seed}"
    out_dir = Path(args.output_dir) / f"{run_id}_{commit[:7]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment ──
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()

    task_id = None
    for i, task in enumerate(task_suite.tasks):
        if args.task.replace("_", " ") in task.name.replace("_", " "):
            task_id = i
            break
    if task_id is None:
        available = [t.name for t in task_suite.tasks]
        raise ValueError(f"Task '{args.task}' not found. Available: {available}")

    task = task_suite.tasks[task_id]
    task_suite.set_task_id(task_id)

    env_args = {
        "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
        "camera_heights": 224,
        "camera_widths": 224,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": ["frontview", "robot0_eye_in_hand"],
        "control_freq": 20,
        "env_specific_kwargs": {
            "env_state_id": args.state_id,
        },
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    # ── Model ──
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # ── Unnormalize ──
    from libero.libero.utils import get_libero_action_bounds
    import pickle
    norm_path = Path(LIBERO_ROOT) / "libero/libero/benchmark/libero_object_bounds.pkl"
    with open(norm_path, "rb") as f:
        bounds = pickle.load(f)
    low = bounds["low"]
    high = bounds["high"]
    center = (low + high) / 2.0
    n_bins = 256
    bin_width = (high - low) / n_bins

    # ── Run clean episode ──
    obs = env.reset()
    task_name = task.name
    instruction = task_suite.get_task_instruction(task_id)
    done = False
    step = 0
    trace_rows = []

    for step in range(args.max_steps):
        # Process image
        img = obs["frontview_image"]
        img_bytes = (img * 255).astype(np.uint8).tobytes()
        img_sha = sha256_bytes(img_bytes)

        inputs = processor(
            images=img, text=instruction, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        tokens = outputs.sequences[0, -7:].cpu().tolist()
        token_str = ",".join(str(t) for t in tokens)

        # Decode to raw actions
        disc = np.array([t - 1 for t in tokens], dtype=np.float32)
        raw = center + disc * bin_width
        raw = np.clip(raw, low, high)

        # Step environment
        obs, reward, done, info = env.step(raw)

        # Gripper state from MuJoCo
        try:
            qpos = env.env.get_object("gripper_states")["gripper_qpos"]
            qpos_abs = abs(qpos[0]) if hasattr(qpos, '__len__') else abs(qpos)
        except Exception:
            qpos = [0.0, 0.0]
            qpos_abs = 0.0

        # EEF position
        eef_xyz = env.env.get_object("eef_states")["eef_pos"]

        # Object position
        try:
            obj_xyz = obs["object_states"]
        except Exception:
            obj_xyz = np.zeros(3)

        # Target position
        try:
            target_xyz = obs["target_states"] if "target_states" in obs else np.zeros(3)
        except Exception:
            target_xyz = np.zeros(3)

        # Decode gripper semantics
        gripper_disc = disc[-1]
        decoded_open_bool = int(gripper_disc > 127)
        cleaned_raw_gripper = float(raw[-1])
        cleaned_env_gripper = 1.0 if cleaned_raw_gripper > 0.5 else -1.0
        cleaned_close = int(cleaned_raw_gripper <= 0.5)

        # close_onset and close_streak (from trace context)
        close_onset = 0
        close_streak = 0
        if step > 0:
            prev_close = trace_rows[-1].get("clean_close", 0)
            if cleaned_close and not prev_close:
                close_onset = 1
                close_streak = 1
            elif cleaned_close and prev_close:
                close_onset = 0
                close_streak = trace_rows[-1].get("close_streak", 0) + 1
            else:
                close_onset = 0
                close_streak = 0
        elif cleaned_close:
            close_onset = 1
            close_streak = 1

        # Compute distances
        eef_to_obj = float(np.linalg.norm(np.array(eef_xyz) - np.array(obj_xyz)))
        obj_to_target = float(np.linalg.norm(np.array(obj_xyz) - np.array(target_xyz)))

        row = {
            "step": step,
            "clean_gripper_env": cleaned_env_gripper,
            "clean_gripper_raw": cleaned_raw_gripper,
            "gripper_qpos_before": float(qpos[0]) if hasattr(qpos, '__len__') else float(qpos),
            "gripper_qpos_after": float(qpos[0]) if hasattr(qpos, '__len__') else float(qpos),
            "qpos_abs_before": float(qpos_abs),
            "qpos_abs_after": float(qpos_abs),
            "eef_x": float(eef_xyz[0]), "eef_y": float(eef_xyz[1]), "eef_z": float(eef_xyz[2]),
            "clean_close": cleaned_close,
            "close_onset": close_onset,
            "close_streak": close_streak,
            "decoded_open_bool": decoded_open_bool,
            "obj_x": float(obj_xyz[0]), "obj_y": float(obj_xyz[1]), "obj_z": float(obj_xyz[2]),
            "target_obj_x": float(target_xyz[0]), "target_obj_y": float(target_xyz[1]),
            "target_obj_z": float(target_xyz[2]),
            "eef_to_obj_distance": eef_to_obj,
            "obj_to_target_distance": obj_to_target,
            "exact_tokens": token_str,
            "success": int(done),
            "done": int(done),
            "instruction": instruction,
            "image_sha256": img_sha,
        }
        trace_rows.append(row)

        if done:
            break

    env.close()

    # ── Write trace CSV ──
    csv_path = out_dir / f"clean_trace_{args.task}_s{args.state_id}_seed{args.seed}.csv"
    fieldnames = list(trace_rows[0].keys()) if trace_rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(trace_rows)

    # ── Write manifest ──
    manifest = {
        "repo_commit": commit,
        "dirty_status": dirty or "clean",
        "hostname": hostname,
        "runner_sha256": sha256_file(__file__),
        "model_path": args.model_path,
        "task": args.task,
        "state_id": args.state_id,
        "seed": args.seed,
        "eval_seed": args.seed,
        "success": int(done),
        "episode_length": len(trace_rows),
        "clean_trace_csv": str(csv_path),
        "trace_sha256": sha256_file(str(csv_path)),
        "max_steps": args.max_steps,
        "instruction": instruction,
    }
    manifest_path = out_dir / "clean_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Run command file ──
    cmd_path = out_dir / "run_command.txt"
    with open(cmd_path, "w") as f:
        f.write(f"python {__file__} " + " ".join(sys.argv[1:]) + "\n")

    # ── Environment file ──
    env_path = out_dir / "environment.txt"
    with open(env_path, "w") as f:
        f.write(f"hostname: {hostname}\n")
        f.write(f"commit: {commit}\n")
        f.write(f"dirty: {dirty or 'clean'}\n")
        f.write(f"python: {sys.executable}\n")
        f.write(f"cuda_visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')}\n")

    print(f"Trace:      {csv_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Steps:      {len(trace_rows)}")
    print(f"Success:    {done}")
    print(f"Trace SHA:  {manifest['trace_sha256']}")


if __name__ == "__main__":
    main()
