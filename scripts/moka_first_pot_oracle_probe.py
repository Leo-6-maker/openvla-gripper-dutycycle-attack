#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_episode_record(run_dir: Path) -> dict:
    path = run_dir / "episode_records.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    last: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def scan_steps(run_dir: Path) -> dict:
    path = run_dir / "step_records.jsonl"
    out = {
        "attacks": 0,
        "exec_open_rate": "",
        "max_object_z_delta_during_attack": "",
        "qpos_abs_after_max": "",
    }
    if not path.exists() or path.stat().st_size == 0:
        return out
    attack_steps = []
    open_hits = 0
    max_z = None
    qpos_max = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bool(row.get("attack_active")):
                attack_steps.append(row)
                if bool(row.get("force_open_sign_target_ok")) or bool(row.get("oracle_env_override_active")):
                    open_hits += 1
                z = row.get("grasp_bowl_z_delta")
                if isinstance(z, (int, float)):
                    max_z = float(z) if max_z is None else max(max_z, float(z))
            q = row.get("qpos_abs_after_max")
            if isinstance(q, (int, float)):
                qpos_max = float(q) if qpos_max is None else max(qpos_max, float(q))
    out["attacks"] = len(attack_steps)
    out["exec_open_rate"] = "" if not attack_steps else open_hits / float(len(attack_steps))
    out["max_object_z_delta_during_attack"] = "" if max_z is None else max_z
    out["qpos_abs_after_max"] = "" if qpos_max is None else qpos_max
    return out


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    print("[cmd]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(proc.returncode)


def run_eval(args: argparse.Namespace, env: dict[str, str], condition: str, seed: int, window: tuple[int, int] | None) -> dict:
    if condition == "clean_reference":
        run_id = f"moka_clean_state{args.state}_seed{seed}_fresh"
        trigger = "clean"
        rho = "0"
        objective_args: list[str] = []
        env2 = dict(env)
    else:
        assert window is not None
        w0, w1 = window
        run_id = f"moka_oracle_state{args.state}_seed{seed}_w{w0}_{w1}"
        trigger = "fixed_step_window_budgeted"
        rho = "1.0"
        objective_args = [
            "--attack_objective", "oracle_env_gripper_open",
            "--epsilon", str(args.epsilon),
            "--step_size", str(args.step_size),
            "--attack_steps", str(args.attack_steps),
            "--temporal_init", "none",
            "--cw_margin", str(args.cw_margin),
        ]
        env2 = dict(env)
        env2["V4_FIXED_ATTACK_START"] = str(w0)
        env2["V4_FIXED_ATTACK_END"] = str(w1)
        env2["V4_ORACLE_FORCE_GRIPPER_ENV_VALUE"] = "-1.0"
        env2["V4_ORACLE_GRIPPER_PATTERN"] = "continuous_open"

    out_root = Path(args.output_root).resolve()
    cmd = [
        args.python,
        "scripts/v4_run_eval_openvla.py",
        "--tasks_config", args.tasks_config,
        "--attack_config", args.attack_config,
        "--directions_config", args.directions_config,
        "--task_id", "libero10_moka_pots",
        "--trigger", trigger,
        "--rho", rho,
        "--seed", str(seed),
        "--episodes", "1",
        "--max_steps_override", str(args.max_steps),
        "--model_path", args.model_path,
        "--base_model_code_dir", args.base_model_code_dir,
        "--unnorm_key", "libero_10",
        "--camera_obs_key", "agentview_image",
        "--model_gpu_device_id", "-1",
        "--render_gpu_device_id", str(args.render_gpu_device_id),
        "--image_size", "256",
        "--openvla_resize_size", "224",
        "--success_metric", "done",
        "--auto_patch_compat",
        "--libero_official_preprocess",
        "--center_crop",
        "--postprocess_gripper",
        "--deterministic_init_states",
        "--state_ids", str(args.state),
        "--output_root", str(out_root),
        "--run_id", run_id,
        *objective_args,
    ]
    rc = run_cmd(cmd, env2, ROOT)
    ep = load_episode_record(out_root / run_id)
    step_summary = scan_steps(out_root / run_id)
    video_path = ""
    for pattern in ("*.mp4", "*.avi"):
        hits = sorted((out_root / run_id).glob(pattern))
        if hits:
            video_path = str(hits[0])
            break
    return {
        "run_id": run_id,
        "state": args.state,
        "seed": seed,
        "condition": condition,
        "window_start": "" if window is None else window[0],
        "window_end": "" if window is None else window[1],
        "return_code": rc,
        "steps": ep.get("num_steps", ""),
        "SR": bool(ep.get("success")) if ep else False,
        "attacks": step_summary["attacks"],
        "exec_open_rate": step_summary["exec_open_rate"],
        "qpos_abs_after_max": step_summary["qpos_abs_after_max"],
        "max_object_z_delta_during_attack": step_summary["max_object_z_delta_during_attack"],
        "failure_phase": ep.get("failure_phase", ""),
        "video_path": video_path,
        "manual_outcome": "",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_windows(text: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-", 1)
        windows.append((int(a), int(b)))
    return windows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run fixed-window first-pot Moka oracle probes.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--attack_config", default="configs/paper_black_bowl_attack.yaml")
    ap.add_argument("--directions_config", default="configs/v4_directions.yaml")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model_code_dir", required=True)
    ap.add_argument("--output_root", default="/data/liuyu/outputs/moka_first_pot_oracle_probe_20260512")
    ap.add_argument("--state", type=int, default=1)
    ap.add_argument("--seeds", default="2")
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--windows", default="790-805,780-805,798-809")
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--skip_clean", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--step_size", type=float, default=0.020)
    ap.add_argument("--attack_steps", type=int, default=20)
    ap.add_argument("--cw_margin", type=float, default=5.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    windows = parse_windows(args.windows)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["V4_TARGET_OBJECT_NAME"] = "moka_pot_1"
    env["V4_TARGET_RECEPTACLE_NAME"] = "flat_stove_1"

    rows: list[dict] = []
    for seed in seeds:
        if not args.skip_clean:
            rows.append(run_eval(args, env, "clean_reference", seed, None))
            write_rows(Path(args.output_root) / "tables" / "moka_first_pot_oracle_probe_summary.csv", rows)
        for window in windows:
            rows.append(run_eval(args, env, "oracle_firstpot_preplace", seed, window))
            write_rows(Path(args.output_root) / "tables" / "moka_first_pot_oracle_probe_summary.csv", rows)
    print(f"[ok] wrote {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
