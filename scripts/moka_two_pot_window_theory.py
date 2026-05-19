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


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    print("[cmd]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(p.returncode)


def load_episode_record(run_dir: Path) -> dict:
    p = run_dir / "episode_records.jsonl"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    return json.loads(line) if line else {}


def gate_clean_runs(args: argparse.Namespace, env: dict[str, str], out_root: Path) -> tuple[bool, list[dict]]:
    rows: list[dict] = []
    success_n = 0
    for rep in range(1, int(args.clean_repeats) + 1):
        run_id = f"moka_clean_gate_state{args.state}_seed{args.seed}_rep{rep}"
        cmd = [
            args.python,
            "scripts/v4_run_eval_openvla.py",
            "--tasks_config", args.tasks_config,
            "--attack_config", args.attack_config,
            "--directions_config", args.directions_config,
            "--task_id", "libero10_moka_pots",
            "--trigger", "clean",
            "--rho", "0",
            "--seed", str(args.seed),
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
            "--moka_two_pot_mode",
            "--moka_stage_stable_steps", str(args.moka_stage_stable_steps),
            "--moka_second_window_start", str(args.second_window_start),
            "--moka_second_window_end", str(args.second_window_end),
        ]
        rc = run_cmd(cmd, env, ROOT)
        ep = load_episode_record(out_root / run_id)
        ok = bool(ep.get("success")) if rc == 0 and ep else False
        if ok:
            success_n += 1
        rows.append(
            {
                "phase": "clean_gate",
                "run_id": run_id,
                "return_code": rc,
                "success": ok,
                "failure_phase": ep.get("failure_phase", ""),
                "num_steps": ep.get("num_steps", ""),
                "num_attack_active_steps": ep.get("num_attack_active_steps", ""),
                "moka_first_phase_attack_steps": ep.get("moka_first_phase_attack_steps", ""),
                "moka_second_phase_attack_steps": ep.get("moka_second_phase_attack_steps", ""),
                "valid_attack_run": "",
                "invalid_reason": "",
            }
        )
    return success_n >= int(args.clean_success_min), rows


def run_attack_phase(args: argparse.Namespace, env: dict[str, str], out_root: Path, phase: str) -> dict:
    if phase == "oracle":
        trigger = "moka_second_pot_relative_window_budgeted"
        objective = "oracle_env_gripper_open"
        temporal_init = "none"
        extras = {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        }
    else:
        trigger = "moka_second_pot_relative_window_budgeted"
        objective = "gripper_logit_margin_cw"
        temporal_init = "prev_delta"
        extras = {}
    env2 = dict(env)
    env2.update(extras)
    run_id = f"moka_{phase}_state{args.state}_seed{args.seed}_rel{args.second_window_start}_{args.second_window_end}"
    cmd = [
        args.python,
        "scripts/v4_run_eval_openvla.py",
        "--tasks_config", args.tasks_config,
        "--attack_config", args.attack_config,
        "--directions_config", args.directions_config,
        "--task_id", "libero10_moka_pots",
        "--trigger", trigger,
        "--rho", "1.0",
        "--seed", str(args.seed),
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
        "--attack_objective", objective,
        "--epsilon", str(args.epsilon),
        "--step_size", str(args.step_size),
        "--attack_steps", str(args.attack_steps),
        "--temporal_init", temporal_init,
        "--cw_margin", str(args.cw_margin),
        "--moka_two_pot_mode",
        "--moka_stage_stable_steps", str(args.moka_stage_stable_steps),
        "--moka_second_window_start", str(args.second_window_start),
        "--moka_second_window_end", str(args.second_window_end),
    ]
    rc = run_cmd(cmd, env2, ROOT)
    ep = load_episode_record(out_root / run_id)
    return {
        "phase": phase,
        "run_id": run_id,
        "return_code": rc,
        "success": bool(ep.get("success")) if ep else False,
        "failure_phase": ep.get("failure_phase", ""),
        "num_steps": ep.get("num_steps", ""),
        "num_attack_active_steps": ep.get("num_attack_active_steps", ""),
        "moka_first_phase_attack_steps": ep.get("moka_first_phase_attack_steps", ""),
        "moka_second_phase_attack_steps": ep.get("moka_second_phase_attack_steps", ""),
        "valid_attack_run": "",
        "invalid_reason": "",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Moka two-pot window theory pipeline: clean gate -> oracle -> vis.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--attack_config", default="configs/v4_attack.yaml")
    ap.add_argument("--directions_config", default="configs/v4_directions.yaml")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model_code_dir", required=True)
    ap.add_argument("--output_root", default="outputs/v4/moka_twopot_window_theory")
    ap.add_argument("--state", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--clean_repeats", type=int, default=5)
    ap.add_argument("--clean_success_min", type=int, default=3)
    ap.add_argument("--second_window_start", type=int, default=0)
    ap.add_argument("--second_window_end", type=int, default=30)
    ap.add_argument("--moka_stage_stable_steps", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--step_size", type=float, default=0.020)
    ap.add_argument("--attack_steps", type=int, default=20)
    ap.add_argument("--cw_margin", type=float, default=5.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["V4_TARGET_OBJECT_NAME"] = "moka_pot_1"
    env["V4_TARGET_RECEPTACLE_NAME"] = "flat_stove_1"

    rows: list[dict] = []
    gate_pass, gate_rows = gate_clean_runs(args, env, out_root)
    rows.extend(gate_rows)
    if not gate_pass:
        rows.append(
            {
                "phase": "gate_fail",
                "run_id": "",
                "return_code": 0,
                "success": False,
                "failure_phase": "clean_gate_failed",
                "num_steps": "",
                "num_attack_active_steps": "",
                "moka_first_phase_attack_steps": "",
                "moka_second_phase_attack_steps": "",
                "valid_attack_run": "",
                "invalid_reason": "",
            }
        )
        write_rows(out_root / "tables" / "moka_two_pot_window_theory_summary.csv", rows)
        print("[stop] clean gate failed; skip oracle/vis.", flush=True)
        return 0

    oracle_row = run_attack_phase(args, env, out_root, "oracle")
    rows.append(oracle_row)
    oracle_attacks = int(oracle_row.get("num_attack_active_steps") or 0)
    oracle_second_phase_attacks = int(oracle_row.get("moka_second_phase_attack_steps") or 0)
    if oracle_attacks <= 0 or oracle_second_phase_attacks <= 0:
        oracle_row["valid_attack_run"] = False
        oracle_row["invalid_reason"] = "oracle_no_attack"
        rows.append(
            {
                "phase": "oracle_invalid_stop_vis",
                "run_id": oracle_row.get("run_id", ""),
                "return_code": 0,
                "success": False,
                "failure_phase": "oracle_no_attack",
                "num_steps": "",
                "num_attack_active_steps": "",
                "moka_first_phase_attack_steps": "",
                "moka_second_phase_attack_steps": "",
                "valid_attack_run": False,
                "invalid_reason": "oracle_no_attack",
            }
        )
        write_rows(out_root / "tables" / "moka_two_pot_window_theory_summary.csv", rows)
        print("[stop] oracle produced no second-phase attacks; skip vis.", flush=True)
        return 0
    oracle_row["valid_attack_run"] = True
    if bool(oracle_row.get("success")):
        rows.append(
            {
                "phase": "oracle_not_failed_stop_vis",
                "run_id": oracle_row.get("run_id", ""),
                "return_code": 0,
                "success": True,
                "failure_phase": "oracle_not_failed",
                "num_steps": "",
                "num_attack_active_steps": "",
                "moka_first_phase_attack_steps": "",
                "moka_second_phase_attack_steps": "",
                "valid_attack_run": "",
                "invalid_reason": "",
            }
        )
        write_rows(out_root / "tables" / "moka_two_pot_window_theory_summary.csv", rows)
        print("[stop] oracle did not fail; skip vis by policy.", flush=True)
        return 0

    vis_row = run_attack_phase(args, env, out_root, "vis")
    rows.append(vis_row)
    write_rows(out_root / "tables" / "moka_two_pot_window_theory_summary.csv", rows)
    print("[ok] moka two-pot window theory pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
