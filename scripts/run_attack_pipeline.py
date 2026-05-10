#!/usr/bin/env python3
"""Stable public CLI for reproducing the gripper duty-cycle attack matrix.

This wrapper exposes paper-facing condition names and dispatches to the legacy
OpenVLA evaluation runner. Historical V4 script names are retained for traceability,
but new users should start here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "black_bowl": {
        "task_id": "libero_spatial_black_bowl",
        "unnorm_key": "libero_spatial",
        "model_env": "OPENVLA_SPATIAL_MODEL_PATH",
        "default_model_suffix": "openvla-7b-finetuned-libero-spatial",
        "object": "akita_black_bowl_1",
        "receptacle": "plate_1",
    }
}

CONDITIONS = {
    "clean": {
        "trigger": "clean",
        "rho": "0.0",
        "extra_env": {},
    },
    "oracle_continuous": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "extra_env": {
            "V4_ATTACK_OBJECTIVE": "oracle_env_gripper_open",
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_OPEN_PATTERN": "continuous_open",
        },
    },
    "vis_margin_prevdelta": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "extra_env": {
            "V4_ATTACK_OBJECTIVE": "gripper_logit_margin_cw",
            "V4_TEMPORAL_INIT": "prev_delta",
            "V4_CW_MARGIN": "5.0",
        },
    },
    "ctrl_same_gate_zero_margin": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "extra_env": {
            "V4_ATTACK_OBJECTIVE": "gripper_logit_margin_cw",
            "V4_TEMPORAL_INIT": "none",
            "V4_CW_MARGIN": "0.0",
        },
    },
    "ctrl_random_direction": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "extra_env": {
            "V4_ATTACK_OBJECTIVE": "untargeted_arm_clean_token_ce",
            "V4_TEMPORAL_INIT": "none",
        },
    },
    "constant_delta_pregrasp": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "extra_env": {
            "V4_ATTACK_OBJECTIVE": "constant_delta_pregrasp",
            "V4_CONSTANT_DELTA_GRIPPER": "-1.0",
        },
    },
}

STATE_WINDOWS = {
    7: (96, 112),
    5: (78, 87),
}


def env_path(primary: str, root_var: str, suffix: str) -> str:
    if os.environ.get(primary):
        return os.environ[primary]
    root = os.environ.get(root_var, "")
    if root:
        return str(Path(root) / suffix)
    return "${" + primary + "}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run public gripper duty-cycle attack conditions.")
    ap.add_argument("--task", choices=sorted(TASKS), default="black_bowl")
    ap.add_argument("--state", type=int, required=True, help="Deterministic LIBERO init state, e.g. 7 or 5.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--window_start", type=int, default=None)
    ap.add_argument("--window_end", type=int, default=None)
    ap.add_argument("--output_root", default=os.environ.get("OPENVLA_OUTPUT_ROOT", "outputs/repro_black_bowl"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--model_path", default="")
    ap.add_argument("--base_model_code_dir", default=os.environ.get("OPENVLA_BASE_MODEL_DIR", ""))
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--attack_config", default="configs/v4_attack.yaml")
    ap.add_argument("--directions_config", default="configs/v4_directions.yaml")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--dry_run", action="store_true", help="Print the dispatched command without executing it.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    task = TASKS[args.task]
    condition = CONDITIONS[args.condition]
    w0, w1 = STATE_WINDOWS.get(args.state, (args.window_start, args.window_end))
    if args.window_start is not None:
        w0 = args.window_start
    if args.window_end is not None:
        w1 = args.window_end
    if w0 is None or w1 is None:
        raise SystemExit("No default attack window for this state; pass --window_start and --window_end.")

    model_path = args.model_path or env_path(task["model_env"], "OPENVLA_MODEL_ROOT", task["default_model_suffix"])
    base_model = args.base_model_code_dir or os.environ.get("OPENVLA_BASE_MODEL_DIR", "${OPENVLA_BASE_MODEL_DIR}")
    run_id = f"{args.task}_state{args.state}_seed{args.seed}_{args.condition}"

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("V4_TARGET_OBJECT_NAME", task["object"])
    env.setdefault("V4_TARGET_RECEPTACLE_NAME", task["receptacle"])
    env.setdefault("V4_FIXED_WINDOW_START", str(w0))
    env.setdefault("V4_FIXED_WINDOW_END", str(w1))
    env.update(condition["extra_env"])

    cmd = [
        args.python,
        "scripts/v4_run_eval_openvla.py",
        "--tasks_config", args.tasks_config,
        "--attack_config", args.attack_config,
        "--directions_config", args.directions_config,
        "--task_id", task["task_id"],
        "--trigger", condition["trigger"],
        "--rho", condition["rho"],
        "--seed", str(args.seed),
        "--episodes", str(args.episodes),
        "--max_steps_override", str(args.max_steps),
        "--output_root", args.output_root,
        "--run_id", run_id,
        "--model_path", model_path,
        "--base_model_code_dir", base_model,
        "--unnorm_key", task["unnorm_key"],
        "--camera_obs_key", "agentview_image",
        "--libero_official_preprocess",
        "--center_crop",
        "--postprocess_gripper",
        "--deterministic_init_states",
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", args.thresholds])

    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
