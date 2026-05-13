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
        "objective": "",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {},
    },
    "oracle_continuous": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "oracle_env_gripper_open",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
    },
    "vis_margin_prevdelta": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "prev_delta",
        "cw_margin": "5.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
    },
    "ctrl_same_gate_zero_margin": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "none",
        "cw_margin": "0.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
    },
    "ctrl_random_direction": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "untargeted_arm_clean_token_ce",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {},
    },
    "constant_delta_pregrasp": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "constant_delta_pregrasp",
        "temporal_init": "none",
        "cw_margin": None,
        "window": (35, 45),
        "extra_env": {
            "V4_CONSTANT_DELTA_GRIPPER": "-1.0",
        },
    },
    "noise_baseline": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "noise_baseline",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {},
    },
    "oracle_continuous_guard": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "oracle_env_gripper_open",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
        "guard_enabled": True,
        "guard_mode": "conservative",
    },
    "vis_margin_prevdelta_guard": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "prev_delta",
        "cw_margin": "5.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
        "guard_enabled": True,
        "guard_mode": "conservative",
    },
    "ctrl_same_gate_zero_margin_guard": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "none",
        "cw_margin": "0.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
        "guard_enabled": True,
        "guard_mode": "conservative",
    },
    "oracle_continuous_guard_strict": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "oracle_env_gripper_open",
        "temporal_init": "none",
        "cw_margin": None,
        "window": None,
        "extra_env": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
        "guard_enabled": True,
        "guard_mode": "strict_after_close",
    },
    "vis_margin_prevdelta_guard_strict": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "prev_delta",
        "cw_margin": "5.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
        "guard_enabled": True,
        "guard_mode": "strict_after_close",
    },
    "ctrl_same_gate_zero_margin_guard_strict": {
        "trigger": "fixed_step_window_budgeted",
        "rho": "1.0",
        "objective": "gripper_logit_margin_cw",
        "temporal_init": "none",
        "cw_margin": "0.0",
        "force_open_raw_gripper": "1.0",
        "window": None,
        "extra_env": {},
        "guard_enabled": True,
        "guard_mode": "strict_after_close",
    },
}

STATE_WINDOWS = {
    7: (75, 84),
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
    ap = argparse.ArgumentParser(
        description="Run public gripper duty-cycle attack conditions. "
                    "See docs/reproducibility.md for the full condition matrix "
                    "and docs/claim_and_evidence.md for claim boundaries."
    )
    ap.add_argument("--task", choices=sorted(TASKS), default="black_bowl")
    ap.add_argument("--state", type=int, choices=[5, 7], required=True,
                    help="Deterministic LIBERO init state (5 or 7).")
    ap.add_argument("--seed", type=int, required=True,
                    help="Random seed for reproducibility. See docs/reproducibility.md for paper seeds per state.")
    ap.add_argument("--condition", choices=sorted(CONDITIONS), required=True,
                    help="Attack condition name. See docs/reproducibility.md for full descriptions.")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--window_start", type=int, default=None)
    ap.add_argument("--window_end", type=int, default=None)
    ap.add_argument("--output_root", default=os.environ.get("OPENVLA_OUTPUT_ROOT", "outputs/repro_black_bowl"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--model_path", default="")
    ap.add_argument("--base_model_code_dir", default=os.environ.get("OPENVLA_BASE_MODEL_DIR", ""))
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--attack_config", default="configs/paper_black_bowl_attack.yaml")
    ap.add_argument("--directions_config", default="configs/v4_directions.yaml")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--epsilon", default="0.10")
    ap.add_argument("--step_size", default="0.020")
    ap.add_argument("--attack_steps", default="20")
    ap.add_argument("--dry_run", action="store_true", help="Print the dispatched command without executing it.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    task = TASKS[args.task]
    condition = CONDITIONS[args.condition]
    condition_window = condition.get("window")
    if args.condition == "constant_delta_pregrasp" and (args.window_start is not None or args.window_end is not None):
        raise SystemExit("constant_delta_pregrasp is fixed to the 35-45 pregrasp/contact window; do not pass --window_start/--window_end.")
    if condition_window is not None:
        w0, w1 = condition_window
    else:
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
    env["V4_TARGET_OBJECT_NAME"] = task["object"]
    env["V4_TARGET_RECEPTACLE_NAME"] = task["receptacle"]
    env["V4_FIXED_ATTACK_START"] = str(w0)
    env["V4_FIXED_ATTACK_END"] = str(w1)
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
        "--epsilon", str(args.epsilon),
        "--step_size", str(args.step_size),
        "--attack_steps", str(args.attack_steps),
        "--temporal_init", str(condition["temporal_init"]),
        "--libero_official_preprocess",
        "--center_crop",
        "--postprocess_gripper",
        "--deterministic_init_states",
        "--state_ids", str(args.state),
    ]
    if condition["objective"]:
        cmd.extend(["--attack_objective", str(condition["objective"])])
    if condition["cw_margin"] is not None:
        cmd.extend(["--cw_margin", str(condition["cw_margin"])])
    if condition.get("force_open_raw_gripper") is not None:
        cmd.extend(["--force_open_raw_gripper", str(condition["force_open_raw_gripper"])])
    if condition.get("guard_enabled"):
        cmd.append("--guard_enabled")
        cmd.extend(["--guard_mode", str(condition.get("guard_mode", "conservative"))])
    if args.thresholds:
        cmd.extend(["--thresholds", args.thresholds])

    if args.dry_run:
        audit_env_keys = [
            "V4_FIXED_ATTACK_START",
            "V4_FIXED_ATTACK_END",
            "V4_TARGET_OBJECT_NAME",
            "V4_TARGET_RECEPTACLE_NAME",
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE",
            "V4_ORACLE_GRIPPER_PATTERN",
            "V4_CONSTANT_DELTA_GRIPPER",
        ]
        visible_env = [f"{key}={env[key]}" for key in audit_env_keys if key in env]
        if visible_env:
            print("ENV " + " ".join(visible_env))
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
