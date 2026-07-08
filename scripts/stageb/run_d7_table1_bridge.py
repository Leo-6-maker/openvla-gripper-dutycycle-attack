#!/usr/bin/env python3
"""D7 Table1 cross-suite bridge wrapper.

Parameterizes run_v2_vis_sc5_mlp_bridge.py for all four LIBERO suites.
Key changes from Object-only bridge:
  --suite: libero_10, libero_goal, libero_object, libero_spatial
  --model_path, --unnorm_key: parameterized per suite
  --task_idx, --state_id, --seed_id: from D7 manifest
  Object site telemetry: best-effort optional (no crash on missing).

Attack logic: VIS/RAND from existing bridge, detector trigger from C2e3.

DO NOT run this directly — use the D7 manifest queue to drive batch execution.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, Optional

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ============ Suite Configuration ============
SUITE_CONFIG: Dict[str, Dict[str, Any]] = {
    "libero_10": {
        "unnorm_key": "libero_10",
        "action_dim": 7,
        "benchmark_init": "libero_10",
        "task_range": range(0, 10),
        "model_path_default": "",  # Must be provided via --model-path
    },
    "libero_goal": {
        "unnorm_key": "libero_goal",
        "action_dim": 7,
        "benchmark_init": "libero_goal",
        "task_range": range(0, 10),
        "model_path_default": "",
    },
    "libero_object": {
        "unnorm_key": "libero_object",
        "action_dim": 7,
        "benchmark_init": "libero_object",
        "task_range": range(0, 10),
        "model_path_default": "",
    },
    "libero_spatial": {
        "unnorm_key": "libero_spatial",
        "action_dim": 7,
        "benchmark_init": "libero_spatial",
        "task_range": range(0, 10),
        "model_path_default": "",
    },
}

# ============ Condition Protocol ============
CONDITION_PROTOCOLS = {
    "CLEAN": {
        "attack": False,
        "intervention": "none",
        "objective": "clean_baseline",
        "timing": "n/a",
        "eval": "ITT",
    },
    "TRUE_T10": {
        "attack": True,
        "intervention": "force_gripper_open_token_ce",
        "objective": "targeted_gripper_duty_cycle",
        "timing": "detector_trigger",
        "eval": "ITT",
        "epsilon": 0.25,
        "step_size": 0.050,
        "attack_steps": 60,
        "force_open_raw_gripper": 1.0,
    },
    "RAND_T10": {
        "attack": True,
        "intervention": "random_direction_payload",
        "objective": "direction_specificity_control",
        "timing": "detector_trigger_same",
        "eval": "ITT",
        "attack_steps": 60,
    },
    "COMMAND_OPEN_ORACLE": {
        "attack": True,
        "intervention": "command_open_oracle",
        "objective": "mechanistic_upper_bound",
        "timing": "detector_trigger_same",
        "eval": "emission_matched",
        "attack_steps": 60,
    },
}

# ============ Episode Summary Template ============
def build_episode_summary(
    suite: str,
    task_idx: int,
    state_id: int,
    seed: int,
    condition: str,
    parent_key: str,
    success: bool,
    n_steps: int,
    detector_emitted: bool,
    emit_step: int,
    attack_frames: int,
    token_open_duty: float,
    env_open_duty: float,
    arm_duty: float,
    qpos_open_response: float,
    failure_taxonomy: str,
    source_commit: str,
    detector_sha256: str,
    threshold_sha256: str,
    **extra,
) -> Dict[str, Any]:
    return {
        "suite": suite,
        "task_idx": task_idx,
        "state_id": state_id,
        "seed": seed,
        "condition": condition,
        "clean_parent_key": parent_key,
        "task_success": success,
        "n_steps": n_steps,
        "detector_emitted": detector_emitted,
        "emit_step": emit_step,
        "attack_frames": attack_frames,
        "token_open_duty": token_open_duty,
        "env_open_duty": env_open_duty,
        "arm_duty": arm_duty,
        "qpos_open_response": qpos_open_response,
        "failure_taxonomy": failure_taxonomy,
        "source_commit": source_commit,
        "detector_checkpoint_sha256": detector_sha256,
        "threshold_sha256": threshold_sha256,
    }


def main():
    """CLI entry point — single-episode bridge invocation."""
    ap = argparse.ArgumentParser(description="D7 Table1 cross-suite bridge (single episode)")
    ap.add_argument("--suite", required=True, choices=list(SUITE_CONFIG.keys()))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--task-idx", type=int, required=True)
    ap.add_argument("--state-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", required=True, choices=list(CONDITION_PROTOCOLS.keys()))
    ap.add_argument("--parent-key", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--detector-checkpoint", required=True)
    ap.add_argument("--threshold-json", required=True)
    ap.add_argument("--trigger-step-override", type=int, default=None)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    suite_cfg = SUITE_CONFIG[args.suite]
    unnorm_key = args.unnorm_key or suite_cfg["unnorm_key"]
    protocol = CONDITION_PROTOCOLS[args.condition]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({
            "suite": args.suite,
            "condition": args.condition,
            "parent_key": args.parent_key,
            "task_idx": args.task_idx,
            "state_id": args.state_id,
            "seed": args.seed,
            "unnorm_key": unnorm_key,
            "model_path": args.model_path,
            "protocol": protocol,
            "detector_checkpoint": args.detector_checkpoint,
            "trigger_step_override": args.trigger_step_override,
            "dry_run": True,
        }, indent=2))
        return 0

    # === Actual bridge execution ===
    # This would call into the existing VIS/attack pipeline.
    # For now, output the parameterized invocation record.
    invocation = {
        "bridge": "run_d7_table1_bridge.py",
        "suite": args.suite,
        "condition": args.condition,
        "parent_key": args.parent_key,
        "task_idx": args.task_idx,
        "state_id": args.state_id,
        "seed": args.seed,
        "unnorm_key": unnorm_key,
        "model_path": args.model_path,
        "detector_checkpoint": args.detector_checkpoint,
        "threshold_json": args.threshold_json,
        "trigger_step_override": args.trigger_step_override,
        "protocol": protocol,
        "source_commit": args.source_commit,
        "timestamp_unix": time.time(),
    }
    invocation_path = out_dir / "bridge_invocation.json"
    invocation_path.write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"status": "INVOCATION_RECORDED", "suite": args.suite,
                      "condition": args.condition, "parent_key": args.parent_key,
                      "output": str(invocation_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
