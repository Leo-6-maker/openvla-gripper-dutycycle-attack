#!/usr/bin/env python3
"""CPU/mock entrypoint for M3 arm-v5.2 frame-group artifact contracts.

This script does not run OpenVLA, PGD, RAND, shuffled-gradient, or LIBERO. It is
used to smoke-test the fixed-frame artifact layout before real GPU execution is
authorized after V5.1 input freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.m3_v5_attack_harness import (  # noqa: E402
    V5_2_CANDIDATE_COUNT,
    V5_2_CONDITIONS,
    V5_2_FROZEN_SEED,
    audit_frame_group,
    write_candidate_artifact,
    write_json,
)


def mock_payload(*, condition: str, candidate_index: int) -> dict[str, object]:
    base_margin = {
        "TRUE_PGD21_SELECTIVE": 10.0,
        "RAND21_SELECTIVE": 1.0,
        "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE": 0.5,
    }[condition]
    return {
        "official_gripper_token": 31744,
        "official_exact_7_tokens": [1, 2, 3, 4, 5, 6, 31744],
        "arm_match_count": 6,
        "official_target_margin": base_margin + float(candidate_index) / 100.0,
        "linf": 0.0,
        "score_invariant_status": "PASS",
        "route_status": "PASS",
        "libero_rollout_used": False,
        "candidate_mode": "CPU_MOCK_ZERO_PERTURBATION",
    }


def run_mock_zero(args: argparse.Namespace) -> None:
    if int(args.seed) != V5_2_FROZEN_SEED:
        raise SystemExit(f"mock V5.2 harness requires frozen seed {V5_2_FROZEN_SEED}")
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise SystemExit("--output_dir must be new or empty")
    frames = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    for frame_id in frames:
        for condition in V5_2_CONDITIONS:
            for idx in range(V5_2_CANDIDATE_COUNT):
                write_candidate_artifact(root, frame_id=frame_id, condition=condition, candidate_index=idx, payload=mock_payload(condition=condition, candidate_index=idx))
    result = audit_frame_group(root, frame_ids=frames, seed=int(args.seed))
    write_json(root / "m3_arm_v5_frame_group_mock_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock_zero_perturbation"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frame_ids", required=True)
    ap.add_argument("--seed", type=int, default=V5_2_FROZEN_SEED)
    args = ap.parse_args()
    if args.mode == "mock_zero_perturbation":
        run_mock_zero(args)


if __name__ == "__main__":
    main()
