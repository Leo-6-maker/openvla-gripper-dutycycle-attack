#!/usr/bin/env python3
"""Independent CPU auditor for M3 arm-v5.2 fixed-frame candidate groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.m3_v5_attack_harness import audit_frame_group, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_root", required=True)
    ap.add_argument("--frame_ids", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--audit_output", default="")
    args = ap.parse_args()
    frames = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    try:
        result = audit_frame_group(Path(args.artifact_root), frame_ids=frames, seed=int(args.seed))
    except Exception as exc:
        result = {
            "audit_status": "FAIL",
            "failure_reason": repr(exc),
            "artifact_root": str(args.artifact_root),
            "seed": int(args.seed),
            "frame_ids": frames,
        }
    output = Path(args.audit_output) if args.audit_output else Path(args.artifact_root) / "m3_arm_v5_frame_group_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["audit_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
