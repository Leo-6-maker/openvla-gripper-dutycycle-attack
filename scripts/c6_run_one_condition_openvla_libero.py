#!/usr/bin/env python3
"""Fail-closed C6 one-condition shim for legacy OpenVLA/LIBERO runners."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRIMARY = {"libero_goal", "libero_object", "libero_spatial"}
CONDS = {"CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"}


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-id", required=True)
    ap.add_argument("--episode-key", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--legacy-runner", default="scripts/v4_run_eval_openvla.py")
    ap.add_argument("--model-path")
    ap.add_argument("--benchmark-root")
    ap.add_argument("--seed", type=int, default=2026070401)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.suite not in PRIMARY:
        raise SystemExit(f"unsupported suite for C6 primary shim: {args.suite}")
    if args.condition not in CONDS:
        raise SystemExit(f"unsupported C6 condition: {args.condition}")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "HOLD_PARENT_RESET_UNBOUND",
        "parent_id": args.parent_id,
        "episode_key": args.episode_key,
        "suite": args.suite,
        "task_id": args.task_id,
        "condition": args.condition,
        "legacy_runner": args.legacy_runner,
        "work_dir": str(work),
        "reason": (
            "Legacy runner is not yet bound to Clean2000 parent reset, exact-prefix "
            "condition execution, or C6 metric extraction. Refusing to run instead "
            "of fabricating legacy_result_json."
        ),
        "boundary": {
            "OpenVLA": "NOT_PERFORMED",
            "LIBERO": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
        },
    }
    write_json(args.output_json, report)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
