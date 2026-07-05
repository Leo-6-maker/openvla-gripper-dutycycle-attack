#!/usr/bin/env python3
"""Fail-closed C6 one-condition shim for legacy OpenVLA/LIBERO runners."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PRIMARY = {"libero_goal", "libero_object", "libero_spatial"}
CONDS = {"CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"}
LEGACY_TASK_ID_BY_SUITE_INDEX = {
    ("libero_goal", 1): "libero_goal_open_middle_drawer",
    ("libero_object", 1): "libero_object_alphabet_soup",
    ("libero_spatial", 1): "libero_spatial_black_bowl",
}


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_suffix_task_index(parent_id: str) -> int | None:
    m = re.search(r"(?:^|/)task_(\d+)(?:/|$)", str(parent_id or ""))
    if not m:
        return None
    return int(m.group(1))


def parse_task_index(args: argparse.Namespace) -> int | None:
    from_parent = parent_suffix_task_index(args.parent_id)
    if from_parent is not None:
        return from_parent
    try:
        return int(str(args.task_id))
    except ValueError:
        return None


def resolve_legacy_task_id(args: argparse.Namespace) -> str:
    override = str(getattr(args, "legacy_task_id", "") or "").strip()
    if override:
        return override
    task_index = parse_task_index(args)
    if task_index is not None:
        mapped = LEGACY_TASK_ID_BY_SUITE_INDEX.get((str(args.suite), int(task_index)))
        if mapped:
            return mapped
    return str(args.task_id)


def base_report(args: argparse.Namespace, work: Path) -> dict:
    legacy_task_id = resolve_legacy_task_id(args)
    return {
        "parent_id": args.parent_id,
        "episode_key": args.episode_key,
        "suite": args.suite,
        "task_id": args.task_id,
        "legacy_task_id": legacy_task_id,
        "condition": args.condition,
        "initial_state_hash": args.initial_state_hash,
        "state_id": args.state_id,
        "legacy_runner": args.legacy_runner,
        "work_dir": str(work),
    }


def not_performed_boundary() -> dict:
    return {
        "legacy_runner_execution": "NOT_PERFORMED",
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "attack_condition": "NOT_PERFORMED",
    }


def legacy_runner_argv_preview(args: argparse.Namespace, work: Path) -> list[str]:
    legacy_task_id = resolve_legacy_task_id(args)
    argv = [
        sys.executable,
        args.legacy_runner,
        "--task_id",
        str(legacy_task_id),
        "--trigger",
        "state_id_binding_preview_only",
        "--rho",
        "0.0",
        "--episodes",
        "1",
        "--output_root",
        str(work / "legacy_preview"),
        "--dry_run",
    ]
    if args.state_id is not None:
        argv.extend(["--state_ids", str(args.state_id)])
    if args.model_path:
        argv.extend(["--model_path", str(args.model_path)])
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-id", required=True)
    ap.add_argument("--episode-key", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--legacy-task-id", default="")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--legacy-runner", default="scripts/v4_run_eval_openvla.py")
    ap.add_argument("--model-path")
    ap.add_argument("--benchmark-root")
    ap.add_argument("--initial-state-hash")
    ap.add_argument("--state-id", type=int)
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

    if args.dry_run:
        report = base_report(args, work)
        if not args.initial_state_hash:
            report.update(
                {
                    "status": "HOLD_RESET_FIELD_MISSING",
                    "reason": "Dry-run reset binding proof requires --initial-state-hash.",
                    "boundary": not_performed_boundary(),
                    "boundaries": not_performed_boundary(),
                }
            )
            write_json(args.output_json, report)
            return 2
        status = "PASS_SHIM_DRY_RUN_STATE_ID_BOUND" if args.state_id is not None else "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND"
        report.update(
            {
                "status": status,
                "legacy_runner_execution": "NOT_PERFORMED",
                "state_id_binding": {
                    "provided": args.state_id is not None,
                    "state_id": args.state_id,
                    "binding_mode": "DRY_RUN_METADATA_ONLY" if args.state_id is not None else "NOT_PROVIDED",
                },
                "legacy_task_binding": {
                    "source_task_id": str(args.task_id),
                    "source_parent_id": str(args.parent_id),
                    "legacy_task_id": resolve_legacy_task_id(args),
                    "binding_mode": "STATIC_PARENT_TASK_TO_LEGACY_TASK_ID",
                },
                "legacy_runner_argv_preview": legacy_runner_argv_preview(args, work),
                "legacy_runner_argv_preview_mode": "NOT_EXECUTED_DRY_RUN_METADATA_ONLY",
                "boundary": not_performed_boundary(),
                "boundaries": not_performed_boundary(),
                "raw_logs": {
                    "work_dir": str(work),
                    "legacy_runner": args.legacy_runner,
                    "trace_files": [],
                },
            }
        )
        write_json(args.output_json, report)
        return 0

    report = base_report(args, work)
    report.update(
        {
            "status": "HOLD_PARENT_RESET_UNBOUND",
            "reason": (
                "Legacy runner is not yet bound to Clean2000 parent reset, exact-prefix "
                "condition execution, or C6 metric extraction. Refusing to run instead "
                "of fabricating legacy_result_json."
            ),
            "boundary": not_performed_boundary(),
            "boundaries": not_performed_boundary(),
        }
    )
    write_json(args.output_json, report)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
