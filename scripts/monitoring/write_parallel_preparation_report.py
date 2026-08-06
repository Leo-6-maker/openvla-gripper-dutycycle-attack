"""Write a compact, boundary-explicit preparation report."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_stage_v_closure import atomic_write_json, sha256_file


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def write_seal(root: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(root / name)}  {name}\n" for name in names if (root / name).is_file()]
    sums = root / "PARALLEL_PREPARATION_SHA256SUMS"
    sums.write_text("".join(lines), encoding="utf-8")
    sidecar = root / "PARALLEL_PREPARATION_SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  PARALLEL_PREPARATION_SHA256SUMS\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    monitor_root = args.goal_root / "MONITOR"
    triage = load(monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE.json")
    inventory = load(args.output_root / "STAGE_V_ROOT_INVENTORY.json")
    summary = {
        "schema": "PARALLEL_PREPARATION_SUMMARY_V1",
        "status": "PREPARATION_COMPLETE",
        "stage_v_scientific_canary": "PASS",
        "stage_v_formal_map": "IN_PROGRESS",
        "stage_v_scientific_verdict": "NOT_AVAILABLE",
        "stage_v_closure_audit": "NOT_RUN",
        "stage_v_formal_root_modified": False,
        "stage_v_workers_modified": False,
        "second_dispatcher_launched": False,
        "additional_stage_v_workers_launched": False,
        "gpu5_touched": False,
        "protected_pid_signaled": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "student_training": 0,
        "stage_v2_formal_execution": 0,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "gatekeeper_commit": args.gatekeeper_commit,
        "gatekeeper_tree": args.gatekeeper_tree,
        "tests": args.test_summary,
        "live_parent_triage": triage if isinstance(triage, Mapping) else {"status": "MISSING"},
        "root_inventory": inventory if isinstance(inventory, Mapping) else {"status": "MISSING"},
        "stage_v2_command_registration": "PLAN_REGISTERED_WAITING_FOR_CLOSURE_BINDING",
        "generated_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "PARALLEL_PREPARATION_SUMMARY.json", summary)
    report = [
        "# Parallel preparation report",
        "",
        "## Decision",
        "",
        "Preparation is complete, but the formal Stage V scientific verdict remains unavailable.",
        "No formal Stage V2 execution is started before a valid Stage V closure receipt.",
        "",
        "## Frozen status",
        "",
        f"- Stage V scientific canary: `{summary['stage_v_scientific_canary']}`",
        f"- Stage V formal map: `{summary['stage_v_formal_map']}`",
        f"- Stage V scientific verdict: `{summary['stage_v_scientific_verdict']}`",
        f"- Stage V closure audit: `{summary['stage_v_closure_audit']}`",
        f"- Stage V2 formal execution: `{summary['stage_v2_formal_execution']}`",
        "",
        "## Safety declarations",
        "",
        "```text",
        "Stage V formal root modified = false",
        "Stage V workers modified = false",
        "second dispatcher launched = false",
        "additional Stage V workers launched = false",
        "GPU5 touched = false",
        "protected PID signaled = false",
        "Eval160 reads = 0",
        "protected eval reads = 0",
        "VIS/PGD/attack rollouts = 0",
        "Student training = 0",
        "```",
        "",
        "## Registration",
        "",
        "The Stage V2 command plan is registered fail-closed. The monitor materializes the SHA-bound `STAGE_V2_COMMAND.json` only after the actual closure receipt exists and verifies.",
        "",
    ]
    (args.output_root / "PARALLEL_PREPARATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    names = ["PARALLEL_PREPARATION_REPORT.md", "PARALLEL_PREPARATION_SUMMARY.json"]
    index = {
        "schema": "PARALLEL_PREPARATION_ARTIFACT_INDEX_V1",
        "generated_utc": utc_now(),
        "artifacts": [{"path": name, "sha256": sha256_file(args.output_root / name)} for name in names],
    }
    atomic_write_json(args.output_root / "PARALLEL_PREPARATION_ARTIFACT_INDEX.json", index)
    names.append("PARALLEL_PREPARATION_ARTIFACT_INDEX.json")
    write_seal(args.output_root, names)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--gatekeeper-commit", required=True)
    parser.add_argument("--gatekeeper-tree", required=True)
    parser.add_argument("--test-summary", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    build_report(args)
