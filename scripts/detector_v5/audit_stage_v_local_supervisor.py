"""Read-only audit for a local Stage V supervisor run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from run_stage_v_local_supervisor import (
    SCHEMA,
    atomic_write_json,
    discover_workers,
    pid_alive,
    validate_workers,
)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _accepted_artifacts(value: Mapping[str, Any]) -> list[Any] | None:
    for key in ("accepted_parent_artifacts", "accepted_parents", "accepted_parent_results"):
        item = value.get(key)
        if isinstance(item, list):
            return item
    return None


def audit_run(
    run_root: Path,
    *,
    expected_source_commit: str | None = None,
    expected_source_tree: str | None = None,
    planned_parents: int | None = None,
    approved_gpus: list[int] | None = None,
    final: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    heartbeat = load_json(run_root / "LOCAL_HEARTBEAT.json")
    start = load_json(run_root / "SUPERVISOR_START.json")
    complete = load_json(run_root / "SUPERVISOR_COMPLETE.json")
    aborted = load_json(run_root / "ABORTED_INCOMPLETE.json")
    if not isinstance(heartbeat, Mapping):
        errors.append("heartbeat_missing_or_invalid")
    if not isinstance(start, Mapping):
        errors.append("supervisor_start_missing_or_invalid")
    if isinstance(heartbeat, Mapping):
        if heartbeat.get("schema") != SCHEMA:
            errors.append("heartbeat_schema_mismatch")
        if heartbeat.get("ssh_is_hard_stop") is not False:
            errors.append("ssh_hard_stop_must_be_false")
        if heartbeat.get("control_plane_mode") != "LOCAL_AUTONOMOUS":
            errors.append("control_plane_mode_mismatch")
        if expected_source_commit and heartbeat.get("source_commit") != expected_source_commit:
            errors.append("heartbeat_source_commit_mismatch")
        if expected_source_tree and heartbeat.get("source_tree") != expected_source_tree:
            errors.append("heartbeat_source_tree_mismatch")
        gpu_assignments = heartbeat.get("gpu_assignments", [])
        if not isinstance(gpu_assignments, list):
            errors.append("heartbeat_gpu_assignments_invalid")
        else:
            errors.extend(validate_workers(gpu_assignments, approved_gpus or [], require_live=False))
    if isinstance(start, Mapping):
        if expected_source_commit and start.get("source_commit") != expected_source_commit:
            errors.append("start_source_commit_mismatch")
        if expected_source_tree and start.get("source_tree") != expected_source_tree:
            errors.append("start_source_tree_mismatch")
        if planned_parents is not None and _int(start.get("planned_parents")) != planned_parents:
            errors.append("start_planned_parent_mismatch")

    supervisor_pid = _int(heartbeat.get("supervisor_pid")) if isinstance(heartbeat, Mapping) else None
    dispatcher_pid = _int(heartbeat.get("dispatcher_pid")) if isinstance(heartbeat, Mapping) else None
    active_worker_pids = heartbeat.get("active_worker_pids", []) if isinstance(heartbeat, Mapping) else []
    live_pids = [pid for pid in (supervisor_pid, dispatcher_pid) if pid_alive(pid)]
    live_pids.extend(pid for pid in active_worker_pids if isinstance(pid, int) and pid_alive(pid))
    if final and live_pids:
        errors.append(f"processes_still_alive:{sorted(set(live_pids))}")
    elif live_pids:
        warnings.append(f"healthy_live_processes:{sorted(set(live_pids))}")
    if final and isinstance(heartbeat, Mapping) and heartbeat.get("status") == "ABORTED_INCOMPLETE":
        errors.append("final_heartbeat_aborted")

    if approved_gpus is not None and isinstance(heartbeat, Mapping):
        errors.extend(validate_workers(heartbeat.get("gpu_assignments", []), approved_gpus, require_live=False))
    if isinstance(heartbeat, Mapping) and heartbeat.get("gpu_xid_status") == "XID_DETECTED":
        errors.append("gpu_xid_detected")
    if isinstance(heartbeat, Mapping) and heartbeat.get("resource_errors"):
        errors.extend(str(item) for item in heartbeat["resource_errors"])
    for key in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts"):
        if isinstance(complete, Mapping) and _int(complete.get(key)) not in (None, 0):
            errors.append(f"{key}_not_zero")
        if isinstance(aborted, Mapping) and _int(aborted.get(key)) not in (None, 0):
            errors.append(f"{key}_not_zero")

    if final:
        if isinstance(aborted, Mapping):
            if _int(aborted.get("accepted_parent_results")) != 0:
                errors.append("aborted_root_accepted_parent_count_not_zero")
            if isinstance(complete, Mapping):
                errors.append("complete_and_aborted_markers_both_present")
        elif not isinstance(complete, Mapping):
            errors.append("completion_marker_missing")
        if isinstance(complete, Mapping):
            if complete.get("status") != "PASS":
                errors.append("completion_status_not_pass")
            accepted = _int(complete.get("accepted_parent_results"))
            if planned_parents is not None and accepted != planned_parents:
                errors.append(f"accepted_parent_count:{accepted}/{planned_parents}")
            artifacts = _accepted_artifacts(complete)
            if artifacts is None:
                errors.append("accepted_parent_artifact_audit_missing")
            else:
                for index, item in enumerate(artifacts):
                    if not isinstance(item, Mapping) or item.get("artifact_audit_verdict") != "PASS":
                        errors.append(f"accepted_parent_artifact_audit_failed:{index}")
        if isinstance(heartbeat, Mapping):
            if _int(heartbeat.get("accepted_parent_results")) != (
                _int(complete.get("accepted_parent_results")) if isinstance(complete, Mapping) else 0
            ):
                errors.append("heartbeat_completion_acceptance_mismatch")
    report = {
        "schema": "STAGE_V_LOCAL_SUPERVISOR_AUDIT_V1",
        "verdict": "PASS" if not errors else "FAIL",
        "run_root": str(run_root),
        "final": final,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "control_plane_mode": "LOCAL_AUTONOMOUS",
        "ssh_is_hard_stop": False,
        "supervisor_pid": supervisor_pid,
        "dispatcher_pid": dispatcher_pid,
        "heartbeat_count": _int(heartbeat.get("heartbeat_count")) if isinstance(heartbeat, Mapping) else None,
        "ssh_probe_failure_count": _int(heartbeat.get("ssh_probe_failure_count")) if isinstance(heartbeat, Mapping) else None,
        "accepted_parent_results": _int(complete.get("accepted_parent_results")) if isinstance(complete, Mapping) else 0,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
    }
    if write_report:
        atomic_write_json(run_root / "LOCAL_SUPERVISOR_AUDIT.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    parser.add_argument("--planned-parents", type=int)
    parser.add_argument("--approved-gpus", type=lambda raw: [int(part) for part in raw.split(",") if part])
    parser.add_argument("--live", action="store_true", help="audit a live run without requiring dead PIDs")
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_run(
        args.run_root,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
        planned_parents=args.planned_parents,
        approved_gpus=args.approved_gpus,
        final=not args.live,
        write_report=not args.no_write,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
