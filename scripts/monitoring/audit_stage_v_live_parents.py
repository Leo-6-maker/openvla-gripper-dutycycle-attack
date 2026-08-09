"""Read-only triage for Stage V parents while the producer is still running.

This tool only writes its own report.  It never edits a parent artifact, starts
or stops a worker, or turns an intermediate producer status into a closure
verdict.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.monitoring.audit_stage_v_closure import atomic_write_json, parent_progress


SCHEMA = "STAGE_V_LIVE_PARENT_TRIAGE_V1"
ACTIVE_STATUSES = {"RUNNING", "IN_PROGRESS", "PENDING", "STARTED", "WRITING"}
FAIL_STATUSES = {"FAIL", "FAILED", "ERROR", "ABORTED", "CANCELLED"}
KNOWN_STATES = {
    "RUNNING_WRITING",
    "RUNNING_STALLED",
    "QUIESCENT_INCOMPLETE",
    "TERMINAL_PRODUCER_FAIL",
    "BRANCH_COMPLETE_PENDING_AUDIT",
    "DUPLICATE_IDENTITY",
    "UNKNOWN",
}


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        return subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        ).returncode == 0
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return False


def _process_command(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _worker_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pid_file in sorted(root.glob("worker_gpu*.pid")):
        match = re.fullmatch(r"worker_gpu(\d+)\.pid", pid_file.name)
        if not match:
            continue
        pid = _int(pid_file.read_text(encoding="utf-8", errors="replace").strip())
        if pid is None:
            continue
        exit_path = root / f"worker_gpu{match.group(1)}.exitcode"
        exit_code = None
        if exit_path.is_file():
            exit_code = _int(exit_path.read_text(encoding="utf-8", errors="replace").strip())
        records.append(
            {
                "pid": pid,
                "gpu": int(match.group(1)),
                "alive": _pid_alive(pid),
                "command": _process_command(pid),
                "exit_code": exit_code,
            }
        )
    return records


def _heartbeat_mentions(heartbeat: Any, key: str) -> bool:
    if not isinstance(heartbeat, Mapping):
        return False
    encoded = json.dumps(heartbeat, sort_keys=True)
    return key in encoded


def _artifact_snapshot(parent_dir: Path) -> dict[str, Any]:
    latest = 0.0
    file_count = 0
    temporary_files: list[str] = []
    try:
        for path in parent_dir.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            latest = max(latest, path.stat().st_mtime)
            if path.name.endswith((".tmp", ".partial", ".inprogress")) or ".tmp." in path.name:
                temporary_files.append(str(path.relative_to(parent_dir)))
    except OSError:
        return {"latest_mtime": None, "file_count": file_count, "temporary_files": temporary_files, "read_error": True}
    return {"latest_mtime": latest or None, "file_count": file_count, "temporary_files": sorted(temporary_files)}


def _branch_counts(parent_dir: Path, result: Mapping[str, Any] | None) -> tuple[int | None, int]:
    probe_count = _int(result.get("probe_count")) if isinstance(result, Mapping) else None
    expected = probe_count * 3 if probe_count and probe_count > 0 else None
    branch_count = _int(result.get("branch_count")) if isinstance(result, Mapping) else None
    if branch_count is None:
        branch_file = parent_dir / "COUNTERFACTUAL_BRANCHES.jsonl"
        if branch_file.is_file():
            try:
                branch_count = sum(1 for line in branch_file.open(encoding="utf-8") if line.strip())
            except OSError:
                branch_count = 0
    return expected, branch_count or 0


def _classify(
    *,
    parent_dir: Path,
    result: Mapping[str, Any] | None,
    worker_records: list[Mapping[str, Any]],
    heartbeat: Any,
    now: float,
    recent_seconds: float,
    terminal_quiescent_seconds: float,
) -> tuple[str, dict[str, Any]]:
    snapshot = _artifact_snapshot(parent_dir)
    latest = snapshot.get("latest_mtime")
    age = None if latest is None else max(0.0, now - float(latest))
    key = str(result.get("canonical_parent_key", "")) if isinstance(result, Mapping) else parent_dir.as_posix()
    matching = [item for item in worker_records if key and key in str(item.get("command", ""))]
    alive_workers = [item for item in matching if item.get("alive")]
    result_status = str(result.get("status", "")).upper() if isinstance(result, Mapping) else "MISSING"
    expected, completed = _branch_counts(parent_dir, result)
    branch_complete = expected is not None and completed == expected
    active = bool(alive_workers) or _heartbeat_mentions(heartbeat, key)
    recent = age is not None and age <= recent_seconds
    details: dict[str, Any] = {
        "canonical_parent_key": key,
        "parent_dir": str(parent_dir),
        "producer_status": result_status,
        "expected_branch_count": expected,
        "completed_branch_count": completed,
        "branch_complete": branch_complete,
        "worker_records": matching,
        "artifact": {**snapshot, "age_seconds": age},
        "recent_write": recent,
        "producer_active": active,
        "parent_result_exists": result is not None,
    }
    if not isinstance(result, Mapping):
        if active or recent:
            return ("RUNNING_WRITING" if recent else "RUNNING_STALLED"), details
        return "QUIESCENT_INCOMPLETE", details
    if len(matching) > 1:
        return "DUPLICATE_IDENTITY", details
    if active:
        return ("RUNNING_WRITING" if recent else "RUNNING_STALLED"), details
    if result_status == "PASS" and branch_complete:
        return "BRANCH_COMPLETE_PENDING_AUDIT", details
    if recent:
        return "RUNNING_WRITING", details
    if result_status in ACTIVE_STATUSES:
        return "RUNNING_STALLED" if age is not None and age > terminal_quiescent_seconds else "RUNNING_WRITING", details
    exit_code = _int(result.get("exit_code"))
    if result_status in FAIL_STATUSES or (exit_code is not None and exit_code != 0) or not branch_complete:
        return "TERMINAL_PRODUCER_FAIL", details
    return "UNKNOWN", details


def triage(
    stage_v_root: Path,
    parent_manifest: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    only_nonpass: bool = True,
    recent_seconds: float = 300.0,
    terminal_quiescent_seconds: float = 600.0,
) -> dict[str, Any]:
    progress = parent_progress(
        stage_v_root,
        parent_manifest=parent_manifest,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        full_audit=False,
    )
    start = load_json(stage_v_root / "SUPERVISOR_START.json")
    heartbeat = load_json(stage_v_root / "LOCAL_HEARTBEAT.json")
    worker_records = _worker_records(stage_v_root)
    now = time.time()
    items: list[dict[str, Any]] = []
    for row in progress.get("parents", []):
        if only_nonpass and row.get("audit_status") != "FAIL":
            continue
        parent_dir = stage_v_root / str(row.get("canonical_parent_key", ""))
        result = load_json(parent_dir / "PARENT_RESULT.json")
        if "duplicate_parent_identity" in row.get("errors", []):
            state, details = "DUPLICATE_IDENTITY", {"canonical_parent_key": row.get("canonical_parent_key"), "parent_dir": str(parent_dir), "producer_status": "DUPLICATE"}
        else:
            state, details = _classify(
                parent_dir=parent_dir,
                result=result if isinstance(result, Mapping) else None,
                worker_records=worker_records,
                heartbeat=heartbeat,
                now=now,
                recent_seconds=recent_seconds,
                terminal_quiescent_seconds=terminal_quiescent_seconds,
            )
        details["state"] = state
        items.append(details)
    return {
        "schema": SCHEMA,
        "status": "RUNNING" if not (stage_v_root / "SUPERVISOR_COMPLETE.json").is_file() else "PRODUCER_COMPLETE",
        "scientific_verdict": "NOT_AVAILABLE",
        "formal_map_root": str(stage_v_root),
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "parent_manifest": str(parent_manifest),
        "parent_progress": {
            key: progress.get(key)
            for key in (
                "planned_parent_count",
                "started_parent_count",
                "branch_complete_parent_count",
                "audited_parent_count",
                "accepted_parent_count",
                "invalid_parent_count",
                "missing_branch_count",
                "duplicate_identity_count",
                "orphan_artifact_count",
            )
        },
        "triaged_count": len(items),
        "parents": items,
        "generated_utc": utc_now(),
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage V live parent triage",
        "",
        f"- status: `{report.get('status')}`",
        "- scientific verdict: `NOT_AVAILABLE`",
        f"- triaged parents: `{report.get('triaged_count', 0)}`",
        "",
        "| parent | state | producer | worker alive | recent write | branches |",
        "|---|---|---|---|---|---:|",
    ]
    for item in report.get("parents", []):
        workers = item.get("worker_records", [])
        lines.append(
            f"| `{item.get('canonical_parent_key')}` | `{item.get('state')}` | "
            f"`{item.get('producer_status')}` | `{any(w.get('alive') for w in workers)}` | "
            f"`{item.get('recent_write')}` | {item.get('completed_branch_count', 0)}/{item.get('expected_branch_count') or '?'} |"
        )
    lines.extend(["", "Intermediate states are diagnostic only; no closure verdict is inferred while the producer runs.", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--monitor-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--all-parents", action="store_true")
    parser.add_argument("--recent-seconds", type=float, default=300.0)
    parser.add_argument("--terminal-quiescent-seconds", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.recent_seconds < 0 or args.terminal_quiescent_seconds < args.recent_seconds:
        raise SystemExit("invalid quiescence thresholds")
    report = triage(
        args.stage_v_root.resolve(),
        args.parent_manifest.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
        only_nonpass=not args.all_parents,
        recent_seconds=args.recent_seconds,
        terminal_quiescent_seconds=args.terminal_quiescent_seconds,
    )
    atomic_write_json(args.monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE.json", report)
    atomic_write_text(args.monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE.md", markdown(report))
    print(json.dumps({"status": report["status"], "triaged_count": report["triaged_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
