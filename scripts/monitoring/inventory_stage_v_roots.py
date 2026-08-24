"""Inventory Stage V roots without changing any experiment artifact."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.monitoring.audit_stage_v_closure import atomic_write_json, sha256_file


SCHEMA = "STAGE_V_ROOT_INVENTORY_V1"
ROOT_RE = re.compile(r"^STAGE_V_COUNTERFACTUAL_MAP_[A-Za-z0-9]+_\d{8}T\d{6}Z$")
METADATA_NAMES = {
    "RUN_MANIFEST.json",
    "SUPERVISOR_START.json",
    "SUPERVISOR_COMPLETE.json",
    "LOCAL_HEARTBEAT.json",
    "ABORTED_INCOMPLETE.json",
    "HARD_STOP_RECEIPT.json",
    "STAGE_V_CLOSURE_RECEIPT.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
}


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


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


def _sum_bytes(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return total
    return total


def _sha_lines(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            token = line.split(None, 1)[0] if line.split(None, 1) else ""
            if re.fullmatch(r"[0-9a-fA-F]{64}", token):
                values.add(token.lower())
    except OSError:
        pass
    return values


def _metadata_text(root: Path) -> str:
    chunks: list[str] = []
    for name in sorted(METADATA_NAMES):
        path = root / name
        if path.is_file():
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    monitor = root / "MONITOR"
    for name in ("STAGE_V_CLOSURE_AUDIT.json", "STAGE_V_MONITOR_STATE.json"):
        path = monitor / name
        if path.is_file():
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _parent_counts(root: Path) -> dict[str, int]:
    progress = load_json(root / "MONITOR" / "STAGE_V_PARENT_PROGRESS.json")
    if isinstance(progress, Mapping):
        return {
            key: _int(progress.get(key)) or 0
            for key in (
                "planned_parent_count",
                "started_parent_count",
                "branch_complete_parent_count",
                "audited_parent_count",
                "accepted_parent_count",
                "invalid_parent_count",
                "missing_branch_count",
            )
        }
    started = 0
    try:
        started = sum(1 for _ in root.rglob("PARENT_RESULT.json"))
    except OSError:
        pass
    return {
        "planned_parent_count": 0,
        "started_parent_count": started,
        "branch_complete_parent_count": 0,
        "audited_parent_count": 0,
        "accepted_parent_count": 0,
        "invalid_parent_count": 0,
        "missing_branch_count": 0,
    }


def _classify(root: Path, active_root: Path) -> str:
    if root.resolve() == active_root.resolve():
        return "ACTIVE_FORMAL_ROOT"
    if (root / "DIAGNOSTIC_CANARY_ONLY").exists() or "CANARY" in root.name.upper():
        return "CLOSED_DIAGNOSTIC_CANARY"
    aborted = load_json(root / "ABORTED_INCOMPLETE.json")
    if isinstance(aborted, Mapping):
        reason = json.dumps(aborted, sort_keys=True).lower()
        if any(token in reason for token in ("control_plane", "control-plane", "ssh", "lock", "monitor")):
            return "ABORTED_INCOMPLETE_CONTROL_PLANE"
        if any(token in reason for token in ("oom", "xid", "worker", "filesystem", "provenance")):
            return "ABORTED_INCOMPLETE_ENGINEERING"
        return "ABORTED_INCOMPLETE_UNKNOWN"
    if (root / "STAGE_V_CLOSURE_RECEIPT.json").is_file():
        return "OTHER"
    return "OTHER"


def inventory(goal_root: Path, active_root: Path) -> dict[str, Any]:
    roots = [p for p in sorted(goal_root.iterdir()) if p.is_dir() and ROOT_RE.match(p.name)]
    active_text = _metadata_text(active_root)
    active_shas = _sha_lines(active_root / "SHA256SUMS")
    records: list[dict[str, Any]] = []
    for root in roots:
        run_manifest = load_json(root / "RUN_MANIFEST.json")
        start = load_json(root / "SUPERVISOR_START.json")
        complete = load_json(root / "SUPERVISOR_COMPLETE.json")
        aborted = load_json(root / "ABORTED_INCOMPLETE.json")
        closure = load_json(root / "STAGE_V_CLOSURE_RECEIPT.json")
        old_shas = _sha_lines(root / "SHA256SUMS")
        path_reference = str(root) in active_text if root.resolve() != active_root.resolve() else False
        source_record = run_manifest if isinstance(run_manifest, Mapping) else start if isinstance(start, Mapping) else {}
        records.append(
            {
                "root": str(root.resolve()),
                "name": root.name,
                "created_utc": _datetime.datetime.fromtimestamp(root.stat().st_ctime, _datetime.timezone.utc).isoformat(),
                "modified_utc": _datetime.datetime.fromtimestamp(root.stat().st_mtime, _datetime.timezone.utc).isoformat(),
                "classification": _classify(root, active_root),
                "source_commit": source_record.get("source_commit"),
                "source_tree": source_record.get("source_tree"),
                "parent_manifest_sha256": (run_manifest or {}).get("parent_manifest_sha256") if isinstance(run_manifest, Mapping) else None,
                "run_manifest_sha256": sha256_file(root / "RUN_MANIFEST.json") if (root / "RUN_MANIFEST.json").is_file() else None,
                "planned_started_completed": _parent_counts(root),
                "markers": {
                    "aborted_incomplete": isinstance(aborted, Mapping),
                    "hard_stop_receipt": (root / "HARD_STOP_RECEIPT.json").is_file(),
                    "closure_receipt": isinstance(closure, Mapping),
                    "supervisor_complete": isinstance(complete, Mapping),
                },
                "producer_status": (complete or aborted or {}).get("status") if isinstance(complete or aborted, Mapping) else None,
                "supervisor_status": (start or {}).get("status") if isinstance(start, Mapping) else None,
                "dispatcher_status": "COMPLETE" if (root / "DISPATCHER_COMPLETE.json").is_file() else "UNKNOWN",
                "active_root_referenced": path_reference,
                "active_artifact_sha_overlap_count": len(active_shas & old_shas) if old_shas else 0,
                "disk_usage_bytes": _sum_bytes(root),
            }
        )
    aborted = [item for item in records if item["markers"]["aborted_incomplete"]]
    reused = [item for item in records if item["active_root_referenced"] or item["active_artifact_sha_overlap_count"]]
    return {
        "schema": SCHEMA,
        "generated_utc": utc_now(),
        "goal_root": str(goal_root.resolve()),
        "active_formal_root": str(active_root.resolve()),
        "root_count": len(records),
        "aborted_root_count": len(aborted),
        "historical_recorded_aborted_root_count": 4,
        "aborted_count_reconciliation": {
            "recorded_count": 4,
            "physical_inventory_count": len(aborted),
            "delta": len(aborted) - 4,
            "status": "REQUIRES_EXPLANATION" if len(aborted) != 4 else "MATCH",
        },
        "active_root_reuse_of_aborted_roots": bool(reused),
        "active_root_reuse_evidence": reused,
        "roots": records,
    }


def markdown(report: Mapping[str, Any]) -> str:
    recon = report["aborted_count_reconciliation"]
    lines = [
        "# Stage V root inventory and reconciliation",
        "",
        f"- physical Stage V roots: `{report['root_count']}`",
        f"- physical ABORTED_INCOMPLETE roots: `{report['aborted_root_count']}`",
        f"- prior recorded aborted count: `{recon['recorded_count']}`",
        f"- reconciliation: `{recon['status']}` (delta `{recon['delta']}`)",
        f"- active root reused an aborted root: `{report['active_root_reuse_of_aborted_roots']}`",
        "",
        "| root | classification | source | aborted | closure | active reuse | disk bytes |",
        "|---|---|---|---|---|---|---:|",
    ]
    for item in report["roots"]:
        marker = item["markers"]
        lines.append(
            f"| `{item['name']}` | `{item['classification']}` | `{str(item.get('source_commit') or '')[:8]}` | "
            f"`{marker['aborted_incomplete']}` | `{marker['closure_receipt']}` | "
            f"`{item['active_root_referenced'] or bool(item['active_artifact_sha_overlap_count'])}` | {item['disk_usage_bytes']} |"
        )
    lines.extend(
        [
            "",
            "Parent identity overlap is not artifact reuse; reuse is reported only for active-root path references or SHA overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", required=True, type=Path)
    parser.add_argument("--active-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inventory(args.goal_root.resolve(), args.active_root.resolve())
    atomic_write_json(args.output_root / "STAGE_V_ROOT_INVENTORY.json", report)
    (args.output_root / "STAGE_V_ABORTED_ROOT_RECONCILIATION.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"root_count": report["root_count"], "aborted_root_count": report["aborted_root_count"], "reuse": report["active_root_reuse_of_aborted_roots"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
