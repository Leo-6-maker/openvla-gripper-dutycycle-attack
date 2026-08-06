"""Read-only postmortem and timeout-policy freeze for the aborted Stage V root."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, read_json, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, read_json, utc_now


def _timestamp(value: Any) -> float | None:
    if not value:
        return None
    try:
        return _datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _runtime(row: Mapping[str, Any]) -> float | None:
    start = next((_timestamp(row.get(key)) for key in ("start_utc", "started_utc", "claim_utc")), None)
    end = next((_timestamp(row.get(key)) for key in ("end_utc", "completed_utc", "finished_utc")), None)
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def _load_parent_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in root.rglob("PARENT_RESULT.json"):
        value = read_json(path, {})
        if isinstance(value, Mapping):
            row = dict(value)
            row["_path"] = str(path)
            key = str(row.get("canonical_parent_key", ""))
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
    for path in root.glob("WORKER_*_SUMMARY.json"):
        value = read_json(path, {})
        parents = value.get("parents") if isinstance(value, Mapping) else None
        if not isinstance(parents, list):
            continue
        for index, item in enumerate(parents):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row["_path"] = f"{path}#parents[{index}]"
            key = str(row.get("canonical_parent_key", ""))
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def _load_branch_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("COUNTERFACTUAL_BRANCHES.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
        except (OSError, ValueError):
            continue
    return rows


def _text_value(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _manifest_info(root: Path) -> tuple[set[str], dict[str, Any]]:
    keys: set[str] = set()
    run_manifest = read_json(root / "RUN_MANIFEST.json", {})
    referenced = run_manifest.get("parent_manifest") if isinstance(run_manifest, Mapping) else None
    candidates_paths: list[Path] = []
    if referenced:
        path = Path(str(referenced))
        candidates_paths.append(path if path.is_absolute() else root / path)
    candidates_paths.extend(path for path in root.rglob("*.json") if "manifest" in path.name.lower())
    unique_paths: list[Path] = []
    seen_paths: set[str] = set()
    for path in candidates_paths:
        resolved = str(path.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_paths.append(path)
    for path in unique_paths:
        value = read_json(path, {})
        candidates = []
        if isinstance(value, list):
            candidates = value
        elif isinstance(value, Mapping):
            candidates = next((value.get(k) for k in ("selected_parents", "parents", "planned_parents", "rows", "all_candidate_audits") if isinstance(value.get(k), list)), [])
        for item in candidates:
            if isinstance(item, Mapping) and item.get("canonical_parent_key"):
                keys.add(str(item["canonical_parent_key"]))
    manifest_sha = None
    manifest_path = candidates_paths[0] if candidates_paths else None
    if manifest_path and manifest_path.is_file():
        import hashlib
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_sha = run_manifest.get("parent_manifest_sha256") if isinstance(run_manifest, Mapping) else None
    return keys, {
        "path": str(manifest_path) if manifest_path else None,
        "expected_sha256": expected_sha,
        "actual_sha256": manifest_sha,
        "sha256_verified": bool(expected_sha and manifest_sha and expected_sha == manifest_sha),
        "run_manifest_present": isinstance(run_manifest, Mapping),
    }


def _manifest_keys(root: Path) -> set[str]:
    return _manifest_info(root)[0]


def _latest_artifact(root: Path) -> tuple[str | None, str | None]:
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"LOCAL_HEARTBEAT.json", "STALE_LOCK_AUDIT.json"}
    ]
    if not candidates:
        return None, None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    return _datetime.datetime.fromtimestamp(path.stat().st_mtime, _datetime.timezone.utc).isoformat(), str(path)


def build_postmortem(old_root: Path, *, launcher: Path | None = None,
                     supervisor_source: Path | None = None, expected_parent_count: int = 40) -> dict[str, Any]:
    heartbeat = read_json(old_root / "LOCAL_HEARTBEAT.json", {})
    abort = read_json(old_root / "ABORTED_INCOMPLETE.json", {})
    results = _load_parent_results(old_root)
    branches = _load_branch_rows(old_root)
    keys = {str(row["canonical_parent_key"]) for row in results if row.get("canonical_parent_key")}
    expected, manifest_info = _manifest_info(old_root)
    missing = sorted(expected - keys) if expected else []
    runtimes = [(row, _runtime(row)) for row in results]
    values = [value for _, value in runtimes if value is not None]
    branch_values = [value for row in branches if (value := _runtime(row)) is not None]
    clean = [value for row, value in runtimes if row.get("clean_success") is True and value is not None]
    failed = [value for row, value in runtimes if row.get("clean_success") is False and value is not None]
    latest_time, latest_path = _latest_artifact(old_root)
    receipt_fields = sorted(abort.keys()) if isinstance(abort, Mapping) else []
    supervisor_text = supervisor_source.read_text(encoding="utf-8", errors="replace") if supervisor_source and supervisor_source.is_file() else ""
    root_wide_watchdog = "PARENT_WATCHDOG_TIMEOUT" in supervisor_text and "last_artifact_mtime" in supervisor_text
    launcher_text = launcher.read_text(encoding="utf-8", errors="replace") if launcher and launcher.is_file() else ""
    run_manifest = read_json(old_root / "RUN_MANIFEST.json", {})
    layout6 = (
        "MAP_LAYOUT=\"6\"" in launcher_text
        or "GPUS=(1 2 3 4 6 7)" in launcher_text
        or (isinstance(run_manifest, Mapping) and str(run_manifest.get("map_layout")) == "6")
        or (isinstance(run_manifest, Mapping) and len(run_manifest.get("gpus", [])) == 6)
    )
    return {
        "schema": "STAGE_V_ABORT_POSTMORTEM_V1",
        "old_root": str(old_root),
        "old_root_read_only": True,
        "old_root_status": abort.get("status", "UNKNOWN") if isinstance(abort, Mapping) else "UNKNOWN",
        "abort_reason": abort.get("control_plane_abort_reason", abort.get("reason")) if isinstance(abort, Mapping) else None,
        "aborted_utc": abort.get("aborted_utc", abort.get("end_utc")) if isinstance(abort, Mapping) else None,
        "watchdog_last_valid_artifact_utc": heartbeat.get("last_artifact_utc") or latest_time,
        "watchdog_last_valid_artifact_path": latest_path,
        "worker_snapshot": {
            "active_worker_pids": heartbeat.get("active_worker_pids"),
            "gpu_assignments": heartbeat.get("gpu_assignments"),
            "current_parent": heartbeat.get("current_parent"),
            "current_branch": heartbeat.get("current_branch"),
            "gpu_memory": heartbeat.get("gpu_memory"),
            "gpu_xid_status": heartbeat.get("gpu_xid_status"),
            "updated_utc": heartbeat.get("updated_utc"),
        },
        "gpu1_last_worker": {
            "pid": _text_value(old_root / "worker_gpu1.pid"),
            "exit_code": _text_value(old_root / "worker_gpu1.exitcode"),
            "summary_present": (old_root / "WORKER_GPU1_SUMMARY.json").is_file(),
            "worker_manifest_present": (old_root / "WORKER_GPU1.json").is_file(),
        },
        "failure_mode_assessment": {
            "root_wide_artifact_mtime_watchdog_detected": root_wide_watchdog,
            "long_compute_or_deadlock_or_child_loss": "INSUFFICIENT_IDENTITY_LOGS",
            "static_layout6_gpu_idle_tail": layout6,
            "likely_abort_mechanism": "root-wide artifact mtime timeout without parent identity" if root_wide_watchdog else "INSUFFICIENT_LOGS",
        },
        "parent_counts": {
            "expected": expected_parent_count,
            "manifest_discovered": len(expected),
            "parent_results": len(results),
            "clean_success_true": sum(row.get("clean_success") is True for row in results),
            "clean_success_false": sum(row.get("clean_success") is False for row in results),
            "missing_parent_keys": missing,
            "manifest": manifest_info,
        },
        "runtime_seconds": {
            "all": {"count": len(values), "p50": _percentile(values, .50), "p95": _percentile(values, .95), "max": max(values) if values else None},
            "parent": {"count": len(values), "p50": _percentile(values, .50), "p95": _percentile(values, .95), "max": max(values) if values else None},
            "branch": {"count": len(branch_values), "p50": _percentile(branch_values, .50), "p95": _percentile(branch_values, .95), "max": max(branch_values) if branch_values else None},
            "clean_success_true": {"count": len(clean), "p50": _percentile(clean, .50), "p95": _percentile(clean, .95), "max": max(clean) if clean else None},
            "clean_success_false": {"count": len(failed), "p50": _percentile(failed, .50), "p95": _percentile(failed, .95), "max": max(failed) if failed else None},
            "source": "parent timestamps when present; otherwise insufficient",
        },
        "missing_parent_status": {key: "UNRESOLVED_NO_PARENT_BOUND_RECEIPT" for key in missing},
        "abort_receipt_identity_gap": {
            "receipt_fields": receipt_fields,
            "canonical_parent_key_present": "canonical_parent_key" in abort if isinstance(abort, Mapping) else False,
            "branch_present": "branch" in abort if isinstance(abort, Mapping) else False,
            "conclusion": "receipt lacks canonical parent and branch identity" if isinstance(abort, Mapping) and "canonical_parent_key" not in abort else "NOT_DETECTED",
        },
        "provenance": {
            "source_commit": heartbeat.get("source_commit"),
            "source_tree": heartbeat.get("source_tree"),
            "heartbeat_count": heartbeat.get("heartbeat_count"),
            "oom_kill": heartbeat.get("oom_kill"),
        },
        "generated_utc": utc_now(),
    }


def build_timeout_policy(postmortem: Mapping[str, Any]) -> dict[str, Any]:
    runtimes = postmortem.get("runtime_seconds", {})
    branch_p95 = runtimes.get("branch", {}).get("p95")
    parent_p95 = runtimes.get("parent", runtimes.get("all", {})).get("p95")
    branch_observed = bool(branch_p95)
    parent_observed = bool(parent_p95)
    branch_soft = max(2 * float(branch_p95), 90 * 60) if branch_observed else 90 * 60
    branch_hard = max(4 * float(branch_p95), 4 * 3600) if branch_observed else 4 * 3600
    parent_soft = max(2 * float(parent_p95), 3 * 3600) if parent_observed else 3 * 3600
    parent_hard = max(4 * float(parent_p95), 8 * 3600) if parent_observed else 10 * 3600
    if branch_observed and parent_observed:
        source = "old_root_branch_and_parent_runtime_p95"
    elif branch_observed or parent_observed:
        source = "mixed_old_root_runtime_p95_and_fallback"
    else:
        source = "fallback_due_to_insufficient_timestamps"
    return {
        "schema": "STAGE_V_RERUN_TIMEOUT_POLICY_V1",
        "source": source,
        "branch_soft_seconds": branch_soft,
        "branch_hard_seconds": min(branch_hard, 16 * 3600),
        "parent_soft_seconds": parent_soft,
        "parent_hard_seconds": min(parent_hard, 16 * 3600),
        "hard_cap_seconds": 16 * 3600,
        "root_wide_artifact_mtime_watchdog": False,
        "requires_parent_identity": True,
        "frozen_utc": utc_now(),
    }


def write_outputs(postmortem: Mapping[str, Any], policy: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "STAGE_V_ABORT_POSTMORTEM.json", postmortem)
    atomic_write_json(output_dir / "STAGE_V_RERUN_TIMEOUT_POLICY.json", policy)
    lines = [
        "# Stage V abort postmortem",
        "",
        f"- old root: `{postmortem['old_root']}`",
        f"- status: `{postmortem['old_root_status']}`",
        f"- reason: `{postmortem.get('abort_reason')}`",
        f"- parent results: `{postmortem['parent_counts']['parent_results']}`",
        f"- clean success true/false: `{postmortem['parent_counts']['clean_success_true']}/{postmortem['parent_counts']['clean_success_false']}`",
        f"- missing parents: `{len(postmortem['parent_counts']['missing_parent_keys'])}`",
        f"- root-wide watchdog detected: `{postmortem['failure_mode_assessment']['root_wide_artifact_mtime_watchdog_detected']}`",
        f"- receipt identity gap: `{postmortem['abort_receipt_identity_gap']['conclusion']}`",
        "",
        "The old root was read only; no artifact was changed or resealed.",
    ]
    (output_dir / "STAGE_V_ABORT_POSTMORTEM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--supervisor-source", type=Path)
    parser.add_argument("--expected-parent-count", type=int, default=40)
    args = parser.parse_args(argv)
    old_root = args.old_root.resolve()
    output_dir = args.output_dir.resolve()
    if not old_root.is_dir():
        parser.error(f"old root does not exist: {old_root}")
    if output_dir == old_root or str(output_dir).startswith(str(old_root) + "/") or str(output_dir).startswith(str(old_root) + "\\"):
        parser.error("postmortem output must not be inside the old read-only root")
    postmortem = build_postmortem(
        old_root,
        launcher=args.launcher,
        supervisor_source=args.supervisor_source,
        expected_parent_count=args.expected_parent_count,
    )
    policy = build_timeout_policy(postmortem)
    write_outputs(postmortem, policy, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
