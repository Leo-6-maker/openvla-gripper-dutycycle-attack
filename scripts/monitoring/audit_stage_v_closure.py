"""Independent, fail-closed closure audit for a Stage V formal map.

The audit is intentionally read-only with respect to the Stage V artifacts.  A
caller may use :func:`write_root_seal` after the producer is quiescent, then
run this audit again with ``--require-root-seal``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "STAGE_V_CLOSURE_AUDIT_V1"
REQUIRED_BRANCH_ARMS = {
    "CLEAN",
    "OPEN_T3",
    "OPEN_T5",
    "OPEN_T10",
    "NOOP_T10_REPLAY",
}
OPEN_ARMS = {"OPEN_T3", "OPEN_T5", "OPEN_T10"}
BOUNDARY_KEYS = (
    "eval160_reads",
    "protected_eval_reads",
    "attack_rollouts",
    "vis_rollouts",
    "pgd_rollouts",
    "vis_pgd_attack_rollouts",
)
PARENT_DIR_RE = re.compile(r"^libero_[^/]+/task_[0-9]+/state_[0-9]+$")


class ClosureAuditError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _counter(value: Mapping[str, Any], key: str) -> int:
    number = _int(value.get(key))
    return number if number is not None else 0


def _safe_relative(raw: str) -> Path:
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise ClosureAuditError(f"unsafe seal path: {raw}")
    return rel


def verify_sha_manifest(directory: Path) -> tuple[bool, list[str], int]:
    errors: list[str] = []
    sums = directory / "SHA256SUMS"
    sidecar = directory / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        return False, ["parent_seal_missing"], 0
    try:
        sidecar_tokens = sidecar.read_text(encoding="utf-8").split()
        if not sidecar_tokens or sha256_file(sums) != sidecar_tokens[0]:
            errors.append("parent_sidecar_hash_mismatch")
        lines = [line.strip() for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return False, ["parent_seal_unreadable"], 0
    seen: set[str] = set()
    checked = 0
    for line in lines:
        fields = line.split(None, 1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            errors.append("parent_seal_line_invalid")
            continue
        try:
            rel = _safe_relative(fields[1].strip())
        except ClosureAuditError as exc:
            errors.append(str(exc))
            continue
        key = rel.as_posix()
        if key in seen:
            errors.append(f"parent_seal_duplicate:{key}")
            continue
        seen.add(key)
        target = directory / rel
        if not target.is_file() or sha256_file(target).lower() != fields[0].lower():
            errors.append(f"parent_artifact_hash_mismatch:{key}")
        else:
            checked += 1
    return not errors, sorted(set(errors)), checked


def _branch_status(row: Mapping[str, Any], side: str) -> str | None:
    nested = row.get(side)
    if isinstance(nested, Mapping):
        status = nested.get("status")
        if status is not None:
            return str(status).upper()
    comparison = row.get("comparison")
    if isinstance(comparison, Mapping):
        value = comparison.get(f"{side}_status")
        if value is not None:
            return str(value).upper()
    return None


def verify_parent(
    parent_dir: Path,
    expected: Mapping[str, Any],
    *,
    expected_source_commit: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    errors: list[str] = []
    result = load_json(parent_dir / "PARENT_RESULT.json")
    if not isinstance(result, Mapping):
        return {
            "canonical_parent_key": str(expected.get("canonical_parent_key")),
            "audit_status": "FAIL",
            "seal_status": "FAIL",
            "accepted": False,
            "errors": ["parent_result_missing_or_invalid"],
            "expected_branch_count": None,
            "completed_branch_count": 0,
            "failed_branch_count": 0,
            "missing_branches": ["PARENT_RESULT.json"],
        }
    identity = str(result.get("canonical_parent_key", ""))
    if identity != str(expected.get("canonical_parent_key", "")):
        errors.append("parent_identity_mismatch")
    if result.get("status") != "PASS":
        errors.append("parent_status_not_pass")
    if result.get("clean_success") is not True:
        errors.append("clean_success_not_true")
    if result.get("exact_snapshot_replay") is not True:
        errors.append("exact_snapshot_replay_not_true")
    if result.get("current_source_commit") != expected_source_commit:
        errors.append("parent_source_commit_mismatch")
    if result.get("current_source_tree") != expected_source_tree:
        errors.append("parent_source_tree_mismatch")
    if result.get("current_source_status") != "":
        errors.append("parent_source_dirty")
    for key in BOUNDARY_KEYS:
        if _counter(result, key) != 0:
            errors.append(f"{key}_not_zero")
    required_arms = {str(item) for item in result.get("branch_arms", [])}
    if not REQUIRED_BRANCH_ARMS.issubset(required_arms):
        errors.append("required_branch_arm_missing")
    probe_count = _int(result.get("probe_count"))
    expected_branch_count = probe_count * 3 if probe_count is not None and probe_count > 0 else None
    rows: list[Mapping[str, Any]] = []
    branch_file = parent_dir / "COUNTERFACTUAL_BRANCHES.jsonl"
    if not branch_file.is_file():
        errors.append("counterfactual_branch_file_missing")
    else:
        try:
            for line in branch_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError("branch row is not an object")
                    rows.append(row)
                    if not _finite(row):
                        errors.append("branch_metric_non_finite")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"counterfactual_branch_file_invalid:{exc}")
    if expected_branch_count is not None and len(rows) != expected_branch_count:
        errors.append(f"branch_count:{len(rows)}/{expected_branch_count}")
    arms = [str(row.get("arm", "")) for row in rows]
    if set(arms) != OPEN_ARMS:
        errors.append("open_branch_arm_set_mismatch")
    if expected_branch_count is not None:
        identities = {(str(row.get("arm")), _int(row.get("probe_step"))) for row in rows}
        if len(identities) != len(rows):
            errors.append("duplicate_branch_identity")
    failed_branches = 0
    for row in rows:
        comparison = row.get("comparison")
        if not isinstance(comparison, Mapping):
            errors.append("branch_comparison_missing")
        else:
            if comparison.get("label_status") != "VALID":
                errors.append("branch_label_not_valid")
            if comparison.get("control_task_success") is not True:
                errors.append("branch_control_task_failure")
        if _branch_status(row, "control") != "PASS" or _branch_status(row, "opened") != "PASS":
            failed_branches += 1
            errors.append("branch_exit_status_not_pass")
    seal_ok, seal_errors, checked = verify_sha_manifest(parent_dir)
    errors.extend(seal_errors)
    return {
        "canonical_parent_key": identity or str(expected.get("canonical_parent_key")),
        "suite": result.get("suite"),
        "task_index": result.get("task_index"),
        "state_index": result.get("state_index"),
        "expected_branch_count": expected_branch_count,
        "completed_branch_count": len(rows),
        "failed_branch_count": failed_branches,
        "missing_branches": [] if rows else ["COUNTERFACTUAL_BRANCHES.jsonl"],
        "audit_status": "PASS" if not errors else "FAIL",
        "seal_status": "PASS" if seal_ok else "FAIL",
        "accepted": not errors,
        "local_positive_count": sum(
            isinstance(row.get("comparison"), Mapping)
            and row["comparison"].get("local_vulnerability") is True
            for row in rows
        ),
        "task_positive_count": sum(
            isinstance(row.get("comparison"), Mapping)
            and row["comparison"].get("task_vulnerability") is True
            for row in rows
        ),
        "artifact_files_verified": checked,
        "errors": sorted(set(errors)),
    }


def _expected_parents(manifest: Any) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    if not isinstance(manifest, Mapping):
        return {}, ["parent_manifest_missing_or_invalid"]
    rows = manifest.get("selected_parents")
    if not isinstance(rows, list):
        return {}, ["selected_parents_missing_or_invalid"]
    expected: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("canonical_parent_key"):
            errors.append("selected_parent_identity_invalid")
            continue
        key = str(row["canonical_parent_key"])
        if key in expected:
            errors.append(f"duplicate_frozen_parent:{key}")
        expected[key] = row
    return expected, errors


def parent_progress(
    root: Path,
    *,
    parent_manifest: Path,
    expected_source_commit: str,
    expected_source_tree: str,
    full_audit: bool = True,
) -> dict[str, Any]:
    expected, manifest_errors = _expected_parents(load_json(parent_manifest))
    found: dict[str, list[Path]] = {}
    invalid_results = 0
    for path in sorted(root.rglob("PARENT_RESULT.json")):
        value = load_json(path)
        key = str(value.get("canonical_parent_key", "")) if isinstance(value, Mapping) else ""
        if not key:
            invalid_results += 1
            continue
        found.setdefault(key, []).append(path.parent)
    duplicate_count = sum(max(0, len(paths) - 1) for paths in found.values())
    orphan_count = invalid_results + sum(len(paths) for key, paths in found.items() if key not in expected)
    rows: list[dict[str, Any]] = []
    accepted = 0
    audited = 0
    branch_complete = 0
    missing_branch_count = 0
    invalid_parent_count = 0
    local_positive = 0
    task_positive = 0
    for key in sorted(expected):
        paths = found.get(key, [])
        if len(paths) != 1:
            item = {
                "canonical_parent_key": key,
                "expected_branch_count": None,
                "completed_branch_count": 0,
                "failed_branch_count": 0,
                "missing_branches": ["PARENT_RESULT.json"] if not paths else [],
                "audit_status": "FAIL" if len(paths) != 1 else "PENDING",
                "seal_status": "FAIL" if len(paths) != 1 else "PENDING",
                "accepted": False,
                "errors": ["missing_parent_result"] if not paths else ["duplicate_parent_identity"],
            }
        elif full_audit:
            item = verify_parent(
                paths[0],
                expected[key],
                expected_source_commit=expected_source_commit,
                expected_source_tree=expected_source_tree,
            )
        else:
            result = load_json(paths[0] / "PARENT_RESULT.json")
            probe_count = _int(result.get("probe_count")) if isinstance(result, Mapping) else None
            expected_branch_count = probe_count * 3 if probe_count and probe_count > 0 else None
            branch_file = paths[0] / "COUNTERFACTUAL_BRANCHES.jsonl"
            try:
                completed_branch_count = sum(1 for line in branch_file.open(encoding="utf-8") if line.strip())
            except OSError:
                completed_branch_count = 0
            missing = []
            if expected_branch_count is None:
                missing.append("PARENT_RESULT.probe_count")
            elif completed_branch_count != expected_branch_count:
                missing.append(f"COUNTERFACTUAL_BRANCHES.jsonl:{completed_branch_count}/{expected_branch_count}")
            basic_errors = []
            if not isinstance(result, Mapping):
                basic_errors.append("parent_result_missing_or_invalid")
            elif result.get("current_source_commit") != expected_source_commit or result.get("current_source_tree") != expected_source_tree:
                basic_errors.append("parent_source_binding_mismatch")
            elif result.get("status") != "PASS" or result.get("clean_success") is not True:
                basic_errors.append("parent_producer_status_not_pass")
            item = {
                "canonical_parent_key": key,
                "expected_branch_count": expected_branch_count,
                "completed_branch_count": completed_branch_count,
                "failed_branch_count": 0,
                "missing_branches": missing,
                "audit_status": "PENDING" if not basic_errors else "FAIL",
                "seal_status": "PENDING",
                "accepted": False,
                "errors": basic_errors,
            }
        if item["audit_status"] == "PASS":
            audited += 1
            branch_complete += 1
            local_positive += int(item.get("local_positive_count", 0))
            task_positive += int(item.get("task_positive_count", 0))
        elif item["audit_status"] == "PENDING":
            if not item.get("missing_branches"):
                branch_complete += 1
            missing_branch_count += len(item.get("missing_branches", []))
        else:
            if item.get("missing_branches"):
                missing_branch_count += len(item.get("missing_branches", []))
            invalid_parent_count += 1
        if item.get("accepted"):
            accepted += 1
        rows.append(item)
    progress = {
        "schema": "STAGE_V_PARENT_PROGRESS_V1",
        "planned_parent_count": len(expected),
        "started_parent_count": len(found),
        "branch_complete_parent_count": branch_complete,
        "audited_parent_count": audited,
        "accepted_parent_count": accepted,
        "invalid_parent_count": invalid_parent_count + len(manifest_errors),
        "missing_branch_count": missing_branch_count,
        "duplicate_identity_count": duplicate_count,
        "orphan_artifact_count": orphan_count,
        "local_positive_count": local_positive,
        "task_positive_count": task_positive,
        "parents": rows,
    }
    return progress


def _boundary_errors(*values: Any) -> list[str]:
    errors: list[str] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in BOUNDARY_KEYS:
            if _counter(value, key) != 0:
                errors.append(f"{key}_not_zero")
    return errors


def verify_root_seal(root: Path) -> tuple[bool, list[str], int]:
    return verify_sha_manifest(root)


def write_root_seal(root: Path) -> dict[str, Any]:
    """Create a deterministic root seal after the producer is quiescent."""

    excluded = {
        "SHA256SUMS",
        "SHA256SUMS.sha256",
        "STAGE_V_CLOSURE_RECEIPT.json",
        "STAGE_V_CLOSURE_RECEIPT.sha256",
    }
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "MONITOR" in path.relative_to(root).parts:
            continue
        if path.name in excluded:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append((sha256_file(path), rel))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{digest}  {rel}\n" for digest, rel in entries), encoding="utf-8")
    with sums.open("r+b") as handle:
        os.fsync(handle.fileno())
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    with sidecar.open("r+b") as handle:
        os.fsync(handle.fileno())
    return {"files": len(entries), "sha256sums_sha256": sha256_file(sums)}


def audit_closure(
    root: Path,
    *,
    parent_manifest: Path,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_parent_count: int,
    require_root_seal: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    start = load_json(root / "SUPERVISOR_START.json")
    heartbeat = load_json(root / "LOCAL_HEARTBEAT.json")
    complete = load_json(root / "SUPERVISOR_COMPLETE.json")
    producer_audit = load_json(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json")
    manifest = load_json(root / "RUN_MANIFEST.json")
    if not isinstance(start, Mapping):
        errors.append("supervisor_start_missing_or_invalid")
    if not isinstance(heartbeat, Mapping):
        errors.append("heartbeat_missing_or_invalid")
    if not isinstance(complete, Mapping):
        errors.append("supervisor_complete_missing_or_invalid")
    if not isinstance(producer_audit, Mapping) or producer_audit.get("verdict") != "PASS":
        errors.append("independent_producer_audit_missing_or_not_pass")
    if isinstance(start, Mapping):
        if start.get("source_commit") != expected_source_commit:
            errors.append("start_source_commit_mismatch")
        if start.get("source_tree") != expected_source_tree:
            errors.append("start_source_tree_mismatch")
        if _int(start.get("planned_parents")) != expected_parent_count:
            errors.append("start_planned_parent_mismatch")
    for value_name, value in (("heartbeat", heartbeat), ("complete", complete), ("manifest", manifest), ("producer_audit", producer_audit)):
        if isinstance(value, Mapping):
            if value.get("source_commit") not in (None, expected_source_commit):
                errors.append(f"{value_name}_source_commit_mismatch")
            if value.get("source_tree") not in (None, expected_source_tree):
                errors.append(f"{value_name}_source_tree_mismatch")
            errors.extend(_boundary_errors(value))
    if isinstance(complete, Mapping):
        if complete.get("status") != "PASS":
            errors.append("supervisor_complete_not_pass")
        for key in ("planned_parents", "completed_parents", "accepted_parent_results"):
            if _int(complete.get(key)) != expected_parent_count:
                errors.append(f"complete_{key}_mismatch")
        artifacts = complete.get("accepted_parent_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != expected_parent_count:
            errors.append("accepted_parent_artifact_audit_missing")
        elif any(not isinstance(item, Mapping) or item.get("artifact_audit_verdict") != "PASS" for item in artifacts):
            errors.append("accepted_parent_artifact_audit_not_pass")
    progress = parent_progress(
        root,
        parent_manifest=parent_manifest,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
    )
    if progress["planned_parent_count"] != expected_parent_count:
        errors.append("frozen_parent_count_mismatch")
    if progress["started_parent_count"] != expected_parent_count:
        errors.append("started_parent_count_mismatch")
    if progress["branch_complete_parent_count"] != expected_parent_count:
        errors.append("branch_complete_parent_count_mismatch")
    if progress["audited_parent_count"] != expected_parent_count:
        errors.append("audited_parent_count_mismatch")
    if progress["accepted_parent_count"] != expected_parent_count:
        errors.append("accepted_parent_count_mismatch")
    for key in ("invalid_parent_count", "missing_branch_count", "duplicate_identity_count", "orphan_artifact_count"):
        if progress[key] != 0:
            errors.append(f"{key}_nonzero")
    producer_parent_count = _int(producer_audit.get("parent_count")) if isinstance(producer_audit, Mapping) else None
    if producer_parent_count != expected_parent_count:
        errors.append("producer_audit_parent_count_mismatch")
    root_seal_ok, root_seal_errors, root_seal_files = verify_root_seal(root)
    if require_root_seal and not root_seal_ok:
        errors.extend(root_seal_errors)
    elif not root_seal_ok:
        root_seal_errors = [f"deferred:{item}" for item in root_seal_errors]
    report = {
        "schema": SCHEMA,
        "verdict": "PASS" if not errors else "FAIL",
        "run_root": str(root),
        "expected_parent_count": expected_parent_count,
        "planned_parent_count": progress["planned_parent_count"],
        "started_parent_count": progress["started_parent_count"],
        "completed_parent_count": progress["branch_complete_parent_count"],
        "audited_parent_count": progress["audited_parent_count"],
        "accepted_parent_count": progress["accepted_parent_count"],
        "branch_complete_parent_count": progress["branch_complete_parent_count"],
        "invalid_parent_count": progress["invalid_parent_count"],
        "missing_branch_count": progress["missing_branch_count"],
        "duplicate_identity_count": progress["duplicate_identity_count"],
        "orphan_artifact_count": progress["orphan_artifact_count"],
        "local_positive_count": progress["local_positive_count"],
        "task_positive_count": progress["task_positive_count"],
        "root_seal_status": "PASS" if root_seal_ok else "MISSING_OR_FAIL",
        "root_seal_files": root_seal_files,
        "root_seal_errors": root_seal_errors,
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_rollouts": 0,
        "pgd_rollouts": 0,
        "attack_rollouts": 0,
        "parents": progress["parents"],
        "errors": sorted(set(errors)),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-parent-count", required=True, type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-missing-root-seal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_closure(
        args.root.resolve(),
        parent_manifest=args.parent_manifest.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
        expected_parent_count=args.expected_parent_count,
        require_root_seal=not args.allow_missing_root_seal,
    )
    report_path = args.report or args.root / "MONITOR" / "STAGE_V_CLOSURE_AUDIT.json"
    atomic_write_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("verdict", "accepted_parent_count", "branch_complete_parent_count", "errors")}, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
