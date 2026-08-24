"""Independently audit a completed Stage O observability root."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


BOUNDARY_FIELDS = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")
IDENTITY_FIELDS = ("cell_id", "suite", "split", "canonical_parent_key", "seed", "mode")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _exit_code(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(root: Path, *, expected_source_commit: str | None = None, expected_source_tree: str | None = None) -> dict[str, Any]:
    manifest_path = root / "STAGE_O_MANIFEST.json"
    report_path = root / "STAGE_O_REPORT.json"
    reasons: list[str] = []
    manifest: Mapping[str, Any] = {}
    report: Mapping[str, Any] = {}
    try:
        manifest_value = _read_json(manifest_path)
        report_value = _read_json(report_path)
        if isinstance(manifest_value, Mapping):
            manifest = manifest_value
        else:
            reasons.append("MANIFEST_NOT_OBJECT")
        if isinstance(report_value, Mapping):
            report = report_value
        else:
            reasons.append("REPORT_NOT_OBJECT")
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"ROOT_INPUT_INVALID:{type(exc).__name__}")

    source_commit = expected_source_commit or str(manifest.get("source_commit", ""))
    source_tree = expected_source_tree or str(manifest.get("source_tree", ""))
    if manifest.get("source_commit") != source_commit or manifest.get("source_tree") != source_tree:
        reasons.append("MANIFEST_SOURCE_MISMATCH")
    if report.get("source_commit") != source_commit or report.get("source_tree") != source_tree:
        reasons.append("REPORT_SOURCE_MISMATCH")
    jobs_value = manifest.get("job_specs", [])
    jobs = [dict(item) for item in jobs_value] if isinstance(jobs_value, list) and all(isinstance(item, Mapping) for item in jobs_value) else []
    if not jobs:
        reasons.append("MANIFEST_JOB_SPECS_MISSING")
    expected_by_id = {str(job.get("cell_id")): job for job in jobs}
    expected_ids = list(expected_by_id)
    if len(expected_ids) != len(jobs):
        reasons.append("MANIFEST_DUPLICATE_JOB_IDS")

    result_paths = sorted(root.glob("jobs/*/JOB_RESULT.json"))
    results: list[Mapping[str, Any]] = []
    duplicate_ids: list[str] = []
    missing_job_json: list[str] = []
    invalid_results: list[str] = []
    provenance_mismatches: list[str] = []
    boundary_violations: list[str] = []
    seen_ids: set[str] = set()
    for result_path in result_paths:
        try:
            value = _read_json(result_path)
        except (OSError, json.JSONDecodeError):
            invalid_results.append(str(result_path.relative_to(root)))
            continue
        if not isinstance(value, Mapping):
            invalid_results.append(str(result_path.relative_to(root)))
            continue
        result = dict(value)
        results.append(result)
        cell_id = str(result.get("cell_id", ""))
        if cell_id in seen_ids:
            duplicate_ids.append(cell_id)
        seen_ids.add(cell_id)
        job = expected_by_id.get(cell_id)
        job_path = result_path.parent / "JOB.json"
        if not job_path.is_file():
            missing_job_json.append(cell_id)
        else:
            try:
                job_value = _read_json(job_path)
                if not isinstance(job_value, Mapping) or job is None or any(job_value.get(field) != job.get(field) for field in IDENTITY_FIELDS):
                    provenance_mismatches.append(f"{cell_id}:job")
            except (OSError, json.JSONDecodeError):
                provenance_mismatches.append(f"{cell_id}:job_parse")
        if job is None or any(result.get(field) != job.get(field) for field in IDENTITY_FIELDS):
            provenance_mismatches.append(cell_id)
        if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
            provenance_mismatches.append(f"{cell_id}:source")
        if result.get("status") != "PASS" or _exit_code(result.get("exit_code", 1)) != 0 or result.get("queue_commit") is not True:
            invalid_results.append(cell_id)
        if not _finite(result):
            invalid_results.append(f"{cell_id}:NONFINITE")
        for field in BOUNDARY_FIELDS:
            if result.get(field) != 0:
                boundary_violations.append(f"{cell_id}:{field}")

    missing_ids = sorted(set(expected_ids) - seen_ids)
    if missing_ids:
        reasons.append("MISSING_JOB_RESULTS")
    if duplicate_ids:
        reasons.append("DUPLICATE_JOB_RESULTS")
    if missing_job_json:
        reasons.append("MISSING_JOB_IDENTITY")
    if invalid_results:
        reasons.append("INVALID_JOB_RESULTS")
    if provenance_mismatches:
        reasons.append("PROVENANCE_MISMATCH")
    if boundary_violations:
        reasons.append("BOUNDARY_NONZERO")
    if report.get("status") != "PASS":
        reasons.append("RUN_REPORT_NOT_PASS")
    if report.get("jobs") != len(jobs) or report.get("completed_jobs") != len(results):
        reasons.append("RUN_REPORT_COUNT_MISMATCH")

    verdict = "PASS" if not reasons and not missing_ids and not duplicate_ids and len(results) == len(jobs) else "FAIL"
    return {
        "schema": "STAGE_O_INDEPENDENT_AUDIT_V3",
        "verdict": verdict,
        "root": str(root),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "report_sha256": _sha256(report_path) if report_path.is_file() else None,
        "planned_job_count": len(jobs),
        "completed_job_count": len(results),
        "missing_job_count": len(missing_ids),
        "missing_job_ids": missing_ids,
        "duplicate_job_ids": sorted(set(duplicate_ids)),
        "missing_job_json": sorted(set(missing_job_json)),
        "invalid_results": sorted(set(invalid_results)),
        "provenance_mismatches": sorted(set(provenance_mismatches)),
        "boundary_violations": sorted(set(boundary_violations)),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args()
    result = audit(args.root, expected_source_commit=args.source_commit, expected_source_tree=args.source_tree)
    _write_json(args.root / "STAGE_O_INDEPENDENT_AUDIT.json", result)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
