from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_json, is_valid_sha256, load_json, write_json


REQUIRED_JOB_FIELDS = [
    "job_key",
    "attempt",
    "pid",
    "start_time",
    "actual_loaded_worker_sha",
    "bridge_sha",
    "manifest_sha",
    "provenance_source",
]


def _rows(data: object) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("jobs", "rows", "worker_runtime_provenance"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("runtime provenance must be a list or contain jobs/rows")


def _equivalent(report: dict | None) -> bool:
    if not report:
        return False
    return any(bool(report.get(k)) for k in ("valid_row_equivalence_pass", "valid_rows_equivalent", "execution_equivalent"))


def evaluate(provenance: dict | list, *, spec_worker_sha: str, disk_worker_sha: str, expected_jobs: int, equivalence: dict | None = None) -> dict:
    problems: list[dict] = []
    if not is_valid_sha256(spec_worker_sha):
        problems.append({"class": "invalid_spec_worker_sha"})
    if not is_valid_sha256(disk_worker_sha):
        problems.append({"class": "invalid_disk_worker_sha"})

    rows = _rows(provenance)
    if len(rows) != expected_jobs:
        problems.append({"class": "job_count_mismatch", "expected": expected_jobs, "actual": len(rows)})

    seen: set[tuple[str, int]] = set()
    shas: list[str] = []
    for i, row in enumerate(rows):
        missing = [f for f in REQUIRED_JOB_FIELDS if row.get(f) in (None, "")]
        if missing:
            problems.append({"class": "missing_runtime_provenance_field", "row": i, "fields": missing})
        key = (str(row.get("job_key", "")), int(row.get("attempt", -1)) if str(row.get("attempt", "")).lstrip("-").isdigit() else -1)
        if key in seen:
            problems.append({"class": "duplicate_job_attempt", "job_key": key[0], "attempt": key[1]})
        seen.add(key)
        sha = str(row.get("actual_loaded_worker_sha", ""))
        if not is_valid_sha256(sha):
            problems.append({"class": "invalid_actual_loaded_worker_sha", "row": i, "sha": sha})
        else:
            shas.append(sha)

    counts = dict(sorted(Counter(shas).items()))
    unique = set(counts)
    if problems:
        binding = "RUNTIME_BINDING_P0_HOLD"
        decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
    elif unique == {spec_worker_sha}:
        binding = "CASE_A_SPEC_BOUND_WORKER"
        decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
    elif unique == {disk_worker_sha}:
        if _equivalent(equivalence):
            binding = "CASE_B_POSTLAUNCH_RUNTIME_DEVIATION_REVIEW"
            decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
        else:
            binding = "RUNTIME_BINDING_P0_HOLD"
            decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
            problems.append({"class": "disk_worker_without_valid_row_equivalence"})
    elif len(unique) > 1:
        binding = "VIS_RUNTIME_QUARANTINE_HOLD"
        decision = "VIS_RUNTIME_QUARANTINE_HOLD"
        problems.append({"class": "mixed_worker_versions", "sha_counts": counts})
    else:
        binding = "RUNTIME_BINDING_P0_HOLD"
        decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
        problems.append({"class": "actual_worker_sha_matches_neither_spec_nor_reported_disk", "sha_counts": counts})

    return {
        "schema_version": "worker_runtime_binding_validation.v1",
        "decision_state": decision,
        "worker_binding_status": binding,
        "result_acceptance": "RESULT_ACCEPTANCE_HOLD",
        "new_condition_launch": "NEW_CONDITION_LAUNCH_HOLD",
        "spec_worker_sha": spec_worker_sha,
        "reported_disk_worker_sha": disk_worker_sha,
        "expected_jobs": expected_jobs,
        "observed_jobs": len(rows),
        "actual_loaded_worker_sha_counts": counts,
        "equivalence_report_present": equivalence is not None,
        "valid_row_equivalence_pass": _equivalent(equivalence),
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify TRUE_T10 worker runtime binding from offline provenance JSON.")
    add_path_arg(ap, "--runtime-provenance", required=True)
    add_path_arg(ap, "--output-json", required=True)
    add_path_arg(ap, "--equivalence-report")
    ap.add_argument("--spec-worker-sha", required=True)
    ap.add_argument("--disk-worker-sha", required=True)
    ap.add_argument("--expected-jobs", type=int, default=162)
    args = ap.parse_args()
    result = evaluate(
        load_json(args.runtime_provenance),
        spec_worker_sha=args.spec_worker_sha,
        disk_worker_sha=args.disk_worker_sha,
        expected_jobs=args.expected_jobs,
        equivalence=load_json(args.equivalence_report) if args.equivalence_report else None,
    )
    write_json(args.output_json, result)
    print(canonical_json({"worker_binding_status": result["worker_binding_status"], "problems": len(result["problems"])}), end="")
    return 0 if result["worker_binding_status"] in {"CASE_A_SPEC_BOUND_WORKER", "CASE_B_POSTLAUNCH_RUNTIME_DEVIATION_REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
