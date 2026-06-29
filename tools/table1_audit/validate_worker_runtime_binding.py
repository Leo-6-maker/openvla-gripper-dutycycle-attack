from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_json, is_valid_sha256, load_json, load_jsonl, sha256_file, write_json


REQUIRED_JOB_FIELDS = [
    "job_key",
    "attempt",
    "accepted",
    "pid",
    "start_time",
    "actual_loaded_worker_sha",
    "bridge_sha",
    "manifest_sha",
    "telemetry_schema_sha",
    "provenance_source",
    "provenance_evidence_sha256",
]

ALLOWED_PROVENANCE_SOURCES = {
    "immutable_deployment_copy",
    "container_image_digest",
    "launch_bundle_checksum",
    "episode_recorded_worker_sha",
    "verified_pyc_source_pairing",
    "process_specific_deployment_tree",
}

EQUIVALENCE_REQUIRED = [
    "old_worker_sha256",
    "new_worker_sha256",
    "manifest_sha256",
    "bridge_sha256",
    "test_harness_sha256",
    "test_vector_inventory_sha256",
    "tested_valid_row_count",
    "expected_valid_row_count",
    "condition_resolution_diff_count",
    "attack_activation_diff_count",
    "env_action_diff_count",
    "arm_lock_diff_count",
    "termination_diff_count",
    "retry_behavior_diff_count",
    "overall_pass",
]

DIFF_COUNTS = [
    "condition_resolution_diff_count",
    "attack_activation_diff_count",
    "env_action_diff_count",
    "arm_lock_diff_count",
    "termination_diff_count",
    "retry_behavior_diff_count",
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


def _parse_time(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _manifest_job_keys(rows: list[dict] | None) -> set[str] | None:
    if rows is None:
        return None
    return {str(r.get("job_key", "")) for r in rows if r.get("job_key")}


def _equivalent(report: dict | None, *, spec_worker_sha: str, disk_worker_sha: str, expected_manifest_sha: str, expected_bridge_sha: str, expected_jobs: int, problems: list[dict]) -> bool:
    if not report:
        return False
    missing = [k for k in EQUIVALENCE_REQUIRED if k not in report]
    if missing:
        problems.append({"class": "equivalence_report_missing_field", "fields": missing})
        return False
    expected = {
        "old_worker_sha256": spec_worker_sha,
        "new_worker_sha256": disk_worker_sha,
        "manifest_sha256": expected_manifest_sha,
        "bridge_sha256": expected_bridge_sha,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            problems.append({"class": "equivalence_report_sha_mismatch", "field": field, "expected": value, "actual": report.get(field)})
    for field in ["test_harness_sha256", "test_vector_inventory_sha256"]:
        if not is_valid_sha256(report.get(field)):
            problems.append({"class": "equivalence_report_invalid_sha", "field": field})
    if report.get("tested_valid_row_count") != expected_jobs or report.get("expected_valid_row_count") != expected_jobs:
        problems.append({"class": "equivalence_report_row_count_mismatch"})
    for field in DIFF_COUNTS:
        if report.get(field) != 0:
            problems.append({"class": "equivalence_report_nonzero_diff", "field": field, "count": report.get(field)})
    if report.get("overall_pass") is not True:
        problems.append({"class": "equivalence_report_overall_not_pass"})
    return not any(p["class"].startswith("equivalence_report_") for p in problems)


def evaluate(provenance: dict | list, *, spec_worker_sha: str, disk_worker_sha: str, expected_jobs: int, expected_manifest_sha: str, expected_bridge_sha: str, expected_telemetry_schema_sha: str, manifest_rows: list[dict] | None = None, equivalence: dict | None = None) -> dict:
    problems: list[dict] = []
    if expected_jobs <= 0:
        problems.append({"class": "invalid_expected_jobs"})
    if not is_valid_sha256(spec_worker_sha):
        problems.append({"class": "invalid_spec_worker_sha"})
    if not is_valid_sha256(disk_worker_sha):
        problems.append({"class": "invalid_disk_worker_sha"})
    if spec_worker_sha == disk_worker_sha:
        problems.append({"class": "spec_worker_equals_disk_worker"})
    for name, value in [("expected_manifest_sha", expected_manifest_sha), ("expected_bridge_sha", expected_bridge_sha), ("expected_telemetry_schema_sha", expected_telemetry_schema_sha)]:
        if not is_valid_sha256(value):
            problems.append({"class": "invalid_expected_sha", "field": name})

    rows = _rows(provenance)
    manifest_keys = _manifest_job_keys(manifest_rows)
    if manifest_keys is not None and len(manifest_keys) != expected_jobs:
        problems.append({"class": "manifest_job_count_mismatch", "expected": expected_jobs, "actual": len(manifest_keys)})

    seen: set[tuple[str, int]] = set()
    shas: list[str] = []
    accepted_keys: list[str] = []
    for i, row in enumerate(rows):
        missing = [f for f in REQUIRED_JOB_FIELDS if row.get(f) in (None, "")]
        if missing:
            problems.append({"class": "missing_runtime_provenance_field", "row": i, "fields": missing})
        key = (str(row.get("job_key", "")), int(row.get("attempt", -1)) if str(row.get("attempt", "")).lstrip("-").isdigit() else -1)
        if key in seen:
            problems.append({"class": "duplicate_job_attempt", "job_key": key[0], "attempt": key[1]})
        seen.add(key)
        if key[1] < 0:
            problems.append({"class": "invalid_attempt", "row": i})
        if not isinstance(row.get("accepted"), bool):
            problems.append({"class": "invalid_accepted_flag", "row": i})
        elif row["accepted"]:
            accepted_keys.append(key[0])
        if not isinstance(row.get("pid"), int) or row.get("pid", 0) <= 0:
            problems.append({"class": "invalid_pid", "row": i})
        if not _parse_time(row.get("start_time")):
            problems.append({"class": "invalid_start_time", "row": i})
        if row.get("provenance_source") not in ALLOWED_PROVENANCE_SOURCES:
            problems.append({"class": "untrusted_provenance_source", "row": i, "source": row.get("provenance_source")})
        for field, expected in [("bridge_sha", expected_bridge_sha), ("manifest_sha", expected_manifest_sha), ("telemetry_schema_sha", expected_telemetry_schema_sha)]:
            value = row.get(field)
            if not is_valid_sha256(value):
                problems.append({"class": "invalid_runtime_sha", "row": i, "field": field, "sha": value})
            elif value != expected:
                problems.append({"class": "runtime_sha_mismatch", "row": i, "field": field, "expected": expected, "actual": value})
        if not is_valid_sha256(row.get("provenance_evidence_sha256")):
            problems.append({"class": "invalid_provenance_evidence_sha", "row": i})
        sha = str(row.get("actual_loaded_worker_sha", ""))
        if not is_valid_sha256(sha):
            problems.append({"class": "invalid_actual_loaded_worker_sha", "row": i, "sha": sha})
        else:
            shas.append(sha)

    accepted_counts = Counter(accepted_keys)
    duplicate_accepted = sorted(k for k, v in accepted_counts.items() if v > 1)
    if duplicate_accepted:
        problems.append({"class": "duplicate_canonical_accepted_job", "job_keys": duplicate_accepted})
    if manifest_keys is not None:
        accepted_set = set(accepted_keys)
        missing = sorted(manifest_keys - accepted_set)
        extra = sorted(accepted_set - manifest_keys)
        if missing or extra:
            problems.append({"class": "canonical_job_set_mismatch", "missing": missing, "extra": extra})
    elif len(accepted_keys) != expected_jobs:
        problems.append({"class": "accepted_job_count_mismatch", "expected": expected_jobs, "actual": len(accepted_keys)})

    counts = dict(sorted(Counter(shas).items()))
    unique = set(counts)
    equivalence_pass = _equivalent(equivalence, spec_worker_sha=spec_worker_sha, disk_worker_sha=disk_worker_sha, expected_manifest_sha=expected_manifest_sha, expected_bridge_sha=expected_bridge_sha, expected_jobs=expected_jobs, problems=problems)
    if len(unique) > 1:
        binding = "VIS_RUNTIME_QUARANTINE_HOLD"
        decision = "VIS_RUNTIME_QUARANTINE_HOLD"
        problems.append({"class": "mixed_worker_versions", "sha_counts": counts})
    elif problems:
        binding = "RUNTIME_BINDING_P0_HOLD"
        decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
    elif unique == {spec_worker_sha}:
        binding = "CASE_A_SPEC_BOUND_WORKER"
        decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
    elif unique == {disk_worker_sha}:
        if equivalence_pass:
            binding = "CASE_B_POSTLAUNCH_RUNTIME_DEVIATION_REVIEW"
            decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
        else:
            binding = "RUNTIME_BINDING_P0_HOLD"
            decision = "VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT"
            problems.append({"class": "disk_worker_without_valid_row_equivalence"})
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
        "observed_attempt_rows": len(rows),
        "observed_accepted_jobs": len(accepted_keys),
        "actual_loaded_worker_sha_counts": counts,
        "equivalence_report_present": equivalence is not None,
        "valid_row_equivalence_pass": equivalence_pass,
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify TRUE_T10 worker runtime binding from offline provenance JSON.")
    add_path_arg(ap, "--runtime-provenance", required=True)
    add_path_arg(ap, "--output-json", required=True)
    add_path_arg(ap, "--manifest", required=True)
    add_path_arg(ap, "--equivalence-report")
    ap.add_argument("--spec-worker-sha", required=True)
    ap.add_argument("--disk-worker-sha", required=True)
    ap.add_argument("--expected-manifest-sha", required=True)
    ap.add_argument("--expected-bridge-sha", required=True)
    ap.add_argument("--expected-telemetry-schema-sha", required=True)
    ap.add_argument("--expected-jobs", type=int, default=162)
    ap.add_argument("--allow-case-b-review", action="store_true")
    args = ap.parse_args()
    result = evaluate(
        load_json(args.runtime_provenance),
        spec_worker_sha=args.spec_worker_sha,
        disk_worker_sha=args.disk_worker_sha,
        expected_jobs=args.expected_jobs,
        expected_manifest_sha=args.expected_manifest_sha,
        expected_bridge_sha=args.expected_bridge_sha,
        expected_telemetry_schema_sha=args.expected_telemetry_schema_sha,
        manifest_rows=load_jsonl(args.manifest),
        equivalence=load_json(args.equivalence_report) if args.equivalence_report else None,
    )
    actual_manifest_sha = sha256_file(args.manifest)
    if actual_manifest_sha != args.expected_manifest_sha:
        result["problems"].append({"class": "manifest_file_sha_mismatch", "expected": args.expected_manifest_sha, "actual": actual_manifest_sha})
        if result["worker_binding_status"] != "VIS_RUNTIME_QUARANTINE_HOLD":
            result["worker_binding_status"] = "RUNTIME_BINDING_P0_HOLD"
    write_json(args.output_json, result)
    print(canonical_json({"worker_binding_status": result["worker_binding_status"], "problems": len(result["problems"])}), end="")
    if result["worker_binding_status"] == "CASE_A_SPEC_BOUND_WORKER":
        return 0
    if result["worker_binding_status"] == "CASE_B_POSTLAUNCH_RUNTIME_DEVIATION_REVIEW":
        return 0 if args.allow_case_b_review else 3
    if result["worker_binding_status"] == "VIS_RUNTIME_QUARANTINE_HOLD":
        return 4
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
