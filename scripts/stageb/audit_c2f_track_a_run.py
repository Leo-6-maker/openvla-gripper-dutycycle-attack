#!/usr/bin/env python3
"""CPU-only C2F Track A completion audit with closed-world run accounting."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


CONDITIONS = ["CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"]
PROTOCOL_NAME = "C2F_TRACK_A_CMDOPEN_ACTION_SPACE"
PROTOCOL_VERSION = "2026-07-10.v2"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_runtime_valid_metadata(path: Path) -> bool:
    try:
        return bool(load_json(path).get("runtime_valid") is True)
    except Exception:
        return False


def episode_completion(
    metadata_path: Path,
    *,
    expected_commit: str = "",
    expected_condition: str = "",
    expected_parent_key: str = "",
    expected_protocol_name: str = PROTOCOL_NAME,
    expected_protocol_version: str = PROTOCOL_VERSION,
) -> Dict[str, Any]:
    reasons: List[str] = []
    try:
        meta = load_json(metadata_path)
    except Exception as exc:
        return {"complete": False, "reasons": ["METADATA_MISSING_OR_INVALID"], "error": str(exc), "step_record_count": 0}
    if meta.get("runtime_valid") is not True:
        reasons.append("RUNTIME_INVALID")
    if type(meta.get("success")) is not bool:
        reasons.append("SUCCESS_NOT_BOOLEAN")
    condition = str(meta.get("condition", ""))
    if condition not in CONDITIONS or (expected_condition and condition != expected_condition):
        reasons.append("CONDITION_MISMATCH")
    if expected_parent_key and str(meta.get("parent_key", "")) != expected_parent_key:
        reasons.append("PARENT_KEY_MISMATCH")
    commit = str(meta.get("git_commit", ""))
    if not FULL_SHA_RE.fullmatch(commit) or (expected_commit and commit != expected_commit):
        reasons.append("COMMIT_MISMATCH")
    if str(meta.get("protocol_name", "")) != expected_protocol_name:
        reasons.append("PROTOCOL_NAME_MISMATCH")
    if str(meta.get("protocol_version", "")) != expected_protocol_version:
        reasons.append("PROTOCOL_VERSION_MISMATCH")
    steps_path = metadata_path.with_name("step_records.jsonl")
    step_count = 0
    if not steps_path.is_file():
        reasons.append("STEP_RECORDS_MISSING")
    else:
        try:
            for line in steps_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
                    step_count += 1
        except Exception as exc:
            reasons.append("STEP_RECORDS_INVALID")
            return {"complete": False, "reasons": reasons, "error": str(exc), "metadata": meta, "step_record_count": step_count}
        if step_count == 0:
            reasons.append("STEP_RECORDS_EMPTY")
    return {"complete": not reasons, "reasons": reasons, "metadata": meta, "step_record_count": step_count}


def archive_invalid_attempt(
    output_root: Path,
    parent_key: str,
    condition: str,
    archive_root: Path,
    *,
    expected_commit: str = "",
    expected_protocol_name: str = PROTOCOL_NAME,
    expected_protocol_version: str = PROTOCOL_VERSION,
) -> bool:
    ep_dir = output_root / parent_key / condition
    meta = ep_dir / "episode_metadata.json"
    if not ep_dir.exists():
        return False
    if meta.exists() and episode_completion(
        meta,
        expected_commit=expected_commit,
        expected_condition=condition,
        expected_parent_key=parent_key,
        expected_protocol_name=expected_protocol_name,
        expected_protocol_version=expected_protocol_version,
    )["complete"]:
        return False
    dest = archive_root / parent_key / condition
    i = 1
    while (dest / f"attempt_{i:03d}").exists():
        i += 1
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ep_dir), str(dest / f"attempt_{i:03d}"))
    return True


def _job_from_artifact(path: Path, output_root: Path) -> tuple[str, str] | None:
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 3:
        return None
    return "/".join(parts[:-2]), parts[-2]


def _expected_jobs(parent_manifest: Path, jobs_file: Path | None) -> tuple[list[dict[str, str]], list[str], list[str]]:
    duplicate_jobs: list[str] = []
    duplicate_parents: list[str] = []
    if jobs_file:
        jobs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for line in jobs_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parent, condition, *_ = line.split("|")
            key = (parent, condition)
            if key in seen:
                duplicate_jobs.append(f"{parent}|{condition}")
            seen.add(key)
            jobs.append({"parent_key": parent, "condition": condition})
        return jobs, sorted(set(duplicate_jobs)), duplicate_parents
    parents: list[str] = []
    seen_parents: set[str] = set()
    for line in parent_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parent = str(json.loads(line)["parent_key"])
        if parent in seen_parents:
            duplicate_parents.append(parent)
        seen_parents.add(parent)
        parents.append(parent)
    jobs = [{"parent_key": parent, "condition": condition} for parent in parents for condition in CONDITIONS]
    return jobs, duplicate_jobs, sorted(set(duplicate_parents))


def audit_run(
    run_root: Path,
    output_root: Path,
    parent_manifest: Path,
    jobs_file: Path | None = None,
    *,
    expected_commit: str = "",
    expected_protocol_name: str = PROTOCOL_NAME,
    expected_protocol_version: str = PROTOCOL_VERSION,
) -> Dict[str, Any]:
    expected_jobs, duplicate_expected_jobs, duplicate_expected_parents = _expected_jobs(parent_manifest, jobs_file)
    expected_parents = sorted({job["parent_key"] for job in expected_jobs})
    expected_keys = {(job["parent_key"], job["condition"]) for job in expected_jobs}
    invalid: List[Dict[str, Any]] = []
    complete_jobs: List[Dict[str, Any]] = []
    step_record_row_count = 0
    missing_step_records: List[str] = []
    empty_step_records: List[str] = []
    commit_mismatches: List[str] = []
    protocol_mismatches: List[str] = []
    metadata_paths = sorted(output_root.glob("**/episode_metadata.json"))
    step_paths = sorted(output_root.glob("**/step_records.jsonl"))
    actual_metadata_keys = {key for path in metadata_paths if (key := _job_from_artifact(path, output_root)) is not None}
    actual_step_keys = {key for path in step_paths if (key := _job_from_artifact(path, output_root)) is not None}
    unexpected_episode_keys = sorted(actual_metadata_keys - expected_keys)
    unexpected_step_record_keys = sorted(actual_step_keys - expected_keys)
    missing: list[dict[str, str]] = []
    for job in expected_jobs:
        path = output_root / job["parent_key"] / job["condition"] / "episode_metadata.json"
        if not path.exists():
            missing.append(job)
            continue
        status = episode_completion(
            path,
            expected_commit=expected_commit,
            expected_condition=job["condition"],
            expected_parent_key=job["parent_key"],
            expected_protocol_name=expected_protocol_name,
            expected_protocol_version=expected_protocol_version,
        )
        step_record_row_count += int(status["step_record_count"])
        if status["complete"]:
            row = dict(status["metadata"])
            row["_path"] = str(path)
            complete_jobs.append(row)
            continue
        reasons = status["reasons"]
        invalid.append({"path": str(path), "parent_key": job["parent_key"], "condition": job["condition"], "reasons": reasons})
        if "STEP_RECORDS_MISSING" in reasons:
            missing_step_records.append(str(path.with_name("step_records.jsonl")))
        if "STEP_RECORDS_EMPTY" in reasons:
            empty_step_records.append(str(path.with_name("step_records.jsonl")))
        if "COMMIT_MISMATCH" in reasons:
            commit_mismatches.append(str(path))
        if any(reason.startswith("PROTOCOL_") for reason in reasons):
            protocol_mismatches.append(str(path))
    for path in metadata_paths:
        key = _job_from_artifact(path, output_root)
        if key not in expected_keys:
            status = episode_completion(
                path,
                expected_commit=expected_commit,
                expected_protocol_name=expected_protocol_name,
                expected_protocol_version=expected_protocol_version,
            )
            if not status["complete"]:
                invalid.append({"path": str(path), "parent_key": key[0] if key else "", "condition": key[1] if key else "", "reasons": status["reasons"]})
    invalid_by_path: Dict[str, Dict[str, Any]] = {}
    for row in invalid:
        current = invalid_by_path.setdefault(row["path"], dict(row))
        current["reasons"] = sorted(set(current.get("reasons", [])) | set(row.get("reasons", [])))
        if row.get("error"):
            current["error"] = row["error"]
    invalid = list(invalid_by_path.values())
    unique_valid_keys = {(str(meta.get("parent_key", "")), str(meta.get("condition", ""))) for meta in complete_jobs}
    delivery = [int(meta.get("delivery_count", 0)) for meta in complete_jobs if meta.get("condition") in CONDITIONS[1:]]
    no_emit = [
        {"parent_key": meta.get("parent_key"), "condition": meta.get("condition")}
        for meta in complete_jobs
        if meta.get("condition") in CONDITIONS[1:] and int(meta.get("attack_window_start", -1)) < 0
    ]
    exact_job_set_match = unique_valid_keys == expected_keys and actual_metadata_keys == expected_keys and actual_step_keys == expected_keys
    complete = (
        not missing and not invalid and not duplicate_expected_jobs and not duplicate_expected_parents
        and not unexpected_episode_keys and not unexpected_step_record_keys and exact_job_set_match
    )
    audit = {
        "expected_parents": len(expected_parents),
        "expected_episodes": len(expected_jobs),
        "unique_expected_job_count": len(expected_keys),
        "metadata_count": len(metadata_paths),
        "step_record_file_count": len(step_paths),
        "step_record_row_count": step_record_row_count,
        "valid_complete_job_count": len(complete_jobs),
        "unique_valid_job_count": len(unique_valid_keys),
        "valid_episode_count": len(complete_jobs),
        "invalid_episode_count": len(invalid),
        "runtime_invalid_job_count": sum(1 for row in invalid if "RUNTIME_INVALID" in row.get("reasons", [])),
        "by_condition_valid": dict(Counter(meta.get("condition") for meta in complete_jobs)),
        "by_suite_valid": dict(Counter(meta.get("suite") for meta in complete_jobs)),
        "missing": missing,
        "invalid": invalid,
        "missing_step_records": missing_step_records,
        "empty_step_records": empty_step_records,
        "commit_mismatches": commit_mismatches,
        "protocol_mismatches": protocol_mismatches,
        "duplicate_expected_jobs": duplicate_expected_jobs,
        "duplicate_expected_parents": duplicate_expected_parents,
        "unexpected_episode_keys": [f"{parent}|{condition}" for parent, condition in unexpected_episode_keys],
        "unexpected_step_record_keys": [f"{parent}|{condition}" for parent, condition in unexpected_step_record_keys],
        "exact_job_set_match": exact_job_set_match,
        "expected_commit": expected_commit,
        "expected_protocol_name": expected_protocol_name,
        "expected_protocol_version": expected_protocol_version,
        "complete": complete,
        "no_emit": no_emit,
        "delivery_count_min": min(delivery) if delivery else None,
        "delivery_count_max": max(delivery) if delivery else None,
        "delivery_count_mean": sum(delivery) / len(delivery) if delivery else None,
    }
    (run_root / "postrun_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["parent_key,suite,task_index,state_id,condition,success,runtime_valid,total_steps,attack_window_start,attack_window_end,delivery_count"]
    for meta in complete_jobs:
        rows.append(",".join(str(meta.get(key, "")) for key in [
            "parent_key", "suite", "task_index", "state_id", "condition", "success", "runtime_valid",
            "total_steps", "attack_window_start", "attack_window_end", "delivery_count",
        ]))
    (run_root / "summary_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-complete")
    parser.add_argument("--archive-invalid-output-root")
    parser.add_argument("--parent-key")
    parser.add_argument("--condition")
    parser.add_argument("--invalid-archive-root")
    parser.add_argument("--run-root")
    parser.add_argument("--output-root")
    parser.add_argument("--parent-manifest")
    parser.add_argument("--jobs-file")
    parser.add_argument("--expected-git-commit", default="")
    parser.add_argument("--expected-condition", default="")
    parser.add_argument("--expected-parent-key", default="")
    parser.add_argument("--expected-protocol-name", default=PROTOCOL_NAME)
    parser.add_argument("--expected-protocol-version", default=PROTOCOL_VERSION)
    args = parser.parse_args()
    if args.metadata_complete:
        status = episode_completion(
            Path(args.metadata_complete),
            expected_commit=args.expected_git_commit,
            expected_condition=args.expected_condition,
            expected_parent_key=args.expected_parent_key,
            expected_protocol_name=args.expected_protocol_name,
            expected_protocol_version=args.expected_protocol_version,
        )
        return 0 if status["complete"] else 1
    if args.archive_invalid_output_root:
        moved = archive_invalid_attempt(
            Path(args.archive_invalid_output_root),
            args.parent_key or "",
            args.condition or "",
            Path(args.invalid_archive_root or "invalid_attempts"),
            expected_commit=args.expected_git_commit,
            expected_protocol_name=args.expected_protocol_name,
            expected_protocol_version=args.expected_protocol_version,
        )
        print(json.dumps({"archived": moved}, sort_keys=True))
        return 0
    audit = audit_run(
        Path(args.run_root),
        Path(args.output_root),
        Path(args.parent_manifest),
        Path(args.jobs_file) if args.jobs_file else None,
        expected_commit=args.expected_git_commit,
        expected_protocol_name=args.expected_protocol_name,
        expected_protocol_version=args.expected_protocol_version,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
