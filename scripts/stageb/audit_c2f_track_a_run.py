#!/usr/bin/env python3
"""CPU-only C2F Track A completion audit."""
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
    if not meta.exists() or episode_completion(
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
    if jobs_file:
        expected_jobs = [
            {"parent_key": p, "condition": c}
            for p, c, *_ in (line.split("|") for line in jobs_file.read_text(encoding="utf-8").splitlines() if line.strip())
        ]
        expected_parents = sorted({j["parent_key"] for j in expected_jobs})
    else:
        expected_parents = [load_json_line["parent_key"] for load_json_line in (
            json.loads(line) for line in parent_manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        )]
        expected_jobs = [{"parent_key": p, "condition": c} for p in expected_parents for c in CONDITIONS]
    metas: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    complete_jobs: List[Dict[str, Any]] = []
    step_record_row_count = 0
    missing_step_records: List[str] = []
    empty_step_records: List[str] = []
    commit_mismatches: List[str] = []
    protocol_mismatches: List[str] = []
    metadata_paths = sorted(output_root.glob("**/episode_metadata.json"))
    step_paths = sorted(output_root.glob("**/step_records.jsonl"))
    for path in metadata_paths:
        try:
            meta = load_json(path)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        meta["_path"] = str(path)
        metas.append(meta)
    missing = []
    expected_paths = set()
    for job in expected_jobs:
        path = output_root / job["parent_key"] / job["condition"] / "episode_metadata.json"
        expected_paths.add(str(path))
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
        if any(r.startswith("PROTOCOL_") for r in reasons):
            protocol_mismatches.append(str(path))

    for meta in metas:
        path = Path(meta["_path"])
        if str(path) in expected_paths:
            continue
        status = episode_completion(
            path,
            expected_commit=expected_commit,
            expected_protocol_name=expected_protocol_name,
            expected_protocol_version=expected_protocol_version,
        )
        if not status["complete"]:
            invalid.append({"path": str(path), "parent_key": str(meta.get("parent_key", "")), "condition": str(meta.get("condition", "")), "reasons": status["reasons"]})

    invalid_by_path: Dict[str, Dict[str, Any]] = {}
    for row in invalid:
        current = invalid_by_path.setdefault(row["path"], dict(row))
        current["reasons"] = sorted(set(current.get("reasons", [])) | set(row.get("reasons", [])))
        if row.get("error"):
            current["error"] = row["error"]
    invalid = list(invalid_by_path.values())

    valid = complete_jobs
    delivery = [int(m.get("delivery_count", 0)) for m in valid if m.get("condition") in CONDITIONS[1:]]
    no_emit = [
        {"parent_key": m.get("parent_key"), "condition": m.get("condition")}
        for m in valid
        if m.get("condition") in CONDITIONS[1:] and int(m.get("attack_window_start", -1)) < 0
    ]
    audit = {
        "expected_parents": len(expected_parents),
        "expected_episodes": len(expected_jobs),
        "metadata_count": len(metadata_paths),
        "step_record_file_count": len(step_paths),
        "step_record_row_count": step_record_row_count,
        "valid_complete_job_count": len(valid),
        "valid_episode_count": len(valid),
        "invalid_episode_count": len(invalid),
        "runtime_invalid_job_count": sum(1 for row in invalid if "RUNTIME_INVALID" in row.get("reasons", [])),
        "by_condition_valid": dict(Counter(m.get("condition") for m in valid)),
        "by_suite_valid": dict(Counter(m.get("suite") for m in valid)),
        "missing": missing,
        "invalid": invalid,
        "missing_step_records": missing_step_records,
        "empty_step_records": empty_step_records,
        "commit_mismatches": commit_mismatches,
        "protocol_mismatches": protocol_mismatches,
        "expected_commit": expected_commit,
        "expected_protocol_name": expected_protocol_name,
        "expected_protocol_version": expected_protocol_version,
        "complete": len(missing) == 0 and len(invalid) == 0,
        "no_emit": no_emit,
        "delivery_count_min": min(delivery) if delivery else None,
        "delivery_count_max": max(delivery) if delivery else None,
        "delivery_count_mean": sum(delivery) / len(delivery) if delivery else None,
    }
    (run_root / "postrun_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["parent_key,suite,task_index,state_id,condition,success,runtime_valid,total_steps,attack_window_start,attack_window_end,delivery_count"]
    for m in valid:
        rows.append(",".join(str(m.get(k, "")) for k in [
            "parent_key", "suite", "task_index", "state_id", "condition", "success", "runtime_valid",
            "total_steps", "attack_window_start", "attack_window_end", "delivery_count",
        ]))
    (run_root / "summary_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-complete")
    ap.add_argument("--archive-invalid-output-root")
    ap.add_argument("--parent-key")
    ap.add_argument("--condition")
    ap.add_argument("--invalid-archive-root")
    ap.add_argument("--run-root")
    ap.add_argument("--output-root")
    ap.add_argument("--parent-manifest")
    ap.add_argument("--jobs-file")
    ap.add_argument("--expected-git-commit", default="")
    ap.add_argument("--expected-condition", default="")
    ap.add_argument("--expected-parent-key", default="")
    ap.add_argument("--expected-protocol-name", default=PROTOCOL_NAME)
    ap.add_argument("--expected-protocol-version", default=PROTOCOL_VERSION)
    args = ap.parse_args()

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
