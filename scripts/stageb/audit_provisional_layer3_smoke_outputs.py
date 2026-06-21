#!/usr/bin/env python3
"""Audit provisional Layer3 smoke rollout outputs against a frozen manifest.

This is a metadata/integrity auditor for engineering smoke runs. It does not
score attack effectiveness and should not be used to claim VIS superiority.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAIL_FIELDS = [
    "missing_summary",
    "missing_step_telemetry",
    "telemetry_length_mismatch",
    "raw_video_decode_failure",
    "overlay_video_decode_failure",
    "student_trigger_contract_failure",
    "arm_preservation_contract_failure",
    "invalid_feature_episode_count",
    "checkpoint_sha_mismatch",
    "dataset_sha_mismatch",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_data_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return None


def video_frame_count(path: Path) -> tuple[bool, int | None, str]:
    if not path.exists():
        return False, None, "missing"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, None, repr(exc)
    if proc.returncode != 0:
        return False, None, (proc.stderr or proc.stdout).strip()
    text = (proc.stdout or "").strip().splitlines()
    if not text:
        return False, None, "no frame count"
    if text[0].strip().upper() == "N/A":
        return True, None, "frame count unavailable"
    try:
        return True, int(text[0].strip()), ""
    except ValueError:
        return False, None, f"non-integer frame count: {text[0]!r}"


def requested_condition(manifest_condition: str) -> str:
    return "VIS" if manifest_condition == "VIS" else manifest_condition


def audit_job(row: dict[str, str], expected_dataset_sha: str | None) -> dict[str, Any]:
    out_dir = Path(row["output_dir"])
    summary_path = out_dir / "episode_summary.json"
    telemetry_path = out_dir / "step_telemetry.csv"
    raw_video_path = out_dir / "rollout_raw.mp4"
    overlay_video_path = out_dir / "rollout_overlay.mp4"

    result: dict[str, Any] = {
        "job_id": row.get("job_id", ""),
        "parent_key": row.get("parent_key", ""),
        "suite": row.get("suite", ""),
        "task_idx": row.get("task_idx", ""),
        "state_id": row.get("state_id", ""),
        "condition": row.get("condition", ""),
        "output_dir": str(out_dir),
        "expected_detector_sha256": row.get("expected_detector_sha256", ""),
        "status": "COMPLETE",
    }
    for field in FAIL_FIELDS:
        result[field] = 0

    if not summary_path.exists():
        result["missing_summary"] = 1
        result["status"] = "MISSING_SUMMARY"
        return result

    try:
        summary = load_json(summary_path)
    except Exception as exc:
        result["missing_summary"] = 1
        result["status"] = f"SUMMARY_READ_ERROR:{exc!r}"
        return result

    n_steps = summary.get("n_steps")
    result.update(
        {
            "summary_condition": summary.get("condition"),
            "requested_condition": summary.get("requested_condition"),
            "task_success": summary.get("task_success"),
            "n_steps": n_steps,
            "mlp_triggered": summary.get("mlp_triggered"),
            "mlp_emit_step": summary.get("mlp_emit_step"),
            "attack_frames": summary.get("attack_frames"),
            "token_open_duty": summary.get("token_open_duty"),
            "env_open_duty": summary.get("env_open_duty"),
            "arm_duty": summary.get("arm_duty"),
            "invalid_feature_steps": summary.get("invalid_feature_steps"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "dataset_sha256": summary.get("dataset_sha256"),
            "privileged_detector_input_used": summary.get("privileged_detector_input_used"),
            "manual_anchor_used": summary.get("manual_anchor_used"),
            "arm_action_preservation_mode": summary.get("arm_action_preservation_mode"),
        }
    )

    telemetry_rows = count_csv_data_rows(telemetry_path)
    result["step_telemetry_rows"] = telemetry_rows
    if telemetry_rows is None:
        result["missing_step_telemetry"] = 1
    elif isinstance(n_steps, int) and telemetry_rows != n_steps:
        result["telemetry_length_mismatch"] = 1

    raw_ok, raw_frames, raw_error = video_frame_count(raw_video_path)
    overlay_ok, overlay_frames, overlay_error = video_frame_count(overlay_video_path)
    result["raw_video_frames"] = raw_frames
    result["overlay_video_frames"] = overlay_frames
    result["raw_video_error"] = raw_error
    result["overlay_video_error"] = overlay_error
    if not raw_ok:
        result["raw_video_decode_failure"] = 1
    if not overlay_ok:
        result["overlay_video_decode_failure"] = 1

    if summary.get("privileged_detector_input_used") is not False or summary.get("manual_anchor_used") is not False:
        result["student_trigger_contract_failure"] = 1

    if summary.get("arm_action_preservation_mode") != "execute_clean_arm_with_attacked_gripper":
        result["arm_preservation_contract_failure"] = 1

    if summary.get("invalid_feature_steps") != 0:
        result["invalid_feature_episode_count"] = 1

    if summary.get("checkpoint_sha256") != row.get("expected_detector_sha256"):
        result["checkpoint_sha_mismatch"] = 1

    if expected_dataset_sha and summary.get("dataset_sha256") != expected_dataset_sha:
        result["dataset_sha_mismatch"] = 1

    if summary.get("requested_condition") != requested_condition(row.get("condition", "")):
        result["status"] = "REQUESTED_CONDITION_MISMATCH"

    if any(result[field] for field in FAIL_FIELDS):
        result["status"] = "AUDIT_FAIL"

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-manifest", required=True, type=Path)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-dataset-sha", default=None)
    parser.add_argument("--accepted-label", default="ENGINEERING_PASS")
    args = parser.parse_args()

    jobs = read_csv(args.job_manifest)
    rows = [audit_job(job, args.expected_dataset_sha) for job in jobs]

    key_counts = Counter((r["parent_key"], r["condition"]) for r in rows)
    duplicate_keys = sum(1 for count in key_counts.values() if count > 1)
    fail_counts = {field: sum(int(r.get(field, 0) or 0) for r in rows) for field in FAIL_FIELDS}

    by_suite: dict[str, dict[str, int]] = defaultdict(lambda: {"planned": 0, "complete": 0, "audit_fail": 0})
    for row in rows:
        suite = str(row.get("suite", ""))
        by_suite[suite]["planned"] += 1
        if row.get("status") == "COMPLETE":
            by_suite[suite]["complete"] += 1
        else:
            by_suite[suite]["audit_fail"] += 1

    all_fail_counts_zero = all(value == 0 for value in fail_counts.values())
    complete_jobs = sum(1 for r in rows if r.get("status") == "COMPLETE")
    accepted = (
        len(rows) == len(jobs)
        and complete_jobs == len(jobs)
        and duplicate_keys == 0
        and all_fail_counts_zero
    )

    summary = {
        "stage": args.stage,
        "result_class": args.accepted_label if accepted else "AUDIT_FAIL",
        "accepted_engineering_smoke": accepted,
        "planned_jobs": len(jobs),
        "audited_jobs": len(rows),
        "complete_jobs": complete_jobs,
        "duplicate_parent_condition_keys": duplicate_keys,
        "all_fail_counts_zero": all_fail_counts_zero,
        "fail_counts": fail_counts,
        "by_suite": dict(by_suite),
        "expected_dataset_sha": args.expected_dataset_sha,
        "job_manifest": str(args.job_manifest),
    }

    write_csv(args.rows_csv, rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
