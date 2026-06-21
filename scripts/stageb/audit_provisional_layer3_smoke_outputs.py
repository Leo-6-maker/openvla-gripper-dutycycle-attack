#!/usr/bin/env python3
"""Audit provisional Layer3 smoke rollout outputs against a frozen manifest.

This is a metadata/integrity auditor for engineering smoke runs. It does not
score attack effectiveness and should not be used to claim VIS superiority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    "manifest_identity_mismatch",
    "command_ledger_mismatch",
    "attack_timing_contract_failure",
    "attack_count_contract_failure",
    "video_frame_mismatch",
    "parent_condition_set_failure",
    "matched_emit_mismatch",
]

CONDITIONS = {"CLEAN", "VIS", "RAND", "SHUFFLED"}
MAX_ATTACK_FRAMES = 10


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def video_full_decode_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, repr(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, ""


def requested_condition(manifest_condition: str) -> str:
    return "VIS" if manifest_condition == "VIS" else manifest_condition


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def command_contains(command: str, *parts: str) -> bool:
    return all(part in command for part in parts)


def audit_command_ledger(row: dict[str, str], ledger_row: dict[str, str] | None) -> tuple[int, str]:
    if ledger_row is None:
        return 1, "missing ledger row"
    if ledger_row.get("status") != "COMPLETE" or str(ledger_row.get("returncode")) != "0":
        return 1, "ledger row not COMPLETE/0"
    if ledger_row.get("output_dir") != row.get("output_dir"):
        return 1, "ledger output_dir mismatch"
    command = ledger_row.get("command", "")
    checks = [
        f"--condition {row.get('condition')}",
        f"--suite {row.get('suite')}",
        f"--model_path {row.get('model_path')}",
        f"--unnorm_key {row.get('unnorm_key')}",
        f"--task_idx {row.get('task_idx')}",
        f"--state_id {row.get('state_id')}",
        f"--anchor {row.get('teacher_anchor')}",
        f"--seed_id {row.get('attack_seed')}",
        f"--output_dir {row.get('output_dir')}",
        f"--render_gpu {row.get('render_gpu')}",
        f"--mlp_path {row.get('detector_path')}",
    ]
    missing = [part for part in checks if part not in command]
    if missing:
        return 1, "command missing: " + ";".join(missing)
    return 0, ""


def audit_telemetry_contract(
    rows: list[dict[str, str]] | None,
    manifest_condition: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "attack_timing_contract_failure": 0,
        "attack_count_contract_failure": 0,
        "attack_this_rows": None,
        "max_attack_count_observed": None,
        "first_attack_step": None,
        "last_attack_step": None,
        "telemetry_contract_note": "",
    }
    if rows is None:
        out["attack_timing_contract_failure"] = 1
        out["telemetry_contract_note"] = "telemetry unreadable"
        return out

    emit = parse_int(summary.get("mlp_emit_step"))
    attack_frames_summary = parse_int(summary.get("attack_frames")) or 0
    triggered = parse_bool(summary.get("mlp_triggered"))

    attack_steps: list[int] = []
    attack_counts: list[int] = []
    prev_count = 0
    for row in rows:
        step = parse_int(row.get("step"))
        attack_this = parse_bool(row.get("attack_this")) is True
        count = parse_int(row.get("attack_count"))
        if count is not None:
            if count < prev_count:
                out["attack_count_contract_failure"] = 1
            prev_count = count
            attack_counts.append(count)
        if attack_this:
            if step is None:
                out["attack_timing_contract_failure"] = 1
                continue
            attack_steps.append(step)
            if emit is None or emit < 0 or step < emit:
                out["attack_timing_contract_failure"] = 1

    out["attack_this_rows"] = len(attack_steps)
    out["max_attack_count_observed"] = max(attack_counts) if attack_counts else 0
    out["first_attack_step"] = min(attack_steps) if attack_steps else None
    out["last_attack_step"] = max(attack_steps) if attack_steps else None

    should_have_zero_attack = manifest_condition == "CLEAN" or triggered is False or emit is None or emit < 0
    if should_have_zero_attack:
        if attack_steps or attack_frames_summary != 0 or out["max_attack_count_observed"] != 0:
            out["attack_timing_contract_failure"] = 1
            out["attack_count_contract_failure"] = 1
        return out

    if attack_frames_summary != len(attack_steps):
        out["attack_count_contract_failure"] = 1
    if len(attack_steps) > MAX_ATTACK_FRAMES or out["max_attack_count_observed"] > MAX_ATTACK_FRAMES:
        out["attack_count_contract_failure"] = 1
    if attack_steps:
        expected = list(range(attack_steps[0], attack_steps[0] + len(attack_steps)))
        if attack_steps != expected:
            out["attack_timing_contract_failure"] = 1
        expected_counts = list(range(1, len(attack_steps) + 1))
        actual_attack_counts = [
            parse_int(row.get("attack_count"))
            for row in rows
            if parse_bool(row.get("attack_this")) is True
        ]
        if actual_attack_counts != expected_counts:
            out["attack_count_contract_failure"] = 1
    return out


def audit_identity(row: dict[str, str], summary: dict[str, Any]) -> tuple[int, str]:
    checks = {
        "suite": summary.get("suite"),
        "task_idx": summary.get("task_idx"),
        "state_id": summary.get("state_id"),
        "teacher_anchor": summary.get("teacher_anchor"),
        "unnorm_key": summary.get("unnorm_key"),
        "requested_condition": summary.get("requested_condition"),
    }
    expected = {
        "suite": row.get("suite"),
        "task_idx": parse_int(row.get("task_idx")),
        "state_id": parse_int(row.get("state_id")),
        "teacher_anchor": parse_int(row.get("teacher_anchor")),
        "unnorm_key": row.get("unnorm_key"),
        "requested_condition": requested_condition(row.get("condition", "")),
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual = checks[key]
        if key in {"task_idx", "state_id", "teacher_anchor"}:
            actual = parse_int(actual)
        if actual != expected_value:
            mismatches.append(f"{key}: actual={actual!r} expected={expected_value!r}")
    return (1, "; ".join(mismatches)) if mismatches else (0, "")


def audit_job(
    row: dict[str, str],
    expected_dataset_sha: str | None,
    ledger_by_job: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
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
    identity_failed, identity_note = audit_identity(row, summary)
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
            "manifest_identity_note": identity_note,
        }
    )
    result["manifest_identity_mismatch"] = identity_failed

    if ledger_by_job is not None:
        ledger_failed, ledger_note = audit_command_ledger(row, ledger_by_job.get(row.get("job_id", "")))
        result["command_ledger_mismatch"] = ledger_failed
        result["command_ledger_note"] = ledger_note

    telemetry_data = read_csv_rows(telemetry_path)
    telemetry_rows = None if telemetry_data is None else len(telemetry_data)
    result["step_telemetry_rows"] = telemetry_rows
    if telemetry_data is None:
        result["missing_step_telemetry"] = 1
    elif isinstance(n_steps, int) and telemetry_rows != n_steps:
        result["telemetry_length_mismatch"] = 1
    if telemetry_data is not None:
        result.update(audit_telemetry_contract(telemetry_data, row.get("condition", ""), summary))

    raw_ok, raw_frames, raw_error = video_frame_count(raw_video_path)
    overlay_ok, overlay_frames, overlay_error = video_frame_count(overlay_video_path)
    raw_decode_ok, raw_decode_error = video_full_decode_ok(raw_video_path)
    overlay_decode_ok, overlay_decode_error = video_full_decode_ok(overlay_video_path)
    result["raw_video_frames"] = raw_frames
    result["overlay_video_frames"] = overlay_frames
    result["raw_video_error"] = raw_error
    result["overlay_video_error"] = overlay_error
    result["raw_video_decode_error"] = raw_decode_error
    result["overlay_video_decode_error"] = overlay_decode_error
    if not (raw_ok and raw_decode_ok):
        result["raw_video_decode_failure"] = 1
    if not (overlay_ok and overlay_decode_ok):
        result["overlay_video_decode_failure"] = 1
    if isinstance(n_steps, int) and (raw_frames != n_steps or overlay_frames != n_steps):
        result["video_frame_mismatch"] = 1

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

    detector_path = Path(row.get("detector_path", ""))
    if detector_path.exists():
        result["detector_file_sha256"] = sha256_file(detector_path)
        if result["detector_file_sha256"] != row.get("expected_detector_sha256"):
            result["checkpoint_sha_mismatch"] = 1
    else:
        result["detector_file_sha256"] = ""
        result["checkpoint_sha_mismatch"] = 1

    if summary.get("requested_condition") != requested_condition(row.get("condition", "")):
        result["status"] = "REQUESTED_CONDITION_MISMATCH"

    if any(result[field] for field in FAIL_FIELDS):
        result["status"] = "AUDIT_FAIL"

    return result


def add_group_contracts(rows: list[dict[str, Any]]) -> None:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row.get("parent_key", ""))].append(row)

    for parent_key, parent_rows in by_parent.items():
        conds = {str(row.get("condition", "")) for row in parent_rows}
        if conds != CONDITIONS or len(parent_rows) != len(CONDITIONS):
            for row in parent_rows:
                row["parent_condition_set_failure"] = 1
                row["parent_condition_set_note"] = f"{parent_key}: conditions={sorted(conds)} rows={len(parent_rows)}"

        emits = {
            str(row.get("condition", "")): (
                str(row.get("mlp_triggered", "")),
                str(row.get("mlp_emit_step", "")),
            )
            for row in parent_rows
        }
        if len(set(emits.values())) > 1:
            for row in parent_rows:
                row["matched_emit_mismatch"] = 1
                row["matched_emit_note"] = json.dumps(emits, sort_keys=True)


def build_recursive_sha_manifest(
    jobs: list[dict[str, str]],
    rows: list[dict[str, Any]],
    ledger_path: Path | None,
    output_path: Path,
    extra_paths: list[Path],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add_path(path: Path, scope: str, job_id: str = "") -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        if path.is_file():
            records.append(
                {
                    "scope": scope,
                    "job_id": job_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    for job in jobs:
        out_dir = Path(job["output_dir"])
        if out_dir.exists():
            for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
                add_path(path, "episode_output", job.get("job_id", ""))

    if ledger_path is not None:
        add_path(ledger_path, "worker_ledger")
    for row in rows:
        log_path = row.get("ledger_log_path") or row.get("log_path")
        if log_path:
            add_path(Path(str(log_path)), "worker_log", str(row.get("job_id", "")))
    for path in extra_paths:
        add_path(path, "extra")

    write_csv(output_path, records)
    sha = sha256_file(output_path)
    return {"recursive_sha_manifest": str(output_path), "recursive_sha_manifest_sha256": sha, "sealed_file_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-manifest", required=True, type=Path)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-dataset-sha", default=None)
    parser.add_argument("--accepted-label", default="ENGINEERING_PASS")
    parser.add_argument("--worker-ledger", type=Path, default=None)
    parser.add_argument("--recursive-sha-csv", type=Path, default=None)
    parser.add_argument("--seal-path", action="append", type=Path, default=[])
    args = parser.parse_args()

    jobs = read_csv(args.job_manifest)
    ledger_by_job = None
    if args.worker_ledger is not None:
        ledger_rows = read_csv(args.worker_ledger)
        ledger_by_job = {row.get("job_id", ""): row for row in ledger_rows}
    rows = [audit_job(job, args.expected_dataset_sha, ledger_by_job=ledger_by_job) for job in jobs]
    if ledger_by_job is not None:
        for row in rows:
            ledger = ledger_by_job.get(str(row.get("job_id", "")))
            if ledger:
                row["ledger_log_path"] = ledger.get("log_path", "")
                row["ledger_duration_sec"] = ledger.get("duration_sec", "")
                row["ledger_returncode"] = ledger.get("returncode", "")
    add_group_contracts(rows)

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
        "worker_ledger": str(args.worker_ledger) if args.worker_ledger else None,
        "runtime_contract_level": {
            "trigger_timing": "telemetry_row_audited",
            "arm_preservation": "source_level_and_summary_mode_audited",
            "arm_vector_runtime_note": "step telemetry does not store clean and executed 6D arm vectors, so per-dimension runtime equality is not independently auditable from these outputs",
        },
    }

    write_csv(args.rows_csv, rows)
    if args.recursive_sha_csv is not None:
        seal = build_recursive_sha_manifest(
            jobs=jobs,
            rows=rows,
            ledger_path=args.worker_ledger,
            output_path=args.recursive_sha_csv,
            extra_paths=[args.job_manifest, *args.seal_path],
        )
        summary.update(seal)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
