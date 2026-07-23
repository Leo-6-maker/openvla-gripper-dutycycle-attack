#!/usr/bin/env python3
"""B2: Validate pilot execution — K=10 closure, condition parity, arm parity, video/telemetry."""
from __future__ import annotations

import argparse, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from pilot_integrity import sha256_file, load_strict_json, seal_output_dir, is_64char_hex

SELF_SHA = None
EXPECTED_CONDITIONS = {"CLEAN", "COMMAND_OPEN_ORACLE", "RAND_T10", "TRUE_T10", "RANDOM_TIME_T10"}
REQUIRED_RUN_FIELDS = (
    "parent_id", "condition", "run_index", "k_requested", "k_executed",
    "attack_start_step", "attack_end_step", "checkpoint_sha256",
    "detector_triggered", "arm_contact", "video_path", "telemetry_path",
)


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-job-matrix", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index", type=Path, required=True)
    ap.add_argument("--pilot-video-index", type=Path, required=True)
    ap.add_argument("--pilot-parent-manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    job_matrix = load_strict_json(args.pilot_job_matrix, "JOB_MATRIX")
    run_ledger = load_strict_json(args.pilot_run_ledger, "RUN_LEDGER")
    telemetry = load_strict_json(args.pilot_telemetry_index, "TELEMETRY")
    video = load_strict_json(args.pilot_video_index, "VIDEO")
    parents = load_strict_json(args.pilot_parent_manifest, "PARENTS")

    errors: list[str] = []
    parent_ids = {p["parent_id"] for p in parents.get("parents", [])}
    expected_runs = len(parent_ids) * len(EXPECTED_CONDITIONS)

    runs = run_ledger.get("runs", run_ledger.get("entries", []))
    if not isinstance(runs, list):
        errors.append("RUN_LEDGER_NOT_LIST")
        runs = []

    # Build run index
    run_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        pid = run.get("parent_id", "")
        cond = run.get("condition", "")
        run_index.setdefault((pid, cond), []).append(run)

    telemetry_paths = set(telemetry.get("entries", telemetry.get("paths", [])))
    video_paths = set(video.get("entries", video.get("paths", [])))
    invalid_runs: list[dict[str, Any]] = []
    budget_parity: list[dict[str, Any]] = []

    for pid in sorted(parent_ids):
        for cond in sorted(EXPECTED_CONDITIONS):
            key = (pid, cond)
            cond_runs = run_index.get(key, [])

            if not cond_runs:
                errors.append(f"MISSING_RUNS: {pid}/{cond}")
                continue

            for run in cond_runs:
                # K enforcement
                k_req = run.get("k_requested", 0)
                k_exec = run.get("k_executed", 0)
                if k_req != 10:
                    errors.append(f"K_REQUESTED_NOT_10: {pid}/{cond} k_req={k_req}")
                if k_exec != 10:
                    errors.append(f"K_EXECUTED_NOT_10: {pid}/{cond} k_exec={k_exec}")
                    invalid_runs.append({"parent_id": pid, "condition": cond,
                                         "reason": f"k_executed={k_exec}"})

                # Condition check
                if cond not in EXPECTED_CONDITIONS:
                    errors.append(f"UNKNOWN_CONDITION: {pid}/{cond}")

                # Required fields
                for fld in REQUIRED_RUN_FIELDS:
                    if fld not in run:
                        errors.append(f"MISSING_{fld}: {pid}/{cond}")

                # Video/telemetry
                vp = run.get("video_path", "")
                if vp and vp not in video_paths:
                    errors.append(f"VIDEO_MISSING: {pid}/{cond} {vp}")
                tp = run.get("telemetry_path", "")
                if tp and tp not in telemetry_paths:
                    errors.append(f"TELEMETRY_MISSING: {pid}/{cond} {tp}")

                # Arm contact
                if not run.get("arm_contact", True):
                    errors.append(f"ARM_NO_CONTACT: {pid}/{cond}")

            # Budget parity
            budget_parity.append({
                "parent_id": pid, "condition": cond, "n_runs": len(cond_runs),
                "k_requested": cond_runs[0].get("k_requested", 0) if cond_runs else 0,
                "k_executed_total": sum(r.get("k_executed", 0) for r in cond_runs),
                "valid": all(r.get("k_executed", 0) == 10 for r in cond_runs),
            })

    receipt = {
        "schema": "PILOT_EXECUTION_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_parents": len(parent_ids), "n_conditions": len(EXPECTED_CONDITIONS),
        "n_runs": len(runs), "n_expected": expected_runs,
        "n_errors": len(errors), "n_invalid_runs": len(invalid_runs),
        "errors": errors[:100],
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_EXECUTION_VALIDATION_V0.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # Budget parity CSV
    with open(staging / "PILOT_BUDGET_PARITY_V0.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "n_runs", "k_requested", "k_executed_total", "valid"])
        for bp in budget_parity:
            w.writerow([bp["parent_id"], bp["condition"], bp["n_runs"], bp["k_requested"], bp["k_executed_total"], bp["valid"]])

    # Invalid runs CSV
    with open(staging / "PILOT_INVALID_RUNS_V0.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "reason"])
        for ir in invalid_runs:
            w.writerow([ir["parent_id"], ir["condition"], ir["reason"]])

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)
    print(f"Pilot Execution Validation: {receipt['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
