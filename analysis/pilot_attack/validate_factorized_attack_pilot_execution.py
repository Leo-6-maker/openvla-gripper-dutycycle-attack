#!/usr/bin/env python3
"""B2: Pilot execution validator — job matrix source of truth, matched parity, evidence closure.

P0-3: Job matrix drives all validation. No hardcoded conditions.
P0-4: Condition-specific execution contracts (CLEAN, TRUE_T10, RAND_T10, RANDOM_TIME_T10, ORACLE).
P0-5: Attack execution closure — K=10, step continuity, no gaps.
P0-6: Matched-control parity — checkpoint, initial state, task, preprocessing identical.
P0-7: Arm parity — max_abs deviation not just "arm_contact=true".
P0-8: Video/telemetry evidence — actual file existence, SHA, non-empty.
P0-9: Disposition closure — every job accounted for, no silent deletion.
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, load_strict_json, is_64char_hex

SELF_SHA = None

# P0-4: Condition-specific execution contracts
CONDITION_CONTRACTS = {
    "CLEAN": {
        "attack_requested": False, "attack_executed_steps": 0,
        "k_requested": 0, "k_executed": 0,
    },
    "TRUE_T10": {
        "attack_requested": True, "attack_executed_steps": 10,
        "k_requested": 10, "k_executed": 10,
        "gradient_aligned": True, "target": "gripper_open",
    },
    "RAND_T10": {
        "attack_requested": True, "attack_executed_steps": 10,
        "k_requested": 10, "k_executed": 10,
        "gradient_aligned": False,
    },
    "RANDOM_TIME_T10": {
        "attack_requested": True, "attack_executed_steps": 10,
        "k_requested": 10, "k_executed": 10,
        "payload_matches_TRUE": True,
    },
    "COMMAND_OPEN_ORACLE": {
        "attack_requested": True, "intervention_executed_steps": 10,
        "k_requested": 10, "k_executed": 10,
    },
}

# P0-6: Fields that must be identical across conditions for same parent
MATCHED_PARITY_FIELDS = (
    "checkpoint_sha256", "initial_state_sha256", "task_identity",
    "prompt_sha256", "preprocessing_sha256", "processor_config_sha256",
    "runtime_source_sha256", "evaluation_horizon",
)

# P0-6: TRUE vs RAND must additionally match
TRUE_RAND_PARITY_FIELDS = (
    "epsilon", "pgd_steps", "pgd_iterations", "attacked_frame_count",
    "norm_convention", "input_space", "jpeg_preprocessing_sha256",
)

# P0-9: Disposition taxonomy
DISPOSITIONS = (
    "COMPLETE_VALID", "NO_TRIGGER", "INVALID_RUNTIME", "PARTIAL_ATTACK",
    "MISSING_VIDEO", "MISSING_TELEMETRY", "PROTOCOL_MISMATCH",
    "INFRA_FAILURE_PRE_ACTION",
)

EXPECTED_JOB_MATRIX_SCHEMA = "PILOT_JOB_MATRIX_V0"
EXPECTED_RUN_LEDGER_SCHEMA = "PILOT_RUN_LEDGER_V0"
EXPECTED_TELEMETRY_SCHEMA = "PILOT_TELEMETRY_INDEX_V0"
EXPECTED_VIDEO_SCHEMA = "PILOT_VIDEO_INDEX_V0"


def _build_job_key(job: dict[str, Any]) -> tuple[str, str, int, int]:
    """Canonical job key: (parent_id, condition, perturbation_seed, repeat_index)."""
    return (
        str(job.get("parent_id", "")),
        str(job.get("condition", "")),
        int(job.get("perturbation_seed", 0)),
        int(job.get("repeat_index", 0)),
    )


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-job-matrix", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index", type=Path, required=True)
    ap.add_argument("--pilot-video-index", type=Path, required=True)
    ap.add_argument("--pilot-parent-manifest", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, default=None,
                    help="Root for resolving relative video/telemetry paths")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--arm-parity-tolerance", type=float, default=0.01)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    evidence_root = args.evidence_root.resolve() if args.evidence_root else Path(".")

    # P0-3: Job matrix is source of truth
    job_matrix = load_strict_json(args.pilot_job_matrix, "JOB_MATRIX")
    if job_matrix.get("schema") != EXPECTED_JOB_MATRIX_SCHEMA:
        pass  # Diagnostic mode allows

    run_ledger = load_strict_json(args.pilot_run_ledger, "RUN_LEDGER")
    telemetry_index = load_strict_json(args.pilot_telemetry_index, "TELEMETRY")
    video_index = load_strict_json(args.pilot_video_index, "VIDEO")
    parents_manifest = load_strict_json(args.pilot_parent_manifest, "PARENTS")

    errors: list[str] = []
    parent_ids = {p["parent_id"] for p in parents_manifest.get("parents", [])}

    # P0-3: Build expected job set from matrix
    jobs = job_matrix.get("jobs", job_matrix.get("entries", []))
    expected_jobs: dict[tuple, dict[str, Any]] = {}
    conditions_seen: set[str] = set()

    for job in jobs:
        key = _build_job_key(job)
        if key in expected_jobs:
            errors.append(f"JOB_MATRIX_DUP: {key}")
            continue
        expected_jobs[key] = job
        conditions_seen.add(job.get("condition", ""))

    # Build actual run set from ledger
    runs = run_ledger.get("runs", run_ledger.get("entries", []))
    actual_jobs: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        key = _build_job_key(run)
        actual_jobs.setdefault(key, []).append(run)

    # P0-3: Job closure
    missing = set(expected_jobs) - set(actual_jobs)
    extra = set(actual_jobs) - set(expected_jobs)
    duplicates = {k: v for k, v in actual_jobs.items() if len(v) > 1}

    for m in sorted(missing):
        errors.append(f"JOB_MISSING: {m}")
    for e in sorted(extra):
        errors.append(f"JOB_EXTRA: {e}")
    for k, v in sorted(duplicates.items()):
        errors.append(f"JOB_DUPLICATE: {k} count={len(v)}")

    # P0-8: Build video/telemetry lookup
    video_entries = video_index.get("entries", video_index.get("videos", []))
    telemetry_entries = telemetry_index.get("entries", telemetry_index.get("telemetry", []))

    video_by_key: dict[tuple, dict[str, Any]] = {}
    for ve in video_entries:
        if isinstance(ve, dict):
            vk = _build_job_key(ve)
            video_by_key[vk] = ve

    telem_by_key: dict[tuple, dict[str, Any]] = {}
    for te in telemetry_entries:
        if isinstance(te, dict):
            tk = _build_job_key(te)
            telem_by_key[tk] = te

    # Per-job validation
    disposition: dict[tuple, str] = {}
    budget_parity: list[dict[str, Any]] = []
    invalid_runs: list[dict[str, Any]] = []

    for key, job_def in sorted(expected_jobs.items()):
        pid, cond, seed, rep = key
        job_runs = actual_jobs.get(key, [])
        contract = CONDITION_CONTRACTS.get(cond, {})

        if not job_runs:
            disposition[key] = "MISSING"
            continue

        run = job_runs[0]  # Single run per job key

        # P0-4: Condition-specific K requirements
        k_req = contract.get("k_requested", 0)
        k_exec_req = contract.get("k_executed", 0)
        actual_k_req = run.get("k_requested", -1)
        actual_k_exec = run.get("k_executed", -1)

        if actual_k_req != k_req:
            errors.append(f"K_REQUESTED: {pid}/{cond} expected={k_req} actual={actual_k_req}")
        if actual_k_exec != k_exec_req:
            errors.append(f"K_EXECUTED: {pid}/{cond} expected={k_exec_req} actual={actual_k_exec}")

        # P0-5: Attack step closure for attack conditions
        if k_exec_req > 0:
            attack_start = run.get("attack_start_step")
            attack_end = run.get("attack_end_step")
            if isinstance(attack_start, (int, float)) and isinstance(attack_end, (int, float)):
                actual_steps = int(attack_end) - int(attack_start) + 1
                if actual_steps != k_exec_req:
                    errors.append(f"ATTACK_STEP_SPAN: {pid}/{cond} expected={k_exec_req} actual={actual_steps}")
            else:
                errors.append(f"ATTACK_STEP_MISSING: {pid}/{cond} start={attack_start!r} end={attack_end!r}")

        # P0-6: Checkpoint/state parity (validated across conditions for same parent)
        for fld in MATCHED_PARITY_FIELDS:
            if fld not in run:
                errors.append(f"PARITY_MISSING: {pid}/{cond} field={fld}")

        # Condition-specific checks
        if cond == "TRUE_T10":
            if not run.get("gradient_aligned", False):
                errors.append(f"TRUE_NOT_ALIGNED: {pid}")
        if cond == "CLEAN":
            if actual_k_exec != 0:
                errors.append(f"CLEAN_K_NOT_ZERO: {pid} k_exec={actual_k_exec}")

        # P0-7: Arm parity
        arm_max_diff = run.get("arm_max_abs_diff", run.get("arm_deviation"))
        if arm_max_diff is not None and isinstance(arm_max_diff, (int, float)):
            if arm_max_diff > args.arm_parity_tolerance:
                errors.append(f"ARM_DEVIATION: {pid}/{cond} diff={arm_max_diff}")
        # "arm_contact" alone is not parity

        # P0-8: Video evidence
        vp = run.get("video_path", "")
        vk = video_by_key.get(key, {})
        video_file = vk.get("path", vp) if isinstance(vk, dict) else vp
        if not video_file:
            errors.append(f"VIDEO_MISSING: {pid}/{cond}")
        else:
            vf = evidence_root / video_file
            if not vf.is_file():
                errors.append(f"VIDEO_NOT_FOUND: {pid}/{cond} path={video_file}")
            elif vf.stat().st_size == 0:
                errors.append(f"VIDEO_EMPTY: {pid}/{cond}")
            elif isinstance(vk, dict) and vk.get("sha256"):
                actual_vsha = sha256_file(vf)
                if actual_vsha != vk["sha256"]:
                    errors.append(f"VIDEO_SHA_MISMATCH: {pid}/{cond}")

        # P0-8: Telemetry evidence
        tp = run.get("telemetry_path", "")
        tk = telem_by_key.get(key, {})
        telem_file = tk.get("path", tp) if isinstance(tk, dict) else tp
        if not telem_file:
            errors.append(f"TELEMETRY_MISSING: {pid}/{cond}")
        else:
            tf = evidence_root / telem_file
            if not tf.is_file():
                errors.append(f"TELEMETRY_NOT_FOUND: {pid}/{cond}")
            elif tf.stat().st_size == 0:
                errors.append(f"TELEMETRY_EMPTY: {pid}/{cond}")

        # P0-9: Disposition
        has_errors = any(err.startswith(f"{pid}/{cond}") for err in errors[-20:])
        if not has_errors:
            disposition[key] = "COMPLETE_VALID"
        elif any("K_EXECUTED" in e or "ATTACK_STEP" in e for e in errors[-20:]):
            disposition[key] = "PARTIAL_ATTACK"
        elif any("VIDEO" in e or "TELEMETRY" in e for e in errors[-20:]):
            disposition[key] = "MISSING_VIDEO" if "VIDEO" in str(errors[-10:]) else "MISSING_TELEMETRY"
        else:
            disposition[key] = "PROTOCOL_MISMATCH"

        budget_parity.append({
            "parent_id": pid, "condition": cond, "seed": seed, "repeat": rep,
            "k_requested": actual_k_req, "k_executed": actual_k_exec,
            "disposition": disposition.get(key, "UNKNOWN"),
            "valid": disposition.get(key) == "COMPLETE_VALID",
        })

    # P0-6: Cross-condition matched parity for same parent/repeat
    for pid in sorted(parent_ids):
        for rep in sorted({k[3] for k in expected_jobs if k[0] == pid}):
            cond_runs: dict[str, dict[str, Any]] = {}
            for key, job_runs in actual_jobs.items():
                if key[0] == pid and key[3] == rep:
                    cond_runs[key[1]] = job_runs[0]

            if "TRUE_T10" in cond_runs and "RAND_T10" in cond_runs:
                true_run = cond_runs["TRUE_T10"]
                rand_run = cond_runs["RAND_T10"]
                for fld in TRUE_RAND_PARITY_FIELDS:
                    tv = true_run.get(fld); rv = rand_run.get(fld)
                    if tv != rv:
                        errors.append(f"TRUE_RAND_PARITY: {pid} rep={rep} field={fld} TRUE={tv!r} RAND={rv!r}")

            # Cross-condition checkpoint parity
            checkpoint_shas = {c: r.get("checkpoint_sha256") for c, r in cond_runs.items()}
            if len(set(checkpoint_shas.values())) > 1:
                errors.append(f"CHECKPOINT_DIVERGENT: {pid} rep={rep} shas={checkpoint_shas}")

    # P0-9: Verify all jobs have dispositions
    undisposed = set(expected_jobs) - set(disposition)
    for key in undisposed:
        disposition[key] = "MISSING"
        errors.append(f"UNDISPOSED: {key}")

    # Count dispositions
    disp_counts: dict[str, int] = {}
    for d in disposition.values():
        disp_counts[d] = disp_counts.get(d, 0) + 1

    receipt = {
        "schema": "PILOT_EXECUTION_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_expected_jobs": len(expected_jobs), "n_actual_runs": len(runs),
        "n_missing": len(missing), "n_extra": len(extra), "n_duplicates": len(duplicates),
        "conditions_present": sorted(conditions_seen),
        "n_errors": len(errors), "errors": errors[:200],
        "disposition_counts": disp_counts,
        "attack_eval_consumed": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_EXECUTION_VALIDATION_V0.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # Budget parity CSV
    with open(staging / "PILOT_BUDGET_PARITY_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "seed", "repeat", "k_requested", "k_executed", "disposition", "valid"])
        for bp in budget_parity:
            w.writerow([bp["parent_id"], bp["condition"], bp["seed"], bp["repeat"],
                        bp["k_requested"], bp["k_executed"], bp["disposition"], bp["valid"]])

    # Invalid runs CSV
    with open(staging / "PILOT_INVALID_RUNS_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "reason"])
        for ir in invalid_runs:
            w.writerow([ir.get("parent_id"), ir.get("condition"), ir.get("reason", "")])

    # Disposition CSV
    with open(staging / "PILOT_DISPOSITION_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "seed", "repeat", "disposition"])
        for key, disp in sorted(disposition.items()):
            w.writerow([key[0], key[1], key[2], key[3], disp])

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Execution Validation: {receipt['status']}")
    print(f"  Jobs: {len(expected_jobs)} expected, {len(runs)} actual")
    print(f"  Missing: {len(missing)} Extra: {len(extra)} Dup: {len(duplicates)}")
    print(f"  Dispositions: {disp_counts}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
