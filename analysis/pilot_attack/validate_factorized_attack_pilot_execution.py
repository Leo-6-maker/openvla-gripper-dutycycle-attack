#!/usr/bin/env python3
"""B2: Pilot execution validator — strict job key, matched parity, evidence closure, per-job disposition."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import (
    sha256_file, load_strict_json, is_64char_hex, is_strict_int,
    require_schema, guard_path_safe, verify_evidence_file,
)

SELF_SHA = None
EXPECTED_JOB_SCHEMA = "PILOT_JOB_MATRIX_V0"
EXPECTED_LEDGER_SCHEMA = "PILOT_RUN_LEDGER_V0"
EXPECTED_TELEMETRY_SCHEMA = "PILOT_TELEMETRY_INDEX_V0"
EXPECTED_VIDEO_SCHEMA = "PILOT_VIDEO_INDEX_V0"

CONDITION_CONTRACTS = {
    "CLEAN": {"attack_requested": False, "k_requested": 0, "k_executed": 0},
    "TRUE_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": True},
    "RAND_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": False},
    "RANDOM_TIME_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10},
    "COMMAND_OPEN_ORACLE": {"attack_requested": True, "k_requested": 10, "k_executed": 10},
}

MATCHED_PARITY_FIELDS = (
    "checkpoint_sha256", "initial_state_sha256", "task_identity",
    "prompt_sha256", "preprocessing_sha256", "processor_config_sha256",
    "runtime_source_sha256", "evaluation_horizon",
)

TRUE_RAND_PARITY_FIELDS = (
    "epsilon", "pgd_steps", "pgd_iterations", "attacked_frame_count",
    "norm_convention", "input_space", "jpeg_preprocessing_sha256",
)


def _build_job_key(job: dict[str, Any], label: str = "") -> tuple:
    """Fix 2: Strict job key — reject bool-as-int, missing, non-integer."""
    pid = job.get("parent_id", "")
    cond = job.get("condition", "")
    seed = job.get("perturbation_seed")
    rep = job.get("repeat_index")
    if not pid or not cond:
        raise SystemExit(f"{label}_JOB_KEY_MISSING: pid={pid!r} cond={cond!r}")
    if not is_strict_int(seed):
        raise SystemExit(f"{label}_JOB_SEED_INVALID: {seed!r}")
    if not is_strict_int(rep):
        raise SystemExit(f"{label}_JOB_REPEAT_INVALID: {rep!r}")
    return (str(pid), str(cond), int(seed), int(rep))


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-job-matrix", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index", type=Path, required=True)
    ap.add_argument("--pilot-video-index", type=Path, required=True)
    ap.add_argument("--pilot-parent-manifest", type=Path, required=True)
    ap.add_argument("--pilot-arm-parity-protocol", type=Path, default=None)
    ap.add_argument("--evidence-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    evidence_root = args.evidence_root.resolve() if args.evidence_root else Path(".")

    # Fix 1: Schema hard-reject
    job_matrix = load_strict_json(args.pilot_job_matrix, "JOB_MATRIX")
    require_schema(job_matrix, EXPECTED_JOB_SCHEMA, "JOB_MATRIX")
    run_ledger = load_strict_json(args.pilot_run_ledger, "RUN_LEDGER")
    require_schema(run_ledger, EXPECTED_LEDGER_SCHEMA, "RUN_LEDGER")
    telemetry_index = load_strict_json(args.pilot_telemetry_index, "TELEMETRY")
    require_schema(telemetry_index, EXPECTED_TELEMETRY_SCHEMA, "TELEMETRY")
    video_index = load_strict_json(args.pilot_video_index, "VIDEO")
    require_schema(video_index, EXPECTED_VIDEO_SCHEMA, "VIDEO")
    parents_manifest = load_strict_json(args.pilot_parent_manifest, "PARENTS")

    # Fix 7: Arm parity tolerance from sealed protocol
    arm_tolerance = 0.01  # default
    if args.pilot_arm_parity_protocol and args.pilot_arm_parity_protocol.is_file():
        arm_proto = load_strict_json(args.pilot_arm_parity_protocol, "ARM_PROTO")
        arm_tolerance = float(arm_proto.get("max_abs_tolerance", 0.01))

    parent_ids_by_manifest = {p["parent_id"] for p in parents_manifest.get("parents", [])}

    # Fix 3: Conditions from job matrix, not hardcoded
    allowed_conditions: set[str] = set()
    for job in job_matrix.get("jobs", []):
        cond = job.get("condition", "")
        if cond in CONDITION_CONTRACTS:
            allowed_conditions.add(cond)

    # Build expected job set
    expected_jobs: dict[tuple, dict[str, Any]] = {}
    for job in job_matrix.get("jobs", []):
        try:
            key = _build_job_key(job, "JOB_MATRIX")
        except SystemExit as e:
            # Will be caught below as errors
            continue
        if key in expected_jobs:
            raise SystemExit(f"JOB_MATRIX_DUP_KEY: {key}")
        cond = job.get("condition", "")
        # Fix 3: Reject unknown conditions
        if cond not in CONDITION_CONTRACTS:
            raise SystemExit(f"JOB_MATRIX_UNKNOWN_CONDITION: {cond} key={key}")
        expected_jobs[key] = job

    # Build actual run set
    runs = run_ledger.get("runs", [])
    actual_jobs: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        try:
            key = _build_job_key(run, "RUN_LEDGER")
        except SystemExit:
            continue
        actual_jobs.setdefault(key, []).append(run)

    # Fix 3 + job closure
    missing = set(expected_jobs) - set(actual_jobs)
    extra = set(actual_jobs) - set(expected_jobs)
    duplicates = {k: v for k, v in actual_jobs.items() if len(v) > 1}

    # Build video/telemetry lookup (Fix 7: reject duplicate index keys)
    video_by_key: dict[tuple, dict[str, Any]] = {}
    for ve in video_index.get("entries", []):
        if not isinstance(ve, dict): continue
        try:
            vk = _build_job_key(ve, "VIDEO_INDEX")
        except SystemExit:
            continue
        if vk in video_by_key:
            raise SystemExit(f"VIDEO_INDEX_DUP: {vk}")
        video_by_key[vk] = ve

    telem_by_key: dict[tuple, dict[str, Any]] = {}
    for te in telemetry_index.get("entries", []):
        if not isinstance(te, dict): continue
        try:
            tk = _build_job_key(te, "TELEMETRY_INDEX")
        except SystemExit:
            continue
        if tk in telem_by_key:
            raise SystemExit(f"TELEMETRY_INDEX_DUP: {tk}")
        telem_by_key[tk] = te

    # Fix 8: Per-job error tracking
    job_errors: dict[tuple, list[str]] = {k: [] for k in expected_jobs}
    disposition: dict[tuple, str] = {}
    budget_parity: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for key in sorted(missing):
        err = f"JOB_MISSING: {key}"
        job_errors[key] = [err]; all_errors.append(err)
        disposition[key] = "MISSING"

    for key in sorted(extra):
        err = f"JOB_EXTRA: {key}"
        job_errors[key] = [err]; all_errors.append(err)
        disposition[key] = "EXTRA"

    for key, run_list in sorted(duplicates.items()):
        err = f"JOB_DUPLICATE: {key} count={len(run_list)}"
        job_errors[key].append(err); all_errors.append(err)

    # Validate each expected job
    for key in sorted(expected_jobs):
        pid, cond, seed, rep = key
        je = job_errors.setdefault(key, [])
        contract = CONDITION_CONTRACTS.get(cond, {})
        job_runs = actual_jobs.get(key, [])

        if not job_runs:
            if not je: disposition[key] = "MISSING"
            continue

        run = job_runs[0]
        k_req_expected = contract.get("k_requested", 0)
        k_exec_expected = contract.get("k_executed", 0)

        actual_k_req = run.get("k_requested")
        actual_k_exec = run.get("k_executed")

        # Fix 4: K must be strict int
        if not is_strict_int(actual_k_req) or actual_k_req != k_req_expected:
            je.append(f"K_REQUESTED: expected={k_req_expected} actual={actual_k_req!r}")
        if not is_strict_int(actual_k_exec) or actual_k_exec != k_exec_expected:
            je.append(f"K_EXECUTED: expected={k_exec_expected} actual={actual_k_exec!r}")

        # Fix 4: Attack step closure
        if k_exec_expected > 0:
            astart = run.get("attack_start_step"); aend = run.get("attack_end_step")
            if not is_strict_int(astart) or not is_strict_int(aend):
                je.append(f"ATTACK_STEP_TYPE: start={astart!r} end={aend!r}")
            else:
                span = int(aend) - int(astart) + 1
                if span != k_exec_expected:
                    je.append(f"ATTACK_STEP_SPAN: expected={k_exec_expected} actual={span}")

            # Check attack step ledger
            attack_ledger = run.get("attack_step_ledger", run.get("attack_steps", []))
            if isinstance(attack_ledger, list):
                if len(attack_ledger) != k_exec_expected:
                    je.append(f"ATTACK_LEDGER_COUNT: expected={k_exec_expected} actual={len(attack_ledger)}")
                # Check step uniqueness and continuity
                steps = [s.get("step", s) if isinstance(s, dict) else s for s in attack_ledger]
                if all(is_strict_int(s) for s in steps):
                    for i, s in enumerate(sorted(steps)):
                        if s != astart + i:
                            je.append(f"ATTACK_LEDGER_GAP: expected_step={astart + i} actual={s}")
                            break
            elif k_exec_expected > 0:
                je.append("ATTACK_LEDGER_MISSING")

        # Condition-specific checks
        if cond == "CLEAN" and actual_k_exec != 0:
            je.append(f"CLEAN_K_NOT_ZERO: k_exec={actual_k_exec}")
        if cond == "TRUE_T10" and not run.get("gradient_aligned", False):
            je.append(f"TRUE_NOT_GRADIENT_ALIGNED")

        # Fix 5: All matched parity fields must be present
        for fld in MATCHED_PARITY_FIELDS:
            if fld not in run:
                je.append(f"PARITY_MISSING: {fld}")
            elif run.get(fld) is None:
                je.append(f"PARITY_NULL: {fld}")

        # Fix 7: Arm parity mandatory for attack conditions
        if k_exec_expected > 0 and cond != "CLEAN":
            arm_diff = run.get("arm_max_abs_diff")
            if arm_diff is None:
                je.append("ARM_PARITY_MISSING")
            elif isinstance(arm_diff, (int, float)) and not isinstance(arm_diff, bool):
                if arm_diff > arm_tolerance:
                    je.append(f"ARM_DEVIATION: diff={arm_diff} tolerance={arm_tolerance}")
            else:
                je.append(f"ARM_PARITY_TYPE: {arm_diff!r}")

        # Fix 7: Evidence closure
        vp = run.get("video_path", "")
        vk = video_by_key.get(key, {})
        video_file = vk.get("path", vp) if vk else vp
        try:
            verify_evidence_file(evidence_root, video_file, vk.get("sha256") if vk else None, f"VIDEO_{pid}_{cond}")
        except SystemExit as e:
            je.append(f"VIDEO: {e}")

        tp = run.get("telemetry_path", "")
        tk = telem_by_key.get(key, {})
        telem_file = tk.get("path", tp) if tk else tp
        try:
            verify_evidence_file(evidence_root, telem_file, tk.get("sha256") if tk else None, f"TELEM_{pid}_{cond}")
        except SystemExit as e:
            je.append(f"TELEMETRY: {e}")

        # Fix 8: Per-job disposition
        if not je:
            disposition[key] = "COMPLETE_VALID"
        elif any("K_" in e or "ATTACK_" in e for e in je):
            disposition[key] = "PARTIAL_ATTACK"
        elif any("VIDEO" in e for e in je):
            disposition[key] = "MISSING_VIDEO"
        elif any("TELEMETRY" in e for e in je):
            disposition[key] = "MISSING_TELEMETRY"
        else:
            disposition[key] = "PROTOCOL_MISMATCH"

        all_errors.extend(je)

        budget_parity.append({
            "parent_id": pid, "condition": cond, "seed": seed, "repeat": rep,
            "k_requested": actual_k_req, "k_executed": actual_k_exec,
            "disposition": disposition.get(key, "UNKNOWN"),
            "valid": not je,
        })

    # Fix 5: Cross-condition matched parity (same parent + repeat)
    for pid in sorted(parent_ids_by_manifest):
        for rep in sorted({k[3] for k in expected_jobs if k[0] == pid}):
            cond_runs: dict[str, dict[str, Any]] = {}
            for key, job_runs in actual_jobs.items():
                if key[0] == pid and key[3] == rep:
                    cond_runs[key[1]] = job_runs[0]

            # All matched parity fields must be equal across conditions
            for fld in MATCHED_PARITY_FIELDS:
                values_by_cond = {c: r.get(fld) for c, r in cond_runs.items() if fld in r}
                if len(set(values_by_cond.values())) > 1:
                    all_errors.append(f"PARITY_DIVERGENT: {pid} rep={rep} field={fld} values={values_by_cond}")

            # TRUE vs RAND parity
            if "TRUE_T10" in cond_runs and "RAND_T10" in cond_runs:
                for fld in TRUE_RAND_PARITY_FIELDS:
                    tv = cond_runs["TRUE_T10"].get(fld)
                    rv = cond_runs["RAND_T10"].get(fld)
                    if tv != rv:
                        all_errors.append(f"TRUE_RAND_PARITY: {pid} field={fld} TRUE={tv!r} RAND={rv!r}")

            # Checkpoint parity
            checkpoints = {c: r.get("checkpoint_sha256") for c, r in cond_runs.items()}
            if len(set(checkpoints.values())) > 1:
                all_errors.append(f"CHECKPOINT_DIVERGENT: {pid} rep={rep} shas={checkpoints}")

    # Disposition counts
    disp_counts: dict[str, int] = {}
    for d in disposition.values(): disp_counts[d] = disp_counts.get(d, 0) + 1

    receipt = {
        "schema": "PILOT_EXECUTION_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not all_errors else "HOLD",
        "n_expected_jobs": len(expected_jobs), "n_actual_runs": len(runs),
        "n_missing": len(missing), "n_extra": len(extra), "n_duplicates": len(duplicates),
        "allowed_conditions": sorted(allowed_conditions),
        "n_errors": len(all_errors), "errors": all_errors[:200],
        "disposition_counts": disp_counts,
        "attack_eval_consumed": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_EXECUTION_VALIDATION_V0.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with open(staging / "PILOT_BUDGET_PARITY_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "seed", "repeat", "k_requested", "k_executed", "disposition", "valid"])
        for bp in budget_parity:
            w.writerow([bp["parent_id"], bp["condition"], bp["seed"], bp["repeat"], bp["k_requested"], bp["k_executed"], bp["disposition"], bp["valid"]])

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
    print(f"  Jobs: {len(expected_jobs)} expected, {len(runs)} actual, missing={len(missing)} extra={len(extra)} dup={len(duplicates)}")
    print(f"  Dispositions: {disp_counts}")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
