#!/usr/bin/env python3
"""B2 v2.2: Pilot execution validator — sealed roots, strict keys, matched parity, evidence closure."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import (
    sha256_file, is_64char_hex, is_strict_int, load_strict_json,
    require_schema, require_nonempty_list, consume_sealed_root,
    guard_path_safe, verify_evidence_file, seal_output_dir,
)

SELF_SHA = None
EXPECTED_JOB_SCHEMA = "PILOT_JOB_MATRIX_V0"
EXPECTED_LEDGER_SCHEMA = "PILOT_RUN_LEDGER_V0"
EXPECTED_TELEMETRY_SCHEMA = "PILOT_TELEMETRY_INDEX_V0"
EXPECTED_VIDEO_SCHEMA = "PILOT_VIDEO_INDEX_V0"
EXPECTED_ARM_PROTO_SCHEMA = "PILOT_ARM_PARITY_PROTOCOL_V0"

CONDITION_CONTRACTS = {
    "CLEAN": {"attack_requested": False, "k_requested": 0, "k_executed": 0},
    "TRUE_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": True},
    "RAND_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": False},
    "RANDOM_TIME_T10": {"attack_requested": True, "k_requested": 10, "k_executed": 10, "payload_matches_TRUE": True},
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

TRUE_RANDOM_TIME_PARITY_FIELDS = (
    "epsilon", "pgd_steps", "pgd_iterations", "attacked_frame_count",
    "norm_convention", "input_space", "jpeg_preprocessing_sha256",
    "payload_config_sha256",
)


def _build_job_key(job: dict[str, Any], label: str) -> tuple:
    """Fix 2: Strict job key — hard-fail on invalid, no silent continue."""
    pid = job.get("parent_id", "")
    cond = job.get("condition", "")
    seed = job.get("perturbation_seed")
    rep = job.get("repeat_index")
    mgid = job.get("matched_group_id")
    if not isinstance(pid, str) or not pid:
        raise SystemExit(f"{label}_JOB_KEY_PID: {pid!r}")
    if not isinstance(cond, str) or not cond:
        raise SystemExit(f"{label}_JOB_KEY_COND: {cond!r}")
    if not is_strict_int(seed):
        raise SystemExit(f"{label}_JOB_KEY_SEED: {seed!r}")
    if not is_strict_int(rep):
        raise SystemExit(f"{label}_JOB_KEY_REP: {rep!r}")
    # matched_group_id is optional but must be valid if present
    if mgid is not None and not isinstance(mgid, str):
        raise SystemExit(f"{label}_JOB_KEY_MGID: {mgid!r}")
    return (str(pid), str(cond), int(seed), int(rep))


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    # Fix 1: All inputs are sealed roots
    ap.add_argument("--pilot-job-matrix-root", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger-root", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index-root", type=Path, required=True)
    ap.add_argument("--pilot-video-index-root", type=Path, required=True)
    ap.add_argument("--pilot-parent-manifest-root", type=Path, required=True)
    # Fix 6: Arm protocol sealed and mandatory
    ap.add_argument("--pilot-arm-parity-protocol-root", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    evidence_root = args.evidence_root.resolve()

    # Fix 1: Consume all inputs as sealed roots
    job_matrix, _ = consume_sealed_root(args.pilot_job_matrix_root, EXPECTED_JOB_SCHEMA, "JOB_MATRIX")
    run_ledger, _ = consume_sealed_root(args.pilot_run_ledger_root, EXPECTED_LEDGER_SCHEMA, "RUN_LEDGER")
    telemetry_index, _ = consume_sealed_root(args.pilot_telemetry_index_root, EXPECTED_TELEMETRY_SCHEMA, "TELEMETRY")
    video_index, _ = consume_sealed_root(args.pilot_video_index_root, EXPECTED_VIDEO_SCHEMA, "VIDEO")
    parents_data, _ = consume_sealed_root(args.pilot_parent_manifest_root, "PILOT_PARENT_MANIFEST_V0", "PARENTS")

    # Fix 6: Arm protocol sealed and mandatory
    arm_proto, _ = consume_sealed_root(args.pilot_arm_parity_protocol_root, EXPECTED_ARM_PROTO_SCHEMA, "ARM_PROTO")
    arm_tolerance = arm_proto.get("max_abs_tolerance")
    if not isinstance(arm_tolerance, (int, float)) or isinstance(arm_tolerance, bool):
        raise SystemExit(f"ARM_PROTO_TOLERANCE_INVALID: {arm_tolerance!r}")
    arm_tolerance = float(arm_tolerance)

    parent_ids_by_manifest = {p["parent_id"] for p in parents_data.get("parents", [])}

    # Fix 3: Non-empty enforcement
    jobs = require_nonempty_list(job_matrix.get("jobs", []), "JOB_MATRIX_JOBS")
    runs = require_nonempty_list(run_ledger.get("runs", []), "RUN_LEDGER_RUNS")
    telem_entries = require_nonempty_list(telemetry_index.get("entries", []), "TELEMETRY_ENTRIES")
    video_entries = require_nonempty_list(video_index.get("entries", []), "VIDEO_ENTRIES")

    # Collect allowed conditions from matrix
    allowed_conditions: set[str] = set()
    for job in jobs:
        cond = job.get("condition", "")
        if cond not in CONDITION_CONTRACTS:
            raise SystemExit(f"JOB_MATRIX_UNKNOWN_CONDITION: {cond}")
        allowed_conditions.add(cond)

    # Fix 2: Build expected jobs — hard-fail on any invalid key
    expected_jobs: dict[tuple, dict[str, Any]] = {}
    for job in jobs:
        key = _build_job_key(job, "JOB_MATRIX")
        if key in expected_jobs:
            raise SystemExit(f"JOB_MATRIX_DUP_KEY: {key}")
        expected_jobs[key] = job

    # Build actual runs — hard-fail on any invalid key
    actual_jobs: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        key = _build_job_key(run, "RUN_LEDGER")
        actual_jobs.setdefault(key, []).append(run)

    # Fix 3: Video/telemetry index closure — hard-fail on invalid keys
    video_by_key: dict[tuple, dict[str, Any]] = {}
    for ve in video_entries:
        if not isinstance(ve, dict):
            raise SystemExit(f"VIDEO_INDEX_ENTRY_NOT_OBJECT: {ve!r}")
        vk = _build_job_key(ve, "VIDEO_INDEX")
        if vk in video_by_key:
            raise SystemExit(f"VIDEO_INDEX_DUP_KEY: {vk}")
        video_by_key[vk] = ve

    telem_by_key: dict[tuple, dict[str, Any]] = {}
    for te in telem_entries:
        if not isinstance(te, dict):
            raise SystemExit(f"TELEMETRY_INDEX_ENTRY_NOT_OBJECT: {te!r}")
        tk = _build_job_key(te, "TELEMETRY_INDEX")
        if tk in telem_by_key:
            raise SystemExit(f"TELEMETRY_INDEX_DUP_KEY: {tk}")
        telem_by_key[tk] = te

    # Job closure
    missing = set(expected_jobs) - set(actual_jobs)
    extra = set(actual_jobs) - set(expected_jobs)
    duplicates = {k: v for k, v in actual_jobs.items() if len(v) > 1}

    # Fix 8: Per-job error tracking
    job_errors: dict[tuple, list[str]] = {k: [] for k in expected_jobs}
    disposition: dict[tuple, str] = {}
    budget_parity: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for key in sorted(missing):
        err = f"JOB_MISSING: {key}"
        job_errors[key] = [err]; all_errors.append(err); disposition[key] = "MISSING"

    for key in sorted(extra):
        err = f"JOB_EXTRA: {key}"
        job_errors[key] = [err]; all_errors.append(err); disposition[key] = "EXTRA"

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
        if not is_strict_int(actual_k_req) or actual_k_req != k_req_expected:
            je.append(f"K_REQUESTED: expected={k_req_expected} actual={actual_k_req!r}")
        if not is_strict_int(actual_k_exec) or actual_k_exec != k_exec_expected:
            je.append(f"K_EXECUTED: expected={k_exec_expected} actual={actual_k_exec!r}")

        # Attack step closure
        if k_exec_expected > 0:
            astart = run.get("attack_start_step"); aend = run.get("attack_end_step")
            if not is_strict_int(astart) or not is_strict_int(aend):
                je.append(f"ATTACK_STEP_TYPE: start={astart!r} end={aend!r}")
            else:
                span = int(aend) - int(astart) + 1
                if span != k_exec_expected:
                    je.append(f"ATTACK_STEP_SPAN: expected={k_exec_expected} actual={span}")

            attack_ledger = run.get("attack_step_ledger", run.get("attack_steps", []))
            if isinstance(attack_ledger, list) and len(attack_ledger) == k_exec_expected:
                steps = [s.get("step", s) if isinstance(s, dict) else s for s in attack_ledger]
                if not all(is_strict_int(s) for s in steps):
                    je.append("ATTACK_LEDGER_STEP_TYPE")
                else:
                    sorted_steps = sorted(int(s) for s in steps)
                    for i, s in enumerate(sorted_steps):
                        if s != int(astart) + i:
                            je.append(f"ATTACK_LEDGER_GAP: expected={int(astart)+i} actual={s}")
                            break
            elif k_exec_expected > 0:
                je.append("ATTACK_LEDGER_MISSING_OR_WRONG_COUNT")

        # Condition-specific
        if cond == "CLEAN" and actual_k_exec != 0:
            je.append(f"CLEAN_K_NOT_ZERO")
        if cond == "TRUE_T10" and not run.get("gradient_aligned", False):
            je.append("TRUE_NOT_GRADIENT_ALIGNED")

        # Fix 5: All matched parity fields present and non-null
        for fld in MATCHED_PARITY_FIELDS:
            if fld not in run or run.get(fld) is None:
                je.append(f"PARITY_MISSING: {fld}")

        # Fix 6: Arm parity mandatory for attack conditions
        if k_exec_expected > 0 and cond != "CLEAN":
            arm_diff = run.get("arm_max_abs_diff")
            if arm_diff is None:
                je.append("ARM_PARITY_MISSING")
            elif isinstance(arm_diff, (int, float)) and not isinstance(arm_diff, bool):
                if float(arm_diff) > arm_tolerance:
                    je.append(f"ARM_DEVIATION: diff={arm_diff} tolerance={arm_tolerance}")
            else:
                je.append(f"ARM_PARITY_TYPE: {arm_diff!r}")

        # Fix 7: Evidence closure — SHA mandatory
        vp = run.get("video_path", "")
        vk = video_by_key.get(key)
        if not vk:
            je.append(f"VIDEO_INDEX_MISSING_KEY: {key}")
        elif not vp:
            je.append("VIDEO_PATH_EMPTY_IN_RUN")
        else:
            index_path = vk.get("path", "")
            if index_path != vp:
                je.append(f"VIDEO_PATH_MISMATCH: run={vp} index={index_path}")
            vsha = vk.get("sha256", "")
            try:
                verify_evidence_file(evidence_root, vp, vsha, f"VIDEO_{pid}_{cond}")
            except SystemExit as e:
                je.append(f"VIDEO: {e}")

        tp = run.get("telemetry_path", "")
        tk = telem_by_key.get(key)
        if not tk:
            je.append(f"TELEMETRY_INDEX_MISSING_KEY: {key}")
        elif not tp:
            je.append("TELEMETRY_PATH_EMPTY_IN_RUN")
        else:
            index_path = tk.get("path", "")
            if index_path != tp:
                je.append(f"TELEMETRY_PATH_MISMATCH: run={tp} index={index_path}")
            tsha = tk.get("sha256", "")
            try:
                verify_evidence_file(evidence_root, tp, tsha, f"TELEM_{pid}_{cond}")
            except SystemExit as e:
                je.append(f"TELEMETRY: {e}")

        # Per-job disposition
        if not je:
            disposition[key] = "COMPLETE_VALID"
        elif any("K_" in e or "ATTACK_" in e for e in je):
            disposition[key] = "PARTIAL_ATTACK"
        elif any("VIDEO" in e for e in je):
            disposition[key] = "MISSING_VIDEO"
        elif any("TELEMETRY" in e for e in je):
            disposition[key] = "MISSING_TELEMETRY"
        elif any("ARM_" in e for e in je):
            disposition[key] = "ARM_PARITY_FAIL"
        else:
            disposition[key] = "PROTOCOL_MISMATCH"

        all_errors.extend(je)
        budget_parity.append({
            "parent_id": pid, "condition": cond, "seed": seed, "repeat": rep,
            "k_requested": actual_k_req, "k_executed": actual_k_exec,
            "disposition": disposition.get(key, "UNKNOWN"), "valid": not je,
        })

    # Fix 5 + Fix 16: Cross-condition matched parity, errors write back to jobs
    for pid in sorted(parent_ids_by_manifest):
        for rep in sorted({k[3] for k in expected_jobs if k[0] == pid}):
            cond_runs: dict[str, tuple[tuple, dict[str, Any]]] = {}
            for key, job_runs in actual_jobs.items():
                if key[0] == pid and key[3] == rep:
                    cond_runs[key[1]] = (key, job_runs[0])

            affected_keys: set[tuple] = set()

            for fld in MATCHED_PARITY_FIELDS:
                values_by_cond = {c: r.get(fld) for c, (_, r) in cond_runs.items() if fld in r}
                if len(set(values_by_cond.values())) > 1:
                    err = f"PARITY_DIVERGENT: {pid} rep={rep} field={fld} values={values_by_cond}"
                    all_errors.append(err)
                    affected_keys.update(k for _, (k, _) in cond_runs.items())

            if "TRUE_T10" in cond_runs and "RAND_T10" in cond_runs:
                true_key, true_run = cond_runs["TRUE_T10"]
                rand_key, rand_run = cond_runs["RAND_T10"]
                for fld in TRUE_RAND_PARITY_FIELDS:
                    tv = true_run.get(fld); rv = rand_run.get(fld)
                    if tv is None or rv is None:
                        err = f"TRUE_RAND_PARITY_MISSING: {pid} field={fld} TRUE={tv!r} RAND={rv!r}"
                        all_errors.append(err); affected_keys.update([true_key, rand_key])
                    elif tv != rv:
                        err = f"TRUE_RAND_PARITY: {pid} field={fld} TRUE={tv!r} RAND={rv!r}"
                        all_errors.append(err); affected_keys.update([true_key, rand_key])

            # Fix 5: TRUE vs RANDOM_TIME parity
            if "TRUE_T10" in cond_runs and "RANDOM_TIME_T10" in cond_runs:
                true_key, true_run = cond_runs["TRUE_T10"]
                rt_key, rt_run = cond_runs["RANDOM_TIME_T10"]
                for fld in TRUE_RANDOM_TIME_PARITY_FIELDS:
                    tv = true_run.get(fld); rv = rt_run.get(fld)
                    if tv is None or rv is None:
                        err = f"TRUE_RT_PARITY_MISSING: {pid} field={fld}"
                        all_errors.append(err); affected_keys.update([true_key, rt_key])
                    elif tv != rv:
                        err = f"TRUE_RT_PARITY: {pid} field={fld} TRUE={tv!r} RT={rv!r}"
                        all_errors.append(err); affected_keys.update([true_key, rt_key])

            checkpoints = {c: r.get("checkpoint_sha256") for c, (_, r) in cond_runs.items()}
            if len(set(checkpoints.values())) > 1:
                err = f"CHECKPOINT_DIVERGENT: {pid} rep={rep} shas={checkpoints}"
                all_errors.append(err)
                affected_keys.update(k for _, (k, _) in cond_runs.items())

            # Fix 16: Write parity errors back to job dispositions
            for ak in affected_keys:
                if ak in job_errors and disposition.get(ak) == "COMPLETE_VALID":
                    disposition[ak] = "PARITY_MISMATCH"

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
        "arm_tolerance": arm_tolerance,
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
        for bp in budget_parity: w.writerow([bp[k] for k in ["parent_id","condition","seed","repeat","k_requested","k_executed","disposition","valid"]])

    with open(staging / "PILOT_DISPOSITION_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "condition", "seed", "repeat", "disposition"])
        for key, disp in sorted(disposition.items()): w.writerow([key[0], key[1], key[2], key[3], disp])

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Execution Validation: {receipt['status']} errors={len(all_errors)}")
    print(f"  Dispositions: {disp_counts}")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
