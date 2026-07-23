#!/usr/bin/env python3
"""B2 v2.3.1: Pilot execution validator — cross-artifact binding, duplicate rejection, unified dispositions."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import (
    sha256_file, is_64char_hex, is_strict_int, is_finite_number, load_strict_json,
    require_schema, require_nonempty_list, consume_sealed_root,
    guard_path_safe, verify_evidence_file,
)

SELF_SHA = None
EXPECTED_JOB_SCHEMA = "PILOT_JOB_MATRIX_V0"
EXPECTED_LEDGER_SCHEMA = "PILOT_RUN_LEDGER_V0"
EXPECTED_TELEMETRY_SCHEMA = "PILOT_TELEMETRY_INDEX_V0"
EXPECTED_VIDEO_SCHEMA = "PILOT_VIDEO_INDEX_V0"
EXPECTED_ARM_PROTO_SCHEMA = "PILOT_ARM_PARITY_PROTOCOL_V0"

CONDITION_CONTRACTS = {
    "CLEAN":              {"attack_requested": False, "k_requested": 0, "k_executed": 0},
    "TRUE_T10":           {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": True},
    "RAND_T10":           {"attack_requested": True, "k_requested": 10, "k_executed": 10, "gradient_aligned": False},
    "RANDOM_TIME_T10":    {"attack_requested": True, "k_requested": 10, "k_executed": 10, "payload_matches_TRUE": True},
    "COMMAND_OPEN_ORACLE":{"attack_requested": True, "k_requested": 10, "k_executed": 10, "oracle_type": "command_intervention"},
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

NUMERIC_FIELDS_REQUIRE_FINITE = frozenset({
    "arm_max_abs_diff", "epsilon", "k_requested", "k_executed",
    "attack_start_step", "attack_end_step", "evaluation_horizon",
    "pgd_steps", "pgd_iterations", "attacked_frame_count",
    "perturbation_seed", "repeat_index",
})

IMMUTABLE_FIELDS = ("matched_group_id", "parent_id", "condition", "perturbation_seed", "repeat_index")


def _validate_entry_id(entry: dict[str, Any], label: str) -> tuple[str, str]:
    jid = entry.get("job_id", "")
    mgid = entry.get("matched_group_id", "")
    if not isinstance(jid, str) or not jid:
        raise SystemExit(f"{label}_MISSING_JOB_ID: {json.dumps(entry)[:120]}")
    if not isinstance(mgid, str) or not mgid:
        raise SystemExit(f"{label}_MISSING_MATCHED_GROUP_ID: jid={jid!r}")
    return jid, mgid


def _reject_non_finite(entry: dict[str, Any], label: str) -> list[str]:
    errs: list[str] = []
    for fld in NUMERIC_FIELDS_REQUIRE_FINITE:
        v = entry.get(fld)
        if v is not None and not is_finite_number(v):
            errs.append(f"{label}_NON_FINITE: {fld}={v!r}")
    return errs


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-job-matrix-root", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger-root", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index-root", type=Path, required=True)
    ap.add_argument("--pilot-video-index-root", type=Path, required=True)
    ap.add_argument("--pilot-parent-manifest-root", type=Path, required=True)
    ap.add_argument("--pilot-arm-parity-protocol-root", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    evidence_root = args.evidence_root.resolve()

    # ── Load all sealed roots ─────────────────────────────────────────────
    job_matrix,  job_matrix_seal  = consume_sealed_root(args.pilot_job_matrix_root, EXPECTED_JOB_SCHEMA, "JOB_MATRIX")
    run_ledger,  run_ledger_seal  = consume_sealed_root(args.pilot_run_ledger_root, EXPECTED_LEDGER_SCHEMA, "RUN_LEDGER")
    telem_index, telem_index_seal = consume_sealed_root(args.pilot_telemetry_index_root, EXPECTED_TELEMETRY_SCHEMA, "TELEMETRY")
    video_index, video_index_seal = consume_sealed_root(args.pilot_video_index_root, EXPECTED_VIDEO_SCHEMA, "VIDEO")
    parents_data, parent_seal    = consume_sealed_root(args.pilot_parent_manifest_root, "PILOT_PARENT_MANIFEST_V0", "PARENTS")
    arm_proto,    arm_proto_seal  = consume_sealed_root(args.pilot_arm_parity_protocol_root, EXPECTED_ARM_PROTO_SCHEMA, "ARM_PROTO")

    arm_tolerance = arm_proto.get("max_abs_tolerance")
    if not is_finite_number(arm_tolerance) or float(arm_tolerance) <= 0:
        raise SystemExit(f"ARM_PROTO_TOLERANCE_INVALID: {arm_tolerance!r}")
    arm_tolerance = float(arm_tolerance)

    parent_ids_by_manifest = {p["parent_id"] for p in parents_data.get("parents", [])}

    jobs = require_nonempty_list(job_matrix.get("jobs", []), "JOB_MATRIX_JOBS")
    runs = require_nonempty_list(run_ledger.get("runs", []), "RUN_LEDGER_RUNS")
    telem_entries = require_nonempty_list(telem_index.get("entries", []), "TELEMETRY_ENTRIES")
    video_entries = require_nonempty_list(video_index.get("entries", []), "VIDEO_ENTRIES")

    # ── Build job_id-indexed maps ────────────────────────────────────────
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        jid, mgid = _validate_entry_id(job, "JOB_MATRIX")
        if jid in jobs_by_id:
            raise SystemExit(f"JOB_MATRIX_DUP_JOB_ID: {jid}")
        cond = job.get("condition", "")
        if cond not in CONDITION_CONTRACTS:
            raise SystemExit(f"JOB_MATRIX_UNKNOWN_CONDITION: jid={jid} cond={cond!r}")
        pid = job.get("parent_id", "")
        if not isinstance(pid, str) or not pid:
            raise SystemExit(f"JOB_MATRIX_BAD_PARENT_ID: jid={jid}")
        seed = job.get("perturbation_seed")
        if not is_strict_int(seed):
            raise SystemExit(f"JOB_MATRIX_BAD_SEED: jid={jid} seed={seed!r}")
        rep = job.get("repeat_index")
        if not is_strict_int(rep):
            raise SystemExit(f"JOB_MATRIX_BAD_REP: jid={jid} rep={rep!r}")
        jobs_by_id[jid] = job

    runs_by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        jid, mgid = _validate_entry_id(run, "RUN_LEDGER")
        if jid in runs_by_id:
            raise SystemExit(f"RUN_LEDGER_DUP_JOB_ID: {jid}")
        runs_by_id[jid] = run

    video_by_id: dict[str, dict[str, Any]] = {}
    for ve in video_entries:
        if not isinstance(ve, dict):
            raise SystemExit(f"VIDEO_INDEX_ENTRY_NOT_OBJECT: {ve!r}")
        jid, mgid = _validate_entry_id(ve, "VIDEO_INDEX")
        if jid in video_by_id:
            raise SystemExit(f"VIDEO_INDEX_DUP_JOB_ID: {jid}")
        video_by_id[jid] = ve

    telem_by_id: dict[str, dict[str, Any]] = {}
    for te in telem_entries:
        if not isinstance(te, dict):
            raise SystemExit(f"TELEMETRY_INDEX_ENTRY_NOT_OBJECT: {te!r}")
        jid, mgid = _validate_entry_id(te, "TELEMETRY_INDEX")
        if jid in telem_by_id:
            raise SystemExit(f"TELEMETRY_INDEX_DUP_JOB_ID: {jid}")
        telem_by_id[jid] = te

    # ── Exact set closure on job_id ──────────────────────────────────────
    matrix_ids = set(jobs_by_id)
    ledger_ids = set(runs_by_id)
    video_ids  = set(video_by_id)
    telem_ids  = set(telem_by_id)

    all_errors: list[str] = []

    missing_runs = matrix_ids - ledger_ids
    extra_runs   = ledger_ids - matrix_ids
    for jid in sorted(missing_runs):
        all_errors.append(f"RUN_LEDGER_MISSING_JOB: {jid}")
    for jid in sorted(extra_runs):
        all_errors.append(f"RUN_LEDGER_EXTRA_JOB: {jid}")

    missing_video = matrix_ids - video_ids
    extra_video   = video_ids - matrix_ids
    for jid in sorted(missing_video):
        all_errors.append(f"VIDEO_INDEX_MISSING_JOB: {jid}")
    for jid in sorted(extra_video):
        all_errors.append(f"VIDEO_INDEX_EXTRA_JOB: {jid}")

    missing_telem = matrix_ids - telem_ids
    extra_telem   = telem_ids - matrix_ids
    for jid in sorted(missing_telem):
        all_errors.append(f"TELEMETRY_INDEX_MISSING_JOB: {jid}")
    for jid in sorted(extra_telem):
        all_errors.append(f"TELEMETRY_INDEX_EXTRA_JOB: {jid}")

    # ── Cross-artifact immutable field binding ────────────────────────────
    for jid in sorted(matrix_ids & ledger_ids):
        job = jobs_by_id[jid]
        run = runs_by_id[jid]
        for fld in IMMUTABLE_FIELDS:
            jv = job.get(fld); rv = run.get(fld)
            if jv != rv:
                all_errors.append(f"RUN_FIELD_BINDING: jid={jid} field={fld} matrix={jv!r} run={rv!r}")

    for jid in sorted(matrix_ids & video_ids):
        job = jobs_by_id[jid]
        ve = video_by_id[jid]
        if job.get("matched_group_id") != ve.get("matched_group_id"):
            all_errors.append(f"VIDEO_FIELD_BINDING: jid={jid} field=matched_group_id")

    for jid in sorted(matrix_ids & telem_ids):
        job = jobs_by_id[jid]
        te = telem_by_id[jid]
        if job.get("matched_group_id") != te.get("matched_group_id"):
            all_errors.append(f"TELEM_FIELD_BINDING: jid={jid} field=matched_group_id")

    # ── Reject duplicate condition per matched_group ─────────────────────
    mgid_cond_counts: dict[tuple[str, str], int] = {}
    for jid in sorted(matrix_ids):
        job = jobs_by_id[jid]
        key = (job["matched_group_id"], job["condition"])
        mgid_cond_counts[key] = mgid_cond_counts.get(key, 0) + 1
    for (mgid, cond), count in mgid_cond_counts.items():
        if count > 1:
            all_errors.append(f"DUPLICATE_CONDITION_IN_GROUP: mgid={mgid} cond={cond} count={count}")

    # ── Per-job validation ───────────────────────────────────────────────
    job_errors: dict[str, list[str]] = {jid: [] for jid in matrix_ids}

    for jid in sorted(matrix_ids):
        job = jobs_by_id[jid]
        run = runs_by_id.get(jid)
        je = job_errors[jid]
        contract = CONDITION_CONTRACTS.get(job["condition"], {})
        pid = job["parent_id"]
        cond = job["condition"]

        if run is None:
            continue

        # ── NaN/Inf rejection on all numeric fields ──────────────────
        je.extend(_reject_non_finite(run, f"RUN_{jid}"))

        # ── k contract ──────────────────────────────────────────────
        k_req_expected = contract.get("k_requested", 0)
        k_exec_expected = contract.get("k_executed", 0)
        actual_k_req = run.get("k_requested")
        actual_k_exec = run.get("k_executed")
        if not is_strict_int(actual_k_req) or actual_k_req != k_req_expected:
            je.append(f"K_REQUESTED: expected={k_req_expected} actual={actual_k_req!r}")
        if not is_strict_int(actual_k_exec) or actual_k_exec != k_exec_expected:
            je.append(f"K_EXECUTED: expected={k_exec_expected} actual={actual_k_exec!r}")

        # ── attack_requested MUST be strict bool, match contract ────
        attack_req_expected = contract.get("attack_requested")
        actual_attack_req = run.get("attack_requested")
        if not isinstance(actual_attack_req, bool):
            je.append(f"ATTACK_REQUESTED_NOT_BOOL: {actual_attack_req!r}")
        elif actual_attack_req != attack_req_expected:
            je.append(f"ATTACK_REQUESTED: expected={attack_req_expected} actual={actual_attack_req!r}")

        # ── Condition-specific contract checks ──────────────────────
        if cond == "CLEAN":
            if actual_k_exec != 0:
                je.append("CLEAN_K_NOT_ZERO")

        if cond == "TRUE_T10":
            ga = run.get("gradient_aligned")
            if ga is not True:
                je.append(f"TRUE_NOT_GRADIENT_ALIGNED: {ga!r}")

        if cond == "RAND_T10":
            ga = run.get("gradient_aligned")
            if ga is not False:
                je.append(f"RAND_NOT_GRADIENT_UNALIGNED: {ga!r}")

        if cond == "RANDOM_TIME_T10":
            pmt = run.get("payload_matches_TRUE")
            if pmt is not True:
                je.append(f"RANDOM_TIME_PAYLOAD_NOT_MATCHING: {pmt!r}")

        if cond == "COMMAND_OPEN_ORACLE":
            ot = run.get("oracle_type", "")
            if ot != "command_intervention":
                je.append(f"ORACLE_TYPE_NOT_COMMAND: {ot!r}")

        # ── Attack step ledger per-row validation ───────────────────
        if k_exec_expected > 0:
            astart = run.get("attack_start_step"); aend = run.get("attack_end_step")
            if not is_strict_int(astart) or not is_strict_int(aend):
                je.append(f"ATTACK_STEP_TYPE: start={astart!r} end={aend!r}")
            elif int(astart) < 0 or int(aend) < int(astart):
                je.append(f"ATTACK_STEP_ORDER: start={astart} end={aend}")
            else:
                span = int(aend) - int(astart) + 1
                if span != k_exec_expected:
                    je.append(f"ATTACK_STEP_SPAN: expected={k_exec_expected} actual={span}")

                eval_horizon = run.get("evaluation_horizon")
                if is_strict_int(eval_horizon) and int(aend) >= int(eval_horizon):
                    je.append(f"ATTACK_END_EXCEEDS_HORIZON: end={aend} horizon={eval_horizon}")

            attack_ledger = run.get("attack_step_ledger", run.get("attack_steps", []))
            if isinstance(attack_ledger, list) and len(attack_ledger) == k_exec_expected:
                seen_steps: set[int] = set()
                for si, step in enumerate(attack_ledger):
                    if not isinstance(step, dict):
                        je.append(f"ATTACK_LEDGER_STEP_NOT_OBJECT: idx={si}")
                        continue
                    s = step.get("step")
                    if not is_strict_int(s):
                        je.append(f"ATTACK_LEDGER_STEP_TYPE: idx={si} step={s!r}")
                        continue
                    st = int(s)
                    if st in seen_steps:
                        je.append(f"ATTACK_LEDGER_DUP_STEP: {st}")
                    seen_steps.add(st)
                    if not step.get("armed", False):
                        je.append(f"ATTACK_LEDGER_NOT_ARMED: idx={si} step={st}")
                    if not step.get("executed", False):
                        je.append(f"ATTACK_LEDGER_NOT_EXECUTED: idx={si} step={st}")
                if is_strict_int(astart) and seen_steps:
                    expected_steps = set(range(int(astart), int(astart) + k_exec_expected))
                    if seen_steps != expected_steps:
                        je.append(f"ATTACK_LEDGER_GAP: expected={sorted(expected_steps)} actual={sorted(seen_steps)}")
            elif k_exec_expected > 0:
                je.append("ATTACK_LEDGER_MISSING_OR_WRONG_COUNT")

        # ── Matched parity fields ───────────────────────────────────
        for fld in MATCHED_PARITY_FIELDS:
            if fld not in run or run.get(fld) is None:
                je.append(f"PARITY_MISSING: {fld}")

        # ── Arm parity mandatory for attack conditions ──────────────
        if k_exec_expected > 0 and cond != "CLEAN":
            arm_diff = run.get("arm_max_abs_diff")
            if arm_diff is None:
                je.append("ARM_PARITY_MISSING")
            elif is_finite_number(arm_diff):
                if float(arm_diff) < 0:
                    je.append(f"ARM_DEVIATION_NEGATIVE: {arm_diff}")
                elif float(arm_diff) > arm_tolerance:
                    je.append(f"ARM_DEVIATION: diff={arm_diff} tolerance={arm_tolerance}")
            else:
                je.append(f"ARM_PARITY_NON_FINITE: {arm_diff!r}")

        # ── Evidence closure ────────────────────────────────────────
        ve = video_by_id.get(jid)
        vp = run.get("video_path", "")
        if ve is None:
            if jid not in missing_video:
                je.append(f"VIDEO_INDEX_MISSING_JOB: {jid}")
        elif not vp:
            je.append("VIDEO_PATH_EMPTY_IN_RUN")
        else:
            index_path = ve.get("path", "")
            if index_path != vp:
                je.append(f"VIDEO_PATH_MISMATCH: run={vp} index={index_path}")
            vsha = ve.get("sha256", "")
            if not is_64char_hex(vsha):
                je.append(f"VIDEO_SHA_INVALID: {vsha[:40]!r}")
            else:
                try:
                    verify_evidence_file(evidence_root, vp, vsha, f"VIDEO_{jid}")
                except SystemExit as e:
                    je.append(f"VIDEO: {e}")

        te = telem_by_id.get(jid)
        tp = run.get("telemetry_path", "")
        if te is None:
            if jid not in missing_telem:
                je.append(f"TELEMETRY_INDEX_MISSING_JOB: {jid}")
        elif not tp:
            je.append("TELEMETRY_PATH_EMPTY_IN_RUN")
        else:
            index_path = te.get("path", "")
            if index_path != tp:
                je.append(f"TELEMETRY_PATH_MISMATCH: run={tp} index={index_path}")
            tsha = te.get("sha256", "")
            if not is_64char_hex(tsha):
                je.append(f"TELEMETRY_SHA_INVALID: {tsha[:40]!r}")
            else:
                try:
                    verify_evidence_file(evidence_root, tp, tsha, f"TELEM_{jid}")
                except SystemExit as e:
                    je.append(f"TELEMETRY: {e}")

        all_errors.extend(je)

    # ── Cross-condition parity by matched_group_id ───────────────────────
    groups: dict[str, list[str]] = {}
    for jid in sorted(matrix_ids):
        mgid = jobs_by_id[jid]["matched_group_id"]
        groups.setdefault(mgid, []).append(jid)

    for mgid in sorted(groups):
        jids = groups[mgid]
        cond_entries: dict[str, tuple[str, dict[str, Any]]] = {}
        for jid in jids:
            run = runs_by_id.get(jid)
            if run is not None:
                cond = jobs_by_id[jid]["condition"]
                if cond in cond_entries:
                    # Duplicate condition in same matched_group — already flagged above,
                    # but also refuse to silently overwrite during parity comparison
                    all_errors.append(f"PARITY_DUP_CONDITION: mgid={mgid} cond={cond}")
                    continue
                cond_entries[cond] = (jid, run)

        if len(cond_entries) < 2:
            continue

        affected_jids: set[str] = set()

        # ── Shared matched parity ───────────────────────────────────
        for fld in MATCHED_PARITY_FIELDS:
            vals = {c: r.get(fld) for c, (_, r) in cond_entries.items() if fld in r}
            if len(set(vals.values())) > 1:
                all_errors.append(f"PARITY_DIVERGENT: mgid={mgid} field={fld} values={vals}")
                affected_jids.update(jid for _, (jid, _) in cond_entries.items())

        # ── TRUE vs RAND parity ─────────────────────────────────────
        if "TRUE_T10" in cond_entries and "RAND_T10" in cond_entries:
            true_jid, true_run = cond_entries["TRUE_T10"]
            rand_jid, rand_run = cond_entries["RAND_T10"]
            for fld in TRUE_RAND_PARITY_FIELDS:
                tv = true_run.get(fld); rv = rand_run.get(fld)
                if tv is None or rv is None:
                    all_errors.append(f"TRUE_RAND_PARITY_MISSING: mgid={mgid} field={fld}")
                    affected_jids.update([true_jid, rand_jid])
                elif tv != rv:
                    all_errors.append(f"TRUE_RAND_PARITY: mgid={mgid} field={fld} TRUE={tv!r} RAND={rv!r}")
                    affected_jids.update([true_jid, rand_jid])
            ts = true_run.get("attack_start_step"); rs = rand_run.get("attack_start_step")
            if ts is not None and rs is not None and ts != rs:
                all_errors.append(f"TRUE_RAND_START_STEP_MISMATCH: mgid={mgid} TRUE={ts} RAND={rs}")
                affected_jids.update([true_jid, rand_jid])

        # ── TRUE vs RANDOM_TIME parity ──────────────────────────────
        if "TRUE_T10" in cond_entries and "RANDOM_TIME_T10" in cond_entries:
            true_jid, true_run = cond_entries["TRUE_T10"]
            rt_jid, rt_run = cond_entries["RANDOM_TIME_T10"]
            for fld in TRUE_RANDOM_TIME_PARITY_FIELDS:
                tv = true_run.get(fld); rv = rt_run.get(fld)
                if tv is None or rv is None:
                    all_errors.append(f"TRUE_RT_PARITY_MISSING: mgid={mgid} field={fld}")
                    affected_jids.update([true_jid, rt_jid])
                elif tv != rv:
                    all_errors.append(f"TRUE_RT_PARITY: mgid={mgid} field={fld} TRUE={tv!r} RT={rv!r}")
                    affected_jids.update([true_jid, rt_jid])

        # ── Checkpoint cross-check ──────────────────────────────────
        checkpoints = {c: r.get("checkpoint_sha256") for c, (_, r) in cond_entries.items()}
        if len(set(checkpoints.values())) > 1:
            all_errors.append(f"CHECKPOINT_DIVERGENT: mgid={mgid} shas={checkpoints}")
            affected_jids.update(jid for _, (jid, _) in cond_entries.items())

        # ── Write parity errors back to job_errors ──────────────────
        for ajid in affected_jids:
            if ajid in job_errors:
                je_list = job_errors[ajid]
                if "PARITY_MISMATCH" not in str(je_list):
                    je_list.append("PARITY_MISMATCH")

    # ── Compute final dispositions (UNIFIED source of truth) ─────────────
    disposition: dict[str, str] = {}
    for jid in sorted(matrix_ids):
        je = job_errors[jid]
        run = runs_by_id.get(jid)
        if run is None:
            disposition[jid] = "MISSING"
        elif not je:
            disposition[jid] = "COMPLETE_VALID"
        elif any("PARITY_MISMATCH" in e for e in je):
            disposition[jid] = "PARITY_MISMATCH"
        elif any("K_" in e or "ATTACK_" in e for e in je):
            disposition[jid] = "PARTIAL_ATTACK"
        elif any("VIDEO" in e for e in je):
            disposition[jid] = "MISSING_VIDEO"
        elif any("TELEMETRY" in e for e in je):
            disposition[jid] = "MISSING_TELEMETRY"
        elif any("ARM_" in e for e in je):
            disposition[jid] = "ARM_PARITY_FAIL"
        else:
            disposition[jid] = "PROTOCOL_MISMATCH"

    # ── Compute disposition counts ───────────────────────────────────────
    disp_counts: dict[str, int] = {}
    for d in disposition.values():
        disp_counts[d] = disp_counts.get(d, 0) + 1

    # ── Budget parity rows (derived from final disposition) ──────────────
    budget_parity: list[dict[str, Any]] = []
    for jid in sorted(matrix_ids):
        job = jobs_by_id[jid]
        run = runs_by_id.get(jid)
        final_disp = disposition.get(jid, "UNKNOWN")
        budget_parity.append({
            "job_id": jid, "matched_group_id": job["matched_group_id"],
            "parent_id": job["parent_id"], "condition": job["condition"],
            "seed": job["perturbation_seed"], "repeat": job["repeat_index"],
            "k_requested": run.get("k_requested") if run else None,
            "k_executed": run.get("k_executed") if run else None,
            "disposition": final_disp,
            "valid": final_disp == "COMPLETE_VALID",
        })

    # ══════════════════════════════════════════════════════════════════════
    # Receipt
    # ══════════════════════════════════════════════════════════════════════
    receipt = {
        "schema": "PILOT_EXECUTION_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not all_errors else "HOLD",
        "n_expected_jobs": len(jobs_by_id), "n_actual_runs": len(runs_by_id),
        "n_missing_runs": len(missing_runs), "n_extra_runs": len(extra_runs),
        "n_missing_video": len(missing_video), "n_extra_video": len(extra_video),
        "n_missing_telem": len(missing_telem), "n_extra_telem": len(extra_telem),
        "allowed_conditions": sorted(set(j["condition"] for j in jobs_by_id.values())),
        "arm_tolerance": arm_tolerance,
        "n_errors": len(all_errors), "errors": all_errors[:200],
        "disposition_counts": disp_counts,
        "attack_eval_consumed": False,
        "input_seals": {
            "job_matrix": job_matrix_seal,
            "run_ledger": run_ledger_seal,
            "video_index": video_index_seal,
            "telemetry_index": telem_index_seal,
            "parent_manifest": parent_seal,
            "arm_protocol": arm_proto_seal,
        },
    }

    # ── Write all outputs (AFTER all validation + parity) ─────────────────
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_EXECUTION_VALIDATION_V0.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with open(staging / "PILOT_BUDGET_PARITY_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "matched_group_id", "parent_id", "condition", "seed", "repeat",
                     "k_requested", "k_executed", "disposition", "valid"])
        for bp in budget_parity:
            w.writerow([bp[k] for k in ["job_id", "matched_group_id", "parent_id", "condition",
                         "seed", "repeat", "k_requested", "k_executed", "disposition", "valid"]])

    with open(staging / "PILOT_DISPOSITION_V0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "matched_group_id", "parent_id", "condition", "seed", "repeat", "disposition"])
        for jid in sorted(matrix_ids):
            job = jobs_by_id[jid]
            w.writerow([jid, job["matched_group_id"], job["parent_id"], job["condition"],
                        job["perturbation_seed"], job["repeat_index"], disposition.get(jid, "UNKNOWN")])

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
