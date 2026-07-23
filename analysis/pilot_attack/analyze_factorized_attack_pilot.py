#!/usr/bin/env python3
"""B3 v2.3: Pilot paired analysis — seal binding, matched_group pairing, real GO/NO-GO rule application."""
from __future__ import annotations

import argparse, csv, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, is_finite_number, consume_sealed_root

SELF_SHA = None
OUTCOME_FIELDS = ("official_success", "gripper_opened", "object_dropped", "transport_complete",
                  "placement_success", "contact_quality_failure")


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-execution-validation-root", type=Path, required=True)
    ap.add_argument("--pilot-job-matrix-root", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger-root", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index-root", type=Path, required=True)
    ap.add_argument("--pilot-go-no-go-rules-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # ── Consume all sealed roots ──────────────────────────────────────────
    exec_val, exec_val_seal = consume_sealed_root(
        args.pilot_execution_validation_root, "PILOT_EXECUTION_VALIDATION_V0", "EXEC_VAL")
    if exec_val.get("status") != "PASS":
        raise SystemExit("EXEC_VALIDATION_NOT_PASS: cannot run analysis on HOLD execution")

    job_matrix, job_matrix_seal = consume_sealed_root(
        args.pilot_job_matrix_root, "PILOT_JOB_MATRIX_V0", "JOB_MATRIX")
    run_ledger, run_ledger_seal = consume_sealed_root(
        args.pilot_run_ledger_root, "PILOT_RUN_LEDGER_V0", "RUN_LEDGER")
    telem_index, telem_index_seal = consume_sealed_root(
        args.pilot_telemetry_index_root, "PILOT_TELEMETRY_INDEX_V0", "TELEMETRY")
    go_rules, go_rules_seal = consume_sealed_root(
        args.pilot_go_no_go_rules_root, "PILOT_GO_NO_GO_RULES_V0", "GO_RULES")

    # ── Verify input seal binding against execution receipt ───────────────
    declared_seals = exec_val.get("input_seals", {})
    binding_errors: list[str] = []
    actual_seals = {
        "job_matrix": job_matrix_seal,
        "run_ledger": run_ledger_seal,
        "telemetry_index": telem_index_seal,
    }
    for key, actual in actual_seals.items():
        declared = declared_seals.get(key, "")
        if declared != actual:
            binding_errors.append(
                f"SEAL_BINDING_MISMATCH: {key} declared={declared[:16]!r} actual={actual[:16]!r}")

    if binding_errors:
        raise SystemExit("CROSS_RECEIPT_SUBSTITUTION: " + "; ".join(binding_errors))

    # ── Load runs, index by job_id ────────────────────────────────────────
    runs = run_ledger.get("runs", [])
    jobs = job_matrix.get("jobs", [])

    runs_by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        jid = run.get("job_id", "")
        if jid:
            runs_by_id[jid] = run

    jobs_by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        jid = job.get("job_id", "")
        if jid:
            jobs_by_id[jid] = job

    # ── Build matched groups from job matrix ──────────────────────────────
    groups: dict[str, list[str]] = {}
    for jid, job in jobs_by_id.items():
        mgid = job.get("matched_group_id", "")
        if mgid:
            groups.setdefault(mgid, []).append(jid)

    # ── Per-group paired analysis ─────────────────────────────────────────
    paired_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for mgid in sorted(groups):
        jids = groups[mgid]
        cond_runs: dict[str, dict[str, Any]] = {}
        for jid in jids:
            run = runs_by_id.get(jid)
            if run:
                cond_runs[run.get("condition", "UNKNOWN")] = run

        COMPARISONS = [
            ("TRUE_T10", "CLEAN"),
            ("TRUE_T10", "RAND_T10"),
            ("TRUE_T10", "RANDOM_TIME_T10"),
            ("COMMAND_OPEN_ORACLE", "CLEAN"),
        ]

        for cond_a, cond_b in COMPARISONS:
            a_run = cond_runs.get(cond_a)
            b_run = cond_runs.get(cond_b)
            pair_key = f"{mgid}/{cond_a}_vs_{cond_b}"
            result: dict[str, Any] = {"matched_group_id": mgid, "condition_a": cond_a, "condition_b": cond_b}
            if a_run is not None and b_run is not None:
                for field in OUTCOME_FIELDS:
                    av = a_run.get(field); bv = b_run.get(field)
                    if av is not None and bv is not None:
                        result[f"{field}_a"] = av
                        result[f"{field}_b"] = bv
                result["both_present"] = True
            else:
                result["both_present"] = False
            paired_results[pair_key] = result
            summary_rows.append({"matched_group_id": mgid, "pair": pair_key,
                                 "a_present": a_run is not None, "b_present": b_run is not None})

    # ── Compute GO/NO-GO metrics ──────────────────────────────────────────
    n_groups = len(groups)
    n_oracle_physical = 0
    n_true_over_rand = 0
    n_true_over_random_time = 0
    n_oracle_over_clean = 0
    n_complete_groups = 0

    for mgid in sorted(groups):
        jids = groups[mgid]
        cond_runs: dict[str, dict[str, Any]] = {}
        for jid in jids:
            run = runs_by_id.get(jid)
            if run:
                cond_runs[run.get("condition", "UNKNOWN")] = run

        all_expected = {"TRUE_T10", "RAND_T10", "RANDOM_TIME_T10", "COMMAND_OPEN_ORACLE", "CLEAN"}
        present = set(cond_runs)
        if all_expected.issubset(present):
            n_complete_groups += 1

        oracle_run = cond_runs.get("COMMAND_OPEN_ORACLE")
        if oracle_run and oracle_run.get("gripper_opened") is True:
            n_oracle_physical += 1

        clean_run = cond_runs.get("CLEAN")
        if oracle_run and clean_run:
            o_success = oracle_run.get("official_success", False)
            c_success = clean_run.get("official_success", False)
            if o_success and not c_success:
                n_oracle_over_clean += 1

        true_run = cond_runs.get("TRUE_T10")
        rand_run = cond_runs.get("RAND_T10")
        if true_run and rand_run:
            t_fail = not true_run.get("official_success", True)
            r_fail = not rand_run.get("official_success", True)
            if t_fail and not r_fail:
                n_true_over_rand += 1

        rt_run = cond_runs.get("RANDOM_TIME_T10")
        if true_run and rt_run:
            t_fail = not true_run.get("official_success", True)
            rt_fail = not rt_run.get("official_success", True)
            if t_fail and not rt_fail:
                n_true_over_random_time += 1

    # ── Apply rules from sealed GO/NO-GO rules ────────────────────────────
    rules = go_rules.get("rules", {})
    min_valid_pairs = rules.get("min_valid_pairs", 4)
    min_oracle_physical = rules.get("min_oracle_physical_parents", 1)
    min_true_vs_rand = rules.get("min_true_over_rand_parents", 1)
    min_true_vs_rt = rules.get("min_true_over_random_time_parents", 1)
    require_all_conditions = rules.get("require_all_conditions_per_group", True)

    disp_counts = exec_val.get("disposition_counts", {})
    n_missing_video = disp_counts.get("MISSING_VIDEO", 0)
    n_missing_telem = disp_counts.get("MISSING_TELEMETRY", 0)
    max_missing_evidence = rules.get("max_missing_evidence", 0)

    checks: dict[str, Any] = {
        "n_groups": n_groups,
        "n_complete_groups": n_complete_groups,
        "n_oracle_physical": n_oracle_physical,
        "n_true_over_rand": n_true_over_rand,
        "n_true_over_random_time": n_true_over_random_time,
        "n_oracle_over_clean": n_oracle_over_clean,
        "n_missing_video": n_missing_video,
        "n_missing_telem": n_missing_telem,
    }

    blocker_reasons: list[str] = []

    if n_groups < min_valid_pairs:
        blocker_reasons.append(f"INSUFFICIENT_GROUPS: {n_groups} < {min_valid_pairs}")
    if n_oracle_physical < min_oracle_physical:
        blocker_reasons.append(f"INSUFFICIENT_ORACLE_PHYSICAL: {n_oracle_physical} < {min_oracle_physical}")
    if n_true_over_rand < min_true_vs_rand:
        blocker_reasons.append(f"TRUE_NOT_BEATING_RAND: {n_true_over_rand} < {min_true_vs_rand}")
    if n_true_over_random_time < min_true_vs_rt:
        blocker_reasons.append(f"TRUE_NOT_BEATING_RANDOM_TIME: {n_true_over_random_time} < {min_true_vs_rt}")
    if require_all_conditions and n_complete_groups < n_groups:
        blocker_reasons.append(f"INCOMPLETE_GROUPS: {n_complete_groups}/{n_groups}")
    if n_missing_video > max_missing_evidence:
        blocker_reasons.append(f"MISSING_VIDEO: {n_missing_video} > {max_missing_evidence}")
    if n_missing_telem > max_missing_evidence:
        blocker_reasons.append(f"MISSING_TELEMETRY: {n_missing_telem} > {max_missing_evidence}")

    # ── Determine recommendation ──────────────────────────────────────────
    if blocker_reasons:
        recommendation = "STOP"
    elif n_true_over_rand < min_true_vs_rand and n_true_over_random_time >= min_true_vs_rt:
        recommendation = "STOP_TIMING"
    elif n_oracle_physical < min_oracle_physical:
        recommendation = "STOP_WINDOW"
    elif n_groups < min_valid_pairs:
        recommendation = "MODIFY_DETECTOR"
    else:
        recommendation = "CONTINUE"

    go_result = {
        "schema": "PILOT_GO_NO_GO_V0",
        "recommendation": recommendation,
        "blocker_reasons": blocker_reasons,
        "checks": checks,
        "sealed_rules_sha256": go_rules_seal,
        "direction_only": True,
        "paper_table1_eligible": False,
        "attack_authorized": False,
        "execution_validation_seal_sha256": exec_val_seal,
    }

    # ── Mechanism diagnosis ───────────────────────────────────────────────
    diagnosis_lines = [
        f"# Pilot Mechanism Diagnosis",
        f"",
        f"Execution receipt seal: {exec_val_seal}",
        f"GO/NO-GO rules seal: {go_rules_seal}",
        f"Recommendation: **{recommendation}**",
        f"",
        f"## Blockers",
    ]
    if blocker_reasons:
        for b in blocker_reasons:
            diagnosis_lines.append(f"- {b}")
    else:
        diagnosis_lines.append("- (none)")
    diagnosis_lines.append("")
    diagnosis_lines.append("## Per-Group Summary")
    diagnosis_lines.append("")
    for mgid in sorted(groups):
        jids = groups[mgid]
        conds = [jobs_by_id[jid]["condition"] for jid in jids if jid in jobs_by_id]
        diagnosis_lines.append(f"- {mgid}: {sorted(conds)}")
    diagnosis_lines.append("")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    with open(staging / "PILOT_ATTACK_SUMMARY_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matched_group_id", "pair", "a_present", "b_present"])
        w.writeheader()
        for row in summary_rows: w.writerow(row)

    (staging / "PILOT_PAIRED_RESULTS_V0.json").write_text(
        json.dumps(paired_results, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_GO_NO_GO_V0.json").write_text(
        json.dumps(go_result, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_MECHANISM_DIAGNOSIS_V0.md").write_text("\n".join(diagnosis_lines) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Analysis: recommendation={recommendation}")
    print(f"  Checks: {json.dumps(checks)}")
    return 0 if recommendation == "CONTINUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
