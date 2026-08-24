#!/usr/bin/env python3
"""B3 v2.3.3: Pilot paired analysis — graceful re-validation, Oracle Plan A, separated automated signals."""
from __future__ import annotations

import argparse, csv, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, is_64char_hex, is_finite_number, is_strict_int, consume_sealed_root

SELF_SHA = None
OUTCOME_FIELDS = ("official_success", "gripper_opened", "object_dropped", "transport_complete",
                  "placement_success", "contact_quality_failure")

REQUIRED_RULE_FIELDS = (
    "min_valid_pairs", "min_oracle_actuation_parents",
    "min_true_over_rand_parents", "min_true_over_random_time_parents",
    "max_missing_evidence", "require_all_conditions_per_group",
)


def _validate_rules(rules: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for fld in REQUIRED_RULE_FIELDS:
        if fld not in rules:
            errs.append(f"GO_RULES_MISSING_FIELD: {fld}")
            continue
        v = rules[fld]
        if fld == "require_all_conditions_per_group":
            if not isinstance(v, bool):
                errs.append(f"GO_RULES_NOT_BOOL: {fld}={v!r}")
        else:
            if not is_strict_int(v) or v < 0:
                errs.append(f"GO_RULES_NOT_NONNEG_INT: {fld}={v!r}")
    return errs


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

    # ── Strict GO rules validation ────────────────────────────────────────
    raw_rules = go_rules.get("rules", {})
    rule_errors = _validate_rules(raw_rules)
    if rule_errors:
        raise SystemExit("GO_RULES_INVALID: " + "; ".join(rule_errors))

    # ── Verify input seal binding against execution receipt ───────────────
    REQUIRED_SEAL_KEYS = {"job_matrix", "run_ledger", "telemetry_index"}
    declared_seals = exec_val.get("input_seals", {})
    binding_errors: list[str] = []
    actual_seals: dict[str, str] = {
        "job_matrix": job_matrix_seal,
        "run_ledger": run_ledger_seal,
        "telemetry_index": telem_index_seal,
    }
    declared_valid_keys = {k for k, v in declared_seals.items() if is_64char_hex(v)}
    if declared_valid_keys:
        missing = REQUIRED_SEAL_KEYS - declared_valid_keys
        if missing:
            raise SystemExit(
                "SEAL_DECLARATION_INCOMPLETE: "
                f"missing={sorted(missing)}")
        for key in REQUIRED_SEAL_KEYS:
            if declared_seals[key] != actual_seals[key]:
                binding_errors.append(
                    f"SEAL_BINDING_MISMATCH: {key} declared={declared_seals[key][:16]} actual={actual_seals[key][:16]}")
    if binding_errors:
        raise SystemExit("CROSS_RECEIPT_SUBSTITUTION: " + "; ".join(binding_errors))

    # ── Load runs and jobs ────────────────────────────────────────────────
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

    # ── Re-validation: collect errors, produce results (no SystemExit) ────
    infrastructure_errors: list[str] = []
    matrix_ids = set(jobs_by_id)
    ledger_ids = set(runs_by_id)
    extra_in_ledger = ledger_ids - matrix_ids
    missing_from_ledger = matrix_ids - ledger_ids
    for jid in sorted(missing_from_ledger):
        infrastructure_errors.append(f"ANALYSIS_MISSING_RUN: {jid}")
    for jid in sorted(extra_in_ledger):
        infrastructure_errors.append(f"ANALYSIS_EXTRA_RUN: {jid}")

    # Build matched groups
    groups: dict[str, list[str]] = {}
    for jid, job in jobs_by_id.items():
        mgid = job.get("matched_group_id", "")
        if mgid:
            groups.setdefault(mgid, []).append(jid)

    n_groups = len(groups)
    if n_groups == 0:
        infrastructure_errors.append("ANALYSIS_ZERO_GROUPS")

    dup_cond_errors: list[str] = []
    for mgid, jids in groups.items():
        seen_conds: set[str] = set()
        for jid in jids:
            cond = jobs_by_id[jid].get("condition", "")
            if cond in seen_conds:
                dup_cond_errors.append(f"ANALYSIS_DUP_CONDITION: mgid={mgid} cond={cond}")
            seen_conds.add(cond)
    infrastructure_errors.extend(dup_cond_errors)

    # ── Apply rules ───────────────────────────────────────────────────────
    min_valid_pairs = raw_rules["min_valid_pairs"]
    min_oracle_actuation = raw_rules["min_oracle_actuation_parents"]
    min_true_vs_rand = raw_rules["min_true_over_rand_parents"]
    min_true_vs_rt = raw_rules["min_true_over_random_time_parents"]
    max_missing_evidence = raw_rules["max_missing_evidence"]
    require_all_conditions = raw_rules["require_all_conditions_per_group"]

    disp_counts = exec_val.get("disposition_counts", {})
    n_missing_video = disp_counts.get("MISSING_VIDEO", 0)
    n_missing_telem = disp_counts.get("MISSING_TELEMETRY", 0)

    evidence_fail = (n_missing_video > max_missing_evidence or
                     n_missing_telem > max_missing_evidence)
    if evidence_fail:
        infrastructure_errors.append(
            f"EVIDENCE_GAPS: missing_video={n_missing_video} missing_telem={n_missing_telem}")

    # ── Compute GO/NO-GO metrics ──────────────────────────────────────────
    n_oracle_actuation = 0
    n_oracle_degradation = 0
    n_true_over_rand = 0
    n_true_over_random_time = 0
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
            n_oracle_actuation += 1

        clean_run = cond_runs.get("CLEAN")
        if oracle_run and clean_run:
            c_success = clean_run.get("official_success", False)
            o_success = oracle_run.get("official_success", False)
            if c_success and not o_success:
                n_oracle_degradation += 1

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

    # ── Automated mechanism signals ───────────────────────────────────────
    infrastructure_ready = (len(infrastructure_errors) == 0)
    oracle_actuation_supported = n_oracle_actuation >= min_oracle_actuation
    oracle_physical_effect_supported = n_oracle_degradation >= min_oracle_actuation
    payload_specificity_signal = n_true_over_rand >= min_true_vs_rand
    timing_specificity_signal = n_true_over_random_time >= min_true_vs_rt

    checks: dict[str, Any] = {
        "n_groups": n_groups,
        "n_complete_groups": n_complete_groups,
        "n_oracle_actuation": n_oracle_actuation,
        "n_oracle_degradation": n_oracle_degradation,
        "n_true_over_rand": n_true_over_rand,
        "n_true_over_random_time": n_true_over_random_time,
        "n_missing_video": n_missing_video,
        "n_missing_telem": n_missing_telem,
    }

    # ── Decision tree: mutually exclusive ─────────────────────────────────
    blocker_reasons: list[str] = []
    incomplete = require_all_conditions and n_complete_groups < n_groups
    insufficient_groups = (not infrastructure_errors) and n_groups < min_valid_pairs
    oracle_actuation_fail = n_oracle_actuation < min_oracle_actuation
    true_rand_fail = n_true_over_rand < min_true_vs_rand
    true_rt_fail = n_true_over_random_time < min_true_vs_rt

    if not infrastructure_ready:
        blocker_reasons.extend(infrastructure_errors)
    if incomplete:
        blocker_reasons.append(f"INCOMPLETE_GROUPS: {n_complete_groups}/{n_groups}")
    if insufficient_groups:
        blocker_reasons.append(f"INSUFFICIENT_GROUPS: {n_groups} < {min_valid_pairs}")

    if not infrastructure_ready:
        automated_recommendation = "STOP"
    elif insufficient_groups:
        automated_recommendation = "MODIFY_DETECTOR"
    elif incomplete:
        automated_recommendation = "STOP"
    elif oracle_actuation_fail:
        blocker_reasons.append(
            f"ORACLE_ACTUATION_FAIL: n_oracle_actuation={n_oracle_actuation} < {min_oracle_actuation}")
        automated_recommendation = "STOP_WINDOW"
    elif true_rand_fail:
        blocker_reasons.append(
            f"TRUE_NOT_BEATING_RAND: n_true_over_rand={n_true_over_rand} < {min_true_vs_rand}")
        if true_rt_fail:
            blocker_reasons.append(
                f"TRUE_NOT_BEATING_RANDOM_TIME: n_true_over_random_time={n_true_over_random_time} < {min_true_vs_rt}")
            automated_recommendation = "STOP"
        else:
            automated_recommendation = "STOP_TIMING"
    elif true_rt_fail:
        blocker_reasons.append(
            f"TRUE_NOT_BEATING_RANDOM_TIME: n_true_over_random_time={n_true_over_random_time} < {min_true_vs_rt}")
        automated_recommendation = "MODIFY_DETECTOR"
    else:
        automated_recommendation = "CONTINUE"

    # ── Per-group paired results ──────────────────────────────────────────
    paired_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    COMPARISONS = [
        ("TRUE_T10", "CLEAN"),
        ("TRUE_T10", "RAND_T10"),
        ("TRUE_T10", "RANDOM_TIME_T10"),
        ("COMMAND_OPEN_ORACLE", "CLEAN"),
    ]

    for mgid in sorted(groups):
        jids = groups[mgid]
        cond_runs: dict[str, dict[str, Any]] = {}
        for jid in jids:
            run = runs_by_id.get(jid)
            if run:
                cond_runs[run.get("condition", "UNKNOWN")] = run

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

    go_result = {
        "schema": "PILOT_AUTOMATED_GO_NO_GO_V0",
        "automated_recommendation": automated_recommendation,
        "blocker_reasons": blocker_reasons,
        "checks": checks,
        "infrastructure_ready": infrastructure_ready,
        "oracle_actuation_supported": oracle_actuation_supported,
        "oracle_physical_effect_supported": oracle_physical_effect_supported,
        "payload_specificity_signal": payload_specificity_signal,
        "timing_specificity_signal": timing_specificity_signal,
        "sealed_rules_sha256": go_rules_seal,
        "direction_only": True,
        "paper_table1_eligible": False,
        "attack_authorized": False,
        "scientific_go_no_go_authorized": False,
        "execution_validation_seal_sha256": exec_val_seal,
    }

    # ── Mechanism diagnosis ───────────────────────────────────────────────
    diagnosis_lines = [
        "# Pilot Automated GO/NO-GO Diagnosis",
        "",
        f"Execution receipt seal: {exec_val_seal}",
        f"GO/NO-GO rules seal: {go_rules_seal}",
        f"Automated recommendation: **{automated_recommendation}**",
        "",
        "## Infrastructure",
        f"- ready: {infrastructure_ready}",
        f"- infrastructure_errors: {len(infrastructure_errors)}",
        "",
        "## Mechanism Signals",
        f"- oracle_actuation: {oracle_actuation_supported} (n={n_oracle_actuation}, min={min_oracle_actuation})",
        f"- oracle_physical_effect: {oracle_physical_effect_supported} (n={n_oracle_degradation})",
        f"- payload_specificity: {payload_specificity_signal} (n={n_true_over_rand}, min={min_true_vs_rand})",
        f"- timing_specificity: {timing_specificity_signal} (n={n_true_over_random_time}, min={min_true_vs_rt})",
        "",
        "## Blockers",
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
    (staging / "PILOT_AUTOMATED_GO_NO_GO_V0.json").write_text(
        json.dumps(go_result, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_MECHANISM_DIAGNOSIS_V0.md").write_text("\n".join(diagnosis_lines) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Automated GO/NO-GO: automated_recommendation={automated_recommendation}")
    print(f"  Infrastructure ready: {infrastructure_ready}")
    print(f"  Checks: {json.dumps(checks)}")
    return 0 if automated_recommendation == "CONTINUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
