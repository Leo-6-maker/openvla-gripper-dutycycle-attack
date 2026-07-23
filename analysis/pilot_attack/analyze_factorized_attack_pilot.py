#!/usr/bin/env python3
"""B3 v2.2: Pilot paired analysis — consumes sealed execution PASS receipt, applies GO rules."""
from __future__ import annotations

import argparse, csv, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, consume_sealed_root

SELF_SHA = None
OUTCOME_FIELDS = ("official_success", "gripper_opened", "object_dropped", "transport_complete",
                  "placement_success", "contact_quality_failure")
COMPARISONS = [("TRUE_T10", "CLEAN"), ("TRUE_T10", "RAND_T10"),
               ("TRUE_T10", "RANDOM_TIME_T10"), ("COMMAND_OPEN_ORACLE", "CLEAN")]


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-execution-validation-root", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger-root", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index-root", type=Path, required=True)
    ap.add_argument("--pilot-go-no-go-rules-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    exec_val, exec_val_seal = consume_sealed_root(
        args.pilot_execution_validation_root, "PILOT_EXECUTION_VALIDATION_V0", "EXEC_VAL")
    if exec_val.get("status") != "PASS":
        raise SystemExit("EXEC_VALIDATION_NOT_PASS: cannot run analysis on HOLD execution")

    run_ledger, _ = consume_sealed_root(args.pilot_run_ledger_root, "PILOT_RUN_LEDGER_V0", "RUN_LEDGER")
    consume_sealed_root(args.pilot_telemetry_index_root, "PILOT_TELEMETRY_INDEX_V0", "TELEMETRY")
    go_rules, _ = consume_sealed_root(args.pilot_go_no_go_rules_root, "PILOT_GO_NO_GO_RULES_V0", "GO_RULES")

    runs = run_ledger.get("runs", [])
    disp_counts = exec_val.get("disposition_counts", {})
    n_valid = disp_counts.get("COMPLETE_VALID", 0)

    parents: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        pid = run.get("parent_id", ""); cond = run.get("condition", "")
        parents.setdefault(pid, {}).setdefault(cond, []).append(run)

    paired_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for pid in sorted(parents):
        conds = parents[pid]
        for cond_a, cond_b in COMPARISONS:
            a_runs = conds.get(cond_a, []); b_runs = conds.get(cond_b, [])
            pair_key = f"{pid}/{cond_a}_vs_{cond_b}"
            result: dict[str, Any] = {"parent_id": pid, "condition_a": cond_a, "condition_b": cond_b}
            for field in OUTCOME_FIELDS:
                a_vals = [r.get(field) for r in a_runs if field in r]
                b_vals = [r.get(field) for r in b_runs if field in r]
                if a_vals and b_vals:
                    result[f"{field}_a_rate"] = sum(1 for v in a_vals if v) / len(a_vals)
                    result[f"{field}_b_rate"] = sum(1 for v in b_vals if v) / len(b_vals)
                    result[f"{field}_n_a"] = len(a_vals); result[f"{field}_n_b"] = len(b_vals)
            paired_results[pair_key] = result
            summary_rows.append({"parent_id": pid, "pair": pair_key, "n_a": len(a_runs), "n_b": len(b_runs)})

    # GO/NO-GO checks
    go_checks: dict[str, Any] = {
        "execution_validation_PASS": exec_val.get("status") == "PASS",
        "all_jobs_closure": disp_counts.get("MISSING", 0) == 0 and disp_counts.get("EXTRA", 0) == 0,
        "no_evidence_gaps": disp_counts.get("MISSING_VIDEO", 0) == 0 and disp_counts.get("MISSING_TELEMETRY", 0) == 0,
        "n_valid_pairs": n_valid,
        "ITT_denominator": exec_val.get("n_expected_jobs", 0),
    }

    blocker_reasons: list[str] = []
    if not go_checks["execution_validation_PASS"]: blocker_reasons.append("execution validation not PASS")
    if not go_checks["all_jobs_closure"]: blocker_reasons.append("job closure incomplete")
    if not go_checks["no_evidence_gaps"]: blocker_reasons.append("video/telemetry gaps")

    recommendation = "STOP" if blocker_reasons else "CONTINUE"
    if not blocker_reasons and n_valid < 4:
        recommendation = "MODIFY_DETECTOR"

    go_result = {
        "schema": "PILOT_GO_NO_GO_V0", "recommendation": recommendation,
        "blocker_reasons": blocker_reasons, "checks": go_checks,
        "direction_only": True, "paper_table1_eligible": False, "attack_authorized": False,
        "execution_validation_seal_sha256": exec_val_seal,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    with open(staging / "PILOT_ATTACK_SUMMARY_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parent_id", "pair", "n_a", "n_b"]); w.writeheader()
        for row in summary_rows: w.writerow(row)
    (staging / "PILOT_PAIRED_RESULTS_V0.json").write_text(json.dumps(paired_results, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_GO_NO_GO_V0.json").write_text(json.dumps(go_result, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_MECHANISM_DIAGNOSIS_V0.md").write_text(
        f"# Pilot Mechanism Diagnosis\n\nExecution receipt: {exec_val_seal}\n"
        f"Recommendation: {recommendation}\nBlockers: {blocker_reasons}\n\n"
        + "\n".join(f"- {pid}: {list(conds.keys())}" for pid, conds in sorted(parents.items())) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Analysis: recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
