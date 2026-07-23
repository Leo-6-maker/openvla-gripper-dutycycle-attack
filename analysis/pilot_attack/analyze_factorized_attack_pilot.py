#!/usr/bin/env python3
"""B3: Pilot paired analysis — TRUE vs CLEAN/RAND/RANDOM_TIME, no overstatement."""
from __future__ import annotations

import argparse, csv, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from pilot_integrity import sha256_file, load_strict_json, seal_output_dir

SELF_SHA = None
PAIRS = [("TRUE_T10", "CLEAN"), ("TRUE_T10", "RAND_T10"), ("TRUE_T10", "RANDOM_TIME_T10"),
         ("COMMAND_OPEN_ORACLE", "CLEAN")]
OUTCOME_FIELDS = ("official_success", "gripper_opened", "object_dropped", "transport_complete",
                  "placement_success", "contact_quality_failure")


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-telemetry-index", type=Path, required=True)
    ap.add_argument("--pilot-go-no-go-rules", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    run_ledger = load_strict_json(args.pilot_run_ledger, "LEDGER")
    telemetry = load_strict_json(args.pilot_telemetry_index, "TELEMETRY")
    go_rules = load_strict_json(args.pilot_go_no_go_rules, "GO_RULES")

    runs = run_ledger.get("runs", run_ledger.get("entries", []))
    parents: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        pid = run.get("parent_id", "")
        cond = run.get("condition", "")
        parents.setdefault(pid, {}).setdefault(cond, []).append(run)

    summary_rows: list[dict[str, Any]] = []
    paired_results: dict[str, Any] = {}

    for pid in sorted(parents):
        conds = parents[pid]
        for cond_a, cond_b in PAIRS:
            a_runs = conds.get(cond_a, [])
            b_runs = conds.get(cond_b, [])
            pair_key = f"{pid}/{cond_a}_vs_{cond_b}"
            result: dict[str, Any] = {"parent_id": pid, "condition_a": cond_a, "condition_b": cond_b}

            for field in OUTCOME_FIELDS:
                a_vals = [r.get(field) for r in a_runs if field in r]
                b_vals = [r.get(field) for r in b_runs if field in r]
                if a_vals and b_vals:
                    a_rate = sum(1 for v in a_vals if v) / len(a_vals) if a_vals else None
                    b_rate = sum(1 for v in b_vals if v) / len(b_vals) if b_vals else None
                    result[f"{field}_a"] = a_rate
                    result[f"{field}_b"] = b_rate
                    result[f"{field}_n_a"] = len(a_vals)
                    result[f"{field}_n_b"] = len(b_vals)

            paired_results[pair_key] = result
            summary_rows.append({
                "parent_id": pid, "pair": pair_key,
                "n_a": len(a_runs), "n_b": len(b_runs),
            })

    # GO/NO-GO from rules contract
    rules = go_rules.get("rules", go_rules.get("go_no_go_rules", {}))
    go_result: dict[str, Any] = {"schema": "PILOT_GO_NO_GO_V0", "direction_only": True,
                                  "paper_table1_eligible": False, "attack_authorized": False}

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    with open(staging / "PILOT_ATTACK_SUMMARY_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parent_id", "pair", "n_a", "n_b"])
        w.writeheader()
        for row in summary_rows: w.writerow(row)

    (staging / "PILOT_PAIRED_RESULTS_V0.json").write_text(json.dumps(paired_results, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_GO_NO_GO_V0.json").write_text(json.dumps(go_result, indent=2, sort_keys=True) + "\n")
    (staging / "PILOT_MECHANISM_DIAGNOSIS_V0.md").write_text(
        "# Pilot Mechanism Diagnosis\n\nDirectional only. Not paper Table 1 eligible.\n\n"
        "## Per-parent observations\n\n"
        + "\n".join(f"- {pid}: {len(conds)} conditions" for pid, conds in sorted(parents.items()))
        + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)
    print(f"Pilot Analysis: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
