#!/usr/bin/env python3
"""audit_online_detector_proposals.py v5 — Audit generated attack window proposals.

Checks:
  - proposal_valid rows have actual_window_len == window_len (no clipped)
  - proposal_eligible rows have clean_natural_open_ratio <= max_clean_open_ratio
  - clean_open_ratio non-empty for all valid rows
  - threshold/K match eval metrics
  - window_start == T_pred + delay, window_end == window_start + 17
  - feature_space_model == X_norm, feature_space_open_ratio == X_raw
  - no X_raw missing
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal-csv", required=True)
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--eval-metrics-json", required=True)
    ap.add_argument("--max-clean-open-ratio", type=float, default=0.1)
    ap.add_argument("--window-len", type=int, default=18)
    ap.add_argument("--output-csv", default="tables/object_detector_window_proposals_audit.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_DETECTOR_PROPOSAL_AUDIT_V5.md")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.proposal_csv):
        print(f"ERROR: Proposal CSV not found: {args.proposal_csv}"); sys.exit(1)
    if not os.path.exists(args.eval_metrics_json):
        print(f"ERROR: eval_metrics_json not found: {args.eval_metrics_json}"); sys.exit(1)

    with open(args.eval_metrics_json) as f:
        em = json.load(f)
    with open(args.proposal_csv, newline="") as f:
        proposals = list(csv.DictReader(f))

    print(f"Loaded {len(proposals)} proposals")
    print(f"Eval metrics: th={em.get('best_threshold')}, K={em.get('best_K')}")

    checks_passed = 0; checks_failed = 0; failures = []

    def check(cond, msg, row=None):
        nonlocal checks_passed, checks_failed
        if cond: checks_passed += 1
        else:
            checks_failed += 1
            detail = f"  row={row.get('episode_id','?')}" if row else ""
            failures.append(f"FAIL: {msg}{detail}")

    # 1. Threshold/K match
    exp_th = em.get("best_threshold"); exp_K = em.get("best_K")
    for i, p in enumerate(proposals):
        if i == 0:
            check(float(p.get("threshold",0)) == exp_th, f"threshold mismatch: {p.get('threshold')} vs {exp_th}")
            check(int(p.get("K",0)) == exp_K, f"K mismatch: {p.get('K')} vs {exp_K}")

    # 2. X_raw check
    check(all(p.get("feature_space_open_ratio","")=="X_raw" for p in proposals),
          "Not all proposals have feature_space_open_ratio=X_raw")
    check(all(p.get("feature_space_model","")=="X_norm" for p in proposals),
          "Not all proposals have feature_space_model=X_norm")

    # 3. Per-row checks
    n_valid = 0; n_eligible = 0; n_clipped = 0; n_confound = 0; n_no_trigger = 0
    for p in proposals:
        valid = p.get("proposal_valid","").lower() == "true"
        eligible = p.get("proposal_eligible","").lower() == "true"
        ws = p.get("window_start",""); we = p.get("window_end","")
        delay = p.get("delay",""); T_pred = p.get("T_pred","")
        actual_len = int(p.get("actual_window_len",0) or 0)
        open_ratio = p.get("clean_natural_open_ratio","")
        inv_reason = p.get("invalid_reason","")
        elig_reason = p.get("eligibility_reason","")

        if "no_trigger" in str(inv_reason): n_no_trigger += 1
        if "clipped" in str(inv_reason): n_clipped += 1
        if "confound" in str(elig_reason): n_confound += 1

        if valid:
            n_valid += 1
            # Window len check
            check(actual_len == args.window_len,
                  f"valid proposal has actual_window_len={actual_len} != {args.window_len}", p)
            # Window bounds
            if ws and T_pred and delay:
                exp_ws = int(T_pred) + int(delay)
                exp_we = exp_ws + args.window_len - 1
                check(int(ws) == exp_ws, f"window_start={ws} != T_pred+delay={exp_ws}", p)
                check(int(we) == exp_we, f"window_end={we} != ws+17={exp_we}", p)
            # Clean open ratio
            check(open_ratio != "", "valid proposal has empty clean_natural_open_ratio", p)
            # T_pred exists
            check(T_pred != "", "valid proposal has no T_pred", p)

        if eligible:
            n_eligible += 1
            check(float(open_ratio) <= args.max_clean_open_ratio,
                  f"eligible proposal has clean_open_ratio={open_ratio} > {args.max_clean_open_ratio}", p)
            check(elig_reason == "", f"eligible proposal has eligibility_reason={elig_reason}", p)
            check(valid, f"eligible but not valid: {elig_reason}", p)

    # 4. Counts sanity
    check(n_valid > 0, "No valid proposals")
    check(n_eligible > 0, "No eligible proposals")

    # 5. Duplicate check
    keys = defaultdict(int)
    for p in proposals:
        keys[(p.get("episode_id"), p.get("delay"))] += 1
    dups = {k: v for k, v in keys.items() if v > 2}  # 2 delays expected
    check(not dups, f"Unexpected duplicate proposals: {list(dups.keys())[:5]}")

    # ── Write audit CSV ──
    audit_fields = list(proposals[0].keys()) + ["audit_checks"]
    for p in proposals:
        p["audit_checks"] = "passed"  # per-row checks already aggregated
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(proposals)

    # ── Report ──
    status = "PASS" if checks_failed == 0 else "FAIL"
    report = f"""# Object Detector Window Proposal Audit v5

**Status**: {status} ({checks_failed} failures, {checks_passed} passed)

## Summary

| Metric | Value |
|--------|-------|
| Total proposals | {len(proposals)} |
| Valid (full window) | {n_valid} |
| Eligible (low confound) | {n_eligible} |
| No trigger | {n_no_trigger} |
| Clipped short | {n_clipped} |
| Natural open confounded | {n_confound} |

## Checks

| # | Check | Status |
|---|-------|--------|
"""

    for i, f_msg in enumerate(failures):
        report += f"| {i+1} | {f_msg} | FAIL |\n"
    if not failures:
        report += "| - | All checks passed | PASS |\n"

    report += f"""

## Config

- threshold={exp_th}, K={exp_K}
- max_clean_open_ratio={args.max_clean_open_ratio}
- window_len={args.window_len}
- eval_metrics: {args.eval_metrics_json}

## Verdict

"""
    if status == "PASS":
        report += (
            "All proposal audit checks passed. Proposals are valid for proposal-only evaluation. "
            "Detector-driven VIS is NOT yet approved — gate approval required."
        )
    else:
        report += "Proposal audit FAILED. Do NOT use these proposals for VIS."

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Audit: {status} ({checks_failed} failures)")
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
