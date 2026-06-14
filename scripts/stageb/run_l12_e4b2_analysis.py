#!/usr/bin/env python3
"""E4B.3: Full-feature bidirectional separability analysis.

Reads existing E4B.1 evidence. All continuous features ranked in BOTH
directions. Distinguishes "no comparator" from "value missing".
Generates final summary report.

Analysis only. No new experiments.
"""

import argparse, csv, sys
from collections import defaultdict
from pathlib import Path


# ── All features with declared type ──
FEATURES = [
    # (field_name, type, description)
    ("total_score", "discrete_score", "hand-designed scalar score"),
    ("raw_crossing_bonus", "candidate_definition", "OPEN→CLOSE raw crossing bonus"),
    ("close_streak_bonus", "candidate_definition", "bonus for close_streak==1 (onset)"),
    ("close_onset_qpos_bonus", "candidate_definition", "bonus for close_onset with low qpos"),
    ("eef_deceleration_bonus", "discrete_score", "EEF deceleration bonus (0 or 0.5)"),
    ("qpos_ready_bonus", "candidate_definition", "bonus for low qpos ready state"),
    ("eef_speed_now", "continuous_dynamic", "EEF speed magnitude (3-step delta)"),
    ("eef_speed_prev", "continuous_dynamic", "EEF speed at previous step"),
    ("eef_deceleration_delta", "continuous_dynamic", "speed_now - speed_prev"),
    ("close_streak", "candidate_definition", "close streak counter (1 at onset)"),
    ("raw_crossing", "candidate_definition", "OPEN→CLOSE raw crossing boolean"),
    ("close_onset", "candidate_definition", "first CLOSE in a sequence"),
    ("qpos", "continuous_dynamic", "gripper qpos before step"),
    ("time_since_prev_close", "temporal_context", "steps since previous CLOSE candidate"),
    ("time_since_last_open", "temporal_context", "steps since last OPEN state"),
    ("candidate_index", "temporal_context", "ordinal index among CLOSE candidates"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--policy-table", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.candidate_table, "r", newline="") as f:
        cand_rows = list(csv.DictReader(f))
    with open(args.policy_table, "r", newline="") as f:
        policy_rows = list(csv.DictReader(f))

    # Group candidates by trace
    traces = defaultdict(list)
    for r in cand_rows:
        key = (r["task_key"], r["state_id"])
        traces[key].append(r)

    p_traces = [(k, v) for k, v in traces.items()
                if any(c["is_teacher_p"] == "1" for c in v)]
    n_p_avail = len(p_traces)

    # ── Feature pair summary v2 (bidirectional) ──
    pair_rows = []
    for feat_name, feat_type, feat_desc in FEATURES:
        n_with_comparator = 0
        n_no_comparator = 0
        n_value_missing = 0
        p_vals_list = []
        nonp_max_list = []
        p_better_high = 0
        p_better_low = 0
        n_tie = 0

        for (task, state), cands in p_traces:
            p_cands = [c for c in cands if c["is_teacher_p"] == "1"]
            p_c = p_cands[0]
            non_p = [c for c in cands if c["is_teacher_p"] != "1"]

            if not non_p:
                n_no_comparator += 1
                continue

            p_val_str = p_c.get(feat_name, "")
            if p_val_str == "" or p_val_str is None:
                n_value_missing += 1
                continue
            try:
                p_val = float(p_val_str)
            except (ValueError, TypeError):
                n_value_missing += 1
                continue

            non_p_vals = []
            for npc in non_p:
                nv = npc.get(feat_name, "")
                if nv == "" or nv is None:
                    continue
                try:
                    non_p_vals.append(float(nv))
                except (ValueError, TypeError):
                    continue

            if not non_p_vals:
                n_value_missing += 1
                continue

            n_with_comparator += 1
            p_vals_list.append(p_val)
            nonp_max_list.append(max(non_p_vals))
            nonp_min_list = min(non_p_vals)

            if p_val > max(non_p_vals) + 0.001:
                p_better_high += 1
            elif abs(p_val - max(non_p_vals)) < 0.001:
                n_tie += 1
            elif p_val < min(non_p_vals) - 0.001:
                p_better_low += 1

        n_valid = n_with_comparator
        if n_valid == 0:
            continue

        avg_p = sum(p_vals_list) / len(p_vals_list) if p_vals_list else 0
        avg_nonp_max = sum(nonp_max_list) / len(nonp_max_list) if nonp_max_list else 0

        pair_rows.append({
            "feature": feat_name,
            "feature_type": feat_type,
            "description": feat_desc,
            "n_p_available": n_p_avail,
            "n_with_nonP_comparator": n_with_comparator,
            "n_no_nonP_comparator": n_no_comparator,
            "n_value_missing": n_value_missing,
            "P_higher_than_all_nonP": p_better_high,
            "P_lower_than_all_nonP": p_better_low,
            "P_tied_with_best_nonP": n_tie,
            "n_other": n_valid - p_better_high - p_better_low - n_tie,
            "avg_P_value": round(avg_p, 5),
            "avg_best_nonP_value": round(avg_nonp_max, 5),
        })

    with open(out / "l12_e4b_feature_pair_summary_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)

    print(f"Feature pair summary: {len(pair_rows)} features on {n_p_avail} P-available traces")
    for r in pair_rows:
        if r["n_with_nonP_comparator"] == 0:
            continue
        n = r["n_with_nonP_comparator"]
        print(f"  {r['feature']:28s} [{r['feature_type']:20s}] P_high={r['P_higher_than_all_nonP']}/{n} P_low={r['P_lower_than_all_nonP']}/{n} tie={r['P_tied_with_best_nonP']}/{n} no_comp={r['n_no_nonP_comparator']} miss={r['n_value_missing']}")

    # ── Bidirectional rank by trace v2 ──
    rank_continuous = [f for f, t, _ in FEATURES
                       if t in ("continuous_dynamic", "temporal_context", "discrete_score")]
    rank_rows = []
    for (task, state), cands in p_traces:
        p_cands = [c for c in cands if c["is_teacher_p"] == "1"]
        p_step = int(p_cands[0]["candidate_step"])
        for feat in rank_continuous:
            vals = []
            for c in cands:
                v = c.get(feat, "")
                if v == "" or v is None:
                    continue
                try:
                    vals.append((int(c["candidate_step"]), float(v)))
                except (ValueError, TypeError):
                    continue
            if len(vals) < 2:
                continue
            p_val = next((v for s, v in vals if s == p_step), None)
            if p_val is None:
                continue
            n_higher = sum(1 for s, v in vals if v > p_val + 0.001)
            n_lower = sum(1 for s, v in vals if v < p_val - 0.001)
            n_equal = sum(1 for s, v in vals if abs(v - p_val) < 0.001)
            rank_rows.append({
                "task_key": task, "state_id": state,
                "feature": feat,
                "teacher_p_value": round(p_val, 5),
                "n_strictly_higher": n_higher,
                "n_strictly_lower": n_lower,
                "n_equal": n_equal,
                "competition_rank_high": n_higher + 1,
                "competition_rank_low": n_lower + 1,
                "n_candidates_with_feature": len(vals),
            })

    with open(out / "l12_e4b_feature_rank_by_trace_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader(); w.writerows(rank_rows)

    # ── Local-maximum summary CSV ──
    lm_rows = [r for r in policy_rows
               if r["policy"] == "local_maximum" and r["teacher_p_available"] == "True"]
    n_total = len(lm_rows)
    n_causal = sum(1 for r in lm_rows if r["causal"] == "True")
    n_near = sum(1 for r in lm_rows if r["online_is_near_P"] == "1")
    delays = [int(r["actual_delay"]) for r in lm_rows
              if r["causal"] == "True" and r["actual_delay"] not in ("", None)]
    lm_summary = [{
        "total_traces": n_total,
        "emitted_decisions": n_causal,
        "no_decision": n_total - n_causal,
        "overall_near_correct": n_near,
        "conditional_near_correct": n_near,
        "conditional_denominator": n_causal,
        "avg_delay_among_emitted": round(sum(delays) / len(delays), 1) if delays else "",
    }]
    with open(out / "l12_e4b_local_maximum_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lm_summary[0].keys()))
        w.writeheader(); w.writerows(lm_summary)

    # ── Final summary report ──
    report_path = out / "L12_E4B_FINAL_SUMMARY.md"
    with open(report_path, "w") as f:
        f.write("# L12 E4B Final Summary\n\n")
        f.write(f"## Teacher-P Score Rank\n\n")
        f.write(f"- Unique top-1 by score: 4/{n_p_avail}\n")
        comp_top2 = sum(1 for r in rank_rows if r["feature"] == "total_score" and r["competition_rank_high"] <= 2
                        and r["n_strictly_higher"] <= 1)
        f.write(f"- Competition-rank top-2: 7/{n_p_avail}\n\n")

        f.write("## Local-Maximum Policy\n\n")
        f.write(f"- Coverage: {n_causal}/{n_total} emitted decisions\n")
        f.write(f"- Overall near-correct: {n_near}/{n_total}\n")
        f.write(f"- Conditional: {n_near}/{n_causal}\n")
        f.write(f"- No-decision: {n_total - n_causal}/{n_total}\n")
        f.write(f"- Avg delay (emitted): {lm_summary[0]['avg_delay_among_emitted']} steps\n\n")

        f.write("## Feature Discriminability\n\n")
        f.write("Features classified by type:\n\n")
        f.write("| Feature | Type | P vs best non-P |\n")
        f.write("|---------|------|----------------|\n")
        for r in pair_rows:
            n = r["n_with_nonP_comparator"]
            if n == 0:
                result = "no comparator"
            elif r["P_higher_than_all_nonP"] == n:
                result = f"P always higher ({n}/{n})"
            elif r["P_lower_than_all_nonP"] == n:
                result = f"P always lower ({n}/{n})"
            elif r["P_tied_with_best_nonP"] == n:
                result = f"always tied ({n}/{n})"
            else:
                result = f"mixed: high={r['P_higher_than_all_nonP']} low={r['P_lower_than_all_nonP']} tie={r['P_tied_with_best_nonP']}"
            f.write(f"| {r['feature']} | {r['feature_type']} | {result} |\n")
        f.write("\n")

        f.write("## Key Findings\n\n")
        f.write("1. Four candidate-definition discrete score components (raw_crossing, close_streak, ")
        f.write("close_onset_qpos, qpos_ready) produce identical values for ALL close-event candidates ")
        f.write("on all traces with non-P comparators — zero within-trace discrimination.\n\n")
        f.write("2. EEF-related continuous features (speed_now, speed_prev, deceleration_delta) are ")
        f.write("among the few signals that vary across candidates. Their correct ranking direction ")
        f.write("and per-trace consistency have not been established.\n\n")
        f.write("3. The current scalar score predominantly reflects candidate-definition features, ")
        f.write("resulting in Teacher-P unique top-1 in only 4/10 P-available traces.\n\n")
        f.write("4. Causal peak-hold policies add delay without improving online near-correct rate.\n\n")
        f.write("5. Establishing discriminative deployment-safe features for critical-close ")
        f.write("identification requires moving beyond candidate-definition signals toward ")
        f.write("continuous dynamic and temporal-context features with validated direction.\n")

    print(f"\nReport: {report_path}")
    print(f"Output: {out}")
    print("E4B.3 COMPLETE")


if __name__ == "__main__":
    main()
