#!/usr/bin/env python3
"""E4B.2: Separability summary + reporting repair.

Reads existing E4B.1 evidence tables, generates:
  - feature_pair_summary: P vs non-P candidate feature diffs
  - feature_rank_by_trace: per-feature single-trace Teacher-P rank
  - teacher_p_rank_v2: competition rank, n_higher, n_equal, is_unique_top1
  - local_maximum corrected summary

No new experiments. Analysis only.
"""

import argparse, csv, sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--policy-table", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load existing data
    with open(args.candidate_table, "r", newline="") as f:
        cand_rows = list(csv.DictReader(f))
    with open(args.policy_table, "r", newline="") as f:
        policy_rows = list(csv.DictReader(f))

    # ── Teacher-P competition rank ──
    traces = defaultdict(list)
    for r in cand_rows:
        key = (r["task_key"], r["state_id"])
        traces[key].append(r)

    rank_v2_rows = []
    for (task, state), cands in traces.items():
        p_cands = [c for c in cands if c["is_teacher_p"] == "1"]
        if not p_cands:
            continue
        p_step = int(p_cands[0]["candidate_step"])
        p_score = float(p_cands[0]["total_score"])
        n_higher = sum(1 for c in cands if float(c["total_score"]) > p_score + 0.001)
        n_equal = sum(1 for c in cands if abs(float(c["total_score"]) - p_score) < 0.001)
        comp_rank = n_higher + 1
        # Ordinal: sort by score desc, stable — index in sorted list
        sorted_cands = sorted(cands, key=lambda c: (-float(c["total_score"]), int(c["candidate_step"])))
        ord_rank = next(i + 1 for i, c in enumerate(sorted_cands) if int(c["candidate_step"]) == p_step)
        rank_v2_rows.append({
            "task_key": task, "state_id": state,
            "teacher_p_step": p_step, "teacher_p_score": p_score,
            "n_strictly_higher": n_higher,
            "n_equal_to_P": n_equal,
            "competition_rank": comp_rank,
            "ordinal_rank": ord_rank,
            "is_unique_top1": int(n_higher == 0 and n_equal == 1),
            "total_close_candidates": len(cands),
            "max_candidate_score": max(float(c["total_score"]) for c in cands),
        })

    with open(out / "l12_e4b_teacher_p_rank_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_v2_rows[0].keys()))
        w.writeheader(); w.writerows(rank_v2_rows)

    n_unique_top1 = sum(1 for r in rank_v2_rows if r["is_unique_top1"])
    n_comp_top2 = sum(1 for r in rank_v2_rows if r["competition_rank"] <= 2)
    print(f"Teacher-P unique top-1: {n_unique_top1}/{len(rank_v2_rows)}")
    print(f"Teacher-P competition top-2: {n_comp_top2}/{len(rank_v2_rows)}")

    # ── Feature pair summary ──
    numeric_features = [
        "total_score", "raw_crossing_bonus", "close_streak_bonus",
        "close_onset_qpos_bonus", "eef_deceleration_bonus", "qpos_ready_bonus",
        "eef_speed_now", "close_streak", "raw_crossing",
    ]
    pair_rows = []
    for feat in numeric_features:
        diffs = []
        p_vals = []
        non_p_vals = []
        n_traces_p_better = 0
        n_traces_nonp_better = 0
        n_missing = 0
        n_tie = 0

        for (task, state), cands in traces.items():
            p_cands = [c for c in cands if c["is_teacher_p"] == "1"]
            if not p_cands:
                continue
            p_c = p_cands[0]
            non_p = [c for c in cands if c["is_teacher_p"] != "1"]
            p_val_str = p_c.get(feat, "")
            if p_val_str == "" or p_val_str is None:
                n_missing += 1
                continue
            try:
                p_val = float(p_val_str)
            except (ValueError, TypeError):
                n_missing += 1
                continue

            non_p_vals_list = []
            for npc in non_p:
                nv = npc.get(feat, "")
                if nv == "" or nv is None:
                    continue
                try:
                    non_p_vals_list.append(float(nv))
                except (ValueError, TypeError):
                    continue

            if not non_p_vals_list:
                n_missing += 1
                continue

            p_vals.append(p_val)
            non_p_vals.append(sum(non_p_vals_list) / len(non_p_vals_list))
            max_non_p = max(non_p_vals_list)
            if p_val > max_non_p + 0.001:
                n_traces_p_better += 1
            elif abs(p_val - max_non_p) < 0.001:
                n_tie += 1
            else:
                n_traces_nonp_better += 1
            diffs.append(p_val - max_non_p)

        n_valid = len(diffs)
        if n_valid == 0:
            continue
        avg_diff = sum(diffs) / n_valid
        pair_rows.append({
            "feature": feat,
            "n_valid_traces": n_valid,
            "n_missing_or_empty": n_missing,
            "n_P_better_than_best_nonP": n_traces_p_better,
            "n_tie_with_best_nonP": n_tie,
            "n_nonP_better_than_P": n_traces_nonp_better,
            "avg_P_minus_best_nonP_diff": round(avg_diff, 4),
            "P_better_rate": round(n_traces_p_better / n_valid, 3) if n_valid else 0,
        })

    with open(out / "l12_e4b_feature_pair_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)

    print("\nFeature P-vs-nonP discriminability:")
    for r in sorted(pair_rows, key=lambda x: -x["P_better_rate"]):
        print(f"  {r['feature']:30s} P_better={r['n_P_better_than_best_nonP']}/{r['n_valid_traces']} tie={r['n_tie_with_best_nonP']} nonP_better={r['n_nonP_better_than_P']} avg_diff={r['avg_P_minus_best_nonP_diff']}")

    # ── Feature rank by trace ──
    rank_fields = ["total_score", "close_streak", "raw_crossing",
                   "close_onset_qpos_bonus", "eef_deceleration_bonus", "qpos_ready_bonus"]
    rank_by_trace = []
    for (task, state), cands in traces.items():
        p_cands = [c for c in cands if c["is_teacher_p"] == "1"]
        if not p_cands:
            continue
        p_step = int(p_cands[0]["candidate_step"])
        for feat in rank_fields:
            vals = []
            for c in cands:
                v = c.get(feat, "")
                if v == "" or v is None:
                    continue
                try:
                    vals.append((int(c["candidate_step"]), float(v)))
                except (ValueError, TypeError):
                    continue
            if not vals:
                continue
            # Higher score is "better" — ascending rank for other features
            p_val = next((v for s, v in vals if s == p_step), None)
            if p_val is None:
                continue
            n_higher = sum(1 for s, v in vals if v > p_val + 0.001)
            n_equal = sum(1 for s, v in vals if abs(v - p_val) < 0.001)
            rank_by_trace.append({
                "task_key": task, "state_id": state,
                "feature": feat,
                "teacher_p_value": round(p_val, 4),
                "n_higher": n_higher,
                "n_equal": n_equal,
                "competition_rank": n_higher + 1,
                "n_candidates_with_feature": len(vals),
                "max_value": round(max(v for _, v in vals), 4),
                "min_value": round(min(v for _, v in vals), 4),
            })

    with open(out / "l12_e4b_feature_rank_by_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_by_trace[0].keys()))
        w.writeheader(); w.writerows(rank_by_trace)

    # ── Local-maximum corrected summary ──
    lm_rows = [r for r in policy_rows if r["policy"] == "local_maximum" and r["teacher_p_available"] == "True"]
    n_total = len(lm_rows)
    n_causal = sum(1 for r in lm_rows if r["causal"] == "True")
    n_near = sum(1 for r in lm_rows if r["online_is_near_P"] == "1")
    n_no_decision = n_total - n_causal
    delays = [int(r["actual_delay"]) for r in lm_rows if r["causal"] == "True" and r["actual_delay"] not in ("", None)]
    print(f"\nLocal-maximum corrected:")
    print(f"  coverage: {n_causal}/{n_total} emitted decisions")
    print(f"  overall near-correct: {n_near}/{n_total}")
    print(f"  conditional near-correct: {n_near}/{n_causal}" if n_causal > 0 else "  conditional: N/A")
    print(f"  no-decision: {n_no_decision}/{n_total}")
    print(f"  avg delay (emitted): {sum(delays)/len(delays):.1f}" if delays else "  avg delay: N/A")

    print(f"\nOutput: {out}")
    print("E4B.2 COMPLETE")


if __name__ == "__main__":
    main()
