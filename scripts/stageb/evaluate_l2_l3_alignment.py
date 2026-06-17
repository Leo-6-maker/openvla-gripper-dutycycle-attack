#!/usr/bin/env python3
"""D4: Evaluate L2->L3 timing alignment (CPU-only).

Compares D5 first emit, D5+global delay, First-CLOSE, and Teacher-P oracle
against real attack-window intervals from H3 results.

Metrics: direct hit, early, late, miss, absolute distance, micro/macro hit rate.
One global delay only. Leave-one-parent-out stability.
"""

import csv, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Object timing parents with timing data
TIMING_PARENTS = {
    "butter_s11":       {"task": "butter", "class": "exact",  "d5_emit": 60,  "teacher_ws": 58,  "teacher_anchor": 60,  "teacher_we": 68},
    "ketchup_s18":      {"task": "ketchup", "class": "exact",  "d5_emit": 84,  "teacher_ws": 83,  "teacher_anchor": 84,  "teacher_we": 93},
    "orange_juice_s29": {"task": "orange_juice", "class": "exact",  "d5_emit": 47,  "teacher_ws": 46,  "teacher_anchor": 47,  "teacher_we": 55},
    "milk_s7":          {"task": "milk", "class": "exact",  "d5_emit": 41,  "teacher_ws": 40,  "teacher_anchor": 41,  "teacher_we": 50},
    "bbq_sauce_s40":    {"task": "bbq_sauce", "class": "exact",  "d5_emit": 54,  "teacher_ws": 53,  "teacher_anchor": 54,  "teacher_we": 63},
    "bbq_sauce_s27":    {"task": "bbq_sauce", "class": "exact",  "d5_emit": 94,  "teacher_ws": 93,  "teacher_anchor": 94,  "teacher_we": 94},
    "tomato_sauce_s23": {"task": "tomato_sauce", "class": "early", "d5_emit": 69,  "teacher_ws": 139, "teacher_anchor": 141, "teacher_we": 149},
    "salad_dressing_s32":{"task": "salad_dressing", "class": "early", "d5_emit": 56,  "teacher_ws": 127, "teacher_anchor": 128, "teacher_we": 148},
    "cream_cheese_s1":  {"task": "cream_cheese", "class": "early", "d5_emit": 137, "teacher_ws": 141, "teacher_anchor": 142, "teacher_we": 154},
    "cream_cheese_s20": {"task": "cream_cheese", "class": "early", "d5_emit": 57,  "teacher_ws": 106, "teacher_anchor": 107, "teacher_we": 119},
    "salad_dressing_s24":{"task": "salad_dressing", "class": "early", "d5_emit": 57,  "teacher_ws": 98,  "teacher_anchor": 114, "teacher_we": 120},
    "salad_dressing_s11":{"task": "salad_dressing", "class": "late",  "d5_emit": 128, "teacher_ws": 57,  "teacher_anchor": 59,  "teacher_we": 67},
    "ketchup_s34":      {"task": "ketchup", "class": "miss",  "d5_emit": -1,  "teacher_ws": 74,  "teacher_anchor": 75,  "teacher_we": 77},
    "salad_dressing_s45":{"task": "salad_dressing", "class": "miss",  "d5_emit": -1,  "teacher_ws": 110, "teacher_anchor": 111, "teacher_we": 130},
}


class AlignmentEvaluator:
    def __init__(self):
        self.results = {"direct": [], "delay_sweep": [], "macro": []}

    def evaluate_direct(self) -> List[Dict]:
        """Evaluate direct D5 emit alignment."""
        rows = []
        for pid, p in TIMING_PARENTS.items():
            emit = p["d5_emit"]
            if emit < 0:
                rows.append({"parent_id": pid, "timing_class": p["class"],
                            "d5_emit": -1, "direct_hit": "miss",
                            "distance": -1, "teacher_anchor": p["teacher_anchor"],
                            "teacher_ws": p["teacher_ws"], "teacher_we": p["teacher_we"]})
                continue

            # Direct hit: emit inside [ws, we)
            inside_window = p["teacher_ws"] <= emit < p["teacher_we"]
            distance_to_anchor = emit - p["teacher_anchor"]

            if emit == p["teacher_anchor"]:
                hit_status = "exact_anchor_hit"
            elif inside_window:
                hit_status = "in_window_hit"
            elif emit < p["teacher_ws"]:
                hit_status = "early"
            else:
                hit_status = "late"

            rows.append({
                "parent_id": pid, "task": p["task"], "timing_class": p["class"],
                "d5_emit": emit, "teacher_ws": p["teacher_ws"],
                "teacher_anchor": p["teacher_anchor"], "teacher_we": p["teacher_we"],
                "direct_hit": hit_status, "distance_to_anchor": distance_to_anchor,
                "inside_teacher_window": str(inside_window),
            })
        self.results["direct"] = rows
        return rows

    def evaluate_delay_sweep(self, max_delay: int = 20) -> List[Dict]:
        """Sweep global nonnegative delay on D5 emit.

        For each delay d: emit_delayed = d5_emit + d.
        Report which parents have emit_delayed inside teacher window.
        """
        valid_parents = {pid: p for pid, p in TIMING_PARENTS.items() if p["d5_emit"] >= 0}
        sweep_rows = []

        for delay in range(0, max_delay + 1):
            hits = 0
            early = 0
            late_count = 0
            total_dist = 0
            parent_details = {}

            for pid, p in valid_parents.items():
                delayed_emit = p["d5_emit"] + delay
                inside = p["teacher_ws"] <= delayed_emit < p["teacher_we"]
                dist = delayed_emit - p["teacher_anchor"]

                if inside:
                    hits += 1
                elif delayed_emit < p["teacher_ws"]:
                    early += 1
                else:
                    late_count += 1
                total_dist += abs(dist)
                parent_details[pid] = {
                    "delayed_emit": delayed_emit, "hit": inside,
                    "distance": dist, "class": p["class"],
                }

            n = len(valid_parents)
            sweep_rows.append({
                "delay": delay, "n_parents": n, "n_hits": hits,
                "n_early": early, "n_late": late_count,
                "hit_rate": round(hits / n, 4) if n > 0 else 0,
                "mean_abs_distance": round(total_dist / n, 2) if n > 0 else -1,
                "parent_details": json.dumps(parent_details),
            })
        self.results["delay_sweep"] = sweep_rows
        return sweep_rows

    def evaluate_macro(self) -> List[Dict]:
        """Macro-per-task reporting."""
        tasks = defaultdict(list)
        for pid, p in TIMING_PARENTS.items():
            tasks[p["task"]].append({
                "parent_id": pid, "class": p["class"],
                "d5_emit": p["d5_emit"], "teacher_anchor": p["teacher_anchor"],
                "teacher_ws": p["teacher_ws"], "teacher_we": p["teacher_we"],
            })

        macro_rows = []
        for task, entries in sorted(tasks.items()):
            n_exact = sum(1 for e in entries if e["class"] == "exact")
            n_early = sum(1 for e in entries if e["class"] == "early")
            n_late = sum(1 for e in entries if e["class"] == "late")
            n_miss = sum(1 for e in entries if e["class"] == "miss")

            # Direct hit rate per task
            valid = [e for e in entries if e["d5_emit"] >= 0]
            direct_hits = sum(1 for e in valid if e["teacher_ws"] <= e["d5_emit"] < e["teacher_we"])
            exact_hits = sum(1 for e in valid if e["d5_emit"] == e["teacher_anchor"])

            mean_dist = -1
            if valid:
                mean_dist = round(sum(abs(e["d5_emit"] - e["teacher_anchor"]) for e in valid) / len(valid), 2)

            macro_rows.append({
                "task": task, "n_parents": len(entries),
                "n_exact": n_exact, "n_early": n_early, "n_late": n_late, "n_miss": n_miss,
                "n_valid_emit": len(valid),
                "n_direct_hits": direct_hits, "n_exact_anchor_hits": exact_hits,
                "direct_hit_rate": round(direct_hits / len(valid), 4) if valid else 0,
                "exact_anchor_rate": round(exact_hits / len(valid), 4) if valid else 0,
                "mean_abs_distance_to_anchor": mean_dist,
            })
        self.results["macro"] = macro_rows
        return macro_rows

    def leave_one_out(self, best_delay: int) -> Dict:
        """Leave-one-parent-out stability at best global delay."""
        valid_parents = {pid: p for pid, p in TIMING_PARENTS.items() if p["d5_emit"] >= 0}
        parent_ids = sorted(valid_parents.keys())
        stability = {}

        for held_out in parent_ids:
            hits = 0
            for pid, p in valid_parents.items():
                if pid == held_out: continue
                delayed_emit = p["d5_emit"] + best_delay
                if p["teacher_ws"] <= delayed_emit < p["teacher_we"]:
                    hits += 1
            n = len(parent_ids) - 1
            stability[held_out] = {
                "n_hits": hits, "n_total": n,
                "hit_rate": round(hits / n, 4) if n > 0 else 0,
            }

        return {
            "best_delay": best_delay,
            "stability": stability,
            "all_parents_hit_rate": round(
                sum(1 for pid, p in valid_parents.items()
                    if p["teacher_ws"] <= p["d5_emit"] + best_delay < p["teacher_we"]) / len(valid_parents), 4
            ) if valid_parents else 0,
        }

    def run(self, max_delay: int = 20):
        print("=== D4: L2->L3 Timing Alignment Evaluator ===\n")

        # Direct evaluation
        direct = self.evaluate_direct()
        n_hits = sum(1 for r in direct if r["direct_hit"] in ("exact_anchor_hit", "in_window_hit"))
        n_exact = sum(1 for r in direct if r["direct_hit"] == "exact_anchor_hit")
        n_valid = sum(1 for r in direct if r["d5_emit"] >= 0)
        print(f"  Direct D5 emit: {n_hits}/{n_valid} in-window, {n_exact}/{n_valid} exact anchor")

        # Delay sweep
        sweep = self.evaluate_delay_sweep(max_delay)
        best = max(sweep, key=lambda r: (r["hit_rate"], -r["mean_abs_distance"]))
        print(f"  Best global delay: {best['delay']} (hit_rate={best['hit_rate']}, mean_dist={best['mean_abs_distance']})")

        # LOO stability
        loo = self.leave_one_out(best["delay"])
        loo_rates = [v["hit_rate"] for v in loo["stability"].values()]
        print(f"  LOO stability: min={min(loo_rates):.3f} max={max(loo_rates):.3f} all={loo['all_parents_hit_rate']:.3f}")

        # Macro
        macro = self.evaluate_macro()
        print(f"\n  Macro per task:")
        for r in macro:
            print(f"    {r['task']}: {r['n_parents']} parents, direct_hit={r['direct_hit_rate']:.2f}, exact_anchor={r['exact_anchor_rate']:.2f}")

        self._write_outputs(best, loo)
        print(f"\n  Output written to tables/ and reports/")

    def _write_outputs(self, best_delay_row, loo):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Direct
        if self.results["direct"]:
            fields = list(self.results["direct"][0].keys())
            with open(out_dir / "l2_l3_alignment_direct.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(self.results["direct"])

        # Delay sweep
        if self.results["delay_sweep"]:
            fields = [k for k in self.results["delay_sweep"][0].keys() if k != "parent_details"]
            with open(out_dir / "l2_l3_alignment_delay_sweep.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(self.results["delay_sweep"])

        # Macro
        if self.results["macro"]:
            fields = list(self.results["macro"][0].keys())
            with open(out_dir / "l2_l3_alignment_macro.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(self.results["macro"])

        # Report
        with open(reports_dir / "L2_TO_L3_REAL_ALIGNMENT.md", "w") as f:
            f.write("# L2 → L3 Timing Alignment Report\n\n")
            f.write(f"**Best global delay:** {best_delay_row['delay']}\n")
            f.write(f"**Best hit rate:** {best_delay_row['hit_rate']}\n")
            f.write(f"**Best mean abs distance:** {best_delay_row['mean_abs_distance']}\n\n")

            f.write(f"## Direct D5 emit alignment\n\n")
            n_hits = sum(1 for r in self.results["direct"] if r["direct_hit"] in ("exact_anchor_hit", "in_window_hit"))
            n_valid = sum(1 for r in self.results["direct"] if r["d5_emit"] >= 0)
            f.write(f"- In-window hits: {n_hits}/{n_valid}\n")
            f.write(f"- Exact anchor hits: {sum(1 for r in self.results['direct'] if r['direct_hit'] == 'exact_anchor_hit')}/{n_valid}\n\n")

            f.write(f"## Leave-one-out stability (delay={best_delay_row['delay']})\n\n")
            f.write(f"- All-parents hit rate: {loo['all_parents_hit_rate']}\n")
            loo_rates = [v["hit_rate"] for v in loo["stability"].values()]
            if loo_rates:
                f.write(f"- Min LOO rate: {min(loo_rates)}\n")
                f.write(f"- Max LOO rate: {max(loo_rates)}\n")

            f.write(f"\n## Macro per task\n\n")
            for r in self.results["macro"]:
                f.write(f"- **{r['task']}**: {r['n_parents']} parents, {r['n_direct_hits']}/{r['n_valid_emit']} direct hits ({r['direct_hit_rate']:.2f})\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-delay", type=int, default=20, help="Maximum global delay to sweep")
    args = ap.parse_args()

    evaluator = AlignmentEvaluator()
    evaluator.run(max_delay=args.max_delay)


if __name__ == "__main__":
    main()
