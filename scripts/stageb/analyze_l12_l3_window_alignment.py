#!/usr/bin/env python3
"""Analyze D5 trigger timing against Layer3 attack windows.

Accepts future Layer3 attack-window tables and computes:
  - direct D5 hit rate
  - D5 + fixed delay hit rate
  - per-task offset statistics
  - First-CLOSE comparison
  - oracle comparison

Outputs recommendations only — never changes tau or deployment delay.
"""
import argparse, csv, sys
from collections import defaultdict


def load_d5_handoff(path):
    """Load timing handoff CSV, return list of dicts."""
    return list(csv.DictReader(open(path)))


def compute_offset_stats(rows):
    """Compute per-task offset statistics."""
    tasks = defaultdict(list)
    for r in rows:
        offset = int(r["emit_anchor_offset"])
        tasks[r["task"]].append(offset)

    stats = {}
    for task, offsets in tasks.items():
        n = len(offsets)
        mean = sum(offsets) / n
        sorted_offsets = sorted(offsets)
        p50 = sorted_offsets[n // 2]
        stats[task] = {"n": n, "mean": round(mean, 1), "median": p50,
                       "offsets": offsets}
    return stats


def classify_recommendation(task_stats, window_half_width=5):
    """Classify timing quality for each task."""
    recs = {}
    for task, s in task_stats.items():
        offsets = s["offsets"]
        n_in_window = sum(1 for o in offsets if abs(o) <= window_half_width)
        in_window_pct = n_in_window / len(offsets) * 100

        if in_window_pct >= 80:
            recs[task] = "DIRECT_TRIGGER_CANDIDATE"
        elif abs(s["median"]) <= window_half_width * 2:
            recs[task] = "FIXED_DELAY_CANDIDATE"
        elif any(abs(o) <= window_half_width * 3 for o in offsets):
            recs[task] = "ARMED_WINDOW_CANDIDATE"
        else:
            recs[task] = "UNRELIABLE_TIMING"

    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, help="Timing handoff CSV")
    ap.add_argument("--output", default="", help="Output recommendations CSV")
    args = ap.parse_args()

    rows = load_d5_handoff(args.handoff)
    if not rows:
        print("ERROR: empty handoff", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} handoff entries")

    # Per-task offset stats
    stats = compute_offset_stats(rows)
    recs = classify_recommendation(stats)

    # Summary
    n_direct = sum(1 for v in recs.values() if v == "DIRECT_TRIGGER_CANDIDATE")
    n_fixed = sum(1 for v in recs.values() if v == "FIXED_DELAY_CANDIDATE")
    n_armed = sum(1 for v in recs.values() if v == "ARMED_WINDOW_CANDIDATE")
    n_unreliable = sum(1 for v in recs.values() if v == "UNRELIABLE_TIMING")

    print(f"\n=== Timing Quality Assessment ===")
    print(f"  DIRECT_TRIGGER:  {n_direct} tasks")
    print(f"  FIXED_DELAY:     {n_fixed} tasks")
    print(f"  ARMED_WINDOW:    {n_armed} tasks")
    print(f"  UNRELIABLE:      {n_unreliable} tasks")

    print(f"\n=== Per-Task ===")
    for task in sorted(stats):
        s = stats[task]
        rec = recs[task]
        offsets_str = ", ".join(str(o) for o in s["offsets"])
        print(f"  {task}: n={s['n']} mean={s['mean']} median={s['median']} rec={rec}")
        print(f"    offsets: [{offsets_str}]")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "n", "mean_offset", "median_offset",
                                              "offsets", "recommendation"])
            w.writeheader()
            for task in sorted(stats):
                s = stats[task]
                w.writerow({
                    "task": task, "n": s["n"],
                    "mean_offset": s["mean"], "median_offset": s["median"],
                    "offsets": ",".join(str(o) for o in s["offsets"]),
                    "recommendation": recs[task],
                })
        print(f"\nOutput: {args.output}")

    print("\nNOTE: Recommendations are based on limited internal sample.")
    print("Do NOT tune tau or deployment delay based on this analysis.")
    print("External validation on untouched states required before any deployment decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
