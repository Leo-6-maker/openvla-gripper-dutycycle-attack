#!/usr/bin/env python3
"""Analyze D5 trigger timing against Layer 3 attack windows.

Reads L12 timing handoff v2 and Layer 3 attack-window table.
Computes: direct hit, nonnegative fixed-delay hit, First-CLOSE,
oracle coverage, macro-task metrics, leave-one-task-out stability.

Rules:
  - Fixed delay Δ >= 0 only (cannot attack before emit)
  - Output: DIRECT_TRIGGER_CANDIDATE, FIXED_DELAY_CANDIDATE,
    ARMED_WINDOW_CANDIDATE, UNRELIABLE_TIMING
  - Never changes tau or deployment delay.
"""
import argparse, csv, math, sys
from collections import defaultdict


def load_handoff(path):
    return list(csv.DictReader(open(path)))


def load_attack_windows(path):
    """Load Layer3 attack-window table.
    Expected columns: task, state_id, attack_start, attack_end, condition, ...
    """
    return list(csv.DictReader(open(path)))


def compute_hit(d5_emit, delay, attack_start, attack_end):
    """Check if d5_emit + delay falls within attack window."""
    t = int(d5_emit) + delay
    return int(attack_start) <= t <= int(attack_end)


def compute_direct_hit(d5_emit, attack_start, attack_end):
    return compute_hit(d5_emit, 0, attack_start, attack_end)


def compute_oracle_hit(first_close, attack_start, attack_end):
    """Check if First-CLOSE step falls in attack window."""
    if first_close < 0:
        return False
    return int(attack_start) <= int(first_close) <= int(attack_end)


def best_delay(d5_emit, attack_start, attack_end, delay_max=20):
    """Find smallest nonnegative delay that hits the window, or None."""
    for d in range(0, delay_max + 1):
        if compute_hit(d5_emit, d, attack_start, attack_end):
            return d
    return None


def evaluate(handoff_rows, attack_rows, delay_min=0, delay_max=20):
    """Compute full alignment statistics."""
    # Join handoff and attack on (task, state_id)
    attack_map = {}
    for a in attack_rows:
        key = (a["task"], int(a["state_id"]))
        attack_map[key] = a

    results = []
    per_task = defaultdict(lambda: {"n": 0, "direct": 0, "fixed": 0, "oracle": 0,
                                     "best_delays": [], "d5_emits": [],
                                     "first_closes": []})

    for h in handoff_rows:
        key = (h["task"], int(h["state_id"]))
        atk = attack_map.get(key)
        if atk is None:
            continue

        d5_emit = int(h.get("d5_emit", -1))
        first_close = int(h.get("first_close_step", -1))
        attack_start = int(atk.get("attack_start", atk.get("window_start", -1)))
        attack_end = int(atk.get("attack_end", atk.get("window_end", -1)))
        condition = atk.get("condition", "unknown")

        if attack_start < 0 or attack_end < 0:
            continue

        direct = compute_direct_hit(d5_emit, attack_start, attack_end)
        oracle = compute_oracle_hit(first_close, attack_start, attack_end)
        best = best_delay(d5_emit, attack_start, attack_end, delay_max)
        fixed_hit = best is not None

        r = {
            "task": h["task"], "state_id": h["state_id"],
            "d5_emit": d5_emit, "first_close": first_close,
            "attack_start": attack_start, "attack_end": attack_end,
            "condition": condition,
            "direct_hit": direct,
            "fixed_delay_hit": fixed_hit,
            "best_delay": best if best is not None else -1,
            "oracle_hit": oracle,
        }
        results.append(r)

        t = h["task"]
        per_task[t]["n"] += 1
        if direct:
            per_task[t]["direct"] += 1
        if fixed_hit:
            per_task[t]["fixed"] += 1
            per_task[t]["best_delays"].append(best)
        if oracle:
            per_task[t]["oracle"] += 1
        per_task[t]["d5_emits"].append(d5_emit)
        per_task[t]["first_closes"].append(first_close)

    # Macro metrics
    n_total = len(results)
    n_direct = sum(1 for r in results if r["direct_hit"])
    n_fixed = sum(1 for r in results if r["fixed_delay_hit"])
    n_oracle = sum(1 for r in results if r["oracle_hit"])

    # Task-level
    task_metrics = {}
    for task, s in sorted(per_task.items()):
        task_metrics[task] = {
            "n": s["n"],
            "direct_rate": s["direct"] / s["n"] if s["n"] > 0 else 0,
            "fixed_rate": s["fixed"] / s["n"] if s["n"] > 0 else 0,
            "oracle_rate": s["oracle"] / s["n"] if s["n"] > 0 else 0,
            "best_delays": s["best_delays"],
            "mean_best_delay": sum(s["best_delays"]) / len(s["best_delays"]) if s["best_delays"] else -1,
        }

    # Leave-one-task-out delay stability
    all_best_delays = [r["best_delay"] for r in results if r["best_delay"] >= 0]
    loto_std = -1.0
    if len(task_metrics) >= 3 and all_best_delays:
        per_task_best = {}
        for task, tm in task_metrics.items():
            if tm["best_delays"]:
                per_task_best[task] = max(set(tm["best_delays"]), key=tm["best_delays"].count)
        if len(per_task_best) >= 2:
            vals = list(per_task_best.values())
            mean_v = sum(vals) / len(vals)
            loto_std = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals))

    # Recommendations
    if n_direct / max(n_total, 1) >= 0.5:
        global_rec = "DIRECT_TRIGGER_CANDIDATE"
    elif n_fixed / max(n_total, 1) >= 0.5 and loto_std >= 0 and loto_std <= 5:
        global_rec = "FIXED_DELAY_CANDIDATE"
    elif n_fixed / max(n_total, 1) >= 0.3:
        global_rec = "ARMED_WINDOW_CANDIDATE"
    else:
        global_rec = "UNRELIABLE_TIMING"

    summary = {
        "n_handoff": len(handoff_rows),
        "n_matched": n_total,
        "n_direct": n_direct,
        "direct_rate": n_direct / max(n_total, 1),
        "n_fixed": n_fixed,
        "fixed_rate": n_fixed / max(n_total, 1),
        "n_oracle": n_oracle,
        "oracle_rate": n_oracle / max(n_total, 1),
        "loto_delay_std": round(loto_std, 2),
        "global_recommendation": global_rec,
        "task_metrics": task_metrics,
    }

    return results, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, help="L12 timing handoff v2 CSV")
    ap.add_argument("--attack-windows", default="", help="Layer 3 attack-window CSV")
    ap.add_argument("--delay-min", type=int, default=0)
    ap.add_argument("--delay-max", type=int, default=20)
    ap.add_argument("--output", default="", help="Output CSV")
    args = ap.parse_args()

    handoff = load_handoff(args.handoff)
    print("Handoff: " + str(len(handoff)) + " entries")

    if not args.attack_windows:
        print("No attack-window table provided. Running synthetic self-test.")
        # Self-test: create synthetic attack windows
        attack = []
        for h in handoff:
            d5 = int(h.get("d5_emit", -1))
            ws = int(h.get("teacher_ws", 0))
            we = int(h.get("teacher_we", 0))
            if d5 >= 0 and ws > 0:
                attack.append({
                    "task": h["task"], "state_id": h["state_id"],
                    "attack_start": str(ws), "attack_end": str(we),
                    "condition": "teacher_p_window",
                })
        print("Self-test: using Teacher-P windows as Layer3 attack windows")
    else:
        attack = load_attack_windows(args.attack_windows)
        print("Attack windows: " + str(len(attack)) + " entries")

    if not attack:
        print("ERROR: no attack windows available")
        return 1

    results, summary = evaluate(handoff, attack, args.delay_min, args.delay_max)

    print("\n=== Alignment Summary ===")
    print("Matched parents: " + str(summary["n_matched"]))
    print("Direct hit:       {:.1f}% ({}/{})".format(
        summary["direct_rate"] * 100, summary["n_direct"], summary["n_matched"]))
    print("Fixed-delay hit:  {:.1f}% ({}/{})".format(
        summary["fixed_rate"] * 100, summary["n_fixed"], summary["n_matched"]))
    print("Oracle hit:       {:.1f}% ({}/{})".format(
        summary["oracle_rate"] * 100, summary["n_oracle"], summary["n_matched"]))
    if summary["loto_delay_std"] >= 0:
        print("LOTO delay std:   {:.1f}".format(summary["loto_delay_std"]))
    print("Recommendation:   " + summary["global_recommendation"])

    print("\n=== Per-Task ===")
    for task in sorted(summary["task_metrics"]):
        tm = summary["task_metrics"][task]
        print("  {}: n={} direct={:.0f}% fixed={:.0f}% delays={}".format(
            task, tm["n"],
            tm["direct_rate"] * 100, tm["fixed_rate"] * 100,
            sorted(set(tm["best_delays"])) if tm["best_delays"] else "N/A"))

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "task", "state_id", "d5_emit", "first_close",
                "attack_start", "attack_end", "condition",
                "direct_hit", "fixed_delay_hit", "best_delay", "oracle_hit"])
            w.writeheader()
            w.writerows(results)
        print("\nOutput: " + args.output)

    print("\nNOTE: Do NOT tune tau or deployment delay based on this analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
