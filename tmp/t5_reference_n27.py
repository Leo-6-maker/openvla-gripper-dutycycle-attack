#!/usr/bin/env python3
"""T5: Reference N=27 statistics — Wilson CI, paired McNemar, cell/task macro."""
import os, json, math

BASE = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2"

# 9 reference cells
CELLS = [
    ("alphabet_soup", 0, 0), ("bbq_sauce", 1, 0), ("butter", 2, 0),
    ("butter", 3, 2), ("ketchup", 4, 0), ("milk", 5, 4),
    ("orange_juice", 6, 0), ("salad_dressing", 7, 0), ("tomato_sauce", 8, 0),
]
TASKS = [
    ("alphabet_soup", 0), ("bbq_sauce", 1), ("butter", 2), ("butter", 3),
    ("ketchup", 4), ("milk", 5), ("orange_juice", 6), ("salad_dressing", 7),
    ("tomato_sauce", 8),
]
# Unique tasks for macro (butter aggregated)
UNIQUE_TASKS = [
    ("alphabet_soup", 0), ("bbq_sauce", 1), ("butter", [2, 3]),
    ("ketchup", 4), ("milk", 5), ("orange_juice", 6),
    ("salad_dressing", 7), ("tomato_sauce", 8),
]

def wilson_ci(n_fail, n_total):
    if n_total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = n_fail / n_total
    denom = 1 + z*z/n_total
    center = (p + z*z/(2*n_total)) / denom
    margin = z * math.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total)) / denom
    return (max(0, center - margin), min(1, center + margin))

def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    p_val = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (0.5 ** n)
        if abs(i - n/2) >= abs(b - n/2):
            p_val += prob
    return p_val

CONDS = ["tma_nolock", "tma_armlock", "prefix_nolock", "prefix_armlock"]

for cond in CONDS:
    cp = os.path.join(BASE, cond)
    runs = {}
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        summ = os.path.join(rp, "episode_summary.json")
        if not os.path.isfile(summ):
            continue
        with open(summ) as f:
            s = json.load(f)
        key = (s.get("task_idx"), s.get("state_id"), s.get("perturbation_seed"))
        runs[key] = {
            "task_success": s.get("task_success"),
            "attack_frames": s.get("attack_frames", 0),
            "mlp_triggered": s.get("mlp_triggered", False),
        }

    n_total = len(runs)
    n_fail = sum(1 for r in runs.values() if r["task_success"] is False)
    n_emit = sum(1 for r in runs.values() if r["attack_frames"] > 0)
    emit_runs = {k: v for k, v in runs.items() if v["attack_frames"] > 0}
    n_cond = len(emit_runs)
    n_cond_fail = sum(1 for r in emit_runs.values() if r["task_success"] is False)

    itt_lo, itt_hi = wilson_ci(n_fail, n_total)
    cond_lo, cond_hi = wilson_ci(n_cond_fail, n_cond) if n_cond > 0 else (0, 0)

    print("=" * 60)
    print("Condition: {}".format(cond))
    print("ITT: {}/{} = {:.1%} [CI: {:.1%}, {:.1%}]".format(
        n_fail, n_total, n_fail/n_total if n_total else 0, itt_lo, itt_hi))
    print("Conditional: {}/{} = {:.1%} [CI: {:.1%}, {:.1%}]".format(
        n_cond_fail, n_cond, n_cond_fail/n_cond if n_cond else 0, cond_lo, cond_hi))
    print("Coverage: {}/{} = {:.1%}".format(n_emit, n_total, n_emit/n_total if n_total else 0))
    print()

    # Per-cell
    print("Per-cell ITT FR:")
    for cell_name, task_idx, state_id in CELLS:
        cell_runs = {k: v for k, v in runs.items() if k[0] == task_idx and k[1] == state_id}
        n = len(cell_runs)
        f = sum(1 for v in cell_runs.values() if v["task_success"] is False)
        e = sum(1 for v in cell_runs.values() if v["attack_frames"] > 0)
        lo, hi = wilson_ci(f, n)
        print("  {:20s}: {}/{} = {:.0%} [CI: {:.0%}, {:.0%}] emit={}/{}".format(
            "{}_s{}".format(cell_name, state_id), f, n, f/n if n else 0, lo, hi, e, n))

    # Task macro (butter aggregated)
    print()
    print("Per-task ITT FR:")
    for task_name, ti in UNIQUE_TASKS:
        if isinstance(ti, list):
            task_runs = {k: v for k, v in runs.items() if k[0] in ti}
        else:
            task_runs = {k: v for k, v in runs.items() if k[0] == ti}
        n = len(task_runs)
        f = sum(1 for v in task_runs.values() if v["task_success"] is False)
        e = sum(1 for v in task_runs.values() if v["attack_frames"] > 0)
        lo, hi = wilson_ci(f, n)
        n_s = len(set(k[1] for k in task_runs))
        print("  {:20s}: {}/{} = {:.0%} [CI: {:.0%}, {:.0%}] emit={}/{} ({} cells)".format(
            task_name, f, n, f/n if n else 0, lo, hi, e, n, n_s))

    # Cell macro
    cell_fails = 0; cell_total = 0
    for cell_name, task_idx, state_id in CELLS:
        cell_runs = {k: v for k, v in runs.items() if k[0] == task_idx and k[1] == state_id}
        cell_fails += sum(1 for v in cell_runs.values() if v["task_success"] is False)
        cell_total += len(cell_runs)
    cl, ch = wilson_ci(cell_fails, cell_total)
    print()
    print("Cell-macro FR: {}/{} = {:.1%} [CI: {:.1%}, {:.1%}]".format(
        cell_fails, cell_total, cell_fails/cell_total if cell_total else 0, cl, ch))
    print()

# Paired McNemar
print("=" * 60)
print("PAIRED MCNEMAR (N=27)")
print("=" * 60)

for name, nl_cond, al_cond in [("TMA", "tma_nolock", "tma_armlock"),
                                ("Prefix", "prefix_nolock", "prefix_armlock")]:
    def get_runs(cond_name):
        cp = os.path.join(BASE, cond_name)
        result = {}
        for run_dir in sorted(os.listdir(cp)):
            summ = os.path.join(cp, run_dir, "episode_summary.json")
            if os.path.isfile(summ):
                with open(summ) as f:
                    s = json.load(f)
                key = (s.get("task_idx"), s.get("state_id"), s.get("perturbation_seed"))
                result[key] = s.get("task_success")
        return result

    nl = get_runs(nl_cond)
    al = get_runs(al_cond)
    common = set(nl.keys()) & set(al.keys())

    a = b = c = d = 0
    for k in common:
        nl_f = not nl[k]; al_f = not al[k]
        if nl_f and al_f: a += 1
        elif nl_f and not al_f: b += 1
        elif not nl_f and al_f: c += 1
        else: d += 1

    p = mcnemar_exact_p(b, c)
    print()
    print("{} (N={}):".format(name, len(common)))
    print("  Both fail={}  NL-only fail={}  AL-only fail={}  Both succ={}".format(a, b, c, d))
    print("  McNemar exact p = {:.4f}".format(p))
    if b + c > 0:
        print("  Paired effect = (c-b)/N = {:+.1%}".format((c-b)/len(common)))
