#!/usr/bin/env python3
import os, json, math
from collections import defaultdict

BASE = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/breadth_120"

SLOTS = [
    ("salad_dressing", 2, 1, "salad_s1"),
    ("bbq_sauce", 3, 4, "bbq_s4"),
    ("ketchup", 4, 1, "ketchup_s1"),
    ("milk", 7, 5, "milk_s5"),
    ("butter", 6, 5, "butterA_s5"),
    ("orange_juice", 9, 2, "orange_s2"),
    ("tomato_sauce", 5, 1, "tomato_s1"),
    ("butter", 6, 6, "butterB_s6"),
]
TASKS = [
    ("salad_dressing", 2), ("bbq_sauce", 3), ("ketchup", 4),
    ("milk", 7), ("butter", 6), ("orange_juice", 9), ("tomato_sauce", 5),
]

def wilson_ci(n_fail, n_total):
    if n_total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = n_fail / n_total
    denom = 1 + z*z/n_total
    center = (p + z*z/(2*n_total)) / denom
    margin = z * math.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total)) / denom
    lo = max(0, center - margin)
    hi = min(1, center + margin)
    return (lo, hi)

def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value from discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    p_val = 0.0
    from math import comb
    for i in range(n + 1):
        prob = comb(n, i) * (0.5 ** n)
        if abs(i - n/2) >= abs(b - n/2):
            p_val += prob
    return p_val

for cond in ["tma_nolock", "tma_armlock", "prefix_nolock", "prefix_armlock", "rand"]:
    cp = os.path.join(BASE, cond)
    print("=" * 70)
    print("Condition: {}".format(cond))
    print("=" * 70)

    runs = {}
    for run_dir in sorted(os.listdir(cp)):
        if "_r" in run_dir:
            parts = run_dir.split("_r")
            if len(parts) > 1:
                try:
                    int(parts[-1])
                    continue
                except:
                    pass
        summ_path = os.path.join(cp, run_dir, "episode_summary.json")
        if not os.path.isfile(summ_path):
            continue
        with open(summ_path) as f:
            s = json.load(f)
        key = (s.get("task_idx"), s.get("state_id"), s.get("perturbation_seed"))
        runs[key] = {
            "task_success": s.get("task_success"),
            "attack_frames": s.get("attack_frames", 0),
            "mlp_triggered": s.get("mlp_triggered", False),
            "run_dir": run_dir,
        }

    n_total = len(runs)
    n_fail = sum(1 for r in runs.values() if r["task_success"] is False)
    n_emit = sum(1 for r in runs.values() if r["attack_frames"] > 0)
    emit_runs = {k: v for k, v in runs.items() if v["attack_frames"] > 0}
    n_cond = len(emit_runs)
    n_cond_fail = sum(1 for r in emit_runs.values() if r["task_success"] is False)

    itt_lo, itt_hi = wilson_ci(n_fail, n_total)
    cond_lo, cond_hi = wilson_ci(n_cond_fail, n_cond) if n_cond > 0 else (0, 0)

    print("ITT: {}/{} = {:.1%}  [95% CI: {:.1%}, {:.1%}]".format(
        n_fail, n_total, n_fail/n_total if n_total else 0, itt_lo, itt_hi))
    print("Conditional: {}/{} = {:.1%}  [95% CI: {:.1%}, {:.1%}]".format(
        n_cond_fail, n_cond, n_cond_fail/n_cond if n_cond else 0, cond_lo, cond_hi))
    print("Coverage: {}/{} = {:.1%}".format(n_emit, n_total, n_emit/n_total if n_total else 0))
    print()

    # Per-state-slot
    print("Per-state-slot ITT FR:")
    for task_name, task_idx, state_id, slot_name in SLOTS:
        slot_runs = {k: v for k, v in runs.items() if k[0] == task_idx and k[1] == state_id}
        if not slot_runs:
            continue
        n = len(slot_runs)
        f = sum(1 for v in slot_runs.values() if v["task_success"] is False)
        e = sum(1 for v in slot_runs.values() if v["attack_frames"] > 0)
        lo, hi = wilson_ci(f, n)
        print("  {:12s}: {}/{} = {:.0%} [CI: {:.0%}, {:.0%}] emit={}/{}".format(
            slot_name, f, n, f/n if n else 0, lo, hi, e, n))

    # Task macro
    print()
    print("Per-task ITT FR:")
    for task_name, task_idx in TASKS:
        task_runs = {k: v for k, v in runs.items() if k[0] == task_idx}
        if not task_runs:
            continue
        n = len(task_runs)
        f = sum(1 for v in task_runs.values() if v["task_success"] is False)
        e = sum(1 for v in task_runs.values() if v["attack_frames"] > 0)
        n_slots = len(set(k[1] for k in task_runs))
        lo, hi = wilson_ci(f, n)
        print("  {:16s}: {}/{} = {:.0%} [CI: {:.0%}, {:.0%}] emit={}/{} ({} state-slots)".format(
            task_name, f, n, f/n if n else 0, lo, hi, e, n, n_slots))

    # State-slot macro
    slot_fails = 0; slot_total = 0
    for task_name, task_idx, state_id, slot_name in SLOTS:
        slot_runs = {k: v for k, v in runs.items() if k[0] == task_idx and k[1] == state_id}
        slot_fails += sum(1 for v in slot_runs.values() if v["task_success"] is False)
        slot_total += len(slot_runs)
    slot_lo, slot_hi = wilson_ci(slot_fails, slot_total)
    print()
    print("State-slot macro FR: {}/{} = {:.1%} [95% CI: {:.1%}, {:.1%}]".format(
        slot_fails, slot_total, slot_fails/slot_total if slot_total else 0, slot_lo, slot_hi))
    print()

# Paired McNemar
print("=" * 70)
print("PAIRED MCNEMAR (conditional, exclude tomato_s1 no-emit)")
print("=" * 70)
tomato_keys = {(5, 1, 42), (5, 1, 123), (5, 1, 456)}

for name, nl_cond, al_cond in [("TMA", "tma_nolock", "tma_armlock"),
                                ("Prefix", "prefix_nolock", "prefix_armlock")]:
    def get_runs(cond_name):
        cp = os.path.join(BASE, cond_name)
        result = {}
        for run_dir in sorted(os.listdir(cp)):
            if "_r" in run_dir:
                parts = run_dir.split("_r")
                if len(parts) > 1:
                    try: int(parts[-1]); continue
                    except: pass
            summ_path = os.path.join(cp, run_dir, "episode_summary.json")
            if os.path.isfile(summ_path):
                with open(summ_path) as f:
                    s = json.load(f)
                key = (s.get("task_idx"), s.get("state_id"), s.get("perturbation_seed"))
                result[key] = s.get("task_success")
        return result

    nl = get_runs(nl_cond)
    al = get_runs(al_cond)
    common = set(nl.keys()) & set(al.keys())
    emit_keys = common - tomato_keys

    a_cond = b_cond = c_cond = d_cond = 0
    for k in emit_keys:
        nl_f = not nl[k]; al_f = not al[k]
        if nl_f and al_f: a_cond += 1
        elif nl_f and not al_f: b_cond += 1
        elif not nl_f and al_f: c_cond += 1
        else: d_cond += 1

    p = mcnemar_exact_p(b_cond, c_cond)
    print()
    print("{} (conditional N={}):".format(name, len(emit_keys)))
    print("  Both fail={}  NL-only fail={}  AL-only fail={}  Both succ={}".format(
        a_cond, b_cond, c_cond, d_cond))
    print("  McNemar exact p = {:.4f}".format(p))
    print("  Paired effect = (c-b)/N = ({}-{})/{} = {:+.1%}".format(
        c_cond, b_cond, len(emit_keys), (c_cond - b_cond)/len(emit_keys)))
