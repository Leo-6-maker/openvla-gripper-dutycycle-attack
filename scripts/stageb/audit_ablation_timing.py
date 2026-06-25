#!/usr/bin/env python3
"""Phase 6D.1: Common-support timing audit + failure analysis.
For each seed, compare early/inside rates only on TV episodes triggered
by ALL three models (M1 ∩ M1-OS ∩ M2). Also analyze M2 seed123 and
M1-OS seed123 failures.
"""
import csv, json, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]

# Paths
EVAL_CSV = REPO / "evidence/m1c/phase6d_ablation/ablation_per_episode_seed42.csv"
PER_SEED_CSV = REPO / "evidence/m1c/phase6d_ablation/ablation_per_seed_metrics.csv"
DEV_LABELS = REPO / "evidence/m1c/sc5_v2_dev_combined_labels.csv"
DATASET_CSV = REPO / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"


def load_seed42_episodes():
    """Load per-episode results for seed42."""
    rows = list(csv.DictReader(open(EVAL_CSV)))
    eps = {}
    for r in rows:
        eid = r["episode_id"]
        tv = r["teacher_valid"] == "True"
        eps[eid] = {
            "task": int(r["task"]), "state": int(r["state"]),
            "teacher_valid": tv,
            "corridor_start": int(r.get("corridor_start", -1)),
            "corridor_end": int(r.get("corridor_end", -1)),
            "M1_armed": r["M1_armed"] == "True", "M1_emitted": r["M1_emitted"] == "True",
            "M1_emit_step": int(r.get("M1_emit_step", -1) or -1),
            "M1_emit_before": r["M1_emit_before"] == "True",
            "M1_emit_inside": r["M1_emit_inside"] == "True",
            "M1OS_armed": r["M1OS_armed"] == "True", "M1OS_emitted": r["M1OS_emitted"] == "True",
            "M1OS_emit_step": int(r.get("M1OS_emit_step", -1) or -1),
            "M1OS_emit_before": r["M1OS_emit_before"] == "True",
            "M1OS_emit_inside": r["M1OS_emit_inside"] == "True",
            "M2_armed": r["M2_armed"] == "True", "M2_emitted": r["M2_emitted"] == "True",
            "M2_emit_step": int(r.get("M2_emit_step", -1) or -1),
            "M2_emit_before": r["M2_emit_before"] == "True",
            "M2_emit_inside": r["M2_emit_inside"] == "True",
        }
    return eps


def common_support_analysis(eps):
    """On TV episodes, compute common-trigger sets and timing stats."""
    tv_eps = {eid: v for eid, v in eps.items() if v["teacher_valid"]}

    C_all = {eid for eid, v in tv_eps.items()
             if v["M1_emitted"] and v["M1OS_emitted"] and v["M2_emitted"]}
    C_m1_m2 = {eid for eid, v in tv_eps.items()
               if v["M1_emitted"] and v["M2_emitted"]}
    C_m1os_m2 = {eid for eid, v in tv_eps.items()
                 if v["M1OS_emitted"] and v["M2_emitted"]}

    print("=== COMMON-SUPPORT TIMING AUDIT (seed42) ===")
    print(f"TV episodes: {len(tv_eps)}")
    print(f"M1 emitted: {sum(1 for v in tv_eps.values() if v['M1_emitted'])}")
    print(f"M1-OS emitted: {sum(1 for v in tv_eps.values() if v['M1OS_emitted'])}")
    print(f"M2 emitted: {sum(1 for v in tv_eps.values() if v['M2_emitted'])}")
    print(f"\nCommon sets:")
    print(f"  M1 ∩ M1-OS ∩ M2: {len(C_all)} episodes")
    print(f"  M1 ∩ M2: {len(C_m1_m2)} episodes")
    print(f"  M1-OS ∩ M2: {len(C_m1os_m2)} episodes")

    results = {}
    for label, cset in [("M1∩M1-OS∩M2", C_all), ("M1∩M2", C_m1_m2), ("M1-OS∩M2", C_m1os_m2)]:
        if len(cset) == 0:
            print(f"\n{label}: empty set, skipping")
            continue

        m1_early, m1os_early, m2_early = [], [], []
        m1_inside, m1os_inside, m2_inside = [], [], []
        m1_delta, m1os_delta, m2_delta = [], [], []

        for eid in sorted(cset):
            v = tv_eps[eid]
            cs = v["corridor_start"]
            if v["M1_emit_step"] >= 0 and cs >= 0:
                m1_delta.append(v["M1_emit_step"] - cs)
            if v["M1OS_emit_step"] >= 0 and cs >= 0:
                m1os_delta.append(v["M1OS_emit_step"] - cs)
            if v["M2_emit_step"] >= 0 and cs >= 0:
                m2_delta.append(v["M2_emit_step"] - cs)
            m1_early.append(1 if v["M1_emit_before"] else 0)
            m1os_early.append(1 if v["M1OS_emit_before"] else 0)
            m2_early.append(1 if v["M2_emit_before"] else 0)
            m1_inside.append(1 if v["M1_emit_inside"] else 0)
            m1os_inside.append(1 if v["M1OS_emit_inside"] else 0)
            m2_inside.append(1 if v["M2_emit_inside"] else 0)

        r = {
            "n_episodes": len(cset),
            "M1_early_rate": np.mean(m1_early), "M1_inside_rate": np.mean(m1_inside),
            "M1OS_early_rate": np.mean(m1os_early), "M1OS_inside_rate": np.mean(m1os_inside),
            "M2_early_rate": np.mean(m2_early), "M2_inside_rate": np.mean(m2_inside),
        }
        if m1_delta:
            r["M1_delta_median"] = float(np.median(m1_delta))
            r["M2_delta_median"] = float(np.median(m2_delta))
            r["M1_delta_mean"] = float(np.mean(m1_delta))
            r["M2_delta_mean"] = float(np.mean(m2_delta))
        if m1os_delta:
            r["M1OS_delta_median"] = float(np.median(m1os_delta))

        results[label] = r
        print(f"\n{label} (n={len(cset)}):")
        print(f"  M1:   early={r['M1_early_rate']:.3f} inside={r['M1_inside_rate']:.3f} |Δ|_median={r.get('M1_delta_median', 'N/A')}")
        print(f"  M1-OS: early={r['M1OS_early_rate']:.3f} inside={r['M1OS_inside_rate']:.3f}")
        print(f"  M2:   early={r['M2_early_rate']:.3f} inside={r['M2_inside_rate']:.3f} |Δ|_median={r.get('M2_delta_median', 'N/A')}")

    return results


def analyze_m2_seed123_failures():
    """Analyze M2 seed123 TV=27/36 failure."""
    print("\n" + "=" * 60)
    print("M2 seed123 TV RECALL FAILURE ANALYSIS")

    # Load labels
    labels = {}
    for lr in csv.DictReader(open(DEV_LABELS)):
        key = (int(lr["task"]), int(lr["state"]), lr["source"])
        labels[key] = lr

    # M2 seed123 results from per-seed CSV
    seed_data = {}
    for r in csv.DictReader(open(PER_SEED_CSV)):
        if r["group"] == "M2" and r["seed"] == "123":
            seed_data[r["slice"]] = r
    print(f"M2 seed123 primary: TV={seed_data.get('primary_dev', {}).get('tv_triggered', '?')}/{seed_data.get('primary_dev', {}).get('tv_total', '?')}"
          f" NC={seed_data.get('primary_dev', {}).get('nc_false_trigger', '?')}/{seed_data.get('primary_dev', {}).get('nc_total', '?')}")

    # We need per-episode data for M2 seed123, M1 seed123, M1-OS seed123
    # Since only seed42 has per-episode CSV, we need to re-evaluate seed123
    print("(Requires per-episode re-evaluation for seed123 — see below)")

    # Quick check: what seeds have the 27/36 issue?
    # M2 seed123 = 27, M2 others = 34-35
    # Let's check if this is task-specific using M2 seed42 per-episode data
    eps = load_seed42_episodes()
    tv_eps = {eid: v for eid, v in eps.items() if v["teacher_valid"]}

    # For seed42, M2 missed 1 TV: which one?
    m2_missed_42 = [eid for eid, v in tv_eps.items() if not v["M2_emitted"]]
    print(f"\nM2 seed42 missed TV episodes: {m2_missed_42}")
    for eid in m2_missed_42:
        v = tv_eps[eid]
        print(f"  {eid}: task={v['task']} state={v['state']} M1_emit={v['M1_emitted']} M1OS_emit={v['M1OS_emitted']}")

    # Check which tasks have lowest coverage across seed42
    task_counts = defaultdict(lambda: {"total": 0, "M1": 0, "M1OS": 0, "M2": 0})
    for eid, v in tv_eps.items():
        t = v["task"]
        task_counts[t]["total"] += 1
        for model in ["M1", "M1OS", "M2"]:
            if v[f"{model}_emitted"]:
                task_counts[t][model] += 1

    print(f"\nTask-level TV recall (seed42):")
    for t in sorted(task_counts):
        c = task_counts[t]
        print(f"  task{t}: M1={c['M1']}/{c['total']} M1-OS={c['M1OS']}/{c['total']} M2={c['M2']}/{c['total']}")

    return m2_missed_42


def analyze_m1os_seed123_nc():
    """Analyze M1-OS seed123 NC false triggers."""
    print("\n" + "=" * 60)
    print("M1-OS seed123 NC FALSE TRIGGER ANALYSIS")
    print("(Requires per-episode re-evaluation for seed123 — see below)")

    # Check NC corridor margins for seed42 to understand baseline
    eps = load_seed42_episodes()
    nc_eps = {eid: v for eid, v in eps.items() if not v["teacher_valid"]}

    # M1-OS seed42 NC episodes with highest corridor_p would need
    # re-evaluation of M1-OS seed123 to identify the 2 false triggers.
    # For now, report the NC episode task/state list.
    print(f"\nPrimary-dev NC episodes (seed42, n={len([e for e in nc_eps.values() if e['corridor_start']<0])}):")
    for eid, v in sorted(nc_eps.items()):
        if v["corridor_start"] < 0:
            triggered = []
            for model in ["M1", "M1OS", "M2"]:
                if v[f"{model}_emitted"]:
                    triggered.append(model)
            if triggered:
                print(f"  {eid}: task={v['task']} state={v['state']} TRIGGERED_BY={triggered}")
            else:
                pass  # not triggered — expected


def main():
    print("Phase 6D.1: Common-support timing audit + failure analysis")
    eps = load_seed42_episodes()
    common_support_analysis(eps)
    analyze_m2_seed123_failures()
    analyze_m1os_seed123_nc()

    print("\n" + "=" * 60)
    print("NEXT: For seed123 per-episode analysis, re-run evaluator on seed123 checkpoints.")
    print("M2 seed123 path: outputs/sc5_v2_seed123/sc5_mlp_v2.pt")
    print("M1 seed123 path: outputs/sc5_ablation_primary_seed123/sc5_mlp_v2.pt")
    print("M1-OS seed123 path: outputs/sc5_ablation_oversampled_seed123/sc5_mlp_v2.pt")


if __name__ == "__main__":
    main()
