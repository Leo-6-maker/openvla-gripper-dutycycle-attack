#!/usr/bin/env python3
"""Phase 6D ablation evaluator: evaluate M1, M1-OS, M2 (all 5 seeds) on dev 90.
Reports paired per-seed comparisons and aggregate metrics.
"""
import csv, hashlib, json, math, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C = 0.3; TAU_R = 0.3; GUARD = 5

SEEDS = [42, 123, 456, 789, 1024]

MODEL_GROUPS = {
    "M1": {
        "label": "Primary-only",
        "paths": {s: REPO / f"outputs/sc5_ablation_primary_seed{s}/sc5_mlp_v2.pt" for s in SEEDS},
    },
    "M1_OS": {
        "label": "Primary-oversampled",
        "paths": {s: REPO / f"outputs/sc5_ablation_oversampled_seed{s}/sc5_mlp_v2.pt" for s in SEEDS},
    },
    "M2": {
        "label": "Primary+Reserve (current)",
        "paths": {s: REPO / f"outputs/sc5_v2_seed{s}/sc5_mlp_v2.pt" for s in SEEDS},
    },
}

DATASET_CSV = REPO / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"
DEV_LABELS_CSV = REPO / "evidence/m1c/sc5_v2_dev_combined_labels.csv"


def load_runtime(ckpt_path):
    return SC5DetectorRuntime(str(ckpt_path), tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)


def eval_trajectory(rt, rows):
    rt.reset()
    arm_step = -1; emit_step = -1
    max_corridor_p = 0.0
    corridor_above_streak = 0; max_corridor_streak = 0; current_streak = 0
    for r in rows:
        if rt.emitted:
            break
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            val = r.get(fn, "")
            if val in ("", "nan", "NaN", None):
                ok = False; break
            try:
                feats[fn] = float(val)
            except (ValueError, TypeError):
                ok = False; break
        if not ok:
            continue
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            continue
        step = int(r.get("step_idx", 0))
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)

        cp = dec.get("corridor_p", 0)
        if cp is not None and not math.isnan(cp):
            max_corridor_p = max(max_corridor_p, cp)
            if cp > TAU_C:
                current_streak += 1
                max_corridor_streak = max(max_corridor_streak, current_streak)
            else:
                current_streak = 0

        if rt.state == "ARMED" and arm_step < 0:
            arm_step = step
        if dec.get("emitted"):
            emit_step = step

    corridor_margin = max_corridor_streak - GUARD if max_corridor_streak > 0 else -GUARD

    return {
        "armed": rt.state == "ARMED" or arm_step >= 0, "emitted": rt.emitted,
        "arm_step": arm_step, "emit_step": emit_step,
        "max_corridor_p": round(max_corridor_p, 6),
        "max_corridor_streak": max_corridor_streak,
        "corridor_margin": corridor_margin,
    }


def main():
    # Load dev labels
    dev_labels = {}
    for lr in csv.DictReader(open(DEV_LABELS_CSV)):
        key = (int(lr["task"]), int(lr["state"]), lr["source"])
        dev_labels[key] = lr

    # Load val episodes
    all_rows = list(csv.DictReader(open(DATASET_CSV)))
    val_episodes = defaultdict(list)
    for r in all_rows:
        if r["split"] != "val":
            continue
        val_episodes[r["episode_id"]].append(r)

    # Build episode metadata
    ep_meta = {}
    for eid, rows in val_episodes.items():
        task = int(rows[0]["task_idx"])
        state = int(rows[0]["parent_state_id"])
        source = rows[0]["source_pool"]
        key = (task, state, source)
        if key not in dev_labels:
            continue
        lbl = dev_labels[key]
        tv = lbl.get("teacher_valid") == "True"
        anchor = int(lbl.get("teacher_anchor", -1))
        corr_s = anchor if anchor >= 0 else -1
        corr_e = anchor + 10 if anchor >= 0 else -1
        ep_meta[eid] = {
            "task": task, "state": state, "source": source,
            "teacher_valid": tv, "corridor_start": corr_s, "corridor_end": corr_e,
            "slice": "primary_dev" if source == "primary" else "reserve_dev",
        }

    dev_eps = sorted(ep_meta.keys())
    print(f"Dev episodes: {len(dev_eps)} (primary={sum(1 for m in ep_meta.values() if m['slice']=='primary_dev')}, reserve={sum(1 for m in ep_meta.values() if m['slice']=='reserve_dev')})")

    # Evaluate all models
    all_results = {}  # {group: {seed: {eid: {...}}}}
    slice_metrics = {}  # {group: {seed: {slice: {...}}}}

    for group_name, group_info in MODEL_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {group_name} ({group_info['label']})...")
        all_results[group_name] = {}
        slice_metrics[group_name] = {}

        for seed in SEEDS:
            ckpt = group_info["paths"][seed]
            if not ckpt.exists():
                print(f"  seed{seed}: CHECKPOINT MISSING ({ckpt})")
                continue
            rt = load_runtime(ckpt)
            print(f"  seed{seed}: loaded (sha={rt.checkpoint_sha256[:16]})")

            ep_res = {}
            sl_accum = {sl: {"tv": [], "nc": []} for sl in ["primary_dev", "reserve_dev"]}

            for eid in dev_eps:
                rows = val_episodes[eid]
                meta = ep_meta[eid]
                res = eval_trajectory(rt, rows)

                corr_s = meta["corridor_start"]
                corr_e = meta["corridor_end"]
                emit_before = res["emitted"] and corr_s >= 0 and res["emit_step"] < corr_s
                emit_inside = res["emitted"] and corr_s >= 0 and corr_s <= res["emit_step"] <= corr_e

                ep_res[eid] = {
                    "episode_id": eid, "task": meta["task"], "state": meta["state"],
                    "teacher_valid": meta["teacher_valid"],
                    "corridor_start": corr_s, "corridor_end": corr_e,
                    "armed": res["armed"], "arm_step": res["arm_step"],
                    "emitted": res["emitted"], "emit_step": res["emit_step"],
                    "emit_before": emit_before, "emit_inside": emit_inside,
                    "max_corridor_p": res["max_corridor_p"],
                    "max_corridor_streak": res["max_corridor_streak"],
                    "corridor_margin": res["corridor_margin"],
                }

                sl = meta["slice"]
                entry = {"emitted": res["emitted"], "armed": res["armed"],
                         "emit_before": emit_before, "emit_inside": emit_inside,
                         "max_corridor_p": res["max_corridor_p"],
                         "max_corridor_streak": res["max_corridor_streak"],
                         "corridor_margin": res["corridor_margin"]}
                if meta["teacher_valid"]:
                    sl_accum[sl]["tv"].append(entry)
                else:
                    sl_accum[sl]["nc"].append(entry)

            all_results[group_name][seed] = ep_res

            # Compute slice metrics
            sm = {}
            for sl in ["primary_dev", "reserve_dev"]:
                tv = sl_accum[sl]["tv"]
                nc = sl_accum[sl]["nc"]
                tv_trig = sum(1 for v in tv if v["emitted"])
                nc_trig = sum(1 for v in nc if v["emitted"])
                emit_before_n = sum(1 for v in tv if v["emit_before"])
                emit_inside_n = sum(1 for v in tv if v["emit_inside"])
                armed_n = sum(1 for v in tv + nc if v["armed"])
                emit_n = sum(1 for v in tv + nc if v["emitted"])
                armed_not_emit_n = sum(1 for v in tv if v["armed"] and not v["emitted"])

                sm[sl] = {
                    "tv_recall": tv_trig / max(len(tv), 1), "tv_total": len(tv), "tv_triggered": tv_trig,
                    "nc_abstain": 1.0 - nc_trig / max(len(nc), 1), "nc_total": len(nc), "nc_false_trigger": nc_trig,
                    "emit_inside": emit_inside_n, "emit_before": emit_before_n,
                    "armed_count": armed_n, "emitted_count": emit_n,
                    "armed_not_emit": armed_not_emit_n,
                }

            # Combined
            tv_all = []
            nc_all = []
            for sl in ["primary_dev", "reserve_dev"]:
                tv_all += sl_accum[sl]["tv"]
                nc_all += sl_accum[sl]["nc"]
            tv_trig = sum(1 for v in tv_all if v["emitted"])
            nc_trig = sum(1 for v in nc_all if v["emitted"])
            sm["combined_dev"] = {
                "tv_recall": tv_trig / max(len(tv_all), 1), "tv_total": len(tv_all), "tv_triggered": tv_trig,
                "nc_abstain": 1.0 - nc_trig / max(len(nc_all), 1), "nc_total": len(nc_all), "nc_false_trigger": nc_trig,
            }

            slice_metrics[group_name][seed] = sm

            r = sm["primary_dev"]
            print(f"    Primary: TV={r['tv_triggered']}/{r['tv_total']} NC={r['nc_false_trigger']}/{r['nc_total']} "
                  f"armed={r['armed_count']} emit={r['emitted_count']} emit_before={r['emit_before']} "
                  f"emit_inside={r['emit_inside']} armed_not_emit={r['armed_not_emit']}")

    # ── Save per-seed detailed CSV ──
    out_dir = REPO / "evidence/m1c/phase6d_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ablation_per_seed_metrics.csv"
    fields = ["group", "seed", "slice",
              "tv_recall", "tv_total", "tv_triggered",
              "nc_abstain", "nc_total", "nc_false_trigger",
              "armed_count", "emitted_count", "armed_not_emit",
              "emit_before", "emit_inside"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for group_name in ["M1", "M1_OS", "M2"]:
            for seed in SEEDS:
                if seed not in slice_metrics.get(group_name, {}):
                    continue
                for sl in ["primary_dev", "reserve_dev", "combined_dev"]:
                    r = slice_metrics[group_name][seed].get(sl, {})
                    row = {"group": group_name, "seed": seed, "slice": sl}
                    row.update(r)
                    w.writerow(row)
    print(f"\nSaved per-seed metrics: {csv_path}")

    # ── Per-episode output for seed42 comparison ──
    ep_csv = out_dir / "ablation_per_episode_seed42.csv"
    ep_fields = ["episode_id", "task", "state", "source", "teacher_valid",
                 "corridor_start", "corridor_end",
                 "M1_armed", "M1_emitted", "M1_emit_step", "M1_emit_before", "M1_emit_inside",
                 "M1OS_armed", "M1OS_emitted", "M1OS_emit_step", "M1OS_emit_before", "M1OS_emit_inside",
                 "M2_armed", "M2_emitted", "M2_emit_step", "M2_emit_before", "M2_emit_inside"]
    with open(ep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction="ignore")
        w.writeheader()
        for eid in dev_eps:
            row = {"episode_id": eid}
            m1 = all_results.get("M1", {}).get(42, {}).get(eid, {})
            m1os = all_results.get("M1_OS", {}).get(42, {}).get(eid, {})
            m2 = all_results.get("M2", {}).get(42, {}).get(eid, {})
            for k in ["task", "state", "source", "teacher_valid", "corridor_start", "corridor_end"]:
                row[k] = m1.get(k, "")
            for prefix, src in [("M1", m1), ("M1OS", m1os), ("M2", m2)]:
                row[f"{prefix}_armed"] = src.get("armed", "")
                row[f"{prefix}_emitted"] = src.get("emitted", "")
                row[f"{prefix}_emit_step"] = src.get("emit_step", "")
                row[f"{prefix}_emit_before"] = src.get("emit_before", "")
                row[f"{prefix}_emit_inside"] = src.get("emit_inside", "")
            w.writerow(row)
    print(f"Saved per-episode seed42: {ep_csv}")

    # ── Summary JSON ──
    summary = {
        "gate": "SC5_V2_DATA_ABLATION_EVALUATION",
        "tau_corridor": TAU_C, "tau_release": TAU_R, "guard": GUARD,
        "dataset_sha256": hashlib.sha256(open(DATASET_CSV, "rb").read()).hexdigest(),
        "models": {},
    }

    for group_name in ["M1", "M1_OS", "M2"]:
        seeds_data = {}
        for seed in SEEDS:
            if seed not in slice_metrics.get(group_name, {}):
                continue
            seeds_data[str(seed)] = slice_metrics[group_name][seed]

        primary = [seeds_data[str(s)]["primary_dev"] for s in SEEDS if str(s) in seeds_data]
        summary["models"][group_name] = {
            "label": MODEL_GROUPS[group_name]["label"],
            "n_seeds": len(primary),
            "primary_dev_mean": {
                "tv_recall": float(np.mean([r["tv_recall"] for r in primary])),
                "tv_recall_std": float(np.std([r["tv_recall"] for r in primary])),
                "nc_abstain": float(np.mean([r["nc_abstain"] for r in primary])),
                "nc_false_trigger_mean": float(np.mean([r["nc_false_trigger"] for r in primary])),
                "nc_false_trigger_total": sum(r["nc_false_trigger"] for r in primary),
                "armed_not_emit_mean": float(np.mean([r["armed_not_emit"] for r in primary])),
                "emit_before_mean": float(np.mean([r["emit_before"] for r in primary])),
                "emit_inside_mean": float(np.mean([r["emit_inside"] for r in primary])),
            },
            "per_seed": seeds_data,
        }

    json_path = out_dir / "ablation_evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary: {json_path}")

    # ── Print comparison table ──
    print(f"\n{'='*60}")
    print("COMPARISON: M1 vs M1-OS vs M2 (Primary Dev, mean over 5 seeds)")
    print(f"{'Group':<10} {'TV recall':>10} {'NC abstain':>10} {'NC false':>8} {'armed_not':>9} {'emit_b4':>8} {'emit_in':>8}")
    for group_name in ["M1", "M1_OS", "M2"]:
        if group_name not in summary["models"]:
            continue
        m = summary["models"][group_name]["primary_dev_mean"]
        print(f"{group_name:<10} {m['tv_recall']:>10.3f} {m['nc_abstain']:>10.3f} {m['nc_false_trigger_total']:>8.0f} {m['armed_not_emit_mean']:>9.1f} {m['emit_before_mean']:>8.1f} {m['emit_inside_mean']:>8.1f}")

    # ── Paired comparisons ──
    print(f"\n=== PAIRED SEED COMPARISONS ===")
    for seed in SEEDS:
        m1_r = slice_metrics.get("M1", {}).get(seed, {}).get("primary_dev", {})
        m1os_r = slice_metrics.get("M1_OS", {}).get(seed, {}).get("primary_dev", {})
        m2_r = slice_metrics.get("M2", {}).get(seed, {}).get("primary_dev", {})
        if not all([m1_r, m1os_r, m2_r]):
            continue
        print(f"  seed{seed}: M1 TV={m1_r['tv_triggered']}/{m1_r['tv_total']} NC={m1_r['nc_false_trigger']}/{m1_r['nc_total']} | "
              f"M1-OS TV={m1os_r['tv_triggered']}/{m1os_r['tv_total']} NC={m1os_r['nc_false_trigger']}/{m1os_r['nc_total']} | "
              f"M2 TV={m2_r['tv_triggered']}/{m2_r['tv_total']} NC={m2_r['nc_false_trigger']}/{m2_r['nc_total']}")

    # NC corridor probability margins (for Reserve where all may be 0/40)
    print(f"\n=== NC CORRIDOR PROBABILITY MARGINS (Primary Dev, seed42) ===")
    for group_name in ["M1", "M1_OS", "M2"]:
        ep = all_results.get(group_name, {}).get(42, {})
        nc_margins = [e["corridor_margin"] for e in ep.values() if not e["teacher_valid"]]
        nc_max_cp = [e["max_corridor_p"] for e in ep.values() if not e["teacher_valid"]]
        if nc_margins:
            print(f"  {group_name}: corridor_margin P50={np.percentile(nc_margins, 50):.1f} P10={np.percentile(nc_margins, 10):.1f} max_cp P50={np.percentile(nc_max_cp, 50):.4f} P90={np.percentile(nc_max_cp, 90):.4f} max={max(nc_max_cp):.4f}")


if __name__ == "__main__":
    main()
