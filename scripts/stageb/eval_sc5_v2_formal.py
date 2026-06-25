#!/usr/bin/env python3
"""Formal SC5-V2 evaluator v3: original step_idx, slice-scoped metrics, per-episode CSV, clean output."""
import csv, json, hashlib, math, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

V1_CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
V2_CKPTS = {
    42: REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
    123: REPO / "outputs/sc5_v2_seed123/sc5_mlp_v2.pt",
    456: REPO / "outputs/sc5_v2_seed456/sc5_mlp_v2.pt",
    789: REPO / "outputs/sc5_v2_seed789/sc5_mlp_v2.pt",
    1024: REPO / "outputs/sc5_v2_seed1024/sc5_mlp_v2.pt",
}
DATASET_CSV = REPO / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"
DEV_LABELS_CSV = REPO / "evidence/m1c/sc5_v2_dev_combined_labels.csv"
TAU_C = 0.3; TAU_R = 0.3; GUARD = 5

def load_runtime(ckpt_path):
    return SC5DetectorRuntime(str(ckpt_path), tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)

def get_corridor_range(anchor_str):
    try:
        a = int(anchor_str)
        if a < 0: return -1, -1
        return a, a + 10  # K=10 corridor end
    except (ValueError, TypeError):
        return -1, -1

def eval_trajectory(rt, rows):
    """Replay detector using original step_idx from telemetry."""
    rt.reset()
    armed = False; emitted = False; emit_step = -1; arm_step = -1
    for r in rows:
        if emitted: break
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            val = r.get(fn, "")
            if val in ("", "nan", "NaN", None): ok = False; break
            try: feats[fn] = float(val)
            except: ok = False; break
        if not ok: continue
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)): continue
        step = int(r.get("step_idx", 0))
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)
        if rt.state == "ARMED" and arm_step < 0: arm_step = step; armed = True
        if dec.get("emitted"): emitted = True; emit_step = step
    return armed, emitted, arm_step, emit_step

def main():
    # Load dev labels (fail-closed)
    dev_labels = {}
    for lr in csv.DictReader(open(DEV_LABELS_CSV)):
        key = (int(lr["task"]), int(lr["state"]), lr["source"])
        dev_labels[key] = lr

    all_rows = list(csv.DictReader(open(DATASET_CSV)))
    val_episodes = defaultdict(list)
    for r in all_rows:
        if r["split"] != "val": continue
        val_episodes[r["episode_id"]].append(r)

    # Build episode metadata (fail-closed label lookup)
    ep_meta = {}
    for eid, rows in val_episodes.items():
        task = int(rows[0]["task_idx"]); state = int(rows[0]["parent_state_id"])
        source = rows[0]["source_pool"]
        key = (task, state, source)
        if key not in dev_labels:
            raise RuntimeError("MISSING_LABEL: episode=%s task=%d state=%d source=%s" % (eid, task, state, source))
        lbl = dev_labels[key]
        tv = lbl.get("teacher_valid") == "True"
        anchor = lbl.get("teacher_anchor", "-1")
        corr_s, corr_e = get_corridor_range(anchor)
        ep_meta[eid] = {
            "task": task, "state": state, "source": source,
            "teacher_valid": tv, "teacher_anchor": anchor,
            "corridor_start": corr_s, "corridor_end": corr_e,
            "slice": "primary_dev" if source == "primary" else "reserve_dev",
        }
    print("Episodes: %d (primary=%d reserve=%d TV=%d NC=%d)" % (
        len(val_episodes),
        sum(1 for m in ep_meta.values() if m["slice"]=="primary_dev"),
        sum(1 for m in ep_meta.values() if m["slice"]=="reserve_dev"),
        sum(1 for m in ep_meta.values() if m["teacher_valid"]),
        sum(1 for m in ep_meta.values() if not m["teacher_valid"])))

    # Evaluate
    models = {"V1": load_runtime(V1_CKPT)}
    for seed, path in V2_CKPTS.items():
        models["V2_s%d" % seed] = load_runtime(path)

    all_ep_results = {}  # {model_name: {eid: {...}}}
    slice_results = {}   # {slice_name: {model_name: {...}}}

    for name, rt in models.items():
        print("\nEvaluating %s..." % name)
        ep_res = {}
        # Per-slice accumulators
        sl_accum = {sl: {"tv_eps": [], "nc_eps": []} for sl in ["primary_dev", "reserve_dev"]}

        for eid in sorted(val_episodes.keys()):
            rows = val_episodes[eid]
            meta = ep_meta[eid]
            armed, emitted, arm_s, emit_s = eval_trajectory(rt, rows)
            corr_s = meta["corridor_start"]; corr_e = meta["corridor_end"]
            emit_before = emitted and corr_s >= 0 and emit_s < corr_s
            emit_inside = emitted and corr_s >= 0 and corr_s <= emit_s <= corr_e
            emit_after = emitted and corr_e >= 0 and emit_s > corr_e

            ep_res[eid] = {
                "episode_id": eid, "task": meta["task"], "state": meta["state"],
                "source": meta["source"], "teacher_valid": meta["teacher_valid"],
                "corridor_start": corr_s, "corridor_end": corr_e,
                "armed": armed, "arm_step": arm_s,
                "emitted": emitted, "emit_step": emit_s,
                "emit_before": emit_before, "emit_inside": emit_inside, "emit_after": emit_after,
                "n_feature_valid_steps": len(rows),
            }

            sl = meta["slice"]
            entry = {"emitted": emitted, "armed": armed, "emit_before": emit_before,
                     "emit_inside": emit_inside, "emit_after": emit_after, "arm_step": arm_s, "emit_step": emit_s}
            if meta["teacher_valid"]:
                sl_accum[sl]["tv_eps"].append(entry)
            else:
                sl_accum[sl]["nc_eps"].append(entry)

        all_ep_results[name] = ep_res

        # Compute slice metrics
        for sl in ["primary_dev", "reserve_dev"]:
            tv = sl_accum[sl]["tv_eps"]; nc = sl_accum[sl]["nc_eps"]
            slice_eps = tv + nc
            tv_trig = sum(1 for v in tv if v["emitted"])
            nc_trig = sum(1 for v in nc if v["emitted"])
            emit_inside_n = sum(1 for v in tv if v["emit_inside"])
            emit_before_n = sum(1 for v in tv if v["emit_before"])
            armed_n = sum(1 for v in slice_eps if v["armed"])
            emit_n = sum(1 for v in slice_eps if v["emitted"])

            slice_results.setdefault(sl, {})[name] = {
                "tv_recall": tv_trig / max(len(tv), 1), "tv_total": len(tv), "tv_triggered": tv_trig,
                "nc_abstain": 1.0 - nc_trig / max(len(nc), 1), "nc_total": len(nc), "nc_false_trigger": nc_trig,
                "emit_inside": emit_inside_n, "emit_before": emit_before_n,
                "armed_count": armed_n, "emitted_count": emit_n,
            }
            r = slice_results[sl][name]
            print("  %s: TV=%d/%d (%.3f) NC=%d/%d (%.3f) armed=%d emit=%d inside=%d before=%d" % (
                sl, r["tv_triggered"], r["tv_total"], r["tv_recall"],
                r["nc_false_trigger"], r["nc_total"], r["nc_abstain"],
                r["armed_count"], r["emitted_count"], r["emit_inside"], r["emit_before"]))

        # Combined
        tv_all = []; nc_all = []
        for sl in ["primary_dev", "reserve_dev"]:
            tv_all += sl_accum[sl]["tv_eps"]; nc_all += sl_accum[sl]["nc_eps"]
        tv_trig = sum(1 for v in tv_all if v["emitted"]); nc_trig = sum(1 for v in nc_all if v["emitted"])
        slice_results["combined_dev"] = slice_results.get("combined_dev", {})
        slice_results["combined_dev"][name] = {
            "tv_recall": tv_trig / max(len(tv_all), 1), "tv_total": len(tv_all), "tv_triggered": tv_trig,
            "nc_abstain": 1.0 - nc_trig / max(len(nc_all), 1), "nc_total": len(nc_all), "nc_false_trigger": nc_trig,
        }

    # Save per-episode CSV for seed42
    ep_csv_path = REPO / "evidence/m1c/per_episode_results.csv"
    ep_fields = ["episode_id","task","state","source","teacher_valid","corridor_start","corridor_end",
                 "V1_armed","V1_emitted","V1_emit_step","V1_emit_before","V1_emit_inside","V1_emit_after",
                 "V2_armed","V2_emitted","V2_emit_step","V2_emit_before","V2_emit_inside","V2_emit_after"]
    with open(ep_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction="ignore")
        w.writeheader()
        for eid in sorted(val_episodes.keys()):
            v1r = all_ep_results["V1"][eid]
            v2r = all_ep_results["V2_s42"][eid]
            row = {}
            for k in ep_fields:
                if k.startswith("V1_"): row[k] = v1r.get(k[3:], "")
                elif k.startswith("V2_"): row[k] = v2r.get(k[3:], "")
                else: row[k] = v1r.get(k, "")
            w.writerow(row)
    print("\nPer-episode CSV: %s (%d rows)" % (ep_csv_path, len(val_episodes)))

    # Tie-break tied seeds
    print("\n=== TIE-BREAK (42 vs 456 vs 789) ===")
    for name in ["V2_s42", "V2_s456", "V2_s789"]:
        ep = all_ep_results[name]
        tv_eps = {eid: v for eid, v in ep.items() if ep_meta[eid]["teacher_valid"] and ep_meta[eid]["slice"]=="primary_dev"}
        emit_before = sum(1 for v in tv_eps.values() if v["emit_before"])
        emit_inside = sum(1 for v in tv_eps.values() if v["emit_inside"])
        # Task-level TV
        task_tv = defaultdict(lambda: [0, 0])
        for eid, v in tv_eps.items():
            t = ep_meta[eid]["task"]
            task_tv[t][1] += 1
            if v["emitted"]: task_tv[t][0] += 1
        task_recalls = {t: c[0]/max(c[1],1) for t, c in task_tv.items()}
        worst = min(task_recalls.values()) if task_recalls else 0
        armed_not_emit = sum(1 for v in tv_eps.values() if v["armed"] and not v["emitted"])
        print("  %s: inside=%d before=%d worst_task=%.3f armed_not_emit=%d" % (
            name, emit_inside, emit_before, worst, armed_not_emit))

    # Save results
    out = {
        "gate": "SC5_V2_FORMAL_EVALUATOR_V3",
        "tau_corridor": TAU_C, "tau_release": TAU_R, "guard": GUARD,
        "v1_sha256": hashlib.sha256(open(V1_CKPT,"rb").read()).hexdigest(),
        "v2_seed42_sha256": hashlib.sha256(open(V2_CKPTS[42],"rb").read()).hexdigest(),
        "dataset_sha256": hashlib.sha256(open(DATASET_CSV,"rb").read()).hexdigest(),
        "per_slice": {sl: {name: r for name, r in models.items()} for sl, models in slice_results.items()},
    }
    out_path = REPO / "evidence/m1c/formal_evaluator_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Verify no SSH warnings in output
    raw = open(out_path).read()
    assert raw.startswith("{"), "JSON does not start with '{' — possible SSH warning contamination"
    print("\nSaved: %s (clean JSON verified)" % out_path)

    # Selection summary
    v1_tv = slice_results["primary_dev"]["V1"]["tv_recall"]
    tv_gate = max(0.80, v1_tv - 0.05)
    print("\n=== SELECTION (TV>=%.3f, NC>=0.90) ===" % tv_gate)
    for name in sorted(models.keys()):
        if name == "V1": continue
        r = slice_results["primary_dev"][name]
        ok = r["tv_recall"] >= tv_gate and r["nc_abstain"] >= 0.90
        print("  %s: TV=%.3f NC=%.3f %s" % (name, r["tv_recall"], r["nc_abstain"], "PASS" if ok else "FAIL"))

if __name__ == "__main__":
    main()
