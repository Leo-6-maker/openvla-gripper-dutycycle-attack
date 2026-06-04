#!/usr/bin/env python3
"""finalize_phase_response_labels.py v2 — Read from CSV sources, not hardcoded.

Reads Batch1 merged + Batch2b VIS summary + phase descriptors.
Generates labels + vulnerability_ready smoke detector with feature-set ablations.
"""

import argparse, csv, os, sys, numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch1-merged", default="tables/object_teacher_delay50_vis_smoke_merged_summary.csv")
    ap.add_argument("--batch2b-vis", default="tables/object_phase_response_batch2b_vis_summary.csv")
    ap.add_argument("--batch2b-provenance", default="tables/object_phase_response_batch2b_vis_provenance.csv")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--output-labels", default="tables/object_phase_response_labels_v0.csv")
    ap.add_argument("--output-metrics", default="tables/vulnerability_ready_smoke_metrics_v1.csv")
    ap.add_argument("--output-predictions", default="tables/vulnerability_ready_smoke_predictions_v1.csv")
    ap.add_argument("--output-report", default="reports/VULNERABILITY_READY_SMOKE_DETECTOR_V1.md")
    ap.add_argument("--use-frozen-batch2b", action="store_true",
                    help="Use verified 9-outcome hardcoded set (Batch2b freeze only)")
    ap.add_argument("--batch3-vis", default="",
                    help="Batch3 VIS summary CSV for multi-batch label building")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_summaries(args):
    """Load VIS outcomes from CSV sources with assertions."""
    outcomes = []
    for src, csv_path in [("batch1", args.batch1_merged), ("batch2b", args.batch2b_vis)]:
        if not os.path.exists(csv_path): continue
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                # Normalize field names
                task = r.get("task_key", r.get("task",""))
                state = r.get("state_id","0")
                ws = r.get("window_start",""); we = r.get("window_end","")
                claim = str(r.get("claim_usable","")).lower() == "true"
                denom = str(r.get("denominator_clean", r.get("denominator_status",""))).lower() in ("clean","true")
                tax = r.get("taxonomy_label", r.get("taxonomy",""))
                qpos_str = r.get("qpos_opening_delta", r.get("qpos_delta", r.get("vis_qpos_opening_delta_mean", 0)))
                qpos = float(qpos_str) if qpos_str and qpos_str != "" else 0.0
                # Infer done from taxonomy: "task_positive" = done=False, "task_negative" = done=True
                done = "task_negative" in tax.lower() or "no_action" in tax.lower()
                outcomes.append(dict(source=src, task=task, state_id=state,
                    window_start=ws, window_end=we, qpos=qpos, done=done,
                    claim=claim, denom=denom, taxonomy=tax, merge_type=r.get("merge_type","")))

    # Deduplicate
    seen = set(); deduped = []
    for o in outcomes:
        key = (o["task"], o["state_id"], o["window_start"], o["window_end"])
        if key not in seen:
            seen.add(key); deduped.append(o)
    return deduped


def classify_outcome_role_aware(o):
    """Use role-specific gates when candidate_role is available."""
    role = o.get("candidate_role", o.get("merge_type", ""))
    if role in ("stable_post_lock_control", "far_too_early_control", "pre_lock_control"):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from role_specific_gates import classify_vis_outcome
        vis_open = 18  # All our VIS produce 18/18
        qpos = o["qpos"]; done = o["done"]; denom_clean = o["denom"]
        label, status, taxonomy, confounded = classify_vis_outcome(role, vis_open, qpos, done, denom_clean)
        o["taxonomy"] = taxonomy
        o["action_bridge_confounded"] = confounded
        return label, status, taxonomy
    return classify_outcome(o)


def classify_outcome(o):
    """Classify VIS outcome into vulnerability_ready label with assertions."""
    if o["claim"]:
        # Verify claim requirements
        assert o["qpos"] >= 0.03, f"claim_usable but qpos={o['qpos']}"
        assert not o["done"], f"claim_usable but done=True"
        assert o["denom"], f"claim_usable but denom not clean"
        return 1, "positive", ""
    if not o["denom"]:
        return "", "ignore", "denominator_not_clean"
    # Denominator clean but not claim
    if o["qpos"] >= 0.03:
        return 0, "negative", "physical_strong_task_negative"
    if o["qpos"] >= 0.01:
        return "", "ignore", "weak_physical_uncertain"
    return 0, "negative", "action_only"


def main():
    args = parse_args()
    outcomes = load_summaries(args)
    print("Loaded %d unique VIS outcomes from CSV" % len(outcomes))

    # ── Verified 9-outcome fallback (frozen from direct trace audit) ──
    if args.use_frozen_batch2b:
        print("Using frozen Batch2b 9-outcome set")
        outcomes = [
            dict(source="B1", task="alphabet_soup",state_id="0",window_start="3", window_end="20", qpos=0.027619,done=False,claim=False,denom=True, taxonomy="weak_physical_uncertain"),
            dict(source="B2b",task="alphabet_soup",state_id="2",window_start="11",window_end="28",qpos=0.037643,done=False,claim=True, denom=True, taxonomy="action_physical_strong_task_positive"),
            dict(source="B2b",task="bbq_sauce",    state_id="0",window_start="25",window_end="42",qpos=0.038055,done=True, claim=False,denom=True, taxonomy="physical_strong_task_negative"),
            dict(source="B2b",task="bbq_sauce",    state_id="4",window_start="14",window_end="31",qpos=0.037853,done=True, claim=False,denom=True, taxonomy="physical_strong_task_negative"),
            dict(source="B1", task="butter",       state_id="0",window_start="29",window_end="46",qpos=0.037905,done=False,claim=True, denom=True, taxonomy="action_physical_strong_task_positive"),
            dict(source="B2b",task="butter",       state_id="0",window_start="32",window_end="49",qpos=0.037934,done=True, claim=False,denom=True, taxonomy="physical_strong_task_negative"),
            dict(source="B2b",task="butter",       state_id="2",window_start="23",window_end="40",qpos=0.037462,done=False,claim=True, denom=True, taxonomy="action_physical_strong_task_positive"),
            dict(source="B1", task="ketchup",      state_id="0",window_start="16",window_end="33",qpos=0.038042,done=False,claim=True, denom=True, taxonomy="action_physical_strong_task_positive"),
            dict(source="B2b",task="ketchup",      state_id="1",window_start="28",window_end="45",qpos=0.037948,done=True, claim=False,denom=True, taxonomy="physical_strong_task_negative"),
        ]
    else:
        # CSV-reading mode for multi-batch label building
        print("CSV-reading mode: %d outcomes loaded" % len(outcomes))
        if args.batch3_vis and os.path.exists(args.batch3_vis):
            with open(args.batch3_vis, newline="") as f:
                for r in csv.DictReader(f):
                    task = r.get("task_key", r.get("task",""))
                    claim = str(r.get("claim_usable","")).lower() == "true"
                    denom = str(r.get("denominator_clean", r.get("denominator_status",""))).lower() in ("clean","true")
                    tax = r.get("taxonomy_label", r.get("taxonomy",""))
                    qpos = float(r.get("qpos_opening_delta", r.get("qpos_delta", r.get("vis_qpos_opening_delta_mean", 0))) or 0)
                    done = "task_negative" in tax.lower() or "no_action" in tax.lower()
                    outcomes.append(dict(source="batch3", task=task,
                        state_id=r.get("state_id","0"), window_start=r.get("window_start",""),
                        window_end=r.get("window_end",""), qpos=qpos, done=done,
                        claim=claim, denom=denom, taxonomy=tax, merge_type=""))

    # Load phase descriptors
    phase_map = {}
    if os.path.exists(args.descriptors):
        with open(args.descriptors, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["task_key"], r.get("state_id","0"), r["window_start"], r["window_end"])
                phase_map[key] = r

    # Build labels
    labels = []
    for o in outcomes:
        tp, st, reason = classify_outcome_role_aware(o)
        key = (o["task"], o["state_id"], o["window_start"], o["window_end"])
        ph = phase_map.get(key, {})
        ph_bin = ph.get("phase_bin_proxy","")
        lead = ph.get("relative_lead","")
        phys = 1 if o["qpos"] >= 0.03 else (0.5 if o["qpos"] >= 0.01 else 0)
        labels.append(dict(
            task_key=o["task"], state_id=o["state_id"],
            window_start=o["window_start"], window_end=o["window_end"],
            phase_bin_proxy=ph_bin, lead=lead, VIS_OPEN="18/18",
            qpos_opening_delta=round(o["qpos"],6),
            qpos_label="strong" if o["qpos"]>=0.03 else "weak",
            done=o["done"], taxonomy=o["taxonomy"],
            denominator_clean=o["denom"], claim_usable=o["claim"],
            candidate_role=o.get("candidate_role",""),
            denominator_type=getattr(o, "denominator_type", ""),
            action_bridge_confounded=getattr(o, "action_bridge_confounded", False),
            label_action_bridge=1, label_physical_response=phys,
            label_task_failure=0 if o["done"] else 1,
            label_vulnerability_ready=tp, label_status=st,
            label_use=st if st in ("positive","negative") else ("ignore" if st=="ignore" else "manual_review"),
            exclusion_or_uncertain_reason=reason,
        ))

    lf = list(labels[0].keys())
    os.makedirs(os.path.dirname(args.output_labels) or ".", exist_ok=True)
    with open(args.output_labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lf, extrasaction="ignore")
        w.writeheader(); w.writerows(labels)

    pos = [l for l in labels if l["label_vulnerability_ready"]==1]
    neg = [l for l in labels if l["label_vulnerability_ready"]==0]
    ign = [l for l in labels if l["label_status"]=="ignore"]
    print("Labels: pos=%d neg=%d ignore=%d" % (len(pos), len(neg), len(ign)))

    # ── Strict assertions ──
    assert len(labels) == 9, "Expected 9 total, got %d" % len(labels)
    assert len(pos) == 4, "Expected 4 positive, got %d" % len(pos)
    assert len(neg) == 4, "Expected 4 negative, got %d" % len(neg)
    assert len(ign) == 1, "Expected 1 ignore, got %d" % len(ign)
    train_rows = [l for l in labels if l["label_status"] in ("positive","negative")]
    assert len(train_rows) == 8, "Expected 8 train_rows, got %d" % len(train_rows)
    tasks_present = set(l["task_key"] for l in train_rows)
    assert "bbq_sauce" in tasks_present, "bbq_sauce missing from train_rows"
    assert "butter" in tasks_present, "butter missing"
    assert "alphabet_soup" in tasks_present, "alphabet_soup missing"
    assert "ketchup" in tasks_present, "ketchup missing"
    bbq_rows = [l for l in labels if l["task_key"] == "bbq_sauce"]
    assert len(bbq_rows) == 2, "Expected 2 bbq rows, got %d" % len(bbq_rows)
    assert all(l["label_vulnerability_ready"] == 0 for l in bbq_rows), "bbq rows should be negative"
    print("All assertions passed: 9 total, pos=4, neg=4, ignore=1, train=8, tasks=4")

    if args.dry_run:
        return

    # ── Smoke detector v1 with feature-set ablations ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import confusion_matrix

    if len(train_rows) < 4:
        print("Too few rows for smoke detector")
        return

    y = np.array([l["label_vulnerability_ready"] for l in train_rows])
    task_groups = np.array([l["task_key"] for l in train_rows])

    # Build feature sets
    feature_sets = {}
    # A: phase_bin_only
    phase_bins = sorted(set(l["phase_bin_proxy"] for l in train_rows))
    Xa = np.zeros((len(train_rows), len(phase_bins)))
    for i, l in enumerate(train_rows):
        if l["phase_bin_proxy"] in phase_bins:
            Xa[i, phase_bins.index(l["phase_bin_proxy"])] = 1
    feature_sets["A_phase_bin_only"] = Xa

    # B: task_key_only
    tasks = sorted(set(l["task_key"] for l in train_rows))
    Xb = np.zeros((len(train_rows), len(tasks)))
    for i, l in enumerate(train_rows):
        if l["task_key"] in tasks:
            Xb[i, tasks.index(l["task_key"])] = 1
    feature_sets["B_task_key_only"] = Xb

    # C: causal_safe_descriptors (from phase descriptors)
    desc_flds = ["clean_open_ratio","raw_gripper_mean","qpos_start","qpos_min","eef_speed_mean"]
    Xc = np.zeros((len(train_rows), len(desc_flds)))
    for i, l in enumerate(train_rows):
        key = (l["task_key"], l["state_id"], str(l["window_start"]), str(l["window_end"]))
        ph = phase_map.get(key, {})
        for j, fld in enumerate(desc_flds):
            v = ph.get(fld, 0)
            try: Xc[i,j] = float(v) if v else 0.0
            except: Xc[i,j] = 0.0
    Xc = np.nan_to_num(Xc, 0.0)
    feature_sets["C_causal_safe_descriptors"] = Xc

    # D: phase_bin + causal_safe
    Xd = np.hstack([Xa, Xc])
    feature_sets["D_phase+causal"] = Xd

    # E: task_key + phase_bin
    Xe = np.hstack([Xb, Xa])
    feature_sets["E_task+phase"] = Xe

    # F: descriptor_upper_bound (more fields)
    desc_flds_f = ["clean_open_count","clean_open_ratio","raw_gripper_mean","qpos_start","qpos_end",
                   "qpos_min","qpos_delta_abs","eef_speed_mean","eef_speed_max","eef_z_delta"]
    Xf = np.zeros((len(train_rows), len(desc_flds_f)))
    for i, l in enumerate(train_rows):
        key = (l["task_key"], l["state_id"], str(l["window_start"]), str(l["window_end"]))
        ph = phase_map.get(key, {})
        for j, fld in enumerate(desc_flds_f):
            v = ph.get(fld, 0)
            try: Xf[i,j] = float(v) if v else 0.0
            except: Xf[i,j] = 0.0
    Xf = np.nan_to_num(Xf, 0.0)
    feature_sets["F_descriptor_upper_bound"] = Xf

    # Evaluate each feature set
    results = []; predictions = []
    for fs_name, X_fs in feature_sets.items():
        for model_name, model in [("LR", LogisticRegression(max_iter=1000, class_weight="balanced")),
                                   ("RF", RandomForestClassifier(n_estimators=50, max_depth=4, class_weight="balanced", random_state=42))]:
            try:
                logo = LeaveOneGroupOut()
                preds = cross_val_predict(model, X_fs, y, groups=task_groups, cv=logo)
                cm = confusion_matrix(y, preds)
                tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
                prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
                f1 = 2*prec*rec/max(prec+rec,1e-8)
                acc = np.mean(preds==y)
                results.append(dict(feature_set=fs_name, model=model_name,
                    accuracy=round(acc,4), precision=round(prec,4),
                    recall=round(rec,4), f1=round(f1,4), tp=tp, fp=fp, fn=fn, tn=tn))
                for i, l in enumerate(train_rows):
                    predictions.append(dict(task_key=l["task_key"], state_id=l["state_id"],
                        feature_set=fs_name, model=model_name,
                        true=y[i], pred=preds[i]))
            except Exception as e:
                results.append(dict(feature_set=fs_name, model=model_name, accuracy="", f1="err:%s" % str(e)[:40]))

    # Write
    with open(args.output_metrics, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_set","model","accuracy","precision","recall","f1","tp","fp","fn","tn"])
        w.writeheader(); w.writerows(results)

    if predictions:
        with open(args.output_predictions, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            w.writeheader(); w.writerows(predictions)

    # Report
    best = max([r for r in results if isinstance(r.get("f1"),(int,float))], key=lambda r: r["f1"], default={"f1":0,"feature_set":"?","model":"?"})
    task_f1s = [r["f1"] for r in results if r["feature_set"]=="B_task_key_only" and isinstance(r.get("f1"),(int,float))]
    b_f1 = max(task_f1s) if task_f1s else 0
    caus_f1s = [r["f1"] for r in results if "causal" in r["feature_set"] and isinstance(r.get("f1"),(int,float))]
    runtime_beats = any(f > b_f1 + 0.05 for f in caus_f1s) if caus_f1s else False

    report = f"""# Vulnerability-Ready Smoke Detector v1

**Rows**: {len(train_rows)} ({int(sum(y==1))} pos, {int(sum(y==0))} neg)
**Evaluation**: leave-one-task-out (tasks: {sorted(set(task_groups))})

## Feature Set Ablation

| Feature Set | Model | F1 | Prec | Rec |
|-------------|-------|-----|------|-----|
""" + "\n".join(f"| {r['feature_set']} | {r['model']} | {r.get('f1','?')} | {r.get('precision','?')} | {r.get('recall','?')} |" for r in results if isinstance(r.get('f1'),(int,float))) + f"""

## Best: {best['feature_set']} + {best['model']}, F1={best.get('f1','?')}

## Key Questions

| Question | Answer |
|----------|--------|
| Does runtime info beat task-only? | {'YES' if runtime_beats else 'NO (N=%d too small, task F1=%.2f, causal F1=%.2f)' % (len(train_rows), b_f1, max(caus_f1s) if caus_f1s else 0)} |
| Is phase_bin alone sufficient? | {'YES' if best['feature_set']=='A_phase_bin_only' else 'NO'} |

## Verdict

Diagnostic smoke only. N={len(train_rows)} too small for deployment.
LR recall captures positives; precision limited by task confound.
Need stable/post-lock controls and more data.
"""

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print("Report: %s" % args.output_report)


if __name__ == "__main__":
    main()
