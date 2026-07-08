#!/usr/bin/env python3
"""C2e2C: suite-conditioned calibration gate.

Uses C2e2 v2 selected model scores (emit_p, suppress_p). For each suite,
selects tau_emit/tau_suppress on validation split only, then evaluates on
test split once. No training, no model changes.

CPU-only. No OpenVLA, no LIBERO, no env.step, no rollout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

# ---- model (same as C2e2) ----
class PoolingMLP(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+"\n")

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})

def load_stats(path):
    obj = json.loads(open(path).read())
    return {"temporal_mean": np.asarray(obj.get("temporal_feature_mean",[]), dtype=np.float32),
            "temporal_std": np.asarray(obj.get("temporal_feature_std",[]), dtype=np.float32),
            "context_mean": np.asarray(obj.get("context_feature_mean",[]), dtype=np.float32),
            "context_std": np.asarray(obj.get("context_feature_std",[]), dtype=np.float32)}

def pooled_features(xt, xc, stats):
    tm, ts = stats["temporal_mean"].reshape(1,1,-1), stats["temporal_std"].reshape(1,1,-1)
    xt = (xt.astype(np.float32) - tm) / np.maximum(ts, 1e-8)
    if xc.shape[1] > 0:
        cm, cs = stats["context_mean"].reshape(1,-1), stats["context_std"].reshape(1,-1)
        xc = (xc.astype(np.float32) - cm) / np.maximum(cs, 1e-8)
    last = xt[:,-1,:]; mean = xt.mean(axis=1); std = xt.std(axis=1); delta = xt[:,-1,:] - xt[:,0,:]
    return np.concatenate([last, mean, std, delta, xc], axis=1).astype(np.float32)

def load_window_dataset(c2e1_root, window):
    npz = np.load(Path(c2e1_root) / f"c2e1_w{window:02d}_temporal_dataset.npz", allow_pickle=True)
    stats = load_stats(Path(c2e1_root) / f"c2e1_w{window:02d}_normalization_stats_train_only.json")
    x = pooled_features(np.asarray(npz["X_temporal"], dtype=np.float32), np.asarray(npz["X_context"], dtype=np.float32), stats)
    return {"x": x, "y": npz["y"].astype(np.int64), "split": np.asarray(npz["split"]).astype(str),
            "suite": np.asarray(npz["suite"]).astype(str), "row_index": npz["row_index"].astype(np.int64)}

def predict_logits(model, x, batch_size=4096):
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            outs.append(model(torch.from_numpy(x[s:s+batch_size]).float()).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,2), dtype=np.float32)

def metrics_from_pred(y, pred):
    pos, neg = y==1, y==0
    tp, fn = int((pred&pos).sum()), int(((~pred)&pos).sum())
    fp, tn = int((pred&neg).sum()), int(((~pred)&neg).sum())
    rec = tp/max(1, tp+fn); fpr = fp/max(1, fp+tn); prec = tp/max(1, tp+fp)
    f1 = 2*prec*rec/max(1e-12, prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1,"acc":(tp+tn)/max(1,len(y))}

def sweep(y, ep, sp, tau_vals, min_recall, max_fp):
    rows = []
    for te in tau_vals:
        for ts in tau_vals:
            pred = (ep>=te)&(sp<=ts); m = metrics_from_pred(y, pred)
            feasible = m["recall"]>=min_recall and m["fp_rate"]<=max_fp
            score = m["f1"]+0.5*m["recall"]-3*max(0,min_recall-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            rows.append({"tau_emit":te,"tau_suppress":ts,"feasible":feasible,"score":score,**m})
    rows.sort(key=lambda r: (bool(r["feasible"]), r["score"], r["f1"], r["recall"], -r["fp_rate"]), reverse=True)
    return rows

def main():
    ap = argparse.ArgumentParser(description="C2e2C suite-conditioned calibration gate")
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--c2e2-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--min-recall", type=float, default=0.70)
    ap.add_argument("--max-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-fp", type=float, default=0.50)
    ap.add_argument("--min-object-fp", type=float, default=0.30)
    ap.add_argument("--min-l10-recall", type=float, default=0.50)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    c2e1_root = Path(args.c2e1_root); c2e2_root = Path(args.c2e2_root)

    # Load config
    for name in ["c2e2_v2_selected_model_config.json", "c2e2_selected_model_config.json"]:
        cfg_path = c2e2_root / name
        if cfg_path.exists(): break
    cfg = json.loads(cfg_path.read_text())
    sel_cfg = cfg.get("selected_config", cfg)
    window = int(sel_cfg["window"])
    hidden = [int(x) for x in str(sel_cfg["hidden"]).split("-")]
    for name in ["c2e2_v2_selected_model.pt", "c2e2_selected_model.pt"]:
        model_path = c2e2_root / name
        if model_path.exists(): break

    print(f"C2e2C: window={window} hidden={hidden}")

    # Load data + model
    dataset = load_window_dataset(c2e1_root, window)
    ckpt = torch.load(model_path, map_location="cpu")
    model = PoolingMLP(ckpt.get("input_dim", dataset["x"].shape[1]), hidden, sel_cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model_state_dict"]); model.cpu().eval()
    logits = predict_logits(model, dataset["x"])
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    y, split, suite, ri = dataset["y"], dataset["split"], dataset["suite"], dataset["row_index"]
    tau_vals = [round(x,3) for x in np.linspace(0.01, 0.99, 99).tolist()]
    suites = sorted(set(suite))

    # Per-suite val sweep
    suite_thresholds = {}
    val_sweeps = {}
    for s in suites:
        vm = (suite==s) & (split=="val")
        if vm.sum() == 0: continue
        sw = sweep(y[vm], ep[vm], sp[vm], tau_vals, args.min_recall, args.max_fp)
        val_sweeps[s] = sw[:10]  # top 10
        best = sw[0]
        suite_thresholds[s] = {"tau_emit": best["tau_emit"], "tau_suppress": best["tau_suppress"],
                                "val_feasible": best["feasible"], "val_recall": best["recall"],
                                "val_fp_rate": best["fp_rate"], "val_f1": best["f1"]}

    # Apply per-suite thresholds to all splits
    pred_cal = np.zeros(len(y), dtype=bool)
    for s in suites:
        sm = suite == s
        if s in suite_thresholds:
            t = suite_thresholds[s]
            pred_cal[sm] = (ep[sm] >= t["tau_emit"]) & (sp[sm] <= t["tau_suppress"])

    # Test metrics
    tm_test = split == "test"
    test_rows = []
    for s in suites:
        m = (suite==s) & tm_test
        if m.sum() == 0: continue
        met = metrics_from_pred(y[m], pred_cal[m])
        test_rows.append({"suite":s,"n":int(m.sum()),**met})

    overall = metrics_from_pred(y[tm_test], pred_cal[tm_test])

    # Violations
    violations = []
    if overall["recall"] < args.min_recall:
        violations.append(f"LOW_OVERALL_RECALL:{overall['recall']:.4f}")
    if overall["fp_rate"] > args.max_fp:
        violations.append(f"HIGH_OVERALL_FP:{overall['fp_rate']:.4f}")
    for r in test_rows:
        if r["fp_rate"] > args.max_suite_fp:
            violations.append(f"HIGH_SUITE_FP:{r['suite']}={r['fp_rate']:.4f}")
    obj_row = next((r for r in test_rows if r["suite"]=="libero_object"), None)
    if obj_row and obj_row["fp_rate"] > args.min_object_fp:
        violations.append(f"HIGH_OBJECT_FP:{obj_row['fp_rate']:.4f}")
    l10_row = next((r for r in test_rows if r["suite"]=="libero_10"), None)
    if l10_row and l10_row["recall"] < args.min_l10_recall:
        violations.append(f"LOW_L10_RECALL:{l10_row['recall']:.4f}")

    status = "PASS_C2E2C_SUITE_CALIBRATION" if not violations else "HOLD_C2E2C_SUITE_CALIBRATION"

    # Write outputs
    write_csv(out / "c2e2c_suite_thresholds.csv",
              [{"suite":s,**t} for s,t in suite_thresholds.items()],
              ["suite","tau_emit","tau_suppress","val_feasible","val_recall","val_fp_rate","val_f1"])
    write_csv(out / "c2e2c_test_metrics_by_suite.csv", test_rows,
              ["suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])

    pred_rows = []
    for i in np.where(tm_test)[0]:
        pred_rows.append({"row_index":int(ri[i]),"suite":str(suite[i]),"y":int(y[i]),
            "pred":int(pred_cal[i]),"emit_p":float(ep[i]),"suppress_p":float(sp[i]),
            "error_type":"FP" if pred_cal[i] and not y[i] else ("FN" if not pred_cal[i] and y[i] else "")})
    write_csv(out / "c2e2c_test_predictions.csv", pred_rows,
              ["row_index","suite","y","pred","emit_p","suppress_p","error_type"])
    write_csv(out / "c2e2c_violations.csv", [{"violation":v} for v in violations], ["violation"])

    # Val sweep detail
    sweep_rows = []
    for s, sw in val_sweeps.items():
        for r in sw:
            r2 = dict(r); r2["suite"] = s; sweep_rows.append(r2)
    write_csv(out / "c2e2c_val_sweep_by_suite.csv", sweep_rows,
              ["suite","tau_emit","tau_suppress","feasible","score","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])

    report = {
        "gate": "C2E2C_SUITE_CONDITIONED_CALIBRATION",
        "status": status,
        "reason": "violations=0" if not violations else f"violations={len(violations)}",
        "created_at_unix": time.time(), "runtime_seconds": time.time()-t0,
        "git_commit": args.git_commit,
        "selected_window": window,
        "suite_thresholds": suite_thresholds,
        "test_overall": overall,
        "test_by_suite": {r["suite"]: {"fp_rate":r["fp_rate"],"recall":r["recall"],"f1":r["f1"]} for r in test_rows},
        "violations": violations,
        "recommendation": ("proceed_to_C2E3_post_training_audit" if not violations else
                          "proceed_to_C2E2D_sequence_model" if obj_row and obj_row["fp_rate"]<=args.min_object_fp else
                          "object_score_unreliable_consider_c2e2d_or_c2f"),
        "boundaries": {
            "CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"NOT_PERFORMED",
            "env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED",
        },
    }
    write_json(out / "c2e2c_suite_calibration_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and fn != "checksum_report.json": csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":status,"test_overall":overall,"suite_thresholds":suite_thresholds,
        "test_by_suite":report["test_by_suite"],"violations":violations}, indent=2, sort_keys=True))
    return 0 if not violations else 1

if __name__ == "__main__":
    raise SystemExit(main())
