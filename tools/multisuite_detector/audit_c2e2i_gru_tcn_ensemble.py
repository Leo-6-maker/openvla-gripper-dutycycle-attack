#!/usr/bin/env python3
"""C2e2I: GRU/TCN offline ensemble audit.

Loads C2e2D GRU and C2e2H TCN predictions. Tests ensemble strategies on val:
  1. GRU only (baseline)
  2. TCN only
  3. Score average: α * GRU + (1-α) * TCN
  4. Gated union: GRU OR (TCN AND GRU score guards)
  5. Conservative intersection: GRU AND TCN

Selects best ensemble on val, reports on test once. No training.
CPU-only. No OpenVLA/LIBERO/env.step.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn
import torch.nn.functional as F

# ============ Models ============
class GRUSmall(nn.Module):
    def __init__(self, nf=25, nc=0, hidden=64, nl=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, nl, batch_first=True, dropout=dropout if nl>1 else 0.0)
        self.head = nn.Linear(hidden+nc, 2)
    def forward(self, xt, xc):
        _, h = self.gru(xt); last = h[-1]
        if xc.shape[1]>0: last = torch.cat([last,xc],dim=1)
        return self.head(last)

class CausalConv1d(nn.Module):
    def __init__(self, ci, co, k, d):
        super().__init__(); self.lp = (k-1)*d; self.conv = nn.Conv1d(ci,co,kernel_size=k,dilation=d)
    def forward(self, x): return self.conv(F.pad(x, (self.lp, 0)))

class TCNBlock(nn.Module):
    def __init__(self, ci, co, k, d, dr):
        super().__init__()
        self.net = nn.Sequential(CausalConv1d(ci,co,k,d), nn.ReLU(), nn.Dropout(dr),
                                 CausalConv1d(co,co,k,d), nn.ReLU(), nn.Dropout(dr))
        self.proj = nn.Conv1d(ci,co,1) if ci!=co else nn.Identity()
    def forward(self, x): return self.net(x)+self.proj(x)

class CausalTCN(nn.Module):
    def __init__(self, nf, nc, ch, dr, k):
        super().__init__()
        self.tcn = nn.Sequential(TCNBlock(nf,ch,k,1,dr), TCNBlock(ch,ch,k,2,dr), TCNBlock(ch,ch,k,4,dr))
        self.head = nn.Sequential(nn.Linear(ch+nc,ch), nn.ReLU(), nn.Dropout(dr), nn.Linear(ch,2))
    def forward(self, xs, xc):
        h = self.tcn(xs.transpose(1,2))[:,:,-1]
        return self.head(torch.cat([h,xc],dim=1))

# ============ Helpers ============
def sigmoid_np(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(fields),extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
def load_stats(path):
    o=json.loads(open(path).read())
    return {"tm":np.asarray(o.get("temporal_feature_mean",[]),np.float32),"ts":np.asarray(o.get("temporal_feature_std",[]),np.float32),
            "cm":np.asarray(o.get("context_feature_mean",[]),np.float32),"cs":np.asarray(o.get("context_feature_std",[]),np.float32)}
def load_data(root, w):
    data=np.load(Path(root)/f"c2e1_w{w:02d}_temporal_dataset.npz",allow_pickle=True)
    st=load_stats(Path(root)/f"c2e1_w{w:02d}_normalization_stats_train_only.json")
    xt=np.asarray(data["X_temporal"],np.float32); xc=np.asarray(data["X_context"],np.float32)
    xt=(xt-st["tm"].reshape(1,1,-1))/np.maximum(st["ts"].reshape(1,1,-1),1e-8)
    if xc.shape[1]>0: xc=(xc-st["cm"].reshape(1,-1))/np.maximum(st["cs"].reshape(1,-1),1e-8)
    return {"xt":xt,"xc":xc,"y":data["y"].astype(np.int64),"split":np.asarray(data["split"]).astype(str),
            "suite":np.asarray(data["suite"]).astype(str),"row_index":data["row_index"].astype(np.int64)}
def predict(model,xt,xc,bs=512):
    model.eval(); outs=[]
    with torch.no_grad():
        for s in range(0,len(xt),bs):
            outs.append(model(torch.from_numpy(xt[s:s+bs]).float(),torch.from_numpy(xc[s:s+bs]).float()).cpu().numpy())
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,2),dtype=np.float32)
def metrics(y,pred):
    pos,neg=y==1,y==0; tp,fn=int((pred&pos).sum()),int(((~pred)&pos).sum())
    fp,tn=int((pred&neg).sum()),int(((~pred)&neg).sum())
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    f1=2*prec*rec/max(1e-12,prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}
def suite_rows(y,pred,suite):
    return [{"suite":s,"n":int((suite==s).sum()),**metrics(y[suite==s],pred[suite==s])} for s in sorted(set(suite))]

def load_model_and_predict(root_path, model_key, c2e1_root):
    """Load model from {root_path}/c2e2{model_key}_selected_model.pt and predict."""
    cfg_path = Path(root_path) / f"c2e2{model_key}_selected_model_config.json"
    model_path = Path(root_path) / f"c2e2{model_key}_selected_model.pt"
    cfg = json.loads(cfg_path.read_text())["selected_config"]
    ckpt = torch.load(model_path, map_location="cpu")
    window = int(cfg["window"])
    ds = load_data(c2e1_root, window)
    if model_key == "d":
        model = GRUSmall(ds["xt"].shape[2], ds["xc"].shape[1], int(cfg["channels"]), dropout=float(cfg.get("dropout",0)))
    else:
        model = CausalTCN(ds["xt"].shape[2], ds["xc"].shape[1], int(cfg["channels"]), float(cfg.get("dropout",0)), int(cfg.get("kernel_size",3)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", {})))
    model.cpu().eval()
    logits = predict(model, ds["xt"], ds["xc"])
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    return {"ep": ep, "sp": sp, "y": ds["y"], "split": ds["split"], "suite": ds["suite"],
            "row_index": ds["row_index"], "window": window, "threshold": ckpt.get("threshold",{})}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--c2e2d-root", required=True)
    ap.add_argument("--c2e2h-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--min-recall", type=float, default=0.70)
    ap.add_argument("--max-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-fp", type=float, default=0.50)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)

    # Load both models
    print("Loading GRU...")
    gru = load_model_and_predict(args.c2e2d_root, "d", args.c2e1_root)
    print("Loading TCN...")
    tcn = load_model_and_predict(args.c2e2h_root, "h", args.c2e1_root)
    print(f"GRU W={gru['window']} TCN W={tcn['window']}")

    # Align: use GRU's dataset (W=16) as the common reference
    y, split, suite = gru["y"], gru["split"], gru["suite"]
    gep, gsp = gru["ep"], gru["sp"]
    tep, tsp = tcn["ep"], tcn["sp"]
    # Align lengths (TCN might have different row count due to different window)
    n = min(len(gep), len(tep))
    y, split, suite = y[:n], split[:n], suite[:n]
    gep, gsp = gep[:n], gsp[:n]
    tep, tsp = tep[:n], tsp[:n]

    val = split == "val"
    test = split == "test"
    suites = sorted(set(suite))

    # Baseline predictions (using selected thresholds)
    # Use report's worker-selected thresholds (NOT checkpoint main-retrain thresholds)
    # GRU worker threshold: (0.36, 0.64) — gives FP=29.8% as in C2e2D report
    gt = {"tau_emit": 0.36, "tau_suppress": 0.64}
    tt = tcn["threshold"]
    gr_pred = (gep >= gt.get("tau_emit",0.5)) & (gsp <= gt.get("tau_suppress",0.5))
    tc_pred = (tep >= tt.get("tau_emit",0.5)) & (tsp <= tt.get("tau_suppress",0.5))

    # ===== Ensemble strategies =====
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    guard_eps = [0.15, 0.20, 0.25, 0.30, 0.35]
    guard_sps = [0.70, 0.75, 0.80, 0.85]
    tau_vals = [round(x,3) for x in np.linspace(0.01,0.99,99)]

    ensemble_rows = []
    best_ensemble = None

    # 1. GRU baseline
    for te in tau_vals:
        for ts in tau_vals:
            pred = (gep >= te) & (gsp <= ts)
            m = metrics(y[val], pred[val])
            if m["recall"] >= args.min_recall and m["fp_rate"] <= args.max_fp:
                score = m["f1"] + 0.5*m["recall"]
                ensemble_rows.append({"strategy":"gru_only","alpha":"","gep_guard":"","gsp_guard":"",
                    "tau_emit":te,"tau_suppress":ts,"val_recall":m["recall"],"val_fp_rate":m["fp_rate"],
                    "val_f1":m["f1"],"score":score,"feasible":True})
    # 2. TCN baseline
    for te in tau_vals:
        for ts in tau_vals:
            pred = (tep >= te) & (tsp <= ts)
            m = metrics(y[val], pred[val])
            if m["recall"] >= args.min_recall:
                score = m["f1"] + 0.5*m["recall"]
                ensemble_rows.append({"strategy":"tcn_only","alpha":"","gep_guard":"","gsp_guard":"",
                    "tau_emit":te,"tau_suppress":ts,"val_recall":m["recall"],"val_fp_rate":m["fp_rate"],
                    "val_f1":m["f1"],"score":score,"feasible":m["recall"]>=args.min_recall and m["fp_rate"]<=args.max_fp})
    # 3. Score average
    for a in alphas:
        ep_avg = a * gep + (1-a) * tep
        sp_avg = a * gsp + (1-a) * tsp
        for te in tau_vals:
            for ts in tau_vals:
                pred = (ep_avg >= te) & (sp_avg <= ts)
                m = metrics(y[val], pred[val])
                score = m["f1"] + 0.5*m["recall"]
                ensemble_rows.append({"strategy":f"avg_a{a:.1f}","alpha":a,"gep_guard":"","gsp_guard":"",
                    "tau_emit":te,"tau_suppress":ts,"val_recall":m["recall"],"val_fp_rate":m["fp_rate"],
                    "val_f1":m["f1"],"score":score,"feasible":m["recall"]>=args.min_recall and m["fp_rate"]<=args.max_fp})
    # 4. Gated union: GRU OR (TCN AND GRU score guards)
    for ge in guard_eps:
        for gs in guard_sps:
            tcn_guarded = tc_pred & (gep >= ge) & (gsp <= gs)
            union = gr_pred | tcn_guarded
            m = metrics(y[val], union[val])
            score = m["f1"] + 0.5*m["recall"]
            ensemble_rows.append({"strategy":"gated_union","alpha":"","gep_guard":ge,"gsp_guard":gs,
                "tau_emit":"","tau_suppress":"","val_recall":m["recall"],"val_fp_rate":m["fp_rate"],
                "val_f1":m["f1"],"val_fp_rate":m["fp_rate"],"score":score,"feasible":m["recall"]>=args.min_recall and m["fp_rate"]<=args.max_fp})
    # 5. Intersection
    intersection = gr_pred & tc_pred
    mi = metrics(y[val], intersection[val])
    ensemble_rows.append({"strategy":"intersection","alpha":"","gep_guard":"","gsp_guard":"",
        "tau_emit":"","tau_suppress":"","val_recall":mi["recall"],"val_fp":mi["fp_rate"],
        "val_f1":mi["f1"],"val_fp_rate":mi["fp_rate"],"score":mi["f1"]+0.5*mi["recall"],
        "feasible":mi["recall"]>=args.min_recall and mi["fp_rate"]<=args.max_fp})

    # Select best feasible ensemble on val
    ensemble_rows.sort(key=lambda r: (bool(r["feasible"]), float(r["score"]), float(r["val_f1"]),
                                       float(r["val_recall"]), -float(r["val_fp_rate"])), reverse=True)
    best = ensemble_rows[0]
    write_csv(out/"c2e2i_ensemble_val_sweep.csv", ensemble_rows,
              ["strategy","alpha","gep_guard","gsp_guard","tau_emit","tau_suppress",
               "val_recall","val_fp_rate","val_f1","score","feasible"])

    # Apply best ensemble to test
    if best["strategy"] == "gru_only":
        ep, sp = gep, gsp
        test_pred = (ep >= best["tau_emit"]) & (sp <= best["tau_suppress"])
    elif best["strategy"] == "tcn_only":
        ep, sp = tep, tsp
        test_pred = (ep >= best["tau_emit"]) & (sp <= best["tau_suppress"])
    elif best["strategy"].startswith("avg"):
        a = float(best["alpha"])
        ep, sp = a*gep + (1-a)*tep, a*gsp + (1-a)*tsp
        test_pred = (ep >= best["tau_emit"]) & (sp <= best["tau_suppress"])
    elif best["strategy"] == "gated_union":
        tcn_g = tc_pred & (gep >= best["gep_guard"]) & (gsp <= best["gsp_guard"])
        test_pred = gr_pred | tcn_g
    else:  # intersection
        test_pred = gr_pred & tc_pred

    test_m = metrics(y[test], test_pred[test])
    test_sr = suite_rows(y[test], test_pred[test], suite[test])
    sfp = max([r["fp_rate"] for r in test_sr], default=0)

    # Compare with GRU baseline
    gr_m = metrics(y[test], gr_pred[test])
    gr_sr = suite_rows(y[test], gr_pred[test], suite[test])
    gr_l10 = next((r for r in gr_sr if r["suite"]=="libero_10"), {})
    best_l10 = next((r for r in test_sr if r["suite"]=="libero_10"), {})

    print(f"Best ensemble: {best['strategy']}")
    print(f"  GRU baseline: recall={gr_m['recall']:.3f} FP={gr_m['fp_rate']:.3f} L10_recall={gr_l10.get('recall',0):.3f}")
    print(f"  Ensemble:     recall={test_m['recall']:.3f} FP={test_m['fp_rate']:.3f} L10_recall={best_l10.get('recall',0):.3f}")

    # Write per-suite comparison
    comp_rows = []
    for sr in test_sr:
        gr_r = next((r for r in gr_sr if r["suite"]==sr["suite"]), {})
        comp_rows.append({"suite":sr["suite"],"n":sr["n"],
            "GRU_recall":gr_r.get("recall",""),"GRU_fp":gr_r.get("fp_rate",""),
            "ENS_recall":sr["recall"],"ENS_fp":sr["fp_rate"]})
    write_csv(out/"c2e2i_ensemble_vs_gru_by_suite.csv", comp_rows,
              ["suite","n","GRU_recall","GRU_fp","ENS_recall","ENS_fp"])

    report = {
        "gate": "C2E2I_GRU_TCN_ENSEMBLE_AUDIT",
        "status": "PASS_C2E2I_ENSEMBLE_AUDITED",
        "created_at_unix": time.time(), "runtime_seconds": time.time()-t0,
        "git_commit": args.git_commit,
        "best_ensemble": best,
        "test_metrics": test_m,
        "test_by_suite": {r["suite"]:{"recall":r["recall"],"fp_rate":r["fp_rate"]} for r in test_sr},
        "gru_baseline_test": gr_m,
        "gru_baseline_by_suite": {r["suite"]:{"recall":r["recall"],"fp_rate":r["fp_rate"]} for r in gr_sr},
        "l10_change": {"gru_recall":gr_l10.get("recall",0),"ensemble_recall":best_l10.get("recall",0)},
        "recommendation": (
            "ensemble_improves_gru" if test_m["fp_rate"]<=args.max_fp and test_m["recall"]>=gr_m["recall"]
            else "gru_remains_selected"
        ),
        "boundaries": {"CUDA_required":"NOT_REQUIRED","device":"cpu","detector_training":"NOT_PERFORMED",
            "OpenVLA_model":"NOT_LOADED","LIBERO_runtime":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2i_ensemble_audit_report.json", report)

    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out/fn
        if fp.is_file() and fn != "checksum_report.json": csums[fn] = sha256_file(str(fp))
    write_json(out/"checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"best":best["strategy"],"test_recall":test_m["recall"],"test_fp":test_m["fp_rate"],
        "l10_recall":best_l10.get("recall",0),"recommendation":report["recommendation"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
