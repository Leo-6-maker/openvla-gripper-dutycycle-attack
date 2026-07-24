#!/usr/bin/env python3
"""C2e2B: temporal detector failure forensics audit.

Reads C2e2 v2 selected model + C2e1 dataset. No training, no model changes,
no OpenVLA, no LIBERO, no env.step, no rollout.

Analyzes:
  - per-suite test metrics under selected global threshold
  - score distribution (emit_p, suppress_p) by split/suite/label
  - suite-conditioned threshold audit (per-suite val selection → test eval)
  - test FP/FN error rows with margins

CPU-only diagnostic gate.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

# ---- model (same as C2e2) ----
class PoolingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

# ---- io helpers ----
def sha256_file(path): h = hashlib.sha256(); [h.update(c) for c in iter(lambda: open(path,"rb").read(1<<20), b"")]; return h.hexdigest()
def write_json(path, obj): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+"\n")
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})

# ---- load C2e1 ----
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

def suite_metrics(y, pred, suite, split_name):
    return [{"split":split_name,"suite":s,"n":int((suite==s).sum()), **metrics_from_pred(y[suite==s], pred[suite==s])} for s in sorted(set(suite))]

# ---- threshold sweep ----
def sweep_thresholds(y, logits, tau_vals, min_recall, max_fp):
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
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
    ap = argparse.ArgumentParser(description="C2e2B failure forensics audit")
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--c2e2-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--min-recall", type=float, default=0.70)
    ap.add_argument("--max-fp", type=float, default=0.30)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    c2e1_root = Path(args.c2e1_root); c2e2_root = Path(args.c2e2_root)

    # ---- load selected config ----
    for name in ["c2e2_v2_selected_model_config.json", "c2e2_selected_model_config.json"]:
        cfg_path = c2e2_root / name
        if cfg_path.exists(): break
    cfg = json.loads(cfg_path.read_text())
    sel_cfg = cfg.get("selected_config", cfg)
    sel_thresh = cfg.get("selected_threshold", {"tau_emit": 0.5, "tau_suppress": 0.5})
    window = int(sel_cfg["window"])
    hidden = [int(x) for x in str(sel_cfg["hidden"]).split("-")]

    for name in ["c2e2_v2_selected_model.pt", "c2e2_selected_model.pt"]:
        model_path = c2e2_root / name
        if model_path.exists(): break

    print(f"C2e2B: window={window} hidden={hidden} threshold={sel_thresh}")

    # ---- load data ----
    dataset = load_window_dataset(c2e1_root, window)
    # Load model
    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = ckpt.get("input_dim", dataset["x"].shape[1])
    model = PoolingMLP(input_dim, hidden, sel_cfg.get("dropout", 0.0))
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    model.cpu().eval()

    # ---- predict ----
    logits = predict_logits(model, dataset["x"])
    ep = sigmoid_np(logits[:,0]); sp = sigmoid_np(logits[:,1])
    te, ts = float(sel_thresh["tau_emit"]), float(sel_thresh["tau_suppress"])
    pred = (ep >= te) & (sp <= ts)
    y, split, suite, ri = dataset["y"], dataset["split"], dataset["suite"], dataset["row_index"]
    tau_vals = [round(x,3) for x in np.linspace(0.01,0.99,99).tolist()]

    # ==== Part 1: per-suite metrics under selected threshold ====
    rows1 = []
    for sp_name in ["train","val","test"]:
        m = split == sp_name
        if m.sum() == 0: continue
        rows1.extend(suite_metrics(y[m], pred[m], suite[m], sp_name))
    write_csv(out / "c2e2b_selected_threshold_metrics_by_suite.csv", rows1,
              ["split","suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])

    # ==== Part 2: score distribution ====
    dist_rows = []
    for sp_name in ["train","val","test"]:
        m = split == sp_name
        if m.sum() == 0: continue
        for s in sorted(set(suite[m])):
            for label in [0,1]:
                mk = (suite==s) & (split==sp_name) & (y==label)
                n = int(mk.sum())
                if n == 0: continue
                ep_s = ep[mk]; sp_s = sp[mk]
                q = lambda a, qq: float(np.percentile(a, qq)) if len(a)>0 else 0.0
                dist_rows.append({"split":sp_name,"suite":s,"label":label,"n":n,
                    "emit_p_q05":q(ep_s,5),"emit_p_q25":q(ep_s,25),"emit_p_q50":q(ep_s,50),
                    "emit_p_q75":q(ep_s,75),"emit_p_q95":q(ep_s,95),"emit_p_mean":float(ep_s.mean()),
                    "suppress_p_q05":q(sp_s,5),"suppress_p_q25":q(sp_s,25),"suppress_p_q50":q(sp_s,50),
                    "suppress_p_q75":q(sp_s,75),"suppress_p_q95":q(sp_s,95),"suppress_p_mean":float(sp_s.mean()),
                    "pred_rate": float(((ep_s>=te)&(sp_s<=ts)).mean())})
    write_csv(out / "c2e2b_score_distribution_by_suite_split_label.csv", dist_rows,
              ["split","suite","label","n","emit_p_q05","emit_p_q25","emit_p_q50","emit_p_q75","emit_p_q95","emit_p_mean",
               "suppress_p_q05","suppress_p_q25","suppress_p_q50","suppress_p_q75","suppress_p_q95","suppress_p_mean","pred_rate"])

    # ==== Part 3: suite-conditioned threshold audit ====
    sc_rows = []
    for s in sorted(set(suite)):
        val_m = (suite==s) & (split=="val")
        test_m = (suite==s) & (split=="test")
        if val_m.sum() == 0: continue
        sweep = sweep_thresholds(y[val_m], logits[val_m], tau_vals, args.min_recall, args.max_fp)
        best = sweep[0]
        vte, vts = best["tau_emit"], best["tau_suppress"]
        sc_pred = (ep >= vte) & (sp <= vts)
        vmet = metrics_from_pred(y[val_m], sc_pred[val_m])
        tmet = metrics_from_pred(y[test_m], sc_pred[test_m]) if test_m.sum()>0 else {}
        sc_rows.append({"suite":s,"val_tau_emit":vte,"val_tau_suppress":vts,
            "val_feasible":best["feasible"],"val_recall":vmet["recall"],"val_fp_rate":vmet["fp_rate"],"val_f1":vmet["f1"],
            "test_recall":tmet.get("recall",""),"test_fp_rate":tmet.get("fp_rate",""),"test_f1":tmet.get("f1",""),
            "test_tp":tmet.get("tp",""),"test_fn":tmet.get("fn",""),"test_fp":tmet.get("fp",""),"test_tn":tmet.get("tn","")})
    write_csv(out / "c2e2b_suite_conditioned_threshold_audit.csv", sc_rows,
              ["suite","val_tau_emit","val_tau_suppress","val_feasible","val_recall","val_fp_rate","val_f1",
               "test_recall","test_fp_rate","test_f1","test_tp","test_fn","test_fp","test_tn"])

    # ==== Part 4: test error rows ====
    test_m = split == "test"
    err_rows = []
    for i in np.where(test_m & (pred != y))[0]:
        err_rows.append({"row_index":int(ri[i]),"suite":str(suite[i]),"split":"test",
            "y":int(y[i]),"pred":int(pred[i]),"error_type":"FP" if pred[i] and not y[i] else "FN",
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),"margin":float(ep[i]-sp[i])})
    write_csv(out / "c2e2b_test_error_rows.csv", err_rows,
              ["row_index","suite","split","y","pred","error_type","emit_p","suppress_p","margin"])

    # ---- error summary ----
    err_by_suite = defaultdict(lambda: defaultdict(int))
    for r in err_rows: err_by_suite[r["suite"]][r["error_type"]] += 1
    err_summary = {s: dict(d) for s,d in err_by_suite.items()}

    # ---- report ----
    report = {
        "gate": "C2E2B_TEMPORAL_DETECTOR_FAILURE_FORENSICS",
        "status": "PASS_C2E2B_TEMPORAL_DETECTOR_FAILURE_FORENSICS_BUILT",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time()-t0,
        "git_commit": args.git_commit,
        "selected_window": window,
        "selected_config": sel_cfg,
        "selected_threshold": sel_thresh,
        "global_test_metrics": metrics_from_pred(y[test_m], pred[test_m]),
        "test_error_counts_by_suite_type": err_summary,
        "suite_conditioned_summary": {
            s: {"val_te":r["val_tau_emit"],"val_ts":r["val_tau_suppress"],"test_recall":r["test_recall"],"test_fp":r["test_fp_rate"]}
            for s,r in zip([x["suite"] for x in sc_rows], sc_rows)
        },
        "recommendation": (
            "suite_conditioned_calibration" if all(float(r.get("test_fp_rate",1))<0.5 for r in sc_rows)
            else "consider_tcn_gru_or_observation_features"
        ),
        "boundaries": {
            "CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","simulator_runtime":"NOT_PERFORMED",
            "detector_training":"NOT_PERFORMED","env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED",
        },
    }
    write_json(out / "c2e2b_failure_forensics_report.json", report)

    # checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file(): csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":report["status"],"window":window,"runtime":report["runtime_seconds"],
        "test_fp":report["global_test_metrics"]["fp_rate"],"test_recall":report["global_test_metrics"]["recall"],
        "suite_conditioned":report["suite_conditioned_summary"],"recommendation":report["recommendation"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
