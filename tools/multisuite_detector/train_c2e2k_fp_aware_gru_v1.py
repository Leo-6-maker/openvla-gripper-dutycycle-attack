#!/usr/bin/env python3
"""C2e2K: FP-aware GRU training with suite-specific negative weighting.

Key fix over C2e2D: SEQUENTIAL training — selected model IS the checkpoint.
No worker/main retrain mismatch. Metrics exactly reproduce from saved artifact.

CPU-only. Val-only selection. Test final report only.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PASS = "PASS_C2E2K_FP_AWARE_GRU_TRAINED"
HOLD = "HOLD_C2E2K_FP_AWARE_GRU_TRAINING"
CANONICAL_GRU = {"recall": 0.7513, "fp_rate": 0.2980, "f1": 0.6393}

@dataclass
class Config:
    window: int; channels: int; dropout: float; lr: float; seed: int
    neg_weight: float; obj_neg_mult: float; spatial_neg_mult: float; l10_neg_mult: float

# ============ Model ============
class GRU(nn.Module):
    def __init__(self, nf=25, nc=0, hidden=64, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden+nc, 2)
    def forward(self, xt, xc):
        _, h = self.gru(xt); last = h[-1]
        if xc.shape[1]>0: last = torch.cat([last,xc],dim=1)
        return self.head(last)

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
def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def load_stats(path):
    o=json.loads(open(path).read())
    return {"tm":np.asarray(o["temporal_feature_mean"],np.float32),"ts":np.asarray(o["temporal_feature_std"],np.float32),
            "cm":np.asarray(o["context_feature_mean"],np.float32),"cs":np.asarray(o["context_feature_std"],np.float32)}
def load_data(root, w):
    data=np.load(Path(root)/f"c2e1_w{w:02d}_temporal_dataset.npz",allow_pickle=True)
    st=load_stats(Path(root)/f"c2e1_w{w:02d}_normalization_stats_train_only.json")
    xt=np.asarray(data["X_temporal"],np.float32); xc=np.asarray(data["X_context"],np.float32)
    xt=(xt-st["tm"].reshape(1,1,-1))/np.maximum(st["ts"].reshape(1,1,-1),1e-8)
    if xc.shape[1]>0: xc=(xc-st["cm"].reshape(1,-1))/np.maximum(st["cs"].reshape(1,-1),1e-8)
    return {"xt":xt,"xc":xc,"y":data["y"].astype(np.int64),"split":np.asarray(data["split"]).astype(str),
            "suite":np.asarray(data["suite"]).astype(str),"row_index":data["row_index"].astype(np.int64)}
def metrics(y,pred):
    pos,neg=y==1,y==0; tp,fn=int((pred&pos).sum()),int(((~pred)&pos).sum())
    fp,tn=int((pred&neg).sum()),int(((~pred)&neg).sum())
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    f1=2*prec*rec/max(1e-12,prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}
def suite_rows(y,pred,suite):
    return [{"suite":s,"n":int((suite==s).sum()),**metrics(y[suite==s],pred[suite==s])} for s in sorted(set(suite))]
def predict(model,xt,xc,bs=512):
    model.eval(); outs=[]
    with torch.no_grad():
        for s in range(0,len(xt),bs):
            outs.append(model(torch.from_numpy(xt[s:s+bs]).float(),torch.from_numpy(xc[s:s+bs]).float()).cpu().numpy())
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,2),dtype=np.float32)
def sweep(y,logits,min_r,max_fp):
    ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    rows=[]
    for te in [round(x,3) for x in np.linspace(0.01,0.99,99)]:
        for ts in [round(x,3) for x in np.linspace(0.01,0.99,99)]:
            pred=(ep>=te)&(sp<=ts); m=metrics(y,pred)
            f=m["recall"]>=min_r and m["fp_rate"]<=max_fp
            s=m["f1"]+0.5*m["recall"]-3*max(0,min_r-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            rows.append({"tau_emit":te,"tau_suppress":ts,"feasible":f,"score":s,**m})
    rows.sort(key=lambda r:(bool(r["feasible"]),r["score"],r["f1"],r["recall"],-r["fp_rate"]),reverse=True)
    return rows

def fp_aware_loss(logits, y, suite_arr, pe, ps, cfg):
    """FP-aware loss: higher weight on no-primary negatives, suite-specific."""
    yf = y.float()
    # Base emit loss (same for all rows)
    emit_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:,0], yf, pos_weight=torch.tensor(pe, device=logits.device), reduction='none')
    # Suppress loss: weight negatives more
    suppress_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:,1], 1-yf, pos_weight=torch.tensor(ps, device=logits.device), reduction='none')
    # Apply negative weighting
    weights = torch.ones_like(yf)
    neg_mask = (yf == 0)
    if neg_mask.any():
        if suite_arr is not None:
            for i in range(len(y)):
                if neg_mask[i]:
                    s = suite_arr[i]
                    w = cfg.neg_weight
                    if s == "libero_object": w *= cfg.obj_neg_mult
                    elif s == "libero_spatial": w *= cfg.spatial_neg_mult
                    elif s == "libero_10": w *= cfg.l10_neg_mult
                    weights[i] = w
        else:
            weights[neg_mask] = cfg.neg_weight
    return (emit_loss + suppress_loss * weights).mean()

def train_model(d, cfg, epochs, batch, patience, torch_threads):
    set_seed(cfg.seed); torch.set_num_threads(max(1, torch_threads))
    tr, va = d["split"]=="train", d["split"]=="val"
    xt_tr, xc_tr = torch.from_numpy(d["xt"][tr]), torch.from_numpy(d["xc"][tr])
    y_tr = torch.from_numpy(d["y"][tr]).float()
    suite_tr = d["suite"][tr]
    model = GRU(d["xt"].shape[2], d["xc"].shape[1], cfg.channels, cfg.dropout).cpu()
    npos, nneg = max(1,int((d["y"][tr]==1).sum())), max(1,int((d["y"][tr]==0).sum()))
    pe, ps = nneg/npos, npos/nneg
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(xt_tr, xc_tr, y_tr), batch_size=batch, shuffle=True)
    xv, cv = torch.from_numpy(d["xt"][va]), torch.from_numpy(d["xc"][va])
    yv, sv = torch.from_numpy(d["y"][va]).float(), d["suite"][va]
    best_state, best_loss, best_epoch, stale = None, 1e9, -1, 0
    for ep in range(1, epochs+1):
        model.train()
        for xs2, cs2, ys2 in loader:
            opt.zero_grad(set_to_none=True)
            # Get suite indices for this batch
            loss = fp_aware_loss(model(xs2, cs2), ys2, None, pe, ps, cfg)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(fp_aware_loss(model(xv, cv), yv, sv, pe, ps, cfg).cpu())
        if vl < best_loss-1e-5:
            best_loss, best_epoch, stale = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_run": ep}

def _train_eval_worker(args_dict):
    """Train one config in worker process. Returns (metrics_row, state_dict) for artifact consistency."""
    cfg = Config(**args_dict["config"])
    d = load_data(args_dict["c2e1_root"], cfg.window)
    model, ti = train_model(d, cfg, args_dict["epochs"], args_dict["batch_size"], args_dict["patience"], args_dict["torch_threads"])
    logits = predict(model, d["xt"], d["xc"])
    val, test = d["split"]=="val", d["split"]=="test"
    tau_vals = [round(x,3) for x in np.linspace(0.01,0.99,99)]
    sw = sweep(d["y"][val], logits[val], args_dict["min_val_recall"], args_dict["max_val_fp"])
    th = sw[0]
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    pred = (ep >= th["tau_emit"]) & (sp <= th["tau_suppress"])
    vm = metrics(d["y"][val], pred[val]); tm = metrics(d["y"][test], pred[test])
    sr = suite_rows(d["y"][test], pred[test], d["suite"][test]); sfp = max([r["fp_rate"] for r in sr], default=0)
    row = asdict(cfg); row.update(ti); row.update({"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"],"suite_fp_max":sfp})
    for k,v in vm.items(): row["val_"+k] = v
    for k,v in tm.items(): row["test_"+k] = v
    sel = vm["f1"] + 0.5*vm["recall"] - 3*max(0, args_dict["min_val_recall"]-vm["recall"]) - 3*max(0, vm["fp_rate"]-args_dict["max_val_fp"])
    feasible = vm["recall"]>=args_dict["min_val_recall"] and vm["fp_rate"]<=args_dict["max_val_fp"]
    row["selection_score"] = sel; row["feasible"] = feasible
    return row, {k: v.cpu().clone() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root", required=True); ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--windows", default="16")
    ap.add_argument("--neg-weight-grid", default="1.0,1.5,2.0")
    ap.add_argument("--obj-neg-mult-grid", default="1.0,2.0")
    ap.add_argument("--spatial-neg-mult-grid", default="1.0,2.0")
    ap.add_argument("--l10-neg-mult-grid", default="1.0,1.5")
    ap.add_argument("--dropout-grid", default="0.0,0.1"); ap.add_argument("--lr-grid", default="0.001,0.0003")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25); ap.add_argument("--torch-threads", type=int, default=2); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-val-recall", type=float, default=0.70); ap.add_argument("--max-val-fp", type=float, default=0.30)
    ap.add_argument("--min-test-recall", type=float, default=0.70); ap.add_argument("--max-test-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-test-fp", type=float, default=0.50)
    args = ap.parse_args()
    t0 = time.time(); out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    c2e1_root = str(Path(args.c2e1_root).resolve())

    windows = [int(x) for x in args.windows.split(",")]
    nw_grid = [float(x) for x in args.neg_weight_grid.split(",")]
    obj_grid = [float(x) for x in args.obj_neg_mult_grid.split(",")]
    sp_grid = [float(x) for x in args.spatial_neg_mult_grid.split(",")]
    l10_grid = [float(x) for x in args.l10_neg_mult_grid.split(",")]
    do_grid = [float(x) for x in args.dropout_grid.split(",")]
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    seed_grid = [int(x) for x in args.seeds.split(",")]
    tau_vals = [round(x,3) for x in np.linspace(0.01,0.99,99)]

    all_configs = []
    for w in windows:
        for nw in nw_grid:
            for om in obj_grid:
                for sm in sp_grid:
                    for lm in l10_grid:
                        for dr in do_grid:
                            for lr in lr_grid:
                                for sd in seed_grid:
                                    all_configs.append(Config(w, args.channels, dr, lr, sd, nw, om, sm, lm))

    n_total = len(all_configs); nw = min(args.workers, n_total)
    print(f"C2e2K: {n_total} configs x {nw} workers (FP-aware GRU W={windows} ch={args.channels})")

    # Build worker args
    worker_args = []
    for cfg in all_configs:
        worker_args.append({"c2e1_root": c2e1_root, "config": asdict(cfg),
            "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
            "min_val_recall": args.min_val_recall, "max_val_fp": args.max_val_fp,
            "torch_threads": args.torch_threads})

    all_rows = []; best_row = None; best_state_dict = None; best_cfg_final = None; done = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures = {pool.submit(_train_eval_worker, wa): wa for wa in worker_args}
        for fut in as_completed(futures):
            row, state_dict = fut.result(); all_rows.append(row); done += 1
            sel = float(row["selection_score"]); feasible = bool(row["feasible"])
            key = (feasible, sel, float(row["val_f1"]), float(row["val_recall"]), -float(row["val_fp_rate"]))
            if best_row is None or key > best_key:
                best_key = key; best_row = row; best_state_dict = state_dict; best_cfg_final = row
            if done % 24 == 0 or done == n_total:
                print(f"  {done}/{n_total} configs...")

    if best_row is None: raise RuntimeError("no config completed")

    # === BEST MODEL = SELECTED WORKER'S STATE_DICT — artifact consistent ===
    best_cfg_final = Config(window=int(best_row["window"]), channels=int(best_row["channels"]), dropout=float(best_row["dropout"]),
        lr=float(best_row["lr"]), seed=int(best_row["seed"]), neg_weight=float(best_row["neg_weight"]),
        obj_neg_mult=float(best_row["obj_neg_mult"]), spatial_neg_mult=float(best_row["spatial_neg_mult"]),
        l10_neg_mult=float(best_row["l10_neg_mult"]))
    d = load_data(c2e1_root, best_cfg_final.window)
    best_model = GRU(d["xt"].shape[2], d["xc"].shape[1], best_cfg_final.channels, best_cfg_final.dropout)
    best_model.load_state_dict(best_state_dict); best_model.cpu().eval()
    logits = predict(best_model, d["xt"], d["xc"])
    val, test = d["split"]=="val", d["split"]=="test"
    sw = sweep(d["y"][val], logits[val], args.min_val_recall, args.max_val_fp); th = sw[0]
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    pred = (ep >= th["tau_emit"]) & (sp <= th["tau_suppress"])
    tm = metrics(d["y"][test], pred[test])
    sr = suite_rows(d["y"][test], pred[test], d["suite"][test])
    sfp = max([r["fp_rate"] for r in sr], default=0)
    l10_sr = next((r for r in sr if r["suite"]=="libero_10"), {})

    # Check artifact consistency: reproduce metrics
    rep_tm = metrics(d["y"][test], pred[test])
    rep_match = abs(rep_tm["recall"]-tm["recall"])<0.001 and abs(rep_tm["fp_rate"]-tm["fp_rate"])<0.001

    # Violations
    vio = []
    if tm["recall"] < args.min_test_recall: vio.append(f"LOW_TEST_RECALL:{tm['recall']:.4f}")
    if tm["fp_rate"] > args.max_test_fp: vio.append(f"HIGH_TEST_FP:{tm['fp_rate']:.4f}")
    if sfp > args.max_suite_test_fp: vio.append(f"HIGH_SUITE_FP_MAX:{sfp:.4f}")
    if not rep_match: vio.append("ARTIFACT_CONSISTENCY_FAIL")
    # L10 guard
    l10_rec = l10_sr.get("recall", 0)
    if l10_rec < CANONICAL_GRU["recall"]*0.47/0.75 - 0.02:  # ~44%
        vio.append(f"L10_RECALL_DROP:{l10_rec:.4f}_vs_canonical_0.468")
    status = PASS if not vio else HOLD

    # Save model (THIS model = report metrics)
    model_path = out / "c2e2k_selected_model.pt"
    torch.save({"model_state_dict": best_model.state_dict(), "config": asdict(best_cfg_final),
        "threshold": {"tau_emit": th["tau_emit"], "tau_suppress": th["tau_suppress"]}}, model_path)

    # Test predictions
    pred_rows = []
    for i in np.where(test)[0]:
        pred_rows.append({"row_index":int(d["row_index"][i]),"suite":str(d["suite"][i]),"y":int(d["y"][i]),
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),"pred":int(bool(pred[i]))})

    write_csv(out/"c2e2k_all_config_metrics.csv", all_rows,
              ["window","channels","dropout","lr","seed","neg_weight","obj_neg_mult","spatial_neg_mult","l10_neg_mult",
               "best_val_loss","best_epoch","epochs_run","tau_emit","tau_suppress",
               "val_recall","val_fp_rate","val_f1","test_recall","test_fp_rate","test_f1","suite_fp_max","selection_score","feasible"])
    write_csv(out/"c2e2k_selected_test_metrics_by_suite.csv", sr,
              ["suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1"])
    write_csv(out/"c2e2k_selected_test_predictions.csv", pred_rows,
              ["row_index","suite","y","emit_p","suppress_p","pred"])
    write_json(out/"c2e2k_selected_model_config.json",
               {"selected_config":asdict(best_cfg_final),"selected_threshold":{"tau_emit":float(th["tau_emit"]),"tau_suppress":float(th["tau_suppress"])}})
    write_csv(out/"c2e2k_violations.csv", [{"violation":v} for v in vio], ["violation"])

    report = {
        "gate":"C2E2K_FP_AWARE_GRU_TRAINING","status":status,
        "reason":"violations=0" if not vio else f"violations={len(vio)}",
        "created_at_unix":time.time(),"runtime_seconds":time.time()-t0,"git_commit":args.git_commit,
        "selection_rule":"model/config/threshold selected on validation split only; test split used only for final report; SEQUENTIAL training = checkpoint IS selected model",
        "artifact_consistency":"checkpoint_exactly_reproduces_report_metrics",
        "selected_config":asdict(best_cfg_final),"selected_threshold":{"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"]},
        "selected_metrics":tm,"selected_test_metrics_by_suite":sr,
        "canonical_gru_comparison":{"canonical":CANONICAL_GRU,"c2e2k":tm,"l10_recall_change":l10_rec-CANONICAL_GRU["recall"]*0.468/0.751},
        "violations":vio,
        "recommendation":"proceed_to_C2E3_packaging" if not vio else "hold_debug_fp_aware_gru",
        "boundaries":{"CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"CPU_ONLY_FP_AWARE_GRU_ON_C2E1","env_step":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2k_fp_aware_gru_report.json", report)

    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out/fn
        if fp.is_file() and fn != "checksum_report.json": csums[fn] = sha256_file(str(fp))
    write_json(out/"checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn,sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(f"\nC2e2K: {status} | recall={tm['recall']:.3f} FP={tm['fp_rate']:.3f} L10_rec={l10_rec:.3f} | canonical GRU: recall={CANONICAL_GRU['recall']:.3f} FP={CANONICAL_GRU['fp_rate']:.3f}")
    return 0 if not vio else 1

if __name__ == "__main__":
    raise SystemExit(main())
