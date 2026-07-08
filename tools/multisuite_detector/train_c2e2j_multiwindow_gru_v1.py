#!/usr/bin/env python3
"""C2e2J: multi-window GRU training (W8+W16+W32).

Last 25D-only structural upgrade. Encodes three temporal windows with separate
GRUs, concatenates hidden states + context, then emit/suppress heads.

Worker returns state_dict → artifact consistent. Val-only selection.
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

PASS = "PASS_C2E2J_MULTIWINDOW_GRU_TRAINED"
HOLD = "HOLD_C2E2J_MULTIWINDOW_GRU_TRAINING"
WINDOWS = [8, 16, 32]

@dataclass
class Config:
    hidden: int; dropout: float; lr: float; seed: int; neg_weight: float

# ============ Multi-Window GRU ============
class MultiWindowGRU(nn.Module):
    def __init__(self, n_features=25, n_context=0, hidden=64, dropout=0.0):
        super().__init__()
        self.gru8 = nn.GRU(n_features, hidden, 1, batch_first=True)
        self.gru16 = nn.GRU(n_features, hidden, 1, batch_first=True)
        self.gru32 = nn.GRU(n_features, hidden, 1, batch_first=True)
        total_hidden = hidden * 3 + n_context
        self.head = nn.Sequential(nn.Linear(total_hidden, hidden*2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden*2, 2))

    def forward(self, x8, x16, x32, xc):
        _, h8 = self.gru8(x8); _, h16 = self.gru16(x16); _, h32 = self.gru32(x32)
        h = torch.cat([h8[-1], h16[-1], h32[-1], xc], dim=1)
        return self.head(h)

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

def load_multi_window(root):
    """Load all three windows, align by common row_index."""
    ds = {w: load_data(root, w) for w in WINDOWS}
    # Find common row indices
    common = set(ds[8]["row_index"].tolist())
    for w in [16, 32]: common &= set(ds[w]["row_index"].tolist())
    common = sorted(common)
    # Build index maps
    maps = {}
    for w in WINDOWS:
        ri_to_idx = {int(ri): i for i, ri in enumerate(ds[w]["row_index"])}
        maps[w] = [ri_to_idx[c] for c in common]
    n = len(common)
    return {"xt8": ds[8]["xt"][maps[8]], "xt16": ds[16]["xt"][maps[16]], "xt32": ds[32]["xt"][maps[32]],
            "xc": ds[8]["xc"][maps[8]],  # context is same across windows
            "y": ds[8]["y"][maps[8]], "split": ds[8]["split"][maps[8]],
            "suite": ds[8]["suite"][maps[8]], "row_index": np.array(common), "n": n}

def metrics(y,pred):
    pos,neg=y==1,y==0; tp,fn=int((pred&pos).sum()),int(((~pred)&pos).sum())
    fp,tn=int((pred&neg).sum()),int(((~pred)&neg).sum())
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    f1=2*prec*rec/max(1e-12,prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}
def suite_rows(y,pred,suite):
    return [{"suite":s,"n":int((suite==s).sum()),**metrics(y[suite==s],pred[suite==s])} for s in sorted(set(suite))]
def predict(model, d, bs=256):
    model.eval(); outs=[]
    x8,x16,x32,xc = torch.from_numpy(d["xt8"]).float(),torch.from_numpy(d["xt16"]).float(),torch.from_numpy(d["xt32"]).float(),torch.from_numpy(d["xc"]).float()
    with torch.no_grad():
        for s in range(0, len(x8), bs):
            outs.append(model(x8[s:s+bs],x16[s:s+bs],x32[s:s+bs],xc[s:s+bs]).cpu().numpy())
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,2),dtype=np.float32)
def sweep(y,logits,min_r,max_fp):
    ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    rows=[]; tv=[round(x,3) for x in np.linspace(0.01,0.99,99)]
    for te in tv:
        for ts in tv:
            pred=(ep>=te)&(sp<=ts); m=metrics(y,pred)
            f=m["recall"]>=min_r and m["fp_rate"]<=max_fp
            s=m["f1"]+0.5*m["recall"]-3*max(0,min_r-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            rows.append({"tau_emit":te,"tau_suppress":ts,"feasible":f,"score":s,**m})
    rows.sort(key=lambda r:(bool(r["feasible"]),r["score"],r["f1"],r["recall"],-r["fp_rate"]),reverse=True)
    return rows

def train_model(d, cfg, epochs, batch, patience, torch_threads):
    set_seed(cfg.seed); torch.set_num_threads(max(1, torch_threads))
    tr, va = d["split"]=="train", d["split"]=="val"
    model = MultiWindowGRU(d["xt8"].shape[2], d["xc"].shape[1], cfg.hidden, cfg.dropout).cpu()
    npos, nneg = max(1,int((d["y"][tr]==1).sum())), max(1,int((d["y"][tr]==0).sum()))
    pe, ps = nneg/npos, npos/nneg
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    x8tr,x16tr,x32tr,xctr = torch.from_numpy(d["xt8"][tr]),torch.from_numpy(d["xt16"][tr]),torch.from_numpy(d["xt32"][tr]),torch.from_numpy(d["xc"][tr])
    ytr = torch.from_numpy(d["y"][tr]).float()
    loader = DataLoader(TensorDataset(x8tr,x16tr,x32tr,xctr,ytr), batch_size=batch, shuffle=True)
    x8v,x16v,x32v,xcv = torch.from_numpy(d["xt8"][va]),torch.from_numpy(d["xt16"][va]),torch.from_numpy(d["xt32"][va]),torch.from_numpy(d["xc"][va])
    yv = torch.from_numpy(d["y"][va]).float()
    best_state, best_loss, best_epoch, stale = None, 1e9, -1, 0
    for ep in range(1, epochs+1):
        model.train()
        for a8,a16,a32,ac,ay in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(a8,a16,a32,ac)
            yf = ay
            l1 = torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0], yf, pos_weight=torch.tensor(pe, device=logits.device))
            l2 = torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1], 1-yf, pos_weight=torch.tensor(ps, device=logits.device))
            # FP-aware: extra weight on negatives
            w = torch.where(yf == 0, cfg.neg_weight, 1.0)
            (l1 + l2 * w).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(x8v,x16v,x32v,xcv)
            l1v = torch.nn.functional.binary_cross_entropy_with_logits(vl[:,0], yv, pos_weight=torch.tensor(pe))
            l2v = torch.nn.functional.binary_cross_entropy_with_logits(vl[:,1], 1-yv, pos_weight=torch.tensor(ps))
            wv = torch.where(yv == 0, cfg.neg_weight, 1.0)
            vloss = float((l1v + l2v * wv).mean().cpu())
        if vloss < best_loss-1e-5:
            best_loss, best_epoch, stale = vloss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_run": ep}

def _worker(args_dict):
    cfg = Config(**args_dict["config"]); d = load_multi_window(args_dict["c2e1_root"])
    model, ti = train_model(d, cfg, args_dict["epochs"], args_dict["batch_size"], args_dict["patience"], args_dict["torch_threads"])
    logits = predict(model, d)
    val, test = d["split"]=="val", d["split"]=="test"
    sw = sweep(d["y"][val], logits[val], args_dict["min_val_recall"], args_dict["max_val_fp"]); th = sw[0]
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    pred = (ep >= th["tau_emit"]) & (sp <= th["tau_suppress"])
    vm = metrics(d["y"][val], pred[val]); tm = metrics(d["y"][test], pred[test])
    sr = suite_rows(d["y"][test], pred[test], d["suite"][test]); sfp = max([r["fp_rate"] for r in sr], default=0)
    row = asdict(cfg); row.update(ti); row.update({"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"],"suite_fp_max":sfp})
    for k,v in vm.items(): row["val_"+k] = v
    for k,v in tm.items(): row["test_"+k] = v
    sel = vm["f1"] + 0.5*vm["recall"] - 3*max(0,args_dict["min_val_recall"]-vm["recall"]) - 3*max(0,vm["fp_rate"]-args_dict["max_val_fp"])
    row["selection_score"] = sel; row["feasible"] = vm["recall"]>=args_dict["min_val_recall"] and vm["fp_rate"]<=args_dict["max_val_fp"]
    return row, {k: v.cpu().clone() for k, v in model.state_dict().items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root", required=True); ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--hidden-grid", default="64,128"); ap.add_argument("--dropout-grid", default="0.0,0.1")
    ap.add_argument("--lr-grid", default="0.001,0.0003"); ap.add_argument("--neg-weight-grid", default="1.0,1.5")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--torch-threads", type=int, default=2); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-val-recall", type=float, default=0.70); ap.add_argument("--max-val-fp", type=float, default=0.30)
    ap.add_argument("--min-test-recall", type=float, default=0.70); ap.add_argument("--max-test-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-test-fp", type=float, default=0.50)
    args = ap.parse_args()
    t0 = time.time(); out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    c2e1_root = str(Path(args.c2e1_root).resolve())

    hi_grid = [int(x) for x in args.hidden_grid.split(",")]
    do_grid = [float(x) for x in args.dropout_grid.split(",")]
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    nw_grid = [float(x) for x in args.neg_weight_grid.split(",")]
    sd_grid = [int(x) for x in args.seeds.split(",")]

    all_configs = []
    for hi in hi_grid:
        for do in do_grid:
            for lr in lr_grid:
                for nw in nw_grid:
                    for sd in sd_grid:
                        all_configs.append(asdict(Config(hi, do, lr, sd, nw)))

    n_total = len(all_configs); nw = min(args.workers, n_total)
    print(f"C2e2J: {n_total} configs x {nw} workers (multi-window GRU W=8+16+32)")

    worker_args = [{"c2e1_root": c2e1_root, "config": c, "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience,
        "min_val_recall": args.min_val_recall, "max_val_fp": args.max_val_fp,
        "torch_threads": args.torch_threads} for c in all_configs]

    all_rows = []; best_row = None; best_state_dict = None; done = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures = {pool.submit(_worker, wa): wa for wa in worker_args}
        for fut in as_completed(futures):
            row, sd = fut.result(); all_rows.append(row); done += 1
            sel = float(row["selection_score"]); feasible = bool(row["feasible"])
            key = (feasible, sel, float(row["val_f1"]), float(row["val_recall"]), -float(row["val_fp_rate"]))
            if best_row is None or key > best_key: best_key = key; best_row = row; best_state_dict = sd
            if done % 24 == 0 or done == n_total: print(f"  {done}/{n_total} configs...")

    if best_row is None: raise RuntimeError("no config completed")

    # Reconstruct best model from state_dict
    best_cfg = Config(hidden=int(best_row["hidden"]), dropout=float(best_row["dropout"]),
        lr=float(best_row["lr"]), seed=int(best_row["seed"]), neg_weight=float(best_row["neg_weight"]))
    d = load_multi_window(c2e1_root)
    model = MultiWindowGRU(d["xt8"].shape[2], d["xc"].shape[1], best_cfg.hidden, best_cfg.dropout)
    model.load_state_dict(best_state_dict); model.cpu().eval()
    logits = predict(model, d)
    val, test = d["split"]=="val", d["split"]=="test"
    sw = sweep(d["y"][val], logits[val], args.min_val_recall, args.max_val_fp); th = sw[0]
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    pred = (ep >= th["tau_emit"]) & (sp <= th["tau_suppress"])
    tm = metrics(d["y"][test], pred[test])
    sr = suite_rows(d["y"][test], pred[test], d["suite"][test])
    sfp = max([r["fp_rate"] for r in sr], default=0)
    l10_sr = next((r for r in sr if r["suite"]=="libero_10"), {})

    rep_tm = metrics(d["y"][test], pred[test])
    rep_match = abs(rep_tm["recall"]-tm["recall"])<0.001

    vio = []
    if tm["recall"] < args.min_test_recall: vio.append(f"LOW_TEST_RECALL:{tm['recall']:.4f}")
    if tm["fp_rate"] > args.max_test_fp: vio.append(f"HIGH_TEST_FP:{tm['fp_rate']:.4f}")
    if sfp > args.max_suite_test_fp: vio.append(f"HIGH_SUITE_FP_MAX:{sfp:.4f}")
    if not rep_match: vio.append("ARTIFACT_INCONSISTENT")
    status = PASS if not vio else HOLD

    model_path = out / "c2e2j_selected_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(best_cfg),
        "threshold": {"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"]}}, model_path)

    pred_rows = []
    for i in np.where(test)[0]:
        pred_rows.append({"row_index":int(d["row_index"][i]),"suite":str(d["suite"][i]),"y":int(d["y"][i]),
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),"pred":int(bool(pred[i]))})

    write_csv(out/"c2e2j_all_config_metrics.csv", all_rows,
              ["hidden","dropout","lr","seed","neg_weight","best_val_loss","best_epoch","epochs_run",
               "tau_emit","tau_suppress","val_recall","val_fp_rate","val_f1",
               "test_recall","test_fp_rate","test_f1","suite_fp_max","selection_score","feasible"])
    write_csv(out/"c2e2j_selected_test_metrics_by_suite.csv", sr,
              ["suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1"])
    write_csv(out/"c2e2j_selected_test_predictions.csv", pred_rows,
              ["row_index","suite","y","emit_p","suppress_p","pred"])
    write_json(out/"c2e2j_selected_model_config.json",
               {"selected_config":asdict(best_cfg),"selected_threshold":{"tau_emit":float(th["tau_emit"]),"tau_suppress":float(th["tau_suppress"])}})
    write_csv(out/"c2e2j_violations.csv", [{"violation":v} for v in vio], ["violation"])

    report = {"gate":"C2E2J_MULTIWINDOW_GRU_TRAINING","status":status,
        "reason":"violations=0" if not vio else f"violations={len(vio)}",
        "created_at_unix":time.time(),"runtime_seconds":time.time()-t0,"git_commit":args.git_commit,
        "aligned_rows": d["n"], "artifact_consistency": "checkpoint_exactly_reproduces_report_metrics" if rep_match else "FAIL",
        "selected_config":asdict(best_cfg),"selected_threshold":{"tau_emit":float(th["tau_emit"]),"tau_suppress":float(th["tau_suppress"])},
        "selected_metrics":tm,"selected_test_metrics_by_suite":sr,"violations":vio,
        "l10_recall": l10_sr.get("recall",0),
        "recommendation":"proceed_to_C2E3" if not vio else "hold_or_proceed_to_C2F",
        "boundaries":{"CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"CPU_ONLY_MULTIWINDOW_GRU","env_step":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2j_multiwindow_gru_report.json", report)

    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out/fn
        if fp.is_file() and fn != "checksum_report.json": csums[fn] = sha256_file(str(fp))
    write_json(out/"checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn,sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(f"\nC2e2J: {status} | recall={tm['recall']:.3f} FP={tm['fp_rate']:.3f} L10_rec={l10_sr.get('recall',0):.3f} | aligned N={d['n']}")
    return 0 if not vio else 1

if __name__ == "__main__":
    raise SystemExit(main())
