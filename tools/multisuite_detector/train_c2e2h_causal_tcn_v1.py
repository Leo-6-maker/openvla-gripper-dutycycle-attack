#!/usr/bin/env python3
"""C2e2H causal TCN training on C2e1 temporal tensors.

Offline CPU-only detector experiment. The script loads C2e1 NPZ tensors,
trains small causal TCN models, selects model and thresholds on validation only,
and reports the selected model on test once.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))
from train_c2e2_temporal_pooling_mlp_v1 import metrics_from_pred, sigmoid_np  # type: ignore

PASS = "PASS_C2E2H_CAUSAL_TCN_TRAINED"
HOLD = "HOLD_C2E2H_CAUSAL_TCN_TRAINING"

@dataclass
class Config:
    window: int
    channels: int
    dropout: float
    lr: float
    seed: int
    kernel_size: int = 3

class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel, dilation=dilation)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))

class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(c_in, c_out, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(c_out, c_out, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.proj(x)

class CausalTCN(nn.Module):
    def __init__(self, n_temporal: int, n_context: int, channels: int, dropout: float, kernel: int):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(n_temporal, channels, kernel, 1, dropout),
            TCNBlock(channels, channels, kernel, 2, dropout),
            TCNBlock(channels, channels, kernel, 4, dropout),
        )
        self.head = nn.Sequential(nn.Linear(channels + n_context, channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(channels, 2))
    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x_seq.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([h, x_ctx], dim=1))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")

def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def parse_list(raw: str, typ=float):
    return [typ(x.strip()) for x in str(raw).split(",") if x.strip()]

def load_stats(path: Path) -> Dict[str, np.ndarray]:
    o = json.loads(path.read_text())
    return {"tm":np.asarray(o["temporal_feature_mean"],np.float32), "ts":np.asarray(o["temporal_feature_std"],np.float32), "cm":np.asarray(o["context_feature_mean"],np.float32), "cs":np.asarray(o["context_feature_std"],np.float32)}

def load_data(root: Path, w: int) -> Dict[str, Any]:
    data = np.load(root / f"c2e1_w{w:02d}_temporal_dataset.npz", allow_pickle=True)
    st = load_stats(root / f"c2e1_w{w:02d}_normalization_stats_train_only.json")
    xt = data["X_temporal"].astype(np.float32); xc = data["X_context"].astype(np.float32)
    xt = (xt - st["tm"].reshape(1,1,-1)) / np.maximum(st["ts"].reshape(1,1,-1), 1e-8)
    if xc.shape[1] > 0: xc = (xc - st["cm"].reshape(1,-1)) / np.maximum(st["cs"].reshape(1,-1), 1e-8)
    if not np.isfinite(xt).all() or not np.isfinite(xc).all(): raise ValueError("non-finite normalized input")
    return {"xt":xt.astype(np.float32), "xc":xc.astype(np.float32), "y":data["y"].astype(np.int64), "split":np.asarray(data["split"]).astype(str), "suite":np.asarray(data["suite"]).astype(str), "row_index":data["row_index"].astype(np.int64)}

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def loss_fn(logits: torch.Tensor, y: torch.Tensor, pe: float, ps: float) -> torch.Tensor:
    yf = y.float()
    a = F.binary_cross_entropy_with_logits(logits[:,0], yf, pos_weight=torch.tensor(pe, device=logits.device))
    b = F.binary_cross_entropy_with_logits(logits[:,1], 1-yf, pos_weight=torch.tensor(ps, device=logits.device))
    return a + b

def train_model(d: Dict[str,Any], cfg: Config, epochs: int, batch: int, patience: int) -> tuple[CausalTCN, Dict[str,Any]]:
    seed_all(cfg.seed)
    tr = d["split"] == "train"; va = d["split"] == "val"
    model = CausalTCN(d["xt"].shape[2], d["xc"].shape[1], cfg.channels, cfg.dropout, cfg.kernel_size).cpu()
    npos=max(1,int((d["y"][tr]==1).sum())); nneg=max(1,int((d["y"][tr]==0).sum()))
    pe=nneg/npos; ps=npos/nneg
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(d["xt"][tr]), torch.from_numpy(d["xc"][tr]), torch.from_numpy(d["y"][tr]).float())
    loader=DataLoader(ds,batch_size=batch,shuffle=True)
    xv=torch.from_numpy(d["xt"][va]); cv=torch.from_numpy(d["xc"][va]); yv=torch.from_numpy(d["y"][va]).float()
    best=None; best_loss=1e9; best_epoch=0; stale=0
    for ep in range(1,epochs+1):
        model.train()
        for xs,cs,ys in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xs,cs),ys,pe,ps); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(loss_fn(model(xv,cv),yv,pe,ps).cpu())
        if vl < best_loss-1e-5:
            best_loss=vl; best_epoch=ep; stale=0; best={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best: model.load_state_dict(best)
    return model,{"best_val_loss":best_loss,"best_epoch":best_epoch}

def predict(model: CausalTCN, d: Dict[str,Any]) -> np.ndarray:
    out=[]; model.eval()
    with torch.no_grad():
        for i in range(0,len(d["y"]),4096): out.append(model(torch.from_numpy(d["xt"][i:i+4096]), torch.from_numpy(d["xc"][i:i+4096])).numpy())
    return np.concatenate(out,0)

def sweep(y: np.ndarray, logits: np.ndarray, min_recall: float, max_fp: float) -> List[Dict[str,Any]]:
    ep=sigmoid_np(logits[:,0]); sp=sigmoid_np(logits[:,1]); rows=[]
    for te in np.linspace(0.01,0.99,99):
        for ts in np.linspace(0.01,0.99,99):
            pred=(ep>=te)&(sp<=ts); m=metrics_from_pred(y,pred)
            score=m["f1"]+0.5*m["recall"]-3*max(0,min_recall-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            r={"tau_emit":round(float(te),3),"tau_suppress":round(float(ts),3),"score":score,"feasible":m["recall"]>=min_recall and m["fp_rate"]<=max_fp}; r.update(m); rows.append(r)
    rows.sort(key=lambda r:(r["feasible"],r["score"],r["f1"],r["recall"],-r["fp_rate"]), reverse=True); return rows

def suite_rows(y:np.ndarray,pred:np.ndarray,suite:np.ndarray)->List[Dict[str,Any]]:
    rows=[]
    for s in sorted(set(suite.tolist())):
        m=suite==s; r={"suite":s,"n":int(m.sum())}; r.update(metrics_from_pred(y[m],pred[m])); rows.append(r)
    return rows

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c2e1-root",required=True); ap.add_argument("--output-root",required=True); ap.add_argument("--git-commit",required=True)
    ap.add_argument("--windows",default="8,16,32"); ap.add_argument("--channels-grid",default="64,128"); ap.add_argument("--dropout-grid",default="0.0,0.1"); ap.add_argument("--lr-grid",default="0.001,0.0003"); ap.add_argument("--seeds",default="0,1,2")
    ap.add_argument("--epochs",type=int,default=200); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--patience",type=int,default=25); ap.add_argument("--torch-threads",type=int,default=16)
    ap.add_argument("--min-val-recall",type=float,default=0.70); ap.add_argument("--max-val-fp",type=float,default=0.30); ap.add_argument("--min-test-recall",type=float,default=0.70); ap.add_argument("--max-test-fp",type=float,default=0.30); ap.add_argument("--max-suite-test-fp",type=float,default=0.50)
    args=ap.parse_args(); start=time.time(); torch.set_num_threads(args.torch_threads)
    root=Path(args.c2e1_root).expanduser().resolve(); out=Path(args.output_root).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    best=None; best_model=None; best_data=None; rows=[]
    for w in parse_list(args.windows,int):
        d=load_data(root,w)
        for ch in parse_list(args.channels_grid,int):
          for dr in parse_list(args.dropout_grid,float):
           for lr in parse_list(args.lr_grid,float):
            for sd in parse_list(args.seeds,int):
              cfg=Config(w,ch,dr,lr,sd); model,info=train_model(d,cfg,args.epochs,args.batch_size,args.patience); logits=predict(model,d)
              val=d["split"]=="val"; test=d["split"]=="test"; sw=sweep(d["y"][val],logits[val],args.min_val_recall,args.max_val_fp); th=sw[0]
              ep=sigmoid_np(logits[:,0]); sp=sigmoid_np(logits[:,1]); pred=(ep>=th["tau_emit"])&(sp<=th["tau_suppress"])
              vm=metrics_from_pred(d["y"][val],pred[val]); tm=metrics_from_pred(d["y"][test],pred[test]); sr=suite_rows(d["y"][test],pred[test],d["suite"][test]); sfp=max([x["fp_rate"] for x in sr], default=0)
              row=asdict(cfg); row.update(info); row.update({"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"],"suite_fp_max":sfp})
              for k,v in vm.items(): row["val_"+k]=v
              for k,v in tm.items(): row["test_"+k]=v
              rows.append(row); key=(vm["recall"]>=args.min_val_recall and vm["fp_rate"]<=args.max_val_fp, vm["f1"]+0.5*vm["recall"]-3*max(0,args.min_val_recall-vm["recall"])-3*max(0,vm["fp_rate"]-args.max_val_fp), vm["f1"], vm["recall"], -vm["fp_rate"])
              if best is None or key>best["key"]: best={"key":key,"cfg":cfg,"row":row,"th":th,"logits":logits,"suite":sr,"pred":pred}; best_model=model; best_data=d
              print(json.dumps({"cfg":asdict(cfg),"val_recall":vm["recall"],"val_fp":vm["fp_rate"],"test_recall_report":tm["recall"],"test_fp_report":tm["fp_rate"]},sort_keys=True))
    assert best and best_model and best_data
    b=best; test=best_data["split"]=="test"; pred_rows=[]; ep=sigmoid_np(b["logits"][:,0]); sp=sigmoid_np(b["logits"][:,1])
    for i in np.where(test)[0].tolist(): pred_rows.append({"row_index":int(best_data["row_index"][i]),"suite":str(best_data["suite"][i]),"y":int(best_data["y"][i]),"emit_p":float(ep[i]),"suppress_p":float(sp[i]),"pred":int(bool(b["pred"][i]))})
    vio=[]; r=b["row"]
    if r["test_recall"]<args.min_test_recall: vio.append(f"LOW_TEST_RECALL:{r['test_recall']:.6f}")
    if r["test_fp_rate"]>args.max_test_fp: vio.append(f"HIGH_TEST_FP:{r['test_fp_rate']:.6f}")
    if r["suite_fp_max"]>args.max_suite_test_fp: vio.append(f"HIGH_SUITE_FP:{r['suite_fp_max']:.6f}")
    status=PASS if not vio else HOLD
    model_path=out/"c2e2h_selected_model.pt"; torch.save({"state_dict":best_model.state_dict(),"config":asdict(b["cfg"]),"threshold":{"tau_emit":b["th"]["tau_emit"],"tau_suppress":b["th"]["tau_suppress"]},"input_dim_temporal":25,"context_dim":int(best_data["xc"].shape[1])},model_path)
    write_csv(out/"c2e2h_all_config_metrics.csv",rows,["window","channels","dropout","lr","seed","kernel_size","best_val_loss","best_epoch","tau_emit","tau_suppress","val_recall","val_fp_rate","val_f1","test_recall","test_fp_rate","test_f1","suite_fp_max"])
    write_csv(out/"c2e2h_selected_test_metrics_by_suite.csv",b["suite"],["suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])
    write_csv(out/"c2e2h_selected_test_predictions.csv",pred_rows,["row_index","suite","y","emit_p","suppress_p","pred"])
    write_csv(out/"c2e2h_violations.csv",[{"violation":v} for v in vio],["violation"])
    report={"gate":"C2E2H_CAUSAL_TCN_TRAINING","status":status,"reason":"hard_violation_count=0" if not vio else f"hard_violation_count={len(vio)}","runtime_seconds":time.time()-start,"git_commit":args.git_commit,"selection_rule":"validation_split_only","selected_config":asdict(b["cfg"]),"selected_threshold":{"tau_emit":b["th"]["tau_emit"],"tau_suppress":b["th"]["tau_suppress"]},"selected_metrics":r,"selected_test_metrics_by_suite":b["suite"],"violations":vio,"boundaries":{"device":"cpu","simulator_runtime":"NOT_PERFORMED","detector_training":"CPU_ONLY_CAUSAL_TCN_ON_C2E1_DATASET"}}
    write_json(out/"c2e2h_causal_tcn_report.json",report)
    names=["c2e2h_causal_tcn_report.json","c2e2h_all_config_metrics.csv","c2e2h_selected_test_metrics_by_suite.csv","c2e2h_selected_test_predictions.csv","c2e2h_violations.csv","c2e2h_selected_model.pt"]
    checks=[{"path":n,"sha256":sha256_file(out/n),"bytes":(out/n).stat().st_size} for n in names if (out/n).exists()]
    write_json(out/"checksum_report.json",{"files":checks})
    with (out/"SHA256SUMS").open("w",encoding="utf-8") as f:
        for c in checks: f.write(f"{c['sha256']}  {c['path']}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(out/'SHA256SUMS')}  SHA256SUMS\n",encoding="utf-8")
    print(json.dumps({"status":status,"output_root":str(out),"selected_config":asdict(b["cfg"]),"test_recall":r["test_recall"],"test_fp_rate":r["test_fp_rate"],"suite_fp_max":r["suite_fp_max"],"violations":vio},indent=2,sort_keys=True))
    return 0 if not vio else 2

if __name__ == "__main__":
    raise SystemExit(main())
