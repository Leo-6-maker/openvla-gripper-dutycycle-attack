#!/usr/bin/env python3
"""C2e2H parallel causal TCN training on C2e1 temporal tensors.

Offline CPU-only detector experiment with ProcessPoolExecutor across configs.
Val-only selection, test final report only.
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
import torch.nn.functional as F

PASS = "PASS_C2E2H_CAUSAL_TCN_TRAINED"
HOLD = "HOLD_C2E2H_CAUSAL_TCN_TRAINING"

@dataclass
class Config:
    window: int; channels: int; dropout: float; lr: float; seed: int; kernel_size: int = 3

# ============ Model ============
class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.left_pad = (kernel-1)*dilation; self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel, dilation=dilation)
    def forward(self, x): return self.conv(F.pad(x, (self.left_pad, 0)))

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(CausalConv1d(c_in,c_out,kernel,dilation), nn.ReLU(), nn.Dropout(dropout),
                                 CausalConv1d(c_out,c_out,kernel,dilation), nn.ReLU(), nn.Dropout(dropout))
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x): return self.net(x) + self.proj(x)

class CausalTCN(nn.Module):
    def __init__(self, n_temporal, n_context, channels, dropout, kernel):
        super().__init__()
        self.tcn = nn.Sequential(TCNBlock(n_temporal,channels,kernel,1,dropout),
                                 TCNBlock(channels,channels,kernel,2,dropout),
                                 TCNBlock(channels,channels,kernel,4,dropout))
        self.head = nn.Sequential(nn.Linear(channels+n_context,channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(channels,2))
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
def metrics_from_pred(y,pred):
    pos,neg=y==1,y==0; tp,fn=int((pred&pos).sum()),int(((~pred)&pos).sum())
    fp,tn=int((pred&neg).sum()),int(((~pred)&neg).sum())
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    f1=2*prec*rec/max(1e-12,prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}
def suite_rows(y,pred,suite):
    return [{"suite":s,"n":int((suite==s).sum()),**metrics_from_pred(y[suite==s],pred[suite==s])} for s in sorted(set(suite))]
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
            pred=(ep>=te)&(sp<=ts); m=metrics_from_pred(y,pred)
            f=m["recall"]>=min_r and m["fp_rate"]<=max_fp
            s=m["f1"]+0.5*m["recall"]-3*max(0,min_r-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            rows.append({"tau_emit":te,"tau_suppress":ts,"feasible":f,"score":s,**m})
    rows.sort(key=lambda r:(bool(r["feasible"]),r["score"],r["f1"],r["recall"],-r["fp_rate"]),reverse=True)
    return rows

def train_model(d,cfg,epochs,batch,patience,torch_threads):
    set_seed(cfg.seed); torch.set_num_threads(max(1,torch_threads))
    tr,va=d["split"]=="train",d["split"]=="val"
    model=CausalTCN(d["xt"].shape[2],d["xc"].shape[1],cfg.channels,cfg.dropout,cfg.kernel_size).cpu()
    np_pos=max(1,int((d["y"][tr]==1).sum())); nneg=max(1,int((d["y"][tr]==0).sum()))
    pe,ps=nneg/np_pos,np_pos/nneg
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(d["xt"][tr]),torch.from_numpy(d["xc"][tr]),torch.from_numpy(d["y"][tr]).float())
    loader=DataLoader(ds,batch_size=batch,shuffle=True)
    xv,cv=torch.from_numpy(d["xt"][va]),torch.from_numpy(d["xc"][va])
    yv=torch.from_numpy(d["y"][va]).float()
    best_state,best_loss,best_epoch,stale=None,1e9,-1,0
    for ep in range(1,epochs+1):
        model.train()
        for xs2,cs2,ys2 in loader:
            opt.zero_grad(set_to_none=True)
            yf=ys2; l1=F.binary_cross_entropy_with_logits(model(xs2,cs2)[:,0],yf,pos_weight=torch.tensor(pe,device=model.head[0].weight.device))
            l2=F.binary_cross_entropy_with_logits(model(xs2,cs2)[:,1],1-yf,pos_weight=torch.tensor(ps,device=model.head[0].weight.device))
            (l1+l2).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lo=model(xv,cv)
            vl=float((F.binary_cross_entropy_with_logits(lo[:,0],yv,pos_weight=torch.tensor(pe))+
                      F.binary_cross_entropy_with_logits(lo[:,1],1-yv,pos_weight=torch.tensor(ps))).cpu())
        if vl<best_loss-1e-5: best_loss,best_epoch,stale=vl,ep,0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: break
    if best_state: model.load_state_dict(best_state)
    return model,{"best_val_loss":best_loss,"best_epoch":best_epoch,"epochs_run":ep}

def _worker(args_dict):
    cfg=Config(**args_dict["config"]); d=load_data(args_dict["c2e1_root"],cfg.window)
    model,ti=train_model(d,cfg,args_dict["epochs"],args_dict["batch_size"],args_dict["patience"],args_dict.get("torch_threads",4))
    logits=predict(model,d["xt"],d["xc"])
    val,test=d["split"]=="val",d["split"]=="test"
    sw=sweep(d["y"][val],logits[val],args_dict["min_val_recall"],args_dict["max_val_fp"]); th=sw[0]
    ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    pred=(ep>=th["tau_emit"])&(sp<=th["tau_suppress"])
    vm=metrics_from_pred(d["y"][val],pred[val]); tm=metrics_from_pred(d["y"][test],pred[test])
    sr=suite_rows(d["y"][test],pred[test],d["suite"][test]); sfp=max([x["fp_rate"] for x in sr],default=0)
    row=asdict(cfg); row.update(ti); row.update({"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"],"suite_fp_max":sfp})
    for k,v in vm.items(): row["val_"+k]=v
    for k,v in tm.items(): row["test_"+k]=v
    row["selection_score"]=vm["f1"]+0.5*vm["recall"]-3*max(0,args_dict["min_val_recall"]-vm["recall"])-3*max(0,vm["fp_rate"]-args_dict["max_val_fp"])
    row["feasible"]=bool(vm["recall"]>=args_dict["min_val_recall"] and vm["fp_rate"]<=args_dict["max_val_fp"])
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--c2e1-root",required=True); ap.add_argument("--output-root",required=True)
    ap.add_argument("--git-commit",required=True)
    ap.add_argument("--windows",default="8,16,32"); ap.add_argument("--channels-grid",default="64,128")
    ap.add_argument("--dropout-grid",default="0.0,0.1"); ap.add_argument("--lr-grid",default="0.001,0.0003")
    ap.add_argument("--seeds",default="0,1,2"); ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--patience",type=int,default=25)
    ap.add_argument("--torch-threads",type=int,default=2); ap.add_argument("--workers",type=int,default=32)
    ap.add_argument("--min-val-recall",type=float,default=0.70); ap.add_argument("--max-val-fp",type=float,default=0.30)
    ap.add_argument("--min-test-recall",type=float,default=0.70); ap.add_argument("--max-test-fp",type=float,default=0.30)
    ap.add_argument("--max-suite-test-fp",type=float,default=0.50)
    args=ap.parse_args()
    t0=time.time(); out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    c2e1_root=str(Path(args.c2e1_root).resolve())

    windows=[int(x) for x in args.windows.split(",")]
    channels=[int(x) for x in args.channels_grid.split(",")]
    drops=[float(x) for x in args.dropout_grid.split(",")]
    lrs=[float(x) for x in args.lr_grid.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]

    all_configs=[]
    for w in windows:
        for ch in channels:
            for dr in drops:
                for lr in lrs:
                    for sd in seeds:
                        all_configs.append({"c2e1_root":c2e1_root,
                            "config":{"window":w,"channels":ch,"dropout":dr,"lr":lr,"seed":sd},
                            "epochs":args.epochs,"batch_size":args.batch_size,"patience":args.patience,
                            "min_val_recall":args.min_val_recall,"max_val_fp":args.max_val_fp,
                            "torch_threads":args.torch_threads})

    nw=min(args.workers,len(all_configs))
    print(f"C2e2H TCN: {len(all_configs)} configs x {nw} workers")
    rows=[]; best_row=None; done=0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures={pool.submit(_worker,c):c for c in all_configs}
        for fut in as_completed(futures):
            row=fut.result(); rows.append(row); done+=1
            sel=float(row["selection_score"]); feasible=bool(row["feasible"])
            key=(feasible,sel,float(row["val_f1"]),float(row["val_recall"]),-float(row["val_fp_rate"]))
            if best_row is None or key>best_row["_key"]: row["_key"]=key; best_row=row
            if done%24==0 or done==len(all_configs): print(f"  {done}/{len(all_configs)} configs...")

    if best_row is None: raise RuntimeError("no config completed")
    best_cfg=Config(window=int(best_row["window"]),channels=int(best_row["channels"]),
        dropout=float(best_row["dropout"]),lr=float(best_row["lr"]),seed=int(best_row["seed"]))
    # Retrain best in main process
    torch.set_num_threads(max(1,args.torch_threads*4))
    d=load_data(c2e1_root,best_cfg.window)
    model,ti=train_model(d,best_cfg,args.epochs,args.batch_size,args.patience,args.torch_threads*4)
    logits=predict(model,d["xt"],d["xc"])
    val=d["split"]=="val"; test=d["split"]=="test"
    sw=sweep(d["y"][val],logits[val],args.min_val_recall,args.max_val_fp); th=sw[0]
    ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    pred=(ep>=th["tau_emit"])&(sp<=th["tau_suppress"])
    tm=metrics_from_pred(d["y"][test],pred[test]); sr=suite_rows(d["y"][test],pred[test],d["suite"][test])
    sfp=max([x["fp_rate"] for x in sr],default=0)

    pred_rows=[]
    for i in np.where(test)[0]:
        pred_rows.append({"row_index":int(d["row_index"][i]),"suite":str(d["suite"][i]),"y":int(d["y"][i]),
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),"pred":int(bool(pred[i]))})

    vio=[]
    if tm["recall"]<args.min_test_recall: vio.append(f"LOW_TEST_RECALL:{tm['recall']:.4f}")
    if tm["fp_rate"]>args.max_test_fp: vio.append(f"HIGH_TEST_FP:{tm['fp_rate']:.4f}")
    if sfp>args.max_suite_test_fp: vio.append(f"HIGH_SUITE_FP_MAX:{sfp:.4f}")
    status=PASS if not vio else HOLD

    model_path=out/"c2e2h_selected_model.pt"
    torch.save({"state_dict":model.state_dict(),"config":asdict(best_cfg),
        "threshold":{"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"]}},model_path)

    write_csv(out/"c2e2h_all_config_metrics.csv",rows,
              ["window","channels","dropout","lr","seed","kernel_size","best_val_loss","best_epoch","epochs_run",
               "tau_emit","tau_suppress","val_recall","val_fp_rate","val_f1",
               "test_recall","test_fp_rate","test_f1","suite_fp_max","selection_score","feasible"])
    write_csv(out/"c2e2h_selected_test_metrics_by_suite.csv",sr,
              ["suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1"])
    write_csv(out/"c2e2h_selected_test_predictions.csv",pred_rows,
              ["row_index","suite","y","emit_p","suppress_p","pred"])
    write_csv(out/"c2e2h_violations.csv",[{"violation":v} for v in vio],["violation"])
    write_json(out/"c2e2h_selected_model_config.json",
               {"selected_config":asdict(best_cfg),"selected_threshold":{"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"]}})

    report={"gate":"C2E2H_CAUSAL_TCN_TRAINING","status":status,
        "reason":"violations=0" if not vio else f"violations={len(vio)}",
        "created_at_unix":time.time(),"runtime_seconds":time.time()-t0,"git_commit":args.git_commit,
        "selection_rule":"model/config/threshold selected on validation split only; test split used only for final report",
        "selected_config":asdict(best_cfg),"selected_threshold":{"tau_emit":th["tau_emit"],"tau_suppress":th["tau_suppress"]},
        "selected_metrics":tm,"selected_test_metrics_by_suite":sr,"violations":vio,
        "recommendation":"proceed_to_C2E2I_comparison" if not vio else "hold_debug_tcn",
        "boundaries":{"CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"CPU_ONLY_CAUSAL_TCN_ON_C2E1_DATASET",
            "env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2h_causal_tcn_report.json",report)

    csums={}
    for fn in sorted(os.listdir(str(out))):
        fp=out/fn
        if fp.is_file() and fn!="checksum_report.json": csums[fn]=sha256_file(str(fp))
    write_json(out/"checksum_report.json",csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn,sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":status,"selected":asdict(best_cfg),"test_recall":tm["recall"],
        "test_fp":tm["fp_rate"],"test_f1":tm["f1"],"violations":vio},indent=2,sort_keys=True))
    return 0 if not vio else 1

if __name__=="__main__":
    raise SystemExit(main())
