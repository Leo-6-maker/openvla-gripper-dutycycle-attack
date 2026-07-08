#!/usr/bin/env python3
"""C2e2F: task-context failure audit for GRU detector.

Joins C2e2D GRU predictions with C2e1 dataset context to analyze:
  - L10 FP/FN concentration by task template
  - Per-task error rates
  - Score distribution by task within L10
  - Whether task heterogeneity explains L10 weakness

CPU-only. No training. No OpenVLA/LIBERO/env.step.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

# ============ model (same as C2e2D) ============
class GRUSmall(nn.Module):
    def __init__(self, n_features=25, n_context=0, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers, batch_first=True,
                         dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Linear(hidden+n_context, 2)
    def forward(self, xt, xc):
        _, h = self.gru(xt); last = h[-1]
        if xc.shape[1]>0: last = torch.cat([last,xc],dim=1)
        return self.head(last)

def sigmoid_np(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))
def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=list(fields),extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def load_stats(path):
    obj = json.loads(open(path).read())
    return {"temporal_mean":np.asarray(obj.get("temporal_feature_mean",[]),dtype=np.float32),
            "temporal_std":np.asarray(obj.get("temporal_feature_std",[]),dtype=np.float32),
            "context_mean":np.asarray(obj.get("context_feature_mean",[]),dtype=np.float32),
            "context_std":np.asarray(obj.get("context_feature_std",[]),dtype=np.float32)}

def normalize_data(xt, xc, stats):
    tm,ts=stats["temporal_mean"].reshape(1,1,-1),stats["temporal_std"].reshape(1,1,-1)
    xt=(xt.astype(np.float32)-tm)/np.maximum(ts,1e-8)
    if xc.shape[1]>0:
        cm,cs=stats["context_mean"].reshape(1,-1),stats["context_std"].reshape(1,-1)
        xc=(xc.astype(np.float32)-cm)/np.maximum(cs,1e-8)
    return xt,xc

def load_dataset(c2e1_root, window):
    npz = np.load(Path(c2e1_root)/f"c2e1_w{window:02d}_temporal_dataset.npz", allow_pickle=True)
    stats = load_stats(Path(c2e1_root)/f"c2e1_w{window:02d}_normalization_stats_train_only.json")
    xt,xc = normalize_data(np.asarray(npz["X_temporal"],dtype=np.float32),
                          np.asarray(npz["X_context"],dtype=np.float32),stats)
    return {"xt":xt,"xc":xc,"y":npz["y"].astype(np.int64),
            "split":np.asarray(npz["split"]).astype(str),
            "suite":np.asarray(npz["suite"]).astype(str),
            "row_index":npz["row_index"].astype(np.int64)}

def predict(model, xt, xc, bs=512):
    model.eval(); outs=[]
    with torch.no_grad():
        for s in range(0,len(xt),bs):
            outs.append(model(torch.from_numpy(xt[s:s+bs]).float(),
                            torch.from_numpy(xc[s:s+bs]).float()).cpu().numpy())
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,2),dtype=np.float32)

def metrics(y,pred):
    pos,neg=y==1,y==0
    tp,fn=int((pred&pos).sum()),int(((~pred)&pos).sum())
    fp,tn=int((pred&neg).sum()),int(((~pred)&neg).sum())
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    f1=2*prec*rec/max(1e-12,prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root",required=True); ap.add_argument("--c2e2d-root",required=True)
    ap.add_argument("--context-dataset",required=True)
    ap.add_argument("--output-root",required=True); ap.add_argument("--git-commit",required=True)
    args = ap.parse_args()
    t0=time.time(); out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    c2e1_root=Path(args.c2e1_root); c2e2d_root=Path(args.c2e2d_root)

    # Load GRU model
    cfg=json.loads((c2e2d_root/"c2e2d_selected_model_config.json").read_text())["selected_config"]
    ckpt=torch.load(c2e2d_root/"c2e2d_selected_model.pt",map_location="cpu")
    thresh=ckpt["threshold"]; window=int(cfg["window"])
    print(f"C2e2F: {cfg['model_type']} W={window} te={thresh['tau_emit']} ts={thresh['tau_suppress']}")

    # Load data
    ds=load_dataset(c2e1_root,window)
    xt,xc,y,split,suite,ri=ds["xt"],ds["xc"],ds["y"],ds["split"],ds["suite"],ds["row_index"]

    # Build model
    model=GRUSmall(xt.shape[2],xc.shape[1],int(cfg["channels"]),dropout=float(cfg["dropout"]))
    model.load_state_dict(ckpt["model_state_dict"]); model.cpu().eval()

    # Predict
    logits=predict(model,xt,xc); ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    te,ts=float(thresh["tau_emit"]),float(thresh["tau_suppress"])
    pred=(ep>=te)&(sp<=ts)

    # Load context dataset for task metadata
    ctx_rows=list(csv.DictReader(open(args.context_dataset,encoding="utf-8-sig")))
    # Map row_index → task metadata
    task_map={}
    for r in ctx_rows:
        suite_val=r.get("suite",""); gk=r.get("group_key","")
        if gk not in task_map:
            task_map[gk]={"task_name":"","task_index":-1}
        # Extract task name from group_key
        parts=gk.split("/")
        if len(parts)>=2:
            tn="_".join(parts[1:-2]) if len(parts)>3 else parts[1]
            task_map[gk]["task_name"]=tn
        # Task index from context CSV
        ti=r.get("task_index","")
        if ti:
            try: task_map[gk]["task_index"]=int(ti)
            except ValueError: pass

    # Build task lookup by row_index using the dataset manifest
    # The C2e1 row_manifest has group_key mapping
    manifest_path=c2e1_root/f"c2e1_w{window:02d}_row_manifest.csv"
    if manifest_path.exists():
        manifest_rows=list(csv.DictReader(open(manifest_path)))
        row_to_gk={int(r["row_index"]):r.get("group_key","") for r in manifest_rows}
    else:
        # Fallback: use context dataset alignment (same order)
        row_to_gk={i:ctx_rows[i].get("group_key","") for i in range(min(len(ctx_rows),len(y)))}

    # L10 error analysis by task
    test_mask=split=="test"
    l10_mask=(suite=="libero_10")&test_mask
    l10_err=defaultdict(lambda: {"fp":0,"fn":0,"total":0,"primary":0,"no_primary":0,
                                  "fp_ep_median":[],"fn_ep_median":[]})
    for i in np.where(l10_mask)[0]:
        gk=row_to_gk.get(int(ri[i]),"unknown")
        tn=task_map.get(gk,{}).get("task_name","unknown")
        ti_val=task_map.get(gk,{}).get("task_index",-1)
        key=f"{tn} (ti={ti_val})"
        l10_err[key]["total"]+=1
        if y[i]==1: l10_err[key]["primary"]+=1
        else: l10_err[key]["no_primary"]+=1
        if pred[i] and not y[i]:
            l10_err[key]["fp"]+=1
            l10_err[key]["fp_ep_median"].append(float(ep[i]))
        if not pred[i] and y[i]:
            l10_err[key]["fn"]+=1
            l10_err[key]["fn_ep_median"].append(float(ep[i]))

    # Per-task summary
    task_rows=[]
    for task_name, stats in sorted(l10_err.items()):
        n=stats["total"]; fp=stats["fp"]; fn=stats["fn"]
        fp_rate=fp/max(1,stats["no_primary"]); fn_rate=fn/max(1,stats["primary"])
        task_rows.append({"task":task_name,"n":n,"primary":stats["primary"],
            "no_primary":stats["no_primary"],"fp":fp,"fn":fn,
            "fp_rate":fp_rate,"fn_rate":fn_rate,
            "fp_ep_median":np.median(stats["fp_ep_median"]) if stats["fp_ep_median"] else "",
            "fn_ep_median":np.median(stats["fn_ep_median"]) if stats["fn_ep_median"] else ""})
    task_rows.sort(key=lambda r:(-r["fp"],-r["fn"]))
    write_csv(out/"c2e2f_l10_errors_by_task.csv",task_rows,
              ["task","n","primary","no_primary","fp","fn","fp_rate","fn_rate","fp_ep_median","fn_ep_median"])

    # Concentration analysis
    total_fp=sum(r["fp"] for r in task_rows)
    total_fn=sum(r["fn"] for r in task_rows)
    top3_fp=sum(r["fp"] for r in task_rows[:3])
    top3_fn=sum(r["fn"] for r in task_rows[:3])
    print(f"  L10: {len(task_rows)} tasks, {total_fp} FP, {total_fn} FN")
    print(f"  Top-3 tasks: {top3_fp}/{total_fp} FP ({top3_fp/max(1,total_fp)*100:.0f}%), {top3_fn}/{total_fn} FN")
    for r in task_rows[:5]:
        print(f"    {r['task']}: fp={r['fp']} fn={r['fn']} (n={r['n']})")

    # L10 score distribution by task (top tasks)
    l10_score_rows=[]
    for i in np.where(l10_mask)[0]:
        gk=row_to_gk.get(int(ri[i]),"unknown")
        tn=task_map.get(gk,{}).get("task_name","unknown")
        ti_val=task_map.get(gk,{}).get("task_index",-1)
        l10_score_rows.append({"task":f"{tn} (ti={ti_val})","y":int(y[i]),"pred":int(pred[i]),
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),
            "error":"FP" if pred[i] and not y[i] else ("FN" if not pred[i] and y[i] else "")})
    write_csv(out/"c2e2f_l10_score_distribution_by_task.csv",l10_score_rows,
              ["task","y","pred","emit_p","suppress_p","error"])

    # Overall task-level metrics for all suites
    all_task_rows=[]
    for s in sorted(set(suite)):
        s_mask=(suite==s)&test_mask
        task_stats=defaultdict(lambda: {"y":[],"pred":[]})
        for i in np.where(s_mask)[0]:
            gk=row_to_gk.get(int(ri[i]),"unknown")
            tn=task_map.get(gk,{}).get("task_name","unknown")
            task_stats[tn]["y"].append(int(y[i]))
            task_stats[tn]["pred"].append(int(pred[i]))
        for tn,st in task_stats.items():
            ya=np.array(st["y"]); pa=np.array(st["pred"],dtype=bool)
            m=metrics(ya,pa)
            all_task_rows.append({"suite":s,"task":tn,"n":len(ya),
                "primary":int((ya==1).sum()),"no_primary":int((ya==0).sum()),
                **{f"test_{k}":v for k,v in m.items()}})
    all_task_rows.sort(key=lambda r:(-r["test_fp"],-r["test_fn"]))
    write_csv(out/"c2e2f_per_task_metrics_all_suites.csv",all_task_rows,
              ["suite","task","n","primary","no_primary","test_tp","test_fn","test_fp","test_tn",
               "test_recall","test_fp_rate","test_precision","test_f1"])

    # Recommendations
    l10_task_count=len(task_rows)
    concentrated=top3_fp>total_fp*0.5 if total_fp>0 else False
    heterogeneous=l10_task_count>5

    report={
        "gate":"C2E2F_TASK_CONTEXT_FAILURE_AUDIT",
        "status":"PASS_C2E2F_TASK_FAILURE_ANALYZED",
        "created_at_unix":time.time(),"runtime_seconds":time.time()-t0,
        "git_commit":args.git_commit,
        "l10_summary":{"tasks":l10_task_count,"total_fp":total_fp,"total_fn":total_fn,
            "top3_fp_pct":top3_fp/max(1,total_fp),"top3_fn_pct":top3_fn/max(1,total_fn),
            "fp_concentrated":concentrated,"task_heterogeneous":heterogeneous},
        "top_fp_tasks":[{k:r[k] for k in ["task","fp","fn","n","fp_rate","fn_rate"]} for r in task_rows[:5]],
        "recommendation":(
            "task_aware_conditioning_or_calibration" if concentrated else
            "task_heterogeneous_consider_balanced_loss" if heterogeneous else
            "proprio_action_insufficient_for_l10"
        ),
        "boundaries":{"CUDA_required":"NOT_REQUIRED","device":"cpu",
            "OpenVLA_model":"NOT_LOADED","detector_training":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2f_task_context_audit_report.json",report)

    csums={}
    for fn in sorted(os.listdir(str(out))):
        fp=out/fn
        if fp.is_file() and fn!="checksum_report.json": csums[fn]=sha256_file(str(fp))
    write_json(out/"checksum_report.json",csums)
    (out/"SHA256SUMS").write_text("\n".join(f"{sha}  {fn}" for fn,sha in sorted(csums.items()))+"\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":report["status"],"l10_tasks":l10_task_count,
        "fp_concentrated":concentrated,"recommendation":report["recommendation"]},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
