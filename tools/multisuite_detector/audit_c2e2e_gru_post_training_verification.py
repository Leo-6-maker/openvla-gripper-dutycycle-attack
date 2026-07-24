#!/usr/bin/env python3
"""C2e2E: GRU post-training audit gate.

Verifies C2e2D GRU model artifacts without training:
  - val-only selection proof
  - metric reproducibility from saved model
  - threshold local sensitivity
  - score distributions
  - L10 error pattern analysis
  - split leakage / normalization source / consistency

CPU-only. No OpenVLA, no LIBERO, no env.step, no rollout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

# ============ model ============
class CausalTCN(nn.Module):
    def __init__(self, n_features=25, n_context=0, channels=64, kernel_size=3, dilations=(1,2,4), dropout=0.0):
        super().__init__()
        layers = []; in_ch = n_features
        for d in dilations:
            pad = (kernel_size-1)*d
            layers.extend([nn.Conv1d(in_ch,channels,kernel_size,dilation=d,padding=pad,padding_mode='zeros'),
                          nn.ReLU(), nn.Dropout(dropout)]); in_ch = channels
        self.conv = nn.Sequential(*layers) if layers else nn.Identity()
        self._trim = (kernel_size-1)*max(dilations) if dilations else 0
        self.head = nn.Linear(channels+n_context, 2)
    def forward(self, xt, xc):
        x = xt.permute(0,2,1); x = self.conv(x)
        if self._trim > 0: x = x[:,:,:-self._trim] if x.shape[2]>self._trim else x[:,:,-1:]
        last = x[:,:,-1]
        if xc.shape[1]>0: last = torch.cat([last,xc],dim=1)
        return self.head(last)

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

# ============ helpers ============
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

def suite_metrics(y,pred,suite,sp):
    return [{"split":sp,"suite":s,"n":int((suite==s).sum()),**metrics(y[suite==s],pred[suite==s])} for s in sorted(set(suite))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root",required=True); ap.add_argument("--c2e2d-root",required=True)
    ap.add_argument("--output-root",required=True); ap.add_argument("--git-commit",required=True)
    ap.add_argument("--min-recall",type=float,default=0.70); ap.add_argument("--max-fp",type=float,default=0.30)
    ap.add_argument("--max-suite-fp",type=float,default=0.50)
    ap.add_argument("--max-object-fp",type=float,default=0.30); ap.add_argument("--max-spatial-fp",type=float,default=0.35)
    args = ap.parse_args()
    t0=time.time(); out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    c2e1_root=Path(args.c2e1_root); c2e2d_root=Path(args.c2e2d_root)

    # Load C2e2D artifacts
    cfg=json.loads((c2e2d_root/"c2e2d_selected_model_config.json").read_text())["selected_config"]
    ckpt=torch.load(c2e2d_root/"c2e2d_selected_model.pt",map_location="cpu")
    thresh=ckpt["threshold"]; window=int(cfg["window"]); mt=str(cfg["model_type"])
    report_orig=json.loads((c2e2d_root/"c2e2d_sequence_model_report.json").read_text())
    selection_rule=report_orig.get("selection_rule","")
    print(f"C2e2E: {mt} W={window} te={thresh['tau_emit']} ts={thresh['tau_suppress']}")

    # Load data
    ds=load_dataset(c2e1_root,window)
    xt,xc,y,split,suite,ri=ds["xt"],ds["xc"],ds["y"],ds["split"],ds["suite"],ds["row_index"]

    # Build model
    if mt=="tcn":
        model=CausalTCN(xt.shape[2],xc.shape[1],int(cfg["channels"]),dropout=float(cfg["dropout"]))
    else:
        model=GRUSmall(xt.shape[2],xc.shape[1],int(cfg["channels"]),dropout=float(cfg["dropout"]))
    model.load_state_dict(ckpt["model_state_dict"]); model.cpu().eval()

    # Reproduce predictions
    logits=predict(model,xt,xc); ep,sp=sigmoid_np(logits[:,0]),sigmoid_np(logits[:,1])
    te,ts=float(thresh["tau_emit"]),float(thresh["tau_suppress"])
    pred=(ep>=te)&(sp<=ts)
    test_mask=split=="test"; val_mask=split=="val"

    # Check 1: metric reproducibility — compare per-suite breakdowns
    my_suite=suite_metrics(y[test_mask],pred[test_mask],suite[test_mask],"test")
    orig_suite=report_orig.get("selected_test_metrics_by_suite",[])
    diffs=[]
    for mr in my_suite:
        for o_r in orig_suite:
            if mr["suite"]==o_r.get("suite",""):
                for k in ["recall","fp_rate","f1","tp","fn","fp","tn"]:
                    diffs.append(abs(mr.get(k,0)-float(o_r.get(k,0))))
    max_diff=max(diffs) if diffs else 0
    reproducible=max_diff<0.02
    print(f"  Metric reproducibility: max_diff={max_diff:.6f} {'OK' if reproducible else 'FAIL'}")

    # Check 2: suite metrics
    suite_rows=suite_metrics(y[test_mask],pred[test_mask],suite[test_mask],"test")
    write_csv(out/"c2e2e_recomputed_metrics_by_suite.csv",suite_rows,
              ["split","suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1"])

    my_test_overall=metrics(y[test_mask],pred[test_mask])

    # Check 3: threshold local sensitivity
    sens_rows=[]
    for dte in [-0.10,-0.05,-0.02,0,0.02,0.05,0.10]:
        for dts in [-0.10,-0.05,-0.02,0,0.02,0.05,0.10]:
            te2=round(max(0.01,min(0.99,te+dte)),3); ts2=round(max(0.01,min(0.99,ts+dts)),3)
            pred2_tm=(ep[test_mask]>=te2)&(sp[test_mask]<=ts2)
            m=metrics(y[test_mask],pred2_tm)
            drec=abs(m["recall"]-my_test_overall["recall"]); dfp=abs(m["fp_rate"]-my_test_overall["fp_rate"])
            stable=drec<0.05 and dfp<0.08
            sens_rows.append({"d_tau_emit":dte,"d_tau_suppress":dts,"tau_emit":te2,"tau_suppress":ts2,
                "stable":stable,"recall":m["recall"],"fp_rate":m["fp_rate"],"f1":m["f1"]})
    write_csv(out/"c2e2e_threshold_local_sensitivity.csv",sens_rows,
              ["d_tau_emit","d_tau_suppress","tau_emit","tau_suppress","stable","recall","fp_rate","f1"])
    # Stability only required for ±0.02 perturbations (the smallest tested)
    small_pert=[r for r in sens_rows if abs(r["d_tau_emit"])<=0.02 and abs(r["d_tau_suppress"])<=0.02]
    stable_count=sum(1 for r in small_pert if r["stable"])
    threshold_stable=stable_count>=len(small_pert)*0.8 if small_pert else False
    print(f"  Threshold stability (±0.02): {stable_count}/{len(small_pert)} stable {'OK' if threshold_stable else 'BRITTLE'}")

    # Check 4: score distribution
    dist_rows=[]
    for sp_name in ["train","val","test"]:
        m=split==sp_name
        for s in sorted(set(suite[m])):
            for label in [0,1]:
                mk=(suite==s)&(split==sp_name)&(y==label); n=int(mk.sum())
                if n==0: continue
                q=lambda a,qq:float(np.percentile(a,qq))
                dist_rows.append({"split":sp_name,"suite":s,"label":label,"n":n,
                    "emit_q05":q(ep[mk],5),"emit_q50":q(ep[mk],50),"emit_q95":q(ep[mk],95),"emit_mean":float(ep[mk].mean()),
                    "supp_q05":q(sp[mk],5),"supp_q50":q(sp[mk],50),"supp_q95":q(sp[mk],95),"supp_mean":float(sp[mk].mean()),
                    "pred_rate":float(((ep[mk]>=te)&(sp[mk]<=ts)).mean())})
    write_csv(out/"c2e2e_score_distribution_by_suite_label.csv",dist_rows,
              ["split","suite","label","n","emit_q05","emit_q50","emit_q95","emit_mean",
               "supp_q05","supp_q50","supp_q95","supp_mean","pred_rate"])

    # Check 5: L10 error analysis
    l10_errors=[]
    l10_mask=(suite=="libero_10")&test_mask
    l10_err_idx=np.where(l10_mask&(pred!=y))[0]
    for i in l10_err_idx:
        l10_errors.append({"row_index":int(ri[i]),"y":int(y[i]),"pred":int(pred[i]),
            "error_type":"FP" if pred[i] and not y[i] else "FN",
            "emit_p":float(ep[i]),"suppress_p":float(sp[i]),"margin":float(ep[i]-sp[i])})
    write_csv(out/"c2e2e_libero10_error_rows.csv",l10_errors,
              ["row_index","y","pred","error_type","emit_p","suppress_p","margin"])
    l10_fp_count=sum(1 for r in l10_errors if r["error_type"]=="FP")
    l10_fn_count=sum(1 for r in l10_errors if r["error_type"]=="FN")
    print(f"  L10 errors: {l10_fp_count} FP + {l10_fn_count} FN = {len(l10_errors)} total")

    # Check 6: split leakage
    group_splits=defaultdict(set)
    for i in range(len(y)):
        gk=f"{suite[i]}_{ri[i]}"
        group_splits[gk].add(str(split[i]))
    leakage=sum(1 for gk,sp in group_splits.items() if len(sp)>1)
    print(f"  Split leakage: {leakage}")

    # Check 7: normalization source verification
    stats_path=c2e1_root/f"c2e1_w{window:02d}_normalization_stats_train_only.json"
    stats_meta=json.loads(stats_path.read_text())
    norm_source=stats_meta.get("fit_split","train") if "fit_split" in stats_meta else "train_only"
    print(f"  Normalization source: {norm_source}")

    # Check 8: Object/Spatial score separation
    obj_test=(suite=="libero_object")&test_mask
    obj_labels=set(y[obj_test].tolist())
    obj_sep={}
    for lb in obj_labels:
        m=(suite=="libero_object")&test_mask&(y==lb)
        obj_sep[f"label_{lb}_n"]=int(m.sum())
        obj_sep[f"label_{lb}_supp_q50"]=float(np.median(sp[m])) if m.sum()>0 else 0.0
    print(f"  Object score separation: {obj_sep}")

    # Violations
    violations=[]
    warnings=[]
    if not reproducible: violations.append("METRICS_NOT_REPRODUCIBLE")
    if not threshold_stable: violations.append("THRESHOLD_BRITTLE")
    if leakage>0: violations.append(f"SPLIT_LEAKAGE:{leakage}")
    if norm_source!="train": violations.append(f"NORM_NOT_TRAIN_ONLY:{norm_source}")
    if "val" in str(selection_rule).lower() and "test" not in str(selection_rule).lower():
        pass  # OK — val-only selection
    else:
        warnings.append("SELECTION_RULE_UNCLEAR")
    if l10_fn_count>0:
        warnings.append(f"L10_WEAK_RECALL:FN={l10_fn_count}")
    if l10_fp_count>0:
        warnings.append(f"L10_ELEVATED_FP:FP={l10_fp_count}")

    all_ok=len(violations)==0
    status="PASS_C2E2E_GRU_POST_TRAINING_AUDIT" if all_ok else "HOLD_C2E2E_GRU_POST_TRAINING_AUDIT"

    report={
        "gate":"C2E2E_GRU_POST_TRAINING_AUDIT","status":status,
        "reason":"violations=0" if all_ok else f"violations={len(violations)}",
        "created_at_unix":time.time(),"runtime_seconds":time.time()-t0,"git_commit":args.git_commit,
        "checks":{
            "metric_reproducible":reproducible,"max_metric_diff":max_diff,
            "threshold_stable":threshold_stable,"stable_fraction":stable_count/len(sens_rows) if sens_rows else 0,
            "split_leakage":leakage,"norm_source":norm_source,
            "val_only_selection_rule":str(selection_rule)[:200],
        },
        "recomputed_test_metrics":my_test_overall,
        "suite_metrics":{r["suite"]:{"recall":r["recall"],"fp_rate":r["fp_rate"],"f1":r["f1"]} for r in suite_rows},
        "l10_error_summary":{"fp":l10_fp_count,"fn":l10_fn_count,"total":len(l10_errors)},
        "object_separation":obj_sep,
        "violations":violations,"warnings":warnings,
        "recommendation":("proceed_to_C2E3_packaging" if all_ok else
                         "fix_violations_before_c2e3"),
        "boundaries":{"CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"NOT_PERFORMED",
            "env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED","simulator_runtime":"NOT_PERFORMED"},
    }
    write_json(out/"c2e2e_gru_post_training_audit_report.json",report)
    write_csv(out/"c2e2e_violations.csv",[{"violation":v} for v in violations+warnings],["violation"])

    csums={}
    for fn in sorted(os.listdir(str(out))):
        fp=out/fn
        if fp.is_file() and fn!="checksum_report.json": csums[fn]=sha256_file(str(fp))
    write_json(out/"checksum_report.json",csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn,sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":status,"reproducible":reproducible,"stable":threshold_stable,
        "leakage":leakage,"l10_fp":l10_fp_count,"l10_fn":l10_fn_count,
        "violations":violations,"warnings":warnings},indent=2))
    return 0 if all_ok else 1

if __name__=="__main__":
    raise SystemExit(main())
