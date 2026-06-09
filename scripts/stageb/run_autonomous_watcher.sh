#!/bin/bash
# Autonomous watcher: monitors main extraction, auto-runs readout on completion.
# Server-side, no frontend polling needed.

MAIN_LOG=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/run.log
DUP_LOG=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_duplicate_gpu45/run.log
REPO=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python

echo "[$(date +%H:%M)] AUTONOMOUS WATCHER STARTED"
echo "  Main: $(grep -c 'pre=' $MAIN_LOG 2>/dev/null || echo 0)/38"
echo "  Dup:  $(grep -c 'pre=' $DUP_LOG 2>/dev/null || echo 0)/8"

# Wait for main extraction to complete
while true; do
  DONE=$(grep -c 'pre=' $MAIN_LOG 2>/dev/null || echo 0)
  ERR=$(grep -c 'Error\|Traceback' $MAIN_LOG 2>/dev/null || echo 0)
  if [ "$DONE" -ge 38 ]; then
    echo "[$(date +%H:%M)] MAIN EXTRACTION COMPLETE ($DONE done)"
    break
  fi
  if [ "$ERR" -gt 0 ]; then
    echo "[$(date +%H:%M)] ERRORS IN MAIN EXTRACTION"
    grep -E 'Error|Traceback' $MAIN_LOG | tail -5
    break
  fi
  sleep 300
done

# Run readout
echo "[$(date +%H:%M)] RUNNING ACTION-LOGIT READOUT..."
cd $REPO && PYTHONPATH=src $PY scripts/diagnostics/run_action_logit_readout.py > /data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/readout.log 2>&1
echo "[$(date +%H:%M)] READOUT COMPLETE"

# Run ablation
echo "[$(date +%H:%M)] RUNNING ABLATION..."
cd $REPO && PYTHONPATH=src $PY -c "
import csv, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import Counter

FEATURES = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/action_logit_full_features.csv'
STABLE = 'tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv'

import re
KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
def parse(pk):
    task=sid=ws=we=None
    for tk in KNOWN:
        if tk in pk: task=tk; break
    m_s=re.search(r'_s(\d+)',pk);
    if m_s: sid=m_s.group(1)
    m_w=re.search(r'_w(\d+)_(\d+)',pk)
    if m_w: ws=m_w.group(1); we=m_w.group(2)
    return task,sid,ws,we

stable={}
with open(STABLE) as f:
    for r in csv.DictReader(f): stable[r['parent']]=r

features={}
with open(FEATURES) as f:
    for r in csv.DictReader(f):
        key=(r['task'], int(r['state_id']), int(r['ws']), int(r['we']))
        features[key]=r

rows=[]
for pk,pr in stable.items():
    task,sid,ws,we=parse(pk)
    if not task: task=pr.get('task','?')
    if not ws:
        win=pr.get('window',''); parts=win.replace('_env0','').replace('_env1','').replace('_env2','').split('_')
        if len(parts)>=2: ws,we=int(parts[0]),int(parts[1])
    if not sid: sid=int(win.split('_env')[1] if '_env' in win else 0) if 'win' in dir() else 0
    else: sid=int(sid)
    if not all([task,isinstance(ws,int),isinstance(we,int)]): continue
    key=(task,sid,ws,we)
    if key not in features: continue
    r=features[key]
    rows.append({
        'parent':pk,'task':task,
        'is_rand':1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd':1 if 'cmd_specific' in pr['cmd_label'] else 0,
        'risk':float(pr.get('risk_rand',0)),'yield':float(pr.get('yield_cmd',0)),
        'on_last':float(r['open_norm_last']),'margin_last':float(r['logit_margin_last']),
        'ent_last':float(r['entropy_last']),'t2_last':float(r['top2_margin_last']),
        'on_mean':float(r['open_norm_mean']),'margin_mean':float(r['logit_margin_mean']),
        'ent_mean':float(r['entropy_mean']),'t2_mean':float(r['top2_margin_mean']),
        'rg_mean':float(r['rg_mean']),'rg_std':float(r['rg_std']),
        'ws':ws,'we':we,'n_pre':int(r['n_pre']),
    })

n=len(rows)
X_last=np.column_stack([[r['on_last'] for r in rows],[r['margin_last'] for r in rows],[r['ent_last'] for r in rows],[r['t2_last'] for r in rows]])
X_agg=np.column_stack([[r['on_mean'] for r in rows],[r['margin_mean'] for r in rows],[r['ent_mean'] for r in rows],[r['t2_mean'] for r in rows]])
X_proprio=np.column_stack([[r['rg_mean'] for r in rows],[r['rg_std'] for r in rows]])
ws_a=np.array([r['ws'] for r in rows]); we_a=np.array([r['we'] for r in rows])
wc_a=(ws_a+we_a)/2.0; X_timing=np.column_stack([wc_a,wc_a/300.0])
groups=np.array([r['task'] for r in rows]); y_rand=np.array([r['is_rand'] for r in rows])
nsp=min(3,len(set(groups))); gkf=GroupKFold(n_splits=nsp)

configs={
    'LastOnly': X_last,'AggOnly': X_agg,'Last+Agg': np.column_stack([X_last,X_agg]),
    'Proprio+Last': np.column_stack([X_proprio,X_last,X_timing]),
    'Proprio+Agg': np.column_stack([X_proprio,X_agg,X_timing]),
    'Proprio+Last+Agg': np.column_stack([X_proprio,X_last,X_agg,X_timing]),
}

print('Ablation: last-step vs aggregate')
for name,Xf in configs.items():
    oof=np.zeros(n)
    for ti,tei in gkf.split(Xf,y_rand,groups=groups):
        ss=StandardScaler(); Xt=ss.fit_transform(Xf[ti]); Xe=ss.transform(Xf[tei])
        m=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42,C=0.5)
        m.fit(Xt,y_rand[ti]); oof[tei]=m.predict_proba(Xe)[:,1]
    auc=roc_auc_score(y_rand,oof) if len(set(y_rand))>1 else 0
    fp=fn=0
    for i,r in enumerate(rows):
        if 'k5b_contrast_tomato_far' in r['parent']: fp=oof[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn=oof[i]
    print('  %-20s AUC=%.3f FP=%.4f FN=%.4f' % (name,auc,fp,fn))

with open('/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/ablation_results.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','AUROC','FP_score','FN_score'])
    for name,Xf in configs.items():
        oof=np.zeros(n)
        for ti,tei in gkf.split(Xf,y_rand,groups=groups):
            ss=StandardScaler(); Xt=ss.fit_transform(Xf[ti]); Xe=ss.transform(Xf[tei])
            m=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42,C=0.5)
            m.fit(Xt,y_rand[ti]); oof[tei]=m.predict_proba(Xe)[:,1]
        auc=roc_auc_score(y_rand,oof) if len(set(y_rand))>1 else 0
        fp=fn=0
        for i,r in enumerate(rows):
            if 'k5b_contrast_tomato_far' in r['parent']: fp=oof[i]
            if 'k5b_strict_phys_salad' in r['parent']: fn=oof[i]
        w.writerow([name,round(auc,4),round(fp,4),round(fn,4)])
print('Ablation saved.')
"

echo "[$(date +%H:%M)] AUTONOMOUS WATCHER DONE"
