#!/usr/bin/env python3
"""v0.3.2: Task-balanced randhead training on 250 RAND labels. Clean split, multi-model eval."""
import csv, json, glob, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

T='/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
C='/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/configs'
os.makedirs(T,exist_ok=True);os.makedirs(C,exist_ok=True)

# ── Load all RAND labels ──
all_rand = {}
dirs = [
    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
    '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
    '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
    '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
    '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
    '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613',
    '/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613',
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
]
for d in dirs:
    for f in glob.glob(d+'/summary_*random_linf*.json'):
        s=json.load(open(f))
        key=(s['task'],str(s['state_id']),s['window_start'],s['window_end'],str(s.get('attack_seed','0')))
        all_rand[key]=s

# ── Build labels ──
rows=[]
for key,s in all_rand.items():
    task,sid,ws,we,seed=key
    o=s['decoded_open_count'];st=s['max_open_streak']
    d=s['success_done_any'];to=s.get('timeout',False)
    if to or not d: label='RANDOM_SENSITIVE';y=1
    elif o<=3 and st<=3: label='RAND_STRICT';y=0
    elif o<=5 and st<=5: label='RAND_USABLE';y=0
    else: label='RAND_BORDERLINE';y=-1
    if y>=0: rows.append({'task':task,'state_id':sid,'ws':ws,'we':we,'seed':seed,
        'rand_open':o,'rand_streak':st,'rand_done':d,'rand_timeout':to,
        'rand_steps':s['n_steps'],'rand_label':label,'target':y})

print('Training samples: %d (pos=%d, neg=%d)'%(len(rows),sum(1 for r in rows if r['target']==1),sum(1 for r in rows if r['target']==0)))

# ── Universe for phase features ──
universe={}
for upath in [T+'/s20i_v031_non_random_sensitive_candidates.csv',
              T+'/s20i_clean_expansion_candidate_universe.csv']:
    if os.path.exists(upath):
        with open(upath) as f:
            for r in csv.DictReader(f):
                universe[(r['task'],r['state_id'],int(r['window_start']),int(r['window_end']))]=r

# ── Transition audit ──
trans_audit={}
for fpath in glob.glob(T+'/s20g_close_transition_audit.csv'):
    with open(fpath) as f:
        for r in csv.DictReader(f):
            trans_audit[(r['task'],r['state_id'],int(r['window_start']),int(r['window_end']),r['seed'])]=r

# ── Build features ──
tasks_all=sorted(set(r['task'] for r in rows))
phases_all=['approach','grasp_transition','early_transport','transport','preplace','place_or_done']

X_rows=[]
for r in rows:
    key=(r['task'],r['state_id'],int(r['ws']),int(r['we']))
    u=universe.get(key,{})
    t=trans_audit.get((r['task'],r['state_id'],int(r['ws']),int(r['we']),r['seed']),{})
    phase=u.get('phase',u.get('phase_id','?'))
    fc=float(u.get('first_close_step',-1)or -1);lift=float(u.get('lift_step',-1)or -1)
    ws=int(r['ws']);we=int(r['we']);wc=(ws+we)/2.0
    dl=float(u.get('done_step',280)or 280)
    X_rows.append({
        'task':r['task'],'phase':phase,
        'fc':fc if fc>0 else -1,'lift':lift if lift>0 else -1,
        'ws_fc':ws-fc if fc>0 else 50,'ws_lift':ws-lift if lift>0 else 50,
        'rel_timing':wc/max(dl,1),
        'clean_open':float(u.get('clean_open_count',0)),
        'clean_open_frac':float(u.get('clean_open_frac',0)),
        'post_grasp_open':float(u.get('post_grasp_open_count',0)),
        'qpos_mean':float(u.get('qpos_mean',0)),
        'eef_disp':float(u.get('eef_disp',0)),
        'dist_trans':float(t.get('distance_to_transition',0)or 0),
        'pre_open_streak':float(t.get('pre_open_streak',0)or 0),
        'post_close_streak':float(t.get('post_close_streak',0)or 0),
        'trans_overlap':int(t.get('transition_overlap_center',0)or 0),
        'close_commit':float(t.get('close_commitment_score',0.5)or 0.5),
    })

feature_groups={
    'TaskOnly': lambda x: [1 if x['task']==tk else 0 for tk in tasks_all],
    'PhaseOnly': lambda x: [1 if x['phase']==p else 0 for p in phases_all],
    'Task+Phase': lambda x: [1 if x['task']==tk else 0 for tk in tasks_all]+[1 if x['phase']==p else 0 for p in phases_all],
    'CleanNoTask': lambda x: [x['clean_open'],x['clean_open_frac'],x['post_grasp_open'],x['qpos_mean'],x['eef_disp']],
    'Clean+Transition': lambda x: [x['clean_open'],x['clean_open_frac'],x['qpos_mean'],x['eef_disp'],x['dist_trans'],x['pre_open_streak'],x['post_close_streak'],x['trans_overlap'],x['close_commit']],
    'Phase+Clean+Trans': lambda x: [x['ws_fc'],x['ws_lift'],x['rel_timing'],x['clean_open'],x['clean_open_frac'],x['qpos_mean'],x['eef_disp'],x['dist_trans'],x['pre_open_streak'],x['post_close_streak'],x['trans_overlap'],x['close_commit']],
    'AllNoTask': lambda x: [x['fc'],x['lift'],x['ws_fc'],x['ws_lift'],x['rel_timing'],x['clean_open'],x['clean_open_frac'],x['post_grasp_open'],x['qpos_mean'],x['eef_disp'],x['dist_trans'],x['pre_open_streak'],x['post_close_streak'],x['trans_overlap'],x['close_commit']],
    'AllWithTask': lambda x: [x['fc'],x['lift'],x['ws_fc'],x['ws_lift'],x['rel_timing'],x['clean_open'],x['clean_open_frac'],x['post_grasp_open'],x['qpos_mean'],x['eef_disp'],x['dist_trans'],x['pre_open_streak'],x['post_close_streak'],x['trans_overlap'],x['close_commit']]+[1 if x['task']==tk else 0 for tk in tasks_all],
}

y=np.array([r['target'] for r in rows])
groups=np.array(['%s_%s'%(r['task'],r['state_id']) for r in rows])
task_groups=np.array([r['task'] for r in rows])

# ── Split manifest ──
train_tasks={'ketchup','milk','tomato_sauce'}
val_tasks={'cream_cheese','salad_dressing'}
test_tasks={'alphabet_soup','bbq_sauce','butter','chocolate_pudding','orange_juice'}

# ── Evaluate all feature+model combinations ──
results=[]
for fg_name,fg_fn in feature_groups.items():
    X=np.array([fg_fn(x) for x in X_rows])
    for model_name,ModelCls,needs_scale in [
        ('LR',LogisticRegression,True),('RF',RandomForestClassifier,False),('GB',GradientBoostingClassifier,False)
    ]:
        # In-distribution GroupKFold
        train_idx=[i for i,r in enumerate(rows) if r['task'] in train_tasks]
        val_idx=[i for i,r in enumerate(rows) if r['task'] in val_tasks]
        test_idx=[i for i,r in enumerate(rows) if r['task'] in test_tasks]

        n_in=len(train_idx);n_pos_in=int(sum(y[train_idx]))
        if n_pos_in<3 or n_in-n_pos_in<3:continue

        # GroupKFold on in-distribution
        n_splits=min(5,len(set(groups[train_idx])))
        gkf=GroupKFold(n_splits=n_splits)
        auroc_gkf=[];auprc_gkf=[];prec_gkf=[];recall_gkf=[];false_clean_gkf=[]
        oof=np.zeros(n_in)
        for tr,te in gkf.split(X[train_idx],y[train_idx],groups[train_idx]):
            if len(set(y[train_idx][tr]))<2:continue
            if needs_scale:
                ss=StandardScaler();Xtr=ss.fit_transform(X[train_idx][tr]);Xte=ss.transform(X[train_idx][te])
            else:Xtr=X[train_idx][tr];Xte=X[train_idx][te]
            kw={'max_iter':2000,'class_weight':'balanced','random_state':42}if model_name=='LR' else {'n_estimators':100,'class_weight':'balanced','random_state':42}if model_name=='RF' else {'n_estimators':100,'random_state':42}
            m=ModelCls(**kw);m.fit(Xtr,y[train_idx][tr])
            yp=m.predict_proba(Xte)[:,1];oof[te]=yp
            auroc_gkf.append(roc_auc_score(y[train_idx][te],yp))
            auprc_gkf.append(average_precision_score(y[train_idx][te],yp))
            yh=(yp>=0.5).astype(int)
            if sum(yh)>0:prec_gkf.append(sum((yh==1)&(y[train_idx][te]==0))/max(sum(yh),1));recall_gkf.append(sum((yh==1)&(y[train_idx][te]==1))/max(sum(y[train_idx][te]==1),1))
            # eligible precision at threshold 0.40
            elig=yp<=0.40;n_elig=sum(elig)
            if n_elig>0:false_clean_gkf.append(np.mean(y[train_idx][te][elig]))

        # Validation set
        if needs_scale and len(train_idx)>0 and len(val_idx)>0:
            ss=StandardScaler();Xtr_s=ss.fit_transform(X[train_idx]);Xval_s=ss.transform(X[val_idx])
        elif len(val_idx)>0:Xtr_s=X[train_idx];Xval_s=X[val_idx]
        else:Xtr_s=None;Xval_s=None

        if Xtr_s is not None and len(val_idx)>0:
            kw={'max_iter':2000,'class_weight':'balanced','random_state':42}if model_name=='LR' else {'n_estimators':100,'class_weight':'balanced','random_state':42}if model_name=='RF' else {'n_estimators':100,'random_state':42}
            m=ModelCls(**kw);m.fit(Xtr_s,y[train_idx])
            yp_val=m.predict_proba(Xval_s)[:,1]
            val_auroc=roc_auc_score(y[val_idx],yp_val) if len(set(y[val_idx]))>1 else float('nan')
            elig_val=yp_val<=0.40;n_elig_val=sum(elig_val)
            val_prec=1-np.mean(y[val_idx][elig_val]) if n_elig_val>0 else 0
            val_abstain=1-n_elig_val/len(val_idx)
        else:val_auroc=float('nan');val_prec=0;val_abstain=1

        # Test set
        if Xtr_s is not None and len(test_idx)>0:
            if needs_scale:Xtest_s=ss.transform(X[test_idx])
            else:Xtest_s=X[test_idx]
            yp_test=m.predict_proba(Xtest_s)[:,1]
            test_auroc=roc_auc_score(y[test_idx],yp_test) if len(set(y[test_idx]))>1 else float('nan')
            elig_test=yp_test<=0.40;n_elig_test=sum(elig_test)
            test_prec=1-np.mean(y[test_idx][elig_test]) if n_elig_test>0 else 0
            test_abstain=1-n_elig_test/len(test_idx)
        else:test_auroc=float('nan');test_prec=0;test_abstain=1

        results.append({
            'feature_group':fg_name,'model':model_name,
            'n_train':n_in,'n_pos_train':n_pos_in,
            'auroc_gkf':round(np.mean(auroc_gkf),3)if auroc_gkf else'',
            'prec_gkf':round(np.mean(prec_gkf),3)if prec_gkf else'',
            'false_clean_gkf':round(np.mean(false_clean_gkf),3)if false_clean_gkf else'',
            'val_auroc':round(val_auroc,3)if not np.isnan(val_auroc)else'',
            'val_prec':round(val_prec,3),
            'val_abstain':round(val_abstain,3),
            'test_auroc':round(test_auroc,3)if not np.isnan(test_auroc)else'',
            'test_prec':round(test_prec,3),
            'test_abstain':round(test_abstain,3),
        })

# ── Write metrics ──
with open(T+'/s20m2_v032_metrics.csv','w',newline='')as f:
    fields=['feature_group','model','n_train','n_pos_train','auroc_gkf','prec_gkf','false_clean_gkf','val_auroc','val_prec','val_abstain','test_auroc','test_prec','test_abstain']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)

print('='*90)
print('V0.3.2 TASK-BALANCED RANDHEAD — TRAINING COMPLETE')
print('='*90)
print('Train: %s (n=%d)'%(train_tasks,len(train_idx)))
print('Val: %s (n=%d)'%(val_tasks,len(val_idx)))
print('Test: %s (n=%d)'%(test_tasks,len(test_idx)))
print()
hdr='%-22s %-3s %5s %4s %7s %7s %7s %7s %7s %7s %7s %7s %7s'
print(hdr%('Feature Group','Mdl','N','Pos','GK_AUC','GK_Prec','GK_FC','V_AUC','V_Prec','V_Abst','T_AUC','T_Prec','T_Abst'))
print('-'*90)
for r in results:
    print('%-22s %-3s %5d %4d %7s %7s %7s %7s %7s %7s %7s %7s %7s'%(
        r['feature_group'],r['model'],r['n_train'],r['n_pos_train'],
        r['auroc_gkf'],r['prec_gkf'],r['false_clean_gkf'],
        r['val_auroc'],r['val_prec'],r['val_abstain'],
        r['test_auroc'],r['test_prec'],r['test_abstain']))

# ── Best by test precision ──
valid=[r for r in results if isinstance(r['test_prec'],float) and r['test_prec']>0]
if valid:
    best=max(valid,key=lambda r:r['test_prec'])
    print()
    print('Best on TEST: %s+%s (test_prec=%.3f, test_abstain=%.3f, val_prec=%.3f)'%(
        best['feature_group'],best['model'],best['test_prec'],best['test_abstain'],best['val_prec']))

# Save config
with open(C+'/stageb_detector_v032.yaml','w')as f:
    json.dump({'version':'v0.3.2','train_tasks':list(train_tasks),'val_tasks':list(val_tasks),'test_tasks':list(test_tasks),'n_train':len(train_idx),'n_val':len(val_idx),'n_test':len(test_idx),'best_fg':best['feature_group']if valid else'','best_model':best['model']if valid else'','threshold':0.40},f,indent=2)

print()
print('Config: %s'%(C+'/stageb_detector_v032.yaml'))
print('Metrics: %s'%(T+'/s20m2_v032_metrics.csv'))
