#!/usr/bin/env python3
"""D6C-v3 simple: loop-based GRU replay (no batch, no complex pre-allocation)."""
import sys, json, csv, time, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

class GRU(torch.nn.Module):
    def __init__(self, nf=25, nc=108, hidden=128):
        super().__init__()
        self.gru = torch.nn.GRU(nf, hidden, 1, batch_first=True)
        self.head = torch.nn.Linear(hidden+nc, 2)
    def forward(self, xt, xc):
        _, h = self.gru(xt); return self.head(torch.cat([h[-1], xc], dim=1))

def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))

SC5 = ['gripper_command','gripper_qpos','gripper_opening_proxy','eef_x','eef_y','eef_z',
       'eef_vx','eef_vy','eef_vz','action_dx','action_dy','action_dz','action_gripper',
       'recent_close_streak','recent_open_streak','recent_gripper_flip_count',
       'close_onset','time_since_close','eef_speed','eef_z_delta_since_close',
       'qpos_delta_1','qpos_delta_3','opening_proxy_delta_3','opening_proxy_variance_5','eef_speed_variance_5']

def read_csv(path):
    with open(path,'r',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def load_norm(p):
    o=json.loads(open(p).read())
    return [np.asarray(o[k],np.float32) for k in ['temporal_feature_mean','temporal_feature_std','context_feature_mean','context_feature_std']]

def replay_worker(args):
    path=args['path']; is_rep=args.get('is_rep',False); te=args['te']; ts=args['ts']; W=16
    try: rows=read_csv(path)
    except: return {'path':path,'error':'read','n':0}
    n=len(rows)
    if n<W: return {'path':path,'error':'short','n':n}
    xt_all=[]
    for r in rows:
        vals=[]
        for f in SC5:
            try:
                v=r.get(f, r.get(f'f_{f}','')) if is_rep else r.get(f'f_{f}',r.get(f,''))
                vals.append(float(v) if v and v!='None' else 0.0)
            except: vals.append(0.0)
        xt_all.append(vals)
    xt=np.array(xt_all,dtype=np.float32)
    xt=(xt-args['tm'])/np.maximum(args['ts_n'],1e-8)
    xc=np.zeros((n,108),dtype=np.float32)
    model=GRU(25,108,args['hidden'])
    model.load_state_dict(args['state']); model.cpu().eval()
    tc=0; ft=-1; es=0.0; ss=0.0; nw=n-W+1
    with torch.no_grad():
        for i in range(W-1,n):
            w=torch.from_numpy(xt[i-W+1:i+1]).float().unsqueeze(0)
            c=torch.from_numpy(xc[i]).float().unsqueeze(0)
            lo=model(w,c).numpy()[0]; ep=sig(lo[0]); sp=sig(lo[1])
            es+=ep; ss+=sp
            if ep>=te and sp<=ts:
                tc+=1
                if ft<0: ft=i
    return {'path':path,'n':n,'nw':nw,'tc':tc,'ft':ft,'ep':es/nw if nw>0 else 0,'sp':ss/nw if nw>0 else 0,'err':''}

if __name__=='__main__':
    tm,ts_n,cm,cs=load_norm('/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d4c2e1_temporal_dataset/c2e1_w16_normalization_stats_train_only.json')
    ckpt=torch.load('/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d4c2e3_25d_baseline_package/c2e3_selected_baseline_model.pt',map_location='cpu')
    th=ckpt['threshold']; state=ckpt['model_state_dict']; hidden=ckpt['config']['channels']
    te,ts_v=float(th['tau_emit']),float(th['tau_suppress'])
    print(f'te={te} ts={ts_v} hidden={hidden}')
    ctx=read_csv('/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d4c1b_context_runtime_objective_dataset_v1b/context_detector_dataset_v1b.csv')
    repair=read_csv('/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d4c2e0f_object_base13_repair_v2/c2e0f_repair_manifest.csv')
    repair_map={}
    for r in repair:
        if r.get('status')=='REPAIRED': repair_map[r['original_temporal_path']]=r['repaired_feature_path']
    seen=set(); artifacts=[]; suite_of={}
    for r in ctx:
        tp=r.get('temporal_path','')
        if tp and tp not in seen:
            seen.add(tp); ap=repair_map.get(tp,tp)
            artifacts.append({'path':ap,'is_rep':ap!=tp})
            suite_of[ap]=r.get('suite','?')
    print(f'{len(artifacts)} artifacts ({len(repair_map)} repaired)')
    worker_args=[{'path':a['path'],'is_rep':a['is_rep'],'te':te,'ts':ts_v,'tm':tm,'ts_n':ts_n,'hidden':hidden,'state':state} for a in artifacts]
    t0=time.time(); results=[]; done=0
    with ProcessPoolExecutor(max_workers=64) as pool:
        futures={pool.submit(replay_worker,wa):wa for wa in worker_args}
        for fut in futures:
            results.append(fut.result()); done+=1
            if done%500==0: print(f'  {done}/{len(artifacts)}')
    agg=defaultdict(lambda:{'n':0,'trig':0,'tsum':0,'ft':[],'ep':[],'sp':[],'err':0})
    for rr in results:
        s=suite_of.get(rr['path'],'?'); a=agg[s]; a['n']+=1
        if rr.get('err'): a['err']+=1
        else:
            a['tsum']+=rr['tc']/max(1,rr['nw'])
            if rr['tc']>0: a['trig']+=1
            if rr['ft']>=0: a['ft'].append(rr['ft'])
            a['ep'].append(rr['ep']); a['sp'].append(rr['sp'])
    for s in sorted(agg):
        a=agg[s]; n=a['n']
        print(f"{s}: n={n} any_trig={a['trig']/max(1,n):.3f} trig_rate={a['tsum']/max(1,n-a['err']):.3f} med_ft={np.median(a['ft']) if a['ft'] else 'N/A'} ep={np.mean(a['ep']) if a['ep'] else 0:.3f} sp={np.mean(a['sp']) if a['sp'] else 0:.3f}")
    dt=time.time()-t0; errs=sum(a['err'] for a in agg.values())
    print(f'Done: {dt:.0f}s, {errs} errors')
