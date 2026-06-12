#!/usr/bin/env python3
"""S20L-v2: RAND-only fresh test with clean split. No VIS. Test tasks only."""
import csv, json, glob, os, numpy as np
from collections import Counter

T='/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O='/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613'
os.makedirs(O+'/queues',exist_ok=True)

test_tasks={'alphabet_soup','bbq_sauce','orange_juice','butter','chocolate_pudding'}
held_out={('tomato_sauce','0',70,80),('ketchup','0',150,160)}
W=10;STR=5

def detect_phases(rows):
    def g(row,key,d=0.0):
        try:return float(row.get(key,d) or d)
        except:return d
    is_open=[g(r,'decoded_open_bool') for r in rows]
    fc=None;stk=0
    for i,o in enumerate(is_open):
        if o==0:
            if stk==0:fc=i
            stk+=1
            if stk>=3:break
        else:stk=0;fc=None
    if stk<3:fc=None
    bo=float(np.median([g(r,'obj_z') for r in rows[:5]])) if len(rows)>=5 else 0
    be=float(np.median([g(r,'eef_z') for r in rows[:5]])) if len(rows)>=5 else 0
    lift=None
    for i in range(fc or 0,len(rows)):
        if g(rows[i],'obj_z')-bo>=0.015 or g(rows[i],'eef_z')-be>=0.03:
            lift=i;break
    pp=None;pz=None;ps=None
    if lift is not None:
        for i in range(lift,len(rows)):
            z=g(rows[i],'eef_z')
            if pz is None or z>pz:pz=z;ps=i
            if ps is not None and i>ps+3 and pz-z>=0.005:
                pp=i;break
    done=None
    for i,r in enumerate(rows):
        if int(r.get('success_primary','0') or '0')==1 or int(r.get('success_done','0') or '0')==1:
            done=i;break
    if done is None:done=len(rows)-1
    return{'fc':fc,'lift':lift,'pp':pp,'done':done,'ms':len(rows)}

def phase_id(ws,we,ph):
    wc=(ws+we)/2.0
    if wc>=ph['done']-5:return'place_or_done'
    if ph['pp'] is not None and wc>=ph['pp']:return'preplace'
    if ph['lift'] is not None and wc>=ph['lift']+5:return'transport'
    if ph['lift'] is not None and wc>=ph['lift']:return'early_transport'
    if ph['fc'] is not None and wc>=ph['fc']:return'grasp_transition'
    return'approach'

existing=set()
for d in ['/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
          '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
          '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613']:
    for f in glob.glob(d+'/summary_*.json'):
        s=json.load(open(f))
        existing.add((s['task'],str(s['state_id']),s['window_start'],s['window_end']))

candidates=[]
for d in ['/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
          '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612']:
    for sf in sorted(glob.glob(d+'/summary_*clean*.json')):
        s=json.load(open(sf))
        if not s.get('success_done_any',False):continue
        task=s['task'];sid=str(s['state_id'])
        if task not in test_tasks:continue
        tp=d+'/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv'%(task,sid)
        tr=sorted(glob.glob(tp))
        if not tr:continue
        with open(tr[0]) as f:rows=list(csv.DictReader(f))
        ph=detect_phases(rows)
        for ws in range(5,ph['ms']-W,STR):
            we=ws+W
            if we>ph['done']+5:continue
            if(task,sid,ws,we)in held_out:continue
            if(task,sid,ws,we)in existing:continue
            phase=phase_id(ws,we,ph)
            wr=[r for r in rows if ws<=int(r['step'])<we]
            def g(row,key,d=0.0):
                try:return float(row.get(key,d) or d)
                except:return d
            co=sum(1 for r in wr if g(r,'decoded_open_bool')==1)
            cf=co/W
            if phase in('transport','preplace')and cf<=0.1:tier='eligible_strict'
            elif phase in('transport','preplace','grasp_transition','early_transport')and cf<=0.2:tier='eligible_usable'
            elif cf<=0.3:tier='eligible_usable'
            else:tier='predicted_random_sensitive'
            candidates.append({'task':task,'sid':sid,'ws':ws,'we':we,'phase':phase,'tier':tier,'clean_open':co})

print('Candidates: %d'%len(candidates))
tc=Counter(c['task']for c in candidates)
print('Tasks: %s'%dict(tc))
print('Phases: %s'%dict(Counter(c['phase']for c in candidates)))
print('Tiers: %s'%dict(Counter(c['tier']for c in candidates)))

selected=[];task_n=Counter();phase_n=Counter();adj_n=Counter()
tier_limits={'eligible_strict':12,'eligible_usable':8,'predicted_random_sensitive':3}
candidates.sort(key=lambda c:{'eligible_strict':0,'eligible_usable':1,'predicted_random_sensitive':2}.get(c['tier'],9))

for c in candidates:
    if len(selected)>=23:break
    if task_n[c['task']]>=6:continue
    if phase_n[c['phase']]>=23*0.35:continue
    if len([s for s in selected if s['tier']==c['tier']])>=tier_limits.get(c['tier'],99):continue
    adj_key=(c['task'],c['sid'])
    if adj_n[adj_key]>=2:continue
    selected.append(c)
    task_n[c['task']]+=1;phase_n[c['phase']]+=1;adj_n[adj_key]+=1

print()
print('Selected: %d'%len(selected))
print('Tasks: %s'%dict(task_n))
print('Phases: %s'%dict(phase_n))
print('Tiers: %s'%dict(Counter(c['tier']for c in selected)))

jobs=[];jid=270000
for c in selected:
    cid='%s_s%s_w%d_%d'%(c['task'],c['sid'],c['ws'],c['we'])
    jid+=1;jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['sid'],'window_start':str(c['ws']),'window_end':str(c['we']),'condition':'random_linf','attack_seed':'89','random_control_seed':'89','seed':'0','candidate_id':cid,'tier':'L2_'+c['tier'],'track':'S20Lv2','status':'pending'})

queues={'gpu10':[],'gpu26':[],'gpu45':[]}
gpus=['gpu10','gpu26','gpu45']
for i,j in enumerate(jobs):
    queues[gpus[i%3]].append(j)

for gpu,gj in queues.items():
    qp=O+'/queues/s20l_v2_%s.csv'%gpu
    with open(qp,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(jobs[0].keys()));w.writeheader();w.writerows(gj)
    print('%s: %d RAND jobs'%(gpu,len(gj)))

with open(T+'/s20l_v2_randonly_manifest.csv','w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=['task','state_id','ws','we','phase','tier','clean_open'])
    w.writeheader()
    for c in selected:w.writerow({k:c[k] for k in['task','sid','ws','we','phase','tier','clean_open']})

print('Total: %d RAND-only (seed89, no VIS)'%len(jobs))
print('Output: %s'%O)
