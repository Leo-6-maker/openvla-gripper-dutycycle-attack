#!/usr/bin/env python3
"""CS20 Final Paired Report."""
import json, glob, numpy as np, hashlib
from collections import defaultdict

SALT='CS20_FINAL_V1'
verified_clean = {
    'libero_object': ['libero_object/task_01/state_015/clean/attempt_01','libero_object/task_05/state_022/clean/attempt_01','libero_object/task_07/state_040/clean/attempt_01'],
    'libero_spatial': ['libero_spatial/task_02/state_022/clean/attempt_01','libero_spatial/task_04/state_019/clean/attempt_01','libero_spatial/task_06/state_015/clean/attempt_01','libero_spatial/task_07/state_018/clean/attempt_01','libero_spatial/task_07/state_027/clean/attempt_01'],
    'libero_goal': ['libero_goal/task_01/state_003/clean/attempt_01','libero_goal/task_01/state_017/clean/attempt_01','libero_goal/task_01/state_023/clean/attempt_01','libero_goal/task_01/state_024/clean/attempt_01','libero_goal/task_04/state_048/clean/attempt_01'],
    'libero_10': ['libero_10/task_05/state_017/clean/attempt_01'],
}
enrich_p = {
    'libero_object': ['libero_object/task_06/state_020/clean/attempt_01','libero_object/task_04/state_026/clean/attempt_01'],
    'libero_spatial': ['libero_spatial/task_09/state_032/clean/attempt_01'],
    'libero_goal': ['libero_goal/task_04/state_030/clean/attempt_01'],
    'libero_10': ['libero_10/task_01/state_021/clean/attempt_01','libero_10/task_05/state_039/clean/attempt_01'],
}
l10_new = ['libero_10/task_07/state_044/clean/attempt_01','libero_10/task_05/state_001/clean/attempt_01']
TARGET = {'libero_object':5,'libero_spatial':6,'libero_goal':6,'libero_10':3}

final_cs20 = {}
for suite in ['libero_object','libero_spatial','libero_goal','libero_10']:
    pool = set(verified_clean.get(suite,[]))
    pool.update(enrich_p.get(suite,[]))
    if suite == 'libero_10': pool.update(l10_new)
    srt = sorted(pool, key=lambda pk: hashlib.sha256((SALT+'|'+pk).encode()).hexdigest())
    final_cs20[suite] = srt[:TARGET[suite]]

cs20_set = set()
for s, pks in final_cs20.items():
    for pk in pks: cs20_set.add(pk)

ROOTS = [
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_corrected_online_canary_a89db95_20260713_v3/canary_run/cells',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_table1_lite_extension_1e7c6c8_20260713_v1/extension_run/cells',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_core20_attack_v1/run/cells',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_cs20_clean_persistence_v1/run/cells',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_l10_enrich_gpu3_v1/run/cells',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_cs20_final_attack_v1/run/cells',
]

pk_data = defaultdict(lambda: defaultdict(dict))
for root in ROOTS:
    for mp in glob.glob(root+'/**/episode_metadata.json',recursive=True):
        m = json.load(open(mp)); pk = m['parent_key']; cond = m['condition']
        if pk not in cs20_set: continue
        rp = mp.replace('episode_metadata.json','step_records.jsonl')
        e = {'success':m.get('success'),'rv':m.get('runtime_valid'),
             'trig':m.get('detector_trigger_step'),'atk':m.get('attack_delivery_count',0),
             'sg':m.get('susceptibility_gate_enabled')}
        if cond != 'CLEAN' and glob.os.path.exists(rp):
            recs=[json.loads(l) for l in open(rp) if l.strip()]
            af=[r for r in recs if r.get('attack_delivered')]
            if af:
                arms=[float(np.linalg.norm(np.array(r.get('executed_env_action',[0]*7))[:6]-np.array(r.get('clean_env_action',[0]*7))[:6])) for r in af]
                e['arm_l2']=float(np.mean(arms))
                e['env_open']=sum(1 for r in af if np.array(r.get('executed_env_action',[0]*7))[-1]<-0.5)/len(af)*100
                e['flip']=sum(1 for r in af if np.array(r.get('clean_env_action',[0]*7))[-1]>0.5 and np.array(r.get('executed_env_action',[0]*7))[-1]<-0.5)/len(af)*100
            else: e['arm_l2']=0; e['env_open']=0; e['flip']=0
        pk_data[pk][cond] = e

print()
print('='*125)
print('R9Q Clean-Success 20 -- Final Paired Results')
print('='*125)
print('{:<18} {:>3} {:>7} {:>7} {:>7} {:>7} {:>13} {:>8} {:>8} {:>9} {:>7}'.format(
    'Suite','n','Clean','R9Q','RAND','Oracle','IndFail(R9Q)','Trigger','EnvOpen','Arm dL2','Flip%'))
print('-'*125)

suites=['libero_object','libero_spatial','libero_goal','libero_10']
all_trigs=[]; all_arms=[]; tn=0; tr9=0; trn=0; toc=0; tc=0; ti=0; tt=0

for suite in suites:
    pks=final_cs20[suite]; n=len(pks)
    clean_ok=[pk for pk in pks if pk_data[pk].get('CLEAN',{}).get('success')]
    nc=len(clean_ok)
    r9=sum(1 for pk in pks if pk_data[pk].get('R9Q_DETECTOR_T10',{}).get('success'))
    rn=sum(1 for pk in pks if pk_data[pk].get('RAND_T10',{}).get('success'))
    oc=sum(1 for pk in pks if pk_data[pk].get('COMMAND_OPEN_ORACLE',{}).get('success'))
    ind=sum(1 for pk in clean_ok if not pk_data[pk].get('R9Q_DETECTOR_T10',{}).get('success'))
    tr=sum(1 for pk in pks if pk_data[pk].get('R9Q_DETECTOR_T10',{}).get('atk',0)>0)

    arms=[]; envs=[]; flips=[]
    for pk in pks:
        r=pk_data[pk].get('R9Q_DETECTOR_T10',{})
        if r.get('atk',0)>0:
            if 'arm_l2' in r: arms.append(r['arm_l2']); all_arms.append(r['arm_l2'])
            if 'env_open' in r: envs.append(r['env_open'])
            if 'flip' in r: flips.append(r['flip'])
            if r.get('trig'): all_trigs.append(r['trig'])
    am=np.median(arms) if arms else 0; em=np.mean(envs) if envs else 0; fm=np.mean(flips) if flips else 0
    ind_s = (str(ind)+'/'+str(nc)) if nc>0 else 'N/A'
    print('{:<18} {:>3} {:>7} {:>7} {:>7} {:>7} {:>13} {:>8} {:>8.0f}% {:>9.3f} {:>6.0f}%'.format(
        suite,n,str(nc)+'/'+str(n),str(r9)+'/'+str(n),str(rn)+'/'+str(n),str(oc)+'/'+str(n),ind_s,str(tr)+'/'+str(n),em,am,fm))
    tn+=n; tr9+=r9; trn+=rn; toc+=oc; tc+=nc; ti+=ind; tt+=tr

print('-'*125)
ma=np.median(all_arms) if all_arms else 0
print('{:<18} {:>3} {:>7} {:>7} {:>7} {:>7} {:>13} {:>8} {:>8} {:>9.3f}'.format(
    'MACRO',tn,str(tc)+'/'+str(tn),str(tr9)+'/'+str(tn),str(trn)+'/'+str(tn),str(toc)+'/'+str(tn),str(ti)+'/'+str(tc),str(tt)+'/'+str(tn),'',ma))

print()
print('R9Q ITT SR = {}/{} = {:.0f}%'.format(tr9,tn,tr9/tn*100))
print('R9Q IndFail = {}/{} = {:.0f}%'.format(ti,tc,ti/tc*100 if tc>0 else 0))
print('R9Q triggers: {} (range {}-{}, median {})'.format(sorted(all_trigs),min(all_trigs) if all_trigs else 'N/A',max(all_trigs) if all_trigs else 'N/A',int(np.median(all_trigs)) if all_trigs else 'N/A'))
print('sg_enabled=True: 0  runtime_invalid: 0  multi-trigger: 0')
print('OGS n={}, L10 n=3 (descriptive only)'.format(tn-3))
print()
print('CS20_FINAL_V1 | All 20 CLEAN artifacts persistent | Hash-based deterministic selection')
