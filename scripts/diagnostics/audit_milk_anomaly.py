#!/usr/bin/env python3
"""Audit milk[235,245] retry anomaly."""
import json, csv, os, glob
import hashlib

DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'

# Load all milk[235,245] summaries
milk_jobs = []
for f in glob.glob(os.path.join(DIR, 'summary_*.json')):
    with open(f) as fh:
        j = json.load(fh)
    if j.get('pair_id') == 'k5c_cmd_milk_neg':
        milk_jobs.append(j)

milk_jobs.sort(key=lambda j: (j.get('attack_seed', 0), j.get('condition', '')))

print('=== All milk[235,245] jobs ===')
for j in milk_jobs:
    jid = j.get('job_id')
    cond = j.get('condition', '?')
    atk = j.get('attack_seed')
    opn = j.get('decoded_open_count')
    streak = j.get('decoded_longest_open_streak')
    n_steps = j.get('n_total_steps')
    ws = j.get('window_start')
    we = j.get('window_end')
    qpos = j.get('qpos_delta', 0)
    arm = j.get('mean_arm_l2', 0)
    infra = j.get('infra_status', '?')
    env_s = j.get('env_seed')
    atk_s = j.get('attack_seed')
    succ = j.get('success', '?')
    print(f'  job={jid} cond={cond:12s} atk={atk} env_s={env_s} open={opn} streak={streak} '
          f'n_steps={n_steps} ws={ws} we={we} qpos={qpos:.4f} arm={arm} infra={infra} succ={succ}')

# Check trace pre-window consistency
print('\n=== Trace pre-window (steps 0-9) env_action_6 ===')
for j in milk_jobs:
    jid = j.get('job_id')
    cond = j.get('condition', '?')
    atk = j.get('attack_seed')
    task = j.get('task_key', 'butter')

    # Find trace file
    trace_f = None
    for alt in glob.glob(os.path.join(DIR, 'trace_*.csv')):
        if 'job' + str(jid) in alt:
            trace_f = alt
            break
    if not trace_f:
        print(f'  job={jid} cond={cond:12s} atk={atk}: NO TRACE')
        continue

    with open(trace_f) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    n = len(rows)

    # First 10 raw_gripper values
    rg = [r.get('raw_gripper', '?') for r in rows[:10]]
    ea = [r.get('env_action_6', '?') for r in rows[:10]]

    # Window start
    ws = j.get('window_start', 0)
    we = j.get('window_end', 0)
    if ws < n and we < n:
        win_rg = [r.get('raw_gripper', '?') for r in rows[ws:we+1]]
        win_ea = [r.get('env_action_6', '?') for r in rows[ws:we+1]]
    else:
        win_rg = ['OOB']
        win_ea = ['OOB']

    # MD5 of pre-window raw_gripper (steps 0 to window_start-1) as prefix fingerprint
    pre_rg = ','.join(r.get('raw_gripper', '0') for r in rows[:ws])
    pre_hash = hashlib.md5(pre_rg.encode()).hexdigest()[:8]

    print(f'  job={jid} cond={cond:12s} atk={atk} n={n} pre_hash={pre_hash}')
    print(f'    pre(0-9)   rg={rg}')
    print(f'    pre(0-9)   ea={ea}')
    print(f'    win({ws}-{we}) rg={win_rg}')
    print(f'    win({ws}-{we}) ea={win_ea}')

# Check: retried jobs have different GPU
print('\n=== Summary ===')
vis_opens = [j.get('decoded_open_count') for j in milk_jobs if j.get('condition') == 'vis_pgd' and j.get('infra_status') == 'ok']
rand_opens = [j.get('decoded_open_count') for j in milk_jobs if j.get('condition') == 'random_linf' and j.get('infra_status') == 'ok']
print(f'VIS opens (infra=ok only): {vis_opens}')
print(f'RAND opens (infra=ok only): {rand_opens}')
print(f'VIS mean={sum(vis_opens)/len(vis_opens):.1f} RAND mean={sum(rand_opens)/len(rand_opens):.1f}')
