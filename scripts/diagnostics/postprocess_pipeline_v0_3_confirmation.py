#!/usr/bin/env python3
"""Pipeline v0.3 Fresh Confirmation Postprocess.
Groups by logical_pair_key = pair_id + '__atk' + attack_seed.
Outputs: job_audit, pair_audit, group_metrics, window_results.
"""
import json, glob, os, csv
from collections import defaultdict, Counter
import numpy as np

DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_confirmation'
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
TABLES = os.path.join(REPO, 'tables')
REPORTS = os.path.join(REPO, 'reports')

os.makedirs(TABLES, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# ── Load ──
summaries = []
for f in glob.glob(os.path.join(DIR, 'summary_*.json')):
    with open(f) as fh:
        summaries.append(json.load(fh))

MANIFEST_PATH = os.path.join(REPO, 'tables/pipeline_v0_3_confirmation_launch_manifest.csv')
manifest = {}
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        for r in csv.DictReader(f):
            manifest[int(r['job_id'])] = r

# ── 1. Job Audit ──
job_rows = []
for j in summaries:
    jid = j.get('job_id', -1)
    m = manifest.get(jid, {})
    job_rows.append({
        'job_id': jid,
        'pair_id': j.get('pair_id', '?'),
        'logical_pair_key': '%s__atk%s' % (j.get('pair_id', '?'), j.get('attack_seed', '?')),
        'task': j.get('task_key', '?'),
        'state_id': j.get('state_id', '?'),
        'env_seed': j.get('env_seed', '?'),
        'attack_seed': j.get('attack_seed', '?'),
        'condition': j.get('condition', '?'),
        'window_start': j.get('window_start', 0),
        'window_end': j.get('window_end', 0),
        'n_total_steps': j.get('n_total_steps', 0),
        'decoded_open_count': j.get('decoded_open_count', 0),
        'qpos_delta': round(j.get('qpos_delta', 0), 6),
        'infra_status': j.get('infra_status', '?'),
        'group': m.get('group', '?'),
        'expected_strategy': m.get('expected_strategy', '?'),
        'worker': m.get('worker', '?'),
        'cuda_visible_devices': m.get('cuda_visible_devices', '?'),
    })

with open(os.path.join(TABLES, 'pipeline_v0_3_confirmation_job_audit.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['job_id','pair_id','logical_pair_key','task','state_id','env_seed','attack_seed',
                'condition','window_start','window_end','n_total_steps','decoded_open_count',
                'qpos_delta','infra_status','group','expected_strategy','worker','cuda_visible_devices'])
    for r in job_rows:
        w.writerow([r['job_id'],r['pair_id'],r['logical_pair_key'],r['task'],r['state_id'],
                    r['env_seed'],r['attack_seed'],r['condition'],r['window_start'],r['window_end'],
                    r['n_total_steps'],r['decoded_open_count'],r['qpos_delta'],r['infra_status'],
                    r['group'],r['expected_strategy'],r['worker'],r['cuda_visible_devices']])

# ── 2. Pair Audit ──
by_lp = defaultdict(list)
for j in summaries:
    lpk = '%s__atk%s' % (j.get('pair_id', '?'), j.get('attack_seed', '?'))
    by_lp[lpk].append(j)

pair_rows = []
for lpk, jobs in sorted(by_lp.items()):
    vis_j = [j for j in jobs if j.get('condition') == 'vis_pgd']
    rand_j = [j for j in jobs if j.get('condition') == 'random_linf']
    v = vis_j[0] if vis_j else None
    r = rand_j[0] if rand_j else None

    win_size = 0
    if v: win_size = v.get('window_end', 0) - v.get('window_start', 0) + 1
    thr = max(1, win_size // 2)

    v_open = v.get('decoded_open_count', 0) if v else -1
    r_open = r.get('decoded_open_count', 0) if r else -1
    v_cmd = 1 if v_open >= thr else 0 if v else -1
    r_cmd = 1 if r_open >= thr else 0 if r else -1
    v_phys = 1 if (v.get('qpos_delta', 0) if v else 0) >= 0.01 else 0
    r_phys = 1 if (r.get('qpos_delta', 0) if r else 0) >= 0.01 else 0
    v_qpos = v.get('qpos_delta', 0) if v else -1
    r_qpos = r.get('qpos_delta', 0) if r else -1

    vis_n = len(vis_j); rand_n = len(rand_j)
    pair_ok = (vis_n == 1 and rand_n == 1)
    group = '?'
    for g in ['A','B','C']:
        if ('conf_' + g) in lpk: group = g; break

    yield_cmd = v_cmd - r_cmd if pair_ok else -99
    is_rand = 1 if r_cmd == 1 else 0
    is_cmd = 1 if (v_cmd == 1 and r_cmd == 0) else 0
    is_abstain = 1 if (r_cmd == 1 or r_phys == 1) else 0

    pair_rows.append({
        'logical_pair_key': lpk, 'group': group,
        'pair_id': lpk.split('__')[0], 'attack_seed': lpk.split('atk')[-1],
        'task': v.get('task_key', '?') if v else '?',
        'window_start': v.get('window_start', 0) if v else 0,
        'window_end': v.get('window_end', 0) if v else 0,
        'window_size': win_size,
        'vis_job_id': v.get('job_id', -1) if v else -1,
        'rand_job_id': r.get('job_id', -1) if r else -1,
        'vis_open': v_open, 'rand_open': r_open,
        'vis_cmd': v_cmd, 'rand_cmd': r_cmd,
        'vis_phys': v_phys, 'rand_phys': r_phys,
        'vis_qpos': round(v_qpos, 6), 'rand_qpos': round(r_qpos, 6),
        'yield_cmd': yield_cmd,
        'is_rand': is_rand, 'is_cmd': is_cmd, 'is_abstain': is_abstain,
        'pair_ok': pair_ok,
        'fp_fn_note': '',
    })

# Tag FP/FN
for r in pair_rows:
    if 'B1_tomato_w55_65' in r['logical_pair_key']:
        r['fp_fn_note'] = 'detector_FP: CleanRand marked high-risk but fresh VIS=7,RAND=0 (GOLD)'
    if 'B4_salad_w70_80' in r['logical_pair_key']:
        r['fp_fn_note'] = 'detector_FN: CleanRand passed but fresh VIS=4,RAND=10-11 (rand)'

with open(os.path.join(TABLES, 'pipeline_v0_3_confirmation_pair_audit.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['logical_pair_key','group','pair_id','attack_seed','task','window_start','window_end',
                'window_size','vis_job_id','rand_job_id','vis_open','rand_open',
                'vis_cmd','rand_cmd','vis_phys','rand_phys','vis_qpos','rand_qpos',
                'yield_cmd','is_rand','is_cmd','is_abstain','pair_ok','fp_fn_note'])
    for r in pair_rows:
        w.writerow([r['logical_pair_key'],r['group'],r['pair_id'],r['attack_seed'],
                    r['task'],r['window_start'],r['window_end'],r['window_size'],
                    r['vis_job_id'],r['rand_job_id'],r['vis_open'],r['rand_open'],
                    r['vis_cmd'],r['rand_cmd'],r['vis_phys'],r['rand_phys'],
                    r['vis_qpos'],r['rand_qpos'],r['yield_cmd'],
                    r['is_rand'],r['is_cmd'],r['is_abstain'],r['pair_ok'],r['fp_fn_note']])

# ── 3. Group Metrics ──
groups = {'A': [], 'B': [], 'C': []}
for r in pair_rows:
    if r['group'] in groups:
        groups[r['group']].append(r)

group_names = {'A': 'CleanRand-pass', 'B': 'TaskOnly baseline', 'C': 'High-risk abstain'}
metric_rows = []
for grp in ['A','B','C']:
    data = groups[grp]
    n = len(data)
    if n == 0: continue
    cmd_hit = sum(1 for d in data if d['is_cmd']) / n
    cmd_rand = sum(1 for d in data if d['is_rand']) / n
    abst_any = sum(1 for d in data if d['is_abstain']) / n
    pV = sum(d['vis_cmd'] for d in data) / n
    pR = sum(d['rand_cmd'] for d in data) / n
    yld = pV - pR
    vq = np.mean([d['vis_qpos'] for d in data])
    rq = np.mean([d['rand_qpos'] for d in data])
    v_open_mean = np.mean([d['vis_open'] for d in data])
    r_open_mean = np.mean([d['rand_open'] for d in data])
    metric_rows.append({
        'group': grp, 'group_name': group_names[grp], 'n_logical_pairs': n,
        'cmd_hit': round(cmd_hit, 4), 'cmd_rand_hit': round(cmd_rand, 4),
        'abstain_any_hit': round(abst_any, 4),
        'pV_cmd': round(pV, 4), 'pR_cmd': round(pR, 4),
        'mean_yield_cmd': round(yld, 4),
        'mean_vis_open': round(v_open_mean, 2), 'mean_rand_open': round(r_open_mean, 2),
        'mean_vis_qpos': round(vq, 6), 'mean_rand_qpos': round(rq, 6),
    })

with open(os.path.join(TABLES, 'pipeline_v0_3_confirmation_group_metrics.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','group_name','n_logical_pairs','cmd_hit','cmd_rand_hit','abstain_any_hit',
                'pV_cmd','pR_cmd','mean_yield_cmd','mean_vis_open','mean_rand_open',
                'mean_vis_qpos','mean_rand_qpos'])
    for r in metric_rows:
        w.writerow([r['group'],r['group_name'],r['n_logical_pairs'],r['cmd_hit'],r['cmd_rand_hit'],
                    r['abstain_any_hit'],r['pV_cmd'],r['pR_cmd'],r['mean_yield_cmd'],
                    r['mean_vis_open'],r['mean_rand_open'],r['mean_vis_qpos'],r['mean_rand_qpos']])

# ── 4. Window Results ──
with open(os.path.join(TABLES, 'pipeline_v0_3_confirmation_window_results.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','pair_id','attack_seed','task','window','vis_open','rand_open',
                'vis_cmd','rand_cmd','vis_phys','rand_phys','yield_cmd','fp_fn_note'])
    for r in pair_rows:
        win = '%d-%d' % (r['window_start'], r['window_end'])
        w.writerow([r['group'],r['pair_id'],r['attack_seed'],r['task'],win,
                    r['vis_open'],r['rand_open'],r['vis_cmd'],r['rand_cmd'],
                    r['vis_phys'],r['rand_phys'],r['yield_cmd'],r['fp_fn_note']])

# ── Print summary ──
print('Postprocess complete.')
print('Job audit: %d rows' % len(job_rows))
print('Pair audit: %d logical pairs, %d bad' % (len(pair_rows), sum(1 for r in pair_rows if not r['pair_ok'])))
print('Group metrics: %d groups' % len(metric_rows))
print()
for r in metric_rows:
    print('  %-22s n=%d  cmd_hit=%.2f  cmd_rand=%.2f  yield=%+.2f' % (
        r['group_name'], r['n_logical_pairs'], r['cmd_hit'], r['cmd_rand_hit'], r['mean_yield_cmd']))
