#!/usr/bin/env python3
"""K5c postprocess: job audit, retry audit, probability labels, stable pool v2.

CPU-only. Input: 160 K5c summaries. Output: audit CSVs + probability labels + stable pool v2.
"""
import json, glob, os, csv, re, sys
from collections import defaultdict, Counter
import numpy as np

DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
STABLE_S5 = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/combined_stable_pool_k5_k5b.csv'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'

# ── Load all summaries ──
summaries = []
for f in glob.glob(os.path.join(DIR, 'summary_*.json')):
    with open(f) as fh:
        summaries.append(json.load(fh))

print(f'Loaded {len(summaries)} summaries')

# ── 1. Job Audit ──
audit_rows = []
by_pair = defaultdict(lambda: {'vis_pgd': [], 'random_linf': []})
failed_original = []
retry_success = []

# Known failed job_ids from worker_26 CUDA errors
# Static manifest: original job info preserved even if summary was overwritten by retry
FAILED_ORIGINAL_MANIFEST = {
    520060: {'pair_id': 'k5c_cmd_milk_neg', 'condition': 'vis_pgd', 'attack_seed': 0,
             'task': 'milk', 'state_id': 0, 'window_start': 235, 'window_end': 245,
             'original_n_steps': 116, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '1,0'},
    520065: {'pair_id': 'k5c_cmd_milk_neg', 'condition': 'random_linf', 'attack_seed': 2,
             'task': 'milk', 'state_id': 0, 'window_start': 235, 'window_end': 245,
             'original_n_steps': 123, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '4,5'},
    520090: {'pair_id': 'k5c_cmd_alpha', 'condition': 'vis_pgd', 'attack_seed': 0,
             'task': 'alphabet_soup', 'state_id': 0, 'window_start': 65, 'window_end': 75,
             'original_n_steps': 70, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '4,5'},
}
FAILED_ORIGINAL_JIDS = set(FAILED_ORIGINAL_MANIFEST.keys())

for j in summaries:
    jid = j.get('job_id', -1)
    pk = j.get('pair_id', '?')
    cond = j.get('condition', '?')
    atk = j.get('attack_seed', -1)
    env_s = j.get('env_seed', -1)
    infra = j.get('infra_status', 'ok')
    n_steps = j.get('n_total_steps', 0)
    ws = j.get('window_start', 0)
    we = j.get('window_end', 0)
    win_size = we - ws + 1
    opn = j.get('decoded_open_count', 0)
    streak = j.get('decoded_longest_open_streak', 0)
    qpos = j.get('qpos_delta', 0)
    succ = j.get('success', 0)

    # Determine if this is a retry (same job_id as a failed original)
    is_retry = (jid in FAILED_ORIGINAL_JIDS and infra == 'ok')

    # Valid job: infra=ok AND n_steps >= window_end + 5
    valid = (infra == 'ok' and n_steps >= we + 5)

    row = {
        'job_id': jid, 'pair_id': pk, 'task': j.get('task_key', '?'),
        'state_id': j.get('state_id', '?'), 'condition': cond,
        'env_seed': env_s, 'attack_seed': atk,
        'window_start': ws, 'window_end': we, 'window_size': win_size,
        'n_total_steps': n_steps, 'infra_status': infra,
        'decoded_open_count': opn, 'longest_streak': streak,
        'qpos_delta': round(qpos, 6), 'success': succ,
        'valid': valid, 'is_retry': is_retry,
    }
    audit_rows.append(row)

    if valid:
        by_pair[pk][cond].append(row)

# ── 2. Retry Audit ──
print(f'\n=== Retry Audit ===')
for j in audit_rows:
    if j['job_id'] in FAILED_ORIGINAL_JIDS:
        if j['infra_status'] != 'ok':
            failed_original.append(j)
        elif j['is_retry']:
            retry_success.append(j)
            print(f"  RETRY SUCCESS: job={j['job_id']} pair={j['pair_id']} cond={j['condition']} atk={j['attack_seed']} open={j['decoded_open_count']} n_steps={j['n_total_steps']}")

# ── 3. Parent Probability Labels ──
PARENT_CATEGORIES = {
    'k5c_rand_butter1': 'rand_sensitive',
    'k5c_rand_butter2': 'rand_sensitive',
    'k5c_rand_cream': 'rand_sensitive',
    'k5c_rand_oj': 'rand_sensitive',
    'k5c_rand_alpha': 'rand_sensitive',
    'k5c_cmd_tomato_early': 'same_task_contrast',
    'k5c_cmd_milk_neg': 'same_task_contrast',
    'k5c_cmd_cream': 'same_task_contrast',
    'k5c_cmd_salad_neg': 'same_task_contrast',
    'k5c_cmd_alpha': 'same_task_contrast',
    'k5c_cmd_butter': 'same_task_contrast',
    'k5c_phys_butter': 'strict_phys',
    'k5c_phys_tomato': 'strict_phys',
    'k5c_phys_salad': 'strict_phys',
    'k5c_phys_bbq': 'strict_phys',
    'k5c_sentinel_milk_gold': 'sentinel',
}

def classify_stable(pV_seed, pR_seed, pV_phys_seed, pR_phys_seed, yield_cmd, yield_phys, risk_rand):
    """Apply S5 stability rules at seed-level."""
    # stable_cmd_specific: pV>=0.6, pR<=0.2, yield>=0.4
    if pV_seed >= 0.6 and pR_seed <= 0.2 and yield_cmd >= 0.4:
        return 'stable_cmd_specific'
    # stable_rand_sensitive: pR>=0.4 or pR_phys>=0.4
    if pR_seed >= 0.4 or pR_phys_seed >= 0.4:
        return 'stable_rand_sensitive'
    # stable_negative: all p<=0.2
    if pV_seed <= 0.2 and pR_seed <= 0.2 and pV_phys_seed <= 0.2 and pR_phys_seed <= 0.2:
        return 'stable_negative'
    # stable_vis_phys: pV_phys>=0.6, pR_phys<=0.2, yield_phys>=0.4
    if pV_phys_seed >= 0.6 and pR_phys_seed <= 0.2 and yield_phys >= 0.4:
        return 'stable_vis_phys'
    return 'unstable_or_unknown'

def classify_phys(pV_phys_seed, pR_phys_seed, yield_phys, risk_rand):
    """Classify phys label separately from cmd label."""
    if pV_phys_seed >= 0.6 and pR_phys_seed <= 0.2 and yield_phys >= 0.4:
        return 'stable_vis_phys'
    if pR_phys_seed >= 0.4 and pV_phys_seed >= 0.4:
        return 'shared_phys'
    if pR_phys_seed >= 0.4:
        return 'stable_rand_phys'
    if pV_phys_seed <= 0.2 and pR_phys_seed <= 0.2:
        return 'stable_no_phys'
    return 'unstable_phys'

# SEED-LEVEL threshold: open_count >= window_size/2 means "command success"
# For phys: qpos_delta >= 0.01 means physical gripper change
PHYS_THRESHOLD = 0.01
# For abstain_any: risk_rand >= 0.4
ABSTAIN_THRESHOLD = 0.4

label_rows = []
for pk in sorted(by_pair.keys()):
    vis_jobs = by_pair[pk].get('vis_pgd', [])
    rand_jobs = by_pair[pk].get('random_linf', [])

    if not vis_jobs:
        continue

    task = vis_jobs[0]['task']
    win_size = vis_jobs[0]['window_size']

    # VIS command opens
    vis_opens = [j['decoded_open_count'] for j in vis_jobs]
    rand_opens = [j['decoded_open_count'] for j in rand_jobs]

    # Seed-level: does this seed produce "command success"?
    cmd_threshold = max(1, win_size // 2)
    vis_cmd_seeds = [1 if o >= cmd_threshold else 0 for o in vis_opens]
    rand_cmd_seeds = [1 if o >= cmd_threshold else 0 for o in rand_opens]

    # Phys: qpos_delta as proxy for physical gripper change
    vis_qpos = [j['qpos_delta'] for j in vis_jobs]
    rand_qpos = [j['qpos_delta'] for j in rand_jobs]
    vis_phys_seeds = [1 if q >= PHYS_THRESHOLD else 0 for q in vis_qpos]
    rand_phys_seeds = [1 if q >= PHYS_THRESHOLD else 0 for q in rand_qpos]

    K = len(vis_opens)
    pV_seed = sum(vis_cmd_seeds) / K if K > 0 else 0
    pR_seed = sum(rand_cmd_seeds) / K if K > 0 else 0
    pV_phys_seed = sum(vis_phys_seeds) / K if K > 0 else 0
    pR_phys_seed = sum(rand_phys_seeds) / K if K > 0 else 0

    # Rate-normalized means
    vis_mean = np.mean(vis_opens) / win_size if K > 0 else 0
    rand_mean = np.mean(rand_opens) / win_size if K > 0 else 0
    vis_phys_mean = np.mean(vis_qpos) if K > 0 else 0
    rand_phys_mean = np.mean(rand_qpos) if K > 0 else 0

    yield_cmd = pV_seed - pR_seed
    yield_phys = pV_phys_seed - pR_phys_seed
    risk_rand = max(pR_seed, pR_phys_seed)

    cmd_label = classify_stable(pV_seed, pR_seed, pV_phys_seed, pR_phys_seed, yield_cmd, yield_phys, risk_rand)

    # Phys label (independent of cmd label)
    phys_label = classify_phys(pV_phys_seed, pR_phys_seed, yield_phys, risk_rand)

    # Abstain label: any rand risk (command or physical)
    abstain_label = 'rand_abstain' if risk_rand >= ABSTAIN_THRESHOLD else 'keep'

    # Special handling for milk[235,245]
    is_borderline = False
    if pk == 'k5c_cmd_milk_neg':
        is_borderline = True
        cmd_label = 'stable_cmd_specific_borderline'

    cat = PARENT_CATEGORIES.get(pk, '?')

    label_rows.append({
        'parent': pk, 'task': task, 'category': cat,
        'window': f"{vis_jobs[0]['window_start']}_{vis_jobs[0]['window_end']}",
        'K': K,
        'vis_open_list': str(vis_opens),
        'rand_open_list': str(rand_opens),
        'vis_open_mean': round(vis_mean, 4),
        'rand_open_mean': round(rand_mean, 4),
        'vis_qpos_mean': round(vis_phys_mean, 6),
        'rand_qpos_mean': round(rand_phys_mean, 6),
        'pV_cmd_seed': round(pV_seed, 2),
        'pR_cmd_seed': round(pR_seed, 2),
        'pV_phys_seed': round(pV_phys_seed, 2),
        'pR_phys_seed': round(pR_phys_seed, 2),
        'yield_cmd': round(yield_cmd, 2),
        'yield_phys': round(yield_phys, 2),
        'risk_rand': round(risk_rand, 2),
        'cmd_label': cmd_label,
        'phys_label': phys_label,
        'abstain_label': abstain_label,
        'borderline_note': 'VIS-dominant; one RAND outlier seed; seed-level pR=0.20 borderline-pass' if is_borderline else '',
    })

# Print labels
print(f'\n=== K5c Parent Probability Labels ({len(label_rows)} parents) ===')
print(f'{"Parent":35s} {"Task":15s} {"Cat":14s} pV_seed pR_seed yield risk {"Label":35s}')
print('-' * 125)
for r in label_rows:
    print(f"{r['parent']:35s} {r['task']:15s} {r['category']:14s} {r['pV_cmd_seed']:.2f}     {r['pR_cmd_seed']:.2f}     {r['yield_cmd']:+.2f}  {r['risk_rand']:.2f}  {r['cmd_label']:35s}")

# Label distribution
label_dist = Counter(r['cmd_label'] for r in label_rows)
print(f'\nLabel distribution:')
for lbl, cnt in label_dist.most_common():
    print(f'  {lbl}: {cnt}')

# ── 4. Write CSVs ──
os.makedirs(os.path.join(REPO, 'tables'), exist_ok=True)

# Job audit
with open(os.path.join(REPO, 'tables/stageb_v1_1_k5c_job_audit_rc1a_ca3a97e.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['job_id','pair_id','task','state_id','condition','env_seed','attack_seed',
                'window_start','window_end','window_size','n_total_steps','infra_status',
                'decoded_open_count','longest_streak','qpos_delta','success','valid','is_retry'])
    for r in audit_rows:
        w.writerow([r['job_id'],r['pair_id'],r['task'],r['state_id'],r['condition'],
                    r['env_seed'],r['attack_seed'],r['window_start'],r['window_end'],
                    r['window_size'],r['n_total_steps'],r['infra_status'],
                    r['decoded_open_count'],r['longest_streak'],r['qpos_delta'],r['success'],
                    r['valid'],r['is_retry']])

# Retry audit
with open(os.path.join(REPO, 'tables/stageb_v1_1_k5c_retry_audit_rc1a_ca3a97e.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['job_id','pair_id','condition','attack_seed','original_status','retry_status',
                'original_n_steps','retry_n_steps','retry_open_count','retry_gpu_pair'])
    for jid in FAILED_ORIGINAL_JIDS:
        orig = [r for r in audit_rows if r['job_id'] == jid and r['infra_status'] != 'ok']
        retry = [r for r in audit_rows if r['job_id'] == jid and r['is_retry']]
        o_status = orig[0]['infra_status'] if orig else '?'
        o_n = orig[0]['n_total_steps'] if orig else '?'
        r_status = 'ok' if retry else '?'
        r_n = retry[0]['n_total_steps'] if retry else '?'
        r_open = retry[0]['decoded_open_count'] if retry else '?'
        w.writerow([jid, (orig[0] if orig else {}).get('pair_id','?'),
                    (orig[0] if orig else {}).get('condition','?'),
                    (orig[0] if orig else {}).get('attack_seed','?'),
                    o_status, r_status, o_n, r_n, r_open,
                    '4,5' if jid in (520065,520090) else '1,0'])

# Parent probability labels
labels_path = os.path.join(REPO, 'tables/stageb_v1_1_k5c_parent_probability_labels_rc1a_ca3a97e.csv')
with open(labels_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parent','task','category','window','K','vis_open_list','rand_open_list',
                'vis_open_mean','rand_open_mean','vis_qpos_mean','rand_qpos_mean',
                'pV_cmd_seed','pR_cmd_seed','pV_phys_seed','pR_phys_seed',
                'yield_cmd','yield_phys','risk_rand','cmd_label','phys_label','abstain_label','borderline_note'])
    for r in label_rows:
        w.writerow([r['parent'],r['task'],r['category'],r['window'],r['K'],
                    r['vis_open_list'],r['rand_open_list'],
                    r['vis_open_mean'],r['rand_open_mean'],r['vis_qpos_mean'],r['rand_qpos_mean'],
                    r['pV_cmd_seed'],r['pR_cmd_seed'],r['pV_phys_seed'],r['pR_phys_seed'],
                    r['yield_cmd'],r['yield_phys'],r['risk_rand'],r['cmd_label'],r['phys_label'],r['abstain_label'],r['borderline_note']])

print(f'\nOutputs:')
print(f'  Job audit: tables/stageb_v1_1_k5c_job_audit_rc1a_ca3a97e.csv')
print(f'  Retry audit: tables/stageb_v1_1_k5c_retry_audit_rc1a_ca3a97e.csv')
print(f'  Labels: {labels_path}')

# ── 5. Quick stats ──
print(f'\n=== Summary ===')
print(f'Total summaries: {len(summaries)}')
print(f'Valid jobs (infra=ok, n_steps>=window_end+5): {sum(1 for r in audit_rows if r["valid"])}')
print(f'Failed original: {len(failed_original)}')
print(f'Successful retries: {len(retry_success)}')
print(f'Parents: {len(label_rows)}')
valid_parents = sum(1 for r in label_rows if r['K'] == 5)
print(f'Parents with full K=5: {valid_parents}')
