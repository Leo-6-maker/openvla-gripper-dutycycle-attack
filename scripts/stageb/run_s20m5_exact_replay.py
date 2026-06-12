#!/usr/bin/env python3
"""S20M5: Freeze partial + fix baseline lookup + build exact replay queue.
bbq_sauce seed99 REPRO_MISMATCH: S20M4b open=0 vs S20M5 open=10.
Exact replay ×3 + canary seed93/99 crosscheck. GPU 4,5 only."""
import csv, json, glob, os, subprocess, sys, hashlib, time, platform
from pathlib import Path
from datetime import datetime
from collections import defaultdict

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O5 = '/data/liuyu/outputs/stageb_s20m5_vis_diagnostics_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O5+'/queues', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# A. Freeze S20M5 partial
# ═══════════════════════════════════════════════════════════════
S20M5_RESULTS = [
    {'candidate_id': 'cream_cheese_s2_w75_85',     'vis_seed': '99',
     'vis_open': 1, 'vis_streak': 1, 's20m4b_open': 0, 'match_m4b': True,
     'diagnostic_role': 'NO_EFFECT_CONTRAST_NEARBY',
     's20m5_class': 'NO_EFFECT_REPRODUCED',
     'notes': 'Consistent with M4b NO_EFFECT. Window IS VIS-resistant.'},
    {'candidate_id': 'bbq_sauce_s0_w125_135',      'vis_seed': '99',
     'vis_open': 10, 'vis_streak': 10, 's20m4b_open': 0, 'match_m4b': False,
     'diagnostic_role': 'NO_EFFECT_CONTRAST_TRANSPORT',
     's20m5_class': 'REPRO_MISMATCH_POSITIVE',
     'notes': 'CRITICAL: S20M4b open=0 vs S20M5 open=10. Same window+seed. Run-level non-determinism. Exact replay required.'},
    {'candidate_id': 'cream_cheese_s2_w80_90',     'vis_seed': '93',
     'vis_open': '', 'vis_streak': '', 's20m4b_open': '', 'match_m4b': '',
     'diagnostic_role': 'CANARY_POSITIVE_CONTROL',
     's20m5_class': 'SKIPPED_BASELINE_LOOKUP_MISSING',
     'notes': 'RAND baseline in S20M3a/M3b not S20M4a. Worker needs cross-directory lookup.'},
    {'candidate_id': 'cream_cheese_s2_w80_90',     'vis_seed': '99',
     'vis_open': '', 'vis_streak': '', 's20m4b_open': '', 'match_m4b': '',
     'diagnostic_role': 'CANARY_SEED_CROSSCHECK',
     's20m5_class': 'SKIPPED_BASELINE_LOOKUP_MISSING',
     'notes': 'Same as above; cross-directory lookup needed.'},
]

with open(T+'/s20m5_vis_failure_diagnostic_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(S20M5_RESULTS[0].keys()))
    w.writeheader(); w.writerows(S20M5_RESULTS)

# Update registry: mark bbq_sauce as REPRO_MISMATCH
reg_path = T + '/layer3_registry.csv'
registry = {}
if os.path.exists(reg_path):
    with open(reg_path) as f:
        for r in csv.DictReader(f):
            registry[r['parent_id']] = r

for r in S20M5_RESULTS:
    pid = r['candidate_id']
    if r['s20m5_class'] == 'REPRO_MISMATCH_POSITIVE':
        if pid in registry:
            registry[pid]['status'] = 'REPRO_MISMATCH_POSITIVE'
            registry[pid]['vis_outcome'] = 'S20M4b=0 S20M5=10 MISMATCH'
            registry[pid]['notes'] = 'Same window+seed; run-level non-determinism. Exact replay pending.'
        else:
            registry[pid] = {'parent_id': pid, 'stage': 'S20M5', 'task': 'bbq_sauce',
                'status': 'REPRO_MISMATCH_POSITIVE', 'rand_stability': 'PROTOCOL_STRICT',
                'vis_outcome': 'S20M4b=0 S20M5=10 MISMATCH',
                'layer3_confirmed': False, 'layer3_class': '', 'eligible_for_vis': False,
                'notes': 'Exact replay required before any claim.'}

with open(T+'/layer3_registry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['parent_id','stage','task','status','rand_stability',
        'vis_outcome','layer3_confirmed','layer3_class','eligible_for_vis','notes'])
    w.writeheader()
    for r in sorted(registry.values(), key=lambda x: (x['task'], x['parent_id'])):
        w.writerow(r)

print('=== S20M5 FREEZE ===')
print('Registry: %d entries, bbq_sauce marked REPRO_MISMATCH_POSITIVE' % len(registry))
print('Tables: %s/s20m5_vis_failure_diagnostic_audit.csv' % T)

# ═══════════════════════════════════════════════════════════════
# C. Exact replay manifest + queue
# ═══════════════════════════════════════════════════════════════

# Baseline lookup across all stages
BASELINE_DIRS = [
    '/data/liuyu/outputs/stageb_s20m4_rand_stability_20260613',
    '/data/liuyu/outputs/stageb_s20m3b_multiseed_confirmation_20260613',
    '/data/liuyu/outputs/stageb_s20m3a_vis_fill_20260613',  # RAND ref for M3a
    '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613',
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
]

def find_baseline(candidate_id, preferred_seeds=None):
    """Find RAND baseline summaries across all stage directories."""
    results = []
    for d in BASELINE_DIRS:
        if not os.path.exists(d): continue
        for f in glob.glob(d+'/summary_*.json'):
            try:
                s = json.load(open(f))
            except: continue
            if s.get('condition') != 'random_linf': continue
            cid = '{task}_s{sid}_w{ws}_{we}'.format(
                task=s['task'], sid=s['state_id'],
                ws=s['window_start'], we=s['window_end'])
            if cid == candidate_id:
                seed = str(s.get('attack_seed',''))
                if preferred_seeds and seed not in preferred_seeds: continue
                results.append(s)
    return results

# Build exact replay jobs
REPLAY_JOBS = [
    {'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': '0',
     'window_start': '125', 'window_end': '135', 'phase': 'transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'repeat_id': 'replay_a', 'diagnostic_role': 'EXACT_REPLAY',
     'purpose': 'Resolve S20M4b(open=0) vs S20M5(open=10) mismatch — replay A'},
    {'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': '0',
     'window_start': '125', 'window_end': '135', 'phase': 'transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'repeat_id': 'replay_b', 'diagnostic_role': 'EXACT_REPLAY',
     'purpose': 'Resolve S20M4b(open=0) vs S20M5(open=10) mismatch — replay B'},
    {'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': '0',
     'window_start': '125', 'window_end': '135', 'phase': 'transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'repeat_id': 'replay_c', 'diagnostic_role': 'EXACT_REPLAY',
     'purpose': 'Resolve S20M4b(open=0) vs S20M5(open=10) mismatch — replay C'},
    {'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '80', 'window_end': '90', 'phase': 'early_transport',
     'condition': 'vis_pgd', 'attack_seed': '93', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'repeat_id': 'canary_a', 'diagnostic_role': 'CANARY_POSITIVE_CONTROL',
     'purpose': 'Reproduce S20M3a known CMD_POSITIVE (+8/+8). Baseline in S20M3a/M3b.'},
    {'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '80', 'window_end': '90', 'phase': 'early_transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'repeat_id': 'crosscheck_a', 'diagnostic_role': 'CANARY_SEED_CROSSCHECK',
     'purpose': 'Check if w80_90 effect is seed93-specific: VIS seed99 on same window.'},
]

# Find baselines for each job
for job in REPLAY_JOBS:
    cid = job['candidate_id']
    bl = find_baseline(cid)
    seeds_found = sorted(set(str(s.get('attack_seed','')) for s in bl))
    opens_found = '/'.join(str(s.get('decoded_open_count','?')) for s in bl)
    job['baseline_n'] = len(bl)
    job['baseline_seeds'] = '|'.join(seeds_found)
    job['baseline_opens'] = opens_found
    job['baseline_sources'] = '|'.join(sorted(set(
        os.path.basename(os.path.dirname(os.path.dirname(s.get('summary_path','')))) for s in bl)))
    print('  %-30s baseline: %d seeds [%s] opens=%s' % (cid, len(bl), job['baseline_seeds'], opens_found))

# Build queue CSV for the worker
jobs = []
for i, d in enumerate(REPLAY_JOBS):
    jid = 350010 + i
    jobs.append({
        'job_id': str(jid), 'candidate_id': d['candidate_id'], 'task': d['task'],
        'state_id': d['state_id'], 'window_start': d['window_start'], 'window_end': d['window_end'],
        'phase': d['phase'], 'condition': d['condition'], 'attack_seed': d['attack_seed'],
        'pgd_steps': d['pgd_steps'], 'eps_raw_pixels': d['eps_raw_pixels'],
        'random_control_seed': '', 'seed': '0',
        'repeat_id': d['repeat_id'], 'diagnostic_role': d['diagnostic_role'],
        'baseline_n': str(d['baseline_n']), 'baseline_seeds': d['baseline_seeds'],
        'tier': 'M5_REPLAY', 'track': 'S20M5_replay', 'status': 'pending',
        'output_dir': O5,
    })

qp = O5+'/queues/s20m5_replay_gpu4.csv'
with open(qp, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)

with open(T+'/s20m5_exact_replay_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['job_id','candidate_id','task','phase','window_start','window_end',
        'attack_seed','repeat_id','diagnostic_role','purpose','baseline_n','baseline_seeds','baseline_opens'])
    w.writeheader()
    for d in REPLAY_JOBS:
        w.writerow({k: d.get(k,'') for k in w.fieldnames})

print()
print('=== S20M5 EXACT REPLAY ===')
print('Jobs: %d (GPU 4,5 only)' % len(jobs))
for d in REPLAY_JOBS:
    print('  %-30s seed=%s repeat=%s role=%s baseline=%d seeds [%s]' %
          (d['candidate_id'], d['attack_seed'], d['repeat_id'], d['diagnostic_role'],
           d['baseline_n'], d['baseline_seeds']))
print()
print('Queue: %s' % qp)
print('Manifest: %s/s20m5_exact_replay_manifest.csv' % T)
print()
print('Decision rules:')
print('  bbq replay deterministic open=0 → S20M5 was anomaly, M4b correct')
print('  bbq replay deterministic open=10 → S20M4b was anomaly, VIS real')
print('  bbq replay non-deterministic → pipeline broken, STOP ALL VIS claims')
print('  canary reproduces +8/+8 → effect is real, window-specific')
print('  canary fails → possible regression in objective/preprocess')
