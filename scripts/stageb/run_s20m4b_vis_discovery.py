#!/usr/bin/env python3
"""S20M4b VIS-fill discovery: 6 PROTOCOL_STRICT parents, seed99 only, GPU 4,5.
Discovery only — NOT confirmation. No seed100/101 auto-expansion."""
import csv, json, os

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O = '/data/liuyu/outputs/stageb_s20m4b_vis_discovery_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O+'/queues', exist_ok=True)

PARENTS = [
    {'candidate_id': 'cream_cheese_s2_w70_80',    'task': 'cream_cheese',    'state_id': '2',
     'window_start': '70', 'window_end': '80', 'phase': 'grasp_transition',
     'rand_opens': '0/0/0', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P0'},
    {'candidate_id': 'cream_cheese_s2_w75_85',    'task': 'cream_cheese',    'state_id': '2',
     'window_start': '75', 'window_end': '85', 'phase': 'grasp_transition',
     'rand_opens': '1/0/1', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P0'},
    {'candidate_id': 'chocolate_pudding_s2_w65_75','task': 'chocolate_pudding','state_id': '2',
     'window_start': '65', 'window_end': '75', 'phase': 'early_transport',
     'rand_opens': '1/0/0', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P0'},
    {'candidate_id': 'bbq_sauce_s0_w125_135',     'task': 'bbq_sauce',       'state_id': '0',
     'window_start': '125', 'window_end': '135', 'phase': 'transport',
     'rand_opens': '0/2/0', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P0'},
    {'candidate_id': 'butter_s2_w105_115',        'task': 'butter',          'state_id': '2',
     'window_start': '105', 'window_end': '115', 'phase': 'transport',
     'rand_opens': '0/0/3', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P0'},
    {'candidate_id': 'orange_juice_s1_w120_130',  'task': 'orange_juice',    'state_id': '1',
     'window_start': '120', 'window_end': '130', 'phase': 'place_or_done',
     'rand_opens': '0/0/0', 'rand_stability_tier': 'PROTOCOL_STRICT',
     'claim_priority': 'P2_DIAGNOSTIC'},
]

# Build VIS jobs
jobs = []; jid = 340000
for p in PARENTS:
    jid += 1
    jobs.append({
        'job_id': str(jid),
        'candidate_id': p['candidate_id'],
        'task': p['task'],
        'state_id': p['state_id'],
        'window_start': p['window_start'],
        'window_end': p['window_end'],
        'phase': p['phase'],
        'condition': 'vis_pgd',
        'attack_seed': '99',
        'pgd_steps': '20',
        'eps_raw_pixels': '6',
        'random_control_seed': '',
        'seed': '0',
        'matched_rand_seeds': '96|97|98',
        'rand_stability_tier': p['rand_stability_tier'],
        'rand_open_values': p['rand_opens'],
        'claim_priority': p['claim_priority'],
        'tier': 'M4b_DISCOVERY',
        'track': 'S20M4b',
        'status': 'pending',
        'output_dir': O,
    })

# Single GPU queue (4,5)
qp = O+'/queues/s20m4b_vis_gpu4.csv'
with open(qp, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)

# Write manifest
with open(T+'/s20m4b_vis_discovery_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'candidate_id','task','phase','window_start','window_end',
        'rand_stability_tier','rand_opens_96_97_98','claim_priority',
        'vis_seed','pgd_steps','eps_raw_pixels','gpu_pair'])
    w.writeheader()
    for p in PARENTS:
        w.writerow({
            'candidate_id': p['candidate_id'], 'task': p['task'],
            'phase': p['phase'], 'window_start': p['window_start'],
            'window_end': p['window_end'],
            'rand_stability_tier': p['rand_stability_tier'],
            'rand_opens_96_97_98': p['rand_opens'],
            'claim_priority': p['claim_priority'],
            'vis_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
            'gpu_pair': '4,5',
        })

print('S20M4b VIS discovery: %d jobs (seed99 only)' % len(jobs))
for p in PARENTS:
    print('  %-35s phase=%-16s rand=%s priority=%s' %
          (p['candidate_id'], p['phase'], p['rand_opens'], p['claim_priority']))
print()
print('Queue: %s' % qp)
print('Manifest: %s/s20m4b_vis_discovery_manifest.csv' % T)
print()
print('Classification rules:')
print('  DISCOVERY_CMD_POSITIVE: VIS_open - RAND_median_open >= 3')
print('  NO_EFFECT: no meaningful gap')
print('  CONFOUNDED_OR_INFRA: timeout/infra issues')
print()
print('NO seed100/101 auto-expansion.')
print('Only DISCOVERY_CMD_POSITIVE parents get confirmation seeds.')
