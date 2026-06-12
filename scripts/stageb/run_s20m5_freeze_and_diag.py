#!/usr/bin/env python3
"""S20M4b freeze + S20M5 VIS failure diagnostic builder.
A. Update S20M4b audit + Layer3 registry + freeze report
C. Build S20M5 canary + contrast diagnostic queue (GPU 4,5 only, 4 jobs max)"""
import csv, json, os

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O5 = '/data/liuyu/outputs/stageb_s20m5_vis_diagnostics_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O5+'/queues', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# A. S20M4b freeze
# ═══════════════════════════════════════════════════════════════
M4B_RESULTS = [
    {'candidate_id': 'cream_cheese_s2_w70_80',       'task': 'cream_cheese',     'phase': 'grasp_transition',
     'rand_opens': '0/0/0', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': 0, 'streak_gap': 0,
     'discovery_class': 'NO_EFFECT', 'claim_priority': 'P0',
     'registry_status': 'NO_EFFECT', 'notes': 'RAND-stable but VIS produces zero openings'},
    {'candidate_id': 'cream_cheese_s2_w75_85',       'task': 'cream_cheese',     'phase': 'grasp_transition',
     'rand_opens': '1/0/1', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': -1, 'streak_gap': -1,
     'discovery_class': 'NO_EFFECT', 'claim_priority': 'P0',
     'registry_status': 'NO_EFFECT', 'notes': 'Adjacent to M3a positive w80_90; VIS zero effect'},
    {'candidate_id': 'chocolate_pudding_s2_w65_75',  'task': 'chocolate_pudding','phase': 'early_transport',
     'rand_opens': '1/0/0', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': 0, 'streak_gap': 0,
     'discovery_class': 'NO_EFFECT', 'claim_priority': 'P0',
     'registry_status': 'NO_EFFECT', 'notes': 'RAND-stable; VIS zero openings'},
    {'candidate_id': 'bbq_sauce_s0_w125_135',        'task': 'bbq_sauce',        'phase': 'transport',
     'rand_opens': '0/2/0', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': 0, 'streak_gap': 0,
     'discovery_class': 'NO_EFFECT', 'claim_priority': 'P0',
     'registry_status': 'NO_EFFECT', 'notes': 'RAND-stable transport; VIS zero openings'},
    {'candidate_id': 'butter_s2_w105_115',           'task': 'butter',           'phase': 'transport',
     'rand_opens': '0/0/3', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': 0, 'streak_gap': 0,
     'discovery_class': 'NO_EFFECT', 'claim_priority': 'P0',
     'registry_status': 'NO_EFFECT', 'notes': 'RAND-stable; VIS zero openings'},
    {'candidate_id': 'orange_juice_s1_w120_130',     'task': 'orange_juice',     'phase': 'place_or_done',
     'rand_opens': '0/0/0', 'vis_seed': '99', 'vis_open': 0, 'vis_streak': 0, 'open_gap': 0, 'streak_gap': 0,
     'discovery_class': 'DIAGNOSTIC_ONLY_NO_EFFECT', 'claim_priority': 'P2_DIAGNOSTIC',
     'registry_status': 'DIAGNOSTIC_ONLY_NO_EFFECT', 'notes': 'place_or_done phase; low interpretability'},
]

# Write S20M4b audit
with open(T+'/s20m4b_vis_discovery_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(M4B_RESULTS[0].keys()))
    w.writeheader(); w.writerows(M4B_RESULTS)

# ═══════════════════════════════════════════════════════════════
# Update Layer3 registry — merge M4b results into existing
# ═══════════════════════════════════════════════════════════════
reg_path = T + '/layer3_registry.csv'
existing_reg = {}
if os.path.exists(reg_path):
    with open(reg_path) as f:
        for r in csv.DictReader(f):
            existing_reg[r['parent_id']] = r

for r in M4B_RESULTS:
    pid = r['candidate_id']
    if pid in existing_reg:
        existing_reg[pid]['status'] = r['registry_status']
        existing_reg[pid]['vis_outcome'] = r['discovery_class']
        existing_reg[pid]['notes'] = r['notes']
    else:
        existing_reg[pid] = {
            'parent_id': pid, 'stage': 'S20M4b', 'task': r['task'],
            'status': r['registry_status'], 'rand_stability': 'PROTOCOL_STRICT',
            'vis_outcome': r['discovery_class'], 'layer3_confirmed': False,
            'layer3_class': '', 'eligible_for_vis': False,
            'notes': r['notes']}

with open(T+'/layer3_registry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['parent_id','stage','task','status','rand_stability',
        'vis_outcome','layer3_confirmed','layer3_class','eligible_for_vis','notes'])
    w.writeheader()
    for r in sorted(existing_reg.values(), key=lambda x: (x['task'], x['parent_id'])):
        w.writerow(r)

# Registry summary
total = len(existing_reg)
no_effect = sum(1 for r in existing_reg.values() if 'NO_EFFECT' in str(r.get('status','')))
confirmed = sum(1 for r in existing_reg.values() if r.get('layer3_confirmed') == 'True')
print('=== S20M4b FREEZE ===')
print('Registry: %d entries, %d NO_EFFECT, %d confirmed' % (total, no_effect, confirmed))
print('Tables: %s/s20m4b_vis_discovery_audit.csv' % T)
print('         %s/layer3_registry.csv (updated)' % T)

# ═══════════════════════════════════════════════════════════════
# C. S20M5 VIS failure diagnostic queue
# ═══════════════════════════════════════════════════════════════
DIAGNOSTIC_JOBS = [
    # Positive-control canary: known single-seed positive from S20M3a
    {'job_id': '350001', 'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '80', 'window_end': '90', 'phase': 'early_transport',
     'condition': 'vis_pgd', 'attack_seed': '93', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'diagnostic_role': 'CANARY_POSITIVE_CONTROL',
     'purpose': 'Reproduce S20M3a known single-seed CMD_POSITIVE (+8/+8)',
     'rand_opens_m3a': '0', 'rand_label_m3a': 'RAND_STRICT',
     'expected_if_healthy': 'VIS open >= 6, streak >= 4',
     'claim_priority': 'DIAGNOSTIC_CANARY'},
    # No-effect contrast: adjacent window to canary
    {'job_id': '350002', 'candidate_id': 'cream_cheese_s2_w75_85', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '75', 'window_end': '85', 'phase': 'grasp_transition',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'diagnostic_role': 'NO_EFFECT_CONTRAST_NEARBY',
     'purpose': 'Contrast: nearby window to canary, seed99 NO_EFFECT in M4b',
     'rand_opens_m4a': '1/0/1', 'rand_label_m4a': 'PROTOCOL_STRICT',
     'expected_if_healthy': 'VIS open = 0 (reproduce M4b NO_EFFECT)',
     'claim_priority': 'DIAGNOSTIC_CONTRAST'},
    # No-effect contrast: different task
    {'job_id': '350003', 'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': '0',
     'window_start': '125', 'window_end': '135', 'phase': 'transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'diagnostic_role': 'NO_EFFECT_CONTRAST_TRANSPORT',
     'purpose': 'Contrast: different task, transport phase, NO_EFFECT in M4b',
     'rand_opens_m4a': '0/2/0', 'rand_label_m4a': 'PROTOCOL_STRICT',
     'expected_if_healthy': 'VIS open = 0 (reproduce M4b NO_EFFECT)',
     'claim_priority': 'DIAGNOSTIC_CONTRAST'},
    # Canary re-run with seed99 on w80_90 to check seed sensitivity
    {'job_id': '350004', 'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '80', 'window_end': '90', 'phase': 'early_transport',
     'condition': 'vis_pgd', 'attack_seed': '99', 'pgd_steps': '20', 'eps_raw_pixels': '6',
     'diagnostic_role': 'CANARY_SEED_CROSSCHECK',
     'purpose': 'Check if w80_90 effect is seed93-specific: VIS seed99 on same window',
     'rand_opens_m4a': 'not in M4a (was M3a seed92)',
     'expected_if_healthy': 'If positive → effect is window-dependent not seed-dependent. If NO_EFFECT → seed93 is unique.',
     'claim_priority': 'DIAGNOSTIC_CANARY'},
]

# Build VIS jobs
jobs = []
for d in DIAGNOSTIC_JOBS:
    jobs.append({
        'job_id': d['job_id'], 'candidate_id': d['candidate_id'], 'task': d['task'],
        'state_id': d['state_id'], 'window_start': d['window_start'], 'window_end': d['window_end'],
        'phase': d['phase'], 'condition': d['condition'], 'attack_seed': d['attack_seed'],
        'pgd_steps': d['pgd_steps'], 'eps_raw_pixels': d['eps_raw_pixels'],
        'random_control_seed': '', 'seed': '0',
        'diagnostic_role': d['diagnostic_role'], 'purpose': d['purpose'],
        'tier': 'M5_DIAGNOSTIC', 'track': 'S20M5', 'status': 'pending',
        'output_dir': O5,
    })

qp = O5+'/queues/s20m5_diag_gpu4.csv'
with open(qp, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)

# Write manifest
with open(T+'/s20m5_vis_failure_diagnostic_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['job_id','candidate_id','task','phase','window_start','window_end',
        'attack_seed','diagnostic_role','purpose','expected_if_healthy'])
    w.writeheader()
    for d in DIAGNOSTIC_JOBS:
        w.writerow({k: d[k] for k in w.fieldnames})

print()
print('=== S20M5 DIAGNOSTIC QUEUE ===')
print('Jobs: %d (GPU 4,5 only)' % len(jobs))
for d in DIAGNOSTIC_JOBS:
    print('  %s %-30s seed=%s role=%s' % (d['job_id'], d['candidate_id'], d['attack_seed'], d['diagnostic_role']))
print()
print('Queue: %s' % qp)
print('Manifest: %s/s20m5_vis_failure_diagnostic_manifest.csv' % T)
print()
print('Decision gates:')
print('  CANARY seed93 reproduces → effect is real, window-specific')
print('  CANARY seed93 fails → possible regression, STOP all VIS claims')
print('  w80_90 seed99 crosses → effect is window-dependent not seed-dependent')
print('  w80_90 seed99 NO_EFFECT → seed93 is isolated event')
print('  Contrasts remain NO_EFFECT → as expected, confirms M4b')
