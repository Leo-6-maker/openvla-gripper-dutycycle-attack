#!/usr/bin/env python3
"""Layer3 registry builder: aggregate all tested parents across S20M3a/M3b/M4a.
Track status, RAND-stability, VIS outcome, confirmation status.
Part of Freeze A — execution snapshot, not claim freeze."""
import csv, json, glob, os
from collections import defaultdict

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'

# ── 1. Load existing registry entries (S20M3b confirmed) ──
m3b_path = T + '/s20m3b_layer3_registry.csv'
m3b_entries = {}
if os.path.exists(m3b_path):
    with open(m3b_path) as f:
        for r in csv.DictReader(f):
            m3b_entries[r['parent_id']] = r

# ── 2. Load S20M3a VIS-fill manifest ──
m3a_path = T + '/s20m3a_vis_fill_manifest.csv'
m3a_results = []
if os.path.exists(m3a_path):
    with open(m3a_path) as f:
        for r in csv.DictReader(f):
            m3a_results.append(r)

# ── 3. Load S20M4a RAND-stability audit ──
m4a_path = T + '/s20m4a_rand_stability_outcome_audit.csv'
m4a_results = []
if os.path.exists(m4a_path):
    with open(m4a_path) as f:
        for r in csv.DictReader(f):
            m4a_results.append(r)

# ── 4. Build unified registry ──
registry = []

# M3b entries (confirmed/not-confirmed parents)
for pid, r in m3b_entries.items():
    registry.append({
        'parent_id': pid,
        'stage': 'S20M3b',
        'task': r.get('task',''),
        'status': r.get('status',''),
        'rand_stability': r.get('rand_stability',''),
        'vis_outcome': r.get('vis_reproducibility',''),
        'layer3_confirmed': False,
        'layer3_class': '',
        'eligible_for_vis': False,
        'notes': r.get('reason',''),
    })

# M3a results (single-seed VIS, not confirmed but informative)
for r in m3a_results:
    pid = r.get('candidate_id','')
    if pid in m3b_entries:
        continue  # already covered
    rand_label = r.get('rand_label','')
    registry.append({
        'parent_id': pid,
        'stage': 'S20M3a',
        'task': r.get('task',''),
        'status': 'SINGLE_SEED_POSITIVE' if r.get('gate_ok') == 'True' else 'GATE_FAILED',
        'rand_stability': 'single-seed %s' % rand_label,
        'vis_outcome': 'Not multiseed confirmed',
        'layer3_confirmed': False,
        'layer3_class': '',
        'eligible_for_vis': False,
        'notes': 'S20M3a single-seed; needs RAND-stability + multiseed confirmation',
    })

# M4a entries (RAND-stability classified)
for r in m4a_results:
    pid = r.get('candidate_id','')
    if pid in m3b_entries:
        continue
    stability = r.get('stability_class','')
    vis_eligible = r.get('vis_eligible','False') == 'True'

    if stability == 'PROTOCOL_STRICT':
        status = 'RAND_STABLE_STRICT'
    elif stability == 'PROTOCOL_USABLE':
        status = 'RAND_STABLE_USABLE'
    elif stability == 'UNSTABLE':
        status = 'RAND_UNSTABLE'
    else:
        status = 'INCOMPLETE'

    registry.append({
        'parent_id': pid,
        'stage': 'S20M4a',
        'task': r.get('task',''),
        'status': status,
        'rand_stability': stability,
        'vis_outcome': 'Not yet tested',
        'layer3_confirmed': False,
        'layer3_class': '',
        'eligible_for_vis': vis_eligible,
        'notes': 'RAND-stability screen; VIS not yet run',
    })

# ── 5. Write registry ──
with open(T+'/layer3_registry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'parent_id','stage','task','status','rand_stability',
        'vis_outcome','layer3_confirmed','layer3_class','eligible_for_vis','notes'])
    w.writeheader()
    for r in sorted(registry, key=lambda x: (x['task'], x['parent_id'])):
        w.writerow(r)

# ── 6. Summary ──
confirmed = [r for r in registry if r['layer3_confirmed']]
eligible = [r for r in registry if r['eligible_for_vis']]
stable_strict = [r for r in registry if r['status'] == 'RAND_STABLE_STRICT']
print('Layer3 registry: %d entries' % len(registry))
print('  Confirmed: %d' % len(confirmed))
print('  RAND_STABLE_STRICT (VIS-eligible): %d' % len(stable_strict))
print('  All VIS-eligible: %d' % len(eligible))
print('  Not confirmed: %d' % (len(registry) - len(confirmed)))
print()
print('Written: %s/layer3_registry.csv' % T)

# ── 7. Claim boundary audit ──
with open(T+'/layer3_claim_boundary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['claim','status','evidence','constraint'])
    w.writeheader()
    w.writerows([
        {'claim': 'Layer1 v0.3.2 fresh-task screening improves denominator',
         'status': 'ALLOWED', 'evidence': 'S20M2 STRONG PASS 69.6%',
         'constraint': 'Task-dependent; not task-agnostic general detector'},
        {'claim': 'Layer2 RAND-veto can identify RAND-clean windows',
         'status': 'ALLOWED', 'evidence': 'S20M2/M4a multi-seed RAND audit',
         'constraint': 'Single-seed RAND-clean ≠ RAND-stable across seeds'},
        {'claim': 'Layer3 VIS can cause gripper opening beyond RAND baseline',
         'status': 'ALLOWED', 'evidence': 'S20M3a CMD_POSITIVE cream_cheese +8/+8',
         'constraint': 'Single-seed only; multiseed confirmation failed (1/3)'},
        {'claim': 'Layer3 VIS causes task degradation on fresh tasks',
         'status': 'ALLOWED', 'evidence': 'S20M3a TASK_EFFECT salad_dressing timeout',
         'constraint': 'Single-seed only; RAND baseline unstable in 2/3 seeds'},
        {'claim': 'Detector-selected multiseed confirmed VIS attack',
         'status': 'NOT_ALLOWED', 'evidence': 'S20M3b 0/2 confirmed',
         'constraint': 'No parent has >=2/3 seed reproducibility yet'},
        {'claim': 'VIS attack effect is task-agnostic',
         'status': 'NOT_ALLOWED', 'evidence': 'Task-dependent RAND sensitivity confirmed',
         'constraint': 'Layer1 itself is task-dependent'},
        {'claim': 'RAND-veto can be skipped',
         'status': 'NOT_ALLOWED', 'evidence': 'S20M3b salad 2/3 RAND BORDERLINE',
         'constraint': 'RAND-veto is required; single-seed RAND-clean not sufficient'},
        {'claim': 'random_sensitive = negative',
         'status': 'NOT_ALLOWED', 'evidence': 'random_sensitive is abstain/veto class',
         'constraint': 'Do not label RS as negative for attack success'},
    ])
print('Written: %s/layer3_claim_boundary.csv' % T)
