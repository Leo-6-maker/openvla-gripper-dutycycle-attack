#!/usr/bin/env python3
"""Build V6 report tables from raw summaries."""
import json, csv, os, glob
from collections import defaultdict

OUT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(OUT, exist_ok=True)
P1 = '/data/liuyu/outputs/stageb_v6_online_trigger_pilot_20260613/phase1_v2'
P2 = '/data/liuyu/outputs/stageb_v6_rand_veto'
P3 = '/data/liuyu/outputs/stageb_v6_vis_pilot'

# ── TABLE 1: Phase 1 Clean ──
rows1 = []
for sf in sorted(glob.glob(P1 + '/summary_*.json')):
    s = json.load(open(sf))
    rows1.append({
        'parent_id': '%s_s%s' % (s['task'], s['state_id']),
        'task': s['task'], 'state_id': s['state_id'],
        'repeat': s.get('job_id', '')[-2:],
        'trigger_found': s['trigger_found'],
        'trigger_step': s['trigger_step'],
        'n_steps': s['n_steps'],
        'success_primary': s['success_primary'],
        'timeout': s.get('timeout', False),
        'infra_status': s['infra_status'],
    })
with open('%s/s20d_v6_online_clean_trigger_complete.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows1[0].keys()))
    w.writeheader(); w.writerows(rows1)
print('T1 clean: %d rows' % len(rows1))

# ── TABLE 2: Phase 2 RAND ──
rows2 = []
p2_data = defaultdict(lambda: {'trigger': 0, 'c2o_ep': 0, 'c2o_cnt': 0, 'att': 0, 'succ': 0, 'n': 0})
for sf in sorted(glob.glob(P2 + '/summary_*.json')):
    s = json.load(open(sf))
    pid = '%s_s%s' % (s['task'], s['state_id'])
    c2o_ep = 1 if s['C2O_count'] > 0 else 0
    att = max(s['attacked_close_count'], 1)
    rows2.append({
        'parent_id': pid, 'seed': s['attack_seed'],
        'trigger_found': s['trigger_found'], 'trigger_step': s['trigger_step'],
        'attacked_close_count': s['attacked_close_count'],
        'C2O_count': s['C2O_count'], 'C2O_episode': c2o_ep,
        'event_C2O_rate': round(s['C2O_count'] / att, 3),
        'success_primary': s['success_primary'],
        'timeout': s.get('timeout', False), 'infra_status': s['infra_status'],
    })
    p2_data[pid]['trigger'] += s['trigger_found']
    p2_data[pid]['c2o_ep'] += c2o_ep
    p2_data[pid]['c2o_cnt'] += s['C2O_count']
    p2_data[pid]['att'] += att
    p2_data[pid]['succ'] += s['success_primary']
    p2_data[pid]['n'] += 1
with open('%s/s20d_v6_online_rand_veto_complete.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
    w.writeheader(); w.writerows(rows2)
print('T2 RAND: %d rows' % len(rows2))

# ── TABLE 3: RAND Classification ──
clean_succ = defaultdict(int)
for r in rows1:
    pid = r['parent_id']
    if r['success_primary']: clean_succ[pid] += 1

rows3 = []
for pid, d in sorted(p2_data.items()):
    task = pid.split('_s')[0]
    if d['c2o_ep'] >= 2:
        cls = 'ONLINE_RANDOM_SENSITIVE_ABSTAIN'; reason = 'C2O episodes >= 2/3'
    elif d['trigger'] < 2:
        cls = 'ONLINE_TRIGGER_UNSTABLE'; reason = 'trigger < 2/3'
    elif d['c2o_ep'] <= 1:
        cls = 'ONLINE_RAND_STRICT'; reason = 'C2O <= 1/3, trigger 3/3, infra valid'
    else:
        cls = 'ONLINE_RAND_USABLE'; reason = 'C2O <= 1/3, trigger >= 2/3'
    rows3.append({
        'parent_id': pid, 'task': task,
        'trigger': '%d/%d' % (d['trigger'], d['n']),
        'C2O_episodes': '%d/%d' % (d['c2o_ep'], d['n']),
        'median_event_C2O_rate': round(d['c2o_cnt'] / max(d['att'], 1), 3),
        'success': '%d/%d' % (d['succ'], d['n']),
        'clean_success': '%d/2' % clean_succ.get(pid, 0),
        'classification': cls, 'reason': reason,
    })
with open('%s/s20d_v6_online_rand_parent_classification.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
    w.writeheader(); w.writerows(rows3)
print('T3 RAND class: %d rows' % len(rows3))

# ── TABLE 4: Phase 3 VIS ──
rows4 = []
p3_data = defaultdict(lambda: {'c2o_ep': 0, 'n': 0})
for sf in sorted(glob.glob(P3 + '/summary_*.json')):
    s = json.load(open(sf))
    pid = '%s_s%s' % (s['task'], s['state_id'])
    c2o_ep = 1 if s['C2O_count'] > 0 else 0
    att = max(s['attacked_close_count'], 1)
    rows4.append({
        'parent_id': pid, 'seed': s['attack_seed'],
        'trigger_found': s['trigger_found'], 'trigger_step': s['trigger_step'],
        'attacked_close_count': att,
        'C2O_count': s['C2O_count'], 'C2O_episode': c2o_ep,
        'event_C2O_rate': round(s['C2O_count'] / att, 3),
        'success_primary': s['success_primary'],
        'timeout': s.get('timeout', False), 'infra_status': s['infra_status'],
    })
    p3_data[pid]['c2o_ep'] += c2o_ep; p3_data[pid]['n'] += 1
with open('%s/s20d_v6_online_vis_pilot_complete.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows4[0].keys()))
    w.writeheader(); w.writerows(rows4)
print('T4 VIS: %d rows' % len(rows4))

# ── TABLE 5: VIS Parent Comparison ──
rows5 = []
for pid in ['butter_s2', 'bbq_sauce_s0', 'chocolate_pudding_s2']:
    task = pid.split('_s')[0]
    vd = p3_data.get(pid, {'c2o_ep': 0, 'n': 0})
    rd = p2_data.get(pid, {'c2o_ep': 0, 'n': 0})
    if vd['c2o_ep'] >= 2 and rd['c2o_ep'] <= 1: cls = 'ONLINE_CMD_CANDIDATE'
    elif vd['c2o_ep'] == 1: cls = 'ONLINE_VIS_PARTIAL'
    else: cls = 'ONLINE_VIS_NO_EFFECT'
    rows5.append({
        'parent_id': pid, 'task': task,
        'VIS_C2O': '%d/%d' % (vd['c2o_ep'], vd['n']),
        'RAND_C2O': '%d/%d' % (rd['c2o_ep'], rd['n']),
        'physical_bridge': 'NOT_ESTABLISHED',
        'task_effect': 'NOT_ESTABLISHED',
        'classification': cls,
    })
with open('%s/s20d_v6_online_vis_parent_comparison.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows5[0].keys()))
    w.writeheader(); w.writerows(rows5)
print('T5 VIS comparison: %d rows' % len(rows5))

# ── TABLE 6: butter_s2 evidence ──
rows6 = []
for r in rows1:
    if r['parent_id'] == 'butter_s2':
        rows6.append({'condition': 'clean_observer', 'seed': 'N/A', 'trigger_step': r['trigger_step'],
            'C2O': 0, 'success': r['success_primary'], 'infra': r['infra_status']})
for r in rows2:
    if r['parent_id'] == 'butter_s2':
        rows6.append({'condition': 'online_random_linf', 'seed': r['seed'], 'trigger_step': r['trigger_step'],
            'C2O': r['C2O_count'], 'success': r['success_primary'], 'infra': r['infra_status']})
for r in rows4:
    if r['parent_id'] == 'butter_s2':
        rows6.append({'condition': 'online_vis_pgd', 'seed': r['seed'], 'trigger_step': r['trigger_step'],
            'C2O': r['C2O_count'], 'success': r['success_primary'], 'infra': r['infra_status']})
with open('%s/s20d_v6_butter_s2_command_candidate_evidence.csv' % OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows6[0].keys()))
    w.writeheader(); w.writerows(rows6)
print('T6 butter evidence: %d rows' % len(rows6))

# ── TABLE 7: Registry update ──
reg_path = '%s/layer3_parent_registry.csv' % OUT
existing = list(csv.DictReader(open(reg_path))) if os.path.exists(reg_path) else []
v6_new = [
    {'parent_id': 'butter_s2', 'stage': 'V6_online_L3', 'task': 'butter',
     'status': 'ONLINE_CMD_CANDIDATE', 'rand_stability': 'ONLINE_RAND_STRICT',
     'vis_outcome': 'VIS_C2O_3/3', 'layer3_confirmed': 'False',
     'layer3_class': 'ONLINE_CMD_CANDIDATE', 'eligible_for_vis': 'True',
     'notes': 'V6 online trigger. VIS 3/3 C2O, RAND 0/3.'},
    {'parent_id': 'bbq_sauce_s0', 'stage': 'V6_online_L3', 'task': 'bbq_sauce',
     'status': 'ONLINE_VIS_PARTIAL', 'rand_stability': 'ONLINE_RAND_STRICT',
     'vis_outcome': 'VIS_C2O_1/3', 'layer3_confirmed': 'False',
     'layer3_class': 'ONLINE_VIS_PARTIAL', 'eligible_for_vis': 'True',
     'notes': 'V6 online trigger. VIS 1/3 C2O.'},
    {'parent_id': 'chocolate_pudding_s2', 'stage': 'V6_online_L3', 'task': 'chocolate_pudding',
     'status': 'ONLINE_VIS_NO_EFFECT', 'rand_stability': 'ONLINE_RAND_STRICT',
     'vis_outcome': 'VIS_C2O_0/3', 'layer3_confirmed': 'False',
     'layer3_class': 'ONLINE_VIS_NO_EFFECT', 'eligible_for_vis': 'True',
     'notes': 'V6 online trigger. VIS 0/3 C2O.'},
]
fnames = list(existing[0].keys()) if existing else list(v6_new[0].keys())
for e in v6_new: existing.append(e)
with open(reg_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
    w.writeheader(); w.writerows(existing)
print('T7 Registry: %d entries' % len(existing))
print('\nAll tables in %s' % OUT)
