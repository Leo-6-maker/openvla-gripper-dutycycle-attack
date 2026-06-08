#!/usr/bin/env python3
"""Freeze 72-pair label pool with RC1a provenance and multi-head taxonomy.

Combines:
- Master labels (45 pairs, d4a3827)
- Smoke pair labels (6 pairs)
- Expansion pair labels (21 pairs)

Output: tables/stageb_v1_1_all_pair_labels_rc1a_14cfabe_72pairs.csv

Provenance:
  data_anchor = d4a3827
  code_commit = 14cfabe
  branch = exp/vis-prefix-margin-repair-20260603
"""
import csv, json, os, sys
from collections import Counter

# ── Config ──
MASTER = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/master_labels_aggregated_d4a3827.csv'
SMOKE_LABELS = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827/smoke_pair_labels.csv'
EXP_LABELS = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827/expansion_pair_labels.csv'
CANDIDATES = '/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'

CMD_T = 6
PHYS_T = 0.01
PROVENANCE = {
    'data_anchor': 'd4a3827',
    'code_commit': '14cfabe',
    'branch': 'exp/vis-prefix-margin-repair-20260603',
    'trace_version': 'corrected_stageb_v1_1',
    'source_snapshot_id': 'f9840cb1',
    'prompt_style': 'official_in_out',
    'image_preprocess_style': 'official_rot180_only',
}

# ── Load candidates for clean features ──
cand_lookup = {}
with open(CANDIDATES, 'r') as f:
    for c in csv.DictReader(f):
        key = (c['task_key'], c['state_id'], c.get('seed', '0'),
               c['window_start'], c['window_end'])
        cand_lookup[key] = c

# ── Load master labels ──
master_rows = []
with open(MASTER, 'r') as f:
    for r in csv.DictReader(f):
        master_rows.append(r)

# ── Load smoke pair labels ──
smoke_rows = []
with open(SMOKE_LABELS, 'r') as f:
    for r in csv.DictReader(f):
        r['source_batch'] = 'smoke'
        smoke_rows.append(r)

# ── Load expansion pair labels ──
exp_rows = []
with open(EXP_LABELS, 'r') as f:
    for r in csv.DictReader(f):
        r['source_batch'] = 'expansion'
        exp_rows.append(r)


# ── Taxonomy classifier ──
def classify(vis_open, vis_streak, rand_open, rand_streak, vis_qpos, rand_qpos,
             n_total_steps, label_tier=''):
    """Multi-head label taxonomy."""
    # Edge/unstable
    if n_total_steps <= 50:
        return _edge_row('unstable_or_edge', 'episode too short (%d steps)' % n_total_steps)

    vo = int(vis_open or 0); vs = int(vis_streak or 0)
    ro = int(rand_open or 0); rs = int(rand_streak or 0)
    vq = abs(float(vis_qpos or 0)); rq = abs(float(rand_qpos or 0))

    vis_cmd = (vo >= CMD_T or vs >= CMD_T)
    rand_cmd = (ro >= CMD_T or rs >= CMD_T)
    vis_phys = vq >= PHYS_T
    rand_phys = rq >= PHYS_T

    # Command head
    if vis_cmd and not rand_cmd:
        cmd_specific = 1; rand_cmd_sensitive = 0; confounded_cmd = 0
    elif rand_cmd and not vis_cmd:
        cmd_specific = 0; rand_cmd_sensitive = 1; confounded_cmd = 0
    elif vis_cmd and rand_cmd:
        cmd_specific = 0; rand_cmd_sensitive = 0; confounded_cmd = 1
    else:
        cmd_specific = 0; rand_cmd_sensitive = 0; confounded_cmd = 0

    # Physical head
    if vis_phys and not rand_phys:
        vis_specific_phys = 1; shared_qpos = 0; rand_phys_confound = 0
    elif rand_phys and not vis_phys:
        vis_specific_phys = 0; shared_qpos = 0; rand_phys_confound = 1
    elif vis_phys and rand_phys:
        vis_specific_phys = 0; shared_qpos = 1; rand_phys_confound = 0
    else:
        vis_specific_phys = 0; shared_qpos = 0; rand_phys_confound = 0

    # Derived
    negative_clean = 1 if (cmd_specific == 0 and rand_cmd_sensitive == 0
                           and confounded_cmd == 0 and vis_specific_phys == 0
                           and shared_qpos == 0 and rand_phys_confound == 0) else 0
    abstain_any = 1 if (rand_cmd_sensitive or confounded_cmd
                        or rand_phys_confound) else 0

    return {
        'cmd_specific': str(cmd_specific),
        'rand_command_sensitive': str(rand_cmd_sensitive),
        'random_command_confounded': str(confounded_cmd),
        'vis_specific_phys': str(vis_specific_phys),
        'shared_qpos_response': str(shared_qpos),
        'rand_phys_confound': str(rand_phys_confound),
        'negative_clean': str(negative_clean),
        'unstable_or_edge': '0',
        'abstain_any': str(abstain_any),
        '_vo': vo, '_ro': ro, '_vq': vq, '_rq': rq,
    }


def _edge_row(unstable_reason, unstable_detail):
    r = {
        'cmd_specific': '0', 'rand_command_sensitive': '0',
        'random_command_confounded': '0', 'vis_specific_phys': '0',
        'shared_qpos_response': '0', 'rand_phys_confound': '0',
        'negative_clean': '0', 'unstable_or_edge': '1',
        'abstain_any': '1',
    }
    return r


# ── Build 72-pair rows ──
all_rows = []

# 1. Master labels (re-classify using pair labels where available)
# Master labels don't have vis/rand open counts — they have pre-computed targets.
# We use the master's existing target_* columns.
for r in master_rows:
    row = {
        'pair_id': r['pair_id'],
        'source_batch': 'master',
        'task_key': r['task_key'],
        'state_id': r['state_id'],
        'seed': r['seed'],
        'window_start': r['window_start'],
        'window_end': r['window_end'],
    }
    # Master labels use cmd_susceptible which already excludes rand_confounded
    # target_cmd_any = cmd_specific in master (see report note)
    row['cmd_specific'] = r.get('cmd_specific', r.get('target_cmd_any', '0'))
    row['rand_command_sensitive'] = '0'  # Not in original master taxonomy
    row['random_command_confounded'] = '0'
    row['vis_specific_phys'] = r.get('vis_specific_physical', r.get('target_phys', '0'))
    row['shared_qpos_response'] = '0'  # N/A in original taxonomy
    row['rand_phys_confound'] = '0'    # N/A
    row['negative_clean'] = '1' if (
        row['cmd_specific'] == '0' and row['vis_specific_phys'] == '0'
        and r.get('random_sensitive', r.get('target_rand', '0')) == '0'
    ) else '0'
    # Master 'random_sensitive' ≈ our rand_command_sensitive OR rand_phys_confound
    # We approximate: if target_rand==1 and cmd_specific==0 → rand_something
    target_rand = r.get('random_sensitive', r.get('target_rand', '0'))
    if target_rand == '1' and row['cmd_specific'] == '0':
        row['rand_command_sensitive'] = '1'  # best guess (master doesn't distinguish)
        row['negative_clean'] = '0'
    row['unstable_or_edge'] = '1' if r.get('label_tier', '') == 'rescue_unstable' else '0'
    row['abstain_any'] = '1' if (
        row['rand_command_sensitive'] == '1' or row['random_command_confounded'] == '1'
        or row['rand_phys_confound'] == '1' or row['unstable_or_edge'] == '1'
    ) else '0'
    row['label_tier'] = r.get('label_tier', '?')
    # Clean features
    key = (r['task_key'], r['state_id'], r['seed'], r['window_start'], r['window_end'])
    c = cand_lookup.get(key, {})
    row['clean_open_count'] = r.get('clean_open_count', c.get('clean_open_count', ''))
    row['clean_open_frac'] = r.get('clean_open_frac', c.get('clean_open_frac', ''))
    row['raw_gripper_mean'] = r.get('raw_gripper_mean', c.get('raw_gripper_mean', ''))
    row['raw_gripper_max'] = r.get('raw_gripper_max', c.get('raw_gripper_max', ''))
    row['qpos_pre'] = r.get('qpos_pre', c.get('qpos_abs_sum_pre', ''))
    row['qpos_mean'] = r.get('qpos_mean', c.get('qpos_abs_sum_window_mean', ''))
    row['actual_max_step'] = r.get('actual_max_step', c.get('actual_max_step', ''))
    row['stratum'] = r.get('stratum', c.get('candidate_stratum', ''))
    # No raw vis/rand counts from master
    row['vis_open_count'] = ''; row['vis_streak'] = ''
    row['rand_open_count'] = ''; row['rand_streak'] = ''
    row['vis_qpos_delta'] = ''; row['rand_qpos_delta'] = ''
    all_rows.append(row)

# 2. Smoke + Expansion labels (fully classified from vis/rand counts)
for source_label, rows_src in [('smoke', smoke_rows), ('expansion', exp_rows)]:
    for r in rows_src:
        task = r.get('task_key', '')
        sid = r.get('state_id', '')
        seed = r.get('seed', '')
        ws = r.get('window_start', '')
        we = r.get('window_end', '')

        vo = int(r.get('vis_open_count', 0)); vs = int(r.get('vis_streak', 0))
        ro = int(r.get('rand_open_count', 0)); rs = int(r.get('rand_streak', 0))
        vq = float(r.get('vis_qpos_delta_shifted', 0))
        rq = float(r.get('rand_qpos_delta_shifted', 0))

        tax = classify(vo, vs, ro, rs, vq, rq, 100)  # n_steps from pair label trace

        row = {
            'pair_id': r['pair_id'],
            'source_batch': source_label,
            'task_key': task,
            'state_id': sid,
            'seed': seed,
            'window_start': ws,
            'window_end': we,
            'cmd_specific': tax['cmd_specific'],
            'rand_command_sensitive': tax['rand_command_sensitive'],
            'random_command_confounded': tax['random_command_confounded'],
            'vis_specific_phys': tax['vis_specific_phys'],
            'shared_qpos_response': tax['shared_qpos_response'],
            'rand_phys_confound': tax['rand_phys_confound'],
            'negative_clean': tax['negative_clean'],
            'unstable_or_edge': tax['unstable_or_edge'],
            'abstain_any': tax['abstain_any'],
            'label_tier': source_label + '_pair',
            'vis_open_count': str(vo),
            'vis_streak': str(vs),
            'rand_open_count': str(ro),
            'rand_streak': str(rs),
            'vis_qpos_delta': str(round(vq, 6)),
            'rand_qpos_delta': str(round(rq, 6)),
        }

        # Clean features from candidates
        key = (task, sid, seed, ws, we)
        c = cand_lookup.get(key, {})
        row['clean_open_count'] = c.get('clean_open_count', '')
        row['clean_open_frac'] = c.get('clean_open_frac', '')
        row['raw_gripper_mean'] = c.get('raw_gripper_mean', '')
        row['raw_gripper_max'] = c.get('raw_gripper_max', '')
        row['qpos_pre'] = c.get('qpos_abs_sum_pre', '')
        row['qpos_mean'] = c.get('qpos_abs_sum_window_mean', '')
        row['actual_max_step'] = c.get('actual_max_step', '')
        row['stratum'] = c.get('candidate_stratum', '')
        all_rows.append(row)

# ── Write ──
fieldnames = [
    'pair_id', 'source_batch', 'task_key', 'state_id', 'seed',
    'window_start', 'window_end',
    'cmd_specific', 'rand_command_sensitive', 'random_command_confounded',
    'vis_specific_phys', 'shared_qpos_response', 'rand_phys_confound',
    'negative_clean', 'unstable_or_edge', 'abstain_any',
    'label_tier',
    'vis_open_count', 'vis_streak', 'rand_open_count', 'rand_streak',
    'vis_qpos_delta', 'rand_qpos_delta',
    'clean_open_count', 'clean_open_frac', 'raw_gripper_mean', 'raw_gripper_max',
    'qpos_pre', 'qpos_mean', 'actual_max_step', 'stratum',
]

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(all_rows)

# ── Audit ──
print('=== 72-PAIR FREEZE AUDIT ===')
print('Total pairs: %d' % len(all_rows))
print('Sources: %s' % dict(Counter(r['source_batch'] for r in all_rows)))

for head in ['cmd_specific', 'rand_command_sensitive', 'random_command_confounded',
             'vis_specific_phys', 'shared_qpos_response', 'rand_phys_confound',
             'negative_clean', 'unstable_or_edge', 'abstain_any']:
    n = sum(1 for r in all_rows if r[head] == '1')
    print('  %-30s %d' % (head, n))

print('\nBy task:')
task_stats = {}
for r in all_rows:
    tk = r['task_key']
    task_stats.setdefault(tk, Counter())
    for head in ['cmd_specific', 'vis_specific_phys', 'abstain_any', 'negative_clean']:
        if r[head] == '1':
            task_stats[tk][head] += 1
for tk in sorted(task_stats):
    s = task_stats[tk]
    print('  %-20s cmd=%2d phys=%2d abstain=%2d neg=%2d' %
          (tk, s.get('cmd_specific', 0), s.get('vis_specific_phys', 0),
           s.get('abstain_any', 0), s.get('negative_clean', 0)))

# Provenance
print('\nProvenance:')
for k, v in PROVENANCE.items():
    print('  %s: %s' % (k, v))
print('\nOutput: %s' % OUT)
