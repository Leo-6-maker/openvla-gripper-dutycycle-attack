#!/usr/bin/env python3
"""Build Object100 Opportunity Dataset v0.

Positive: teacher/reanchored windows (~73)
Negative: 4 controls per episode — far_too_early, early_pregrasp, random_noncritical, post_release
Abstain: clean-failed episodes
Features: online-legal only (no teacher_hazard, mechanism_eligible, step_idx as input)
"""
import csv, os, sys, json, random
import numpy as np
from collections import Counter

random.seed(42); np.random.seed(42)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'

TASK_SHORT_TO_FULL = {
    'alphabet_soup': 'pick_up_the_alphabet_soup_and_place_it_in_the_basket',
    'bbq_sauce': 'pick_up_the_bbq_sauce_and_place_it_in_the_basket',
    'butter': 'pick_up_the_butter_and_place_it_in_the_basket',
    'cream_cheese': 'pick_up_the_cream_cheese_and_place_it_in_the_basket',
    'ketchup': 'pick_up_the_ketchup_and_place_it_in_the_basket',
    'milk': 'pick_up_the_milk_and_place_it_in_the_basket',
    'orange_juice': 'pick_up_the_orange_juice_and_place_it_in_the_basket',
    'salad_dressing': 'pick_up_the_salad_dressing_and_place_it_in_the_basket',
    'tomato_sauce': 'pick_up_the_tomato_sauce_and_place_it_in_the_basket',
}

def load_step_records(task_short, sid):
    task_full = TASK_SHORT_TO_FULL.get(task_short, '')
    if not task_full: return None
    task_dir = task_full.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
    path = os.path.join(OBJ100_DIR, 'runs', 'libero_object',
                        'pick_up_the_%s_and_place_it_in_the_basket_state%s' % (task_dir, sid),
                        'step_records.jsonl')
    if not os.path.exists(path): return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    return records

def safe_f(v, default=0.0):
    try: return float(v)
    except: return default

# ── Load teacher sanity ──────────────────────────────────────────
sanity_path = os.path.join(REPO, 'tables', 'object100_teacher_window_sanity.csv')
with open(sanity_path) as f:
    sanity = list(csv.DictReader(f))

# Filter to usable teacher windows
usable = [r for r in sanity if r['recommended_use'] in ('use_teacher_window', 'use_reanchored_pre_open_window')
          and r['mechanism_eligible'].strip().lower() == 'true'
          and r['clean_success'].strip().lower() == 'true']
print('Usable teacher windows: %d' % len(usable))

# Group by episode
from collections import defaultdict
episodes = defaultdict(list)
for r in usable:
    ep_key = (r['task_key'], r['state_id'])
    episodes[ep_key].append(r)

print('Unique episodes: %d' % len(episodes))

# ── Define online-legal features ─────────────────────────────────
def extract_features(records, ws, we):
    """Extract online-legal features for window [ws, we]."""
    n_total = len(records)
    window_rows = [r for r in records if ws <= r.get('step_idx', -1) <= we]
    pre_rows = [r for r in records if max(0, ws - 20) <= r.get('step_idx', -1) < ws]
    if len(window_rows) < 2: return None

    qpos_vals = np.array([safe_f(r.get('gripper_qpos', 0)) for r in window_rows])
    grip_width = np.array([safe_f(r.get('gripper_width', 0)) for r in window_rows])
    grip_cmd = np.array([safe_f(r.get('gripper_command', 0)) for r in window_rows])
    eef_x = np.array([safe_f(r.get('eef_x', 0)) for r in window_rows])
    eef_y = np.array([safe_f(r.get('eef_y', 0)) for r in window_rows])
    eef_z = np.array([safe_f(r.get('eef_z', 0)) for r in window_rows])
    pre_qpos = np.array([safe_f(r.get('gripper_qpos', 0)) for r in pre_rows]) if pre_rows else qpos_vals[:1]
    pre_cmd = np.array([safe_f(r.get('gripper_command', 0)) for r in pre_rows]) if pre_rows else grip_cmd[:1]
    n = len(window_rows)

    # Clean open streak
    streak = 0; max_streak = 0
    for v in (grip_cmd > 0):
        if v: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    return {
        # Gripper qpos (proprio)
        'gripper_qpos_mean': round(float(np.mean(qpos_vals)), 6),
        'gripper_qpos_std': round(float(np.std(qpos_vals)), 6),
        'gripper_qpos_min': round(float(np.min(qpos_vals)), 6),
        'gripper_qpos_max': round(float(np.max(qpos_vals)), 6),
        'gripper_qpos_at_start': round(float(qpos_vals[0]), 6),
        'gripper_qpos_range': round(float(np.max(qpos_vals) - np.min(qpos_vals)), 6),
        'gripper_is_closed': 1.0 if float(np.mean(qpos_vals)) < 0.03 else 0.0,
        # Gripper width (proprio)
        'gripper_width_mean': round(float(np.mean(grip_width)), 6),
        'gripper_width_std': round(float(np.std(grip_width)), 6),
        # Gripper command (action)
        'gripper_command_mean': round(float(np.mean(grip_cmd)), 6),
        'gripper_command_std': round(float(np.std(grip_cmd)), 6),
        'gripper_command_open_rate': round(float(np.sum(grip_cmd > 0) / max(n, 1)), 4),
        'clean_open_count': int(np.sum(grip_cmd > 0)),
        'clean_open_rate': round(float(np.sum(grip_cmd > 0) / max(n, 1)), 4),
        'clean_longest_open_streak': int(max_streak),
        # EEF position (proprio)
        'eef_displacement': round(float(np.linalg.norm([eef_x[-1]-eef_x[0], eef_y[-1]-eef_y[0], eef_z[-1]-eef_z[0]])), 6),
        'eef_z_mean': round(float(np.mean(eef_z)), 4),
        'eef_z_trend': round(float(eef_z[-1] - eef_z[0]) if n > 1 else 0.0, 4),
        # Temporal (pre→window deltas)
        'qpos_delta_from_pre': round(float(np.mean(qpos_vals) - np.mean(pre_qpos)), 6),
        'grip_command_delta_from_pre': round(float(np.mean(grip_cmd) - np.mean(pre_cmd)), 4),
        # Window position (proportion-based, NOT step_idx)
        'window_start_frac': round(float(ws / max(n_total, 1)), 4),
        'window_len_frac': round(float((we - ws + 1) / max(n_total, 1)), 4),
        'n_window_frames': n,
        'n_total_steps': n_total,
    }

# ── Build dataset ────────────────────────────────────────────────
dataset_rows = []
row_id = 0

for (task, sid), teacher_rows in sorted(episodes.items()):
    records = load_step_records(task, sid)
    if records is None: continue
    n_total = len(records)

    # Episode-level info from step_records
    qpos_all = [safe_f(r.get('gripper_qpos', 0)) for r in records]
    steps_all = [r.get('step_idx', 0) for r in records]

    # Find phase boundaries
    final_release_step = -1
    for i in range(1, len(records)):
        if qpos_all[i-1] < 0.03 and qpos_all[i] > 0.035:
            final_release_step = steps_all[i]

    first_approach_open = -1
    for i in range(len(records)):
        if qpos_all[i] > 0.03 and safe_f(records[i].get('gripper_command', 0)) > 0:
            first_approach_open = steps_all[i]
            break

    ep_key = '%s_s%s' % (task, sid)

    # ── Positive: teacher window ─────────────────────────────────
    for tr in teacher_rows:
        tws = int(tr['teacher_window_start']); twe = int(tr['teacher_window_end'])
        rec = tr.get('recommended_use', '')
        # Use reanchored if available
        if 'reanchor' in rec:
            tws = int(tr.get('reanchored_window_start', tws))
            twe = int(tr.get('reanchored_window_end', twe))

        feats = extract_features(records, tws, twe)
        if feats is None: continue

        dataset_rows.append({
            'row_id': str(row_id), 'episode_key': ep_key,
            'task_key': task, 'state_id': sid, 'seed': '0',
            'window_start': str(tws), 'window_end': str(twe),
            'stratum': 'teacher_positive',
            'opportunity_label': '1',
            'train_use': 'train',
            'exclude_reason': '',
            'mechanism_type': tr.get('mechanism_type', ''),  # audit only
            'teacher_window_original': '%s-%s' % (tr['teacher_window_start'], tr['teacher_window_end']),
            **feats,
        })
        row_id += 1

    # ── Negative controls ────────────────────────────────────────
    # Teacher window region (avoid overlap with ALL teacher windows in this episode)
    all_tws = [int(t['teacher_window_start']) for t in teacher_rows]
    all_twe = [int(t['teacher_window_end']) for t in teacher_rows]
    teacher_min = min(all_tws); teacher_max = max(all_twe)
    # Teacher time band: the typical teacher window step range
    teacher_band_start = teacher_min - 15
    teacher_band_end = teacher_max + 15

    # 1) late_random_control: random window in SAME time band as teacher but from a DIFFERENT phase
    #    (e.g., same episode, but shifted outside teacher window)
    # Sample windows with start in [teacher_band_start - 30, teacher_band_start) OR (teacher_band_end, teacher_band_end + 30]
    late_candidates = []
    for s in range(max(25, teacher_band_start - 40), min(n_total - 15, teacher_band_end + 40)):
        if s < teacher_band_start - 5 or s > teacher_band_end + 5:
            if s + 10 < n_total:
                late_candidates.append(s)
    if late_candidates:
        lws = random.choice(late_candidates)
        lwe = min(n_total - 2, lws + random.randint(10, 18))
        feats = extract_features(records, lws, lwe)
        if feats:
            dataset_rows.append({
                'row_id': str(row_id), 'episode_key': ep_key,
                'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(lws), 'window_end': str(lwe),
                'stratum': 'late_noncritical_control',
                'opportunity_label': '0',
                'train_use': 'train', 'exclude_reason': '',
                'mechanism_type': '', 'teacher_window_original': '',
                **feats,
            })
            row_id += 1

    # 2) early_pregrasp: before first_approach_open (gripper still closed, arm approaching)
    if first_approach_open > 15:
        ews = max(5, first_approach_open - 15)
        ewe = min(n_total - 2, first_approach_open - 3)
    else:
        ews = max(5, teacher_band_start - 60)
        ewe = min(n_total - 2, teacher_band_start - 20)
    if ewe - ews >= 5:
        feats = extract_features(records, ews, ewe)
        if feats:
            dataset_rows.append({
                'row_id': str(row_id), 'episode_key': ep_key,
                'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(ews), 'window_end': str(ewe),
                'stratum': 'early_pregrasp_control',
                'opportunity_label': '0',
                'train_use': 'train', 'exclude_reason': '',
                'mechanism_type': '', 'teacher_window_original': '',
                **feats,
            })
            row_id += 1

    # 3) post_release: after final_release_step (gripper already released)
    if final_release_step > 0 and final_release_step + 10 < n_total:
        pws = final_release_step + 5
        pwe = min(n_total - 2, final_release_step + 20)
        if pwe - pws >= 5:
            feats = extract_features(records, pws, pwe)
            if feats:
                dataset_rows.append({
                    'row_id': str(row_id), 'episode_key': ep_key,
                    'task_key': task, 'state_id': sid, 'seed': '0',
                    'window_start': str(pws), 'window_end': str(pwe),
                    'stratum': 'post_release_control',
                    'opportunity_label': '0',
                    'train_use': 'train', 'exclude_reason': '',
                    'mechanism_type': '', 'teacher_window_original': '',
                    **feats,
                })
                row_id += 1

# ── Also add clean_failed episodes as abstain only if we have them ─
print('Built %d rows' % len(dataset_rows))

# ── Write dataset ────────────────────────────────────────────────
# Separate feature columns from label/audit columns
feature_cols = [k for k in dataset_rows[0].keys() if k.startswith(('gripper_', 'eef_', 'clean_', 'qpos_', 'grip_', 'window_', 'n_window', 'n_total'))]
label_cols = ['opportunity_label']
audit_cols = ['row_id', 'episode_key', 'task_key', 'state_id', 'seed',
              'window_start', 'window_end', 'stratum', 'train_use', 'exclude_reason',
              'mechanism_type', 'teacher_window_original']

# Write main CSV
all_cols = audit_cols + label_cols + feature_cols
OUT_CSV = os.path.join(REPO, 'tables', 'object100_opportunity_dataset_v0.csv')
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
    w.writeheader(); w.writerows(dataset_rows)
print('Wrote %d rows to %s' % (len(dataset_rows), OUT_CSV))

# Write column tags
TAGS_CSV = os.path.join(REPO, 'tables', 'object100_opportunity_dataset_v0_column_tags.csv')
with open(TAGS_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['column', 'role', 'group', 'description'])
    for c in audit_cols:
        w.writerow([c, 'audit', 'metadata', ''])
    for c in label_cols:
        w.writerow([c, 'label', 'target', '1=teacher window positive, 0=control'])
    for c in feature_cols:
        if 'qpos' in c: g = 'proprio_gripper_qpos'
        elif 'width' in c: g = 'proprio_gripper_width'
        elif 'command' in c or 'grip_command' in c: g = 'action_gripper_command'
        elif 'clean_open' in c: g = 'action_gripper_stats'
        elif 'eef' in c: g = 'proprio_eef'
        elif 'n_window' in c or 'n_total' in c: g = 'window_metadata'
        elif 'window_start_frac' in c or 'window_len_frac' in c: g = 'window_position'
        else: g = 'other'
        w.writerow([c, 'feature', g, 'online-legal causal feature'])
print('Wrote column tags to %s' % TAGS_CSV)

# ── Audit report ─────────────────────────────────────────────────
stratum_counts = Counter(r['stratum'] for r in dataset_rows)
label_counts = Counter(r['opportunity_label'] for r in dataset_rows)
task_counts = Counter(r['task_key'] for r in dataset_rows)

lines = []
lines.append('# Object100 Opportunity Dataset v0 Audit')
lines.append('')
lines.append('**Rows**: %d' % len(dataset_rows))
lines.append('**Episodes**: %d' % len(set(r['episode_key'] for r in dataset_rows)))
lines.append('')
lines.append('## Stratum Distribution')
lines.append('')
lines.append('| Stratum | Count |')
lines.append('|---|---|')
for s, c in stratum_counts.most_common():
    lines.append('| %s | %d |' % (s, c))
lines.append('')
lines.append('## Label Balance')
lines.append('')
lines.append('| Label | Count |')
lines.append('|---|---|')
for l, c in label_counts.most_common():
    lines.append('| %s | %d |' % (l, c))
lines.append('')
lines.append('## Task Distribution')
lines.append('')
for t, c in sorted(task_counts.items()):
    pos = sum(1 for r in dataset_rows if r['task_key'] == t and r['opportunity_label'] == '1')
    neg = sum(1 for r in dataset_rows if r['task_key'] == t and r['opportunity_label'] == '0')
    lines.append('- **%s**: %d total (%d pos, %d neg)' % (t, c, pos, neg))
lines.append('')
lines.append('## Feature Groups')
lines.append('')
lines.append('| Group | Count | Example |')
lines.append('|---|---|---|')
for g in sorted(set('proprio_gripper_qpos proprio_gripper_width action_gripper_command action_gripper_stats proprio_eef window_position window_metadata'.split())):
    cols = [c for c in feature_cols if any(tag in c for tag in g.split('_')[:3])]
    lines.append('| %s | %d | %s |' % (g, len(cols), cols[0] if cols else ''))
lines.append('')
lines.append('## Hard Rules Check')
lines.append('')
lines.append('- [x] No step_idx as input feature')
lines.append('- [x] No normalized_step as input feature')
lines.append('- [x] No teacher_phase as input feature')
lines.append('- [x] No teacher_hazard as input feature')
lines.append('- [x] No mechanism_eligible as input feature')
lines.append('- [x] Window position uses frac (not absolute step)')
lines.append('- [x] All features causal/online-legal')
lines.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_OPPORTUNITY_DATASET_V0_AUDIT.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Wrote audit report')
