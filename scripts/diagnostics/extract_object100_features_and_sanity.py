#!/usr/bin/env python3
"""Extract features from Object100 step_records at both legacy VIS windows and teacher windows.
Also run teacher window sanity/re-anchor audit. CPU-only."""
import csv, os, sys, json, glob
import numpy as np
from collections import Counter

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
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

# ── Load step_records ────────────────────────────────────────────
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

# ── Load teacher labels ──────────────────────────────────────────
teacher_path = os.path.join(OBJ100_DIR, 'tables', 'object100_teacher_window_labels.csv')
with open(teacher_path) as f:
    teacher_labels = list(csv.DictReader(f))

teacher_by_key = {}
for r in teacher_labels:
    task_full = r.get('task_name', '').strip(); sid = r.get('state_id', '').strip()
    task_short = None
    for short, full in TASK_SHORT_TO_FULL.items():
        if full == task_full: task_short = short; break
    if task_short is None: continue
    key = (task_short, sid)
    if key not in teacher_by_key: teacher_by_key[key] = []
    teacher_by_key[key].append(r)

# ── Load Existing31 ──────────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    existing31 = list(csv.DictReader(f))

# ── Feature extraction helpers ───────────────────────────────────
def safe_f(v, default=0.0):
    try: return float(v)
    except: return default

def safe_i(v, default=-1):
    try: return int(v)
    except: return default

def extract_window_features(records, ws, we):
    """Extract window-level online-legal features from step_records."""
    n_total = len(records)
    window_rows = [r for r in records if ws <= r.get('step_idx', -1) <= we]
    pre_rows = [r for r in records if max(0, ws - 20) <= r.get('step_idx', -1) < ws]
    if len(window_rows) < 2: return None

    # Gripper qpos
    qpos_vals = np.array([safe_f(r.get('gripper_qpos', 0)) for r in window_rows])
    grip_width = np.array([safe_f(r.get('gripper_width', 0)) for r in window_rows])
    grip_cmd = np.array([safe_f(r.get('gripper_command', 0)) for r in window_rows])
    grip_action = np.array([safe_f(r.get('action_gripper', 0)) for r in window_rows])

    # EEF
    eef_x = np.array([safe_f(r.get('eef_x', 0)) for r in window_rows])
    eef_y = np.array([safe_f(r.get('eef_y', 0)) for r in window_rows])
    eef_z = np.array([safe_f(r.get('eef_z', 0)) for r in window_rows])

    # Env action
    raw_actions = np.array([[safe_f(r.get('raw_action', '0').split(',')[i] if isinstance(r.get('raw_action', ''), str) and ',' in str(r.get('raw_action', '')) else 0) for i in range(7)] for r in window_rows])

    # Pre-window context
    pre_qpos = np.array([safe_f(r.get('gripper_qpos', 0)) for r in pre_rows]) if pre_rows else qpos_vals[:1]
    pre_grip = np.array([safe_f(r.get('gripper_command', 0)) for r in pre_rows]) if pre_rows else grip_cmd[:1]

    n = len(window_rows)

    # Clean open streak
    streak = 0; max_streak = 0
    for v in (grip_cmd > 0):
        if v: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    clean_longest_open_streak = int(max_streak)

    return {
        'n_window_frames': n, 'n_total_steps': n_total,
        # Gripper qpos
        'gripper_qpos_mean': round(float(np.mean(qpos_vals)), 6),
        'gripper_qpos_std': round(float(np.std(qpos_vals)), 6),
        'gripper_qpos_min': round(float(np.min(qpos_vals)), 6),
        'gripper_qpos_max': round(float(np.max(qpos_vals)), 6),
        'gripper_qpos_at_start': round(float(qpos_vals[0]), 6),
        'gripper_qpos_range': round(float(np.max(qpos_vals) - np.min(qpos_vals)), 6),
        'gripper_is_closed': 1.0 if float(np.mean(qpos_vals)) < 0.03 else 0.0,
        'gripper_is_open': 1.0 if float(np.mean(qpos_vals)) > 0.035 else 0.0,
        # Gripper width
        'gripper_width_mean': round(float(np.mean(grip_width)), 6),
        'gripper_width_std': round(float(np.std(grip_width)), 6),
        # Gripper command
        'gripper_command_mean': round(float(np.mean(grip_cmd)), 6),
        'gripper_command_std': round(float(np.std(grip_cmd)), 6),
        'gripper_command_open_rate': round(float(np.sum(grip_cmd > 0) / max(n, 1)), 4),
        # Gripper action
        'grip_action_mean': round(float(np.mean(grip_action)), 6),
        'grip_action_std': round(float(np.std(grip_action)), 6),
        'clean_open_count': int(np.sum(grip_cmd > 0)),
        'clean_open_rate': round(float(np.sum(grip_cmd > 0) / max(n, 1)), 4),
        'clean_longest_open_streak': max_streak,
        # EEF
        'eef_displacement': round(float(np.linalg.norm([eef_x[-1]-eef_x[0], eef_y[-1]-eef_y[0], eef_z[-1]-eef_z[0]])), 6),
        'eef_z_mean': round(float(np.mean(eef_z)), 4),
        'eef_z_trend': round(float(eef_z[-1] - eef_z[0]) if n > 1 else 0.0, 4),
        # Temporal
        'qpos_delta_from_pre': round(float(np.mean(qpos_vals) - np.mean(pre_qpos)), 6),
        'grip_delta_from_pre': round(float(np.mean(grip_cmd) - np.mean(pre_grip)), 4),
        # Position
        'window_start_frac': round(float(ws / max(n_total, 1)), 4),
        'window_center_frac': round(float((ws + we) / 2 / max(n_total, 1)), 4),
        'steps_remaining': n_total - we,
    }

# ── Process all windows ──────────────────────────────────────────
legacy_rows = []
main_rows = []
sanity_rows = []

# Process existing31 for legacy features
for r in existing31:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    ws = int(r['window_start']); we = int(r['window_end'])
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)

    records = load_step_records(task, sid)
    if records is None:
        legacy_rows.append({'window_id': cid, 'features_available': 'no'})
        continue

    feats = extract_window_features(records, ws, we)
    if feats:
        row = {'window_id': cid, 'task_key': task, 'state_id': sid,
               'window_start': str(ws), 'window_end': str(we),
               'feature_source': 'object100_step_records_legacy_vis_window',
               'features_available': 'yes', **feats,
               'vis_open_count': r.get('vis_open_count', ''), 'label_status': r.get('label_status', ''),
               'phys_resp': r.get('label_physical_response', '')}
        legacy_rows.append(row)

# Process teacher windows for mainline features + sanity
seen_episodes = set()
for (task, sid), teachers in teacher_by_key.items():
    for t in teachers:
        tws = safe_i(t.get('window_start', ''), -1); twe = safe_i(t.get('window_end', ''), -1)
        if tws < 0 or twe < 0: continue
        mechanism = t.get('mechanism_type', '').strip()
        eligible = t.get('mechanism_eligible', '').strip().lower() == 'true'
        clean_success = t.get('clean_success', '').strip().lower() == 'true'
        cid = '%s_s%s_teacher_w%d_%d' % (task, sid, tws, twe)
        ep_key = (task, sid)

        records = load_step_records(task, sid)
        if records is None: continue

        feats = extract_window_features(records, tws, twe)
        if feats is None: continue

        # ── Teacher window sanity ────────────────────────────────
        n_total = len(records)
        # Find LAST natural release: where gripper_qpos goes from <0.03 to >0.035 for the FINAL time
        # (the actual object release, not the pre-grasp approach opening)
        qpos_series = [safe_f(r.get('gripper_qpos', 0)) for r in records]
        cmd_series = [safe_f(r.get('gripper_command', 0)) for r in records]
        steps_series = [r.get('step_idx', 0) for r in records]

        # Find final release: last transition from qpos<0.03 to qpos>0.035
        final_release_step = -1
        for i in range(1, len(records)):
            if qpos_series[i-1] < 0.03 and qpos_series[i] > 0.035:
                final_release_step = steps_series[i]

        # Find initial grasp-open: first time gripper opens to approach object
        first_approach_open = -1
        for i in range(len(records)):
            if qpos_series[i] > 0.03 and cmd_series[i] > 0:
                first_approach_open = steps_series[i]
                break

        # Teacher window gripper state
        win_qpos = [safe_f(r.get('gripper_qpos', 0)) for r in records if tws <= r.get('step_idx', -1) <= twe]
        win_cmd = [safe_f(r.get('gripper_command', 0)) for r in records if tws <= r.get('step_idx', -1) <= twe]
        avg_qpos = float(np.mean(win_qpos)) if win_qpos else 0.0
        avg_cmd = float(np.mean(win_cmd)) if win_cmd else 0.0

        # Release safe: window is AFTER final_release_step (gripper already released object)
        release_safe_too_late = final_release_step > 0 and tws >= final_release_step
        # Gripper already commanding OPEN during window: ceiling effect for delta-to-clean
        ceiling_open = avg_cmd > 0.5  # model naturally commands OPEN in window

        # Object lifted: gripper_width > 0.01 (something between fingers)
        grip_width_vals = [safe_f(r.get('gripper_width', 0)) for r in records if tws <= r.get('step_idx', -1) <= twe]
        object_lifted_in_window = len(grip_width_vals) > 0 and float(np.mean(grip_width_vals)) > 0.005

        # Near target
        eef_z_vals = [safe_f(r.get('eef_z', 0)) for r in records if tws <= r.get('step_idx', -1) <= twe]
        eef_z_mean_window = float(np.mean(eef_z_vals)) if eef_z_vals else 0.0

        # Holding phase: gripper is CLOSED (qpos<0.03) but object is being carried
        holding_phase = avg_qpos < 0.03 and object_lifted_in_window

        # Re-anchor
        reanchored_ws = tws; reanchored_we = twe
        if ceiling_open and not release_safe_too_late:
            # Window where model already commands OPEN — look for earlier CLOSE phase
            # Find last step before window where gripper was CLOSED
            last_closed = -1
            for r in records:
                s = r.get('step_idx', -1)
                if s < tws and safe_f(r.get('gripper_qpos', 0)) < 0.03:
                    last_closed = s
            if last_closed > 0:
                reanchored_ws = max(0, last_closed - 5)
                reanchored_we = min(n_total - 1, last_closed + 8)
        elif release_safe_too_late:
            reanchored_ws = max(0, final_release_step - 20)
            reanchored_we = min(n_total - 1, final_release_step - 3)

        recommended_use = 'use_teacher_window'
        if release_safe_too_late:
            if reanchored_ws > 0 and reanchored_we > reanchored_ws:
                recommended_use = 'use_reanchored_pre_release_window'
            else:
                recommended_use = 'exclude_too_late'
        elif ceiling_open and not holding_phase:
            if reanchored_ws != tws:
                recommended_use = 'use_reanchored_pre_open_window'
            else:
                recommended_use = 'ceiling_model_already_open'
        if not clean_success:
            recommended_use = 'exclude_clean_fail'

        # Alignment: before or after first approach open and final release
        if final_release_step > 0 and tws >= final_release_step:
            align = 'after_final_release'
        elif first_approach_open > 0 and tws < first_approach_open:
            align = 'before_approach_open'
        else:
            align = 'between_approach_and_release'

        sanity_rows.append({
            'window_id': cid, 'task_key': task, 'state_id': sid,
            'teacher_window_start': str(tws), 'teacher_window_end': str(twe),
            'mechanism_type': mechanism, 'mechanism_eligible': str(eligible),
            'clean_success': str(clean_success),
            'gripper_command_mean': str(round(avg_cmd, 4)),
            'gripper_qpos_mean': str(round(avg_qpos, 6)),
            'gripper_width_mean': str(round(float(np.mean(grip_width_vals)) if grip_width_vals else 0.0, 6)),
            'clean_open_count_in_window': str(feats['clean_open_count']),
            'clean_open_rate_in_window': str(feats['clean_open_rate']),
            'first_approach_open_step': str(first_approach_open),
            'final_release_step': str(final_release_step),
            'object_lifted_in_window': str(object_lifted_in_window),
            'eef_z_mean_window': str(round(eef_z_mean_window, 4)),
            'ceiling_open': str(ceiling_open),
            'release_safe_or_too_late': str(release_safe_too_late),
            'holding_phase': str(holding_phase),
            'window_alignment_vs_natural_open': align,
            'reanchored_window_start': str(reanchored_ws),
            'reanchored_window_end': str(reanchored_we),
            'recommended_use': recommended_use,
            'n_total_steps': str(n_total),
        })

        # Mainline feature row
        mf = {'window_id': cid, 'task_key': task, 'state_id': sid,
              'window_start': str(tws), 'window_end': str(twe),
              'feature_source': 'object100_step_records_teacher_window',
              'features_available': 'yes', **feats,
              'mechanism_type': mechanism, 'mechanism_eligible': str(eligible),
              'recommended_use': recommended_use}
        main_rows.append(mf)

        # If reanchored, also extract features at reanchored position
        if 'reanchor' in recommended_use:
            refeats = extract_window_features(records, reanchored_ws, reanchored_we)
            if refeats:
                rc = {'window_id': cid + '_reanchored', 'task_key': task, 'state_id': sid,
                      'window_start': str(reanchored_ws), 'window_end': str(reanchored_we),
                      'feature_source': 'object100_step_records_reanchored',
                      'features_available': 'yes', **refeats,
                      'mechanism_type': mechanism, 'recommended_use': recommended_use}
                main_rows.append(rc)

# ── Write outputs ────────────────────────────────────────────────
def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

write_csv(os.path.join(REPO, 'tables', 'online_features_existing31_at_vis_windows.csv'), legacy_rows)
write_csv(os.path.join(REPO, 'tables', 'online_features_object100_teacher_windows.csv'), main_rows)
write_csv(os.path.join(REPO, 'tables', 'object100_teacher_window_sanity.csv'), sanity_rows)
print('Legacy features: %d rows' % len(legacy_rows))
print('Mainline features: %d rows' % len(main_rows))
print('Sanity audit: %d rows' % len(sanity_rows))

# ── Sanity summary ───────────────────────────────────────────────
use_counts = Counter(r['recommended_use'] for r in sanity_rows)
print('\nTeacher window recommendations:')
for u, c in use_counts.most_common():
    print('  %s: %d' % (u, c))
release_late = sum(1 for r in sanity_rows if r['release_safe_or_too_late'] == 'True')
print('Release_safe_too_late: %d/%d' % (release_late, len(sanity_rows)))

# ── Sanity report ────────────────────────────────────────────────
lines = []
lines.append('# Object100 Teacher Window Sanity Audit')
lines.append('')
lines.append('**Windows**: %d' % len(sanity_rows))
lines.append('')
lines.append('## Recommendation Distribution')
lines.append('')
for u, c in use_counts.most_common():
    lines.append('- **%s**: %d' % (u, c))
lines.append('')
lines.append('## Release-Safe / Too-Late Windows')
lines.append('')
late_rows = [r for r in sanity_rows if r['release_safe_or_too_late'] == 'True']
lines.append('%d windows are AFTER natural open start (release_safe_too_late).' % len(late_rows))
lines.append('These should NOT be used for VIS label generation.')
lines.append('')
for r in sorted(late_rows, key=lambda x: x['window_id']):
    lines.append('- **%s**: teacher [%s,%s], nat_open=%s, reanchor=[%s,%s], use=%s' % (
        r['window_id'], r['teacher_window_start'], r['teacher_window_end'],
        r['natural_open_start_step'], r['reanchored_window_start'], r['reanchored_window_end'],
        r['recommended_use']))
lines.append('')
lines.append('## Feature Coverage')
lines.append('')
lines.append('- Legacy at VIS windows: %d rows' % len(legacy_rows))
lines.append('- Mainline at teacher windows: %d rows' % len(main_rows))
lines.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_TEACHER_WINDOW_SANITY_AUDIT.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Wrote sanity report')
