#!/usr/bin/env python3
"""Object100 × Existing31 Window Join Audit.

Classify each existing31 window by Object100 clean trace + VIS label availability.
CPU-only. No rollout.
"""
import csv, os, sys, glob, json

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'
OUT_CSV = os.path.join(REPO, 'tables', 'object100_existing31_join_audit.csv')
OUT_RPT = os.path.join(REPO, 'reports', 'OBJECT100_EXISTING31_JOIN_AUDIT.md')

# ── Task name mapping ────────────────────────────────────────────
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

# ── Load Object100 teacher window labels ─────────────────────────
teacher_path = os.path.join(OBJ100_DIR, 'tables', 'object100_teacher_window_labels.csv')
with open(teacher_path) as f:
    teacher_labels = list(csv.DictReader(f))
print('Object100 teacher labels: %d rows' % len(teacher_labels))

# Index by (task_short, state_id)
teacher_by_key = {}
for r in teacher_labels:
    task_full = r.get('task_name', '').strip()
    sid = r.get('state_id', '').strip()
    # Reverse map task name to short key
    task_short = None
    for short, full in TASK_SHORT_TO_FULL.items():
        if full == task_full:
            task_short = short; break
    if task_short is None:
        continue
    key = (task_short, sid)
    if key not in teacher_by_key:
        teacher_by_key[key] = []
    teacher_by_key[key].append(r)

print('Indexed Object100 teachers: %d unique (task, state)' % len(teacher_by_key))

# ── Load Object100 official manifest ─────────────────────────────
manifest_path = os.path.join(OBJ100_DIR, 'tables', 'official_clean_artifact_rich_manifest.csv')
with open(manifest_path) as f:
    manifest = list(csv.DictReader(f))
print('Object100 manifest: %d episodes' % len(manifest))

# Index by (task_short, state_id, seed)
manifest_by_key = {}
for r in manifest:
    task_full = r.get('task_name', '').strip()
    sid = r.get('state_id', '').strip()
    seed = r.get('seed', '0').strip()
    task_short = None
    for short, full in TASK_SHORT_TO_FULL.items():
        if full == task_full:
            task_short = short; break
    if task_short is None:
        continue
    key = (task_short, sid, seed)
    manifest_by_key[key] = r

print('Indexed Object100 manifest: %d unique episodes' % len(manifest_by_key))

# ── Load Existing31 labels ───────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    existing31 = list(csv.DictReader(f))
print('Existing31 labels: %d windows' % len(existing31))

# ── Check existing clean traces (ProprioNoStep shadow) ───────────
shadow_dir = '/data/liuyu/outputs/proprionostep_shadow_calib_20260607'
def has_shadow_trace(task, sid):
    return bool(glob.glob(os.path.join(shadow_dir, 'vis_%s_s%s_clean_*_trace.csv' % (task, sid))))

# ── Check VIS traces (3R calibration) ────────────────────────────
vis_3r_dir = '/data/liuyu/outputs/vis_calibration_matched_v2_3r_20260606'
def has_vis_3r(task, sid):
    pattern = os.path.join(vis_3r_dir, '%s_s%s' % (task, sid))
    return bool(glob.glob(pattern))

vis_1r_dir = '/data/liuyu/outputs/vis_calibration_matched_v2_1r_20260606'
def has_vis_1r(task, sid):
    pattern = os.path.join(vis_1r_dir, '%s_s%s' % (task, sid))
    return bool(glob.glob(pattern))

# ── Join ─────────────────────────────────────────────────────────
rows = []
for r in existing31:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    ws = r['window_start']; we = r['window_end']
    label_status = r.get('label_status', '?').strip()
    taxonomy = r.get('taxonomy', '?').strip()
    vis_open = r.get('vis_open_count', '0')
    phys_resp = r.get('label_physical_response', '0')
    provenance = r.get('provenance_status', '?').strip()
    cid = '%s_s%s_w%s_%s' % (task, sid, ws, we)

    # Object100 teacher
    teacher_matches = teacher_by_key.get((task, sid), [])
    # Find teacher window that overlaps with existing31 window
    teacher_window = None
    for t in teacher_matches:
        tws = int(t.get('window_start', -1)); twe = int(t.get('window_end', -1))
        if tws <= int(ws) <= twe or tws <= int(we) <= twe or (int(ws) <= tws and int(we) >= twe):
            teacher_window = t; break
    if teacher_window is None and teacher_matches:
        teacher_window = teacher_matches[0]  # best effort

    # Object100 manifest
    obj100_ep = manifest_by_key.get((task, sid, '0'), None)

    # Shadow trace
    shadow_trace_ok = has_shadow_trace(task, sid)

    # VIS traces
    vis_3r_ok = has_vis_3r(task, sid)
    vis_1r_ok = has_vis_1r(task, sid)

    # ── Classification ────────────────────────────────────────────
    clean_exists = obj100_ep is not None or shadow_trace_ok
    clean_success = False
    if obj100_ep:
        clean_success = obj100_ep.get('success', '').strip().lower() == 'true'
    elif shadow_trace_ok:
        clean_success = True  # assume success if trace exists

    label_exists = teacher_window is not None and teacher_window.get('window_detected', '').strip().lower() == 'true'
    vis_label_exists = vis_open not in ('', '0', '?')  # has VIS label
    random_label_available = False  # need to check separately
    cmd_sus_label_available = vis_label_exists and vis_open != '0'

    window_hit = ''
    if clean_exists and label_exists:
        window_hit = 'exact_match_clean_and_label'
        eligible = True
        action = 'use directly; extract online features + attach label'
    elif clean_exists and not label_exists:
        window_hit = 'clean_exists_label_missing'
        eligible = False
        action = 'schedule VIS/random label generation; clean trace already available'
    elif not clean_exists and label_exists:
        window_hit = 'label_exists_clean_missing'
        eligible = False
        action = 'recover/rerun clean rollout only'
    elif not clean_exists and not label_exists:
        # Check if Object100 has the episode at all
        if obj100_ep is None and not shadow_trace_ok:
            window_hit = 'key_mismatch_or_schema_mismatch'
            action = 'fix provenance; episode not found in Object100 manifest'
        else:
            window_hit = 'clean_failed_or_not_train_eligible'
            action = 'audit only; not train eligible'
        eligible = False

    # Provenance check
    if provenance and 'complete' not in provenance.lower():
        if eligible:
            window_hit = window_hit + '_provenance_flagged'
            eligible = False
            action = 'provenance not clean; audit before use'

    # Object100 trace path
    if obj100_ep:
        obj100_run_id = obj100_ep.get('run_id', '?')
        obj100_path = os.path.join(OBJ100_DIR, 'runs', 'libero_object',
                                    TASK_SHORT_TO_FULL.get(task, 'unknown').replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', ''),
                                    'state%s' % sid, 'step_records.jsonl')
        # Actually use the run_id for the path
        task_dir = TASK_SHORT_TO_FULL.get(task, 'unknown')
        task_dir = task_dir.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
        obj100_path = '%s/runs/libero_object/pick_up_the_%s_and_place_it_in_the_basket_state%s/step_records.jsonl' % (OBJ100_DIR, task_dir, sid)
    else:
        obj100_path = ''

    rows.append({
        'window_id': cid,
        'task_key': task, 'state_id': sid,
        'seed': '0',  # Object100 uses seed 0
        'window_start': ws, 'window_end': we,
        'object100_clean_trace_path': obj100_path,
        'object100_episode_found': str(obj100_ep is not None),
        'clean_trace_exists': str(clean_exists),
        'clean_success': str(clean_success),
        'shadow_trace_available': str(shadow_trace_ok),
        'online_features_available': str(shadow_trace_ok),
        'vis_label_available': str(vis_label_exists),
        'vis_3r_available': str(vis_3r_ok),
        'vis_1r_available': str(vis_1r_ok),
        'random_label_available': str(random_label_available),
        'command_susceptible_label_available': str(cmd_sus_label_available),
        'teacher_window_found': str(teacher_window is not None),
        'teacher_window_start': teacher_window.get('window_start', '') if teacher_window else '',
        'teacher_window_end': teacher_window.get('window_end', '') if teacher_window else '',
        'teacher_mechanism': teacher_window.get('mechanism_type', '') if teacher_window else '',
        'teacher_eligible': teacher_window.get('mechanism_eligible', '') if teacher_window else '',
        'label_status': label_status, 'taxonomy': taxonomy,
        'vis_open_count': vis_open, 'phys_resp': phys_resp,
        'provenance_status': provenance,
        'eligible_for_training': str(eligible),
        'class_match': window_hit.split('_provenance')[0] if '_provenance' in window_hit else window_hit,
        'missing_reason': '' if eligible else action,
        'recommended_action': action,
    })

# ── Write CSV ────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('Wrote %d rows to %s' % (len(rows), OUT_CSV))

# ── Report ───────────────────────────────────────────────────────
from collections import Counter
class_counts = Counter(r['class_match'] for r in rows)

lines = []
lines.append('# Object100 × Existing31 Join Audit')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Object100 episodes**: %d' % len(manifest))
lines.append('**Object100 teacher windows**: %d' % len(teacher_labels))
lines.append('**Existing31 diagnostic windows**: %d' % len(existing31))
lines.append('')

lines.append('## Join Summary')
lines.append('')
lines.append('| Class | Count |')
lines.append('|---|---|')
for cls, cnt in class_counts.most_common():
    lines.append('| %s | %d |' % (cls, cnt))
lines.append('')

lines.append('## Eligible for Training')
lines.append('')
eligible = [r for r in rows if r['eligible_for_training'] == 'True']
lines.append('- **%d/%d** windows eligible for detector training' % (len(eligible), len(rows)))
lines.append('')

lines.append('## Per-Window Detail')
lines.append('')
lines.append('| Window | Class | Clean | Shadow | VIS | Teacher | Eligible | Action |')
lines.append('|---|---|---|---|---|---|---|---|')
for r in sorted(rows, key=lambda x: (x['class_match'], x['task_key'], x['state_id'])):
    lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
        r['window_id'], r['class_match'][:25],
        'Y' if r['clean_trace_exists'] == 'True' else 'N',
        'Y' if r['shadow_trace_available'] == 'True' else 'N',
        'Y' if r['vis_label_available'] == 'True' else 'N',
        'Y' if r['teacher_window_found'] == 'True' else 'N',
        'Y' if r['eligible_for_training'] == 'True' else 'N',
        r['recommended_action'][:50]))
lines.append('')

# Actionable summary
lines.append('## Actionable Summary')
lines.append('')
for cls, cnt in class_counts.most_common():
    rows_cls = [r for r in rows if r['class_match'] == cls]
    lines.append('### %s (n=%d)' % (cls, cnt))
    for r in sorted(rows_cls, key=lambda x: (x['task_key'], x['state_id'])):
        lines.append('- **%s** s%s [%s,%s] %s | teacher: %s [%s,%s]' % (
            r['task_key'], r['state_id'], r['window_start'], r['window_end'],
            r['label_status'], r['teacher_mechanism'],
            r['teacher_window_start'], r['teacher_window_end']))
    lines.append('')

lines.append('## Next Steps')
lines.append('')
clean_exists_no_label = class_counts.get('clean_exists_label_missing', 0)
label_exists_no_clean = class_counts.get('label_exists_clean_missing', 0)
exact = class_counts.get('exact_match_clean_and_label', 0)
lines.append('1. **exact_match_clean_and_label** (%d): Extract online features, train detector v0' % exact)
lines.append('2. **clean_exists_label_missing** (%d): Schedule VIS PGD20+random attack labeling' % clean_exists_no_label)
lines.append('3. **label_exists_clean_missing** (%d): Recover clean rollout from Object100 step_records' % label_exists_no_clean)
lines.append('4. **key_mismatch / clean_failed**: Audit manually, do NOT run rollouts')
lines.append('')

with open(OUT_RPT, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote report to %s' % OUT_RPT)
print('DONE')
