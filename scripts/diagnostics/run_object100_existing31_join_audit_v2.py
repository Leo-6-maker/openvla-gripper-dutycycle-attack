#!/usr/bin/env python3
"""Object100 × Existing31 Join Audit v2.

Split provenance into granular fields. Separate legacy (VIS window) from
mainline (teacher window) eligibility. No coarse provenance_status gate.
"""
import csv, os, sys, glob, json

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'
OUT_CSV = os.path.join(REPO, 'tables', 'object100_existing31_join_audit_v2.csv')
OUT_RPT = os.path.join(REPO, 'reports', 'OBJECT100_EXISTING31_JOIN_AUDIT_V2.md')

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

# ── Load Object100 data ──────────────────────────────────────────
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

manifest_path = os.path.join(OBJ100_DIR, 'tables', 'official_clean_artifact_rich_manifest.csv')
with open(manifest_path) as f:
    manifest = list(csv.DictReader(f))

manifest_by_key = {}
for r in manifest:
    task_full = r.get('task_name', '').strip(); sid = r.get('state_id', '').strip()
    seed = r.get('seed', '0').strip()
    task_short = None
    for short, full in TASK_SHORT_TO_FULL.items():
        if full == task_full: task_short = short; break
    if task_short is None: continue
    manifest_by_key[(task_short, sid, seed)] = r

# ── Load Existing31 ──────────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    existing31 = list(csv.DictReader(f))

# ── Check shadow traces ──────────────────────────────────────────
shadow_dir = '/data/liuyu/outputs/proprionostep_shadow_calib_20260607'
def has_shadow_trace(task, sid):
    return bool(glob.glob(os.path.join(shadow_dir, 'vis_%s_s%s_clean_*_trace.csv' % (task, sid))))

# ── Check step_records existence ─────────────────────────────────
def step_records_exists(task_short, sid):
    task_full = TASK_SHORT_TO_FULL.get(task_short, '')
    if not task_full: return False, ''
    task_dir = task_full.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
    path = os.path.join(OBJ100_DIR, 'runs', 'libero_object',
                        'pick_up_the_%s_and_place_it_in_the_basket_state%s' % (task_dir, sid),
                        'step_records.jsonl')
    exists = os.path.exists(path)
    return exists, path

# ── Join ─────────────────────────────────────────────────────────
rows = []
for r in existing31:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    ws = int(r['window_start']); we = int(r['window_end'])
    label_status = r.get('label_status', '?').strip()
    taxonomy = r.get('taxonomy', '?').strip()
    vis_open = r.get('vis_open_count', '0')
    phys_resp = r.get('label_physical_response', '0')
    provenance = r.get('provenance_status', '?').strip()
    source_batch = r.get('source_batch', '?').strip()
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)

    # ── Granular provenance fields ───────────────────────────────
    obj100_ep = manifest_by_key.get((task, sid, '0'), None)
    clean_trace_exists = obj100_ep is not None
    shadow_trace_ok = has_shadow_trace(task, sid)
    sr_exists, sr_path = step_records_exists(task, sid)

    # VIS labels
    vis_label_available = vis_open not in ('', '0', '?') and vis_open != '0'
    has_random = False  # need to check matched random traces
    phys_resp_val = float(phys_resp) if phys_resp and phys_resp != '?' else 0.0

    # ── Window alignment ─────────────────────────────────────────
    teacher_matches = teacher_by_key.get((task, sid), [])
    teacher_best = None
    if teacher_matches:
        # Find teacher window closest to existing31 window
        best_dist = 9999
        for t in teacher_matches:
            tws = int(t.get('window_start', -1)); twe = int(t.get('window_end', -1))
            if tws < 0: continue
            dist = abs((ws + we) / 2 - (tws + twe) / 2)
            if dist < best_dist: best_dist = dist; teacher_best = t

    teacher_ws = int(teacher_best.get('window_start', -1)) if teacher_best else -1
    teacher_we = int(teacher_best.get('window_end', -1)) if teacher_best else -1

    # Window alignment type
    vis_len = we - ws
    teacher_len = teacher_we - teacher_ws
    overlap = max(0, min(we, teacher_we) - max(ws, teacher_ws))
    overlap_frac = overlap / max(min(vis_len, teacher_len), 1)

    if overlap_frac >= 0.7:
        alignment = 'exact_same_window'
    elif teacher_best is not None:
        alignment = 'same_episode_different_window'
    elif teacher_best is None and vis_label_available:
        alignment = 'vis_window_only'
    else:
        alignment = 'teacher_window_only'

    # ── Eligibility ──────────────────────────────────────────────
    # Legacy diagnostic: clean trace + VIS label + window alignment OK
    legacy_eligible = bool(
        clean_trace_exists and sr_exists and
        vis_label_available and
        alignment in ('exact_same_window', 'same_episode_different_window', 'vis_window_only')
    )
    legacy_exclude = ''
    if not clean_trace_exists: legacy_exclude = 'no_clean_trace'
    elif not sr_exists: legacy_exclude = 'no_step_records'
    elif not vis_label_available: legacy_exclude = 'no_vis_label'
    elif alignment not in ('exact_same_window', 'same_episode_different_window', 'vis_window_only'):
        legacy_exclude = 'window_misaligned'

    # Mainline training: teacher window + mechanism_eligible + clean success
    teacher_eligible = teacher_best is not None and teacher_best.get('mechanism_eligible', '').strip().lower() == 'true'
    teacher_ok = bool(
        clean_trace_exists and sr_exists and teacher_eligible and
        obj100_ep.get('success', '').strip().lower() == 'true'
    )
    teacher_exclude = ''
    if not clean_trace_exists: teacher_exclude = 'no_clean_trace'
    elif not sr_exists: teacher_exclude = 'no_step_records'
    elif not teacher_eligible: teacher_exclude = 'teacher_not_mechanism_eligible'
    elif not obj100_ep.get('success', '').strip().lower() == 'true': teacher_exclude = 'clean_episode_failed'

    rows.append({
        'window_id': cid,
        'task_key': task, 'state_id': sid,
        'window_start': str(ws), 'window_end': str(we),
        # Clean trace
        'clean_trace_exists': str(clean_trace_exists),
        'object100_exact_task_state_seed_match': str(obj100_ep is not None),
        'step_records_available': str(sr_exists),
        'object100_step_records_path': sr_path,
        'rgb_available': str(sr_exists),  # frames embedded in step_records dir
        'online_features_available': str(sr_exists or shadow_trace_ok),
        # VIS labels
        'existing31_vis_label_available': str(vis_label_available),
        'existing31_vis_open_count': vis_open,
        'existing31_phys_resp': phys_resp,
        'matched_random_label_available': str(has_random),
        'source_batch': source_batch,
        # Teacher
        'teacher_window_found': str(teacher_best is not None),
        'teacher_window_start': str(teacher_ws), 'teacher_window_end': str(teacher_we),
        'teacher_mechanism_type': teacher_best.get('mechanism_type', '') if teacher_best else '',
        'teacher_mechanism_eligible': teacher_best.get('mechanism_eligible', '') if teacher_best else '',
        'teacher_clean_success': obj100_ep.get('success', '') if obj100_ep else '',
        # Window alignment
        'window_alignment_type': alignment,
        'vis_teacher_overlap_frac': str(round(overlap_frac, 3)),
        'vis_teacher_center_distance': str(abs((ws + we) / 2 - (teacher_ws + teacher_we) / 2)) if teacher_best else '',
        # Eligibility
        'eligible_for_legacy_existing31_diagnostic': str(legacy_eligible),
        'eligible_for_main_teacher_window_training': str(teacher_ok),
        'exclude_reason_legacy': legacy_exclude,
        'exclude_reason_main': teacher_exclude,
        # Metadata
        'label_status': label_status, 'taxonomy': taxonomy,
    })

# ── Write CSV ────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('Wrote %d rows' % len(rows))

# ── Summary ──────────────────────────────────────────────────────
from collections import Counter
legacy_ok = sum(1 for r in rows if r['eligible_for_legacy_existing31_diagnostic'] == 'True')
main_ok = sum(1 for r in rows if r['eligible_for_main_teacher_window_training'] == 'True')
align_counts = Counter(r['window_alignment_type'] for r in rows)

print('Legacy eligible: %d/31' % legacy_ok)
print('Mainline eligible: %d/31 (teacher windows from existing31 episodes)' % main_ok)
for a, c in align_counts.most_common():
    print('  %s: %d' % (a, c))

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Object100 × Existing31 Join Audit v2')
lines.append('')
lines.append('## Summary')
lines.append('')
lines.append('| Metric | Count |')
lines.append('|---|---|')
lines.append('| Object100 clean episodes match | %d/31 |' % sum(1 for r in rows if r['object100_exact_task_state_seed_match'] == 'True'))
lines.append('| Step records available | %d/31 |' % sum(1 for r in rows if r['step_records_available'] == 'True'))
lines.append('| VIS labels available | %d/31 |' % sum(1 for r in rows if r['existing31_vis_label_available'] == 'True'))
lines.append('| Legacy diagnostic eligible | %d/31 |' % legacy_ok)
lines.append('| Mainline teacher training eligible | %d/31 |' % main_ok)
lines.append('')

lines.append('## Window Alignment')
lines.append('')
for a, c in align_counts.most_common():
    lines.append('- **%s**: %d' % (a, c))
lines.append('')

lines.append('## Eligible for Legacy Diagnostic')
lines.append('')
leg_rows = [r for r in rows if r['eligible_for_legacy_existing31_diagnostic'] == 'True']
for r in sorted(leg_rows, key=lambda x: (x['task_key'], x['state_id'])):
    tws = r['teacher_window_start']; twe = r['teacher_window_end']
    lines.append('- **%s** [%s,%s] VIS vs teacher [%s,%s] align=%s overlap=%s' % (
        r['window_id'], r['window_start'], r['window_end'],
        tws, twe, r['window_alignment_type'], r['vis_teacher_overlap_frac']))
lines.append('')

lines.append('## Eligible for Mainline Teacher Training')
lines.append('')
main_rows = [r for r in rows if r['eligible_for_main_teacher_window_training'] == 'True']
for r in sorted(main_rows, key=lambda x: (x['task_key'], x['state_id'])):
    lines.append('- **%s** teacher [%s,%s] mechanism=%s' % (
        r['window_id'], r['teacher_window_start'], r['teacher_window_end'],
        r['teacher_mechanism_type']))
lines.append('')

lines.append('## Excluded Summary')
lines.append('')
leg_excl = Counter(r['exclude_reason_legacy'] for r in rows if r['exclude_reason_legacy'])
for e, c in leg_excl.most_common():
    lines.append('- Legacy exclude: **%s** (%d)' % (e, c))
main_excl = Counter(r['exclude_reason_main'] for r in rows if r['exclude_reason_main'])
for e, c in main_excl.most_common():
    lines.append('- Main exclude: **%s** (%d)' % (e, c))

with open(OUT_RPT, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote report')
