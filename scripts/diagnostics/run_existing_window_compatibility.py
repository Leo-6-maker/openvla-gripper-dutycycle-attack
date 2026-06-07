#!/usr/bin/env python3
"""Step 2: ProprioNoStep existing-window compatibility audit."""
import csv, json, glob, os, sys
import numpy as np
from collections import defaultdict, Counter

out = '/data/liuyu/outputs/proprionostep_shadow_calib_20260607'
repo = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
shared = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'

# Load labels
with open(shared + '/object_phase_response_labels_v2.csv') as f:
    labels = list(csv.DictReader(f))

# Load mechanism taxonomy
mech_path = repo + '/tables/vulnerability_mechanism_taxonomy_audit.csv'
mech_map = {}
if os.path.exists(mech_path):
    with open(mech_path) as f:
        for r in csv.DictReader(f):
            k = (r['task_key'].strip(), r['state_id'].strip(),
                 r['window_start'].strip(), r['window_end'].strip())
            mech_map[k] = r.get('mechanism_type','')

# Index traces by (task, seed)
traces_by_key = {}
for sf in sorted(glob.glob(out + '/*_summary.json')):
    tf = sf.replace('_summary.json', '_trace.csv')
    if not os.path.exists(tf): continue
    with open(sf) as f: summary = json.load(f)
    with open(tf) as f: trace_rows = list(csv.DictReader(f))
    key = (summary['task'], summary['state_id'])
    if key not in traces_by_key:
        traces_by_key[key] = trace_rows

print('Indexed {} unique (task, state) pairs'.format(len(traces_by_key)))

# Trigger configs
configs = [
    ('hzd_001_d5_c20', 'hazard', 0.01, 5, 20),
    ('hzd_003_d3_c20', 'hazard', 0.03, 3, 20),
    ('hzd_003_d5_c20', 'hazard', 0.03, 5, 20),
    ('hzd_003_d8_c20', 'hazard', 0.03, 8, 20),
    ('hzd_005_d3_c20', 'hazard', 0.05, 3, 20),
    ('hzd_005_d5_c20', 'hazard', 0.05, 5, 20),
]

def compute_trigger_steps(trace_rows, mode, thr, dur, cool):
    triggers = []
    trig_counter = 0; cooldown = 0; trig_active = False
    for i, r in enumerate(trace_rows):
        if cooldown > 0: cooldown -= 1; continue
        h = float(r.get('proprionostep_hazard_score', 0))
        p = int(r.get('proprionostep_phase_idx', -1))
        triggered = False
        if mode == 'hazard': triggered = h >= thr
        elif mode == 'phase': triggered = p not in (4,5) and p >= 0
        elif mode == 'hazard_or_phase': triggered = (h >= thr) or (p not in (4,5) and p >= 0)
        if triggered: trig_counter += 1
        else: trig_active = False; trig_counter = 0
        if trig_counter >= dur and not trig_active:
            trig_active = True; triggers.append(i)
            cooldown = cool
    return triggers

# Audit each labeled window
results = []
for r in labels:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    ws = int(r['window_start']); we = int(r['window_end'])
    label_status = r.get('label_status',''); train_use = r.get('label_use','')
    taxonomy = r.get('taxonomy','')
    mech = mech_map.get((task, sid, str(ws), str(we)), '')
    cid = '%s_s%s_w%s_%s' % (task, sid, ws, we)

    trace = traces_by_key.get((task, sid))
    if trace is None:
        results.append({'candidate_id':cid,'task_key':task,'state_id':sid,
            'window_start':str(ws),'window_end':str(we),
            'label_status':label_status,'mechanism_type':mech,'taxonomy':taxonomy,
            'train_use':train_use,'trace_available':'no'})
        continue

    window_rows = [trace[i] for i in range(max(0,ws), min(we+1, len(trace))) if i < len(trace)]
    if len(window_rows) < 2:
        results.append({'candidate_id':cid,'task_key':task,'state_id':sid,
            'window_start':str(ws),'window_end':str(we),
            'label_status':label_status,'mechanism_type':mech,'taxonomy':taxonomy,
            'train_use':train_use,'trace_available':'yes','window_steps_in_trace':str(len(window_rows))})
        continue

    hazards = [float(rr.get('proprionostep_hazard_score', 0)) for rr in window_rows]
    phases = [int(rr.get('proprionostep_phase_idx', -1)) for rr in window_rows]
    releases = [float(rr.get('proprionostep_release_safe_score', 0)) for rr in window_rows]
    confs = [float(rr.get('proprionostep_phase_confidence', 0)) for rr in window_rows]
    phase_mode = Counter(p for p in phases if p >= 0).most_common(1)

    row = {'candidate_id':cid,'task_key':task,'state_id':sid,
        'window_start':str(ws),'window_end':str(we),
        'label_status':label_status,'mechanism_type':mech,'taxonomy':taxonomy,
        'train_use':train_use,'trace_available':'yes',
        'window_steps_in_trace':str(len(window_rows)),
        'hazard_max_in_window':str(round(max(hazards),6)),
        'hazard_mean_in_window':str(round(np.mean(hazards),6)),
        'phase_idx_mode_in_window':str(phase_mode[0][0]) if phase_mode else '-1',
        'phase_confidence_mean_in_window':str(round(np.mean(confs),6)),
        'release_safe_mean_in_window':str(round(np.mean(releases),6)),
    }

    for cfg_name, mode, thr, dur, cool in configs:
        triggers = compute_trigger_steps(trace, mode, thr, dur, cool)
        hit_in = any(ws <= t <= we for t in triggers)
        hit_5 = any(abs(t - ws) <= 5 or abs(t - we) <= 5 for t in triggers)
        hit_10 = any(abs(t - ws) <= 10 or abs(t - we) <= 10 for t in triggers)
        nearest_dist = 999; nearest_step = -1
        for t in triggers:
            d = 0 if ws <= t <= we else (ws - t if t < ws else t - we)
            if d < nearest_dist: nearest_dist = d; nearest_step = t
        row[cfg_name+'_hit_in_window'] = str(hit_in)
        row[cfg_name+'_hit_within_5'] = str(hit_5)
        row[cfg_name+'_hit_within_10'] = str(hit_10)
        row[cfg_name+'_nearest_trigger_step'] = str(nearest_step)
        row[cfg_name+'_nearest_trigger_distance'] = str(nearest_dist)

    results.append(row)

# Write
out_csv = repo + '/tables/proprionostep_existing_window_compatibility.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print('Wrote %d rows to %s' % (len(results), out_csv))

# Summary
primary = 'hzd_001_d5_c20'
print('\n=== Config: %s ===' % primary)
by_mech = defaultdict(lambda: {'total':0,'hit':0,'hit5':0,'hit10':0,'dists':[]})
for r in results:
    mech = r.get('mechanism_type','?')
    by_mech[mech]['total'] += 1
    if r.get(primary+'_hit_in_window') == 'True': by_mech[mech]['hit'] += 1
    if r.get(primary+'_hit_within_5') == 'True': by_mech[mech]['hit5'] += 1
    if r.get(primary+'_hit_within_10') == 'True': by_mech[mech]['hit10'] += 1
    try: by_mech[mech]['dists'].append(int(r.get(primary+'_nearest_trigger_distance','999')))
    except: pass

for mech in sorted(by_mech.keys()):
    d = by_mech[mech]
    md = np.mean(d['dists']) if d['dists'] else 0
    print('%-30s total=%d hit=%d hit5=%d hit10=%d dist=%.0f' % (mech, d['total'], d['hit'], d['hit5'], d['hit10'], md))

by_label = defaultdict(lambda: {'total':0,'hit':0})
for r in results:
    ls = r.get('label_status','?')
    by_label[ls]['total'] += 1
    if r.get(primary+'_hit_in_window') == 'True': by_label[ls]['hit'] += 1
print('\nBy label:')
for ls in sorted(by_label.keys()):
    d = by_label[ls]; print('  %s: %d/%d (%.0f%%)' % (ls, d['hit'], d['total'], 100*d['hit']/max(d['total'],1)))

# Physical bridge
phys = [r for r in results if 'physical_bridge' in r.get('mechanism_type','')]
phys_hit = sum(1 for r in phys if r.get(primary+'_hit_in_window')=='True')
print('\nPhysical bridge: %d/%d hit (%.0f%%)' % (phys_hit, len(phys), 100*phys_hit/max(len(phys),1)))

# Specific per-candidate for claim_usable
claim = [r for r in results if r.get('taxonomy','')=='claim_usable']
print('\nClaim_usable detail:')
for r in claim:
    print('  %s: hazard=%.4f/%0.4f hit=%s dist=%s' % (
        r['candidate_id'], float(r.get('hazard_mean_in_window',0)),
        float(r.get('hazard_max_in_window',0)),
        r.get(primary+'_hit_in_window','?'), r.get(primary+'_nearest_trigger_distance','?')))
