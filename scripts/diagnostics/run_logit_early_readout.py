#!/usr/bin/env python3
"""Early readout: analyze logit features as soon as priority candidates complete.
Checks non-degeneracy and whether claim_usable positives are detected.
"""
import csv, os, sys, glob
import numpy as np
from collections import defaultdict

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OUT = '/data/liuyu/outputs/logit_margin_validation_20260606'

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

# ── Collect traces from all completed candidates ──────────────────
trace_files = glob.glob(os.path.join(OUT, '*_s*_clean_w*_*.log'))
print('Found %d candidate log files' % len(trace_files))

# Per-candidate aggregate
candidates = []
for log_path in sorted(trace_files):
    # Parse candidate identity from filename
    fname = os.path.basename(log_path)
    parts = fname.replace('.log','').split('_')
    # Expected: {task}_s{state}_clean_w{ws}_{we}
    try:
        # Find s{N} pattern
        task = parts[0]
        sid = ''
        ws = ''
        we = ''
        for i, p in enumerate(parts):
            if p.startswith('s') and p[1:].isdigit():
                sid = p[1:]
            if p == 'w' and i+1 < len(parts):
                ws = parts[i+1]
                if i+2 < len(parts):
                    we = parts[i+2]
                break
    except:
        continue

    # Parse log for trace features
    logit_data = []
    with open(log_path) as f:
        for line in f:
            if 'gripper_logit_margin' in line and 'clean_logit_feats' not in line:
                # Try to extract numeric values
                for key in ['gripper_logit_margin', 'gripper_entropy',
                           'gripper_logit_open_mass', 'gripper_logit_close_mass',
                           'gripper_top2_margin', 'all_action_entropy']:
                    if key in line:
                        try:
                            # Extract float after the key
                            import re
                            m = re.search(r"'%s':\s*([-\d.]+)" % key, line)
                            if m:
                                val = float(m.group(1))
                                # Find or create step dict
                                pass
                        except:
                            pass

    # Also check trace CSV files
    ep_dir = os.path.join(OUT, '%s_s%s' % (task, sid))
    trace_csvs = glob.glob(os.path.join(ep_dir, 'traces', '*clean*trace.csv'))
    steps = []
    for tc in trace_csvs:
        with open(tc) as f:
            steps = list(csv.DictReader(f))
        break

    if not steps:
        continue

    # Filter to window
    window_steps = []
    for s in steps:
        try:
            step = int(float(s.get('step', -1)))
        except:
            continue
        if ws and we:
            try:
                if int(ws) <= step <= int(we):
                    window_steps.append(s)
            except:
                window_steps.append(s)

    if len(window_steps) < 2:
        continue

    # Extract logit features
    margins = []
    entropies = []
    open_masses = []
    top2s = []
    for s in window_steps:
        for name, lst in [('gripper_logit_margin', margins),
                          ('gripper_entropy', entropies),
                          ('gripper_logit_open_mass', open_masses),
                          ('gripper_top2_margin', top2s)]:
            try:
                v = float(s.get(name, 0))
                lst.append(v)
            except:
                pass

    if not margins:
        continue

    candidates.append({
        'task_key': task, 'state_id': sid,
        'window_start': ws, 'window_end': we,
        'n_steps': len(window_steps),
        'logit_margin_min': float(np.min(margins)),
        'logit_margin_mean': float(np.mean(margins)),
        'logit_margin_std': float(np.std(margins)),
        'entropy_mean': float(np.mean(entropies)) if entropies else 0,
        'entropy_max': float(np.max(entropies)) if entropies else 0,
        'open_mass_max': float(np.max(open_masses)) if open_masses else 0,
        'top2_margin_min': float(np.min(top2s)) if top2s else 0,
        'margin_unique': len(set(round(m, 6) for m in margins)),
    })

print('Parsed %d candidates with logit features' % len(candidates))

# ── Analysis ──────────────────────────────────────────────────────
if not candidates:
    print('NO CANDIDATES PARSED — check trace files')
    sys.exit(0)

all_margins = [c['logit_margin_mean'] for c in candidates]
all_entropy = [c['entropy_mean'] for c in candidates]
margin_unique = len(set(round(m, 6) for m in all_margins))

print()
print('=== NON-DEGENERACY CHECK ===')
print('Margin range: [%.6f, %.6f]' % (min(all_margins), max(all_margins)))
print('Margin unique values: %d / %d' % (margin_unique, len(candidates)))
print('Entropy range: [%.6f, %.6f]' % (min(all_entropy), max(all_entropy)))

if margin_unique <= 2:
    print('VERDICT: LOGIT MARGIN IS DEGENERATE (near-constant)')
else:
    print('VERDICT: Logit margin HAS variance (%d unique values)' % margin_unique)

# Check per-candidate detail
print()
print('=== PER-CANDIDATE ===')
for c in sorted(candidates, key=lambda x: x['logit_margin_mean']):
    print('%s s%s [%s,%s]: margin_min=%.6f mean=%.6f std=%.6f entropy=%.6f unique=%d' % (
        c['task_key'], c['state_id'], c['window_start'], c['window_end'],
        c['logit_margin_min'], c['logit_margin_mean'], c['logit_margin_std'],
        c['entropy_mean'], c['margin_unique']))

# ── Check if any of the 7 claim_usable show different margins ─────
print()
claim_usable = [
    ('alphabet_soup','4','4','21'), ('bbq_sauce','9','22','39'),
    ('butter','5','25','42'), ('cream_cheese','4','28','45'),
    ('ketchup','1','21','38'), ('milk','1','8','25'), ('milk','4','19','36'),
]
print('=== CLAIM_USABLE POSITIVES ===')
found = 0
for tk, sid, ws, we in claim_usable:
    match = [c for c in candidates if c['task_key']==tk and c['state_id']==sid
             and c['window_start']==ws and c['window_end']==we]
    if match:
        c = match[0]
        found += 1
        print('%s s%s [%s,%s]: margin=%.6f entropy=%.6f' % (
            tk, sid, ws, we, c['logit_margin_mean'], c['entropy_mean']))
    else:
        print('%s s%s [%s,%s]: NOT YET PROCESSED' % (tk, sid, ws, we))
print('Found: %d/7 claim_usable processed' % found)

# Compare with strong positives
strong_pos = [c for c in candidates if
    (c['task_key']=='ketchup' and c['state_id']=='0') or
    (c['task_key']=='butter' and c['state_id']=='0')]
if strong_pos:
    print()
    print('=== STRONG POSITIVES (batch1) ===')
    for c in strong_pos:
        print('%s s%s: margin=%.6f entropy=%.6f' % (
            c['task_key'], c['state_id'], c['logit_margin_mean'], c['entropy_mean']))
