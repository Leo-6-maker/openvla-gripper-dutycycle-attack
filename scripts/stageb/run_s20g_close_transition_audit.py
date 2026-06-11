#!/usr/bin/env python3
"""S20G Step 2: Close-transition audit for all paired windows."""
import json, glob, csv, os
from collections import defaultdict
import numpy as np

OUT = '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611'
TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'

# Load paired label table
paired = []
with open(TABLES + '/s20g_v031_paired_label_table.csv') as f:
    paired = list(csv.DictReader(f))

# Trace finder: map (task, sid) -> clean trace path
TRACE_BASE = '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean'
trace_map = {
    ('ketchup', '1'): TRACE_BASE + '/trace_ketchup_s1_w0_10_s20d_clean_seed0_job960100.csv',
    ('tomato_sauce', '3'): TRACE_BASE + '/trace_tomato_sauce_s3_w0_10_s20d_clean_seed0_job960101.csv',
    ('tomato_sauce', '5'): TRACE_BASE + '/trace_tomato_sauce_s5_w0_10_s20d_clean_seed0_job960102.csv',
}

def load_trace(task, sid):
    path = trace_map.get((task, sid))
    if not path:
        return None
    with open(path) as f:
        return list(csv.DictReader(f))

def find_close_transitions(rows, ws, we):
    """
    Find nearest stable OPEN->CLOSE transition near [ws, we).
    Transition: prev 3 steps >=2 OPEN, next 3 steps >=2 CLOSE.
    """
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d

    trans_steps = []
    for i in range(3, len(rows) - 3):
        # Pre 3 steps
        pre_open = sum(1 for j in range(i-3, i) if g(rows[j], 'decoded_open_bool') == 1)
        # Post 3 steps
        post_close = sum(1 for j in range(i, i+3) if g(rows[j], 'decoded_open_bool') == 0)
        if pre_open >= 2 and post_close >= 2:
            trans_steps.append(i)

    if not trans_steps:
        return None

    wc = (ws + we) / 2.0
    # Find nearest transition to window center
    nearest = min(trans_steps, key=lambda t: abs(t - wc))
    dist = nearest - wc  # positive = transition after window center

    # Pre-open streak before transition
    pre_streak = 0
    for j in range(nearest-1, -1, -1):
        if g(rows[j], 'decoded_open_bool') == 1:
            pre_streak += 1
        else:
            break

    # Post-close streak after transition
    post_streak = 0
    for j in range(nearest, len(rows)):
        if g(rows[j], 'decoded_open_bool') == 0:
            post_streak += 1
        else:
            break

    # Transition overlap with window
    overlap_pre = 1 if nearest - 1 >= ws and nearest - 1 < we else 0  # pre-step in window
    overlap_center = 1 if nearest >= ws and nearest < we else 0  # transition in window
    overlap_post = 1 if nearest + 1 >= ws and nearest + 1 < we else 0

    # Close commitment: how many CLOSE steps in post-window segment
    post_window_closes = sum(1 for j in range(we, min(we + 10, len(rows)))
                            if g(rows[j], 'decoded_open_bool') == 0)
    close_commitment = post_window_closes / min(10, max(1, len(rows) - we))

    return {
        'nearest_transition_step': nearest,
        'distance_to_transition': dist,
        'pre_open_streak': pre_streak,
        'post_close_streak': post_streak,
        'transition_overlap_pre': overlap_pre,
        'transition_overlap_center': overlap_center,
        'transition_overlap_post': overlap_post,
        'close_commitment_score': round(close_commitment, 3),
        'total_transitions_found': len(trans_steps),
    }

# Compute for each paired window
rows = []
for p in paired:
    task = p['task']; sid = p['state_id']
    ws = int(p['window_start']); we = int(p['window_end'])

    trace = load_trace(task, sid)
    if trace is None:
        trans = None
    else:
        trans = find_close_transitions(trace, ws, we)

    row = dict(p)
    if trans:
        row.update(trans)
    else:
        row.update({k: '' for k in ['nearest_transition_step','distance_to_transition',
                     'pre_open_streak','post_close_streak',
                     'transition_overlap_pre','transition_overlap_center','transition_overlap_post',
                     'close_commitment_score','total_transitions_found']})
    rows.append(row)

# Write
fields = list(rows[0].keys())
with open(TABLES + '/s20g_close_transition_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)

# Summary by classification
summary = defaultdict(lambda: {'count':0, 'dist_sum':0.0, 'close_commit_sum':0.0, 'overlap_count':0, 'post_close_sum':0.0})
for r in rows:
    cls = r['classification']
    summary[cls]['count'] += 1
    if r.get('distance_to_transition', '') != '':
        summary[cls]['dist_sum'] += float(r['distance_to_transition'])
        summary[cls]['close_commit_sum'] += float(r['close_commitment_score'])
        summary[cls]['post_close_sum'] += float(r['post_close_streak'])
        if int(r.get('transition_overlap_center', 0)):
            summary[cls]['overlap_count'] += 1

with open(TABLES + '/s20g_close_transition_vs_label_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['classification','count','mean_distance','mean_close_commitment','mean_post_close_streak','transition_overlap_rate'])
    for cls in ['task_effect','contact_effect_weak','cmd_specific','cmd_borderline','no_effect','random_sensitive']:
        s = summary.get(cls)
        if s and s['count'] > 0:
            w.writerow([cls, s['count'],
                        round(s['dist_sum']/s['count'], 1),
                        round(s['close_commit_sum']/s['count'], 3),
                        round(s['post_close_sum']/s['count'], 1),
                        round(s['overlap_count']/s['count'], 2)])

print('Close-transition audit complete')
print('Summary by classification:')
for cls in ['task_effect','contact_effect_weak','cmd_specific','cmd_borderline','no_effect']:
    s = summary.get(cls)
    if s and s['count'] > 0:
        print('  %-22s n=%2d  mean_dist=%6.1f  close_commit=%.3f  post_close=%.1f  overlap=%.2f' % (
            cls, s['count'], s['dist_sum']/s['count'], s['close_commit_sum']/s['count'],
            s['post_close_sum']/s['count'], s['overlap_count']/s['count']))
