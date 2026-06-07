#!/usr/bin/env python3
"""Stage-B v1: Build pair labels from matched VIS/random windows.
Requires patched_stageb_v1 trace_version only."""
import csv, os, sys
import numpy as np
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'tables', 'stageb_v1_windows_for_labels.csv')

with open(INPUT_CSV) as f:
    rows = list(csv.DictReader(f))
print('Input: %d paired windows' % len(rows))

labels = []
for r in rows:
    vis_open = int(r['vis_open']); vis_streak = int(r['vis_streak'])
    rand_open = int(r['rand_open']); rand_streak = int(r['rand_streak'])
    vis_delta = float(r['vis_delta_shifted']); rand_delta = float(r['rand_delta_shifted'])
    vis_delta_post = float(r['vis_delta_post'])

    cmd_pos = (vis_open >= 6 or vis_streak >= 6) and not (rand_open >= 6 or rand_streak >= 6)
    rand_conf = (rand_open >= 6 or rand_streak >= 6)
    phys_sens = vis_delta >= 0.01 or vis_delta_post >= 0.01
    phys_strict = vis_delta >= 0.02 or vis_delta_post >= 0.02
    vis_specific = phys_sens and not (rand_delta >= 0.01)

    labels.append({
        'task_key': r['task_key'], 'window_start': r['window_start'], 'window_end': r['window_end'],
        'vis_open_count': r['vis_open'], 'vis_streak': r['vis_streak'],
        'rand_open_count': r['rand_open'], 'rand_streak': r['rand_streak'],
        'vis_qpos_delta_shifted': r['vis_delta_shifted'],
        'rand_qpos_delta_shifted': r['rand_delta_shifted'],
        'cmd_susceptible': str(int(cmd_pos)),
        'random_confounded': str(int(rand_conf)),
        'physical_response_sensitive': str(int(phys_sens)),
        'physical_response_strict': str(int(phys_strict)),
        'vis_specific_physical_response': str(int(vis_specific)),
    })

out_csv = os.path.join(REPO, 'tables', 'stageb_v1_pair_labels.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
    w.writeheader(); w.writerows(labels)
print('Wrote %d labels to %s' % (len(labels), out_csv))

# Audit
cmd = sum(1 for r in labels if r['cmd_susceptible'] == '1')
phys = sum(1 for r in labels if r['physical_response_sensitive'] == '1')
print('cmd_susceptible: %d/%d (%.1f%%)' % (cmd, len(labels), cmd/max(len(labels),1)*100))
print('physical_response_sensitive: %d/%d (%.1f%%)' % (phys, len(labels), phys/max(len(labels),1)*100))

# Report
task_counts = Counter(r['task_key'] for r in labels)
lines = ['# Stage-B v1 Pair Label Audit', '', '**Rows**: %d' % len(labels), '',
         '| Task | Paired | cmd_pos | phys_sens | vis_spec |', '|---|---|---|---|---|']
for t, c in task_counts.most_common():
    tr = [r for r in labels if r['task_key'] == t]
    lines.append('| %s | %d | %d | %d | %d |' % (t, c,
        sum(1 for r in tr if r['cmd_susceptible']=='1'),
        sum(1 for r in tr if r['physical_response_sensitive']=='1'),
        sum(1 for r in tr if r['vis_specific_physical_response']=='1')))
with open(os.path.join(REPO, 'reports', 'STAGEB_V1_PAIR_LABEL_AUDIT.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Done')
