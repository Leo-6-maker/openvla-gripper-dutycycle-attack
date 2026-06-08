#!/usr/bin/env python3
"""Generate Silver confirmation queue + worker scripts for GPU 2,6 and 4,5.

8 key parents from 72-pair pool, 2 repeat seeds each → 32 total jobs.
"""
import csv, os

PARENTS = [
    ('bbq_sauce', 2, 2, 100, 110, 'cmd_phys_surprise', 'smoke HN misclassified as cmd+phys'),
    ('bbq_sauce', 2, 2, 200, 210, 'cmd_phys_new', 'expansion cmd+phys positive'),
    ('cream_cheese', 1, 1, 145, 155, 'phys_only', 'rare phys_only pattern'),
    ('milk', 0, 0, 70, 80, 'cmd_phys_anchor', 'stable cmd+phys pipeline health'),
    ('milk', 0, 0, 230, 240, 'confounded_both', 'VIS=8 RAND=11 abstain critical'),
    ('tomato_sauce', 2, 2, 150, 160, 'rand_command', 'RAND=11 > VIS=5 rand_cmd signal'),
    ('tomato_sauce', 2, 2, 90, 100, 'rand_phys', 'RAND qpos=0.065 phys confound'),
    ('salad_dressing', 2, 2, 120, 130, 'clean_negative', 'V=2 R=0 hard neg anchor'),
]

OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

# --- Build queue CSV ---
rows = []
for task, sid, orig_seed, ws, we, cat, reason in PARENTS:
    for repeat in range(2):
        seed = orig_seed * 100 + repeat + 1
        rows.append({
            'parent_task': task, 'state_id': str(sid), 'original_seed': str(orig_seed),
            'repeat_seed': str(seed), 'window_start': str(ws), 'window_end': str(we),
            'category': cat, 'reason': reason, 'repeat_idx': str(repeat),
        })

os.makedirs(OUT_DIR, exist_ok=True)
queue_path = os.path.join(OUT_DIR, 'confirmation_queue.csv')
with open(queue_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('Queue: %d parents x 2 repeats = %d jobs' % (len(PARENTS), len(PARENTS)*4))

# --- Generate worker scripts ---
workers = [
    {'name': 'worker_26', 'gpu': '2,6', 'parents': PARENTS[0:4]},
    {'name': 'worker_45', 'gpu': '4,5', 'parents': PARENTS[4:8]},
]
job_id_base = [400000, 400100]

for wi, w in enumerate(workers):
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# Silver confirmation: %s GPU=%s' % (w['name'], w['gpu']))
    lines.append('# %d parents x 2 repeats = %d jobs' % (len(w['parents']), len(w['parents'])*4))
    lines.append('set +e')
    lines.append('')
    lines.append('export CUDA_VISIBLE_DEVICES=%s' % w['gpu'])
    lines.append('')
    lines.append('echo "data_anchor=d4a3827 code_commit=e33b5e4 batch=silver_confirmation"')
    lines.append('echo "[$(date +%%H:%%M:%%S)] %s CONFIRMATION START: %d parents"' % (w['name'], len(w['parents'])))
    lines.append('')

    jid = job_id_base[wi]
    for task, sid, orig_seed, ws, we, cat, reason in w['parents']:
        for repeat in range(2):
            seed = orig_seed * 100 + repeat + 1
            pair_id = 'silver_%s_%s_s%d_w%d_%d_seed%d_r%d' % (cat, task, sid, ws, we, orig_seed, repeat)
            vis_jid = jid; jid += 1
            rand_jid = jid; jid += 1

            # VIS
            lines.append('echo "=== VIS %d: %s s%d [%d,%d] seed=%d %s r%d ==="' % (vis_jid, task, sid, ws, we, seed, cat, repeat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "VIS_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, seed, vis_jid, pair_id, OUT_DIR, vis_jid, pair_id))
            lines.append('')

            # RAND
            lines.append('echo "=== RAND %d: %s s%d [%d,%d] seed=%d %s r%d ==="' % (rand_jid, task, sid, ws, we, seed, cat, repeat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "RAND_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, seed, rand_jid, pair_id, OUT_DIR, rand_jid, pair_id))
            lines.append('')

    lines.append('echo "[$(date +%%H:%%M:%%S)] %s CONFIRMATION DONE"' % w['name'])
    lines.append('')

    script_path = 'scripts/stageb/run_confirmation_%s.sh' % w['name']
    os.makedirs(os.path.dirname(script_path) or '.', exist_ok=True)
    with open(script_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('Generated %s (%d parents -> %d jobs)' % (script_path, len(w['parents']), len(w['parents'])*4))
