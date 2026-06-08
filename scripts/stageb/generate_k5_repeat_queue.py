#!/usr/bin/env python3
"""Phase C: Generate K=5 repeat stability queue (8 parents × 5 seeds × 2 conditions = 80 jobs).

env_seed = original_seed (fixed for deterministic replay)
attack_seed = 0..4 (varied for perturbation diversity)
"""
import csv, os

PARENTS = [
    # task, state, env_seed, ws, we, category
    ('milk', 0, 0, 70, 80, 'cmd_anchor'),
    ('milk', 0, 0, 230, 240, 'confounded_swing'),
    ('tomato_sauce', 2, 2, 150, 160, 'rand_cmd_drift'),
    ('tomato_sauce', 2, 2, 90, 100, 'rand_phys_flip'),
    ('salad_dressing', 2, 2, 120, 130, 'neg_drift'),
    ('bbq_sauce', 2, 2, 100, 110, 'hn_surprise'),
    ('tomato_sauce', 2, 2, 115, 125, 'strict_phys_master'),       # candidate 7
    ('alphabet_soup', 1, 1, 65, 75, 'clean_negative_expansion'),   # candidate 8
]
K = 5
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

os.makedirs(OUT_DIR, exist_ok=True)

# Build queue CSV
queue_rows = []
for task, sid, env_seed, ws, we, cat in PARENTS:
    for k in range(K):
        attack_seed = k
        pair_id = 'k5_%s_%s_s%d_w%d_%d_env%d_atk%d' % (cat, task, sid, ws, we, env_seed, attack_seed)
        queue_rows.append({
            'parent_category': cat, 'task_key': task, 'state_id': str(sid),
            'env_seed': str(env_seed), 'attack_seed': str(attack_seed),
            'window_start': str(ws), 'window_end': str(we),
            'condition': 'vis_pgd', 'pair_id': pair_id, 'repeat_idx': str(k),
        })
        queue_rows.append({
            'parent_category': cat, 'task_key': task, 'state_id': str(sid),
            'env_seed': str(env_seed), 'attack_seed': str(attack_seed),
            'window_start': str(ws), 'window_end': str(we),
            'condition': 'random_linf', 'pair_id': pair_id, 'repeat_idx': str(k),
        })

queue_path = os.path.join(OUT_DIR, 'k5_repeat_queue.csv')
with open(queue_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(queue_rows[0].keys()))
    w.writeheader()
    w.writerows(queue_rows)
print('Queue: %d parents × %d seeds × 2 conditions = %d jobs' % (len(PARENTS), K, len(queue_rows)))

# Generate worker scripts
workers = [
    {'name': 'worker_10', 'gpu': '1,0', 'parents': PARENTS[0:3]},   # 3 parents = 30 jobs
    {'name': 'worker_26', 'gpu': '2,6', 'parents': PARENTS[3:5]},   # 2 parents = 20 jobs
    {'name': 'worker_45', 'gpu': '4,5', 'parents': PARENTS[5:8]},   # 3 parents = 30 jobs
]
job_id_base = [500000, 500030, 500050]

for wi, w in enumerate(workers):
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# K5 repeat stability: %s GPU=%s' % (w['name'], w['gpu']))
    n_jobs = len(w['parents']) * K * 2
    lines.append('# %d parents × %d seeds = %d jobs' % (len(w['parents']), K, n_jobs))
    lines.append('set +e')
    lines.append('')
    lines.append('export CUDA_VISIBLE_DEVICES=%s' % w['gpu'])
    lines.append('')
    lines.append('echo "S5_K5_REPEAT code=a20379f anchor=d4a3827"')
    lines.append('echo "[$(date +%%H:%%M:%%S)] %s K5 START: %d parents, %d jobs"' % (w['name'], len(w['parents']), n_jobs))
    lines.append('')

    jid = job_id_base[wi]
    for task, sid, env_seed, ws, we, cat in w['parents']:
        for k in range(K):
            attack_seed = k
            pair_id = 'k5_%s_%s_s%d_w%d_%d_env%d_atk%d' % (cat, task, sid, ws, we, env_seed, attack_seed)
            vis_jid = jid; jid += 1
            rand_jid = jid; jid += 1

            lines.append('echo "=== VIS %d: %s s%d [%d,%d] env=%d atk=%d %s ==="' % (vis_jid, task, sid, ws, we, env_seed, attack_seed, cat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed %d --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "VIS_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, env_seed, env_seed, attack_seed, vis_jid, pair_id, OUT_DIR, vis_jid, pair_id))
            lines.append('')

            lines.append('echo "=== RAND %d: %s s%d [%d,%d] env=%d atk=%d %s ==="' % (rand_jid, task, sid, ws, we, env_seed, attack_seed, cat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed %d --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "RAND_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, env_seed, env_seed, attack_seed, rand_jid, pair_id, OUT_DIR, rand_jid, pair_id))
            lines.append('')

    lines.append('echo "[$(date +%%H:%%M:%%S)] %s K5 DONE"' % w['name'])
    lines.append('')

    script_path = 'scripts/stageb/run_k5_%s.sh' % w['name']
    os.makedirs(os.path.dirname(script_path) or '.', exist_ok=True)
    with open(script_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('Generated %s (%d parents → %d jobs)' % (script_path, len(w['parents']), n_jobs))

print('Job ranges: worker_10=%d-%d  worker_26=%d-%d  worker_45=%d-%d' %
      (500000, 500029, 500030, 500049, 500050, 500079))
print('Total: 80 jobs')
