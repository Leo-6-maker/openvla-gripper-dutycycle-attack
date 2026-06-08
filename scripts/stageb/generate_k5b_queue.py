#!/usr/bin/env python3
"""Phase 2: Generate K5b targeted repeat-stability queue.

16 parents x K=5 x 2 conditions = 160 jobs.
Categories: same-task contrast (6), strict phys (4), rand-sensitive (3), hard negative (3).
"""
import csv, os

PARENTS = [
    # === A: Same-task contrast (6) ===
    # Milk neighbors around K5 positives
    ('milk', 0, 0, 240, 250, 'contrast_milk_late', 'neighbor of K5 milk[230,240]'),
    ('milk', 0, 0, 75, 85, 'contrast_milk_early', 'neighbor of K5 milk[70,80]'),
    ('milk', 0, 0, 80, 90, 'contrast_milk_mid', 'same-episode contrast window'),
    # Tomato neighbors
    ('tomato_sauce', 2, 2, 155, 165, 'contrast_tomato_late', 'neighbor of K5 tomato[150,160]'),
    ('tomato_sauce', 2, 2, 95, 105, 'contrast_tomato_early', 'neighbor of K5 tomato[90,100]'),
    ('tomato_sauce', 2, 2, 85, 95, 'contrast_tomato_mid', 'same-episode contrast window'),

    # === B: Strict phys candidates (4) ===
    ('cream_cheese', 2, 2, 50, 60, 'strict_phys_cream', 'VIS=8 RAND=0, high qpos, non-edge'),
    ('bbq_sauce', 2, 2, 200, 210, 'strict_phys_bbq', 'VIS=8 RAND=0 Vq=0.037, monitor for edge'),
    ('tomato_sauce', 2, 2, 165, 175, 'strict_phys_tomato', 'neighbor of [155,165], master phys label'),
    ('salad_dressing', 2, 2, 70, 80, 'strict_phys_salad', 'untested phys candidate from master'),

    # === C: Rand-sensitive (3) ===
    ('alphabet_soup', 0, 0, 60, 70, 'rand_alpha', 'VIS=0 RAND=11, sentinel rand_cmd signal'),
    ('salad_dressing', 2, 2, 80, 90, 'rand_salad', 'RAND qpos=0.051, phys confound candidate'),
    ('orange_juice', 2, 2, 20, 30, 'rand_oj', 'V=2 RAND qpos=0.039, phys confound candidate'),

    # === D: Hard negative anchors (3) ===
    ('alphabet_soup', 1, 1, 50, 60, 'neg_alpha', 'V=2 R=0, clean neg from smoke'),
    ('orange_juice', 2, 2, 25, 35, 'neg_oj', 'V=2 R=0, clean neg expansion'),
    ('bbq_sauce', 0, 0, 60, 70, 'neg_bbq', 'V=2 R=0, clean neg, non-bbq-s2 task diversity'),
]
K = 5
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

os.makedirs(OUT_DIR, exist_ok=True)

# Build queue CSV
queue_rows = []
for task, sid, env_seed, ws, we, cat, reason in PARENTS:
    for k in range(K):
        attack_seed = k
        pair_id = 'k5b_%s_%s_s%d_w%d_%d_env%d_atk%d' % (cat, task, sid, ws, we, env_seed, attack_seed)
        queue_rows.append({
            'parent_category': cat, 'task_key': task, 'state_id': str(sid),
            'env_seed': str(env_seed), 'attack_seed': str(attack_seed),
            'window_start': str(ws), 'window_end': str(we),
            'condition': 'vis_pgd', 'pair_id': pair_id, 'repeat_idx': str(k),
            'reason': reason,
        })
        queue_rows.append({
            'parent_category': cat, 'task_key': task, 'state_id': str(sid),
            'env_seed': str(env_seed), 'attack_seed': str(attack_seed),
            'window_start': str(ws), 'window_end': str(we),
            'condition': 'random_linf', 'pair_id': pair_id, 'repeat_idx': str(k),
            'reason': reason,
        })

queue_path = os.path.join(OUT_DIR, 'k5b_queue.csv')
with open(queue_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(queue_rows[0].keys()))
    w.writeheader()
    w.writerows(queue_rows)
print('Queue: %d parents x %d seeds x 2 = %d jobs' % (len(PARENTS), K, len(queue_rows)))

# Worker scripts
workers = [
    {'name': 'worker_10', 'gpu': '1,0', 'parents': PARENTS[0:6]},    # 6 parents = 60 jobs
    {'name': 'worker_26', 'gpu': '2,6', 'parents': PARENTS[6:11]},   # 5 parents = 50 jobs
    {'name': 'worker_45', 'gpu': '4,5', 'parents': PARENTS[11:16]},  # 5 parents = 50 jobs
]
job_id_base = [510000, 510060, 510110]

for wi, w in enumerate(workers):
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# K5b repeat stability: %s GPU=%s' % (w['name'], w['gpu']))
    n_jobs = len(w['parents']) * K * 2
    lines.append('# %d parents x %d seeds = %d jobs' % (len(w['parents']), K, n_jobs))
    lines.append('set +e')
    lines.append('')
    lines.append('export CUDA_VISIBLE_DEVICES=%s' % w['gpu'])
    lines.append('')
    lines.append('echo "S5_K5B_REPEAT code=0e3428f anchor=d4a3827"')
    lines.append('echo \"[\$(date +%%H:%%M:%%S)] %s K5B START: %d parents, %d jobs\"' % (w['name'], len(w['parents']), n_jobs))
    lines.append('')

    jid = job_id_base[wi]
    for task, sid, env_seed, ws, we, cat, reason in w['parents']:
        for k in range(K):
            attack_seed = k
            pair_id = 'k5b_%s_%s_s%d_w%d_%d_env%d_atk%d' % (cat, task, sid, ws, we, env_seed, attack_seed)
            vis_jid = jid; jid += 1
            rand_jid = jid; jid += 1

            lines.append('echo "=== VIS %d: %s s%d [%d,%d] env=%d atk=%d %s ==="' % (vis_jid, task, sid, ws, we, env_seed, attack_seed, cat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed %d --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "VIS_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, env_seed, env_seed, attack_seed, vis_jid, pair_id, OUT_DIR, vis_jid, pair_id))
            lines.append('')
            lines.append('echo "=== RAND %d: %s s%d [%d,%d] env=%d atk=%d %s ==="' % (rand_jid, task, sid, ws, we, env_seed, attack_seed, cat))
            lines.append('%s -u %s --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed %d --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir %s --image_preprocess official_rot180 || echo "RAND_FAIL %d %s"' % (PY, SCRIPT, task, sid, ws, we, env_seed, env_seed, attack_seed, rand_jid, pair_id, OUT_DIR, rand_jid, pair_id))
            lines.append('')

    lines.append('echo \"[\$(date +%%H:%%M:%%S)] %s K5B DONE\"' % w['name'])
    lines.append('')

    script_path = 'scripts/stageb/run_k5b_%s.sh' % w['name']
    os.makedirs(os.path.dirname(script_path) or '.', exist_ok=True)
    with open(script_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('Generated %s (%d parents -> %d jobs)' % (script_path, len(w['parents']), n_jobs))

print('Job ranges: worker_10=%d-%d  worker_26=%d-%d  worker_45=%d-%d' %
      (510000, 510059, 510060, 510109, 510110, 510159))
print('Total: 160 jobs')
print('Output dir: %s' % OUT_DIR)
