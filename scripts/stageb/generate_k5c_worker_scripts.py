#!/usr/bin/env python3
"""Generate K5c worker scripts from queue CSV."""
import csv

QUEUE = 'tables/stageb_v1_1_k5c_queue_rc1a_ca3a97e.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
S = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

# Read queue
with open(QUEUE) as f:
    rows = list(csv.DictReader(f))

# Group by pair_id to get unique parents
parents = {}
for r in rows:
    pid = int(r['pair_id'])
    if pid not in parents:
        parents[pid] = r

# Split across workers
all_pids = sorted(parents.keys())
worker_pids = {
    '10': all_pids[0:6],   # 6 parents = 60 jobs
    '26': all_pids[6:11],  # 5 parents = 50 jobs
    '45': all_pids[11:16], # 5 parents = 50 jobs
}

for worker, pids in worker_pids.items():
    lines = ['#!/bin/bash', 'set +e', '', f'# K5c worker_{worker} — {len(pids)} parents × K=5 = {len(pids)*10} jobs']
    if worker == '10':
        lines.append('export CUDA_VISIBLE_DEVICES=1,0')
    elif worker == '26':
        lines.append('export CUDA_VISIBLE_DEVICES=2,6')
    else:
        lines.append('export CUDA_VISIBLE_DEVICES=4,5')
    lines += [
        f'OUT={OUT}',
        f'mkdir -p $OUT',
        f'PY={PY}',
        f'S={S}',
        '',
        f'echo "[$(date +%H:%M:%S)] K5C WORKER_{worker} START ({len(pids)} parents)"',
        ''
    ]

    for pid in pids:
        p = parents[pid]
        pk = p['parent_key']; task = p['task']; sid = p['state_id']
        env_seed = p['env_seed']; ws = p['window_start']; we = p['window_end']
        cat = p['category']
        lines.append(f'# ── {pk} ({cat}) ──')
        for atk in range(5):  # K=5
            jid_vis = 520000 + pid * 10 + atk * 2
            jid_rand = jid_vis + 1
            lines.append(f'echo "  VIS {pk} atk={atk}"')
            lines.append(f'$PY -u $S --gpu_pair 0,1 --task {task} --state-id {sid} --window_start {ws} --window_end {we} --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed {env_seed} --attack_seed {atk} --job_id {jid_vis} --pair_id {pk} --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL {pk} atk={atk}"')
            lines.append(f'echo "  RAND {pk} atk={atk}"')
            lines.append(f'$PY -u $S --gpu_pair 0,1 --task {task} --state-id {sid} --window_start {ws} --window_end {we} --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed {env_seed} --attack_seed {atk} --job_id {jid_rand} --pair_id {pk} --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL {pk} atk={atk}"')
        lines.append('')

    lines.append(f'echo "[$(date +%H:%M:%S)] K5C WORKER_{worker} DONE"')

    script = '\n'.join(lines) + '\n'
    path = f'scripts/stageb/run_k5c_worker_{worker}.sh'
    with open(path, 'w') as f:
        f.write(script)
    print(f'Written: {path} ({len(pids)} parents, {len(pids)*10} jobs)')
