#!/usr/bin/env python3
"""Generate Layer-2 confirmation worker scripts from queue CSV."""
import csv, os, hashlib, subprocess, time

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
QUEUE = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue.csv')
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation'
os.makedirs(OUT_DIR, exist_ok=True)

with open(QUEUE) as f:
    rows = list(csv.DictReader(f))

for group_name, gpu_pair_env, gpu_pair_arg, script_name in [
    ('H_HiddenSafeRank', '1,0', '0,1', 'run_l2_confirm_group_h.sh'),
    ('B_RandomRank', '4,5', '0,1', 'run_l2_confirm_group_b.sh'),
]:
    group_rows = [r for r in rows if r['queue_group'] == group_name]
    pairs = {}
    for r in group_rows:
        lp = r['logical_pair_key']
        pairs.setdefault(lp, {})[r['condition']] = r

    lines = ['#!/bin/bash', 'set +e',
             'export CUDA_VISIBLE_DEVICES=%s' % gpu_pair_env,
             'OUT=%s' % OUT_DIR, 'mkdir -p $OUT',
             'PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python',
             'S=%s/scripts/run_stageb_vis_labeling.py' % REPO,
             '',
             'echo "[$(date +%%H:%%M:%%S)] L2_CONFIRM_%s START (%d windows)"' % (group_name, len(pairs)),
             '']

    job_id = 800000 if 'H_' in group_name else 810000
    for lp, cond_rows in pairs.items():
        vis_r = cond_rows['VIS']; rand_r = cond_rows['RAND']
        task = vis_r['task']; sid = vis_r['state_id']
        ws = vis_r['window_start']; we = vis_r['window_end']
        atk = vis_r['attack_seed']; pair_id = lp
        env_seed = sid

        for cond_name, cond_flag, cond_r in [('VIS', 'vis_pgd', vis_r), ('RAND', 'random_linf', rand_r)]:
            cmd = ('$PY -u $S --gpu_pair %s --task %s --state-id %s '
                   '--window_start %s --window_end %s --condition %s '
                   '--pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 '
                   '--seed 0 --env_seed %s --attack_seed %s '
                   '--job_id %d --pair_id %s --output_dir $OUT '
                   '--image_preprocess official_rot180 '
                   '|| echo "%s_FAIL %s atk=%s"') % (
                gpu_pair_arg, task, sid, ws, we, cond_flag,
                env_seed, atk, job_id, pair_id,
                cond_name, pair_id, atk)
            lines.append('echo "  %s %s atk=%s"' % (cond_name, pair_id, atk))
            lines.append(cmd)
            job_id += 1

    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] L2_CONFIRM_%s DONE"' % group_name)

    script_path = os.path.join(REPO, 'scripts/stageb', script_name)
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(script_path, 0o755)
    print('Written: %s (%d jobs, %d pairs)' % (script_path, len(group_rows), len(pairs)))

# Runtime manifest
git_head = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
queue_hash = hashlib.sha256(open(QUEUE, 'rb').read()).hexdigest()
manifest = os.path.join(OUT_DIR, 'runtime_manifest.txt')
with open(manifest, 'w') as f:
    f.write('runtime_git_head = %s\n' % git_head)
    f.write('github_merge_head = f49b3f6\n')
    f.write('queue_csv_sha256 = %s\n' % queue_hash)
    f.write('launch_time = %s\n' % time.strftime('%Y-%m-%dT%H:%M:%S'))
    f.write('total_jobs = 32\n')
    f.write('unique_windows = 8\n')
    f.write('logical_pairs = 16\n')
    f.write('gpu_group_h = 1,0\n')
    f.write('gpu_group_b = 4,5\n')
    f.write('gpu_reserve = 2,6\n')
    f.write('gpu_blacklist = 3,7\n')
print('Manifest: %s' % manifest)
print('git_head: %s' % git_head)
print('queue_sha256: %s' % queue_hash)
