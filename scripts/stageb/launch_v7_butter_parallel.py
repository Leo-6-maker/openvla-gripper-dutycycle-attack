#!/usr/bin/env python3
"""V7 butter_s2 mini-replication parallel launcher.

Usage:
  python launch_v7_butter_parallel.py \
    --repo /path/to/repo \
    --output-root /path/to/output \
    --model /path/to/model
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time


def run_job(gpu_pair, cuda_vis, render_gpu, job_id, task, state_id, condition,
            attack_seed, output_dir, repo, model_path, py):
    """Run one rollout job. Returns (job_id, rc, stdout_tail, stderr_tail)."""
    runner = os.path.join(repo, 'scripts', 'stageb',
                          'run_s20d_v6_online_trigger_l3_runner.py')
    cmd = [
        py, '-u', runner,
        '--task', task,
        '--state_id', str(state_id),
        '--condition', condition,
        '--max_steps_override', '280',
        '--success_metric', 'check_success',
        '--num_steps_wait', '10',
        '--model_path', model_path,
        '--render_gpu_device_id', str(render_gpu),
        '--model_gpu_device_id', '-1',
        '--output_dir', output_dir,
        '--job_id', job_id,
        '--seed', '0',
        '--attack_seed', str(attack_seed),
        '--eps_raw_pixels', '6',
        '--pgd_steps', '20',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_vis
    env['OPENVLA_ATTN_IMPLEMENTATION'] = 'eager'
    env['OPENVLA_CUDA_MAX_MEMORY'] = '10000MiB'

    log_path = os.path.join(output_dir, f'log_{job_id}.txt')
    with open(log_path, 'w') as log:
        log.write(f'CMD: {" ".join(cmd)}\n')
        log.write(f'CUDA_VISIBLE_DEVICES={cuda_vis}\n')
        log.write(f'START: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        log.flush()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=900)
        log.write(r.stdout)
        if r.stderr:
            log.write('\nSTDERR:\n')
            log.write(r.stderr[-3000:])
        log.write(f'\nRC={r.returncode}\n')
        log.write(f'END: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    return job_id, r.returncode, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    py = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'

    # GPU pair config
    pairs = {
        'GPU10': {'cuda': '0,1', 'render': 1},
        'GPU26': {'cuda': '2,6', 'render': 2},
        'GPU45': {'cuda': '4,5', 'render': 4},
    }

    # Mini-replication manifest (order within each pair matters)
    jobs = [
        # GPU10: clean rep0 → RAND 401 → VIS 401
        ('GPU10', 'v7_mini_clean_rep0_gpu10', 'butter', 2, 'clean_observer', 0),
        ('GPU10', 'v7_mini_rand_s401_gpu10', 'butter', 2, 'online_random_linf', 401),
        ('GPU10', 'v7_mini_vis_s401_gpu10', 'butter', 2, 'online_vis_pgd', 401),
        # GPU26: VIS 402 → RAND 402 → clean rep1
        ('GPU26', 'v7_mini_vis_s402_gpu26', 'butter', 2, 'online_vis_pgd', 402),
        ('GPU26', 'v7_mini_rand_s402_gpu26', 'butter', 2, 'online_random_linf', 402),
        ('GPU26', 'v7_mini_clean_rep1_gpu26', 'butter', 2, 'clean_observer', 0),
        # GPU45: RAND 403 → VIS 403
        ('GPU45', 'v7_mini_rand_s403_gpu45', 'butter', 2, 'online_random_linf', 403),
        ('GPU45', 'v7_mini_vis_s403_gpu45', 'butter', 2, 'online_vis_pgd', 403),
    ]

    os.makedirs(args.output_root, exist_ok=True)
    # Create per-pair output dirs
    for pair_name in pairs:
        os.makedirs(os.path.join(args.output_root, pair_name), exist_ok=True)

    results = {}

    def run_pair(pair_name, pair_jobs):
        cfg = pairs[pair_name]
        for job_id, task, state_id, condition, attack_seed in pair_jobs:
            pair_out = os.path.join(args.output_root, pair_name)
            jid, rc, log = run_job(
                pair_name, cfg['cuda'], cfg['render'],
                job_id, task, state_id, condition, attack_seed,
                pair_out, args.repo, args.model, py)
            results[jid] = rc
            if rc != 0:
                # Check if retry needed (CUDA OOM, Xid, etc.)
                print(f'FAIL {jid} RC={rc} — see {log}')
                break  # Stop this pair on error
            print(f'PASS {jid}')

    # Group jobs by pair
    pair_jobs = {}
    for pair_name, job_id, task, sid, cond, seed in jobs:
        pair_jobs.setdefault(pair_name, []).append(
            (job_id, task, sid, cond, seed))

    threads = []
    for pair_name in ['GPU10', 'GPU26', 'GPU45']:
        t = threading.Thread(target=run_pair, args=(pair_name, pair_jobs[pair_name]))
        t.start()
        threads.append(t)
        print(f'Started {pair_name} ({len(pair_jobs[pair_name])} jobs)')

    for t in threads:
        t.join()

    # Summary
    n_pass = sum(1 for rc in results.values() if rc == 0)
    n_fail = sum(1 for rc in results.values() if rc != 0)
    print(f'\n=== MINI DONE: {n_pass} PASS, {n_fail} FAIL ===')
    for jid, rc in sorted(results.items()):
        print(f'  {jid}: RC={rc}')


if __name__ == '__main__':
    main()
