#!/usr/bin/env python3
"""Generate per-worker shell scripts for targeted expansion queue.

FIXED: globally unique job_ids across all workers.
Each parent gets unique VIS+RAND with shared seed and pair_id.

Usage:
  python scripts/stageb/generate_expansion_worker_scripts.py \
    --queue /path/to/expansion_queue.csv \
    --output-dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827
"""
import csv, os, sys, argparse

PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

WORKERS = [
    {'name': 'worker_10', 'gpu': '1,0', 'pair': '0,1'},
    {'name': 'worker_26', 'gpu': '2,6', 'pair': '0,1'},
    {'name': 'worker_45', 'gpu': '4,5', 'pair': '0,1'},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--job-id-base', type=int, default=300000,
                    help='Starting job_id (globally unique range)')
    ap.add_argument('--script-dir', default='scripts/stageb')
    args = ap.parse_args()

    # Load queue
    with open(args.queue, 'r') as f:
        queue = list(csv.DictReader(f))
    print('Queue: %d parents' % len(queue))

    # Round-robin assign parents to workers
    worker_parents = {w['name']: [] for w in WORKERS}
    for i, row in enumerate(queue):
        w = WORKERS[i % len(WORKERS)]
        worker_parents[w['name']].append(row)

    job_id = args.job_id_base

    for w in WORKERS:
        parents = worker_parents[w['name']]
        lines = []
        lines.append('#!/bin/bash')
        lines.append('# Expansion worker: %s GPU=%s' % (w['name'], w['gpu']))
        lines.append('# %d parents, %d jobs (job_id range %d-%d)' %
                     (len(parents), len(parents) * 2, job_id, job_id + len(parents) * 2 - 1))
        lines.append('set +e')
        lines.append('')
        lines.append('export CUDA_VISIBLE_DEVICES=%s' % w['gpu'])
        lines.append('')
        lines.append('echo "[$(date +%%H:%%M:%%S)] %s EXPANSION START: %d parents"' %
                     (w['name'], len(parents)))
        lines.append('')

        for row in parents:
            task = row['task_key']
            sid = int(row['state_id'])
            seed = int(row['seed'])
            ws = int(row['window_start'])
            we = int(row['window_end'])
            cat = row['category']
            max_s = int(row.get('actual_max_step', '400'))
            pair_id = 'exp_%s_%s_s%d_w%d_%d_seed%d' % (cat, task, sid, ws, we, seed)

            vis_jid = job_id; job_id += 1
            rand_jid = job_id; job_id += 1

            lines.append('echo "=== VIS %d: %s s%d [%d,%d] seed=%d %s ==="' %
                         (vis_jid, task, sid, ws, we, seed, cat))
            lines.append('%s -u %s \\' % (PY, SCRIPT))
            lines.append('  --gpu_pair %s \\' % w['pair'])
            lines.append('  --task %s --state-id %d --window_start %d --window_end %d \\' %
                         (task, sid, ws, we))
            lines.append('  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \\')
            lines.append('  --max_steps %d --seed %d \\' % (max_s, seed))
            lines.append('  --job_id %d --pair_id %s \\' % (vis_jid, pair_id))
            lines.append('  --output_dir %s \\' % args.output_dir)
            lines.append('  --image_preprocess official_rot180 \\')
            lines.append('  || echo "VIS_FAIL %d %s"' % (vis_jid, pair_id))
            lines.append('')

            lines.append('echo "=== RAND %d: %s s%d [%d,%d] seed=%d %s ==="' %
                         (rand_jid, task, sid, ws, we, seed, cat))
            lines.append('%s -u %s \\' % (PY, SCRIPT))
            lines.append('  --gpu_pair %s \\' % w['pair'])
            lines.append('  --task %s --state-id %d --window_start %d --window_end %d \\' %
                         (task, sid, ws, we))
            lines.append('  --condition random_linf --eps_raw_pixels 6 \\')
            lines.append('  --max_steps %d --seed %d \\' % (max_s, seed))
            lines.append('  --job_id %d --pair_id %s \\' % (rand_jid, pair_id))
            lines.append('  --output_dir %s \\' % args.output_dir)
            lines.append('  --image_preprocess official_rot180 \\')
            lines.append('  || echo "RAND_FAIL %d %s"' % (rand_jid, pair_id))
            lines.append('')

        lines.append('echo "[$(date +%%H:%%M:%%S)] %s EXPANSION DONE"' % w['name'])
        lines.append('')

        script_path = os.path.join(args.script_dir, 'run_expansion_%s.sh' % w['name'])
        os.makedirs(os.path.dirname(script_path) or '.', exist_ok=True)
        with open(script_path, 'w', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('Generated %s (%d parents -> %d jobs)' %
              (script_path, len(parents), len(parents) * 2))

    print('\nTotal: %d jobs (job_ids %d-%d)' %
          (len(queue) * 2, args.job_id_base, job_id - 1))
    print('Output dir: %s' % args.output_dir)


if __name__ == '__main__':
    main()
