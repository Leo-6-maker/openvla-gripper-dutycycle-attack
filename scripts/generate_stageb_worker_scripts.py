#!/usr/bin/env python3
"""Generate per-worker shell driver scripts for Stage-B VIS labeling."""
import csv, os

JOBS_CSV = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/object100_next80_vis_label_jobs.csv'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/run_stageb_vis_labeling.py'
OUT = '/data/liuyu/outputs/overnight_stageb_labels_20260607'
WORKER_GPU = {0: '2,6', 1: '4,5', 2: '1,0'}
WORKER_NAME = {0: 'worker_26', 1: 'worker_45', 2: 'worker_10'}

os.makedirs(OUT, exist_ok=True)

with open(JOBS_CSV) as f:
    jobs = list(csv.DictReader(f))

for w in range(3):
    wj = [j for j in jobs if int(j['worker_id']) == w]
    vis_jobs = [j for j in wj if j['condition'] == 'vis_pgd']
    rand_jobs = {j['paired_vis_job_id']: j for j in wj if j['condition'] == 'random_linf'}

    lines = []
    lines.append('#!/bin/bash')
    lines.append('# %s - GPU %s' % (WORKER_NAME[w], WORKER_GPU[w]))
    lines.append('# %d VIS PGD20 + %d matched random jobs' % (len(vis_jobs), len(rand_jobs)))
    lines.append('set +e  # do not exit on first job failure')
    lines.append('export CUDA_VISIBLE_DEVICES=%s' % WORKER_GPU[w])
    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] %s START: %d paired jobs"' % (WORKER_NAME[w], len(vis_jobs)))

    for j in vis_jobs:
        jid = j['job_id']; task = j['task_key']; sid = j['state_id']
        ws = j['window_start']; we = j['window_end']; stratum = j['stratum']
        lines.append('')
        lines.append('echo "[$(date +%%H:%%M:%%S)] VIS %s: %s s%s [%s,%s] %s"' % (jid, task, sid, ws, we, stratum))
        lines.append('%s -u %s --task %s --state-id %s --window_start %s --window_end %s --condition vis_pgd --gpu_pair 0,1 --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --job_id %s --output_dir %s 2>&1 || echo "VIS_FAIL %s"' % (PY, SCRIPT, task, sid, ws, we, jid, OUT, jid))

        rj = rand_jobs.get(jid)
        if rj:
            rjid = rj['job_id']
            lines.append('echo "[$(date +%%H:%%M:%%S)] RAND %s: %s s%s [%s,%s]"' % (rjid, task, sid, ws, we))
            lines.append('%s -u %s --task %s --state-id %s --window_start %s --window_end %s --condition random_linf --gpu_pair 0,1 --eps_raw_pixels 6 --max_steps 400 --job_id %s --output_dir %s 2>&1 || echo "RAND_FAIL %s"' % (PY, SCRIPT, task, sid, ws, we, rjid, OUT, rjid))

    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] %s DONE"' % WORKER_NAME[w])

    path = '/tmp/run_stageb_%s.sh' % WORKER_NAME[w]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote %s (%d paired jobs)' % (path, len(vis_jobs)))
