#!/usr/bin/env python3
"""Generate worker scripts for pilot12 from candidate queue."""
import csv, sys

O = sys.argv[1] if len(sys.argv) > 1 else '/data/liuyu/outputs/stageb_v1_1_corrected_pilot12_rc1a_20260607'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
R = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'
Q = '/data/liuyu/outputs/stageb_v1_1_corrected_pilot12_queue_v2.csv'

with open(Q) as f:
    rows = list(csv.DictReader(f))

for wi, group in enumerate([rows[0:4], rows[4:8], rows[8:12]], 1):
    gpu = {1: '4,5', 2: '2,6', 3: '1,0'}[wi]
    path = '/tmp/pilot12_w%d.sh' % wi
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        for r in group:
            pair = 'pilot12_%s_s%s_%s_%s' % (r['task_key'], r['state_id'], r['window_start'], r['window_end'])
            jid_vis = 99000 + wi * 100 + int(r['state_id']) * 10 + 1
            jid_rand = 99000 + wi * 100 + int(r['state_id']) * 10 + 2
            f.write('echo "$(date +%%H:%%M:%%S) VIS %s s%s" >> %s/pilot.log\n' % (r['task_key'], r['state_id'], O))
            f.write('CUDA_VISIBLE_DEVICES=%s %s -u %s --task %s --state-id %s --window_start %s --window_end %s --condition vis_pgd --gpu_pair 0,1 --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed %s --pair_id %s --job_id %d --output_dir %s >> %s/w%d.log 2>&1\n' % (gpu, PY, R, r['task_key'], r['state_id'], r['window_start'], r['window_end'], r['seed'], pair, jid_vis, O, O, wi))
            f.write('echo "$(date +%%H:%%M:%%S) RAND %s s%s" >> %s/pilot.log\n' % (r['task_key'], r['state_id'], O))
            f.write('CUDA_VISIBLE_DEVICES=%s %s -u %s --task %s --state-id %s --window_start %s --window_end %s --condition random_linf --gpu_pair 0,1 --eps_raw_pixels 6 --max_steps 400 --seed %s --pair_id %s --job_id %d --output_dir %s >> %s/w%d.log 2>&1\n' % (gpu, PY, R, r['task_key'], r['state_id'], r['window_start'], r['window_end'], r['seed'], pair, jid_rand, O, O, wi))
        f.write('echo "$(date +%%H:%%M:%%S) W%d DONE" >> %s/pilot.log\n' % (wi, O))
    print('W%d: %s (%d windows)' % (wi, path, len(group)))
