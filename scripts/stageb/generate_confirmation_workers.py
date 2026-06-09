#!/usr/bin/env python3
"""Generate 3 parallel confirmation worker scripts."""
import os

OUT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_confirmation'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
S = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

windows = [
    # Group A: CleanRand-pass -> GPU 1,0
    ('conf_A1_milk_w70_80', 'milk', 0, 70, 80, 'A', '1,0', '10'),
    ('conf_A2_butter_w80_90', 'butter', 0, 80, 90, 'A', '1,0', '10'),
    ('conf_A3_cream_w50_60', 'cream_cheese', 2, 50, 60, 'A', '1,0', '10'),
    ('conf_A4_tomato_w150_160', 'tomato_sauce', 2, 150, 160, 'A', '1,0', '10'),
    # Group B: TaskOnly baseline -> GPU 2,6
    ('conf_B1_tomato_w55_65', 'tomato_sauce', 0, 55, 65, 'B', '2,6', '26'),
    ('conf_B2_milk_w75_85', 'milk', 0, 75, 85, 'B', '2,6', '26'),
    ('conf_B3_cream_w85_95', 'cream_cheese', 0, 85, 95, 'B', '2,6', '26'),
    ('conf_B4_salad_w70_80', 'salad_dressing', 2, 70, 80, 'B', '2,6', '26'),
    # Group C: High-risk abstained -> GPU 4,5
    ('conf_C1_milk_w80_90', 'milk', 0, 80, 90, 'C', '4,5', '45'),
    ('conf_C2_butter_w95_105', 'butter', 0, 95, 105, 'C', '4,5', '45'),
    ('conf_C3_alphabet_w60_70', 'alphabet_soup', 0, 60, 70, 'C', '4,5', '45'),
    ('conf_C4_tomato_w115_125', 'tomato_sauce', 2, 115, 125, 'C', '4,5', '45'),
]

for worker in ['10', '26', '45']:
    w_wins = [w for w in windows if w[7] == worker]
    if not w_wins: continue
    gpu = w_wins[0][6]
    lines = ['#!/bin/bash', 'set +e', 'export CUDA_VISIBLE_DEVICES=%s' % gpu,
             'OUT=%s' % OUT, 'mkdir -p $OUT', 'PY=%s' % PY, 'S=%s' % S, '',
             'echo "[$(date +%%H:%%M:%%S)] CONFIRMATION WORKER_%s START (%d windows)"' % (worker, len(w_wins)), '']
    jid = 700000 + (int(worker) * 100)
    for (pid, task, sid, ws, we, grp, gpu, wrk) in w_wins:
        for atk in [5, 6]:
            lines.append('echo "  VIS %s atk=%d"' % (pid, atk))
            lines.append('$PY -u $S --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL %s atk=%d"' % (task, sid, ws, we, sid, atk, jid, pid, pid, atk))
            jid += 1
            lines.append('echo "  RAND %s atk=%d"' % (pid, atk))
            lines.append('$PY -u $S --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL %s atk=%d"' % (task, sid, ws, we, sid, atk, jid, pid, pid, atk))
            jid += 1
        lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] CONFIRMATION WORKER_%s DONE"' % worker)
    path = 'scripts/stageb/run_confirmation_worker_%s.sh' % worker
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Written: %s (%d windows, %d jobs)' % (path, len(w_wins), len(w_wins)*4))
