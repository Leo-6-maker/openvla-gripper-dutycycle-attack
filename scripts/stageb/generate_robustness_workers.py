#!/usr/bin/env python3
"""Generate 3 robustness workers for Phase 1A+1B."""
OUT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_robustness_seed78'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
S = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'

jobs = [
    ('rob_A1_milk_w70_80','milk',0,70,80,'1,0','10'),
    ('rob_A2_butter_w80_90','butter',0,80,90,'1,0','10'),
    ('rob_A3_cream_w50_60','cream_cheese',2,50,60,'2,6','26'),
    ('rob_A4_tomato_w150_160','tomato_sauce',2,150,160,'2,6','26'),
    ('rob_FP_tomato_w55_65','tomato_sauce',0,55,65,'4,5','45'),
    ('rob_FN_salad_w70_80','salad_dressing',2,70,80,'4,5','45'),
]

for worker in ['10','26','45']:
    w_jobs = [j for j in jobs if j[6]==worker]
    if not w_jobs: continue
    gpu = w_jobs[0][5]
    lines = ['#!/bin/bash','set -e','export CUDA_VISIBLE_DEVICES=%s'%gpu,
             'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
             'OUT=%s'%OUT,'mkdir -p $OUT','PY=%s'%PY,'S=%s'%S,'',
             'echo "[$(date +%%H:%%M:%%S)] ROBUSTNESS WORKER_%s START"'%worker,'']
    jid = 704000 + int(worker)*10
    for pid,task,sid,ws,we,gpu,wrk in w_jobs:
        for atk in [7,8]:
            for cond in ['vis_pgd','random_linf']:
                lines.append('echo "  %s %s atk=%d"' % (cond, pid, atk))
                lines.append('$PY -u $S --gpu_pair 0,1 --task %s --state-id %d --window_start %d --window_end %d --condition %s --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed %d --attack_seed %d --job_id %d --pair_id %s --output_dir $OUT --image_preprocess official_rot180 || exit 1' % (task,sid,ws,we,cond,sid,atk,jid,pid))
                jid += 1
        lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] ROBUSTNESS WORKER_%s DONE"'%worker)
    with open('scripts/stageb/run_robustness_worker_%s.sh'%worker,'w') as f:
        f.write('\n'.join(lines)+'\n')
    print('Worker %s: %d jobs, GPU %s' % (worker, len(w_jobs)*4, gpu))
