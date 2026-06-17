#!/usr/bin/env python3
"""V6 Phase 1: Clean trigger pilot worker. Runs clean_observer ×2 per parent."""
import os, subprocess, sys, time

GPU = sys.argv[1]  # GPU pair like "0,1"
RENDER = GPU.split(',')[0]
OUT = sys.argv[2]
PARENTS = sys.argv[3:]  # task,state_id pairs

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = f'{REPO}/scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

os.makedirs(OUT, exist_ok=True)

for i in range(0, len(PARENTS), 2):
    task = PARENTS[i]; state_id = PARENTS[i+1]
    for rep in range(2):
        tag = '%s_s%s_rep%d' % (task, state_id, rep)
        cmd = [PY, '-u', RUNNER,
            '--task', task, '--state_id', state_id,
            '--condition', 'clean_observer',
            '--attack_seed', '0', '--seed', '0',
            '--model_path', MODEL,
            '--render_gpu_device_id', RENDER, '--model_gpu_device_id', '-1',
            '--output_dir', OUT, '--job_id', '990%02d' % (i+rep),
            '--max_steps_override', '280', '--success_metric', 'check_success',
            '--num_steps_wait', '10',
        ]
        print('[%s] %s' % (time.strftime('%H:%M:%S'), tag), flush=True)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU}
        result = subprocess.run(cmd, env=env, cwd=REPO, timeout=600)
        print('[%s] %s done (exit=%d)' % (time.strftime('%H:%M:%S'), tag, result.returncode), flush=True)

print('[%s] Worker done' % time.strftime('%H:%M:%S'), flush=True)
