#!/usr/bin/env python3
"""V5 TokenPrefixPGD full rollout canary: 3 jobs on GPU 4,5.
pgd_steps=10 (matching window smoke). No seed100/101."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

QUEUE_CSV = sys.argv[1] if len(sys.argv) > 1 else ''
GPU = sys.argv[2] if len(sys.argv) > 2 else '4,5'
RENDER = sys.argv[3] if len(sys.argv) > 3 else '4'

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

CANARY_JOBS = [
    {'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': '0',
     'window_start': '125', 'window_end': '135', 'attack_seed': '99',
     'priority': 'P0', 'purpose': 'transport_clean_close_to_adv_open',
     'rand_source': 'S20M4a', 'rand_opens': '0/2/0'},
    {'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '80', 'window_end': '90', 'attack_seed': '93',
     'priority': 'P0_CANARY', 'purpose': 'historical_canary_restored',
     'rand_source': 'S20M3a/M3b', 'rand_opens': '0/0/0'},
    {'candidate_id': 'cream_cheese_s2_w75_85', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': '75', 'window_end': '85', 'attack_seed': '99',
     'priority': 'P1_CONTRAST', 'purpose': 'nearby_window_position_sensitivity',
     'rand_source': 'S20M4a', 'rand_opens': '1/0/1'},
]

OUT = '/data/liuyu/outputs/stageb_s20d_v5_canary_rollout_20260613'
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT+'/queues', exist_ok=True)

ENV = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'OPENVLA_ATTN_IMPLEMENTATION': 'eager', 'CUDA_VISIBLE_DEVICES': GPU, 'DISPLAY': ''}

# Write manifest
T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(T, exist_ok=True)
with open(T+'/s20d_v5_token_pgd_full_rollout_canary_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['job_id','candidate_id','task','state_id','window_start','window_end',
        'attack_seed','pgd_steps','eps_raw_pixels','priority','purpose','rand_source','rand_opens','runner'])
    w.writeheader()
    for i, j in enumerate(CANARY_JOBS):
        w.writerow({'job_id': str(350100+i), **j, 'pgd_steps': '10', 'eps_raw_pixels': '6',
                     'runner': 's20d_v5_token_pgd_fixed_window_l3_runner'})

# Build queue for worker
jobs = []
for i, j in enumerate(CANARY_JOBS):
    jid = 350100 + i
    jobs.append({
        'job_id': str(jid), 'candidate_id': j['candidate_id'], 'task': j['task'],
        'state_id': j['state_id'], 'window_start': j['window_start'], 'window_end': j['window_end'],
        'condition': 'vis_pgd', 'attack_seed': j['attack_seed'], 'pgd_steps': '10',
        'eps_raw_pixels': '6', 'random_control_seed': '', 'seed': '0',
        'priority': j['priority'], 'purpose': j['purpose'],
        'tier': 'V5_CANARY', 'track': 'S20D_V5_CANARY', 'status': 'pending',
        'output_dir': OUT,
    })

qp = OUT+'/queues/s20d_v5_canary_gpu4.csv'
with open(qp, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)

if not QUEUE_CSV:
    QUEUE_CSV = qp

# Run jobs sequentially
print('[%s] V5 CANARY: %d jobs (GPU %s)' % (datetime.now().strftime('%H:%M:%S'), len(jobs), GPU))
print('[%s] Runner: %s' % (datetime.now().strftime('%H:%M:%S'), RUNNER))
print('[%s] Output: %s' % (datetime.now().strftime('%H:%M:%S'), OUT))

for i, j in enumerate(jobs):
    tag = '%s_s%s_w%s_%s_%s_seed%s' % (j['task'], j['state_id'],
        j['window_start'], j['window_end'], 'vis_pgd', j['attack_seed'])
    vid = os.path.join(OUT, 'videos', tag)
    cmd = [PY, '-u', RUNNER, '--task', j['task'], '--state_ids', j['state_id'],
           '--condition', 'vis_pgd', '--window_start', j['window_start'],
           '--window_end', j['window_end'], '--max_steps_override', '280',
           '--success_metric', 'check_success', '--num_steps_wait', '10',
           '--model_path', MODEL, '--render_gpu_device_id', RENDER,
           '--model_gpu_device_id', '-1', '--output_dir', OUT,
           '--save_video_dir', vid, '--job_id', j['job_id'], '--seed', '0',
           '--eps_raw_pixels', j['eps_raw_pixels'], '--attack_seed', j['attack_seed'],
           '--pgd_steps', j['pgd_steps']]

    print('[%s] V5_CANARY %d/%d: %s [%s]' % (datetime.now().strftime('%H:%M:%S'),
          i+1, len(jobs), tag, j['priority']))
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV)
    dt = time.time() - t0
    status = 'done' if r.returncode == 0 else 'failed'
    j['status'] = status; j['runtime_sec'] = str(round(dt, 1))

    # Quick audit
    if status == 'done':
        sf = list(Path(OUT).glob('summary_*vis_pgd_seed%s_job%s.json' % (j['attack_seed'], j['job_id'])))
        if sf:
            s = json.load(open(sf[0]))
            print('[%s]   open=%d streak=%d done=%s timeout=%s steps=%d infra=%s' %
                  (datetime.now().strftime('%H:%M:%S'),
                   s['decoded_open_count'], s['max_open_streak'],
                   s['success_done_any'], s.get('timeout', False),
                   s['n_steps'], s.get('infra_status', '?')))
            am = s.get('attack_method','?')
            pgd = s.get('v5_pgd_applied', s.get('pgd_applied','?'))
            print('[%s]   attack_method=%s pgd_applied=%s adv_decode=%s' %
                  (datetime.now().strftime('%H:%M:%S'), am, pgd, s.get('adv_decode_path','?')))
        else:
            print('[%s]   no summary found' % datetime.now().strftime('%H:%M:%S'))
    else:
        print('[%s]   FAIL (%.0fs)' % (datetime.now().strftime('%H:%M:%S'), dt))

    with open(qp, 'w', newline='') as f:
        fields = list(jobs[0].keys())
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(jobs)

done = sum(1 for j in jobs if j.get('status') == 'done')
print('[%s] DONE: %d/%d' % (datetime.now().strftime('%H:%M:%S'), done, len(jobs)))
