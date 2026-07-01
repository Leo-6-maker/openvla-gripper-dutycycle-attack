#!/usr/bin/env python3
"""SOTA Worker — passes all manifest fields to the bridge (attack_objective, arm_lock, etc.).

Extends run_vis_formal_worker_v2.py with attack_objective, arm_lock, keep_running support.
Used for: TMA, UMA, SHUFFLED, Adapted FreezeVLA, and any condition needing custom objectives.
"""
import json, os, subprocess, sys, time

if len(sys.argv) < 2:
    print("Usage: run_sota_worker.py <gpu_id> [manifest_path]")
    sys.exit(1)

GPU = int(sys.argv[1])
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else None
if MANIFEST is None:
    print("ERROR: manifest path required")
    sys.exit(1)

EVID = '/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1'

# Preload heldout anchors for all folds
anchors_cache = {}
for fold_id in ['01','02','03','04','05','06','07','08','09']:
    with open(EVID + '/fold_' + fold_id + '/FOLD' + fold_id + '_heldout_episode_anchors.json') as f:
        data = json.load(f)
    for ep in data['episodes']:
        anchors_cache[(fold_id, ep['state_id'])] = ep['sc5_anchor']

jobs = []
with open(MANIFEST) as f:
    for line in f:
        if line.strip():
            jobs.append(json.loads(line))

print('GPU {}: {} jobs'.format(GPU, len(jobs)))

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(GPU)
env['TMPDIR'] = EVID + '/tmp'
env['OPENVLA_MODEL_PATH'] = '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object'
env['OPENVLA_DTYPE'] = 'bfloat16'
env['OPENVLA_ATTN_IMPLEMENTATION'] = 'eager'
env['OMP_NUM_THREADS'] = '1'
env['MKL_NUM_THREADS'] = '1'
env['OPENBLAS_NUM_THREADS'] = '1'
env['NUMEXPR_NUM_THREADS'] = '1'

# Use standard VIS bridge for TMA/UMA/SHUFFLED attacks
BRIDGE = '/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py'
PYTHON = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'

def resolve_condition(job):
    cid = job.get('condition_id')
    cond = job.get('condition')
    if cid is not None and cond is not None and cid != cond:
        raise SystemExit('SCHEMA_CONFLICT: condition_id=%r != condition=%r' % (cid, cond))
    if cid is not None:
        return cid
    if cond is not None:
        return cond
    raise SystemExit('MISSING_CONDITION in job %s' % job.get('job_key', '?'))

t0_total = time.time()
failed_jobs = []
for idx, job in enumerate(jobs):
    os.makedirs(job['output_dir'], exist_ok=True)
    cond = resolve_condition(job)
    fold = str(job['fold'])
    state = int(job['state_id'])
    det_seed = int(job['detector_seed'])
    task_id = int(job.get('task_id', 0))
    pert_seed = int(job['perturbation_seed'])
    mlp_path = EVID + '/fold_' + fold + '/training_v3/seed_' + str(det_seed) + '/best_model.pt'
    trigger_step = job.get("trigger_step_override")
    if trigger_step is not None and int(trigger_step) >= 0:
        trigger_override = int(trigger_step)
    else:
        trigger_override = -1
    anchor = anchors_cache.get((fold, state), -1)

    # Key: pass attack_objective from manifest if present
    attack_obj = job.get("attack_objective", "autoregressive_prefix_gripper_target_token_logratio_arm_v3")
    # arm_lock support
    arm_lock = job.get("arm_lock", False)
    # keep_running for random-time conditions
    keep_running = job.get("keep_running", False)

    cmd = [PYTHON, '-u', BRIDGE,
           '--condition', cond,
           '--state_id', str(state),
           '--anchor', str(anchor),
           '--trigger_step_override', str(trigger_override),
           '--attack_objective', str(attack_obj) if attack_obj else 'autoregressive_prefix_gripper_target_token_logratio_arm_v3',
           '--seed_id', str(pert_seed),
           '--task_idx', str(task_id),
           '--mlp_path', mlp_path,
           '--render_gpu', str(GPU),
           '--output_dir', job['output_dir']]
    if arm_lock:
        cmd.append('--arm_lock')
    if keep_running:
        cmd.append('--keep_running')

    label = '[{}/{}] {}'.format(idx+1, len(jobs), job.get('job_key', '?'))
    print('{} START'.format(label))
    t0 = time.time()

    stdout_path = os.path.join(job['output_dir'], 'stdout.log')
    stderr_path = os.path.join(job['output_dir'], 'stderr.log')
    with open(stdout_path, 'w') as out_f, open(stderr_path, 'w') as err_f:
        proc = subprocess.run(cmd, env=env, stdout=out_f, stderr=err_f)

    elapsed = time.time() - t0
    if proc.returncode == 0:
        print('{} COMPLETE ({:.0f}s)'.format(label, elapsed))
    else:
        print('{} FAILED exit={} ({:.0f}s)'.format(label, proc.returncode, elapsed))
        failed_jobs.append({"job_key": job.get('job_key', '?'), "exit_code": proc.returncode,
                            "output_dir": job['output_dir'], "elapsed_s": int(elapsed)})

# Write failure ledger
if failed_jobs:
    with open(os.path.join(os.path.dirname(MANIFEST) if MANIFEST else '/tmp',
              'failure_ledger_gpu{}.json'.format(GPU)), 'w') as f:
        json.dump({"gpu": GPU, "manifest": MANIFEST, "n_total": len(jobs),
                   "n_failed": len(failed_jobs), "failed_jobs": failed_jobs}, f, indent=2)

print('GPU {} DONE: {} jobs in {:.0f}s, {} FAILED'.format(
    GPU, len(jobs), time.time() - t0_total, len(failed_jobs)))
if failed_jobs:
    sys.exit(1)
