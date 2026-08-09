"""Launch formal V2.3 N4 12-split training. Waits for all to complete.

Usage: python launch_v23_formal.py
"""
import subprocess, os, sys, time, json, re

PY = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
SPLITS = 'o0_i0 o0_i1 o0_i2 o1_i0 o1_i1 o1_i2 o2_i0 o2_i1 o2_i2 o3_i0 o3_i1 o3_i2'.split()
GPU_MAP = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 7]
EPOCHS = 20  # Match N4 ceiling recipe exactly

BASE_ARGS = (
    '--gpu 0'
    ' --feat-root /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
    ' --label-roots /tmp/ft_FIT_TRAIN/labels,/tmp/ft_FIT_DEV/labels,/tmp/ft_CAL/labels,/tmp/ft_CHECK/labels,/tmp/ft_H/labels,/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
    ' --split-manifest /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'
    ' --dev2-manifest /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'
    ' --out-root /mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1'
    f' --epochs {EPOCHS}'
)
TRAINER = '/tmp/train_v23_split.py'
LOG_DIR = '/tmp/v23_formal_logs'
os.makedirs(LOG_DIR, exist_ok=True)

print(f'=== FORMAL V2.3 N4 12-SPLIT TRAINING ===')
print(f'Epochs: {EPOCHS}')
print(f'Trainer: {TRAINER}')
print(f'Logs: {LOG_DIR}/')
print()

procs = []
for i, sn in enumerate(SPLITS):
    gpu = GPU_MAP[i]
    log_path = os.path.join(LOG_DIR, f'{sn}.log')
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} {PY} -u {TRAINER} --split-name {sn} {BASE_ARGS} > {log_path} 2>&1'
    p = subprocess.Popen(cmd, shell=True, start_new_session=True)
    procs.append((sn, gpu, p, log_path))
    print(f'  [{i+1:2d}/12] {sn} -> GPU {gpu} (PID {p.pid})')
    time.sleep(0.5)

print(f'\nLaunched {len(procs)} workers. Waiting for completion...')
print(f'Monitor: tail -f {LOG_DIR}/*.log')
sys.stdout.flush()

# Write PID manifest
pid_path = os.path.join(LOG_DIR, 'PIDS.txt')
with open(pid_path, 'w') as f:
    for sn, gpu, p, log in procs:
        f.write(f'{p.pid}\t{sn}\tGPU{gpu}\t{log}\n')

# Wait for all to complete
results = {}
for sn, gpu, p, log_path in procs:
    exit_code = p.wait()
    # Parse last JSON line from log
    try:
        with open(log_path) as f:
            lines = f.read().strip().splitlines()
            last = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{'):
                    last = json.loads(line)
                    break
            results[sn] = last or {'split': sn, 'status': 'FAIL_PARSE_LOG'}
    except Exception as e:
        results[sn] = {'split': sn, 'status': 'FAIL_READ_LOG', 'error': str(e)}

    status = results[sn].get('status', '?')
    auprc = results[sn].get('best_ep_auprc', '?')
    print(f'  {sn}: {status}  auprc={auprc}  (exit={exit_code})')
    sys.stdout.flush()

# Summary
passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
all_pass = passed == 12
aucs = [r['best_ep_auc'] for r in results.values() if r.get('status') == 'PASS']
auprcs = [r['best_ep_auprc'] for r in results.values() if r.get('status') == 'PASS']

print(f'\n{"="*50}')
print(f'FORMAL 12-SPLIT: {passed}/12 PASS')
if aucs:
    print(f'AUROC: mean={sum(aucs)/len(aucs):.4f}  min={min(aucs):.4f}  max={max(aucs):.4f}')
    print(f'AUPRC: mean={sum(auprcs)/len(auprcs):.4f}  min={min(auprcs):.4f}  max={max(auprcs):.4f}')
print(f'STATUS: {"PASS" if all_pass else "FAIL"}')

# Write summary
import json as _json
summary_path = os.path.join(LOG_DIR, 'SUMMARY.json')
with open(summary_path, 'w') as f:
    _json.dump({'status': 'PASS' if all_pass else 'FAIL', 'passed': passed, 'total': 12,
                'mean_auroc': sum(aucs)/len(aucs) if aucs else 0,
                'mean_auprc': sum(auprcs)/len(auprcs) if auprcs else 0,
                'results': results}, f, indent=2)
print(f'Summary: {summary_path}')
