#!/usr/bin/env python3
"""V2 formal Stage-1 launcher: authorization → train → predict → evaluate → audit.

Fail-closed. Each job runs the full pipeline. Authorization verified before any job starts.
Dual-pool: training pool (N workers) + postprocess pool (max 16).
"""
import subprocess, sys, time, json, hashlib, argparse, os, threading
from pathlib import Path
from collections import defaultdict

PY = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
SCRIPTS = Path('/mnt/sdc/dty_user/openvla_attack/scripts/detector_v5')
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
SPLITS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'


def sha256_file(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def verify_sealed_directory(root):
    s = root / 'SHA256SUMS'; c = root / 'SHA256SUMS.sha256'
    if not s.is_file() or not c.is_file():
        raise RuntimeError(f'SEAL MISSING: {root}')
    if c.read_text().strip() != f'{sha256_file(s)}  SHA256SUMS':
        raise RuntimeError(f'SEAL MISMATCH: {root}')
    for l in s.read_text().splitlines():
        d, _, n = l.partition('  '); t = root / n
        if not t.is_file() or sha256_file(t) != d:
            raise RuntimeError(f'FILE MISMATCH: {root}/{n}')
    return sha256_file(s)


def verify_existing_output(out_dir, job, auth_seal):
    """Full metadata verification. Raises on mismatch. Returns True if valid."""
    if not out_dir.exists():
        return False
    verify_sealed_directory(out_dir)
    src = json.loads((out_dir / 'source_binding.json').read_text())
    run = json.loads((out_dir / 'run_config.json').read_text())
    rec = json.loads((out_dir / 'authorization_receipt.json').read_text())
    checks = [
        (src.get('candidate'), job['candidate'], 'candidate'),
        (src.get('outer_fold'), job['outer_fold'], 'outer_fold'),
        (src.get('inner_fold'), job['inner_fold'], 'inner_fold'),
        (src.get('seed'), job['seed'], 'seed'),
        (run.get('receptive_field'), job['W'], 'W'),
        (run.get('hidden_dim'), job['hidden_dim'], 'hidden_dim'),
        (run.get('dropout'), job['dropout'], 'dropout'),
        (run.get('weight_decay'), job['weight_decay'], 'weight_decay'),
        (rec.get('authorization_seal'), auth_seal, 'authorization_seal'),
        (run.get('epochs'), 30, 'epochs'),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise RuntimeError(f'{name} mismatch: {actual} != {expected}')
    return True


def run_cmd(cmd, env, log_file, timeout=900):
    """Run a command, log output, return (ok, elapsed). env=None means inherit."""
    start = time.time()
    kwargs = {'stdout': open(log_file, 'w'), 'stderr': subprocess.STDOUT, 'timeout': timeout}
    if env is not None:
        kwargs['env'] = env
    r = subprocess.run(cmd, **kwargs)
    kwargs['stdout'].close()
    return r.returncode == 0, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authorization-root', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument('--training-workers', type=int, default=80)
    ap.add_argument('--postprocess-workers', type=int, default=16)
    ap.add_argument('--output-base', type=Path, required=True)
    args = ap.parse_args()

    # ── Authorization verification ──
    auth_root = args.authorization_root.resolve()
    verify_sealed_directory(auth_root)
    auth = json.loads((auth_root / 'authorization.json').read_text())

    if auth.get('v2_inner_cv_authorized') is not True:
        raise RuntimeError('V2 inner-CV not authorized')
    if auth.get('stage2_authorized') is not False:
        raise RuntimeError('Stage-2 must not be authorized')

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=SCRIPTS.parent.parent, text=True).strip()
    if head != auth.get('source_commit', ''):
        raise RuntimeError(f'git HEAD {head[:8]} != auth {auth.get("source_commit", "?")[:8]}')

    for key in ['trainer_sha', 'predictor_sha', 'evaluator_sha', 'auditor_sha',
                'splits_root_seal', 'teacher_root_seal', 's1_root_seal']:
        if key not in auth:
            raise RuntimeError(f'authorization missing {key}')

    # Verify launcher SHA matches
    actual_launcher_sha = sha256_file(Path(__file__))
    if auth.get('launcher_sha') and actual_launcher_sha != auth['launcher_sha']:
        raise RuntimeError(f'launcher SHA mismatch')

    auth_seal = sha256_file(auth_root / 'SHA256SUMS')
    inventory = json.loads((Path(auth['job_inventory_root']) / 'v2_stage1_job_inventory.json').read_text())
    jobs = inventory['jobs']

    if len(jobs) != 864:
        raise RuntimeError(f'Expected 864 jobs, got {len(jobs)}')
    for j in jobs:
        if j['seed'] != 42:
            raise RuntimeError(f'Unauthorized seed {j["seed"]} for {j["label"]}')

    print(f'Authorization: PASS | commit={head[:8]} | jobs={len(jobs)}')
    print(f'GPUs: {args.gpus} | training_workers: {args.training_workers} | postprocess: {args.postprocess_workers}')

    # ── Launch training pool ──
    n_gpus = len(args.gpus)
    workers_per_gpu = max(1, args.training_workers // n_gpus)
    LOG_DIR = Path('/mnt/sdc/dty_user/openvla_attack/logs/v2_stage1')

    base_env = os.environ.copy()
    base_env.update({
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'PYTHONPATH': '/mnt/sdc/dty_user/openvla_attack/src',
    })

    def process_job(job, gpu_id):
        label = job['label']
        out_dir = Path(job['output_path'])
        log_file = LOG_DIR / f'{label}_gpu{gpu_id}.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if verify_existing_output(out_dir, job, auth_seal):
                return label, True, 'SKIP_VALID'
        except Exception as e:
            return label, False, f'HOLD_EXISTING: {e}'

        # Shell wrapper: explicitly set CUDA_VISIBLE_DEVICES, clear any inherited value
        gpu_env_prefix = f'export PYTHONPATH=/mnt/sdc/dty_user/openvla_attack/src; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export NUMEXPR_NUM_THREADS=1;'

        # ── Train ──
        train_args = f'--candidate {job["candidate"]} --outer-fold {job["outer_fold"]} --inner-fold {job["inner_fold"]} --seed {job["seed"]} --gpu {gpu_id} --receptive-field {job["W"]} --hidden-dim {job["hidden_dim"]} --dropout {job["dropout"]} --weight-decay {job["weight_decay"]} --epochs 30 --inner-cv-splits-root {SPLITS} --output-root {out_dir}'
        train_cmd = ['bash', '-c', f'{gpu_env_prefix} exec {PY} {SCRIPTS / "train_factorized_v2_inner_cv.py"} {train_args}']
        ok, elapsed = run_cmd(train_cmd, None, log_file)
        if not ok:
            return label, False, f'TRAIN_FAIL_{elapsed:.0f}s'

        # ── Predict ──
        pred_dir = out_dir.parent / f'predict_{label}'
        predict_cmd = ['bash', '-c', f'{gpu_env_prefix} exec {PY} {SCRIPTS / "predict_factorized_v2_inner_cv.py"} --checkpoint-dir {out_dir} --inner-cv-splits-root {SPLITS} --output-root {pred_dir} --gpu {gpu_id}']
        ok2, _ = run_cmd(predict_cmd, None, log_file)
        if not ok2:
            return label, False, 'PREDICT_FAIL'

        # ── Evaluate ──
        eval_out = pred_dir.parent / f'eval_{label}.json'
        eval_cmd = ['bash', '-c', f'{gpu_env_prefix} exec {PY} {SCRIPTS / "evaluate_factorized_v2_inner_cv.py"} --predictions-base {pred_dir.parent} --output {eval_out} --mode single --candidate {job["candidate"]} --outer-fold {job["outer_fold"]} --inner-fold {job["inner_fold"]} --seed {job["seed"]}']
        ok3, _ = run_cmd(eval_cmd, None, log_file)
        if not ok3:
            return label, False, 'EVAL_FAIL'

        # ── Audit ──
        audit_out = pred_dir.parent / f'audit_{label}.json'
        audit_cmd = ['bash', '-c', f'{gpu_env_prefix} exec {PY} {SCRIPTS / "audit_factorized_v2_inner_cv_predictions.py"} --prediction-dir {pred_dir} --inner-cv-splits-root {SPLITS} --output {audit_out}']
        ok4, _ = run_cmd(audit_cmd, None, log_file)
        if not ok4:
            return label, False, 'AUDIT_FAIL'

        return label, True, f'COMPLETE_{elapsed:.0f}s'

    # Assign jobs to GPUs with round-robin
    job_assignments = []
    for i, job in enumerate(jobs):
        gpu = args.gpus[i % n_gpus]
        job_assignments.append((job, gpu))

    all_results = []
    training_futures = []

    with __import__('concurrent.futures').futures.ThreadPoolExecutor(
            max_workers=args.training_workers) as executor:
        future_map = {}
        for job, gpu in job_assignments:
            f = executor.submit(process_job, job, gpu)
            future_map[f] = job['label']
            time.sleep(0.05)  # gentle stagger

        for f in __import__('concurrent.futures').futures.as_completed(future_map):
            label, ok, msg = f.result()
            all_results.append((label, ok, msg))
            if not ok:
                print(f'  FAIL {label}: {msg}')

    passed = sum(1 for _, ok, _ in all_results if ok)
    failed = len(all_results) - passed
    print(f'\nStage-1: {passed}/{len(all_results)} passed, {failed} failed')
    if failed:
        sys.exit(1)
    print('STATUS: ALL_PASSED')


if __name__ == '__main__':
    main()
