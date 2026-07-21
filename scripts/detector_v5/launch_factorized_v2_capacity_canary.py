#!/usr/bin/env python3
"""V2 capacity canary: test 8/32/64/80 worker concurrency.

Dual-pool: training pool (up to N workers) + postprocess pool (max 16).
System telemetry every 10s. Output in capacity-only root.
"""
import argparse, hashlib, json, os, subprocess, sys, time, threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
SCRIPTS = Path('/mnt/sdc/dty_user/openvla_attack/scripts/detector_v5')
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
SPLITS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'
INVENTORY = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1_20260721'

CAPACITY_JOBS = 64  # diverse sample from 864-job pool


def sha256_file(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def sample_telemetry():
    """Collect one telemetry snapshot."""
    try:
        import psutil
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'swap_percent': psutil.swap_memory().percent,
            'disk_read_mb': psutil.disk_io_counters().read_bytes / (1024**2) if psutil.disk_io_counters() else 0,
            'disk_write_mb': psutil.disk_io_counters().write_bytes / (1024**2) if psutil.disk_io_counters() else 0,
            'iowait': psutil.cpu_times_percent().iowait if hasattr(psutil.cpu_times_percent(), 'iowait') else 0,
        }
    except ImportError:
        return {'timestamp': datetime.now().isoformat(), 'note': 'psutil not available'}


def run_one_job(job, gpu_id, output_base, epochs):
    """Run train -> checkpoint for one capacity job."""
    label = job['label']
    out_dir = Path(output_base) / f'capacity_{label}'
    if out_dir.exists():
        try:
            from gripper_attack.b3_training_protocol import verify_sealed_directory
            verify_sealed_directory(out_dir)
            return label, True, 'SKIP_SEALED'
        except Exception:
            pass

    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'PYTHONPATH': '/mnt/sdc/dty_user/openvla_attack/src',
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
    })

    cmd = [
        PY, str(SCRIPTS / 'train_factorized_v2_inner_cv.py'),
        '--candidate', job['candidate'],
        '--outer-fold', str(job['outer_fold']),
        '--inner-fold', str(job['inner_fold']),
        '--seed', str(job['seed']), '--gpu', '0',
        '--receptive-field', str(job['W']),
        '--hidden-dim', str(job['hidden_dim']),
        '--dropout', str(job['dropout']),
        '--weight-decay', str(job['weight_decay']),
        '--epochs', str(epochs),
        '--inner-cv-splits-root', str(SPLITS),
        '--output-root', str(out_dir),
    ]

    start = time.time()
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.STDOUT, timeout=600)
        elapsed = time.time() - start
        return label, r.returncode == 0, f'OK_{elapsed:.0f}s' if r.returncode == 0 else f'EXIT_{r.returncode}'
    except subprocess.TimeoutExpired:
        return label, False, 'TIMEOUT'
    except Exception as e:
        return label, False, str(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True, choices=[8, 32, 64, 80])
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--output-base', type=Path, required=True)
    ap.add_argument('--postprocess-max', type=int, default=16)
    args = ap.parse_args()

    n_gpus = len(args.gpus)
    workers_per_gpu = args.level // n_gpus
    total_workers = workers_per_gpu * n_gpus

    print(f'Capacity canary: level={args.level}, workers={total_workers}, gpus={n_gpus}')
    print(f'Postprocess pool: max {args.postprocess_max}')

    # Load diverse job sample from inventory
    inv = json.loads((INVENTORY / 'v2_stage1_job_inventory.json').read_text())
    all_jobs = inv['jobs']
    rng = __import__('random').Random(42)
    sampled = rng.sample(all_jobs, min(CAPACITY_JOBS, len(all_jobs)))
    print(f'Sampled {len(sampled)} jobs from {len(all_jobs)} inventory')

    # GPU worker pool assignment
    worker_jobs = defaultdict(list)
    for i, job in enumerate(sampled):
        gpu = args.gpus[i % n_gpus]
        wid = (i // n_gpus) % workers_per_gpu
        worker_jobs[(gpu, wid)].append(job)

    # Start telemetry thread
    telemetry = []
    stop_telemetry = threading.Event()

    def telemetry_loop():
        while not stop_telemetry.is_set():
            telemetry.append(sample_telemetry())
            time.sleep(10)

    telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
    telemetry_thread.start()

    # Launch workers
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}
        for (gpu, wid), jobs in worker_jobs.items():
            for job in jobs:
                f = executor.submit(run_one_job, job, gpu, args.output_base, args.epochs)
                futures[f] = job['label']

        for f in as_completed(futures):
            label, ok, msg = f.result()
            results.append((label, ok, msg))

    stop_telemetry.set()
    telemetry_thread.join(timeout=5)
    elapsed = time.time() - start_time

    # Summary
    ok_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - ok_count
    print(f'\nCapacity Level {args.level}: {ok_count}/{len(results)} passed ({elapsed:.0f}s)')

    # Save results
    out_dir = Path(args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'level': args.level, 'total_workers': total_workers,
        'n_gpus': n_gpus, 'epochs': args.epochs,
        'elapsed_seconds': elapsed, 'passed': ok_count, 'failed': fail_count,
        'results': [{'label': l, 'ok': ok, 'msg': m} for l, ok, m in results],
        'telemetry_samples': len(telemetry),
    }
    (out_dir / f'capacity_level_{args.level}_report.json').write_text(
        json.dumps(report, indent=2))

    if fail_count > 0:
        for l, ok, m in results:
            if not ok:
                print(f'  FAIL {l}: {m}')
        sys.exit(1)


if __name__ == '__main__':
    main()
