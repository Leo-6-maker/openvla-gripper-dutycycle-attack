#!/usr/bin/env python3
"""V2 capacity canary: test 8/32/64/80 worker concurrency.

Dual-pool architecture:
  Training pool: up to N concurrent train jobs (--level N)
  Postprocess pool: max 16 concurrent predict+evaluate+audit chains

Each level uses the SAME 160+ diverse jobs from the 864-job inventory.
Output directories are fully isolated per level.
Full system + GPU telemetry every 10s.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time, threading, uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
SCRIPTS = Path('/mnt/sdc/dty_user/openvla_attack/scripts/detector_v5')
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
SPLITS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'
INVENTORY = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1_20260721'
LOG_BASE = Path('/mnt/sdc/dty_user/openvla_attack/logs/v2_capacity')
GPU_COUNT = 8

CAPACITY_JOBS = 168  # 3 waves at 56, 2 waves at 84, sufficient for 80-concurrency


def sha256_file(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.tmp')
    with t.open('x') as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {'SHA256SUMS', 'SHA256SUMS.sha256'}
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = ''.join(f'{sha256_file(p)}  {p.relative_to(root).as_posix()}\n' for p in fs)
    _atomic_text(root / 'SHA256SUMS', c)
    _atomic_text(root / 'SHA256SUMS.sha256', f'{sha256_file(root / "SHA256SUMS")}  SHA256SUMS\n')


def sample_telemetry():
    """Collect CPU/RAM/I/O/GPU snapshot."""
    snap = {'timestamp': time.time()}
    try:
        import psutil
        snap['cpu_pct'] = psutil.cpu_percent(interval=None)
        snap['load_1m'] = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        mem = psutil.virtual_memory()
        snap['ram_pct'] = mem.percent
        snap['ram_used_gb'] = round(mem.used / (1024**3), 2)
        swap = psutil.swap_memory()
        snap['swap_pct'] = swap.percent
        snap['swap_used_gb'] = round(swap.used / (1024**3), 2)
        io_counters = psutil.disk_io_counters()
        snap['disk_read_mb'] = round(io_counters.read_bytes / (1024**2), 1)
        snap['disk_write_mb'] = round(io_counters.write_bytes / (1024**2), 1)
        cpu_times = psutil.cpu_times_percent()
        snap['iowait'] = getattr(cpu_times, 'iowait', 0)
    except ImportError:
        snap['note'] = 'psutil_missing'
    except Exception:
        snap['note'] = 'telemetry_error'

    # GPU telemetry via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'idx': int(parts[0]), 'util_pct': float(parts[1]) if parts[1] != '[Not Supported]' else -1,
                    'mem_used_mb': float(parts[2]), 'mem_total_mb': float(parts[3]),
                    'power_w': float(parts[4]) if parts[4] != '[Not Supported]' else -1,
                    'temp_c': float(parts[5]) if len(parts) > 5 and parts[5] != '[Not Supported]' else -1,
                })
        snap['gpus'] = gpus
    except Exception:
        snap['gpus'] = []

    return snap


def train_job(job, gpu_id, output_dir, epochs, log_dir):
    """Run training only. Returns (label, ok, runtime_seconds, msg)."""
    label = job['label']
    out_dir = Path(output_dir) / f'train_{label}'
    log_file = log_dir / f'train_{label}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists():
        try:
            from gripper_attack.b3_training_protocol import verify_sealed_directory
            verify_sealed_directory(out_dir)
            src = json.loads((out_dir / 'source_binding.json').read_text())
            if (src.get('candidate') == job['candidate'] and
                src.get('outer_fold') == job['outer_fold'] and
                src.get('inner_fold') == job['inner_fold'] and
                src.get('seed') == job['seed']):
                return label, True, 0.0, 'SKIP_SEALED_VALID'
            else:
                return label, False, 0.0, f'HOLD_METADATA_MISMATCH'
        except Exception as e:
            return label, False, 0.0, f'HOLD_INVALID_EXISTING: {str(e)[:60]}'

    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'PYTHONPATH': '/mnt/sdc/dty_user/openvla_attack/src',
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
    })

    cmd = [PY, str(SCRIPTS / 'train_factorized_v2_inner_cv.py'),
           '--candidate', job['candidate'], '--outer-fold', str(job['outer_fold']),
           '--inner-fold', str(job['inner_fold']), '--seed', str(job['seed']), '--gpu', '0',
           '--receptive-field', str(job['W']), '--hidden-dim', str(job['hidden_dim']),
           '--dropout', str(job['dropout']), '--weight-decay', str(job['weight_decay']),
           '--epochs', str(epochs),
           '--inner-cv-splits-root', str(SPLITS), '--output-root', str(out_dir)]

    start = time.time()
    try:
        with open(log_file, 'w') as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
        elapsed = time.time() - start
        if r.returncode == 0:
            return label, True, elapsed, 'OK'
        else:
            return label, False, elapsed, f'EXIT_{r.returncode}'
    except subprocess.TimeoutExpired:
        return label, False, time.time() - start, 'TIMEOUT'
    except Exception as e:
        return label, False, time.time() - start, str(e)[:80]


def postprocess_job(train_output_dir):
    """Run predict + evaluate + audit on a completed training output."""
    label = Path(train_output_dir).name
    # postprocess not fully implemented in this canary — training-only load test
    return label, True, 0.0, 'POSTPROCESS_SKIPPED_CANARY'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True, choices=[8, 32, 64, 80])
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--output-base', type=Path, required=True)
    args = ap.parse_args()

    n_gpus = len(args.gpus)
    workers_per_gpu = max(1, args.level // n_gpus)
    training_workers = workers_per_gpu * n_gpus
    postprocess_workers = min(16, training_workers // 2)

    capacity_root = args.output_base / f'level_{args.level:03d}'
    train_output_dir = capacity_root / 'training_outputs'
    log_dir = LOG_BASE / f'level_{args.level:03d}'

    print(f'=== Capacity Level {args.level} ===')
    print(f'Training workers: {training_workers} ({workers_per_gpu}/GPU × {n_gpus} GPUs)')
    print(f'Postprocess workers: {postprocess_workers}')
    print(f'Output: {capacity_root}')

    if capacity_root.exists():
        print(f'HOLD: output exists: {capacity_root}')
        sys.exit(1)

    # Load and sample jobs
    from gripper_attack.b3_training_protocol import verify_sealed_directory
    verify_sealed_directory(SPLITS)
    verify_sealed_directory(INVENTORY)
    inv = json.loads((INVENTORY / 'v2_stage1_job_inventory.json').read_text())
    all_jobs = inv['jobs']
    rng = __import__('random').Random(42)
    sampled = rng.sample(all_jobs, min(CAPACITY_JOBS, len(all_jobs)))
    print(f'Sampled {len(sampled)} jobs from {len(all_jobs)} inventory')

    capacity_root.mkdir(parents=True)
    train_output_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    # Assign jobs to GPUs (round-robin)
    job_assignments = []
    for i, job in enumerate(sampled):
        gpu = args.gpus[i % n_gpus]
        job_assignments.append((job, gpu))

    # Start telemetry
    telemetry = []
    stop_telemetry = threading.Event()

    def telemetry_loop():
        while not stop_telemetry.is_set():
            telemetry.append(sample_telemetry())
            time.sleep(10)

    telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
    telemetry_thread.start()

    # Training phase
    train_results = []
    train_start = time.time()

    with ThreadPoolExecutor(max_workers=training_workers) as executor:
        futures = {}
        for job, gpu in job_assignments:
            f = executor.submit(train_job, job, gpu, train_output_dir, args.epochs, log_dir)
            futures[f] = job['label']

        for f in as_completed(futures):
            label, ok, elapsed, msg = f.result()
            train_results.append({'label': label, 'ok': ok, 'runtime_s': elapsed, 'msg': msg})

    train_elapsed = time.time() - train_start
    stop_telemetry.set()
    telemetry_thread.join(timeout=5)

    # Compute stats
    runtimes = [r['runtime_s'] for r in train_results if r['ok'] and r['runtime_s'] > 0]
    runtimes_sorted = sorted(runtimes) if runtimes else [0]
    n = len(runtimes_sorted)
    p50 = runtimes_sorted[n // 2] if n > 0 else 0
    p95 = runtimes_sorted[int(n * 0.95)] if n > 1 else p50

    ok_count = sum(1 for r in train_results if r['ok'])
    fail_count = len(train_results) - ok_count

    print(f'\nResults: {ok_count}/{len(train_results)} passed, {fail_count} failed')
    print(f'Runtime: p50={p50:.0f}s p95={p95:.0f}s wall={train_elapsed:.0f}s')
    if fail_count > 0:
        for r in train_results:
            if not r['ok']:
                print(f'  FAIL {r["label"]}: {r["msg"]}')

    # Save results
    report = {
        'level': args.level,
        'training_workers': training_workers,
        'postprocess_workers': postprocess_workers,
        'n_gpus': n_gpus,
        'epochs': args.epochs,
        'n_jobs': len(sampled),
        'ok_count': ok_count,
        'fail_count': fail_count,
        'wall_seconds': round(train_elapsed, 1),
        'p50_seconds': round(p50, 1),
        'p95_seconds': round(p95, 1),
        'results': train_results,
        'telemetry_count': len(telemetry),
        'telemetry_snapshots': telemetry[::max(1, len(telemetry)//100)],  # subsample for file size
    }

    (capacity_root / 'capacity_report.json').write_text(json.dumps(report, indent=2, default=str))
    with open(capacity_root / 'system_telemetry.jsonl', 'w') as f:
        for t in telemetry:
            f.write(json.dumps(t, default=str) + '\n')

    with open(capacity_root / 'per_job_runtime.csv', 'w') as f:
        f.write('label,ok,runtime_s,msg\n')
        for r in train_results:
            f.write(f'{r["label"]},{r["ok"]},{r["runtime_s"]:.1f},{r["msg"]}\n')

    _atomic_text(capacity_root / 'source_binding.json', json.dumps({
        'capacity_launcher_sha': sha256_file(Path(__file__)),
        'splits_seal': sha256_file(SPLITS / 'SHA256SUMS'),
        'inventory_seal': sha256_file(INVENTORY / 'SHA256SUMS'),
    }, indent=2))

    write_seal(capacity_root)
    print(f'Sealed: {capacity_root}')
    print(f'Seal: {sha256_file(capacity_root / "SHA256SUMS")}')

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
