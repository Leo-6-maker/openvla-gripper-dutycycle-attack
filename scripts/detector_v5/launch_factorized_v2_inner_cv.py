#!/usr/bin/env python3
"""V2 inner-CV launcher: train → predict → evaluate. Fail-closed."""
import subprocess, sys, time, json, hashlib, argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[1, 3, 4, 5])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--output-base', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--job-file', type=Path, required=True,
                    help='JSON job inventory file')
    args = ap.parse_args()

    jobs_data = json.loads(args.job_file.read_text())
    jobs = jobs_data['jobs']

    print(f'Total jobs: {len(jobs)} | GPUs: {args.gpus} | workers/GPU: {args.workers}')

    if args.dry_run:
        for j in jobs[:5]:
            print(f'  {j["label"]}')
        print(f'  ... ({len(jobs)} total)')
        return

    worker_assignments = defaultdict(list)
    for i, job in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        wid = (i // len(args.gpus)) % args.workers
        worker_assignments[(gpu, wid)].append(job)

    def launch_worker(gpu, wid, job_list):
        env = {}
        env.update(__import__('os').environ)
        env.update({'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                     'PYTHONPATH': '/mnt/sdc/dty_user/openvla_attack/src',
                     'CUDA_VISIBLE_DEVICES': str(gpu)})
        LOG_DIR = Path('/mnt/sdc/dty_user/openvla_attack/logs/v2_inner_cv')
        results = []
        for job in job_list:
            out_dir = Path(job['output_path'])
            log_file = LOG_DIR / f'{job["label"]}_gpu{gpu}_w{wid}.log'
            log_file.parent.mkdir(parents=True, exist_ok=True)

            if out_dir.exists():
                try:
                    verify_sealed_directory(out_dir)
                    results.append((job['label'], True, 'SKIP_SEALED'))
                    continue
                except Exception as e:
                    results.append((job['label'], False, str(e)[:80]))
                    continue

            train_args = [
                PY, str(SCRIPTS / 'train_factorized_v2_inner_cv.py'),
                '--candidate', job['candidate'],
                '--outer-fold', str(job['outer_fold']),
                '--inner-fold', str(job['inner_fold']),
                '--seed', str(job['seed']),
                '--gpu', '0',
                '--receptive-field', str(job['W']),
                '--hidden-dim', str(job['hidden_dim']),
                '--dropout', str(job['dropout']),
                '--weight-decay', str(job['weight_decay']),
                '--epochs', '30',
                '--inner-cv-splits-root', str(SPLITS),
                '--output-root', str(out_dir),
            ]

            print(f'  [GPU{gpu} W{wid}] START {job["label"]}')
            start = time.time()
            with open(log_file, 'w') as lf:
                r = subprocess.run(train_args, env=env, stdout=lf, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            ok = r.returncode == 0
            if ok:
                print(f'  [GPU{gpu} W{wid}] DONE  {job["label"]} ({elapsed:.0f}s)')
            else:
                print(f'  [GPU{gpu} W{wid}] FAIL  {job["label"]} (exit={r.returncode})')
            results.append((job['label'], ok, 'OK' if ok else f'EXIT_{r.returncode}'))
        return results

    import threading
    all_results = []
    threads = []
    for (gpu, w), wjobs in sorted(worker_assignments.items()):
        t = threading.Thread(target=lambda g=gpu, w=w, wj=wjobs: all_results.extend(launch_worker(g, w, wj)))
        threads.append(t)
        t.start()
        time.sleep(3)

    for t in threads:
        t.join()

    failed = [(l, r) for l, ok, r in all_results if not ok]
    print(f'\nRESULTS: {len(all_results)-len(failed)}/{len(all_results)} passed')
    if failed:
        for l, r in failed:
            print(f'  FAIL {l}: {r}')
        sys.exit(1)


if __name__ == '__main__':
    main()
