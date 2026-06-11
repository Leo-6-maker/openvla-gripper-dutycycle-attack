#!/usr/bin/env python3
"""S20f Auto-progressing watcher: monitors 3 GPU workers, auto-restarts, logs progress.
Runs on server via: nohup python run_s20f_watcher.py > watcher.log 2>&1 &"""
import json, os, subprocess, time, csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
WORKER = f'{REPO}/scripts/stageb/run_s20f_gpu_worker.py'
TRACKA_SCRIPT = f'{REPO}/scripts/stageb/run_s20f_trackA_known_parent_lift.sh'

WORKERS = {
    'gpu26': {
        'type': 'python',
        'gpu': '2,6', 'render': '2',
        'command': [PY, '-u', WORKER,
                    '/data/liuyu/outputs/stageb_s20f_queues_20260611/queue_gpu26.csv',
                    '2,6', '2',
                    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output'],
        'log': '/data/liuyu/outputs/stageb_s20f_queues_20260611/log_gpu26.txt',
        'output_dir': '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
        'queue': '/data/liuyu/outputs/stageb_s20f_queues_20260611/queue_gpu26.csv',
    },
    'gpu45': {
        'type': 'python',
        'gpu': '4,5', 'render': '4',
        'command': [PY, '-u', WORKER,
                    '/data/liuyu/outputs/stageb_s20f_queues_20260611/queue_gpu45.csv',
                    '4,5', '4',
                    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output'],
        'log': '/data/liuyu/outputs/stageb_s20f_queues_20260611/log_gpu45.txt',
        'output_dir': '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
        'queue': '/data/liuyu/outputs/stageb_s20f_queues_20260611/queue_gpu45.csv',
    },
}

STATUS_FILE = '/data/liuyu/outputs/stageb_s20f_watcher_status.json'
SUMMARY_FILE = '/data/liuyu/outputs/stageb_s20f_watcher_summary.txt'
CHECK_INTERVAL = 30  # seconds between checks
LOG_INTERVAL = 10     # checks between summary writes

processes = {}
check_count = 0


def count_outputs(output_dir, prefix='summary_'):
    """Count JSON outputs in directory."""
    d = Path(output_dir)
    if not d.exists():
        return 0
    return len(list(d.glob(f'{prefix}*.json')))


def count_queue_done(queue_path):
    """Count done jobs in queue CSV."""
    if not queue_path or not Path(queue_path).exists():
        return (0, 0)
    with open(queue_path, newline='') as f:
        jobs = list(csv.DictReader(f))
    done = sum(1 for j in jobs if j.get('status') == 'done')
    return (done, len(jobs))


def check_gpu_health():
    """Check nvidia-smi for zombie processes."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'],
                                capture_output=True, text=True, timeout=10)
        lines = [l for l in result.stdout.strip().split('\n') if l and 'pid' not in l]
        return len(lines) > 0  # True if GPUs are in use
    except Exception:
        return False


def start_worker(name, cfg):
    """Start a worker process if not running."""
    if name in processes and processes[name].poll() is None:
        return  # already running

    log_f = open(cfg['log'], 'a') if cfg['log'] else None
    env = os.environ.copy()
    env.update({'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                'OPENVLA_ATTN_IMPLEMENTATION': 'eager',
                'CUDA_VISIBLE_DEVICES': cfg['gpu'], 'DISPLAY': ''})
    p = subprocess.Popen(cfg['command'], stdout=log_f, stderr=subprocess.STDOUT, env=env)
    processes[name] = p
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Started {name} (PID={p.pid})", flush=True)
    return p


def write_status():
    """Write current status to JSON."""
    status = {
        'updated': datetime.now().isoformat(),
        'checks': check_count,
        'workers': {},
    }
    for name, cfg in WORKERS.items():
        n_done, n_total = count_queue_done(cfg.get('queue'))
        n_outputs = count_outputs(cfg['output_dir'])
        running = name in processes and processes[name].poll() is None
        status['workers'][name] = {
            'gpu': cfg['gpu'],
            'running': running,
            'pid': processes[name].pid if running else None,
            'outputs': n_outputs,
            'queue_done': n_done,
            'queue_total': n_total,
        }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def write_summary():
    """Write human-readable summary."""
    lines = [f"=== S20f Watcher Summary @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
             f"Checks: {check_count}",
             ""]
    for name, cfg in WORKERS.items():
        running = name in processes and processes[name].poll() is None
        n_done, n_total = count_queue_done(cfg.get('queue'))
        n_outputs = count_outputs(cfg['output_dir'])
        lines.append(f"{name} (GPU {cfg['gpu']}): {'RUNNING' if running else 'STOPPED'}")
        if n_total > 0:
            lines.append(f"  Queue: {n_done}/{n_total} done")
        lines.append(f"  JSON outputs: {n_outputs}")
        lines.append("")

    # Total from all queues
    total_done = sum(count_queue_done(cfg.get('queue'))[0] for _, cfg in WORKERS.items())
    total_all = sum(count_queue_done(cfg.get('queue'))[1] for _, cfg in WORKERS.items())
    total_outputs = sum(count_outputs(cfg['output_dir']) for _, cfg in WORKERS.items())
    lines.append(f"TOTAL: {total_done}/{total_all} queue jobs done, {total_outputs} JSON outputs")

    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(lines))


def main():
    global check_count, processes
    print(f"[{datetime.now().strftime('%H:%M:%S')}] S20f Watcher starting...", flush=True)
    print(f"Status: {STATUS_FILE}", flush=True)
    print(f"Summary: {SUMMARY_FILE}", flush=True)

    # Start all workers
    for name, cfg in WORKERS.items():
        start_worker(name, cfg)

    while True:
        time.sleep(CHECK_INTERVAL)
        check_count += 1

        # Check each worker
        all_dead = True
        for name, cfg in WORKERS.items():
            if name in processes:
                rc = processes[name].poll()
                if rc is not None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} exited (code={rc}), restarting...", flush=True)
                    start_worker(name, cfg)

            if name in processes and processes[name].poll() is None:
                all_dead = False

            # Check if queue is complete — if so, don't restart
            n_done, n_total = count_queue_done(cfg.get('queue'))
            if n_total > 0 and n_done >= n_total and name in processes:
                if processes[name].poll() is not None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} queue complete ({n_done}/{n_total})", flush=True)

        # Write periodic summary
        if check_count % LOG_INTERVAL == 0:
            write_status()
            write_summary()
            running = sum(1 for p in processes.values() if p.poll() is None)
            q_done = sum(count_queue_done(cfg.get('queue'))[0] for _, cfg in WORKERS.items())
            q_total = sum(count_queue_done(cfg.get('queue'))[1] for _, cfg in WORKERS.items())
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Check #{check_count}: "
                  f"{running} workers running, {q_done}/{q_total} queue jobs done", flush=True)

        if all_dead:
            # Check if ALL queues are done
            total_remaining = sum(
                count_queue_done(cfg.get('queue'))[1] - count_queue_done(cfg.get('queue'))[0]
                for _, cfg in WORKERS.items())
            if total_remaining == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ALL QUEUES COMPLETE. Watcher exiting.", flush=True)
                write_status()
                write_summary()
                break


if __name__ == '__main__':
    main()
