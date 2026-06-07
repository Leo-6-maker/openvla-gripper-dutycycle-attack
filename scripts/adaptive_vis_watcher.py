#!/usr/bin/env python3
"""Adaptive VIS Screening Watcher v2 — P0 hardened calibration monitor.
NEVER auto-launches 145-candidate adaptive screening without explicit approval.
"""

import os, sys, time, subprocess, json, csv, re, glob
from datetime import datetime

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
OUT = '/data/liuyu/outputs/adaptive_vis_watcher_20260605'
CALIB_3R_ROOTS = [
    '/data/liuyu/outputs/vis_calibration_1r3r_20260605',
    '/data/liuyu/outputs/vis_calibration_1r3r_20260605_recovery_3r',
]
CALIB_1R_ROOTS = [
    '/data/liuyu/outputs/vis_calibration_1r3r_20260605_1r',
]
SCRN = '/data/liuyu/outputs/adaptive_vis_1r_screening_20260605'
STATE_CSV = os.path.join(OUT, 'adaptive_jobs_state.csv')
LOG_FILE = os.path.join(OUT, 'events.log')
BLACKLISTED_GPUS = {'3', '7'}
MIN_VALID_PAIRS = 6  # minimum calibration candidates with both 1R and 3R

os.makedirs(OUT, exist_ok=True)

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    line = f'{t} {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def check_gpu_health():
    """Check GPU health: Xid, OOM, memory. Returns healthy pairs."""
    try:
        # Run nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        gpu_mem = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): continue
            idx, mem = line.split(',')
            idx = idx.strip(); mem = int(mem.strip().split()[0])
            gpu_mem[idx] = mem
        # Check for new Xid
        xid_result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=10)
        new_xids = []
        for line in xid_result.stdout.strip().split('\n'):
            if 'Xid' in line and 'NVRM' in line:
                new_xids.append(line.strip())
        return gpu_mem, new_xids
    except Exception as e:
        return {}, [f'GPU_HEALTH_CHECK_ERROR:{e}']

def is_done(log_path):
    """Check if chain completed: 'Results:' in THIS specific log file.
    Does NOT check other files — avoids false positives from precheck CSVs."""
    if not log_path or not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as f:
            return 'Results:' in f.read()
    except:
        return False

def write_state(phase, jobs, smoke_gate=False, adaptive_gate=False, calib_verdict=''):
    """Write persistent state CSV."""
    rows = []
    for job_id, job in jobs.items():
        rows.append({
            'stage': phase,
            'job_id': job_id,
            'gpu_pair': job.get('gpu', ''),
            'csv_name': job.get('csv', ''),
            'log_path': job.get('log', ''),
            'pid': job.get('pid', ''),
            'started_at': job.get('started', ''),
            'finished_at': job.get('finished', ''),
            'status': 'DONE' if job.get('done') else 'RUNNING',
            'attempt_count': str(job.get('attempts', 1)),
            'last_error': job.get('error', ''),
        })
    with open(STATE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Also write JSON state
    state = {
        'time': str(datetime.now()),
        'phase': phase,
        'smoke_gate': smoke_gate,
        'adaptive_gate': adaptive_gate,
        'calibration_verdict': calib_verdict,
        'job_count': len(rows),
    }
    with open(os.path.join(OUT, 'watcher_status.json'), 'w') as f:
        json.dump(state, f, indent=2)

def _summaries_identical(path_a, path_b):
    """Check if two files are byte-identical (detect copy)."""
    try:
        with open(path_a, 'rb') as f:
            a = f.read()
        with open(path_b, 'rb') as f:
            b = f.read()
        return a == b
    except:
        return False

def _read_summary_rows(path):
    """Read audit summary, handling both DictReader and raw parsing."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize keys (strip BOM and whitespace)
            normalized = {}
            for k, v in r.items():
                nk = k.lstrip('﻿').strip().lower()
                normalized[nk] = (v or '').strip()
            rows.append(normalized)
    return rows

def _row_key(r):
    """Extract (task_key, state_id, window_start) from a summary row.
    Compatible with: task_key/task/object_task, state_id/state, window_start/start."""
    tk = r.get('task_key') or r.get('task') or r.get('object_task') or ''
    sid = r.get('state_id') or r.get('state') or ''
    ws = r.get('window_start') or r.get('start') or ''
    if not tk:
        return (None, None, None)
    return (tk, str(sid), str(ws))

# === MAIN ===
log('=== Adaptive VIS Watcher v2 (P0 hardened) Started ===')
log('SAFETY: Will NOT auto-launch 145-candidate adaptive screening')

# Discover existing 3R log files
calib_3r_jobs = {}
for pair_tag in ['01', '45', '26']:
    gpu_pair = ','.join(pair_tag)
    log_path = os.path.join(CALIB_3R_ROOTS[0], f'vis3r_{pair_tag}.log')
    csv_name = f'calib_pcheck_{pair_tag}.csv'
    exists = os.path.exists(log_path)
    done = is_done(log_path)
    calib_3r_jobs[pair_tag] = {
        'gpu': gpu_pair, 'csv': csv_name, 'log': log_path,
        'done': done, 'started': str(datetime.now()) if exists else '',
        'attempts': 1, 'error': '',
    }
    log(f'3R GPU={gpu_pair}: log_exists={exists} done={done} path={log_path}')

# Discover recovery chain log
RECOVERY_ROOT = '/data/liuyu/outputs/vis_calibration_1r3r_20260605_recovery_3r'
RECOVERY_LOG = os.path.join(RECOVERY_ROOT, 'chain.log')
RECOVERY_SENTINEL = os.path.join(RECOVERY_ROOT, 'RECOVERY_DONE')
recovery_done = is_done(RECOVERY_LOG) or os.path.exists(RECOVERY_SENTINEL)
log(f'Recovery chain: log_exists={os.path.exists(RECOVERY_LOG)} done={recovery_done}')

started = datetime.now()
phase = 'calibration_3r'
calib_1r_jobs = {}
calib_compare_done = False
calib_verdict = ''

iteration = 0
while True:
    iteration += 1
    time.sleep(60)

    # === GPU HEALTH ===
    gpu_mem, new_xids = check_gpu_health()
    for xid_line in new_xids:
        log(f'GPU_XID: {xid_line}')

    # === PHASE: calibration 3R ===
    if phase == 'calibration_3r':
        all_3r_done = True
        for tag, job in calib_3r_jobs.items():
            if not job['done'] and is_done(job['log']):
                log(f'3R DONE: GPU={job["gpu"]}')
                job['done'] = True
                job['finished'] = str(datetime.now())
            if not job['done']:
                all_3r_done = False

        # Recovery barrier: check recovery chain too
        recovery_now_done = is_done(RECOVERY_LOG) or os.path.exists(RECOVERY_SENTINEL)
        if not recovery_now_done and os.path.exists(RECOVERY_LOG):
            all_3r_done = False  # block until recovery finishes

        if all_3r_done:
            # Write sentinel when all 3R done
            if recovery_now_done and not os.path.exists(RECOVERY_SENTINEL):
                with open(RECOVERY_SENTINEL, 'w') as f:
                    f.write(str(datetime.now()))
            log('=== All 3R calibration done (primary + recovery). Launching 1R comparison ===')
            os.makedirs(CALIB_1R_ROOTS[0], exist_ok=True)
            for tag, job in calib_3r_jobs.items():
                # Launch 1R to SEPARATE output directory (NOT shared with 3R)
                r1_log = os.path.join(CALIB_1R_ROOTS[0], f'vis1r_{tag}.log')
                csv_path = os.path.join(REPO, 'tables', job['csv'])
                if not os.path.exists(csv_path):
                    log(f'SKIP 1R {tag}: CSV missing {csv_path}')
                    continue
                cmd = [PY, '-u', os.path.join(REPO, 'scripts/run_object_teacher_delay50_chain.py'),
                    '--candidate-csv', csv_path,
                    '--gpu-pairs', job['gpu'].replace(',', ','),
                    '--output-dir', CALIB_1R_ROOTS[0],  # SEPARATE from 3R
                    '--pgd-restarts', '1',
                    '--skip-clean', '--skip-random']
                log(f'LAUNCH 1R GPU={job["gpu"]} output={CALIB_1R_ROOTS[0]}')
                with open(r1_log, 'w') as lf:
                    proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
                calib_1r_jobs[tag] = {
                    'gpu': job['gpu'], 'csv': job['csv'], 'log': r1_log,
                    'done': False, 'pid': str(proc.pid),
                    'started': str(datetime.now()), 'attempts': 1, 'error': '',
                }
            phase = 'calibration_1r'
            write_state(phase, {**calib_3r_jobs, **calib_1r_jobs})

    # === PHASE: calibration 1R ===
    elif phase == 'calibration_1r':
        all_1r_done = True
        for tag, job in calib_1r_jobs.items():
            if not job['done'] and is_done(job['log']):
                log(f'1R DONE: GPU={job["gpu"]}')
                job['done'] = True
                job['finished'] = str(datetime.now())
            if not job['done']:
                all_1r_done = False

        if all_1r_done and not calib_compare_done:
            log('=== 1R complete. Running separated calibration audit ===')
            summary_3r = os.path.join(REPO, 'tables/calib_3r_summary.csv')
            summary_1r = os.path.join(REPO, 'tables/calib_1r_summary.csv')
            prov_3r = os.path.join(REPO, 'tables/calib_3r_provenance.csv')
            prov_1r = os.path.join(REPO, 'tables/calib_1r_provenance.csv')

            # Delete stale summaries to prevent reuse
            for fp in [summary_3r, summary_1r, prov_3r, prov_1r]:
                if os.path.exists(fp):
                    os.remove(fp)
                    log(f'Removed stale: {fp}')

            # Run audit on 3R trace dirs ONLY
            traces_3r = []
            for root in CALIB_3R_ROOTS:
                traces_3r.extend(glob.glob(os.path.join(root, '*', 'traces', '*vis_pgd*trace.csv')))
            traces_1r = []
            for root in CALIB_1R_ROOTS:
                traces_1r.extend(glob.glob(os.path.join(root, '*', 'traces', '*vis_pgd*trace.csv')))
            log(f'3R traces: {len(traces_3r)}  1R traces: {len(traces_1r)}')

            if traces_3r:
                dirs_3r = list(set(os.path.dirname(t) for t in traces_3r))
                audit_3r = [PY, os.path.join(REPO, 'scripts/diagnostics/audit_phase_conditioned_vis.py'),
                    '--summary-csv', summary_3r, '--output-csv', prov_3r]
                for d in dirs_3r: audit_3r.extend(['--run-dirs', d])
                subprocess.run(audit_3r, cwd=REPO, capture_output=True, timeout=60)
                log(f'3R audit: exists={os.path.exists(summary_3r)} size={os.path.getsize(summary_3r) if os.path.exists(summary_3r) else 0}')

            if traces_1r:
                dirs_1r = list(set(os.path.dirname(t) for t in traces_1r))
                audit_1r = [PY, os.path.join(REPO, 'scripts/diagnostics/audit_phase_conditioned_vis.py'),
                    '--summary-csv', summary_1r, '--output-csv', prov_1r]
                for d in dirs_1r: audit_1r.extend(['--run-dirs', d])
                subprocess.run(audit_1r, cwd=REPO, capture_output=True, timeout=60)
                log(f'1R audit: exists={os.path.exists(summary_1r)} size={os.path.getsize(summary_1r) if os.path.exists(summary_1r) else 0}')

            # UNIFIED GATE: never PASS without valid paired rows
            if not os.path.exists(summary_3r) or not os.path.exists(summary_1r):
                calib_verdict = 'FAIL_MISSING_AUDIT_OUTPUT'
                log(f'Calibration FAIL: missing summaries (3R={os.path.exists(summary_3r)} 1R={os.path.exists(summary_1r)})')
            elif os.path.getsize(summary_3r) == 0 or os.path.getsize(summary_1r) == 0:
                calib_verdict = 'FAIL_EMPTY_SUMMARY'
                log('Calibration FAIL: empty summary')
            elif _summaries_identical(summary_3r, summary_1r):
                calib_verdict = 'FAIL_1R_3R_SUMMARIES_IDENTICAL'
                log('Calibration FAIL: 1R and 3R summaries are byte-identical')
            else:
                # Parse summaries and count paired candidates
                try:
                    rows_3r = _read_summary_rows(summary_3r)
                    rows_1r = _read_summary_rows(summary_1r)
                    keys_3r = set(_row_key(r) for r in rows_3r if _row_key(r) != (None, None, None))
                    keys_1r = set(_row_key(r) for r in rows_1r if _row_key(r) != (None, None, None))
                    paired = keys_3r & keys_1r
                    log(f'3R rows={len(rows_3r)} 1R rows={len(rows_1r)} paired={len(paired)} min={MIN_VALID_PAIRS}')
                    if len(paired) >= MIN_VALID_PAIRS:
                        # Run compare script
                        compare_cmd = [PY, os.path.join(REPO, 'scripts/diagnostics/compare_vis_1r_vs_3r.py'),
                            '--vis-1r-summary', summary_1r,
                            '--vis-3r-summary', summary_3r,
                            '--candidates', os.path.join(REPO, 'tables/vis_1r_vs_3r_calibration_candidates.csv'),
                            '--output-csv', os.path.join(REPO, 'tables/vis_1r_vs_3r_calibration.csv'),
                            '--output-report', os.path.join(REPO, 'reports/VIS_1R_VS_3R_CALIBRATION.md')]
                        result = subprocess.run(compare_cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
                        if result.returncode == 0:
                            calib_verdict = 'PASS'
                            log('Calibration PASS')
                        else:
                            calib_verdict = 'FAIL_COMPARE_ERROR'
                            log(f'Calibration FAIL: compare exit={result.returncode} stderr={result.stderr[:200]}')
                    else:
                        calib_verdict = f'FAIL_INSUFFICIENT_PAIRS_{len(paired)}_lt_{MIN_VALID_PAIRS}'
                        log(f'Calibration FAIL: {len(paired)} pairs < {MIN_VALID_PAIRS}')
                except Exception as e:
                    calib_verdict = f'FAIL_PAIR_COUNT_PARSE_ERROR:{type(e).__name__}:{e}'
                    log(f'Calibration FAIL: parse error: {e}')

            calib_compare_done = True
            write_state(phase, {**calib_3r_jobs, **calib_1r_jobs},
                       calib_verdict=calib_verdict)

            if calib_verdict == 'PASS':
                log('=== Calibration PASS. Smoke test gate: requires manual approval ===')
                log('SAFETY: 145-candidate adaptive screening NOT auto-launched')
                log('To proceed: review report in reports/VIS_1R_VS_3R_CALIBRATION.md')
            else:
                log('=== Calibration FAIL. Adaptive screening BLOCKED ===')
            phase = 'complete'
            write_state(phase, {**calib_3r_jobs, **calib_1r_jobs},
                       calib_verdict=calib_verdict)

    # === Heartbeat ===
    if iteration % 30 == 0:
        elapsed = (datetime.now() - started).total_seconds() / 3600
        running = sum(1 for j in {**calib_3r_jobs, **calib_1r_jobs}.values() if not j.get('done'))
        log(f'HEARTBEAT: {elapsed:.1f}h phase={phase} running={running} calib_verdict={calib_verdict}')
        write_state(phase, {**calib_3r_jobs, **calib_1r_jobs},
                   calib_verdict=calib_verdict)

    # Safety timeout
    if (datetime.now() - started).total_seconds() > 43200:
        log('12h limit reached.')
        break
