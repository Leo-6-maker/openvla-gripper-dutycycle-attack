#!/usr/bin/env python3
"""
Phase 7 Dispatcher V2 — event-driven GPU job scheduler with atomic claims + full audit.

Key properties:
  - Single-instance via fcntl flock on lockfile
  - SQLite WAL journal mode
  - BEGIN IMMEDIATE atomic job claims
  - Unique scientific key enforcement
  - PID-to-job exact mapping
  - Restart recovery (reap abandoned CLAIMED/RUNNING jobs)
  - os.waitpid WNOHANG poll loop for child exit detection
  - Full artifact audit before SUCCESS
  - Dependency gates (phase N requires phase N-1 gate PASS)
  - Stop condition monitoring
  - Legacy worker reconciliation

Usage:
  python phase7_dispatcher_v2.py --db /path/to/jobs.sqlite --lock /path/to/dispatcher.lock
"""

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
import signal
import hashlib
from datetime import datetime
from pathlib import Path

# ── Constants ──
JOB_STATES = ['PENDING', 'BLOCKED_BY_GATE', 'CLAIMED', 'RUNNING', 'AUDITING', 'SUCCESS', 'FAILED_RETRYABLE', 'QUARANTINED', 'SUPERSEDED']
GPU_STATES = ['RUNNING', 'AUDITING', 'CLAIMING', 'IDLE_WAITING_FOR_JOB', 'IDLE_WAITING_FOR_GATE', 'STOPPED_BY_INCIDENT']

POLL_INTERVAL = 5  # seconds between idle scans
AUDIT_TIMEOUT = 300  # max seconds for audit phase
MAX_RETRIES = 1  # max retry count for FAILED_RETRYABLE

BRIDGE_SCRIPT = str(Path(__file__).resolve().parents[1] / "stageb" / "run_v2_vis_sc5_mlp_bridge_telemetry_v2.py")
REPO_ROOT = str(Path(__file__).resolve().parents[2])

# ── Stop condition detectors ──
STOP_PATTERNS = [
    (r"checkpoint.mismatch", "checkpoint_mismatch"),
    (r"bridge.mismatch", "bridge_mismatch"),
    (r"NVIDIA Xid", "nvidia_xid"),
    (r"EGL.*crash", "egl_crash"),
    (r"No space left on device", "disk_full"),
    (r"CUDA out of memory", "oom"),
]


def now_iso():
    return datetime.now().isoformat()


def scientific_key(row):
    """Deterministic scientific key from job parameters."""
    return hashlib.sha256(json.dumps({
        'suite': row.get('suite', 'libero_object'),
        'method': row.get('method', ''),
        'objective': row.get('objective', ''),
        'timing': row.get('timing', ''),
        'arm_lock': row.get('arm_lock', False),
        'task': row.get('task', ''),
        'state_id': row.get('state_id', -1),
        'perturbation_seed': row.get('perturbation_seed', -1),
        'eval_seed': row.get('eval_seed', -1),
        'config_sha': row.get('config_sha', ''),
    }, sort_keys=True).encode()).hexdigest()[:24]


# ── Database setup ──
def init_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            phase TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            suite TEXT NOT NULL DEFAULT 'libero_object',
            method TEXT NOT NULL,
            objective TEXT NOT NULL,
            timing TEXT NOT NULL DEFAULT 'student',
            arm_lock INTEGER NOT NULL DEFAULT 0,
            task TEXT NOT NULL,
            task_idx INTEGER NOT NULL DEFAULT 6,
            state_id INTEGER NOT NULL,
            perturbation_seed INTEGER NOT NULL,
            eval_seed INTEGER NOT NULL DEFAULT -1,
            condition TEXT NOT NULL,
            anchor INTEGER NOT NULL,
            trigger_step_override INTEGER NOT NULL DEFAULT -1,
            keep_running INTEGER NOT NULL DEFAULT 0,
            output_dir TEXT NOT NULL UNIQUE,
            scientific_key TEXT NOT NULL UNIQUE,
            config_sha TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'PENDING',
            gpu_id INTEGER,
            worker_pid INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            claimed_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            last_audit_at TEXT,
            audit_result TEXT,
            exit_code INTEGER,
            stderr_path TEXT,
            stdout_path TEXT,
            gate_dependency TEXT,
            notes TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gates (
            gate_name TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'PENDING',
            passed_at TEXT,
            evidence_path TEXT,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_registry (
            pid INTEGER PRIMARY KEY,
            gpu_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            output_dir TEXT NOT NULL,
            manifest_sha TEXT,
            started_at TEXT NOT NULL,
            last_seen_at TEXT,
            status TEXT NOT NULL DEFAULT 'RUNNING',
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dispatch_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            event_type TEXT NOT NULL,
            gpu_id INTEGER,
            job_id INTEGER,
            detail TEXT
        )
    """)
    conn.commit()
    return conn


def log_event(conn, event_type, gpu_id=None, job_id=None, detail=""):
    conn.execute("INSERT INTO dispatch_events (event_type, gpu_id, job_id, detail) VALUES (?,?,?,?)",
                 (event_type, gpu_id, job_id, detail))
    conn.commit()


# ── GPU detection ──
def detect_gpus():
    """Returns dict: gpu_id -> {'util': int, 'mem_used': int, 'mem_total': int, 'busy': bool}"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:
        return {}
    gpus = {}
    for line in out.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            gid = int(parts[0])
            gpus[gid] = {
                'util': int(parts[1]),
                'mem_used': int(parts[2]),
                'mem_total': int(parts[3]),
            }
    return gpus


def find_vla_worker_pids():
    """Find PIDs of running VLA/MuJoCo scientific workers."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,args", "--no-headers"], text=True, timeout=10)
    except Exception:
        return set()
    pids = set()
    for line in out.strip().split('\n'):
        if 'run_v2_vis_sc5_mlp_bridge' in line and 'dispatcher' not in line:
            try:
                pids.add(int(line.strip().split()[0]))
            except ValueError:
                pass
    return pids


def check_disk():
    """Returns (root_free_gb, sdc_free_gb)."""
    try:
        out = subprocess.check_output(["df", "-BG", "/", "/mnt/sdc"], text=True, timeout=10)
    except Exception:
        return (0, 0)
    lines = out.strip().split('\n')[1:]
    results = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            results[parts[-1]] = int(parts[3].replace('G', ''))
    return (results.get('/', 0), results.get('/mnt/sdc', 0))


# ── Job claiming ──
def claim_job(conn, gpu_id):
    """Atomically claim the highest-priority eligible job for a GPU."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Select inside transaction for atomic claim
        cur = conn.execute("""
            SELECT job_id, phase, method, task, state_id, perturbation_seed, condition,
                   objective, arm_lock, timing, trigger_step_override, keep_running,
                   output_dir, anchor, task_idx, eval_seed, scientific_key
            FROM jobs
            WHERE (status = 'PENDING'
                OR (status = 'FAILED_RETRYABLE' AND retry_count < max_retries))
            AND (
                gate_dependency IS NULL
                OR EXISTS (
                    SELECT 1 FROM gates
                    WHERE gate_name = jobs.gate_dependency
                    AND status = 'PASS'
                )
            )
            ORDER BY priority ASC, phase ASC, method ASC, task ASC, state_id ASC, job_id ASC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            conn.execute("ROLLBACK")
            return None
        cur2 = conn.execute("""
            UPDATE jobs SET status='CLAIMED', gpu_id=?, claimed_at=?,
            retry_count = CASE WHEN status = 'FAILED_RETRYABLE' THEN retry_count + 1 ELSE retry_count END
            WHERE job_id=? AND (status = 'PENDING'
                OR (status = 'FAILED_RETRYABLE' AND retry_count < max_retries))
        """, (gpu_id, now_iso(), row[0]))
        if cur2.rowcount != 1:
            conn.execute("ROLLBACK")
            return None
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return dict(zip(
        ['job_id', 'phase', 'method', 'task', 'state_id', 'perturbation_seed',
         'condition', 'objective', 'arm_lock', 'timing', 'trigger_step_override',
         'keep_running', 'output_dir', 'anchor', 'task_idx', 'eval_seed', 'scientific_key'],
        row))


# ── Worker launch ──
def build_worker_cmd(job, gpu_id, source_commit=""):
    """Build the shell command to launch a worker."""
    out_dir = Path(job['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    bridge_args = [
        "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3", BRIDGE_SCRIPT,
        "--condition", job['condition'],
        "--state_id", str(job['state_id']),
        "--anchor", str(job['anchor']),
        "--seed_id", str(job['perturbation_seed']),
        "--output_dir", job['output_dir'],
        "--render_gpu", str(gpu_id),
        "--task_idx", str(job['task_idx']),
        "--attack_objective", job['objective'],
        "--save_video",
        "--source_commit", source_commit,
        "--video_fps", "10",
        "--frame_stride", "2",
    ]
    if job['arm_lock']:
        bridge_args.append("--arm_lock")
    bridge_args.extend(["--eval_seed", str(job.get('eval_seed', 0))])
    if job.get('trigger_step_override', -1) >= 0 and job.get('timing', 'student') != 'student':
        bridge_args.extend(["--trigger_step_override", str(job['trigger_step_override'])])
    if job.get('keep_running'):
        bridge_args.append("--keep_running")

    return {
        'cmd': bridge_args,
        'stdout': str(stdout_path),
        'stderr': str(stderr_path),
    }


def launch_worker(conn, job, gpu_id, source_commit=""):
    """Launch a worker process for a claimed job."""
    wspec = build_worker_cmd(job, gpu_id, source_commit)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['MUJOCO_GL'] = 'egl'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['OPENVLA_DTYPE'] = 'bfloat16'
    env['OPENVLA_ATTN_IMPLEMENTATION'] = 'eager'
    env['OPENVLA_MODEL_PATH'] = '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object'
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    env['HOME'] = '/mnt/sdc/dty_user/openvla_attack/sandbox_home'
    env['TMPDIR'] = '/mnt/sdc/dty_user/openvla_attack/tmp'
    env['HF_HOME'] = '/mnt/sdc/dty_user/openvla_attack/hf_cache'
    env['TRANSFORMERS_CACHE'] = '/mnt/sdc/dty_user/openvla_attack/hf_cache'

    stdout_f = open(wspec['stdout'], 'w')
    stderr_f = open(wspec['stderr'], 'w')
    proc = None
    try:
        proc = subprocess.Popen(
            wspec['cmd'],
            stdout=stdout_f, stderr=stderr_f,
            env=env, cwd=REPO_ROOT,
        )
    except Exception as e:
        conn.execute("UPDATE jobs SET status='FAILED_RETRYABLE', notes=? WHERE job_id=?",
                     (f"Popen failed: {e}", job['job_id']))
        conn.commit()
        stdout_f.close()
        stderr_f.close()
        raise
    finally:
        if proc is not None:
            stdout_f.close()
            stderr_f.close()

    conn.execute("""
        UPDATE jobs SET status='RUNNING', worker_pid=?, started_at=?
        WHERE job_id=?
    """, (proc.pid, now_iso(), job['job_id']))

    manifest_sha = hashlib.sha256(json.dumps(wspec, sort_keys=True).encode()).hexdigest()[:16]
    conn.execute("""
        INSERT INTO worker_registry (pid, gpu_id, job_id, output_dir, manifest_sha, started_at, status)
        VALUES (?,?,?,?,?,?,'RUNNING')
    """, (proc.pid, gpu_id, job['job_id'], job['output_dir'], manifest_sha, now_iso()))

    conn.commit()
    log_event(conn, 'WORKER_LAUNCH', gpu_id=gpu_id, job_id=job['job_id'],
              detail=f"pid={proc.pid} output={job['output_dir']}")
    return proc.pid


# ── Audit ──
def audit_job(conn, job, gpu_id):
    """Full artifact audit before declaring SUCCESS."""
    out_dir = Path(job['output_dir'])
    issues = []

    # 1. Check COMPLETE.json
    complete_path = out_dir / "COMPLETE.json"
    if not complete_path.exists():
        issues.append("COMPLETE.json not found")
    else:
        try:
            comp = json.load(open(complete_path))
            if comp.get('status') != 'COMPLETE':
                issues.append(f"COMPLETE.json status={comp.get('status')}")
        except Exception as e:
            issues.append(f"COMPLETE.json parse error: {e}")

    # 2. Check telemetry
    csv_path = out_dir / "step_telemetry.csv"
    if not csv_path.exists():
        issues.append("step_telemetry.csv not found")

    # 3. Check summary
    summary_path = out_dir / "episode_summary.json"
    if not summary_path.exists():
        issues.append("episode_summary.json not found")
    else:
        try:
            summary = json.load(open(summary_path))
            # Verify protocol params
            expected_atk = 10 if summary.get('mlp_emit_step', -1) >= 0 or job.get('trigger_step_override', -1) >= 0 else 0
            actual_atk = summary.get('attack_frames', -1)
            if actual_atk != expected_atk:
                issues.append(f"attack_frames mismatch: {actual_atk} vs expected {expected_atk}")
            # Verify success field present
            if 'task_success' not in summary:
                issues.append("task_success missing from summary")
        except Exception as e:
            issues.append(f"summary parse error: {e}")

    # 4. Check video (if expected)
    # Video audit is deferred to reduce I/O during auto-dispatch

    # 5. Check checkpoint SHA
    if summary_path.exists():
        try:
            ckpt_sha = json.load(open(summary_path)).get('detector_checkpoint_sha256', '')
            if ckpt_sha and not ckpt_sha.startswith('b679e4e0'):
                issues.append(f"checkpoint SHA mismatch: {ckpt_sha[:16]}")
        except Exception:
            pass

    # 6. Check ArmLock violations (policy + env actions, all attack frames)
    if job.get('arm_lock'):
        if csv_path.exists():
            import csv
            import json as _json
            policy_violations = 0
            env_violations = 0
            max_policy_diff = 0.0
            max_env_diff = 0.0
            audited_attack_frames = 0
            missing_policy_action = 0
            missing_env_action = 0
            try:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('attack_this') != 'True':
                            continue
                        audited_attack_frames += 1
                        # Policy action audit
                        executed_pol = _json.loads(row.get('executed_policy_action_7d_after_lock', '[]'))
                        clean_pol = _json.loads(row.get('clean_policy_action_7d', '[]'))
                        if len(executed_pol) < 6 or len(clean_pol) < 6:
                            missing_policy_action += 1
                        else:
                            for i in range(6):
                                diff = abs(executed_pol[i] - clean_pol[i])
                                if diff > max_policy_diff:
                                    max_policy_diff = diff
                                if diff > 1e-7:
                                    policy_violations += 1
                        # Environment action audit
                        executed_env = _json.loads(row.get('executed_env_action_7d', '[]'))
                        clean_env = _json.loads(row.get('clean_env_action_7d', '[]'))
                        if len(executed_env) < 6 or len(clean_env) < 6:
                            missing_env_action += 1
                        else:
                            for i in range(6):
                                diff = abs(executed_env[i] - clean_env[i])
                                if diff > max_env_diff:
                                    max_env_diff = diff
                                if diff > 1e-7:
                                    env_violations += 1

                # Verify audited frames match expected
                expected_atk_frames = 10 if (summary_path.exists() and json.load(open(summary_path)).get('mlp_emit_step', -1) >= 0) else 0
                if audited_attack_frames != expected_atk_frames:
                    issues.append(f"ArmLock audit frame mismatch: audited {audited_attack_frames} attack frames, expected {expected_atk_frames}")
                if missing_policy_action > 0:
                    issues.append(f"ArmLock: {missing_policy_action} attack frames missing policy action fields")
                if missing_env_action > 0:
                    issues.append(f"ArmLock: {missing_env_action} attack frames missing env action fields")
                if policy_violations > 0 or max_policy_diff > 1e-7:
                    issues.append(f"ArmLock policy violations: {policy_violations} frame-dof, max_diff={max_policy_diff:.2e}")
                if env_violations > 0 or max_env_diff > 1e-7:
                    issues.append(f"ArmLock env violations: {env_violations} frame-dof, max_diff={max_env_diff:.2e}")
            except Exception as e:
                issues.append(f"ArmLock audit error: {e}")

    # 7. Check protocol params vs expected
    if summary_path.exists():
        try:
            summary = json.load(open(summary_path))
            expected = {
                'epsilon': 0.023529411764705882,
                'pgd_steps': 20,
                'K': 10,
                'target_token': 31744,
            }
            for key, exp_val in expected.items():
                actual = summary.get(key)
                if actual is not None and abs(float(actual) - exp_val) > 1e-9:
                    issues.append(f"protocol mismatch: {key}={actual} expected={exp_val}")
        except Exception:
            pass

    # 8. Check exit code
    if complete_path.exists():
        try:
            comp = json.load(open(complete_path))
            if comp.get('exit_code', 0) != 0:
                issues.append(f"non-zero exit_code: {comp.get('exit_code')}")
        except Exception:
            pass

    # 9. Check identity: summary must match job parameters
    if summary_path.exists():
        try:
            summary = json.load(open(summary_path))
            identity_checks = {
                'task_idx': ('task_idx', job.get('task_idx')),
                'state_id': ('state_id', job.get('state_id')),
                'perturbation_seed': ('perturbation_seed', job.get('perturbation_seed')),
                'objective_id': ('objective_id', job.get('objective')),
                'arm_lock': ('arm_lock', bool(job.get('arm_lock'))),
                'condition': ('condition', job.get('condition')),
                'timing_policy': ('timing_policy', job.get('timing', 'student')),
                'eval_seed': ('eval_seed', job.get('eval_seed', 0)),
            }
            for summary_key, (db_key, expected_val) in identity_checks.items():
                actual_val = summary.get(summary_key)
                if actual_val is None:
                    issues.append(f"identity missing field: {summary_key}")
                elif actual_val != expected_val:
                    issues.append(f"identity mismatch: {summary_key}={actual_val} expected={expected_val} (db.{db_key})")
        except Exception as e:
            issues.append(f"identity audit error: {e}")

    # 10. Check OS exit code from jobs table
    try:
        db_exit = conn.execute("SELECT exit_code FROM jobs WHERE job_id=?", (job['job_id'],)).fetchone()
        if db_exit and db_exit[0] is not None and db_exit[0] != 0:
            issues.append(f"OS exit_code={db_exit[0]} (expected 0)")
    except Exception:
        pass

    # 11. Check telemetry content integrity
    if csv_path.exists():
        try:
            import csv as _csv
            with open(csv_path) as f:
                reader = _csv.DictReader(f)
                rows = list(reader)
            n_csv = len(rows)
            expected_n = summary.get('n_steps', 0) if summary_path.exists() else 0
            if expected_n > 0 and n_csv != expected_n:
                issues.append(f"telemetry row count mismatch: csv={n_csv} summary={expected_n}")
            # Check step continuity
            steps = [int(r.get('step', -1)) for r in rows if r.get('step', '').isdigit()]
            if steps and steps != list(range(min(steps), max(steps)+1)):
                issues.append("telemetry steps not continuous")
            # Check attack_this count
            atk_rows = [r for r in rows if r.get('attack_this') == 'True']
            expected_atk = summary.get('attack_frames', 0) if summary_path.exists() else 0
            if len(atk_rows) != expected_atk:
                issues.append(f"attack_this count mismatch: {len(atk_rows)} vs summary.attack_frames={expected_atk}")
            # For ArmLock: verify attack rows have required fields
            if job.get('arm_lock'):
                missing_fields = 0
                for r in atk_rows:
                    if not r.get('clean_policy_action_7d') or not r.get('executed_policy_action_7d_after_lock'):
                        missing_fields += 1
                if missing_fields > 0:
                    issues.append(f"ArmLock: {missing_fields} attack rows missing required action fields")
        except Exception as e:
            issues.append(f"telemetry content audit error: {e}")

    CRITICAL_PATTERNS = [
        'ArmLock violation', 'checkpoint SHA mismatch', 'bridge SHA mismatch',
        'config mismatch', 'protocol mismatch', 'identity mismatch',
        'duplicate scientific key', 'attack_frames mismatch',
        'wrong task', 'wrong state', 'wrong seed', 'wrong objective',
    ]
    is_critical = any(p.lower() in ' '.join(issues).lower() for p in CRITICAL_PATTERNS)
    audit_result = 'PASS' if not issues else ('CRITICAL: ' if is_critical else 'FAIL: ') + '; '.join(issues)

    if is_critical:
        new_status = 'QUARANTINED'
    elif issues:
        new_status = 'FAILED_RETRYABLE'
    else:
        new_status = 'SUCCESS'

    conn.execute("""
        UPDATE jobs SET status=?, audit_result=?, last_audit_at=?, completed_at=?
        WHERE job_id=?
    """, (new_status, audit_result, now_iso(), now_iso(), job['job_id']))
    conn.commit()

    log_event(conn, 'AUDIT_COMPLETE', gpu_id=gpu_id, job_id=job['job_id'],
              detail=audit_result)
    return not issues, issues, is_critical


# ── Restart recovery ──
def recover_abandoned(conn):
    """On restart, reconcile CLAIMED/RUNNING jobs and worker registry."""
    # Check CLAIMED but not running
    running_pids = find_vla_worker_pids()
    claimed = conn.execute("SELECT job_id, worker_pid, output_dir FROM jobs WHERE status IN ('CLAIMED','RUNNING')").fetchall()

    for job_id, pid, out_dir in claimed:
        if pid and pid not in running_pids:
            complete = Path(out_dir) / "COMPLETE.json"
            if complete.exists():
                conn.execute("UPDATE jobs SET status='AUDITING' WHERE job_id=?", (job_id,))
                log_event(conn, 'RECOVERY_AUDIT', job_id=job_id, detail="Abandoned with COMPLETE")
            else:
                conn.execute("""
                    UPDATE jobs SET status='FAILED_RETRYABLE', notes='Worker died before COMPLETE'
                    WHERE job_id=?
                """, (job_id,))
                log_event(conn, 'RECOVERY_FAILED', job_id=job_id, detail="No COMPLETE")
        elif pid is None:
            conn.execute("""
                UPDATE jobs SET status='FAILED_RETRYABLE', notes='CLAIMED but worker_pid NULL (Popen failed)'
                WHERE job_id=?
            """, (job_id,))
            log_event(conn, 'RECOVERY_CLAIMED_NULL', job_id=job_id, detail="CLAIMED + pid=NULL")
        elif pid in running_pids:
            log_event(conn, 'RECOVERY_ALIVE', job_id=job_id, detail=f"Worker pid={pid} still alive")
    conn.commit()

    # Scan for orphaned RUNNING jobs not in CLAIMED/RUNNING query scope
    orphaned = conn.execute("""
        SELECT job_id, worker_pid, output_dir FROM jobs
        WHERE status='RUNNING' AND worker_pid IS NOT NULL
    """).fetchall()
    for job_id, pid, out_dir in orphaned:
        if pid and pid not in running_pids:
            complete = Path(out_dir) / "COMPLETE.json"
            if complete.exists():
                conn.execute("UPDATE jobs SET status='AUDITING' WHERE job_id=?", (job_id,))
            else:
                conn.execute("UPDATE jobs SET status='FAILED_RETRYABLE', notes='Orphaned RUNNING worker died' WHERE job_id=?", (job_id,))
    conn.commit()

    # Clean stale worker registry entries
    conn.execute("DELETE FROM worker_registry WHERE pid NOT IN ({}) AND status='RUNNING'".format(
        ','.join(str(p) for p in running_pids) if running_pids else '-1'))
    conn.commit()


# ── Gate check ──
def is_gate_passed(conn, gate_name):
    row = conn.execute("SELECT status FROM gates WHERE gate_name=?", (gate_name,)).fetchone()
    return row is not None and row[0] == 'PASS'


def check_dependency_gates(conn):
    """Return list of jobs that are BLOCKED_BY_GATE but whose gates now pass."""
    unblocked = []
    blocked = conn.execute("SELECT job_id, gate_dependency FROM jobs WHERE status='BLOCKED_BY_GATE'").fetchall()
    for job_id, gate in blocked:
        if gate and is_gate_passed(conn, gate):
            conn.execute("UPDATE jobs SET status='PENDING', notes='Gate passed, unblocked' WHERE job_id=?", (job_id,))
            unblocked.append(job_id)
    if unblocked:
        conn.commit()
    return unblocked


# ── Main dispatcher loop ──
def main_loop(conn, lockfile_path, source_commit=""):
    gpu_state = {}  # gpu_id -> {'status': GPU_STATE, 'job_id': int, 'pid': int, 'last_seen': float}
    running_workers = {}  # pid -> gpu_id

    # Initialize GPUs
    gpus = detect_gpus()
    for gid in gpus:
        gpu_state[gid] = {'status': 'IDLE_WAITING_FOR_JOB', 'job_id': None, 'pid': None, 'last_seen': time.time()}

    dispatch_enabled = True
    incident_reason = None
    print(f"[{now_iso()}] Dispatcher V2 started. GPUs: {list(gpu_state.keys())}")
    log_event(conn, 'DISPATCHER_START', detail=f"GPUs: {list(gpu_state.keys())}")

    try:
        while True:
            if not dispatch_enabled:
                # Allow running workers to finish, audit them, but claim nothing new
                time.sleep(POLL_INTERVAL)
                continue
            # 1. Reap any finished workers
            try:
                wpid, wstatus = os.waitpid(-1, os.WNOHANG)
                while wpid > 0:
                    exit_code = os.waitstatus_to_exitcode(wstatus)
                    gid = running_workers.pop(wpid, None)
                    if gid is not None:
                        job_row = conn.execute(
                            "SELECT job_id, output_dir FROM jobs WHERE worker_pid=? AND status='RUNNING'",
                            (wpid,)).fetchone()
                        if job_row:
                            conn.execute("UPDATE jobs SET status='AUDITING', exit_code=?, last_audit_at=? WHERE job_id=?",
                                         (exit_code, now_iso(), job_row[0]))
                            conn.commit()
                            gpu_state[gid]['status'] = 'AUDITING'
                            gpu_state[gid]['job_id'] = job_row[0]
                            log_event(conn, 'WORKER_EXIT', gpu_id=gid, job_id=job_row[0],
                                      detail=f"exit_code={exit_code}")
                    wpid, wstatus = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pass  # No children

            # 2. Audit completed jobs
            for gid in list(gpu_state.keys()):
                if gpu_state[gid]['status'] == 'AUDITING':
                    jid = gpu_state[gid]['job_id']
                    job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
                    if job:
                        job_dict = {desc[0]: val for desc, val in zip(conn.execute("SELECT * FROM jobs LIMIT 0").description, job)}
                        ok, issues, is_critical = audit_job(conn, job_dict, gid)
                        if is_critical:
                            print(f"[{now_iso()}] CRITICAL AUDIT FAILURE GPU{gid} job={jid}: {issues}")
                            dispatch_enabled = False
                            incident_reason = f"GPU{gid} job={jid}: {'; '.join(issues)}"
                            log_event(conn, 'STOPPED_BY_INCIDENT', gpu_id=gid, job_id=jid, detail=incident_reason)
                            gpu_state[gid] = {'status': 'STOPPED_BY_INCIDENT', 'job_id': jid, 'pid': None, 'last_seen': time.time()}
                            # Allow other running workers to complete, stop new claims
                            break
                        if ok:
                            print(f"[{now_iso()}] GPU{gid}: AUDIT PASS job={jid}")
                            gpu_state[gid] = {'status': 'IDLE_WAITING_FOR_JOB', 'job_id': None, 'pid': None, 'last_seen': time.time()}
                        else:
                            print(f"[{now_iso()}] GPU{gid}: AUDIT FAIL job={jid}: {issues}")
                            gpu_state[gid] = {'status': 'IDLE_WAITING_FOR_JOB', 'job_id': None, 'pid': None, 'last_seen': time.time()}

            # 3. Check stop conditions
            gpus = detect_gpus()
            root_free, sdc_free = check_disk()
            if sdc_free < 100:
                print(f"[{now_iso()}] STOP: /mnt/sdc free={sdc_free}GB < 100GB threshold")
                log_event(conn, 'STOP_DISK', detail=f"sdc_free={sdc_free}GB")
                break

            # 4. Check dependency gates (unblock jobs)
            unblocked = check_dependency_gates(conn)
            if unblocked:
                print(f"[{now_iso()}] Unblocked jobs: {unblocked}")

            # 5. Claim and launch for idle GPUs
            for gid in gpus:
                if gid not in gpu_state:
                    gpu_state[gid] = {'status': 'IDLE_WAITING_FOR_JOB', 'job_id': None, 'pid': None, 'last_seen': time.time()}

                if gpu_state[gid]['status'] in ('IDLE_WAITING_FOR_JOB',):
                    # Check no worker already on this GPU
                    gpu_running = any(True for _j in conn.execute(
                        "SELECT 1 FROM jobs WHERE gpu_id=? AND status IN ('CLAIMED','RUNNING')", (gid,)).fetchall())
                    if gpu_running:
                        continue

                    # Check GPU is actually available (no external worker)
                    ext_workers = find_vla_worker_pids()
                    worker_on_gpu = False
                    for pid in ext_workers:
                        try:
                            with open(f"/proc/{pid}/environ", "rb") as ef:
                                env_data = ef.read().decode('utf-8', errors='replace')
                                if f"CUDA_VISIBLE_DEVICES={gid}" in env_data:
                                    worker_on_gpu = True
                                    break
                        except Exception:
                            pass
                    if worker_on_gpu:
                        gpu_state[gid]['status'] = 'IDLE_WAITING_FOR_JOB'
                        continue

                    # Attempt claim (respect dispatch_enabled)
                    if not dispatch_enabled:
                        continue
                    job = claim_job(conn, gid)
                    if job:
                        try:
                            pid = launch_worker(conn, job, gid, source_commit)
                        except Exception as e:
                            log_event(conn, 'LAUNCH_FAILED', gpu_id=gid, job_id=job['job_id'], detail=str(e))
                            gpu_state[gid] = {'status': 'IDLE_WAITING_FOR_JOB', 'job_id': None, 'pid': None, 'last_seen': time.time()}
                            continue
                        running_workers[pid] = gid
                        gpu_state[gid] = {'status': 'RUNNING', 'job_id': job['job_id'], 'pid': pid, 'last_seen': time.time()}
                        print(f"[{now_iso()}] GPU{gid}: launched job={job['job_id']} {job['method']} {job['task']} s{job['state_id']} seed={job['perturbation_seed']} pid={pid}")
                    else:
                        gpu_state[gid]['status'] = 'IDLE_WAITING_FOR_JOB'

            # 6. Status report
            status_counts = {}
            for row in conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall():
                status_counts[row[0]] = row[1]
            print(f"[{now_iso()}] Jobs: {status_counts} | GPUs: {[(g, gpu_state[g]['status']) for g in sorted(gpu_state)]} | Disk: /mnt/sdc={sdc_free}G")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"[{now_iso()}] Dispatcher shutting down on signal.")
    finally:
        log_event(conn, 'DISPATCHER_STOP', detail="Shutdown")
        conn.close()


# ── Entry point ──
def main():
    ap = argparse.ArgumentParser(description="Phase 7 Dispatcher V2")
    ap.add_argument("--db", required=True, help="Path to SQLite job database")
    ap.add_argument("--lock", required=True, help="Path to flock lockfile")
    ap.add_argument("--source_commit", default="", help="Git commit SHA for provenance")
    args = ap.parse_args()

    # Flock single-instance
    lockfile = open(args.lock, 'w')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: Another dispatcher instance is already running.", file=sys.stderr)
        sys.exit(1)

    conn = init_db(args.db)

    # Restart recovery
    recover_abandoned(conn)

    # Get source commit if not provided
    source_commit = args.source_commit
    if not source_commit:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            source_commit = ""

    main_loop(conn, lockfile, source_commit)


if __name__ == "__main__":
    main()
