"""Atomic task queue for FEC formal matrix execution. SQLite WAL, lease fencing."""
import json, os, sqlite3, time, uuid, threading
from datetime import datetime, timezone

SCHEMA_VERSION = 1
LEASE_TIMEOUT_SEC = 300


class AtomicTaskQueue:
    def __init__(self, db_path, run_id=None):
        self.db_path = db_path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._local = threading.local()
        with self._get_conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS run_meta (
                run_id TEXT PRIMARY KEY, state TEXT NOT NULL DEFAULT 'INIT',
                formal_manifest_sha TEXT, source_sha TEXT, config_sha TEXT,
                capacity_policy_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                cell_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL,
                suite TEXT NOT NULL, task_index INTEGER NOT NULL, state_index INTEGER NOT NULL,
                arm TEXT NOT NULL, task_kind TEXT NOT NULL DEFAULT 'FORMAL_CELL',
                estimated_cost REAL NOT NULL DEFAULT 1.0, priority INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'PENDING', attempt_count INTEGER NOT NULL DEFAULT 0,
                accepted_attempt_id TEXT, lease_owner TEXT, lease_token TEXT,
                lease_epoch INTEGER NOT NULL DEFAULT 0, lease_expires_at TEXT, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, cell_id TEXT NOT NULL, lease_epoch INTEGER NOT NULL,
                worker_id TEXT NOT NULL, hostname TEXT, pid INTEGER, gpu_id INTEGER, slot_id INTEGER,
                loaded_suite TEXT, state TEXT NOT NULL DEFAULT 'INIT', started_at TEXT,
                heartbeat_at TEXT, ended_at TEXT, exit_code INTEGER, error_class TEXT,
                exposure_status TEXT, task_outcome TEXT, output_dir TEXT, receipt_sha TEXT,
                peak_memory_mb REAL, nvml_peak_mb REAL, FOREIGN KEY (cell_id) REFERENCES tasks(cell_id));
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                worker_id TEXT, cell_id TEXT, event_type TEXT NOT NULL, payload_json TEXT);
            CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
            CREATE INDEX IF NOT EXISTS idx_tasks_suite ON tasks(suite, state);
            """)

    def _get_conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close(self):
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _log(self, worker_id, cell_id, event_type, payload=None):
        with self._get_conn() as conn:
            conn.execute("INSERT INTO events (timestamp,worker_id,cell_id,event_type,payload_json) VALUES (?,?,?,?,?)",
                         (self._now(), worker_id, cell_id, event_type, json.dumps(payload) if payload else None))

    # ── Run management ──
    def init_run(self, state='RUNNING', manifest_sha=None, source_sha=None, config_sha=None, capacity_policy=None):
        now = self._now()
        with self._get_conn() as conn:
            conn.execute("""INSERT OR REPLACE INTO run_meta (run_id,state,formal_manifest_sha,source_sha,config_sha,
                           capacity_policy_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                         (self.run_id, state, manifest_sha, source_sha, config_sha,
                          json.dumps(capacity_policy) if capacity_policy else None, now, now))
        self._log(None, None, 'RUN_INIT', {'state': state})

    def get_run_state(self):
        row = self._get_conn().execute("SELECT state FROM run_meta WHERE run_id=?", (self.run_id,)).fetchone()
        return row['state'] if row else 'INIT'

    def set_run_state(self, state):
        with self._get_conn() as conn:
            conn.execute("UPDATE run_meta SET state=?, updated_at=? WHERE run_id=?", (state, self._now(), self.run_id))
        self._log(None, None, 'RUN_STATE_CHANGE', {'new_state': state})

    # ── Task registration ──
    def register_tasks(self, cells):
        now = self._now()
        with self._get_conn() as conn:
            for cell in cells:
                conn.execute("""INSERT OR IGNORE INTO tasks (cell_id,parent_id,suite,task_index,state_index,arm,
                               task_kind,estimated_cost,priority,state,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                             (cell['cell_id'], cell['parent_id'], cell['suite'], cell['task_index'],
                              cell['state_index'], cell['arm'], cell.get('task_kind', 'FORMAL_CELL'),
                              cell.get('estimated_cost', 1.0), cell.get('priority', 0), 'PENDING', now))

    def lock_all_tasks(self):
        with self._get_conn() as conn:
            conn.execute("UPDATE tasks SET state='LOCKED' WHERE state='PENDING'")

    def unlock_tasks(self, cell_ids):
        with self._get_conn() as conn:
            for cid in cell_ids:
                conn.execute("UPDATE tasks SET state='PENDING' WHERE cell_id=? AND state='LOCKED'", (cid,))

    # ── Atomic claim ──
    def claim_task(self, worker_id, hostname=None, pid=None, gpu_id=None, slot_id=None,
                   loaded_suite=None, expected_manifest_sha=None, expected_source_sha=None):
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT state,formal_manifest_sha,source_sha FROM run_meta WHERE run_id=?",
                               (self.run_id,)).fetchone()
            if not run or run['state'] not in ('RUNNING', 'ACTIVE'):
                conn.rollback(); return None
            if expected_manifest_sha and run['formal_manifest_sha'] != expected_manifest_sha:
                conn.rollback(); return None
            if expected_source_sha and run['source_sha'] != expected_source_sha:
                conn.rollback(); return None

            row = conn.execute("""SELECT * FROM tasks WHERE state IN ('PENDING','RETRY_READY')
                ORDER BY CASE WHEN suite=? THEN 0 ELSE 1 END, estimated_cost DESC, cell_id ASC LIMIT 1""",
                               (loaded_suite or '',)).fetchone()
            if not row:
                conn.rollback(); return None

            cell = dict(row)
            lease_token = uuid.uuid4().hex
            new_epoch = cell['lease_epoch'] + 1
            now = self._now()
            expires = datetime.fromtimestamp(time.time() + LEASE_TIMEOUT_SEC, tz=timezone.utc).isoformat()

            cur = conn.execute("""UPDATE tasks SET state='LEASED',lease_owner=?,lease_token=?,lease_epoch=?,
                       lease_expires_at=?,attempt_count=attempt_count+1,updated_at=?
                       WHERE cell_id=? AND state IN ('PENDING','RETRY_READY')""",
                               (worker_id, lease_token, new_epoch, expires, now, cell['cell_id']))
            if cur.rowcount == 0:
                conn.rollback(); return None

            attempt_id = "%s_%d_%s" % (cell['cell_id'], new_epoch, uuid.uuid4().hex[:8])
            conn.execute("""INSERT INTO attempts (attempt_id,cell_id,lease_epoch,worker_id,hostname,pid,
                           gpu_id,slot_id,loaded_suite,state,started_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                         (attempt_id, cell['cell_id'], new_epoch, worker_id, hostname, pid,
                          gpu_id, slot_id, loaded_suite, 'LEASED', now))
            self._log(worker_id, cell['cell_id'], 'TASK_CLAIMED',
                      {'attempt_id': attempt_id, 'epoch': new_epoch})
            conn.commit()
            cell['attempt_id'] = attempt_id
            cell['lease_token'] = lease_token
            cell['lease_epoch'] = new_epoch
            return cell
        except Exception:
            try: conn.rollback()
            except: pass
            raise

    # ── Heartbeat ──
    def heartbeat(self, cell_id, attempt_id, worker_id, lease_token, lease_epoch):
        with self._get_conn() as conn:
            row = conn.execute("SELECT lease_token,lease_epoch FROM tasks WHERE cell_id=? AND lease_owner=?",
                               (cell_id, worker_id)).fetchone()
            if not row or row['lease_token'] != lease_token or row['lease_epoch'] != lease_epoch:
                return False
            conn.execute("UPDATE attempts SET heartbeat_at=?,state='RUNNING' WHERE attempt_id=?",
                         (self._now(), attempt_id))
            expires = datetime.fromtimestamp(time.time() + LEASE_TIMEOUT_SEC, tz=timezone.utc).isoformat()
            conn.execute("UPDATE tasks SET lease_expires_at=? WHERE cell_id=?", (expires, cell_id))
        return True

    # ── Commit result ──
    def commit_result(self, cell_id, attempt_id, worker_id, lease_token, lease_epoch,
                      exit_code=0, error_class=None, exposure_status=None, task_outcome=None,
                      output_dir=None, receipt_sha=None, peak_memory_mb=None, nvml_peak_mb=None):
        """Commit attempt. On success (task_outcome=DONE_VALID): set accepted_attempt_id.
        On failure: mark task FAILED or RETRY_READY, do NOT set accepted_attempt_id."""
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # P0-5: Validate task lease AND attempt row matches
            task = conn.execute("""SELECT * FROM tasks WHERE cell_id=? AND lease_owner=?
                                  AND lease_token=? AND lease_epoch=?""",
                                (cell_id, worker_id, lease_token, lease_epoch)).fetchone()
            if not task:
                conn.rollback(); return False
            if task['accepted_attempt_id'] is not None:
                conn.rollback(); return False

            # Validate attempt row
            attempt = conn.execute("""SELECT * FROM attempts WHERE attempt_id=? AND cell_id=?
                                     AND worker_id=? AND lease_epoch=?""",
                                   (attempt_id, cell_id, worker_id, lease_epoch)).fetchone()
            if not attempt:
                conn.rollback(); return False

            now = self._now()
            is_success = (task_outcome == 'DONE_VALID' or task_outcome == 'DONE')

            if is_success:
                # P0-3: Only set DONE_VALID + accepted_attempt_id on success
                new_state = 'DONE_VALID'
                conn.execute("UPDATE tasks SET state=?,accepted_attempt_id=?,updated_at=? WHERE cell_id=?",
                             (new_state, attempt_id, now, cell_id))
            elif task_outcome == 'FAILED':
                # Infrastructure failure: retry-ready with incremented attempt count
                new_state = 'RETRY_READY'
                conn.execute("""UPDATE tasks SET state=?,attempt_count=attempt_count+1,
                               lease_owner=NULL,lease_token=NULL,updated_at=? WHERE cell_id=?""",
                             (new_state, now, cell_id))
            elif task_outcome == 'CLASSIFIED':
                # Scientific result but not clean PASS (e.g., CLASS_C terminal censor)
                new_state = 'DONE_VALID'
                conn.execute("UPDATE tasks SET state=?,accepted_attempt_id=?,updated_at=? WHERE cell_id=?",
                             (new_state, attempt_id, now, cell_id))
            else:
                new_state = task_outcome or 'DONE_VALID'
                conn.execute("UPDATE tasks SET state=?,accepted_attempt_id=?,updated_at=? WHERE cell_id=?",
                             (new_state, attempt_id, now, cell_id))

            conn.execute("""UPDATE attempts SET state=?,ended_at=?,exit_code=?,error_class=?,
                           exposure_status=?,task_outcome=?,output_dir=?,receipt_sha=?,
                           peak_memory_mb=?,nvml_peak_mb=? WHERE attempt_id=?""",
                         (new_state, now, exit_code, error_class, exposure_status, task_outcome,
                          output_dir, receipt_sha, peak_memory_mb, nvml_peak_mb, attempt_id))
            self._log(worker_id, cell_id, 'TASK_COMMITTED',
                      {'attempt_id': attempt_id, 'outcome': task_outcome, 'state': new_state})
            conn.commit()
            return True
        except Exception:
            try: conn.rollback()
            except: pass
            raise

    # ── Reaper ──
    def reap_expired_leases(self):
        conn = self._get_conn()
        now = self._now()
        expired = conn.execute("SELECT * FROM tasks WHERE state='LEASED' AND lease_expires_at < ?",
                               (now,)).fetchall()
        results = []
        with conn:
            for task in expired:
                cell = dict(task)
                conn.execute("""UPDATE tasks SET state='RETRY_READY',lease_owner=NULL,lease_token=NULL,
                               updated_at=? WHERE cell_id=? AND state='LEASED'""",
                             (now, cell['cell_id']))
                conn.execute("""UPDATE attempts SET state='SUPERSEDED',ended_at=?
                               WHERE cell_id=? AND lease_epoch=? AND state IN ('LEASED','RUNNING')""",
                             (now, cell['cell_id'], cell['lease_epoch']))
                self._log(None, cell['cell_id'], 'LEASE_EXPIRED', {'epoch': cell['lease_epoch']})
                results.append(cell)
        return results

    # ── Progress ──
    def get_progress(self):
        conn = self._get_conn()
        return {
            'total': conn.execute("SELECT COUNT(*) as n FROM tasks").fetchone()['n'],
            'done': conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='DONE_VALID'").fetchone()['n'],
            'running': conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state IN ('LEASED','RUNNING','COMMITTING')").fetchone()['n'],
            'pending': conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='PENDING'").fetchone()['n'],
            'retry_ready': conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='RETRY_READY'").fetchone()['n'],
            'locked': conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='LOCKED'").fetchone()['n'],
            'superseded_attempts': conn.execute("SELECT COUNT(*) as n FROM attempts WHERE state='SUPERSEDED'").fetchone()['n'],
        }
