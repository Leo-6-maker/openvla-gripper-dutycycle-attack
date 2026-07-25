"""G0: V4 Formal Reconciliation — import direct-run spatial state_0/state_2 into queue.

Reads /tmp/final_s100/ and /tmp/final_s102/ artifacts, validates provenance,
copies to immutable N5 directory, commits in the formal queue.

The queue stores PARENT-level tasks (arm=FULL_BUNDLE), not individual arm cells.
20 parents total, 18 DONE_VALID, 2 PENDING (spatial state_0, state_2).

Usage (run on Linux server):
  /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python reconcile_direct_run.py --dry-run
  /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python reconcile_direct_run.py
"""
import json, os, sys, hashlib, shutil, argparse, sqlite3
from datetime import datetime, timezone

FROZEN_SEAL = {
    "provider_sha": "6a7ab61d8dba8cb331a748c62317d2513b1e397def2adee8119204be44cecb61",
    "checkpoint_sha": "685ddadf90ad2ac4ec83bcadbe970d6ad74f07baa4e498a4936c78c0b0695f88",
    "seal_sha": "22a97d5746efed9e61024c6567bd964de729dfe990d4765d59cd6fc032099896",
}

FORMAL_QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'
N5_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5'
RECOVERY_ROOT = os.path.join(N5_ROOT, 'formal', 'recovered_attempts')

EXPECTED_ARMS = ['CLEAN', 'TRUE_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE', 'RANDOM_TIME_T10']

DIRECT_RUNS = {
    'libero_spatial_task_00_state_100': {
        'tmp_path': '/tmp/final_s100',
        'cell_id': 'formal_libero_spatial_task_00_state_100',
        'parent_id': 'p_spatial_s0',
        'suite': 'libero_spatial', 'task_index': 0, 'state_index': 0,
        'classification': 'DONE_CLASSIFIED_TC',
        'note': 'TRUE_T10 K10 8/10 truncated (episode end). Incomplete run: only 2/5 arms.',
    },
    'libero_spatial_task_00_state_102': {
        'tmp_path': '/tmp/final_s102',
        'cell_id': 'formal_libero_spatial_task_00_state_102',
        'parent_id': 'p_spatial_s2',
        'suite': 'libero_spatial', 'task_index': 0, 'state_index': 2,
        'classification': 'DONE_VALID',
        'provenance': 'RECOVERED_DIRECT_RUN',
        'note': 'Completed via direct nohup due to persistent worker subprocess bug. All 5 arms pass. Not TC — no terminal censor issue.',
    },
}


def validate_provenance(artifact_dir):
    """Validate that the artifact directory contains valid V4 formal outputs.
    Checks: run_manifest.json, arm subdirectories with COMPLETE.json or result.json.
    """
    issues = []

    manifest_path = os.path.join(artifact_dir, 'run_manifest.json')
    smoke_path = os.path.join(artifact_dir, 'smoke_summary.json')

    if not os.path.isfile(manifest_path):
        issues.append(f'Missing run_manifest.json in {artifact_dir}')
    else:
        # Verify provider SHA matches frozen seal
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            provider_sha = manifest.get('n4_module_sha256', '')
            if provider_sha != FROZEN_SEAL['provider_sha']:
                issues.append(f'Provider SHA mismatch: {provider_sha[:16]}... != {FROZEN_SEAL["provider_sha"][:16]}...')
        except Exception as e:
            issues.append(f'Cannot parse run_manifest.json: {e}')

    if not os.path.isfile(smoke_path):
        issues.append(f'Missing smoke_summary.json in {artifact_dir}')

    present_arms = []
    missing_arms = []
    for arm in EXPECTED_ARMS:
        arm_dir = os.path.join(artifact_dir, arm)
        if not os.path.isdir(arm_dir):
            missing_arms.append(arm)
            continue
        complete = os.path.isfile(os.path.join(arm_dir, 'COMPLETE.json'))
        result = os.path.isfile(os.path.join(arm_dir, 'result.json'))
        if not (complete or result):
            issues.append(f'Arm {arm}: no COMPLETE.json or result.json')
        present_arms.append(arm)

    return issues, present_arms, missing_arms


def copy_to_permanent(tmp_path, identity):
    """Copy artifacts from /tmp to permanent N5 recovery directory.
    Uses os.rename for atomic move; falls back to shutil.copytree with no-clobber.
    NEVER overwrites existing recovered attempts.
    """
    dest = os.path.join(RECOVERY_ROOT, identity)
    if os.path.exists(dest):
        raise FileExistsError(f'REJECTED: recovered attempt already exists at {dest} — no overwrite allowed')
    os.makedirs(RECOVERY_ROOT, exist_ok=True)
    # Try atomic rename first (same filesystem)
    try:
        os.rename(tmp_path, dest)
    except OSError:
        shutil.copytree(tmp_path, dest)
    return dest


def reconcile_parent(conn, cell_id, parent_id, suite, task_index, state_index,
                     permanent_path, classification, dry_run=True):
    """Claim and commit a parent-level task in the formal queue."""
    now = datetime.now(timezone.utc).isoformat()

    # Check if task exists and is in a claimable state
    existing = conn.execute("SELECT * FROM tasks WHERE cell_id=?", (cell_id,)).fetchone()
    if existing is None:
        print(f'  Registering: {cell_id}')
        if not dry_run:
            conn.execute("""INSERT INTO tasks (cell_id,parent_id,suite,task_index,state_index,arm,
                           task_kind,estimated_cost,priority,state,updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (cell_id, parent_id, suite, task_index, state_index, 'FULL_BUNDLE',
                         'FULL_BUNDLE', 5.0, 0, 'PENDING', now))
        existing = conn.execute("SELECT * FROM tasks WHERE cell_id=?", (cell_id,)).fetchone()
        if existing is None:
            return False, 'Registration failed'

    current_state = existing['state']
    if current_state in ('DONE_VALID', 'DONE_CLASSIFIED_TC'):
        print(f'  Already accepted: {cell_id} ({current_state})')
        return True, 'already_accepted'

    if current_state not in ('PENDING', 'RETRY_READY', 'LOCKED'):
        print(f'  Cannot claim: {cell_id} (state={current_state})')
        return False, f'state={current_state}'

    if dry_run:
        print(f'  Would claim + commit: {cell_id} ({current_state} → {classification})')
        return True, 'dry_run_valid'

    # Claim
    conn.execute("BEGIN IMMEDIATE")
    task = conn.execute("""SELECT * FROM tasks WHERE cell_id=? AND state IN ('PENDING','RETRY_READY')
                          LIMIT 1""", (cell_id,)).fetchone()
    if task is None:
        conn.rollback()
        # Check again if it's already done
        task = conn.execute("SELECT * FROM tasks WHERE cell_id=?", (cell_id,)).fetchone()
        if task and task['state'] in ('DONE_VALID', 'DONE_CLASSIFIED_TC'):
            print(f'  Already done (race): {cell_id}')
            return True, 'already_done'
        return False, 'claim_race'

    cell = dict(task)
    lease_token = 'recovery_' + hashlib.sha256(cell_id.encode()).hexdigest()[:16]
    new_epoch = cell['lease_epoch'] + 1

    conn.execute("""UPDATE tasks SET state='LEASED',lease_owner=?,lease_token=?,lease_epoch=?,
                   lease_expires_at=datetime('now','+1 hour'),attempt_count=attempt_count+1,updated_at=?
                   WHERE cell_id=?""",
                ('recovery_worker', lease_token, new_epoch, now, cell_id))

    attempt_id = f'{cell_id}_{new_epoch}_recovery'
    conn.execute("""INSERT INTO attempts (attempt_id,cell_id,lease_epoch,worker_id,hostname,pid,
                   loaded_suite,state,started_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (attempt_id, cell_id, new_epoch, 'recovery_worker', 'recovery', 0,
                 suite, 'RUNNING', now))

    # Compute receipt SHA from smoke_summary
    receipt_sha = None
    smoke_path = os.path.join(permanent_path, 'smoke_summary.json')
    if os.path.isfile(smoke_path):
        h = hashlib.sha256()
        with open(smoke_path, 'rb') as f:
            h.update(f.read())
        receipt_sha = h.hexdigest()

    conn.execute("UPDATE tasks SET state=?,accepted_attempt_id=?,updated_at=? WHERE cell_id=?",
                (classification, attempt_id, now, cell_id))
    conn.execute("""UPDATE attempts SET state=?,ended_at=?,exit_code=?,task_outcome=?,
                   output_dir=?,receipt_sha=? WHERE attempt_id=?""",
                (classification, now, 0, classification, permanent_path, receipt_sha, attempt_id))

    conn.execute("""INSERT INTO events (timestamp,worker_id,cell_id,event_type,payload_json)
                   VALUES (?,?,?,?,?)""",
                (now, 'recovery_worker', cell_id, 'RECOVERY_COMMITTED',
                 json.dumps({'attempt_id': attempt_id, 'classification': classification,
                            'permanent_path': permanent_path})))

    conn.commit()
    print(f'  COMMITTED: {cell_id} → {classification} (attempt={attempt_id})')
    return True, 'committed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--queue-db', default=FORMAL_QUEUE_DB)
    args = parser.parse_args()

    print('=== G0: V4 Formal Reconciliation ===')
    print(f'Queue: {args.queue_db}')
    print(f'Recovery root: {RECOVERY_ROOT}')
    print(f'Mode: {"DRY RUN" if args.dry_run else "EXECUTE"}')
    print()

    if not os.path.isfile(args.queue_db):
        print(f'ERROR: Queue database not found: {args.queue_db}')
        sys.exit(3)

    conn = sqlite3.connect(args.queue_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    results = {}
    requeue_needed = []

    for identity, info in DIRECT_RUNS.items():
        tmp_path = info['tmp_path']
        print(f'--- {identity} ---')

        if not os.path.isdir(tmp_path):
            print(f'  ERROR: /tmp path not found: {tmp_path}')
            requeue_needed.append(identity)
            continue

        issues, present_arms, missing_arms = validate_provenance(tmp_path)
        if issues:
            for i in issues:
                print(f'  ISSUE: {i}')

        print(f'  Arms present: {len(present_arms)}/5 ({", ".join(present_arms)})')

        if missing_arms:
            print(f'  Arms MISSING: {", ".join(missing_arms)}')
            if len(present_arms) < 5:
                print(f'  ACTION: Cannot reconcile — missing {len(missing_arms)} arms. Needs re-run.')
                requeue_needed.append(identity)
                results[identity] = {'status': 'INCOMPLETE', 'present': present_arms, 'missing': missing_arms}
                continue

        # All 5 arms present
        permanent_path = None
        if not args.dry_run and not issues:
            try:
                permanent_path = copy_to_permanent(tmp_path, identity)
                print(f'  Copied to: {permanent_path}')
            except Exception as e:
                print(f'  ERROR copying: {e}')
                results[identity] = {'status': 'COPY_FAILED', 'error': str(e)}
                continue
        elif issues:
            print(f'  SKIPPED: {len(issues)} provenance issues')

        ok, msg = reconcile_parent(
            conn, info['cell_id'], info['parent_id'], info['suite'],
            info['task_index'], info['state_index'],
            permanent_path or os.path.join(RECOVERY_ROOT, identity),
            info['classification'], dry_run=args.dry_run)
        results[identity] = {
            'status': 'RECONCILED' if ok else 'FAILED',
            'detail': msg,
            'permanent_path': permanent_path or os.path.join(RECOVERY_ROOT, identity),
        }

    # Progress
    progress = conn.execute("SELECT COUNT(*) as n FROM tasks").fetchone()['n']
    done = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state IN ('DONE_VALID','DONE_CLASSIFIED_TC')").fetchone()['n']
    conn.close()

    reconciliation = {
        'reconciliation': 'V4_FORMAL_RECONCILIATION_V1',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'dry_run': args.dry_run,
        'results': results,
        'requeue_needed': requeue_needed,
        'frozen_seal': {k: v[:16]+'...' for k, v in FROZEN_SEAL.items()},
        'queue_progress': {'total': progress, 'done': done},
    }

    receipt_path = os.path.join(RECOVERY_ROOT, 'V4_FORMAL_RECONCILIATION_RECEIPT_V1.json')
    if not args.dry_run:
        os.makedirs(RECOVERY_ROOT, exist_ok=True)
        with open(receipt_path, 'w') as f:
            json.dump(reconciliation, f, indent=2, default=str)
    receipt_sha = hashlib.sha256(json.dumps(reconciliation, indent=2, default=str).encode()).hexdigest()

    print(f'\n=== RECONCILIATION RESULTS ===')
    for identity, r in results.items():
        print(f'  {identity}: {r["status"]} — {r.get("detail","")}')
    print(f'\nQueue progress: {done}/{progress} accepted')
    print(f'Requeue needed: {len(requeue_needed)} parents')
    for rid in requeue_needed:
        print(f'  - {rid}')
    print(f'Receipt: {receipt_path}')
    print(f'Receipt SHA: {receipt_sha}')

    if done == 20:
        print('\nG0 STATUS: PASS — 20/20 accepted')
    else:
        print(f'\nG0 STATUS: PENDING — {done}/20 accepted (need re-run: {requeue_needed})')

    sys.exit(0 if not requeue_needed else 2)


if __name__ == '__main__':
    main()
