"""Atomically unlock exactly 1 parent for Wave-0. Must be run after queue init.
Asserts: 20 total, 20 LOCKED, 0 PENDING, 0 RUNNING, 0 DONE.
Unlocks the first parent (deterministic: sorted cell_id, first).
"""
import sys, os, json, sqlite3
sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v2')
conn = q._get_conn()

try:
    conn.execute("BEGIN IMMEDIATE")

    # Assert preconditions
    total = conn.execute("SELECT COUNT(*) as n FROM tasks").fetchone()['n']
    locked = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='LOCKED'").fetchone()['n']
    pending = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='PENDING'").fetchone()['n']
    running = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state IN ('LEASED','RUNNING')").fetchone()['n']
    done = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='DONE_VALID'").fetchone()['n']

    assert total == 20, 'Expected 20 tasks, got %d' % total
    assert locked == 20, 'Expected 20 LOCKED, got %d' % locked
    assert pending == 0, 'Expected 0 PENDING, got %d' % pending
    assert running == 0, 'Expected 0 RUNNING, got %d' % running
    assert done == 0, 'Expected 0 DONE, got %d' % done

    # Unlock first parent by sorted cell_id
    first = conn.execute("SELECT cell_id FROM tasks WHERE state='LOCKED' ORDER BY cell_id ASC LIMIT 1").fetchone()
    assert first is not None, 'No LOCKED task found'

    conn.execute("UPDATE tasks SET state='PENDING', updated_at=? WHERE cell_id=?",
                 (q._now(), first['cell_id']))

    # Assert postconditions
    locked2 = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='LOCKED'").fetchone()['n']
    pending2 = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='PENDING'").fetchone()['n']
    assert locked2 == 19, 'Expected 19 LOCKED, got %d' % locked2
    assert pending2 == 1, 'Expected 1 PENDING, got %d' % pending2

    conn.commit()
    print('Wave-0 UNLOCKED: %s' % first['cell_id'])
    print('  locked: 20 -> %d' % locked2)
    print('  pending: 0 -> %d' % pending2)
    print('  Verify with: SELECT state, COUNT(*) FROM tasks GROUP BY state')
except Exception as e:
    conn.rollback()
    print('Wave-0 UNLOCK FAILED: %s' % e)
    sys.exit(1)
finally:
    q.close()
