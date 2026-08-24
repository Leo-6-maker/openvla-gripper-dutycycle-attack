"""Self-test for atomic task queue. Covers required scenarios."""
import json, os, sys, tempfile, time, uuid, threading, sqlite3
sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

PASS = 0; FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print("  PASS: %s" % name)
    else: FAIL += 1; print("  FAIL: %s %s" % (name, detail))

def new_queue(name):
    db = os.path.join(tempfile.mkdtemp(), 'q_%s.sqlite' % name)
    q = AtomicTaskQueue(db, run_id=name)
    q.init_run(state='RUNNING')
    return q, db

print("\n=== ATOMIC QUEUE SELF-TEST ===\n")

# ── Test 1: 32 concurrent claimants ──
print("Test 1: 32 concurrent claimants, 100 tasks")
q1, db1 = new_queue('t1')
cells = []
for i in range(100):
    suite = ['libero_10','libero_goal','libero_object','libero_spatial'][i % 4]
    arm = ['CLEAN','TRUE_T10','RAND_T10','ORACLE_T10','RANDOM_TIME_T10'][i % 5]
    cells.append({'cell_id': 'cell_%03d' % i, 'parent_id': 'parent_%02d' % (i//5),
                  'suite': suite, 'task_index': 0, 'state_index': i, 'arm': arm,
                  'task_kind': 'FORMAL_CELL', 'estimated_cost': 10.0 - (i%5)*0.5})
q1.register_tasks(cells)
accepted = {}
dup_errors = []
lock = threading.Lock()

def claimant(wid):
    for _ in range(10):
        t = q1.claim_task('w%d' % wid, loaded_suite='libero_10')
        if t is None: break
        cid = t['cell_id']
        with lock:
            if cid in accepted: dup_errors.append(cid)
            accepted[cid] = wid
        time.sleep(0.001)
        q1.heartbeat(cid, t['attempt_id'], 'w%d'%wid, t['lease_token'], t['lease_epoch'])
        q1.commit_result(cid, t['attempt_id'], 'w%d'%wid, t['lease_token'], t['lease_epoch'],
                         exposure_status='TEST', task_outcome='SUCCESS')

threads = [threading.Thread(target=claimant, args=(i,)) for i in range(32)]
for t in threads: t.start()
for t in threads: t.join()
q1.close()

check("No duplicate claims", len(dup_errors) == 0, str(dup_errors[:3]))
check("All 100 tasks done", len(accepted) == 100, "only %d" % len(accepted))

# ── Test 2: Crash recovery ──
print("\nTest 2: Crash recovery (lease expiry + reaper)")
q2, db2 = new_queue('t2')
q2.register_tasks([{'cell_id': 'crash_%d'%i, 'parent_id': 'p%d'%i, 'suite': 'libero_10',
                    'task_index': 0, 'state_index': i, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL'}
                   for i in range(5)])

claims = [q2.claim_task('crash_w', hostname='test', pid=99999, loaded_suite='libero_10') for _ in range(3)]
for i, t in enumerate(claims):
    check("Claim %d succeeds" % i, t is not None)
q2.close()

# Force-expire leases via raw SQL
raw = sqlite3.connect(db2)
raw.execute("UPDATE tasks SET lease_expires_at='2020-01-01T00:00:00' WHERE lease_owner='crash_w'")
raw.commit(); raw.close()

q2b = AtomicTaskQueue(db2, run_id='t2')
expired = q2b.reap_expired_leases()
check("Reaper found expired", len(expired) == 3, "found %d" % len(expired))
check("Retry-ready count", q2b.get_progress()['retry_ready'] == 3)

for i in range(3):
    t = q2b.claim_task('recovery_w', loaded_suite='libero_10')
    check("Re-claim %d succeeds" % i, t is not None)
    q2b.commit_result(t['cell_id'], t['attempt_id'], 'recovery_w',
                      t['lease_token'], t['lease_epoch'], task_outcome='SUCCESS')

# Also claim remaining 2 pending tasks
for i in range(2):
    t = q2b.claim_task('recovery_w2', loaded_suite='libero_10')
    if t:
        q2b.commit_result(t['cell_id'], t['attempt_id'], 'recovery_w2',
                          t['lease_token'], t['lease_epoch'], task_outcome='SUCCESS')

check("All 5 done", q2b.get_progress()['done'] == 5, "done=%d" % q2b.get_progress()['done'])
check("Superseded tracked", q2b.get_progress()['superseded_attempts'] >= 3)
q2b.close()

# ── Test 3: Stale worker fence ──
print("\nTest 3: Stale worker fencing")
q3, db3 = new_queue('t3')
q3.register_tasks([{'cell_id': 'fence', 'parent_id': 'p0', 'suite': 'libero_10',
                    'task_index': 0, 'state_index': 0, 'arm': 'TRUE_T10', 'task_kind': 'FORMAL_CELL'}])
t_a = q3.claim_task('wA', loaded_suite='libero_10')
check("W A claims", t_a is not None)
q3.close()

raw = sqlite3.connect(db3)
raw.execute("UPDATE tasks SET lease_expires_at='2020-01-01T00:00:00' WHERE cell_id='fence'")
raw.commit(); raw.close()

q3b = AtomicTaskQueue(db3, run_id='t3')
q3b.reap_expired_leases()
t_b = q3b.claim_task('wB', loaded_suite='libero_10')
check("W B re-claims", t_b is not None)
check("W B epoch > W A", t_b['lease_epoch'] > t_a['lease_epoch'],
      "a=%d b=%d" % (t_a['lease_epoch'], t_b['lease_epoch']))

# Stale A tries to commit
r = q3b.commit_result(t_a['cell_id'], t_a['attempt_id'], 'wA',
                      t_a['lease_token'], t_a['lease_epoch'], task_outcome='SUCCESS')
check("Stale A commit REJECTED", not r)

# B commits
r = q3b.commit_result(t_b['cell_id'], t_b['attempt_id'], 'wB',
                      t_b['lease_token'], t_b['lease_epoch'], task_outcome='SUCCESS')
check("W B commit ACCEPTED", r)
q3b.close()

# ── Test 4: Duplicate commit ──
print("\nTest 4: No duplicate accepted attempt")
q4, db4 = new_queue('t4')
q4.register_tasks([{'cell_id': 'dup', 'parent_id': 'p0', 'suite': 'libero_10',
                    'task_index': 0, 'state_index': 0, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL'}])
t = q4.claim_task('w1', loaded_suite='libero_10')
q4.commit_result(t['cell_id'], t['attempt_id'], 'w1', t['lease_token'], t['lease_epoch'],
                 task_outcome='SUCCESS')
t2 = q4.claim_task('w2', loaded_suite='libero_10')
check("No re-claim on done task", t2 is None)
q4.close()

# ── Test 5: DB lock contention ──
print("\nTest 5: DB lock contention")
q5, db5 = new_queue('t5')
q5.register_tasks([{'cell_id': 'lk%d'%i, 'parent_id': 'p%d'%i, 'suite': 'libero_10',
                    'task_index': 0, 'state_index': i, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL'}
                   for i in range(20)])
ok = []
def contend(wid):
    try:
        for _ in range(5):
            t = q5.claim_task('w%d'%wid, loaded_suite='libero_10')
            if t: ok.append(True)
    except: ok.append(False)
threads = [threading.Thread(target=contend, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()
check("No DB corruption", all(ok))
q5.close()

# ── Test 6: Manifest hash mismatch ──
print("\nTest 6: Manifest hash mismatch blocks claim")
q6, db6 = new_queue('t6')
q6.init_run(state='RUNNING', manifest_sha='correct_sha')
q6.register_tasks([{'cell_id': 'hash', 'parent_id': 'p0', 'suite': 'libero_10',
                    'task_index': 0, 'state_index': 0, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL'}])
t = q6.claim_task('w1', loaded_suite='libero_10', expected_manifest_sha='wrong')
check("Claim rejected on wrong SHA", t is None)
t = q6.claim_task('w1', loaded_suite='libero_10', expected_manifest_sha='correct_sha')
check("Claim accepted on correct SHA", t is not None)
q6.close()

# ── Test 7: Scientific failure → DONE_VALID ──
print("\nTest 7: Scientific failure accepted, not retried")
q7, db7 = new_queue('t7')
q7.register_tasks([{'cell_id': 'sfail', 'parent_id': 'p0', 'suite': 'libero_10',
                    'task_index': 0, 'state_index': 0, 'arm': 'TRUE_T10', 'task_kind': 'FORMAL_CELL'}])
t = q7.claim_task('w1', loaded_suite='libero_10')
q7.commit_result(t['cell_id'], t['attempt_id'], 'w1', t['lease_token'], t['lease_epoch'],
                 task_outcome='FAILURE', exposure_status='FULL_K10')
check("Sci failure = DONE_VALID", q7.get_progress()['done'] == 1)
t2 = q7.claim_task('w2', loaded_suite='libero_10')
check("No retry for sci failure", t2 is None)
q7.close()

# ── Test 8: No-emit → DONE_VALID ──
print("\nTest 8: No-emit accepted, not retried")
q8, db8 = new_queue('t8')
q8.register_tasks([{'cell_id': 'noemit', 'parent_id': 'p0', 'suite': 'libero_spatial',
                    'task_index': 0, 'state_index': 0, 'arm': 'TRUE_T10', 'task_kind': 'FORMAL_CELL'}])
t = q8.claim_task('w1', loaded_suite='libero_spatial')
q8.commit_result(t['cell_id'], t['attempt_id'], 'w1', t['lease_token'], t['lease_epoch'],
                 exposure_status='NO_EMIT', task_outcome='SUCCESS')
check("No-emit accepted", q8.get_progress()['done'] == 1)
check("No retry for no-emit", q8.claim_task('w2') is None)
q8.close()

# ── Test 9: HOLD blocks claims ──
print("\nTest 9: HOLD blocks claims")
q9, db9 = new_queue('t9')
q9.register_tasks([{'cell_id': 'h%d'%i, 'parent_id': 'p%d'%i, 'suite': 'libero_10',
                    'task_index': 0, 'state_index': i, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL'}
                   for i in range(3)])
t = q9.claim_task('w1', loaded_suite='libero_10')
check("Pre-HOLD OK", t is not None)
q9.set_run_state('HOLD')
check("Post-HOLD blocked", q9.claim_task('w2', loaded_suite='libero_10') is None)
q9.close()

# ── Test 10: Terminal-censored → DONE_VALID ──
print("\nTest 10: Terminal-censored accepted")
q10, db10 = new_queue('t10')
q10.register_tasks([{'cell_id': 'tc', 'parent_id': 'p0', 'suite': 'libero_goal',
                     'task_index': 0, 'state_index': 0, 'arm': 'RANDOM_TIME_T10', 'task_kind': 'FORMAL_CELL'}])
t = q10.claim_task('w1', loaded_suite='libero_goal')
q10.commit_result(t['cell_id'], t['attempt_id'], 'w1', t['lease_token'], t['lease_epoch'],
                  exposure_status='TERMINAL_CENSORED_K10', task_outcome='FAILURE')
check("TC accepted", q10.get_progress()['done'] == 1)
q10.close()

# ── Test 11: Suite affinity ──
print("\nTest 11: Suite affinity")
q11, db11 = new_queue('t11')
for i in range(10):
    q11.register_tasks([{'cell_id': 'aff_%d'%i, 'parent_id': 'p%d'%i,
        'suite': 'libero_10' if i < 3 else 'libero_goal', 'task_index': 0,
        'state_index': i, 'arm': 'CLEAN', 'task_kind': 'FORMAL_CELL', 'estimated_cost': 10.0-i}])
t = q11.claim_task('w1', loaded_suite='libero_10')
check("Prefers same suite", t and t['suite'] == 'libero_10')
q11.close()

# ── Summary ──
print("\n" + "="*50)
print("RESULTS: %d PASS, %d FAIL" % (PASS, FAIL))
print("="*50)

# Cleanup temp files don't matter on server
sys.exit(0 if FAIL == 0 else 1)
