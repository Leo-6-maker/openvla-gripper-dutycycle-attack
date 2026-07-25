"""Initialize Formal V2 queue. Fail-closed: refuses to overwrite existing DB.
Asserts 20 unique parents, 5 per suite, all LOCKED. Uses Seal V2.
"""
import sys, os, json, hashlib
sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'
FORMAL_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
SEAL_PATH = '/tmp/FEC_FORMAL_SEAL_V2.json'

# Fail if DB already exists
if os.path.isfile(QUEUE_DB):
    print('FATAL: queue DB already exists. Refusing to overwrite.')
    print('  To reset: manually remove %s' % QUEUE_DB)
    sys.exit(1)
attempts_dir = os.path.join(FORMAL_OUT, 'attempts')
if os.path.isdir(attempts_dir) and os.listdir(attempts_dir):
    print('FATAL: attempts directory not empty. Refusing to overwrite.')
    sys.exit(1)

with open(SEAL_PATH) as f:
    seal = json.load(f)
seal_copy = {k: v for k, v in seal.items() if k != 'self_sha256'}
recomputed = hashlib.sha256(json.dumps(seal_copy, sort_keys=True, indent=2).encode()).hexdigest()
assert recomputed == seal['self_sha256'], 'SEAL SELF-HASH FAILED'
manifest_sha = seal['self_sha256']

# Verify cohort SHA
cohort_path = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_phase_b_parent_cohort_20260724T212848Z/FEC_PARENT_MANIFEST_V1.json'
actual_cohort_sha = hashlib.sha256(open(cohort_path, 'rb').read()).hexdigest()
assert actual_cohort_sha == seal['files']['cohort'], 'COHORT SHA MISMATCH'

# Source SHA
worker_sha = seal['files']['worker']; queue_sha = seal['files']['queue']
runner_sha = seal['files']['runner']; config_sha = seal['files']['config']
source_sha = hashlib.sha256((worker_sha + queue_sha + runner_sha + config_sha).encode()).hexdigest()

os.makedirs(FORMAL_OUT, exist_ok=True)
os.makedirs(attempts_dir, exist_ok=True)

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v2')
# Start in INIT, transition to ACTIVE only after full verification
q.init_run(state='INIT', manifest_sha=manifest_sha, source_sha=source_sha, config_sha=config_sha,
           capacity_policy={'GPU_0': 2, 'GPU_2': 2, 'GPU_3': 2, 'GPU_4': 1, 'GPU_5': 1, 'GPU_6': 2, 'GPU_7': 2})

cohort = json.load(open(cohort_path))
parents = cohort['parents']
assert len(parents) == 20, 'Expected 20 parents, got %d' % len(parents)

HORIZONS = {'libero_10': 520, 'libero_goal': 300, 'libero_object': 280, 'libero_spatial': 220}
suite_counts = {}
cells = []
for p in parents:
    key = p['canonical_parent_key']
    suite = p['suite']
    suite_counts[suite] = suite_counts.get(suite, 0) + 1
    parent_id = key.replace('/', '_')
    cells.append({
        'cell_id': 'formal_' + parent_id, 'parent_id': parent_id, 'suite': suite,
        'task_index': int(p['task_idx']),
        'state_index': int(p.get('init_state_index', p.get('state_id', 0))),
        'arm': 'FULL_BUNDLE', 'task_kind': 'FORMAL_PARENT_BUNDLE',
        'estimated_cost': HORIZONS.get(suite, 280) / 10.0 * 5, 'priority': 0,
    })

# Verify per-suite counts
for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
    assert suite_counts.get(suite, 0) == 5, 'Expected 5 %s parents, got %d' % (suite, suite_counts.get(suite, 0))

# Verify unique cell IDs
cell_ids = [c['cell_id'] for c in cells]
assert len(set(cell_ids)) == 20, 'Duplicate cell_ids detected'

q.register_tasks(cells)
q.lock_all_tasks()

# Verify DB state
conn = q._get_conn()
db_total = conn.execute("SELECT COUNT(*) as n FROM tasks").fetchone()['n']
db_locked = conn.execute("SELECT COUNT(*) as n FROM tasks WHERE state='LOCKED'").fetchone()['n']
assert db_total == 20, 'DB total=%d expected 20' % db_total
assert db_locked == 20, 'DB locked=%d expected 20' % db_locked

# All good: transition to ACTIVE
q.set_run_state('ACTIVE')

print('Formal V2 queue initialized (fail-closed):')
print('  DB: %s' % QUEUE_DB)
print('  Seal SHA: %s' % manifest_sha[:16])
print('  Tasks: 20/20 LOCKED (verified in DB)')
print('  Suite coverage: %s' % suite_counts)
print('  Run: unlock_wave0.py to release first parent')
q.close()
