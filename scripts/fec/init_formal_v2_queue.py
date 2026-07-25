"""Initialize Formal V2 queue with 20 parent bundle tasks.
Reads Formal Seal for manifest SHA, registers all tasks LOCKED.
Only Wave-0 unlock script can transition one parent to PENDING.
"""
import sys, os, json, hashlib, sqlite3
sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'
FORMAL_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
SEAL_PATH = '/tmp/FEC_FORMAL_SEAL_V1.json'

with open(SEAL_PATH) as f:
    seal = json.load(f)
manifest_sha = seal['self_sha256']

# Compute source SHA
worker_sha = hashlib.sha256(open('/tmp/persistent_formal_worker_v4.py', 'rb').read()).hexdigest()
queue_sha = seal['files']['queue']
runner_sha = seal['files']['runner']
config_sha = seal['files']['config']
source_sha = hashlib.sha256((worker_sha + queue_sha + runner_sha + config_sha).encode()).hexdigest()

# Wipe old DB
if os.path.isfile(QUEUE_DB):
    os.unlink(QUEUE_DB)
    for suffix in ['-wal', '-shm']:
        p = QUEUE_DB + suffix
        if os.path.isfile(p): os.unlink(p)

os.makedirs(FORMAL_OUT, exist_ok=True)
os.makedirs(os.path.join(FORMAL_OUT, 'attempts'), exist_ok=True)

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v2')
q.init_run(state='ACTIVE', manifest_sha=manifest_sha, source_sha=source_sha, config_sha=config_sha,
           capacity_policy={'GPU_0': 2, 'GPU_2': 2, 'GPU_3': 2, 'GPU_4': 1, 'GPU_5': 1, 'GPU_6': 2, 'GPU_7': 2})

# Register 20 parent bundles
cohort = json.load(open('/mnt/sdc/dty_user/openvla_attack_evidence/fec_phase_b_parent_cohort_20260724T212848Z/FEC_PARENT_MANIFEST_V1.json'))
HORIZONS = {'libero_10': 520, 'libero_goal': 300, 'libero_object': 280, 'libero_spatial': 220}

cells = []
for p in cohort['parents']:
    key = p['canonical_parent_key']
    suite = p['suite']
    parent_id = key.replace('/', '_')
    # Read init_state_index from per-suite manifest
    cells.append({
        'cell_id': 'formal_' + parent_id,
        'parent_id': parent_id,
        'suite': suite,
        'task_index': int(p['task_idx']),
        'state_index': int(p.get('init_state_index', p.get('state_id', 0))),
        'arm': 'FULL_BUNDLE',
        'task_kind': 'FORMAL_PARENT_BUNDLE',
        'estimated_cost': HORIZONS.get(suite, 280) / 10.0 * 5,
        'priority': 0,
    })

q.register_tasks(cells)
# Lock all — Wave-0 unlock script must explicitly unlock
q.lock_all_tasks()

print('Formal V2 queue initialized:')
print('  DB: %s' % QUEUE_DB)
print('  Manifest SHA: %s' % manifest_sha[:16])
print('  Source SHA: %s' % source_sha[:16])
print('  Tasks: %d (all LOCKED)' % len(cells))
print('  Wave-0: run unlock_wave0.py to release first parent')
q.close()
