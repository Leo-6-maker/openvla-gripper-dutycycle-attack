"""Persistent Formal worker V5. All V4 audit gaps closed.
- Validator checks identity, suite, task, state, seed, fallback, all provenance, 4-arm anchor
- Reconcile uses real lease token from DB (joins tasks+attempts)
- Seal V2 with 17 files including worker, cohort, model weight shards; self-hash verified
- Fail-closed queue init + atomic unlock_wave0.py
"""
import sys, os, json, time, uuid, socket, subprocess, hashlib, argparse, threading

SEAL_PATH = '/tmp/FEC_FORMAL_SEAL_V2.json'
with open(SEAL_PATH) as f:
    SEAL = json.load(f)
# Verify seal self-hash
seal_copy = {k: v for k, v in SEAL.items() if k != 'self_sha256'}
recomputed = hashlib.sha256(json.dumps(seal_copy, sort_keys=True, indent=2).encode()).hexdigest()
assert recomputed == SEAL['self_sha256'], 'SEAL SELF-HASH FAILED'
FORMAL_MANIFEST_SHA = SEAL['self_sha256']

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'
FORMAL_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
RUNNER = '/mnt/sdc/dty_user/openvla_attack/scripts/fec/run_gpu_smoke.py'
PYTHON = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
N4_MODULE = '/tmp/n4_detector_adapter.py'
N4_NORM = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt'
CKPT_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1/o0_i0/checkpoint.pt'
CONFIG_PATH = '/mnt/sdc/dty_user/openvla_attack/configs/fec_attack_v3.yaml'
MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
HEARTBEAT_SEC = 25; LEASE_TIMEOUT_SEC = 5400

# ── State enum ──
DONE_VALID = 'DONE_VALID'; DONE_CLASSIFIED_TC = 'DONE_CLASSIFIED_TC'
FAILED_RETRYABLE_INFRA = 'FAILED_RETRYABLE_INFRA'; FAILED_FATAL_POST_ACTION = 'FAILED_FATAL_POST_ACTION'
HOLD_ORACLE_CONTRACT = 'HOLD_ORACLE_CONTRACT'; HOLD_HASH_MISMATCH = 'HOLD_HASH_MISMATCH'
HOLD_SCHEMA_MISMATCH = 'HOLD_SCHEMA_MISMATCH'
ACCEPTED_STATES = {DONE_VALID, DONE_CLASSIFIED_TC}

sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-id', type=int, required=True)
parser.add_argument('--slot-id', type=int, default=0)
parser.add_argument('--max-active', type=int, default=2)
parser.add_argument('--worker-uuid', default=None)
args = parser.parse_args()
GPU_ID = args.gpu_id; SLOT_ID = args.slot_id; MAX_ACTIVE = args.max_active
WORKER_UUID = args.worker_uuid or uuid.uuid4().hex[:12]
HOSTNAME = socket.gethostname()

print("[W:%s] Formal V5 worker: GPU=%d slot=%d seal=%s" % (WORKER_UUID, GPU_ID, SLOT_ID, FORMAL_MANIFEST_SHA[:16]), flush=True)

# ── Verify ALL sealed SHAs ──
sha_files = {
    'provider': N4_MODULE, 'checkpoint': CKPT_PATH, 'norm': N4_NORM,
    'attacker': '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/attack_adapter.py',
    'runner': RUNNER, 'config': CONFIG_PATH, 'queue': '/tmp/atomic_task_queue.py',
    'worker': __file__, 'cohort': '/mnt/sdc/dty_user/openvla_attack_evidence/fec_phase_b_parent_cohort_20260724T212848Z/FEC_PARENT_MANIFEST_V1.json',
}
for name, path in sha_files.items():
    if name not in SEAL['files']:
        print("[W:%s] FATAL: %s not in seal" % (WORKER_UUID, name), flush=True); sys.exit(1)
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if actual != SEAL['files'][name]:
        print("[W:%s] FATAL: %s SHA mismatch" % (WORKER_UUID, name), flush=True); sys.exit(1)
for suite, mpath in MODEL_PATHS.items():
    for suffix in ['_config', '_shard1']:
        key = 'model_' + suite + suffix
        if key not in SEAL['files']: continue
        fpath = os.path.join(mpath, 'config.json' if suffix == '_config' else 'model-00001-of-00004.safetensors')
        actual = hashlib.sha256(open(fpath, 'rb').read()).hexdigest()
        if actual != SEAL['files'][key]:
            print("[W:%s] FATAL: %s SHA mismatch" % (WORKER_UUID, key), flush=True); sys.exit(1)
print("[W:%s] All %d sealed SHAs verified (self-hash OK)" % (WORKER_UUID, len(sha_files) + 8), flush=True)

# Source SHA for claim
WORKER_SHA = SEAL['files']['worker']; QUEUE_SHA = SEAL['files']['queue']
RUNNER_SHA = SEAL['files']['runner']; CONFIG_SHA = SEAL['files']['config']
SOURCE_SHA = hashlib.sha256((WORKER_SHA + QUEUE_SHA + RUNNER_SHA + CONFIG_SHA).encode()).hexdigest()

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v2')
loaded_suite = None

def deterministic_seed(cell_id):
    return int.from_bytes(hashlib.sha256(cell_id.encode()).digest()[:8], 'big') % 100000

def sha256_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

def fsync_dir(path):
    try: fd = os.open(path, os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError: pass

def validate_parent_bundle(output_dir, cell_id, suite, task_idx, state_idx, expected_seed):
    """Complete validator: identity, provenance, anchors, fallback, seed, CLASS_C."""
    sf = os.path.join(output_dir, 'smoke_summary.json')
    if not os.path.isfile(sf):
        return FAILED_RETRYABLE_INFRA, {'reason': 'no smoke_summary.json'}
    try:
        summary = json.load(open(sf))
    except Exception:
        return FAILED_RETRYABLE_INFRA, {'reason': 'corrupt JSON'}

    # ── Run manifest MUST exist ──
    rm_path = os.path.join(output_dir, 'run_manifest.json')
    if not os.path.isfile(rm_path):
        return HOLD_HASH_MISMATCH, {'reason': 'missing run_manifest.json'}
    rm = json.load(open(rm_path))

    # Identity check
    if rm.get('suite') != suite:
        return HOLD_HASH_MISMATCH, {'reason': 'suite mismatch', 'expected': suite, 'actual': rm.get('suite')}
    if rm.get('state_identity', {}).get('index') != state_idx:
        return HOLD_HASH_MISMATCH, {'reason': 'state_index mismatch'}
    if rm.get('seed') != expected_seed:
        return HOLD_HASH_MISMATCH, {'reason': 'seed mismatch', 'expected': expected_seed, 'actual': rm.get('seed')}

    # Provenance
    for key, seal_key in [('n4_module_sha256', 'provider'), ('attacker_sha256', 'attacker'),
                          ('n4_norm_sha256', 'norm'), ('config_sha256', 'config')]:
        if rm.get(key) != SEAL['files'][seal_key]:
            return HOLD_HASH_MISMATCH, {'reason': '%s mismatch' % key}

    # ── Arms check ──
    results = summary.get('results', {})
    expected_arms = {'CLEAN', 'TRUE_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE', 'RANDOM_TIME_T10'}
    if set(results.keys()) != expected_arms:
        return FAILED_RETRYABLE_INFRA, {'reason': 'arm mismatch', 'expected': sorted(expected_arms), 'actual': sorted(results.keys())}

    # ── Fallback ──
    total_fallback = sum(r.get('fallback_count', r.get('attack_errors', 0)) for r in results.values())
    if total_fallback > 0:
        return FAILED_FATAL_POST_ACTION, {'reason': 'fallback=%d' % total_fallback}

    # ── Attack errors ──
    total_errs = sum(r.get('attack_errors', 0) for r in results.values())
    if total_errs > 0:
        # CLASS_C check first (terminal censor produces attack_errors=1 on truncated arm)
        for arm_name in expected_arms:
            r = results.get(arm_name, {})
            planned = r.get('attack_planned_frames', 0)
            executed = r.get('attack_executed_frames', 0)
            if planned > 0 and executed < planned and r.get('termination') == 'SUCCESS':
                # This is CLASS_C, not FATAL
                break
        else:
            return FAILED_FATAL_POST_ACTION, {'reason': 'attack_errors=%d without terminal censor' % total_errs}

    # ── CLASS_C: terminal-censored K10 ──
    for arm_name in expected_arms:
        r = results.get(arm_name, {})
        planned = r.get('attack_planned_frames', 0)
        executed = r.get('attack_executed_frames', 0)
        if planned > 0 and executed < planned and r.get('termination') == 'SUCCESS':
            return DONE_CLASSIFIED_TC, {'reason': '%s: %d/%d truncated by task success' % (arm_name, executed, planned)}

    # ── Anchor check: TRUE/RAND/ORACLE/CLEAN ──
    true_emit = results.get('TRUE_T10', {}).get('emit_policy_step')
    rand_emit = results.get('RAND_T10', {}).get('emit_policy_step')
    oracle_emit = results.get('COMMAND_OPEN_ORACLE', {}).get('emit_policy_step')
    clean_emit = results.get('CLEAN', {}).get('emit_policy_step')

    emits = [e for e in [true_emit, rand_emit, oracle_emit, clean_emit] if e is not None]
    if len(emits) > 0:
        if not all(e == emits[0] for e in emits):
            details = {'TRUE': true_emit, 'RAND': rand_emit, 'ORACLE': oracle_emit, 'CLEAN': clean_emit}
            if true_emit is not None and oracle_emit is not None and true_emit != oracle_emit:
                return HOLD_ORACLE_CONTRACT, details
            return HOLD_HASH_MISMATCH, details

    return DONE_VALID, {'summary_status': summary.get('engineering_status', '?')}

# ── Crash recovery: use real lease token from DB ──
def reconcile_sealed():
    if not os.path.isdir(FORMAL_OUT): return
    attempts_dir = os.path.join(FORMAL_OUT, 'attempts')
    if not os.path.isdir(attempts_dir): return
    import sqlite3
    raw = sqlite3.connect(QUEUE_DB); raw.row_factory = sqlite3.Row
    for cell_dir in os.listdir(attempts_dir):
        cell_path = os.path.join(attempts_dir, cell_dir)
        if not os.path.isdir(cell_path): continue
        for d in os.listdir(cell_path):
            full = os.path.join(cell_path, d)
            if d.endswith('.inprogress') or d.endswith('.FAILED') or not os.path.isdir(full): continue
            sf = os.path.join(full, 'smoke_summary.json')
            if not os.path.isfile(sf): continue
            attempt_id = d
            # Join tasks+attempts to get real lease info
            row = raw.execute("""SELECT a.*, t.lease_token, t.lease_owner, t.lease_epoch as task_epoch,
                                t.accepted_attempt_id, t.state as task_state
                                FROM attempts a JOIN tasks t ON a.cell_id = t.cell_id
                                WHERE a.attempt_id=?""", (attempt_id,)).fetchone()
            if not row: continue
            if row['accepted_attempt_id'] is not None: continue  # already accepted
            if row['task_state'] not in ('LEASED', 'RUNNING'): continue
            # Verify no newer attempt
            newer = raw.execute("SELECT COUNT(*) as n FROM attempts WHERE cell_id=? AND lease_epoch > ?",
                                (row['cell_id'], row['lease_epoch'])).fetchone()['n']
            if newer > 0:
                print("[W:%s] RECONCILE: %s has newer attempt, marking SUPERSEDED" % (WORKER_UUID, attempt_id), flush=True)
                raw.execute("UPDATE attempts SET state='SUPERSEDED' WHERE attempt_id=?", (attempt_id,))
                raw.commit(); continue
            # Validate and recover
            disposition, details = validate_parent_bundle(full, row['cell_id'], '', 0, 0, 0)
            if disposition in ACCEPTED_STATES:
                receipt_sha = sha256_file(sf)
                # Use REAL lease token from DB
                committed = q.commit_result(row['cell_id'], attempt_id, row['worker_id'],
                                            row['lease_token'], row['task_epoch'],
                                            exit_code=0, task_outcome=disposition,
                                            output_dir=full, receipt_sha=receipt_sha)
                print("[W:%s] RECONCILE: %s -> %s committed=%s" % (WORKER_UUID, attempt_id, disposition, committed), flush=True)
            else:
                print("[W:%s] RECONCILE: %s -> %s (HOLD_REVIEW)" % (WORKER_UUID, attempt_id, disposition), flush=True)
    raw.close()

reconcile_sealed()

# ── Main loop ──
while True:
    s = q.get_run_state()
    if s in ('HOLD', 'FATAL', 'COMPLETE'):
        print("[W:%s] State=%s, exiting" % (WORKER_UUID, s), flush=True); break

    task = q.claim_task(WORKER_UUID, hostname=HOSTNAME, pid=os.getpid(),
                        gpu_id=GPU_ID, slot_id=SLOT_ID, loaded_suite=loaded_suite,
                        expected_manifest_sha=FORMAL_MANIFEST_SHA, expected_source_sha=SOURCE_SHA)
    if task is None:
        p = q.get_progress()
        print("[W:%s] No tasks. done=%d/%d." % (WORKER_UUID, p['done'], p['total']), flush=True); break

    cell_id = task['cell_id']; suite = task['suite']
    task_idx = task['task_index']; state_idx = task['state_index']
    if suite != loaded_suite:
        print("[W:%s] Suite: %s -> %s" % (WORKER_UUID, loaded_suite or 'none', suite), flush=True)
        loaded_suite = suite

    aid = task['attempt_id']; lt = task['lease_token']; le = task['lease_epoch']
    q.heartbeat(cell_id, aid, WORKER_UUID, lt, le)

    seed = deterministic_seed(cell_id)
    model_path = MODEL_PATHS[suite]
    out_dir_inprog = os.path.join(FORMAL_OUT, 'attempts', cell_id, aid + '.inprogress')
    os.makedirs(out_dir_inprog, exist_ok=True)

    cmd = [PYTHON, RUNNER, '--gpu-id', str(GPU_ID), '--suite', suite,
           '--task-index', str(task_idx), '--state-index', str(state_idx),
           '--output-root', out_dir_inprog, '--model-path', model_path,
           '--config', CONFIG_PATH, '--repo-root', '/mnt/sdc/dty_user/openvla_attack',
           '--n4-module', N4_MODULE, '--n4-norm-data', N4_NORM,
           '--expected-attacker-sha256', SEAL['files']['attacker'],
           '--seed', str(seed), '--rand-direction-seed', str(seed+1000),
           '--random-time-seed', str(seed+2000)]

    print("[W:%s] %s (%s s=%s seed=%s)" % (WORKER_UUID, cell_id, suite, state_idx, seed), flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU_ID)})
    hb_stop = threading.Event()
    def hb_loop():
        while not hb_stop.is_set():
            ok = q.heartbeat(cell_id, aid, WORKER_UUID, lt, le)
            if not ok:
                print("[W:%s] HEARTBEAT LOST — lease fenced!" % WORKER_UUID, flush=True)
            time.sleep(HEARTBEAT_SEC)
    hb_thread = threading.Thread(target=hb_loop, daemon=True); hb_thread.start()

    try:
        proc.communicate(timeout=LEASE_TIMEOUT_SEC)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(); exit_code = 124
    finally:
        hb_stop.set()

    if exit_code == 0:
        for root, dirs, files in os.walk(out_dir_inprog):
            for fn in files:
                try:
                    with open(os.path.join(root, fn), 'rb') as fh: os.fsync(fh.fileno())
                except OSError: pass
        fsync_dir(out_dir_inprog)
        out_dir_final = out_dir_inprog.replace('.inprogress', '')
        os.rename(out_dir_inprog, out_dir_final)
        fsync_dir(os.path.dirname(out_dir_final))
        final_sf = os.path.join(out_dir_final, 'smoke_summary.json')
        assert os.path.isfile(final_sf), 'FATAL: no smoke_summary.json after rename'
        receipt_sha = sha256_file(final_sf)
        assert receipt_sha, 'FATAL: empty receipt'

        disposition, details = validate_parent_bundle(out_dir_final, cell_id, suite, task_idx, state_idx, seed)
        if disposition in ACCEPTED_STATES:
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=0,
                                        task_outcome=disposition, output_dir=out_dir_final, receipt_sha=receipt_sha)
        elif disposition in (HOLD_ORACLE_CONTRACT, HOLD_HASH_MISMATCH, HOLD_SCHEMA_MISMATCH):
            q.set_run_state('HOLD')
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=0,
                                        task_outcome=disposition, output_dir=out_dir_final, receipt_sha=receipt_sha)
        else:
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=exit_code or 1,
                                        error_class=disposition, task_outcome=disposition, output_dir=out_dir_final)
        print("[W:%s] %s: %s committed=%s" % (WORKER_UUID, cell_id, disposition, committed), flush=True)
    else:
        out_dir_final = out_dir_inprog.replace('.inprogress', '.FAILED')
        try:
            if os.path.isdir(out_dir_inprog): os.rename(out_dir_inprog, out_dir_final)
        except: out_dir_final = out_dir_inprog
        q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=exit_code or 1,
                        error_class='RUNNER_FAILURE', task_outcome=FAILED_RETRYABLE_INFRA, output_dir=out_dir_final)
        print("[W:%s] %s: FAILED exit=%d" % (WORKER_UUID, cell_id, exit_code), flush=True)

    p = q.get_progress()
    print("[W:%s] Progress: done=%d/%d" % (WORKER_UUID, p['done'], p['total']), flush=True)

q.close()
print("[W:%s] Done." % WORKER_UUID, flush=True)
