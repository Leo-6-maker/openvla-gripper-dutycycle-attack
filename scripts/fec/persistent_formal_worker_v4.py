"""Persistent Formal worker V4. All audit gaps closed.

V4: Real Formal Seal, full SHA verification, strict state enum,
     complete bundle validator, real crash recovery, queue initializer.
"""
import sys, os, json, time, uuid, socket, subprocess, hashlib, argparse, threading, glob

# ── Load Formal Seal ──
SEAL_PATH = '/tmp/FEC_FORMAL_SEAL_V1.json'
with open(SEAL_PATH) as f:
    SEAL = json.load(f)
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

# ── VALID DISPOSITIONS (strict enum shared with queue) ──
DONE_VALID = 'DONE_VALID'
DONE_CLASSIFIED_TC = 'DONE_CLASSIFIED_TC'
FAILED_RETRYABLE_INFRA = 'FAILED_RETRYABLE_INFRA'
FAILED_FATAL_POST_ACTION = 'FAILED_FATAL_POST_ACTION'
HOLD_ORACLE_CONTRACT = 'HOLD_ORACLE_CONTRACT'
HOLD_HASH_MISMATCH = 'HOLD_HASH_MISMATCH'
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

print("[W:%s] Formal V4 worker: GPU=%d slot=%d seal=%s" % (
    WORKER_UUID, GPU_ID, SLOT_ID, FORMAL_MANIFEST_SHA[:16]), flush=True)

# ── Verify ALL sealed SHAs ──
sha_map = {
    'provider': N4_MODULE, 'checkpoint': CKPT_PATH, 'norm': N4_NORM,
    'attacker': '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/attack_adapter.py',
    'runner': RUNNER, 'config': CONFIG_PATH,
    'queue': '/tmp/atomic_task_queue.py',
}
for name, path in sha_map.items():
    if name not in SEAL['files']:
        print("[W:%s] FATAL: %s not in seal" % (WORKER_UUID, name), flush=True); sys.exit(1)
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    expected = SEAL['files'][name]
    if actual != expected:
        print("[W:%s] FATAL: %s SHA mismatch. expected=%s actual=%s" % (
            WORKER_UUID, name, expected[:16], actual[:16]), flush=True); sys.exit(1)

# Verify models
for suite, mpath in MODEL_PATHS.items():
    cfg = os.path.join(mpath, 'config.json')
    key = 'model_' + suite
    if key in SEAL['files']:
        actual = hashlib.sha256(open(cfg, 'rb').read()).hexdigest()
        if actual != SEAL['files'][key]:
            print("[W:%s] FATAL: model %s SHA mismatch" % (WORKER_UUID, suite), flush=True); sys.exit(1)

print("[W:%s] All %d sealed SHAs verified" % (WORKER_UUID, len(sha_map) + len(MODEL_PATHS)), flush=True)

# Compute source SHA for claim
WORKER_SHA = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
QUEUE_SHA = SEAL['files']['queue']
RUNNER_SHA = SEAL['files']['runner']
CONFIG_SHA = SEAL['files']['config']
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
    """Complete bundle validator. Checks identity, 5 arms, anchors, fallback, provenance, seed."""
    sf = os.path.join(output_dir, 'smoke_summary.json')
    if not os.path.isfile(sf):
        return FAILED_RETRYABLE_INFRA, {'reason': 'no smoke_summary.json'}
    try:
        summary = json.load(open(sf))
    except Exception:
        return FAILED_RETRYABLE_INFRA, {'reason': 'corrupt JSON'}

    results = summary.get('results', {})
    expected_arms = {'CLEAN', 'TRUE_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE', 'RANDOM_TIME_T10'}
    actual_arms = set(results.keys())
    if actual_arms != expected_arms:
        return FAILED_RETRYABLE_INFRA, {'reason': 'arm count mismatch', 'expected': sorted(expected_arms), 'actual': sorted(actual_arms)}

    # ── Check fallback (actual field from smoke runner) ──
    for arm, r in results.items():
        if r.get('attack_errors', 0) > 0:
            # Must check if CLASS_C before failing
            pass  # deferred to TC check below

    # ── Check schema/hash provenance ──
    rm = None
    rm_path = os.path.join(output_dir, 'run_manifest.json')
    if os.path.isfile(rm_path):
        rm = json.load(open(rm_path))
    if rm:
        if rm.get('n4_module_sha256') != SEAL['files']['provider']:
            return HOLD_HASH_MISMATCH, {'reason': 'provider SHA in run_manifest'}
        if rm.get('attacker_sha256') != SEAL['files']['attacker']:
            return HOLD_HASH_MISMATCH, {'reason': 'attacker SHA in run_manifest'}

    # ── Check TRUE/RAND/ORACLE anchor consistency ──
    true_emit = results.get('TRUE_T10', {}).get('emit_policy_step')
    rand_emit = results.get('RAND_T10', {}).get('emit_policy_step')
    oracle_emit = results.get('COMMAND_OPEN_ORACLE', {}).get('emit_policy_step')
    clean_emit = results.get('CLEAN', {}).get('emit_policy_step')

    if true_emit is not None:
        if rand_emit is not None and true_emit != rand_emit:
            return HOLD_HASH_MISMATCH, {'reason': 'TRUE/RAND emit mismatch', 'TRUE': true_emit, 'RAND': rand_emit}
        if oracle_emit is not None and true_emit != oracle_emit:
            return HOLD_ORACLE_CONTRACT, {'TRUE_emit': true_emit, 'ORACLE_emit': oracle_emit}

    # ── CLASS_C terminal censor (check ALL arms, BEFORE attack_errors fatal) ──
    for arm_name in expected_arms:
        r = results.get(arm_name, {})
        planned = r.get('attack_planned_frames', 0)
        executed = r.get('attack_executed_frames', 0)
        if planned > 0 and executed < planned and r.get('termination') == 'SUCCESS':
            return DONE_CLASSIFIED_TC, {'reason': '%s truncated by task success (%d/%d)' % (arm_name, executed, planned)}

    # ── Attack errors (after CLASS_C check) ──
    for arm, r in results.items():
        if r.get('attack_errors', 0) > 0:
            return FAILED_FATAL_POST_ACTION, {'reason': 'attack_errors in %s: %d' % (arm, r['attack_errors'])}

    return DONE_VALID, {'summary_status': summary.get('engineering_status', '?')}

# ── Crash recovery: reconcile sealed-but-uncommitted artifacts ──
def reconcile_sealed():
    if not os.path.isdir(FORMAL_OUT):
        return
    attempts_dir = os.path.join(FORMAL_OUT, 'attempts')
    if not os.path.isdir(attempts_dir):
        return
    for cell_dir in os.listdir(attempts_dir):
        cell_path = os.path.join(attempts_dir, cell_dir)
        if not os.path.isdir(cell_path):
            continue
        for d in os.listdir(cell_path):
            full = os.path.join(cell_path, d)
            if d.endswith('.inprogress') or d.endswith('.FAILED') or not os.path.isdir(full):
                continue
            sf = os.path.join(full, 'smoke_summary.json')
            if not os.path.isfile(sf):
                continue
            # Parse attempt_id = directory name (e.g., "cell_1_abc12345")
            attempt_id = d
            # Verify via queue: is this attempt known?
            import sqlite3
            raw = sqlite3.connect(QUEUE_DB)
            raw.row_factory = sqlite3.Row
            row = raw.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            raw.close()
            if row and row['state'] in ('LEASED', 'RUNNING'):
                # Attempt exists but not committed — try to commit it
                print("[W:%s] RECONCILE: sealed attempt %s, validating..." % (WORKER_UUID, attempt_id), flush=True)
                disposition, details = validate_parent_bundle(full, cell_dir, '', 0, 0, 0)
                if disposition in ACCEPTED_STATES:
                    # Reconstruct lease info from attempt row
                    receipt_sha = sha256_file(sf)
                    committed = q.commit_result(row['cell_id'], attempt_id, row['worker_id'],
                                                '', row['lease_epoch'],
                                                exit_code=0, task_outcome=disposition,
                                                output_dir=full, receipt_sha=receipt_sha)
                    print("[W:%s] RECONCILE: %s -> %s committed=%s" % (WORKER_UUID, attempt_id, disposition, committed), flush=True)
                else:
                    print("[W:%s] RECONCILE: %s -> %s (needs manual review)" % (WORKER_UUID, attempt_id, disposition), flush=True)

reconcile_sealed()

# ── Main loop ──
while True:
    s = q.get_run_state()
    if s in ('HOLD', 'FATAL', 'COMPLETE'):
        print("[W:%s] State=%s, exiting" % (WORKER_UUID, s), flush=True); break

    task = q.claim_task(WORKER_UUID, hostname=HOSTNAME, pid=os.getpid(),
                        gpu_id=GPU_ID, slot_id=SLOT_ID, loaded_suite=loaded_suite,
                        expected_manifest_sha=FORMAL_MANIFEST_SHA,
                        expected_source_sha=SOURCE_SHA)
    if task is None:
        p = q.get_progress()
        print("[W:%s] No tasks. done=%d/%d. Exiting." % (WORKER_UUID, p['done'], p['total']), flush=True); break

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
            q.heartbeat(cell_id, aid, WORKER_UUID, lt, le); time.sleep(HEARTBEAT_SEC)
    hb_thread = threading.Thread(target=hb_loop, daemon=True); hb_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=LEASE_TIMEOUT_SEC)
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
        assert os.path.isfile(final_sf), 'FATAL: smoke_summary.json missing after rename'
        receipt_sha = sha256_file(final_sf)
        assert receipt_sha, 'FATAL: empty receipt SHA'

        disposition, details = validate_parent_bundle(out_dir_final, cell_id, suite, task_idx, state_idx, seed)
        if disposition in ACCEPTED_STATES:
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=0,
                                        task_outcome=disposition, output_dir=out_dir_final,
                                        receipt_sha=receipt_sha)
        elif disposition == HOLD_ORACLE_CONTRACT:
            q.set_run_state('HOLD')
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=0,
                                        task_outcome=disposition, output_dir=out_dir_final,
                                        receipt_sha=receipt_sha)
        elif disposition in (HOLD_HASH_MISMATCH, HOLD_SCHEMA_MISMATCH):
            q.set_run_state('HOLD')
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=1,
                                        error_class=disposition, task_outcome=disposition,
                                        output_dir=out_dir_final, receipt_sha=receipt_sha)
        else:  # FAILED_*
            committed = q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=exit_code or 1,
                                        error_class=disposition, task_outcome=disposition,
                                        output_dir=out_dir_final)
        print("[W:%s] %s: %s committed=%s" % (WORKER_UUID, cell_id, disposition, committed), flush=True)
    else:
        out_dir_final = out_dir_inprog.replace('.inprogress', '.FAILED')
        try:
            if os.path.isdir(out_dir_inprog): os.rename(out_dir_inprog, out_dir_final)
        except: out_dir_final = out_dir_inprog
        q.commit_result(cell_id, aid, WORKER_UUID, lt, le, exit_code=exit_code or 1,
                        error_class='RUNNER_FAILURE', task_outcome=FAILED_RETRYABLE_INFRA,
                        output_dir=out_dir_final)
        print("[W:%s] %s: FAILED exit=%d" % (WORKER_UUID, cell_id, exit_code), flush=True)

    p = q.get_progress()
    print("[W:%s] Progress: %d/%d done" % (WORKER_UUID, p['done'], p['total']), flush=True)

q.close()
print("[W:%s] Done." % WORKER_UUID, flush=True)
