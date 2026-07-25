"""Persistent Smoke V3 GPU worker. Auto-claims tasks from atomic queue,
runs 5-arm smoke runner, validates, commits, claims next. Stops on HOLD or no tasks."""
import sys, os, json, time, uuid, socket, signal, subprocess, hashlib, argparse, threading
from pathlib import Path
sys.path.insert(0, '/tmp')
from atomic_task_queue import AtomicTaskQueue

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_smoke_v3/queue.sqlite'
SMOKE_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_smoke_v3'
RUNNER = '/mnt/sdc/dty_user/openvla_attack/scripts/fec/run_gpu_smoke.py'
PYTHON = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
N4_MODULE = '/tmp/n4_detector_adapter.py'
N4_NORM = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt'
CONFIG = '/mnt/sdc/dty_user/openvla_attack/configs/fec_attack_v3.yaml'
ATTACKER_SHA = '26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be'
MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
HEARTBEAT_SEC = 25
RUNNER_TIMEOUT_SEC = 5400  # 90 min max per identity

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-id', type=int, required=True)
parser.add_argument('--slot-id', type=int, default=0)
parser.add_argument('--max-active', type=int, default=2)
parser.add_argument('--worker-uuid', default=None)
args = parser.parse_args()

GPU_ID = args.gpu_id
SLOT_ID = args.slot_id
MAX_ACTIVE = args.max_active
WORKER_UUID = args.worker_uuid or uuid.uuid4().hex[:12]
HOSTNAME = socket.gethostname()

print("[W:%s] Persistent Smoke V3 worker starting: GPU=%d slot=%d max_active=%d" % (
    WORKER_UUID, GPU_ID, SLOT_ID, MAX_ACTIVE))

# Verify provider SHA
provider_sha = hashlib.sha256(open(N4_MODULE, 'rb').read()).hexdigest()
EXPECTED_PROVIDER = '6a7ab61d8dba8cb331a748c62317d2513b1e397def2adee8119204be44cecb61'
assert provider_sha == EXPECTED_PROVIDER, 'PROVIDER SHA MISMATCH'

q = AtomicTaskQueue(QUEUE_DB, run_id='smoke_v3')
stop_flag = threading.Event()

def heartbeat_loop():
    while not stop_flag.is_set():
        try:
            q.heartbeat(current_cell_id, current_attempt_id, WORKER_UUID,
                        current_lease_token, current_lease_epoch) if current_cell_id else None
        except: pass
        time.sleep(HEARTBEAT_SEC)

current_cell_id = None; current_attempt_id = None
current_lease_token = None; current_lease_epoch = None

# Heartbeat thread
hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
hb_thread.start()

def run_identity(suite, task_index, state_index, seed, rand_seed, rt_seed, identity_id, output_dir):
    """Run 5-arm smoke runner as subprocess. Returns (success, result_dict)."""
    model_path = MODEL_PATHS[suite]
    cmd = [
        PYTHON, RUNNER,
        '--gpu-id', str(GPU_ID), '--suite', suite,
        '--task-index', str(task_index), '--state-index', str(state_index),
        '--output-root', output_dir, '--model-path', model_path,
        '--config', CONFIG, '--repo-root', '/mnt/sdc/dty_user/openvla_attack',
        '--n4-module', N4_MODULE, '--n4-norm-data', N4_NORM,
        '--expected-attacker-sha256', ATTACKER_SHA,
        '--seed', str(seed), '--rand-direction-seed', str(rand_seed),
        '--random-time-seed', str(rt_seed),
    ]
    print("[W:%s] Running %s: %s/%s state=%s" % (WORKER_UUID, identity_id, suite, task_index, state_index))
    try:
        result = subprocess.run(cmd, timeout=RUNNER_TIMEOUT_SEC, capture_output=True, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU_ID)})
        summary_path = os.path.join(output_dir, 'smoke_summary.json')
        if result.returncode == 0 and os.path.isfile(summary_path):
            d = json.load(open(summary_path))
            print("[W:%s] %s: %s" % (WORKER_UUID, identity_id, d.get('engineering_status', '?')))
            return True, d
        else:
            print("[W:%s] %s: FAILED (exit=%d)" % (WORKER_UUID, identity_id, result.returncode))
            if result.stderr:
                print("[W:%s] STDERR: %s" % (WORKER_UUID, result.stderr[:500]))
            return False, {'engineering_status': 'FAIL', 'error': result.stderr[:1000]}
    except subprocess.TimeoutExpired:
        print("[W:%s] %s: TIMEOUT after %ds" % (WORKER_UUID, identity_id, RUNNER_TIMEOUT_SEC))
        return False, {'engineering_status': 'TIMEOUT'}

def classify_disposition(summary):
    """Classify smoke result into disposition category."""
    status = summary.get('engineering_status', '?')
    valid = summary.get('valid', False)
    results = summary.get('results', {})

    if status == 'PASS' and valid:
        return 'DONE_VALID'

    # Check for Class C terminal-censored (RANDOM_TIME partial K10)
    rt = results.get('RANDOM_TIME_T10', {})
    if rt.get('attack_executed_frames', 0) < rt.get('attack_planned_frames', 10):
        if rt.get('termination') == 'SUCCESS' and rt.get('attack_errors', 0) == 0:
            return 'DONE_CLASS_C_TERMINAL_CENSORED'

    # Check ORACLE emit consistency
    true_emit = results.get('TRUE_T10', {}).get('emit_policy_step')
    oracle_emit = results.get('COMMAND_OPEN_ORACLE', {}).get('emit_policy_step')
    if true_emit is not None and oracle_emit is not None and true_emit != oracle_emit:
        return 'HOLD_ORACLE_CONTRACT'

    # Check for hash/schema mismatch
    if status in ('FAIL', 'ISSUES'):
        # Check if it's infrastructure failure vs scientific
        for arm, r in results.items():
            if r.get('attack_errors', 0) > 0:
                return 'FATAL_POST_ACTION'

    return 'DONE_VALID' if valid else 'FAIL_UNCLASSIFIED'

def compute_receipt_sha(output_dir):
    """Compute SHA256 of all files in attempt directory."""
    files = []
    for root, dirs, fns in os.walk(output_dir):
        for fn in sorted(fns):
            fp = os.path.join(root, fn)
            files.append((os.path.relpath(fp, output_dir), hashlib.sha256(open(fp, 'rb').read()).hexdigest()))
    return hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()

# ── Main loop ──
loaded_suite = None
consecutive_failures = 0

while not stop_flag.is_set():
    run_state = q.get_run_state()
    if run_state in ('HOLD', 'FATAL', 'COMPLETE'):
        print("[W:%s] Run state=%s, stopping" % (WORKER_UUID, run_state))
        break

    # Check GPU capacity
    active_same_gpu = 0  # We'd query DB here; simplified for now
    if active_same_gpu >= MAX_ACTIVE:
        time.sleep(10)
        continue

    task = q.claim_task(WORKER_UUID, hostname=HOSTNAME, pid=os.getpid(),
                        gpu_id=GPU_ID, slot_id=SLOT_ID, loaded_suite=loaded_suite,
                        expected_manifest_sha=None)
    if task is None:
        print("[W:%s] No tasks available, stopping" % WORKER_UUID)
        break

    cell_id = task['cell_id']
    suite = task['suite']
    task_idx = task['task_index']
    state_idx = task['state_index']

    if suite != loaded_suite:
        print("[W:%s] Switching suite: %s -> %s" % (WORKER_UUID, loaded_suite or 'none', suite))
        loaded_suite = suite

    # Parse identity info from cell_id
    identity_id = cell_id
    seed = 40 + task_idx * 10 + state_idx
    rand_seed = 10000 + task_idx * 10 + state_idx
    rt_seed = 20000 + task_idx * 10 + state_idx

    # Set up attempt directory
    attempt_dir = os.path.join(SMOKE_OUT, 'gpu_%d' % GPU_ID, 'worker_%s' % identity_id)

    current_cell_id = cell_id
    current_attempt_id = task['attempt_id']
    current_lease_token = task['lease_token']
    current_lease_epoch = task['lease_epoch']

    q.heartbeat(cell_id, task['attempt_id'], WORKER_UUID, task['lease_token'], task['lease_epoch'])

    # Run 5-arm smoke
    success, summary = run_identity(suite, task_idx, state_idx, seed, rand_seed, rt_seed, identity_id, attempt_dir)

    if not success:
        consecutive_failures += 1
        q.commit_result(cell_id, task['attempt_id'], WORKER_UUID, task['lease_token'], task['lease_epoch'],
                        exit_code=1, error_class='RUNNER_FAILURE', output_dir=attempt_dir)
        if consecutive_failures >= 3:
            print("[W:%s] 3 consecutive failures, stopping" % WORKER_UUID)
            q.set_run_state('HOLD')
            break
        continue

    consecutive_failures = 0
    disposition = classify_disposition(summary)
    receipt_sha = compute_receipt_sha(attempt_dir) if os.path.isdir(attempt_dir) else None

    # Extract arm-level stats
    results = summary.get('results', {})
    exposure_status = 'FULL_K10' if any(
        r.get('attack_executed_frames', 0) == 10 for r in results.values()
    ) else 'NO_EMIT'

    # Check for terminal-censored
    rt = results.get('RANDOM_TIME_T10', {})
    if rt.get('attack_executed_frames', 0) < 10 and rt.get('termination') == 'SUCCESS':
        exposure_status = 'TERMINAL_CENSORED_K10'

    committed = q.commit_result(cell_id, task['attempt_id'], WORKER_UUID,
                                task['lease_token'], task['lease_epoch'],
                                exit_code=0, exposure_status=exposure_status,
                                task_outcome='SUCCESS' if summary.get('valid') else 'CLASSIFIED',
                                output_dir=attempt_dir, receipt_sha=receipt_sha,
                                peak_memory_mb=None)

    if committed:
        print("[W:%s] %s: committed as %s" % (WORKER_UUID, identity_id, disposition))
    else:
        print("[W:%s] %s: commit REJECTED (fenced)" % (WORKER_UUID, identity_id))

    current_cell_id = None; current_attempt_id = None
    current_lease_token = None; current_lease_epoch = None

stop_flag.set()
q.close()
print("[W:%s] Worker exiting" % WORKER_UUID)
