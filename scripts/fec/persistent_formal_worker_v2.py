"""Persistent Formal worker V2. All P0 issues fixed.
Fixes:
  P0-1: Parent-bundle tasks (20 tasks, each = 5 arms), not arm cells
  P0-2: Heartbeat during subprocess via Popen polling every 25s
  P0-3: Failed/timeout attempts → FAILED state, NOT DONE_VALID
  P0-4: Artifact fsync+rename BEFORE DB commit
  P0-5: Attempt validation in commit transaction
  P0-6: Deterministic seed from SHA256(cell_id), not hash()
  P0-7: Pinned runtime dependencies with SHA verification
"""
import sys, os, json, time, uuid, socket, subprocess, hashlib, argparse, threading
from pathlib import Path

# ── P0-7: Pinned runtime dependencies ──
PINNED_SHAS = {
    'provider': '6a7ab61d8dba8cb331a748c62317d2513b1e397def2adee8119204be44cecb61',
    'attacker': '26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be',
}

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'
FORMAL_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
RUNNER = '/mnt/sdc/dty_user/openvla_attack/scripts/fec/run_gpu_smoke.py'
PYTHON = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
N4_MODULE = '/tmp/n4_detector_adapter.py'
N4_NORM = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt'
CONFIG = '/mnt/sdc/dty_user/openvla_attack/configs/fec_attack_v3.yaml'
MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
HEARTBEAT_SEC = 25
LEASE_TIMEOUT_SEC = 5400  # 90 min = worst-case parent runtime

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

print("[W:%s] Formal V2 worker: GPU=%d slot=%d" % (WORKER_UUID, GPU_ID, SLOT_ID), flush=True)

# Verify pinned SHAs
for name, expected in PINNED_SHAS.items():
    path = N4_MODULE if name == 'provider' else (
        '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/attack_adapter.py')
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assert actual == expected, 'P0-7 SHA MISMATCH: %s expected=%s actual=%s' % (name, expected[:16], actual[:16])
print("[W:%s] Pinned SHAs verified" % WORKER_UUID, flush=True)

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v2')
loaded_suite = None

def deterministic_seed(cell_id):
    return int.from_bytes(hashlib.sha256(cell_id.encode()).digest()[:8], 'big') % 100000

def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

while True:
    s = q.get_run_state()
    if s in ('HOLD', 'FATAL', 'COMPLETE'):
        print("[W:%s] State=%s, exiting" % (WORKER_UUID, s), flush=True)
        break

    task = q.claim_task(WORKER_UUID, hostname=HOSTNAME, pid=os.getpid(),
                        gpu_id=GPU_ID, slot_id=SLOT_ID, loaded_suite=loaded_suite,
                        expected_manifest_sha=None)
    if task is None:
        p = q.get_progress()
        print("[W:%s] No tasks. done=%d/%d. Exiting." % (WORKER_UUID, p['done'], p['total']), flush=True)
        break

    cell_id = task['cell_id']; suite = task['suite']
    task_idx = task['task_index']; state_idx = task['state_index']
    if suite != loaded_suite:
        print("[W:%s] Suite: %s -> %s" % (WORKER_UUID, loaded_suite or 'none', suite), flush=True)
        loaded_suite = suite

    aid = task['attempt_id']; lease_token = task['lease_token']; lease_epoch = task['lease_epoch']
    q.heartbeat(cell_id, aid, WORKER_UUID, lease_token, lease_epoch)

    # P0-6: Deterministic seed
    seed = deterministic_seed(cell_id)
    model_path = MODEL_PATHS[suite]

    out_dir_inprog = os.path.join(FORMAL_OUT, 'attempts', cell_id, aid + '.inprogress')
    os.makedirs(out_dir_inprog, exist_ok=True)

    cmd = [PYTHON, RUNNER, '--gpu-id', str(GPU_ID), '--suite', suite,
           '--task-index', str(task_idx), '--state-index', str(state_idx),
           '--output-root', out_dir_inprog, '--model-path', model_path,
           '--config', CONFIG, '--repo-root', '/mnt/sdc/dty_user/openvla_attack',
           '--n4-module', N4_MODULE, '--n4-norm-data', N4_NORM,
           '--expected-attacker-sha256', PINNED_SHAS['attacker'],
           '--seed', str(seed), '--rand-direction-seed', str(seed+1000),
           '--random-time-seed', str(seed+2000)]

    print("[W:%s] %s (%s s=%s seed=%s)" % (WORKER_UUID, cell_id, suite, state_idx, seed), flush=True)

    # P0-2: Heartbeat during subprocess via Popen.poll()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU_ID)})
    hb_stop = threading.Event()
    def heartbeat_proc():
        while not hb_stop.is_set():
            q.heartbeat(cell_id, aid, WORKER_UUID, lease_token, lease_epoch)
            time.sleep(HEARTBEAT_SEC)
    hb_thread = threading.Thread(target=heartbeat_proc, daemon=True)
    hb_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=LEASE_TIMEOUT_SEC)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        exit_code = 124; stdout = stderr = ''
    finally:
        hb_stop.set()

    sf = os.path.join(out_dir_inprog, 'smoke_summary.json')
    success = (exit_code == 0 and os.path.isfile(sf))

    # P0-4: fsync + rename BEFORE DB commit
    commit_state = 'DONE_VALID' if success else 'FAILED'
    exit_code_db = 0 if success else (exit_code or 1)

    if success:
        # Validate summary
        try:
            summary = json.load(open(sf))
        except Exception:
            success = False; commit_state = 'FAILED'; exit_code_db = 99
            summary = {}

        # fsync files
        for root, dirs, files in os.walk(out_dir_inprog):
            for fn in files:
                fp = os.path.join(root, fn)
                with open(fp, 'rb') as fh:
                    os.fsync(fh.fileno())
        fsync_dir(out_dir_inprog)

        # Atomic rename
        out_dir_final = out_dir_inprog.replace('.inprogress', '')
        os.rename(out_dir_inprog, out_dir_final)
        fsync_dir(os.path.dirname(out_dir_final))

        receipt_sha = hashlib.sha256(open(sf, 'rb').read()).hexdigest() if os.path.isfile(sf) else None

        # P0-3 + P0-5: Commit with proper state and attempt validation
        committed = q.commit_result(cell_id, aid, WORKER_UUID, lease_token, lease_epoch,
                                    exit_code=exit_code_db, task_outcome=commit_state,
                                    output_dir=out_dir_final, receipt_sha=receipt_sha,
                                    peak_memory_mb=None)
        status = summary.get('engineering_status', '?') if success else 'FAILED'
        print("[W:%s] %s: %s committed=%s" % (WORKER_UUID, cell_id, status, committed), flush=True)
    else:
        # Failed: mark as FAILED, not DONE_VALID
        if os.path.isdir(out_dir_inprog):
            out_dir_final = out_dir_inprog.replace('.inprogress', '.FAILED')
            try:
                os.rename(out_dir_inprog, out_dir_final)
            except:
                pass
        q.commit_result(cell_id, aid, WORKER_UUID, lease_token, lease_epoch,
                        exit_code=exit_code_db, error_class='RUNNER_FAILURE',
                        task_outcome='FAILED', output_dir=out_dir_final if 'out_dir_final' in dir() else out_dir_inprog)
        print("[W:%s] %s: FAILED exit=%d" % (WORKER_UUID, cell_id, exit_code), flush=True)

    p = q.get_progress()
    print("[W:%s] Progress: %d/%d done" % (WORKER_UUID, p['done'], p['total']), flush=True)

q.close()
print("[W:%s] Done." % WORKER_UUID, flush=True)
