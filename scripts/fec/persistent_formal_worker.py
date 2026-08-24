"""Persistent Formal worker. Auto-claims parent bundles from queue, runs 5-arm smoke runner."""
import sys, os, json, time, uuid, socket, subprocess, hashlib, argparse

QUEUE_DB = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v1/queue.sqlite'
FORMAL_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v1'
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

print("[W:%s] Formal worker: GPU=%d slot=%d" % (WORKER_UUID, GPU_ID, SLOT_ID), flush=True)

provider_sha = hashlib.sha256(open(N4_MODULE, 'rb').read()).hexdigest()
assert provider_sha == '6a7ab61d8dba8cb331a748c62317d2513b1e397def2adee8119204be44cecb61', 'PROVIDER SHA MISMATCH'

q = AtomicTaskQueue(QUEUE_DB, run_id='formal_v1')
loaded_suite = None

while True:
    s = q.get_run_state()
    if s in ('HOLD', 'FATAL', 'COMPLETE'):
        print("[W:%s] State=%s, exiting" % (WORKER_UUID, s), flush=True)
        break

    task = q.claim_task(WORKER_UUID, hostname=HOSTNAME, pid=os.getpid(),
                        gpu_id=GPU_ID, slot_id=SLOT_ID, loaded_suite=loaded_suite)
    if task is None:
        p = q.get_progress()
        print("[W:%s] No tasks. done=%d/%d. Exiting." % (WORKER_UUID, p['done'], p['total']), flush=True)
        break

    cell_id = task['cell_id']; suite = task['suite']
    task_idx = task['task_index']; state_idx = task['state_index']
    if suite != loaded_suite:
        print("[W:%s] Suite: %s -> %s" % (WORKER_UUID, loaded_suite or 'none', suite), flush=True)
        loaded_suite = suite

    aid = task['attempt_id']
    q.heartbeat(cell_id, aid, WORKER_UUID, task['lease_token'], task['lease_epoch'])

    out_dir = os.path.join(FORMAL_OUT, 'attempts', cell_id, aid + '.inprogress')
    os.makedirs(out_dir, exist_ok=True)

    seed = abs(hash(cell_id)) % 10000 + 40
    model_path = MODEL_PATHS[suite]

    cmd = [PYTHON, RUNNER, '--gpu-id', str(GPU_ID), '--suite', suite,
           '--task-index', str(task_idx), '--state-index', str(state_idx),
           '--output-root', out_dir, '--model-path', model_path,
           '--config', CONFIG, '--repo-root', '/mnt/sdc/dty_user/openvla_attack',
           '--n4-module', N4_MODULE, '--n4-norm-data', N4_NORM,
           '--expected-attacker-sha256', ATTACKER_SHA,
           '--seed', str(seed), '--rand-direction-seed', str(seed+1000),
           '--random-time-seed', str(seed+2000)]

    print("[W:%s] %s (%s task=%s state=%s)" % (WORKER_UUID, cell_id, suite, task_idx, state_idx), flush=True)
    try:
        r = subprocess.run(cmd, timeout=5400, capture_output=True, text=True,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU_ID)})
        sf = os.path.join(out_dir, 'smoke_summary.json')
        if r.returncode == 0 and os.path.isfile(sf):
            summary = json.load(open(sf))
            receipt_sha = hashlib.sha256(open(sf, 'rb').read()).hexdigest()
            committed = q.commit_result(cell_id, aid, WORKER_UUID,
                                       task['lease_token'], task['lease_epoch'],
                                       exit_code=0, task_outcome='DONE',
                                       output_dir=out_dir, receipt_sha=receipt_sha)
            final_dir = out_dir.replace('.inprogress', '')
            os.rename(out_dir, final_dir)
            print("[W:%s] %s: %s committed=%s" % (WORKER_UUID, cell_id,
                  summary.get('engineering_status', '?'), committed), flush=True)
        else:
            q.commit_result(cell_id, aid, WORKER_UUID, task['lease_token'], task['lease_epoch'],
                           exit_code=1, error_class='RUNNER_FAILURE', output_dir=out_dir)
            print("[W:%s] %s: FAILED exit=%d" % (WORKER_UUID, cell_id, r.returncode), flush=True)
    except subprocess.TimeoutExpired:
        q.commit_result(cell_id, aid, WORKER_UUID, task['lease_token'], task['lease_epoch'],
                       exit_code=124, error_class='TIMEOUT', output_dir=out_dir)
        print("[W:%s] %s: TIMEOUT" % (WORKER_UUID, cell_id), flush=True)

    p = q.get_progress()
    print("[W:%s] Progress: %d/%d done" % (WORKER_UUID, p['done'], p['total']), flush=True)

q.close()
print("[W:%s] Done." % WORKER_UUID, flush=True)
