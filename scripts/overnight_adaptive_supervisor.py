#!/usr/bin/env python3
"""Overnight Adaptive VIS-1R Screening Supervisor.
NEVER auto-trains detector. NEVER treats 1R failure as negative.
"""

import os, sys, time, csv, json, subprocess, glob
from datetime import datetime

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
ROOT = '/data/liuyu/outputs/adaptive_vis_1r_screening_overnight_20260605'
os.makedirs(ROOT, exist_ok=True)
LOG = os.path.join(ROOT, 'events.log')
STATE_CSV = os.path.join(ROOT, 'jobs_state.csv')
PROGRESS_CSV = os.path.join(ROOT, 'progress.csv')

CAND_CSV = os.path.join(REPO, 'tables/object_phase_response_adaptive_candidates.csv')
BLACKLIST = {'3', '7'}
HEALTHY_PAIRS = ['0,1', '4,5', '2,6']

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    print(f'{t} {msg}')
    with open(LOG, 'a') as f: f.write(f'{t} {msg}\n')

def gpu_idle(pair_str):
    gpus = pair_str.split(',')
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().split('\n'):
            if not line.strip(): continue
            idx, mem = line.split(',')
            if idx.strip() in gpus and int(mem.strip().split()[0]) > 100:
                return False
        return True
    except:
        return False

def load_candidates():
    if not os.path.exists(CAND_CSV):
        log(f'FATAL: {CAND_CSV} not found')
        return []
    with open(CAND_CSV) as f:
        return list(csv.DictReader(f))

def load_state():
    if not os.path.exists(STATE_CSV):
        return {}
    state = {}
    with open(STATE_CSV) as f:
        for r in csv.DictReader(f):
            state[r.get('candidate_id','')] = r
    return state

def save_state(state):
    rows = list(state.values())
    if rows:
        with open(STATE_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

def atomic_claim(candidate_id, gpu_pair):
    """Claim a candidate for a GPU pair via lock dir."""
    lockdir = os.path.join(ROOT, 'locks', candidate_id)
    try:
        os.makedirs(lockdir, exist_ok=False)
        with open(os.path.join(lockdir, 'claimed_by'), 'w') as f:
            f.write(f'{gpu_pair}\n{datetime.now()}')
        return True
    except OSError:
        return False

def run_precheck(candidate, gpu_pair):
    """Run clean+random precheck for single candidate."""
    cid = candidate['candidate_id']
    task = candidate['task_key']; sid = candidate['state_id']
    ws = candidate['window_start']; we = candidate['window_end']
    ep_dir = os.path.join(ROOT, cid)
    os.makedirs(ep_dir, exist_ok=True)

    # Run clean
    log(f'PRECHECK clean: {task}_s{sid} [{ws},{we}] GPU={gpu_pair}')
    clean_cmd = [PY, '-u', os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py'),
        '--task', task, '--state-id', str(sid), '--condition', 'clean',
        '--gpu_pair', gpu_pair, '--perturb_start', str(ws), '--perturb_end', str(we),
        '--eps_raw_pixels', '6', '--objective', 'prefix_locked_gripper_open_margin',
        '--seed', '0']
    clean_log = os.path.join(ep_dir, 'clean.log')
    with open(clean_log, 'w') as lf:
        rc = subprocess.run(clean_cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=600).returncode
    if rc != 0:
        return 'precheck_infra_failed', f'clean_rc={rc}'
    time.sleep(5)

    # Run random
    log(f'PRECHECK random: {task}_s{sid} [{ws},{we}] GPU={gpu_pair}')
    rnd_cmd = [PY, '-u', os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py'),
        '--task', task, '--state-id', str(sid), '--condition', 'random_linf',
        '--gpu_pair', gpu_pair, '--perturb_start', str(ws), '--perturb_end', str(we),
        '--eps_raw_pixels', '6', '--objective', 'prefix_locked_gripper_open_margin',
        '--seed', '0']
    rnd_log = os.path.join(ep_dir, 'random.log')
    with open(rnd_log, 'w') as lf:
        rc = subprocess.run(rnd_cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=600).returncode
    if rc != 0:
        return 'precheck_infra_failed', f'random_rc={rc}'

    # Quick denominator check
    # For now: if both ran OK, mark as clean
    return 'precheck_clean', 'ok'

def run_vis1r(candidate, gpu_pair):
    """Run 1R full-window VIS."""
    cid = candidate['candidate_id']
    task = candidate['task_key']; sid = candidate['state_id']
    ws = candidate['window_start']; we = candidate['window_end']
    ep_dir = os.path.join(ROOT, cid)
    os.makedirs(ep_dir, exist_ok=True)

    log(f'VIS1R: {task}_s{sid} [{ws},{we}] GPU={gpu_pair}')
    vis_cmd = [PY, '-u', os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py'),
        '--task', task, '--state-id', str(sid), '--condition', 'vis_pgd',
        '--gpu_pair', gpu_pair, '--perturb_start', str(ws), '--perturb_end', str(we),
        '--eps_raw_pixels', '6', '--pgd_steps', '40', '--pgd_restarts', '1',
        '--objective', 'prefix_locked_gripper_open_margin', '--seed', '0']
    vis_log = os.path.join(ep_dir, 'vis1r.log')
    with open(vis_log, 'w') as lf:
        rc = subprocess.run(vis_cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
    return rc, vis_log

# === MAIN ===
log('=== Overnight Adaptive VIS-1R Supervisor Started ===')
log(f'Candidates: {CAND_CSV}')
log(f'Output: {ROOT}')
log('LABEL POLICY: 1R failure = pending_negative_1r, NEVER gold negative')

candidates = load_candidates()
state = load_state()

# Initialize pending candidates
for c in candidates:
    cid = c.get('candidate_id', '')
    if not cid: continue
    if cid not in state:
        state[cid] = {
            'candidate_id': cid, 'task_key': c.get('task_key',''),
            'state_id': c.get('state_id',''), 'window_start': c.get('window_start',''),
            'window_end': c.get('window_end',''), 'expected_role': c.get('expected_role',''),
            'status': 'PENDING', 'gpu_pair': '', 'started': '', 'finished': '',
            'result': '', 'runtime_sec': '', 'label_1r': '', 'errors': '',
        }

save_state(state)
pending = sum(1 for v in state.values() if v['status'] == 'PENDING')
log(f'State: {len(state)} total, {pending} PENDING')

inline_smoke_count = 0
started_time = datetime.now()
iteration = 0

while True:
    iteration += 1
    time.sleep(30)

    # Find idle GPU pairs
    idle_pairs = [p for p in HEALTHY_PAIRS if gpu_idle(p)]

    if idle_pairs:
        # Pick next PENDING candidate
        pending_cids = [cid for cid, v in state.items() if v['status'] == 'PENDING']
        if pending_cids and idle_pairs:
            cid = pending_cids[0]
            gpu = idle_pairs[0]
            if atomic_claim(cid, gpu):
                log(f'CLAIMED {cid} on GPU {gpu}')
                state[cid]['status'] = 'CLAIMED'
                state[cid]['gpu_pair'] = gpu
                state[cid]['started'] = str(datetime.now())
                save_state(state)

                # Run precheck
                cand = next((c for c in candidates if c.get('candidate_id')==cid), None)
                if cand:
                    t0 = time.time()
                    precheck_status, precheck_reason = run_precheck(cand, gpu)
                    if precheck_status == 'precheck_clean':
                        # Run VIS 1R
                        rc, vis_log = run_vis1r(cand, gpu)
                        rt = time.time() - t0
                        if rc == 0:
                            state[cid]['status'] = 'VIS1R_DONE'
                            state[cid]['result'] = 'completed'
                            state[cid]['label_1r'] = 'provisional_silver_positive_1r_pending_audit'
                            inline_smoke_count += 1
                        else:
                            state[cid]['status'] = 'VIS1R_INFRA_FAILED'
                            state[cid]['result'] = f'vis_rc={rc}'
                            state[cid]['errors'] = f'vis_rc={rc}'
                        state[cid]['runtime_sec'] = str(round(rt, 1))
                    else:
                        state[cid]['status'] = 'PRECHECK_FAILED'
                        state[cid]['result'] = precheck_status
                        state[cid]['errors'] = precheck_reason

                state[cid]['finished'] = str(datetime.now())
                save_state(state)
                log(f'DONE {cid}: {state[cid]["status"]} rt={state[cid].get("runtime_sec","?")}s')

    # Status heartbeat every 30 min
    if iteration % 60 == 0:
        elapsed = (datetime.now() - started_time).total_seconds() / 3600
        done = sum(1 for v in state.values() if v['status'] in ('VIS1R_DONE','PRECHECK_FAILED','VIS1R_INFRA_FAILED'))
        log(f'HEARTBEAT: {elapsed:.1f}h done={done}/{len(state)} pending={pending} smoke={inline_smoke_count}/8')

    # Safety timeout
    if (datetime.now() - started_time).total_seconds() > 43200:
        log('12h limit reached')
        break

    # Stop if no more pending
    pending_now = sum(1 for v in state.values() if v['status'] == 'PENDING')
    if pending_now == 0:
        log('All candidates processed')
        break

log(f'=== Supervisor done: {inline_smoke_count} completed ===')
