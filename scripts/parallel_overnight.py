import os, sys, time, csv, subprocess, threading
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(REPO)  # go up from scripts/
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
ROOT = '/data/liuyu/outputs/adaptive_vis_1r_screening_overnight_20260605'
STATE_CSV = os.path.join(ROOT, 'jobs_state.csv')
LOG = os.path.join(ROOT, 'events.log')
PAIRS = ['4,5', '2,6']
REPO_ROOT = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    line = t + ' ' + msg
    print(line)
    with open(LOG, 'a') as f: f.write(line + '\n')

def claim(cid, gpu):
    lockdir = os.path.join(ROOT, 'locks', cid)
    try:
        os.makedirs(lockdir, exist_ok=False)
        with open(os.path.join(lockdir, 'claimed_by'), 'w') as f:
            f.write(gpu + '\n' + str(datetime.now()))
        return True
    except: return False

def run(cid, gpu):
    cand_csv = os.path.join(REPO_ROOT, 'tables/object_phase_response_adaptive_candidates.csv')
    with open(cand_csv) as f:
        cands = {c['candidate_id']: c for c in csv.DictReader(f)}
    c = cands.get(cid)
    if not c: return 'MISSING_CANDIDATE'
    tk, sid, ws, we = c['task_key'], c['state_id'], c['window_start'], c['window_end']
    ep = os.path.join(ROOT, cid); os.makedirs(ep, exist_ok=True)
    t0 = time.time()
    V3 = os.path.join(REPO_ROOT, 'scripts/vis_rollout_adaptive_v3.py')

    # Clean
    rc = subprocess.run([PY, '-u', V3, '--task', tk, '--state-id', sid, '--condition', 'clean',
        '--gpu_pair', gpu, '--perturb_start', ws, '--perturb_end', we,
        '--eps_raw_pixels', '6', '--objective', 'prefix_locked_gripper_open_margin', '--seed', '0'],
        cwd=REPO_ROOT, stdout=open(os.path.join(ep,'clean.log'),'w'),
        stderr=subprocess.STDOUT, timeout=600).returncode
    if rc != 0: return 'PRECHECK_INFRA'
    time.sleep(3)

    # Random
    rc = subprocess.run([PY, '-u', V3, '--task', tk, '--state-id', sid, '--condition', 'random_linf',
        '--gpu_pair', gpu, '--perturb_start', ws, '--perturb_end', we,
        '--eps_raw_pixels', '6', '--objective', 'prefix_locked_gripper_open_margin', '--seed', '0'],
        cwd=REPO_ROOT, stdout=open(os.path.join(ep,'random.log'),'w'),
        stderr=subprocess.STDOUT, timeout=600).returncode
    if rc != 0: return 'PRECHECK_INFRA'
    time.sleep(3)

    # VIS 1R
    rc = subprocess.run([PY, '-u', V3, '--task', tk, '--state-id', sid, '--condition', 'vis_pgd',
        '--gpu_pair', gpu, '--perturb_start', ws, '--perturb_end', we,
        '--eps_raw_pixels', '6', '--pgd_steps', '40', '--pgd_restarts', '1',
        '--objective', 'prefix_locked_gripper_open_margin', '--seed', '0'],
        cwd=REPO_ROOT, stdout=open(os.path.join(ep,'vis1r.log'),'w'),
        stderr=subprocess.STDOUT, timeout=3600).returncode
    rt = time.time() - t0
    return 'VIS1R_DONE' if rc == 0 else 'VIS1R_INFRA_FAILED'

def update(cid, status, gpu, rt):
    rows = []
    with open(STATE_CSV) as f: rows = list(csv.DictReader(f))
    for r in rows:
        if r['candidate_id'] == cid:
            r['status'] = status; r['gpu_pair'] = gpu
            r['runtime_sec'] = str(int(rt)); r['finished'] = str(datetime.now())
            if status == 'VIS1R_DONE':
                r['label_1r'] = 'provisional_silver_positive_1r_pending_audit'
    with open(STATE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    # Milestone snapshot every 20 VIS1R_DONE
    vis_count = sum(1 for r in rows if r.get('status') == 'VIS1R_DONE')
    if vis_count > 0 and vis_count % 20 == 0:
        subprocess.run(["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "/tmp/refresh_snapshot.py"], timeout=30)


def worker(gpu):
    while True:
        with open(STATE_CSV) as f: rows = list(csv.DictReader(f))
        pending = [r for r in rows if r['status'] == 'PENDING']
        if not pending: break
        cid = pending[0]['candidate_id']
        # Try each pending candidate until one claims successfully
        claimed = False
        for p in pending:
            if claim(p['candidate_id'], gpu):
                cid = p['candidate_id']
                claimed = True
                break
            time.sleep(0.5)
        if not claimed:
            time.sleep(10)
            continue
        log('CLAIMED %s GPU=%s' % (cid, gpu))
        status = run(cid, gpu)
        rt = 0  # runtime tracked inside run
        update(cid, status, gpu, rt)
        log('DONE %s: %s GPU=%s' % (cid, status, gpu))
        time.sleep(5)

log('=== Parallel Overnight v2 Started ===')
threads = []
for pair in PAIRS:
    t = threading.Thread(target=worker, args=(pair,), daemon=True)
    t.start(); threads.append(t)
    log('Worker started: GPU=%s' % pair)
for t in threads: t.join()
log('=== All done ===')
