#!/usr/bin/env python3
"""
S18 Overnight Watcher — monitors 100-job census across 3 GPU pairs.
Must survive: individual job FAILs, tmux death, SSH drops, partial outputs.
Generates progress log every 5 min; final report when all jobs done or timeout.

Usage: python run_s18_overnight_watcher.py
Resume:  re-run same command — it checks existing summaries and skips completed work.
"""
import json, glob, os, subprocess, time, sys
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
SCRIPTS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb'
CENSUS_DIR = os.path.join(ROOT, 's18_overnight_census')
LOG = os.path.join(ROOT, 's18_overnight_watcher.log')
STATE_FILE = os.path.join(ROOT, 's18_overnight_watcher_state.json')

# Configure
TOTAL_JOBS = 100  # 10 tasks × 5 windows × 2 conditions
GPU_SCRIPTS = {
    'gpu10': 'run_s18_overnight_gpu10.sh',  # 4 tasks, 40 jobs
    'gpu26': 'run_s18_overnight_gpu26.sh',  # 3 tasks, 30 jobs
    'gpu45': 'run_s18_overnight_gpu45.sh',  # 3 tasks, 30 jobs
}
POLL_SEC = 120  # poll interval
STALL_MIN = 30  # if no new summaries for N minutes, flag stall
MAX_HOURS = 10  # max runtime before generating partial report

def log(msg):
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def save_state(state):
    state['updated'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None

def count_summaries():
    """Count summary JSONs by condition, check for FAILs in logs."""
    vis_count = 0; rand_count = 0; fail_count = 0
    vis_files = glob.glob(os.path.join(CENSUS_DIR, 'summary_*vispgd*.json'))
    rand_files = glob.glob(os.path.join(CENSUS_DIR, 'summary_*randomlinf*.json'))
    vis_count = len(vis_files)
    rand_count = len(rand_files)

    # Scan log files for FAIL markers
    log_files = glob.glob(os.path.join(CENSUS_DIR, '*.log'))
    for lf in log_files:
        try:
            with open(lf) as f:
                content = f.read()
            fail_count += content.count('FAIL_')
        except:
            pass

    return vis_count, rand_count, fail_count

def check_gpu_health():
    """Verify GPUs are still running jobs."""
    try:
        r = subprocess.run(
            'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader',
            shell=True, capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().split('\n')
        gpu_status = {}
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpu_status[int(parts[0])] = {
                    'mem_mb': int(parts[1].split()[0]),
                    'util_pct': int(parts[2].split()[0])
                }
        return gpu_status
    except:
        return {}

def check_tmux():
    """Check which tmux sessions are alive."""
    try:
        r = subprocess.run('tmux ls 2>/dev/null', shell=True, capture_output=True, text=True, timeout=5)
        sessions = []
        for line in r.stdout.strip().split('\n'):
            if line:
                sessions.append(line.split(':')[0])
        return sessions
    except:
        return []

def classify_entry(vis_open, vis_streak, rand_open, vis_qpos, rand_qpos):
    """Classify a single parent's command-level result."""
    if rand_open >= 3:
        return 'RANDOM_CONFOUNDED'
    gap = vis_open - rand_open
    if vis_open >= 6 and vis_streak >= 4 and rand_open <= 2 and gap >= 4:
        return 'COMMAND_POSITIVE'
    if vis_open >= 5 and vis_streak >= 3 and rand_open <= 2 and gap >= 3:
        return 'PROMISING_BORDERLINE'
    if vis_open <= 3:
        return 'COMMAND_WEAK'
    return 'BORDERLINE'

def generate_progress_report(vis_ct, rand_ct, fail_ct, gpu_status, tmux_sessions, elapsed):
    """Write a progress summary."""
    pct = (vis_ct + rand_ct) * 100 / TOTAL_JOBS
    log(f'PROGRESS: {vis_ct} VIS + {rand_ct} RAND = {vis_ct+rand_ct}/{TOTAL_JOBS} ({pct:.0f}%) | {fail_ct} FAILs | {elapsed:.0f}min elapsed')
    if gpu_status:
        for gpu_id in [0,1,2,4,5,6]:
            if gpu_id in gpu_status:
                gs = gpu_status[gpu_id]
                log(f'  GPU{gpu_id}: {gs["mem_mb"]}MB {gs["util_pct"]}%')
    log(f'  tmux: {len(tmux_sessions)} sessions')

def generate_candidate_table():
    """Build ranked candidate table from completed summaries."""
    pairs = defaultdict(dict)
    for f in glob.glob(os.path.join(CENSUS_DIR, 'summary_*.json')):
        try:
            with open(f) as fh:
                s = json.load(fh)
            pid = s.get('pair_id', '')
            pairs[pid][s.get('condition', '')] = s
        except:
            pass

    rows = []
    for pid, pair in pairs.items():
        vis = pair.get('vis_pgd', {})
        rand = pair.get('random_linf', {})
        if not vis or not rand:
            continue
        task = vis.get('actual_task_key', vis.get('task', '?'))
        ws = vis.get('window_start', 0)
        we = vis.get('window_end', 0)
        sid = vis.get('state_id', 0)
        vo = vis.get('decoded_open_count', 0)
        vs = vis.get('max_open_streak', 0)
        vp = vis.get('qpos_pos_area', 0)
        ro = rand.get('decoded_open_count', 0)
        rs = rand.get('max_open_streak', 0)
        rp = rand.get('qpos_pos_area', 0)
        cls = classify_entry(vo, vs, ro, vp, rp)
        rows.append((cls, task, sid, ws, we, vo, vs, ro, rs, vp, rp, vo-ro))

    # Sort: COMMAND_POSITIVE first, then PROMISING_BORDERLINE, then others
    priority = {'COMMAND_POSITIVE': 0, 'PROMISING_BORDERLINE': 1, 'BORDERLINE': 2, 'COMMAND_WEAK': 3, 'RANDOM_CONFOUNDED': 4}
    rows.sort(key=lambda r: (priority.get(r[0], 9), -(r[5]-r[7])))

    # Write CSV
    csv_path = os.path.join(ROOT, 's18_candidate_table.csv')
    with open(csv_path, 'w') as f:
        f.write('class,actual_task,state_id,window_start,window_end,VIS_open,VIS_streak,RAND_open,RAND_streak,VIS_qpos,RAND_qpos,VIS_RAND_gap\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')

    return rows, csv_path

# ── MAIN ──
log('=== S18 OVERNIGHT WATCHER STARTED ===')
log(f'Total expected: {TOTAL_JOBS} jobs (10 tasks × 5 windows × 2 conditions)')
log(f'Poll interval: {POLL_SEC}s | Stall threshold: {STALL_MIN}min | Max runtime: {MAX_HOURS}h')

# Check if we should launch or resume
state = load_state()
vis_ct, rand_ct, fail_ct = count_summaries()
existing = vis_ct + rand_ct

if existing == 0:
    log('Fresh start — launching GPU scripts')
    # Kill any old sessions
    for s in ['s18_gpu10', 's18_gpu26', 's18_gpu45']:
        subprocess.run(f'tmux kill-session -t {s} 2>/dev/null', shell=True)
    time.sleep(1)

    # Launch
    os.makedirs(CENSUS_DIR, exist_ok=True)
    for gpu_name, script in GPU_SCRIPTS.items():
        logfile = os.path.join(CENSUS_DIR, f'r_{gpu_name}.log')
        cmd = f"tmux new-session -d -s s18_{gpu_name} 'bash {SCRIPTS}/{script} 2>&1 | tee {logfile}'"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            log(f'  ERR launching {gpu_name}: {r.stderr[:100]}')
        else:
            log(f'  Launched s18_{gpu_name}')

    state = {'phase': 'running', 'launched_at': datetime.now().isoformat(), 'last_new_summary': datetime.now().isoformat()}
    save_state(state)
else:
    log(f'Resume mode: {existing} summaries already exist')
    tmux_sessions = check_tmux()
    log(f'  tmux sessions: {tmux_sessions}')

# Main polling loop
start_time = datetime.now()
stall_start = None
last_count = existing

while True:
    time.sleep(POLL_SEC)
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    vis_ct, rand_ct, fail_ct = count_summaries()
    total = vis_ct + rand_ct
    gpu_status = check_gpu_health()
    tmux_sessions = check_tmux()

    # Progress report every 5 min
    if int(elapsed) % 5 < (POLL_SEC / 60):
        generate_progress_report(vis_ct, rand_ct, fail_ct, gpu_status, tmux_sessions, elapsed)

    # Generate candidate table at 25%, 50%, 75%
    for milestone in [25, 50, 75]:
        if total >= milestone and (total - (vis_ct + rand_ct - (total - milestone))) % 10 == 0:
            rows, csv_path = generate_candidate_table()
            log(f'  Milestone {milestone}%: {len(rows)} candidates → {csv_path}')

    # Check stall
    if total > last_count:
        last_count = total
        stall_start = None
        state['last_new_summary'] = datetime.now().isoformat()
        save_state(state)
    elif stall_start is None:
        stall_start = datetime.now()
    else:
        stall_minutes = (datetime.now() - stall_start).total_seconds() / 60
        if stall_minutes > STALL_MIN:
            log(f'STALL WARNING: no new summaries for {stall_minutes:.0f}min. GPUs may be idle or hung.')
            if not any(s.startswith('s18_') for s in tmux_sessions):
                log('  All tmux sessions dead — jobs may be complete or crashed.')
                break

    # Check completion
    if total >= TOTAL_JOBS:
        log(f'ALL {TOTAL_JOBS} JOBS COMPLETE!')
        break

    # Check max time
    if elapsed > MAX_HOURS * 60:
        log(f'MAX RUNTIME {MAX_HOURS}h reached. Generating partial report.')
        break

# ── FINAL REPORT ──
log('=== GENERATING FINAL REPORT ===')
vis_ct, rand_ct, fail_ct = count_summaries()
total = vis_ct + rand_ct
log(f'Final counts: {vis_ct} VIS + {rand_ct} RAND = {total}/{TOTAL_JOBS} | {fail_ct} FAILs')

rows, csv_path = generate_candidate_table()
log(f'Candidate table: {len(rows)} parents → {csv_path}')

# Summary by class
from collections import Counter
class_counts = Counter(r[0] for r in rows)
log('Classification summary:')
for cls in ['COMMAND_POSITIVE', 'PROMISING_BORDERLINE', 'BORDERLINE', 'COMMAND_WEAK', 'RANDOM_CONFOUNDED']:
    log(f'  {cls}: {class_counts.get(cls, 0)}')

# Top candidates for tomorrow
top = [r for r in rows if r[0] in ('COMMAND_POSITIVE', 'PROMISING_BORDERLINE')]
log(f'Top candidates for confirmation: {len(top)}')
for r in top[:10]:
    cls, task, sid, ws, we, vo, vs, ro, rs, vp, rp, gap = r
    log(f'  [{cls}] {task:20s} s{sid} w{ws}-{we} VIS={vo}/{vs} RAND={ro}/{rs} gap={gap} qpos={vp:.4f}/{rp:.4f}')

log('=== S18 OVERNIGHT WATCHER DONE ===')
log(f'Full log: {LOG}')
log(f'State file: {STATE_FILE}')
log(f'Candidate table: {csv_path}')
