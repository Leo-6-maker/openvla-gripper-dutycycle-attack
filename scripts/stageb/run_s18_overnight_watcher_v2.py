#!/usr/bin/env python3
"""
S18 Overnight Watcher v2 — manifest-driven, pair-based, timeout-protected.
Design: manifest → 3 tmux GPU workers → lock/done/fail/timeout files → pair aggregation → candidate table.
Resume: re-run watcher; it detects completed jobs via done/fail/timeout markers.
"""
import json, glob, os, subprocess, sys, time, csv
from datetime import datetime
from collections import defaultdict

ROOT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
CENSUS_DIR = os.path.join(ROOT, 's18_overnight_census')
MANIFEST_PATH = os.path.join(ROOT, '..', 's18_jobs_manifest.csv')
SCRIPTS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb'
GPU_SCRIPT = os.path.join(SCRIPTS, 'run_s18_overnight_gpu.sh')
LOG = os.path.join(ROOT, 's18_overnight_watcher.log')

GPU_CONFIG = {'gpu10': '1,0', 'gpu26': '2,6', 'gpu45': '4,5'}

def log(msg):
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def read_manifest():
    rows = []
    with open(MANIFEST_PATH, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def check_files(pattern):
    return set(os.path.basename(f) for f in glob.glob(pattern))

def check_tmux():
    try:
        r = subprocess.run('tmux ls 2>/dev/null', shell=True, capture_output=True, text=True, timeout=5)
        return [l.split(':')[0] for l in r.stdout.strip().split('\n') if l]
    except: return []

def launch_gpu(gpu_group):
    """Launch or restart a GPU worker tmux session."""
    tmux_name = f's18_{gpu_group}'
    cuda_dev = GPU_CONFIG[gpu_group]
    log_path = os.path.join(CENSUS_DIR, f'r_{gpu_group}.log')
    cmd = f"tmux new-session -d -s {tmux_name} 'bash {GPU_SCRIPT} {gpu_group} {cuda_dev} 2>&1 | tee {log_path}'"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        # Session may already exist — try kill and recreate
        subprocess.run(f'tmux kill-session -t {tmux_name} 2>/dev/null', shell=True)
        time.sleep(1)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    return r.returncode == 0

def read_summary(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def check_provenance(summary, expected_task):
    """Hard provenance gate."""
    atk = summary.get('actual_task_key', '')
    lang = summary.get('actual_language', '')
    bddl = summary.get('actual_bddl_file', '')
    if atk != expected_task: return False, f'actual_task_key mismatch: {atk} != {expected_task}'
    if expected_task.replace('_',' ') not in lang.lower(): return False, f'language mismatch: {lang}'
    if expected_task not in bddl.lower().replace('_',' ').replace(' ','_'): return False, f'bddl mismatch: {bddl}'
    return True, 'ok'

def classify_pair(vis, rand):
    vo = vis.get('decoded_open_count', 0); vs = vis.get('max_open_streak', 0)
    ro = rand.get('decoded_open_count', 0); rs = rand.get('max_open_streak', 0)
    vp = vis.get('qpos_pos_area', 0); rp = rand.get('qpos_pos_area', 0)
    gap = vo - ro
    if ro >= 3: return 'RANDOM_CONFOUNDED'
    if rp >= vp * 0.5 and rp > 0.01: return 'RANDOM_CONFOUNDED'
    if vo >= 6 and vs >= 4 and ro <= 2 and gap >= 4: return 'COMMAND_POSITIVE'
    if vo >= 5 and vs >= 3 and ro <= 2 and gap >= 3: return 'PROMISING_BORDERLINE'
    if vo <= 3: return 'COMMAND_WEAK'
    return 'BORDERLINE'

# ── MAIN ──
log('=== S18 OVERNIGHT WATCHER V2 STARTED ===')

# Load manifest
manifest = read_manifest()
log(f'Manifest: {len(manifest)} jobs from {MANIFEST_PATH}')

# Expected per GPU
gpu_jobs = defaultdict(list)
for r in manifest: gpu_jobs[r['gpu_group']].append(r)
for g, jobs in gpu_jobs.items(): log(f'  {g}: {len(jobs)} jobs')

expected_pairs = set(r['pair_uid'] for r in manifest)
log(f'Expected pairs: {len(expected_pairs)}')

# Launch all GPU workers
for gpu in GPU_CONFIG:
    if launch_gpu(gpu): log(f'  Launched s18_{gpu}')
    else: log(f'  FAILED to launch s18_{gpu}')

# Main loop
POLL_SEC = 90; STALL_MIN = 40; MAX_HOURS = 10
start_time = datetime.now()
last_active = datetime.now()

while True:
    time.sleep(POLL_SEC)
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    tmux_sessions = check_tmux()

    # Count jobs by status
    done_files = check_files(os.path.join(CENSUS_DIR, 'done', '*.done'))
    fail_files = check_files(os.path.join(CENSUS_DIR, 'failed', '*.failed'))
    to_files = check_files(os.path.join(CENSUS_DIR, 'timeout', '*.timeout'))
    lock_files = check_files(os.path.join(CENSUS_DIR, 'locks', '*.lock'))

    completed = len(done_files)
    failed = len(fail_files)
    timed_out = len(to_files)
    locked = len(lock_files)
    remaining = len(manifest) - completed - failed - timed_out

    # Check tmux health, restart if needed
    for gpu in GPU_CONFIG:
        tmux_name = f's18_{gpu}'
        gpu_remaining = sum(1 for r in manifest
                          if r['gpu_group'] == gpu
                          and f"{r['job_uid']}.done" not in done_files
                          and f"{r['job_uid']}.failed" not in fail_files
                          and f"{r['job_uid']}.timeout" not in to_files)
        if gpu_remaining > 0 and tmux_name not in tmux_sessions:
            log(f'RESTART: {tmux_name} dead, {gpu_remaining} jobs remaining → relaunching')
            launch_gpu(gpu)

    # Progress log every 5 min
    if int(elapsed) % 5 < 2:
        pct = completed * 100 / len(manifest)
        log(f'PROGRESS: {completed}/{len(manifest)} ({pct:.0f}%) done | {failed} fail | {timed_out} timeout | {locked} running | {len(tmux_sessions)} tmux | {elapsed:.0f}min')

    # Build pair-based candidate table every 10 min
    if int(elapsed) % 10 < 2:
        # Aggregate summaries by pair_uid
        vis_summaries = {}; rand_summaries = {}
        for f in glob.glob(os.path.join(CENSUS_DIR, 'summary_*vispgd*.json')):
            s = read_summary(f)
            if s:
                puid = None
                for r in manifest:
                    if str(r['job_id']) == str(s.get('job_id','')):
                        puid = r['pair_uid']; break
                if puid: vis_summaries[puid] = (s, f)
        for f in glob.glob(os.path.join(CENSUS_DIR, 'summary_*randomlinf*.json')):
            s = read_summary(f)
            if s:
                puid = None
                for r in manifest:
                    if str(r['job_id']) == str(s.get('job_id','')):
                        puid = r['pair_uid']; break
                if puid: rand_summaries[puid] = (s, f)

        pairs = []
        prov_fails = 0
        for puid in sorted(vis_summaries.keys() & rand_summaries.keys()):
            vis_s, vis_path = vis_summaries[puid]
            rand_s, rand_path = rand_summaries[puid]
            task = vis_s.get('actual_task_key', vis_s.get('task', '?'))
            ws = vis_s.get('window_start', 0); we = vis_s.get('window_end', 0)
            sid = vis_s.get('state_id', 0)

            # Provenance gate
            prov_ok, prov_msg = check_provenance(vis_s, task)
            prov_ok2, _ = check_provenance(rand_s, task)
            if not prov_ok or not prov_ok2:
                prov_fails += 1
                continue

            cls = classify_pair(vis_s, rand_s)
            vo = vis_s.get('decoded_open_count', 0); vs = vis_s.get('max_open_streak', 0)
            ro = rand_s.get('decoded_open_count', 0); rs = rand_s.get('max_open_streak', 0)
            vp = vis_s.get('qpos_pos_area', 0); rp = rand_s.get('qpos_pos_area', 0)
            pairs.append((cls, task, sid, ws, we, vo, vs, ro, rs, vp, rp, vo-ro, puid))

        priority = {'COMMAND_POSITIVE': 0, 'PROMISING_BORDERLINE': 1, 'BORDERLINE': 2, 'COMMAND_WEAK': 3, 'RANDOM_CONFOUNDED': 4}
        pairs.sort(key=lambda x: (priority.get(x[0], 9), -(x[5]-x[7])))

        csv_path = os.path.join(ROOT, 's18_candidate_table.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class','actual_task','state_id','window_start','window_end',
                       'VIS_open','VIS_streak','RAND_open','RAND_streak',
                       'VIS_qpos','RAND_qpos','VIS_RAND_gap','pair_uid'])
            for p in pairs: w.writerow(p)

        class_cnt = Counter(p[0] for p in pairs)
        log(f'  Candidate table: {len(pairs)} pairs ({len(vis_summaries)} VIS + {len(rand_summaries)} RAND matched) | {prov_fails} provenance fails | {class_cnt.get("COMMAND_POSITIVE",0)} cmd-pos')

    # Stall detection
    if locked > 0 or remaining > 0:
        last_active = datetime.now()
    else:
        stall_min = (datetime.now() - last_active).total_seconds() / 60
        if stall_min > STALL_MIN:
            log(f'STALL: no activity for {stall_min:.0f}min, all jobs appear done/failed/timed-out')

    # Check completion
    if remaining == 0 and locked == 0:
        log('ALL JOBS RESOLVED')
        break

    # Timeout
    if elapsed > MAX_HOURS * 60:
        log(f'MAX RUNTIME {MAX_HOURS}h')
        break

# ── FINAL ──
log('=== FINAL REPORT ===')
done_files = check_files(os.path.join(CENSUS_DIR, 'done', '*.done'))
fail_files = check_files(os.path.join(CENSUS_DIR, 'failed', '*.failed'))
to_files = check_files(os.path.join(CENSUS_DIR, 'timeout', '*.timeout'))

log(f'Job-level: {len(done_files)} done / {len(fail_files)} failed / {len(to_files)} timeout / {len(manifest)} total')

# Read final candidate table
csv_path = os.path.join(ROOT, 's18_candidate_table.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        final_pairs = list(csv.DictReader(f))

    class_cnt = Counter(r['class'] for r in final_pairs)
    log(f'Pair-level ({len(final_pairs)} pairs):')
    for cls in ['COMMAND_POSITIVE','PROMISING_BORDERLINE','BORDERLINE','COMMAND_WEAK','RANDOM_CONFOUNDED']:
        log(f'  {cls}: {class_cnt.get(cls,0)}')

    top = [r for r in final_pairs if r['class'] in ('COMMAND_POSITIVE','PROMISING_BORDERLINE')]
    log(f'Top candidates for tomorrow: {len(top)}')
    for r in top:
        log(f'  [{r["class"]}] {r["actual_task"]:20s} s{r["state_id"]} w{r["window_start"]}-{r["window_end"]} VIS={r["VIS_open"]}/{r["VIS_streak"]} RAND={r["RAND_open"]}/{r["RAND_streak"]} gap={r["VIS_RAND_gap"]}')

    # Write confirmation queue
    queue_path = os.path.join(ROOT, 's18_tomorrow_confirmation_queue.csv')
    with open(queue_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank','actual_task','state_id','window_start','window_end',
                    'VIS_open','VIS_streak','RAND_open','VIS_RAND_gap','class','recommended_next_step'])
        for i, r in enumerate(top):
            cls = r['class']
            if cls == 'COMMAND_POSITIVE':
                step = 'RAND_veto_seeds71_72_73'
            else:
                step = 'RAND_veto_first_ORACLE_if_clean'
            w.writerow([i+1, r['actual_task'], r['state_id'], r['window_start'], r['window_end'],
                       r['VIS_open'], r['VIS_streak'], r['RAND_open'], r['VIS_RAND_gap'], cls, step])
    log(f'Confirmation queue: {queue_path}')

# Task summary
task_summary = defaultdict(lambda: {'cmd_pos':0, 'promising':0, 'borderline':0, 'weak':0, 'confounded':0})
for r in final_pairs:
    task_summary[r['actual_task']][
        {'COMMAND_POSITIVE':'cmd_pos','PROMISING_BORDERLINE':'promising',
         'BORDERLINE':'borderline','COMMAND_WEAK':'weak','RANDOM_CONFOUNDED':'confounded'
        }[r['class']]
    ] += 1

for task in sorted(task_summary):
    ts = task_summary[task]
    log(f'  {task:20s}: pos={ts["cmd_pos"]} promising={ts["promising"]} border={ts["borderline"]} weak={ts["weak"]} confounded={ts["confounded"]}')

log('=== S18 OVERNIGHT WATCHER V2 DONE ===')
