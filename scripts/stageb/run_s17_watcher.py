#!/usr/bin/env python3
"""S17 auto-watcher: Track A → classify → Track B/C → audit → claim matrix."""
import json, glob, os, subprocess, time, sys
from datetime import datetime

ROOT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
SCRIPTS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb'
LOG = os.path.join(ROOT, 's17_watcher.log')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=600):
    log(f'  RUN: {cmd[:120]}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        log(f'  ERR({r.returncode}): {r.stderr[:200]}')
    return r

def read_summaries(subdir):
    path = os.path.join(ROOT, subdir)
    if not os.path.isdir(path):
        return []
    results = []
    for f in sorted(glob.glob(os.path.join(path, 'summary_*.json'))):
        try:
            with open(f) as fh:
                s = json.load(fh)
            results.append(s)
        except:
            pass
    return results

def classify_command(vis, rand):
    vo = vis.get('decoded_open_count', 0)
    vs = vis.get('max_open_streak', 0)
    ro = rand.get('decoded_open_count', 0)
    rs = rand.get('max_open_streak', 0)
    gap = vo - ro
    if ro >= 3:
        return 'RANDOM_CONFOUNDED'
    if vo >= 6 and vs >= 4 and ro <= 2 and gap >= 4 and vs > rs:
        return 'COMMAND_ATTACK_POSITIVE'
    if vo <= 3:
        return 'COMMAND_WEAK'
    return 'BORDERLINE'

def classify_physical(vis, rand, ore):
    vo = vis.get('decoded_open_count', 0)
    vp = vis.get('qpos_pos_area', 0)
    rp = rand.get('qpos_pos_area', 0)
    ro = rand.get('decoded_open_count', 0)
    vn = vp / ore if ore > 0 else 0
    if ro >= 3:
        return 'RAND_CONFOUNDED_ABSTAIN'
    if vn >= 0.20 and vp > rp and ro <= 1:
        return 'PHYSICAL_BRIDGE_PASS'
    if vn < 0.10:
        return 'PHYSICAL_TRANSFER_WEAK'
    if 0.10 <= vn < 0.20:
        return 'PHYSICAL_BORDERLINE'
    return 'MIXED'

# ── Track A: tomato_sauce_s0_w240-250 completion ──
log('=== S17 WATCHER STARTED ===')
log(f'Root: {ROOT}')

TRACK_A_DIR = 's17b_trackA_w240_250'
TRACK_A_N = 7  # ORACLE + seed53 VIS+RAND + seed54 VIS+RAND + seed55 VIS+RAND

log(f'Waiting for Track A: {TRACK_A_N} summaries in {TRACK_A_DIR}')
while True:
    summaries = read_summaries(TRACK_A_DIR)
    if len(summaries) >= TRACK_A_N:
        break
    log(f'  Track A: {len(summaries)}/{TRACK_A_N}')
    time.sleep(30)

log(f'Track A complete: {len(summaries)} summaries')

# Parse Track A results
pairs = {}
oracle_pos = 0
for s in summaries:
    cond = s.get('condition', '')
    seed = s.get('attack_seed', 0)
    if cond == 'oracle_open':
        oracle_pos = s.get('qpos_pos_area', 0)
        log(f'  ORACLE pos={oracle_pos:.4f}')
    key = (seed, cond)
    pairs[key] = s

# Group by seed
seeds = sorted(set(s.get('attack_seed', 0) for s in summaries if s.get('condition') in ('vis_pgd', 'random_linf')))
log(f'  Track A seeds: {seeds}')

a_cmd_pos = 0; a_phys_pos = 0; a_total = 0
for seed in seeds:
    vis = pairs.get((seed, 'vis_pgd'), {})
    rand = pairs.get((seed, 'random_linf'), {})
    if not vis or not rand:
        log(f'  seed{seed}: INCOMPLETE')
        continue
    a_total += 1
    cmd = classify_command(vis, rand)
    phys = classify_physical(vis, rand, oracle_pos) if oracle_pos > 0 else 'NO_ORACLE'
    vn = vis.get('qpos_pos_area', 0) / oracle_pos if oracle_pos > 0 else 0
    log(f'  seed{seed}: VIS open={vis.get("decoded_open_count",0)} str={vis.get("max_open_streak",0)} '
        f'RAND open={rand.get("decoded_open_count",0)} | cmd={cmd} phys={phys} vn={vn:.3f}')
    if cmd == 'COMMAND_ATTACK_POSITIVE':
        a_cmd_pos += 1
    if phys == 'PHYSICAL_BRIDGE_PASS':
        a_phys_pos += 1

log(f'Track A verdict: {a_cmd_pos}/{a_total} cmd-pos, {a_phys_pos}/{a_total} phys-pos')

# Pre-relabel seeds 50,51,52: 50=MIXED(7/2), 51=WEAK(4/0), 52=PASS(7/0)
# Combined: seeds 50,51,52,53,54,55
track_a_pass = a_cmd_pos >= 2  # need >=2/3 new seeds command-positive

if track_a_pass:
    log('Track A: PASS → launching Track B')
    # Launch Track B (6 parents × seed61)
    for gpu, script in [('s17c_gpu10', 'run_s17c_trackB_gpu10.sh'),
                         ('s17c_gpu26', 'run_s17c_trackB_gpu26.sh'),
                         ('s17c_gpu45', 'run_s17c_trackB_gpu45.sh')]:
        cmd = f"tmux new-session -d -s {gpu} 'bash {SCRIPTS}/{script} 2>&1 | tee {ROOT}/s17c_trackB_command_screen/r_{gpu}.log'"
        run(cmd)
    log('Track B launched')

    # Wait for Track B
    TRACK_B_DIR = 's17c_trackB_command_screen'
    TRACK_B_N = 12
    log(f'Waiting for Track B: {TRACK_B_N} summaries')
    while True:
        b_summaries = read_summaries(TRACK_B_DIR)
        if len(b_summaries) >= TRACK_B_N:
            break
        remaining_tmux = subprocess.run('tmux ls 2>/dev/null | grep s17c | wc -l', shell=True, capture_output=True, text=True).stdout.strip()
        log(f'  Track B: {len(b_summaries)}/{TRACK_B_N} (tmux: {remaining_tmux})')
        time.sleep(40)

    log(f'Track B complete: {len(b_summaries)} summaries')
    b_pairs = {}
    for s in b_summaries:
        pid = s.get('pair_id', '')
        b_pairs.setdefault(pid, {})[s.get('condition', '')] = s

    b_pos = 0; b_total = 0
    for pid, pair in sorted(b_pairs.items()):
        vis = pair.get('vis_pgd', {})
        rand = pair.get('random_linf', {})
        if not vis or not rand: continue
        b_total += 1
        cmd = classify_command(vis, rand)
        task = vis.get('actual_task_key', vis.get('task', '?'))
        if cmd == 'COMMAND_ATTACK_POSITIVE':
            b_pos += 1
        log(f'  Track B: {task:20s} {vis.get("window_start",0)}-{vis.get("window_end",0)} '
            f'VIS open={vis.get("decoded_open_count",0)} RAND open={rand.get("decoded_open_count",0)} | {cmd}')
    log(f'Track B: {b_pos}/{b_total} cmd-pos')

else:
    log('Track A: FAIL → launching Track C (anchor-neighborhood sweep)')
    # Track C: tomato_sauce_s0 w60-70, w65-75, w75-85, w80-90 seed62 VIS+RAND
    for gpu, ws, we in [('s17c_gpu10', 60, 70), ('s17c_gpu26', 65, 75), ('s17c_gpu45', 75, 85)]:
        script_content = f'''#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=$([ "$gpu" = "s17c_gpu10" ] && echo "1,0" || ([ "$gpu" = "s17c_gpu26" ] && echo "2,6") || echo "4,5")
OUT={ROOT}/s17c_trackC_neighborhood
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S={SCRIPTS}/run_s9b_phase1_runner_attack_port.py
echo "[$(date +%H:%M:%S)] Track C tomato_sauce_s0_w{ws}-{we} seed62"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start {ws} --window_end {we} --condition vis_pgd --open_duration 10 --attack_seed 62 --pgd_steps 20 --eps_raw_pixels 6 --job_id 953500 --pair_id tomato_sauce_s0_w{ws}_{we}_s17c_seed62 --output_dir $OUT || echo "FAIL_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start {ws} --window_end {we} --condition random_linf --open_duration 10 --attack_seed 62 --eps_raw_pixels 6 --job_id 953501 --pair_id tomato_sauce_s0_w{ws}_{we}_s17c_seed62 --output_dir $OUT || echo "FAIL_RAND"
echo "[$(date +%H:%M:%S)] Track C {ws}-{we} DONE"
'''
        script_path = f'{SCRIPTS}/run_s17c_trackC_{ws}_{we}.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        run(f'chmod +x {script_path}')
        run(f"tmux new-session -d -s s17c_{ws} 'bash {script_path} 2>&1 | tee {ROOT}/s17c_trackC_neighborhood/r_{ws}_{we}.log'")

    log('Track C launched (3 windows)')
    # Wait briefly then report
    time.sleep(120)

# ── Final audit ──
log('=== S17 WATCHER: generating claim matrix ===')
all_summaries = []
for d in ['s17a_patched_runner_smoke', 's17b_trackA_w240_250', 's17c_trackB_command_screen', 's17c_trackC_neighborhood']:
    all_summaries.extend(read_summaries(d))

# Verify provenance on all
prov_ok = 0; prov_fail = 0
for s in all_summaries:
    atk = s.get('actual_task_key', '')
    if atk and atk != 'unknown':
        prov_ok += 1
    else:
        prov_fail += 1
log(f'Provenance audit: {prov_ok} OK, {prov_fail} MISSING (total {len(all_summaries)})')

# Check for FAILs
fails = [s for s in all_summaries if 'fail' in s.get('infra_status', '').lower()]
if fails:
    log(f'INFRA FAILS: {len(fails)}')
    for s in fails:
        log(f'  {s.get("pair_id","?")}: {s.get("infra_status","?")}')

log('=== S17 WATCHER DONE ===')
log(f'Log: {LOG}')
