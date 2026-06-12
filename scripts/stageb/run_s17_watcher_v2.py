#!/usr/bin/env python3
"""S17 v2 auto-watcher: Track A→Track B/C→Reserve D→Reserve E→Reserve F→audit."""
import json, glob, os, subprocess, time, sys
from datetime import datetime

ROOT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
SCRIPTS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb'
LOG = os.path.join(ROOT, 's17_watcher_v2.log')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=600):
    log(f'  RUN: {cmd[:150]}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        log(f'  ERR({r.returncode}): {r.stderr[:200]}')
    return r

def read_summaries(subdir):
    path = os.path.join(ROOT, subdir)
    if not os.path.isdir(path): return []
    results = []
    for f in sorted(glob.glob(os.path.join(path, 'summary_*.json'))):
        try:
            with open(f) as fh: s = json.load(fh)
            results.append(s)
        except: pass
    return results

def classify_command(vis, rand):
    vo=vis.get('decoded_open_count',0); vs=vis.get('max_open_streak',0)
    ro=rand.get('decoded_open_count',0); rs=rand.get('max_open_streak',0)
    gap=vo-ro
    if ro>=3: return 'RANDOM_CONFOUNDED'
    if vo>=6 and vs>=4 and ro<=2 and gap>=4 and vs>rs: return 'COMMAND_ATTACK_POSITIVE'
    if vo<=3: return 'COMMAND_WEAK'
    return 'BORDERLINE'

def wait_summaries(subdir, n, label):
    log(f'Waiting for {label}: {n} summaries in {subdir}')
    while True:
        s = read_summaries(subdir)
        if len(s) >= n: break
        tmux_n = subprocess.run('tmux ls 2>/dev/null | grep s17 | wc -l', shell=True, capture_output=True, text=True).stdout.strip()
        log(f'  {label}: {len(s)}/{n} (tmux: {tmux_n})')
        time.sleep(35)
    log(f'{label} complete: {len(s)} summaries')
    return read_summaries(subdir)

def launch_tmux_script(name, script_name, log_name=None):
    logfile = log_name or f'r_{name}.log'
    cmd = f"tmux new-session -d -s {name} 'bash {SCRIPTS}/{script_name} 2>&1 | tee {ROOT}/{logfile}'"
    run(cmd)

# ── MAIN ──
log('=== S17 WATCHER V2 STARTED ===')

# Track A: tomato_sauce_s0_w240-250
track_a = wait_summaries('s17b_trackA_w240_250', 7, 'Track A')

# Classify Track A
oracle_pos = 0; pairs = {}
for s in track_a:
    if s.get('condition') == 'oracle_open':
        oracle_pos = s.get('qpos_pos_area', 0)
    pairs[(s.get('attack_seed',0), s.get('condition',''))] = s

seeds = sorted(set(k[0] for k in pairs if k[1] in ('vis_pgd','random_linf')))
a_cmd = 0; a_phys = 0
for seed in seeds:
    vis = pairs.get((seed,'vis_pgd'),{}); rand = pairs.get((seed,'random_linf'),{})
    if not vis or not rand: continue
    cmd = classify_command(vis, rand)
    vn = vis.get('qpos_pos_area',0)/oracle_pos if oracle_pos>0 else 0
    rn = rand.get('qpos_pos_area',0)/oracle_pos if oracle_pos>0 else 0
    phys = 'PHYSICAL_BRIDGE_PASS' if (vn>=0.20 and vis.get('qpos_pos_area',0)>rand.get('qpos_pos_area',0) and rand.get('decoded_open_count',0)<=1) else ('PHYSICAL_TRANSFER_WEAK' if vn<0.10 else 'PHYSICAL_BORDERLINE')
    log(f'  Track A seed{seed}: VIS open={vis.get("decoded_open_count",0)} str={vis.get("max_open_streak",0)} RAND open={rand.get("decoded_open_count",0)} | cmd={cmd} phys={phys} vn={vn:.3f} rn={rn:.3f}')
    if cmd == 'COMMAND_ATTACK_POSITIVE': a_cmd += 1
    if phys == 'PHYSICAL_BRIDGE_PASS': a_phys += 1

log(f'Track A: ORACLE={oracle_pos:.4f}, {a_cmd}/{len(seeds)} cmd-pos, {a_phys}/{len(seeds)} phys-pos')

# ── Track B: always launch (corrected command screen) ──
log('=== Track B: corrected command screen ===')
for gpu, script in [('s17c_gpu10','run_s17c_trackB_gpu10.sh'),
                     ('s17c_gpu26','run_s17c_trackB_gpu26.sh'),
                     ('s17c_gpu45','run_s17c_trackB_gpu45.sh')]:
    launch_tmux_script(gpu, script, f's17c_trackB_command_screen/r_{gpu}.log')

track_b = wait_summaries('s17c_trackB_command_screen', 12, 'Track B')

b_pairs = {}
for s in track_b:
    b_pairs.setdefault(s.get('pair_id',''), {})[s.get('condition','')] = s

b_pos = []; b_all = []
for pid, pair in sorted(b_pairs.items()):
    vis = pair.get('vis_pgd',{}); rand = pair.get('random_linf',{})
    if not vis or not rand: continue
    cmd = classify_command(vis, rand)
    task = vis.get('actual_task_key', vis.get('task','?'))
    ws = vis.get('window_start',0); we = vis.get('window_end',0)
    vo = vis.get('decoded_open_count',0); ro = rand.get('decoded_open_count',0)
    b_all.append((task, ws, we, vo, ro, cmd))
    if cmd == 'COMMAND_ATTACK_POSITIVE': b_pos.append((task, ws, we))
    log(f'  Track B: {task:20s} w{ws}-{we} VIS open={vo} RAND open={ro} | {cmd}')
log(f'Track B: {len(b_pos)}/{len(b_all)} cmd-pos')

# ── Reserve Track D: tomato_sauce anchor-neighborhood (always run) ──
log('=== Reserve Track D: anchor-neighborhood sweep ===')
D_WINDOWS = [(60,70),(65,75),(75,85),(80,90)]
for i, (ws, we) in enumerate(D_WINDOWS):
    gpu_name = f's17d_gpu{["10","26","45","10"][i]}'
    script = f'''#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES={["1,0","2,6","4,5","1,0"][i]}
OUT={ROOT}/s17d_trackD_neighborhood
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S={SCRIPTS}/run_s9b_phase1_runner_attack_port.py
echo "[$(date +%H:%M:%S)] Track D tomato_sauce_s0_w{ws}-{we} seed62"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start {ws} --window_end {we} --condition vis_pgd --open_duration 10 --attack_seed 62 --pgd_steps 20 --eps_raw_pixels 6 --job_id 9536{ws+we:02d} --pair_id tomato_sauce_s0_w{ws}_{we}_s17d_seed62 --output_dir $OUT || echo "FAIL_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start {ws} --window_end {we} --condition random_linf --open_duration 10 --attack_seed 62 --eps_raw_pixels 6 --job_id 9536{ws+we+1:02d} --pair_id tomato_sauce_s0_w{ws}_{we}_s17d_seed62 --output_dir $OUT || echo "FAIL_RAND"
echo "[$(date +%H:%M:%S)] Track D w{ws}-{we} DONE"
'''
    script_path = f'{SCRIPTS}/run_s17d_trackD_{ws}_{we}.sh'
    with open(script_path, 'w') as f: f.write(script)
    run(f'chmod +x {script_path}')
    launch_tmux_script(gpu_name, f'run_s17d_trackD_{ws}_{we}.sh', f's17d_trackD_neighborhood/r_{ws}_{we}.log')
    time.sleep(2)  # stagger launches

track_d = wait_summaries('s17d_trackD_neighborhood', 8, 'Track D')

d_pos = []
for s in track_d:
    if s.get('condition') != 'vis_pgd': continue
    pid = s.get('pair_id','')
    ws = s.get('window_start',0); we = s.get('window_end',0)
    vo = s.get('decoded_open_count',0); vs = s.get('max_open_streak',0)
    # find matched RAND
    rand_seed = s.get('attack_seed',0)
    rand_s = [r for r in track_d if r.get('pair_id','')==pid and r.get('condition')=='random_linf']
    ro = rand_s[0].get('decoded_open_count',0) if rand_s else -1
    cmd = classify_command(s, rand_s[0]) if rand_s else 'NO_RAND'
    if cmd == 'COMMAND_ATTACK_POSITIVE': d_pos.append((ws, we))
    log(f'  Track D: tomato_sauce w{ws}-{we} VIS open={vo} str={vs} RAND open={ro} | {cmd}')
log(f'Track D: {len(d_pos)}/{len(D_WINDOWS)} neighbor cmd-pos')

# ── Reserve Track E: confirm strongest new positive from B/D ──
candidates = b_pos + [('tomato_sauce', ws, we) for ws, we in d_pos]
candidates = list(set(candidates))  # dedup
# Filter to non-anchor (not w70-80)
fresh = [(t,sid,ws,we) for t,ws,we in candidates if not (t=='tomato_sauce' and ws==70 and we==80)]
log(f'Reserve Track E candidates: {len(fresh)} fresh positives')

if fresh:
    # Take top 2
    for idx, (task, ws, we) in enumerate(fresh[:2]):
        gpu_name = f's17e_gpu{["26","45"][idx]}'
        cuda_dev = ["2,6","4,5"][idx]
        script = f'''#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES={cuda_dev}
OUT={ROOT}/s17e_trackE_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S={SCRIPTS}/run_s9b_phase1_runner_attack_port.py
for SEED in 63 64; do
  echo "[$(date +%H:%M:%S)] Track E {task}_w{ws}-{we} seed$SEED VIS"
  $PY -u $S --gpu_pair 0,1 --task {task} --state-id 0 --window_start {ws} --window_end {we} --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps 20 --eps_raw_pixels 6 --job_id 953700 --pair_id {task}_w{ws}_{we}_s17e_seed$SEED --output_dir $OUT || echo "FAIL_VIS"
  $PY -u $S --gpu_pair 0,1 --task {task} --state-id 0 --window_start {ws} --window_end {we} --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels 6 --job_id 953701 --pair_id {task}_w{ws}_{we}_s17e_seed$SEED --output_dir $OUT || echo "FAIL_RAND"
done
echo "[$(date +%H:%M:%S)] Track E {task}_w{ws}-{we} DONE"
'''
        script_path = f'{SCRIPTS}/run_s17e_trackE_{task}_{ws}_{we}.sh'
        with open(script_path, 'w') as f: f.write(script)
        run(f'chmod +x {script_path}')
        launch_tmux_script(gpu_name, f'run_s17e_trackE_{task}_{ws}_{we}.sh', f's17e_trackE_confirmation/r_{task}_{ws}_{we}.log')
        time.sleep(2)

    track_e = wait_summaries('s17e_trackE_confirmation', len(fresh[:2])*4, 'Track E')
    log(f'Track E: {len(track_e)} summaries')
else:
    log('Track E: no fresh candidates, skipping')

# ── Reserve Track F: weak controls robustness ──
log('=== Reserve Track F: weak controls ===')
f_script = f'''#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT={ROOT}/s17f_trackF_controls
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S={SCRIPTS}/run_s9b_phase1_runner_attack_port.py
SEED=62; EPS=6; PGD=20

echo "[$(date +%H:%M:%S)] Track F salad_dressing_s2_w50-60"
export CUDA_VISIBLE_DEVICES=1,0
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953800 --pair_id salad_dressing_s2_w50_60_s17f_seed62 --output_dir $OUT || echo "FAIL_F1_VIS"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953801 --pair_id salad_dressing_s2_w50_60_s17f_seed62 --output_dir $OUT || echo "FAIL_F1_RAND"

echo "[$(date +%H:%M:%S)] Track F bbq_sauce_s0_w55-65"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 0 --window_start 55 --window_end 65 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953802 --pair_id bbq_sauce_s0_w55_65_s17f_seed62 --output_dir $OUT || echo "FAIL_F2_VIS"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 0 --window_start 55 --window_end 65 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953803 --pair_id bbq_sauce_s0_w55_65_s17f_seed62 --output_dir $OUT || echo "FAIL_F2_RAND"

echo "[$(date +%H:%M:%S)] Track F DONE"
'''
with open(f'{SCRIPTS}/run_s17f_trackF_controls.sh', 'w') as f: f.write(f_script)
run(f'chmod +x {SCRIPTS}/run_s17f_trackF_controls.sh')
launch_tmux_script('s17f_ctrl', 'run_s17f_trackF_controls.sh', 's17f_trackF_controls/r_ctrl.log')

track_f = wait_summaries('s17f_trackF_controls', 4, 'Track F')
for s in track_f:
    if s.get('condition') != 'vis_pgd': continue
    task = s.get('actual_task_key', s.get('task','?'))
    vo = s.get('decoded_open_count',0); ro = -1
    rand_s = [r for r in track_f if r.get('pair_id','')==s.get('pair_id','') and r.get('condition')=='random_linf']
    if rand_s: ro = rand_s[0].get('decoded_open_count',0)
    log(f'  Track F: {task:20s} VIS open={vo} RAND open={ro}')

# ── FINAL AUDIT ──
log('=== FINAL AUDIT ===')
all_dirs = ['s17a_patched_runner_smoke','s17b_trackA_w240_250','s17c_trackB_command_screen',
            's17d_trackD_neighborhood','s17e_trackE_confirmation','s17f_trackF_controls']
all_s = []
for d in all_dirs: all_s.extend(read_summaries(d))

prov_ok = sum(1 for s in all_s if s.get('actual_task_key','') and s.get('actual_task_key','') != 'unknown')
fails = [s for s in all_s if 'fail' in s.get('infra_status','').lower()]
log(f'Total summaries: {len(all_s)} | provenance OK: {prov_ok} | infra FAILs: {len(fails)}')
for s in fails:
    log(f'  FAIL: {s.get("pair_id","?")}: {s.get("infra_status","?")}')

log('=== S17 WATCHER V2 COMPLETE ===')
