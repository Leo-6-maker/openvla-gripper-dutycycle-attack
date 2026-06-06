#!/usr/bin/env python3
"""Stage-B auto watcher: check gate every 5 min, auto-launch expansion."""
import json, glob, os, subprocess, time, sys
from collections import Counter
from datetime import datetime

OUT_DIR = '/data/liuyu/outputs/overnight_stageb_labels_20260607'
EXPANSION_SCRIPTS = [
    '/tmp/run_stageb_worker_26_expansion.sh',
    '/tmp/run_stageb_worker_45_expansion.sh',
    '/tmp/run_stageb_worker_10_expansion.sh',
]
GPU_MAP = {'worker_26': '2,6', 'worker_45': '4,5', 'worker_10': '1,0'}

def log(msg):
    print('[%s] %s' % (datetime.now().strftime('%H:%M:%S'), msg))

def check_state():
    summaries = []
    for f in glob.glob(OUT_DIR + '/summary_*.json'):
        try:
            with open(f) as fh: summaries.append(json.load(fh))
        except: pass

    vis = {}; rand = {}
    for s in summaries:
        jid = s.get('job_id', -1)
        if s.get('condition') == 'vis_pgd':
            vis[jid] = s
        elif s.get('condition') == 'random_linf':
            rand[jid] = s

    # Pair by window coordinates (task, state, ws, we), not job_id
    from collections import defaultdict
    windows = defaultdict(dict)
    for s in summaries:
        key = (s.get('task_key',''), str(s.get('state_id','')), s.get('window_start',0), s.get('window_end',0))
        windows[key][s.get('condition','')] = s

    paired = 0; vis_only = 0; rand_only = 0; positives = 0
    vis_ok = 0; rand_ok = 0; infra_fail = 0
    for key, conds in windows.items():
        has_vis = 'vis_pgd' in conds; has_rand = 'random_linf' in conds
        if has_vis and has_rand:
            paired += 1
            if conds['vis_pgd'].get('infra_status') == 'ok': vis_ok += 1
            if conds['random_linf'].get('infra_status') == 'ok': rand_ok += 1
            if conds['vis_pgd'].get('infra_status') == 'ok' and conds['vis_pgd']['decoded_open_count'] >= 6:
                positives += 1
        elif has_vis:
            vis_only += 1
            if conds['vis_pgd'].get('infra_status') == 'ok': vis_ok += 1
            if conds['vis_pgd'].get('infra_status') != 'ok': infra_fail += 1
        elif has_rand:
            rand_only += 1
            if conds['random_linf'].get('infra_status') == 'ok': rand_ok += 1
            if conds['random_linf'].get('infra_status') != 'ok': infra_fail += 1

    # Check worker processes via GPU memory (robust across subprocess issues)
    workers_alive = 0
    try:
        nvidia = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'], universal_newlines=True, timeout=5)
        gpu_mem = []
        for line in nvidia.split('\n'):
            val = line.strip().split()[0] if line.strip() else ''
            if val.isdigit(): gpu_mem.append(int(val))
        workers_alive = sum(1 for m in gpu_mem if m > 500)
    except:
        try:
            ps = subprocess.check_output(['pgrep', '-c', 'vis_labeling'], universal_newlines=True, timeout=5)
            workers_alive = int(ps.strip())
        except: pass

    return {
        'total_summaries': len(summaries),
        'vis_ok': vis_ok, 'rand_ok': rand_ok,
        'valid_paired': paired,
        'vis_only': vis_only, 'rand_only': rand_only,
        'infra_fail': infra_fail,
        'infra_rate': infra_fail / max(len(summaries), 1),
        'provisional_positives': positives,
        'positive_rate': positives / max(paired + vis_only, 1),
        'workers_alive': workers_alive,
    }

def gate_pass(state):
    return (state['valid_paired'] >= 20
            and state['infra_rate'] < 0.15
            and state['provisional_positives'] >= 1
            and state['workers_alive'] == 0)  # wait for current workers to finish

def launch_expansion():
    for script in EXPANSION_SCRIPTS:
        if os.path.exists(script):
            wname = os.path.basename(script).replace('run_stageb_', '').replace('_expansion.sh', '')
            gpu = GPU_MAP.get(wname, '0,1')
            log_path = os.path.join(OUT_DIR, '%s_expansion.log' % wname)
            cmd = 'CUDA_VISIBLE_DEVICES=%s nohup bash %s > %s 2>&1 &' % (gpu, script, log_path)
            log('Launching: %s' % cmd)
            os.system(cmd)
            time.sleep(2)
    # Write gate-pass marker
    with open(os.path.join(OUT_DIR, 'EXPANSION_LAUNCHED'), 'w') as f:
        f.write(datetime.now().isoformat())
    log('Expansion launched')

def main():
    launched = os.path.exists(os.path.join(OUT_DIR, 'EXPANSION_LAUNCHED'))
    log('Watcher started. Expansion launched: %s' % launched)

    iteration = 0
    while True:
        iteration += 1
        state = check_state()
        log('Check #%d: paired=%d vis=%d rand=%d vis_only=%d rand_only=%d pos=%d infra=%.1f%% workers=%d' % (
            iteration, state['valid_paired'], state['vis_ok'], state['rand_ok'],
            state['vis_only'], state['rand_only'],
            state['provisional_positives'], state['infra_rate']*100,
            state['workers_alive']))

        if not launched and gate_pass(state):
            log('GATE PASSED — launching expansion')
            launch_expansion()
            launched = True

        if launched and state['workers_alive'] == 0:
            log('All workers finished. Watcher exiting.')
            break

        time.sleep(300)  # 5 min

if __name__ == '__main__':
    main()
