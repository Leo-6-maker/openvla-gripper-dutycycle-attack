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

    vis_ok = {k: v for k, v in vis.items() if v.get('infra_status') == 'ok'}
    rand_ok = {k: v for k, v in rand.items() if v.get('infra_status') == 'ok'}
    paired = set(vis_ok.keys()) & set(rand_ok.keys())
    vis_only = set(vis_ok.keys()) - paired
    rand_only = set(rand_ok.keys()) - paired
    infra_fail = sum(1 for s in summaries if s.get('infra_status') != 'ok')

    vis_opens = [v['decoded_open_count'] for v in vis_ok.values()]
    positives = sum(1 for o in vis_opens if o >= 6)

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
        'vis_ok': len(vis_ok), 'rand_ok': len(rand_ok),
        'valid_paired': len(paired),
        'vis_only': len(vis_only), 'rand_only': len(rand_only),
        'infra_fail': infra_fail,
        'infra_rate': infra_fail / max(len(summaries), 1),
        'provisional_positives': positives,
        'positive_rate': positives / max(len(vis_ok), 1),
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
