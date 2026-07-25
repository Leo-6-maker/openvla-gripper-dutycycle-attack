"""Launch 16 FEC smoke workers across GPUs 2,3,6,7."""
import subprocess, sys, os, time, json, hashlib

GPU_IDS = [2, 3, 6, 7]
WORKERS_PER_GPU = 4
SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
PY = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
SMOKE_SCRIPT = '/tmp/run_gpu_smoke.py'
OUT_BASE = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_gpu_smoke_v1'

# Assign workers: 4 suites × 4 GPUs = one suite per GPU
gpu_suites = {2: 'libero_10', 3: 'libero_goal', 6: 'libero_object', 7: 'libero_spatial'}

procs = []
worker_id = 0
for gpu_id in GPU_IDS:
    suite = gpu_suites[gpu_id]
    for w in range(WORKERS_PER_GPU):
        smoke_state = 111 + w  # SMOKE_ONLY identity
        out_dir = os.path.join(OUT_BASE, 'gpu_{}'.format(gpu_id), 'worker_{:02d}'.format(worker_id))
        os.makedirs(out_dir, exist_ok=True)

        cmd = 'CUDA_VISIBLE_DEVICES={} {} {} {} {} {} {}'.format(
            gpu_id, PY, SMOKE_SCRIPT, gpu_id, smoke_state, suite, out_dir)
        log_path = os.path.join(out_dir, 'worker.log')
        with open(log_path, 'w') as log:
            p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        procs.append((worker_id, gpu_id, suite, smoke_state, p, out_dir))
        print('[worker {:02d}] GPU{} {} state={} PID={}'.format(worker_id, gpu_id, suite, smoke_state, p.pid))
        worker_id += 1
        time.sleep(1)

print('\nLaunched {} workers across GPUs {}'.format(len(procs), GPU_IDS))
print('Output: {}'.format(OUT_BASE))

# Write manifest
manifest = {'gpu_ids': GPU_IDS, 'workers_per_gpu': WORKERS_PER_GPU, 'total_workers': len(procs),
            'output_base': OUT_BASE, 'scientific_role': 'SMOKE_ONLY',
            'formal_matrix_execution': False, 'cs200_access': False,
            'workers': [{'id': w[0], 'gpu': w[1], 'suite': w[2], 'state': w[3], 'pid': w[4].pid, 'out': w[5]}
                        for w in procs]}
with open(os.path.join(OUT_BASE, 'launch_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

# Wait for all to complete
print('\nWaiting for workers...')
for wid, gpu, suite, state, p, out_dir in procs:
    p.wait()
    # Check result
    result_path = os.path.join(out_dir, 'smoke_summary.json')
    status = '?'
    if os.path.isfile(result_path):
        try:
            r = json.load(open(result_path))
            status = 'PASS' if r.get('valid') else 'ISSUES'
        except:
            status = 'PARSE_ERROR'
    print('[worker {:02d}] GPU{} {} state={} exit={} status={}'.format(wid, gpu, suite, state, p.returncode, status))

print('\nAll workers complete.')
