"""Launch 20-worker detector validation on 5 GPUs (0,2,3,6,7), 4 workers each."""
import subprocess, sys, os, json, time, hashlib

GPUS = [0, 2, 3, 6, 7]
WORKERS_PER_GPU = 4
PYTHON = '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python'
WORKER_SCRIPT = '/tmp/fec_worker.py'

# Write the per-worker script
with open(WORKER_SCRIPT, 'w') as f:
    f.write(r'''import sys, os, json, hashlib, time, numpy as np, torch
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
os.environ['MUJOCO_GL'] = 'egl'

suite, task_id, state_id, seed, arm, out_dir = sys.argv[1:7]
task_id = int(task_id); state_id = int(state_id); seed = int(seed)

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
MODEL_PATH = MODEL_PATHS[suite]

# SELF-CHECK: attacker SHA
import gripper_attack.attack_adapter as aa
def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1048576), b''): d.update(chunk)
    return d.hexdigest()
attacker_sha = sha256_file(os.path.realpath(aa.__file__))
EXPECTED = '26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be'
assert attacker_sha == EXPECTED, f'SHA MISMATCH: {attacker_sha[:16]} != {EXPECTED[:16]}'

# Load N4 Detector
sys.path.insert(0, '/tmp')
from n4_detector_adapter import N4DetectorAdapter
adapter = N4DetectorAdapter(device='cuda:0', norm_data_path=EVIDENCE + '/fec_implementation_v1/n4_norms_o0i0.pt')

# Load OpenVLA
from transformers import AutoProcessor, AutoModelForVision2Seq
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.openvla_preprocess import prepare_openvla_image

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
model.eval()

bench_obj = benchmark.get_benchmark_dict()[suite]()
task = bench_obj.get_task(task_id)
bddl_file = bench_obj.get_task_bddl_file_path(task_id)
instruction = str(task)

def prompt(ins): return f'In: What action should the robot take to {ins.lower()}?\nOut:'

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=224, camera_widths=224, render_gpu_device_id=0, horizon=600)
obs = env.reset()

adapter.reset_episode()
max_steps = 600
step_records = []
emit_step = None
task_success = False

for step in range(max_steps):
    # OpenVLA forward
    image = prepare_openvla_image(obs['agentview_image'])
    inputs = processor(prompt(instruction), image, return_tensors='pt')
    inputs.pop('attention_mask', None)
    for key, val in list(inputs.items()):
        if torch.is_floating_point(val): inputs[key] = val.to(device='cuda', dtype=torch.bfloat16)
        else: inputs[key] = val.to(device='cuda')

    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=7, do_sample=False, return_dict_in_generate=True, output_scores=True)

    token_ids = gen.sequences[0, -7:].detach().cpu().numpy()
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, a_min=0, a_max=model.bin_centers.shape[0]-1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(suite)
    mask = stats.get('mask', np.ones_like(stats['q01'], dtype=bool))
    high, low = np.array(stats['q99']), np.array(stats['q01'])
    action = np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)

    # N4 Detector: compute features from action and gripper state
    # Simplified 51D: use action + zeros for missing fields (full integration needs proper feature extraction)
    f25d = np.zeros(25, dtype=np.float32)
    f25d[0] = float(action[6])  # gripper command
    f25d[6:9] = action[0:3]     # EEF deltas
    p9d = np.zeros(9, dtype=np.float32)
    g9d = np.zeros(9, dtype=np.float32)
    cc = action[6] < -0.1  # simple candidate_close heuristic

    n4_result = adapter.step(f25d, p9d, g9d, cc)
    if n4_result['emitted_this_step'] and emit_step is None:
        emit_step = step

    # Step environment
    obs, reward, done, info = env.step(action)
    step_records.append({'step': step, 'raw_logit': n4_result['raw_logit'],
                         'cal_prob': n4_result['calibrated_prob'], 'cc': cc,
                         'emit': n4_result['emitted_this_step'],
                         'gripper_cmd': float(action[6])})

    if done:
        task_success = info.get('success', False)
        break

env.close()

result = {
    'suite': suite, 'state_id': state_id, 'seed': seed, 'arm': arm,
    'task_success': task_success, 'total_steps': step + 1,
    'detector_emitted': emit_step is not None, 'emit_step': emit_step,
    'n_steps_logged': len(step_records),
    'attacker_sha_ok': True
}

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'result.json'), 'w') as f:
    json.dump(result, f)
with open(os.path.join(out_dir, 'step_records.jsonl'), 'w') as f:
    for rec in step_records:
        f.write(json.dumps(rec) + '\n')

print(json.dumps(result))
''')

print(f'Worker script written: {WORKER_SCRIPT}')
print(f'GPUs: {GPUS} × {WORKERS_PER_GPU} = {len(GPUS) * WORKERS_PER_GPU} workers')
