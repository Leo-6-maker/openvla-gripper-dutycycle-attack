#!/usr/bin/env python3
"""S8 ORACLE OPEN physical scan — force gripper OPEN, measure qpos response."""
import csv, os, sys, argparse, json, hashlib, time
from datetime import datetime
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--task', required=True)
ap.add_argument('--state-id', type=int, required=True)
ap.add_argument('--window_start', type=int, required=True)
ap.add_argument('--window_end', type=int, required=True)
ap.add_argument('--condition', choices=['clean', 'oracle_open'], required=True)
ap.add_argument('--open_duration', type=int, default=10, help='L: number of steps to force OPEN')
ap.add_argument('--job_id', type=int, default=0)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--max_steps', type=int, default=400)
args, _ = ap.parse_known_args()

import torch; from PIL import Image

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
visible = [int(x) for x in _VISIBLE.split(',')]
render_physical_gpu = visible[gpu_ids[1]] if len(visible) > gpu_ids[1] else gpu_ids[1]

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

from transformers import AutoModelForVision2Seq, AutoProcessor
print('[%s] Loading model...' % datetime.now().strftime('%H:%M:%S'))
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
UNNORM_KEY = 'libero_object'; action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,'salad_dressing':3,'bbq_sauce':4,'milk':5,'alphabet_soup':6,'tomato_sauce':7,'orange_juice':8}
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

def decode_tokens(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

cfg = TASK_CFG.get(args.task)
if cfg is None: print('Unknown task:', args.task); sys.exit(1)
task_obj = task_suite.get_task(cfg); init_states = task_suite.get_task_init_states(cfg)
if args.state_id >= len(init_states): print('state_id out of range'); sys.exit(1)

instruction = task_obj.language if hasattr(task_obj,'language') else args.task.replace('_',' ')
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                         has_renderer=False, has_offscreen_renderer=True,
                         use_camera_obs=True, camera_names=['agentview'],
                         control_freq=20, render_gpu_device_id=render_physical_gpu)
env.seed(args.state_id); env.reset(); env.set_init_state(init_states[args.state_id])
prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()

ws = args.window_start; we = args.window_end
oracle_we = min(ws + args.open_duration, args.max_steps)
pair_id = '%s_s%d_w%d_%d_L%d_%s' % (args.task, args.state_id, ws, we, args.open_duration, args.condition)

os.makedirs(args.output_dir, exist_ok=True)
out_json = os.path.join(args.output_dir, 'summary_%s_job%d.json' % (args.condition, args.job_id))
out_trace = os.path.join(args.output_dir, 'trace_%s_job%d.csv' % (args.condition, args.job_id))

print('[%s] %s s%d w[%d,%d] L=%d %s' % (datetime.now().strftime('%H:%M:%S'), args.task, args.state_id, ws, we, args.open_duration, args.condition))

done = False; step = 0; max_steps = max(we, oracle_we) + 20
trace_rows = []; decoded_opens = []; qpos_history = []; arm_l2_history = []
oracle_active = False

while not done and step < max_steps:
    img = env.sim.render(256, 256, camera_name='agentview')
    img_pil = Image.fromarray(img.astype(np.uint8)).rotate(180)
    inp = processor(prompt, img_pil)
    inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
           if isinstance(v,torch.Tensor) else v for k,v in inp.items()}
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
    raw_action = decode_tokens(token_ids)
    raw_gripper = float(raw_action[-1])

    # ORACLE: force OPEN during the window
    oracle_active = (args.condition == 'oracle_open' and ws <= step < oracle_we)
    if oracle_active:
        env_action = raw_action.copy()
        env_action[-1] = -1.0  # force OPEN in env space (env_action_6 < -0.5 = OPEN)
    else:
        # Standard binarize + invert via env convention
        env_action = raw_action.copy()
        norm_grip = float(np.where(raw_gripper > 0.5, 1.0, -1.0))
        env_action[-1] = -norm_grip  # raw_gripper_to_env_gripper

    obs, reward, done, info = env.step(env_action)
    env_action_6 = float(env_action[-1])
    is_open = env_action_6 < -0.5

    # Collect qpos from sim
    try:
        qpos = env.sim.data.qpos.copy()
        gripper_qpos = float(qpos[7]) if len(qpos) > 7 else 0.0  # LIBERO gripper joint
        arm_qpos = qpos[:7]
    except:
        gripper_qpos = 0.0; arm_qpos = np.zeros(7)

    arm_l2 = float(np.linalg.norm(arm_qpos[:3])) if len(arm_qpos) >= 3 else 0.0
    qpos_history.append(gripper_qpos); arm_l2_history.append(arm_l2)

    trace_rows.append({
        'step': step, 'raw_gripper': round(raw_gripper, 6),
        'env_action_6': round(env_action_6, 6),
        'is_open': int(is_open), 'oracle_active': int(oracle_active),
        'gripper_qpos': round(gripper_qpos, 8),
        'arm_l2': round(arm_l2, 8),
    })
    decoded_opens.append(int(is_open))
    step += 1

env.close(); torch.cuda.empty_cache()

# ── Compute metrics ──
ws_idx = ws
we_idx = min(we, len(trace_rows))
oracle_we_idx = min(oracle_we, len(trace_rows))
post_start = we_idx
post_end = min(len(trace_rows), we_idx + 40)  # 40-step post-window horizon

pre_qpos = np.array(qpos_history[:ws_idx]) if ws_idx > 0 else np.array([0.0])
window_qpos = np.array(qpos_history[ws_idx:oracle_we_idx]) if oracle_we_idx > ws_idx else np.array([0.0])
post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([0.0])
pre_arm = np.array(arm_l2_history[:ws_idx]) if ws_idx > 0 else np.array([0.0])
window_arm = np.array(arm_l2_history[ws_idx:we_idx]) if we_idx > ws_idx else np.array([0.0])

baseline_qpos = np.median(pre_qpos) if len(pre_qpos) > 0 else 0.0
qpos_delta = np.max(window_qpos) - baseline_qpos if len(window_qpos) > 0 else 0.0
qpos_peak = float(np.max(post_qpos) - baseline_qpos) if len(post_qpos) > 0 else 0.0
qpos_area = float(np.sum(np.maximum(post_qpos - baseline_qpos, 0))) if len(post_qpos) > 0 else 0.0

# Response delay: first step in post-window where qpos exceeds baseline + threshold
threshold = 0.005
response_delay = -1
for i, q in enumerate(post_qpos):
    if q - baseline_qpos > threshold:
        response_delay = i; break

decoded_open_count = sum(1 for i in range(ws_idx, we_idx) if i < len(decoded_opens) and decoded_opens[i])

summary = {
    'job_id': args.job_id, 'pair_id': pair_id,
    'task': args.task, 'state_id': args.state_id,
    'window_start': ws, 'window_end': we,
    'oracle_duration': args.open_duration, 'oracle_end': oracle_we,
    'condition': args.condition,
    'n_steps': step, 'n_window_steps': we_idx - ws_idx,
    'decoded_open_count': decoded_open_count,
    'qpos_baseline': round(float(baseline_qpos), 8),
    'qpos_delta': round(qpos_delta, 8),
    'qpos_peak': round(qpos_peak, 8),
    'qpos_area': round(qpos_area, 8),
    'response_delay': response_delay,
    'mean_arm_l2_pre': round(float(np.mean(pre_arm)), 8) if len(pre_arm) > 0 else 0,
    'mean_arm_l2_window': round(float(np.mean(window_arm)), 8) if len(window_arm) > 0 else 0,
    'infra_status': 'ok',
    'runner': 's8_oracle_open_physical_scan_v0',
}

with open(out_json, 'w') as f: json.dump(summary, f)
with open(out_trace, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
    w.writeheader()
    for r in trace_rows: w.writerow(r)

print('[%s] Done: open=%d qpos_delta=%.6f qpos_peak=%.6f qpos_area=%.6f delay=%d infra=ok' % (
    datetime.now().strftime('%H:%M:%S'), decoded_open_count, qpos_delta, qpos_peak, qpos_area, response_delay))
