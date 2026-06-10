#!/usr/bin/env python3
"""S8 Phase 2 extended VIS/RAND physical diagnostic — qpos measurement + attack."""
import csv, os, sys, argparse, json, hashlib, time
from datetime import datetime
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--task', required=True)
ap.add_argument('--state-id', type=int, required=True)
ap.add_argument('--window_start', type=int, required=True)
ap.add_argument('--window_end', type=int, required=True)
ap.add_argument('--condition', choices=['vis_pgd', 'random_linf'], required=True)
ap.add_argument('--attack_seed', type=int, required=True)
ap.add_argument('--env_seed', type=int, default=None)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--job_id', type=int, default=0)
ap.add_argument('--pair_id', default='')
ap.add_argument('--output_dir', required=True)
ap.add_argument('--max_steps', type=int, default=400)
args, _ = ap.parse_known_args()

if args.env_seed is None: args.env_seed = args.state_id

import torch; from PIL import Image

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE: print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

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
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result

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
env.seed(args.env_seed); env.reset(); env.set_init_state(init_states[args.state_id])
prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()

# Attack setup
rng = np.random.RandomState(args.attack_seed)
rand_gen = torch.Generator(device=model_device)
rand_gen.manual_seed(int(args.attack_seed))
_eps = args.eps_raw_pixels / 255.0
attacker_cfg = dict(epsilon=_eps, step_size=_eps / max(args.pgd_steps, 1) * 1.5,
                    num_steps=args.pgd_steps, loss='open_logit', rand_gen=rand_gen)

ws = args.window_start; we = args.window_end
attack_window = min(we, args.max_steps)
pair_id = args.pair_id or '%s_s%d_w%d_%d_%s_atk%d' % (args.task, args.state_id, ws, we, args.condition, args.attack_seed)

os.makedirs(args.output_dir, exist_ok=True)
out_json = os.path.join(args.output_dir, 'summary_%s_job%d.json' % (args.condition.replace('_',''), args.job_id))

print('[%s] %s s%d w[%d,%d] %s atk=%d' % (datetime.now().strftime('%H:%M:%S'), args.task, args.state_id, ws, we, args.condition, args.attack_seed))

done = False; step = 0; max_steps = max(we + 20, args.max_steps)
decoded_opens = []; qpos_history = []; arm_qpos_history = []
infra_status = 'ok'; pgd_applied = 0

while not done and step < max_steps:
    img = env.sim.render(256, 256, camera_name='agentview')
    img_pil = Image.fromarray(img.astype(np.uint8)).rotate(180)

    in_window = ws <= step < attack_window
    perturbed = False
    if in_window and args.condition in ('vis_pgd', 'random_linf'):
        try:
            if args.condition == 'vis_pgd':
                attacker = TokenPrefixPGDAttacker(model, processor, **attacker_cfg)
                result = attacker.attack(prompt, img_pil, max_new_tokens=action_dim)
                inp = get_adv_inputs_from_attack_result(result, model_device)
                pgd_applied += 1; perturbed = True
            elif args.condition == 'random_linf':
                inp = processor(prompt, img_pil)
                inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
                       if isinstance(v,torch.Tensor) else v for k,v in inp.items()}
                if 'pixel_values' in inp:
                    noise = torch.rand_like(inp['pixel_values'].float()) * 2 * _eps - _eps
                    inp['pixel_values'] = (inp['pixel_values'].float() + noise).clamp(0, 1).to(inp['pixel_values'].dtype)
                perturbed = True
        except Exception as e:
            infra_status = '%s_error: %s' % (args.condition, str(e)[:60])
            inp = processor(prompt, img_pil)
            inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
                   if isinstance(v,torch.Tensor) else v for k,v in inp.items()}

    if not perturbed:
        inp = processor(prompt, img_pil)
        inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
               if isinstance(v,torch.Tensor) else v for k,v in inp.items()}

    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
    raw_action = decode_tokens(token_ids)
    raw_gripper = float(raw_action[-1])
    env_action = raw_action.copy()
    norm_grip = float(np.where(raw_gripper > 0.5, 1.0, -1.0))
    env_action[-1] = -norm_grip
    is_open = env_action[-1] < -0.5

    obs, reward, done, info = env.step(env_action)

    try:
        qpos = env.sim.data.qpos.copy()
        gripper_qpos = float(qpos[7]) if len(qpos) > 7 else 0.0
        arm_qpos = qpos[:7]
    except:
        gripper_qpos = 0.0; arm_qpos = np.zeros(7)
    arm_norm = float(np.linalg.norm(arm_qpos[:3])) if len(arm_qpos) >= 3 else 0.0
    qpos_history.append(gripper_qpos); arm_qpos_history.append(arm_norm)
    decoded_opens.append(int(is_open))
    step += 1

env.close(); torch.cuda.empty_cache()

# ── Metrics ──
ws_idx = ws; we_idx = min(we, len(qpos_history))
post_start = we_idx; post_end = min(len(qpos_history), we_idx + 40)
pre_qpos = np.array(qpos_history[:ws_idx]) if ws_idx > 0 else np.array([0.0])
post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([])
baseline_qpos = float(np.median(pre_qpos)) if len(pre_qpos) > 0 else 0.0

if len(post_qpos) > 0:
    qpos_diff = post_qpos - baseline_qpos
    qpos_pos_peak = float(np.max(qpos_diff))
    qpos_pos_area = float(np.sum(np.maximum(qpos_diff, 0)))
    qpos_neg_peak = float(np.max(-qpos_diff))
    qpos_neg_area = float(np.sum(np.maximum(-qpos_diff, 0)))
    qpos_abs_peak = float(np.max(np.abs(qpos_diff)))
    qpos_abs_area = float(np.sum(np.abs(qpos_diff)))
else:
    qpos_pos_peak = qpos_pos_area = qpos_neg_peak = qpos_neg_area = qpos_abs_peak = qpos_abs_area = 0.0

threshold = 0.005
response_delay_pos = -1; response_delay_neg = -1
for i, q in enumerate(post_qpos):
    d = q - baseline_qpos
    if response_delay_pos < 0 and d > threshold: response_delay_pos = i
    if response_delay_neg < 0 and baseline_qpos - q > threshold: response_delay_neg = i

open_count = sum(1 for i in range(ws_idx, min(we_idx, len(decoded_opens))) if decoded_opens[i])
orig_ws_steps = max(we_idx - ws_idx, 1)

# Longest open streak in window
streak = 0; max_streak = 0
for i in range(ws_idx, min(we_idx, len(decoded_opens))):
    if decoded_opens[i]: streak += 1; max_streak = max(max_streak, streak)
    else: streak = 0

summary = {
    'job_id': args.job_id, 'pair_id': pair_id,
    'task': args.task, 'state_id': args.state_id,
    'window_start': ws, 'window_end': we,
    'condition': args.condition, 'attack_seed': args.attack_seed, 'env_seed': args.env_seed,
    'n_steps': step, 'n_window_steps': we_idx - ws_idx,
    'infra_status': infra_status,
    # Command
    'decoded_open_count': open_count,
    'open_rate': round(open_count / orig_ws_steps, 4),
    'max_open_streak': max_streak,
    # Physical
    'qpos_baseline': round(baseline_qpos, 8),
    'qpos_pos_peak': round(qpos_pos_peak, 8), 'qpos_pos_area': round(qpos_pos_area, 8),
    'qpos_neg_peak': round(qpos_neg_peak, 8), 'qpos_neg_area': round(qpos_neg_area, 8),
    'qpos_abs_peak': round(qpos_abs_peak, 8), 'qpos_abs_area': round(qpos_abs_area, 8),
    'response_delay_pos': response_delay_pos, 'response_delay_neg': response_delay_neg,
    # Attack
    'pgd_applied': pgd_applied,
    'runner': 's8_extended_visrand_physical_v1',
}

with open(out_json, 'w') as f: json.dump(summary, f)

print('[%s] Done: open=%d streak=%d pos_area=%.6f abs_area=%.6f d_pos=%d infra=%s' % (
    datetime.now().strftime('%H:%M:%S'), open_count, max_streak,
    qpos_pos_area, qpos_abs_area, response_delay_pos, infra_status))
