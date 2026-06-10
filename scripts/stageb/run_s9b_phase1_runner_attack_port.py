#!/usr/bin/env python3
"""S9b Phase1-runner attack port — Phase1 ORACLE dynamics + S6 VIS/RAND attack."""
import csv, os, sys, argparse, json, hashlib, time, subprocess
from datetime import datetime
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--task', required=True)
ap.add_argument('--state-id', type=int, required=True)
ap.add_argument('--window_start', type=int, required=True)
ap.add_argument('--window_end', type=int, required=True)
ap.add_argument('--condition', choices=['clean','oracle_open','vis_pgd','random_linf'], required=True)
ap.add_argument('--open_duration', type=int, default=10)
ap.add_argument('--attack_seed', type=int, default=0)
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
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result

print('[%s] Loading model...' % datetime.now().strftime('%H:%M:%S'))
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
UNNORM_KEY = 'libero_object'; action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

def decode_tokens(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

def generate_action_from_inputs(input_ids, pixel_values):
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                             max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    tids = gen.sequences[0, -action_dim:].cpu().numpy()
    return decode_tokens(tids)

# S6 attack config
_eps_eff = args.eps_raw_pixels / 255.0
_attack_seed = args.attack_seed
attacker_config = {
    'epsilon': _eps_eff,
    'step_size': _eps_eff / max(args.pgd_steps, 1) * 1.5,
    'num_steps': args.pgd_steps,
    'random_start': True,
    'objective': 'prefix_locked_gripper_open_margin',
    'arm_preserve_weight': 0.5,
    'gripper_margin': 5.0,
}
attacker = TokenPrefixPGDAttacker(
    model=model, processor=processor, config=attacker_config, seed=_attack_seed,
    device='cuda:%d' % gpu_ids[0], preprocess_kwargs={'postprocess_gripper': True})
attacker._freeze_model()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# TASK_CFG removed — task index resolved from actual LIBERO task metadata at runtime
def _canonical_task_key(task_obj):
    text = ' '.join([getattr(task_obj, 'language', ''),
                     getattr(task_obj, 'problem_folder', ''),
                     getattr(task_obj, 'bddl_file', '')]).lower()
    if 'alphabet' in text: return 'alphabet_soup'
    if 'cream' in text: return 'cream_cheese'
    if 'salad' in text: return 'salad_dressing'
    if 'bbq' in text or 'barbecue' in text: return 'bbq_sauce'
    if 'ketchup' in text: return 'ketchup'
    if 'tomato' in text: return 'tomato_sauce'
    if 'butter' in text: return 'butter'
    if 'milk' in text: return 'milk'
    if 'chocolate' in text: return 'chocolate_pudding'
    if 'orange' in text and 'juice' in text: return 'orange_juice'
    raise ValueError('Unknown task: %s' % text)
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

_actual_by_key = {}
for _i in range(len(task_suite.tasks)):
    _tobj = task_suite.get_task(_i)
    _key = _canonical_task_key(_tobj)
    if _key in _actual_by_key:
        raise RuntimeError('duplicate canonical task key: %s (idx %d and %d)' % (_key, _actual_by_key[_key], _i))
    _actual_by_key[_key] = _i
assert len(_actual_by_key) == len(task_suite.tasks), 'canonical coverage: %d keys != %d tasks' % (len(_actual_by_key), len(task_suite.tasks))

cfg = _actual_by_key.get(args.task)
if cfg is None: print('Unknown task:', args.task, 'available:', sorted(_actual_by_key.keys())); sys.exit(1)
task_obj = task_suite.get_task(cfg); init_states = task_suite.get_task_init_states(cfg)
actual_task_key = _canonical_task_key(task_obj)
assert actual_task_key == args.task, 'FATAL: canonical task key %r != requested task %r' % (actual_task_key, args.task)
if args.state_id >= len(init_states): print('state_id out of range'); sys.exit(1)

instruction = task_obj.language if hasattr(task_obj,'language') else args.task.replace('_',' ')
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

# Phase1 init: seed → reset → set_init_state (no qvel zero)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                         has_renderer=False, has_offscreen_renderer=True,
                         use_camera_obs=True, camera_names=['agentview'],
                         control_freq=20, render_gpu_device_id=render_physical_gpu)
env.seed(args.env_seed); env.reset(); env.set_init_state(init_states[args.state_id])
prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()

ws = args.window_start; we = args.window_end
oracle_we = min(ws + args.open_duration, args.max_steps)
pair_id = args.pair_id or '%s_s%d_w%d_%d_L%d_%s_atk%d' % (args.task, args.state_id, ws, we, args.open_duration, args.condition, args.attack_seed)

os.makedirs(args.output_dir, exist_ok=True)
safe_pair = pair_id.replace('/','_').replace('\\','_')
out_json = os.path.join(args.output_dir, 'summary_%s_%s_job%d.json' % (safe_pair, args.condition.replace('_',''), args.job_id))
out_trace = os.path.join(args.output_dir, 'trace_%s_%s_job%d.csv' % (safe_pair, args.condition.replace('_',''), args.job_id))

print('[%s] %s s%d w[%d,%d] L=%d %s atk=%d' % (datetime.now().strftime('%H:%M:%S'), args.task, args.state_id, ws, we, args.open_duration, args.condition, args.attack_seed))

done = False; step = 0; max_steps_local = max(we, oracle_we) + 20
trace_rows = []; decoded_open_bools = []; qpos_history = []; arm_l2_history = []
infra_status = 'ok'

while not done and step < max_steps_local:
    img = env.sim.render(256, 256, camera_name='agentview')
    img_pil = Image.fromarray(img.astype(np.uint8)).rotate(180)
    inp = processor(prompt, img_pil)
    inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
           if isinstance(v,torch.Tensor) else v for k,v in inp.items()}

    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
    clean_action = decode_tokens(token_ids)
    raw_action = clean_action.copy()
    raw_gripper = float(clean_action[-1])

    # qpos
    try:
        qpos = env.sim.data.qpos.copy()
        gripper_qpos = float(qpos[7]) if len(qpos) > 7 else 0.0
        arm_qpos = qpos[:7]
    except:
        gripper_qpos = 0.0; arm_qpos = np.zeros(7)
    arm_l2 = float(np.linalg.norm(arm_qpos[:3])) if len(arm_qpos) >= 3 else 0.0
    qpos_history.append(gripper_qpos); arm_l2_history.append(arm_l2)

    in_window = ws <= step < we  # half-open
    attack_this_step = in_window and args.condition not in ('clean','oracle_open')
    oracle_active = in_window and args.condition == 'oracle_open'

    pgd_applied = 0; attacks_applied = 0
    random_seed_str = ''; noise_linf = '0'; noise_l2 = '0'
    perturbation_space = 'none'
    env_grip_raw = raw_gripper

    if oracle_active:
        raw_action[-1] = 1.0  # binarize → OPEN
        perturbation_space = 'oracle_forced_open'

    elif attack_this_step:
        if args.condition == 'vis_pgd':
            try:
                result = attacker.attack(observation=img_pil, instruction=instruction.lower(),
                                          target_action=clean_action, unnorm_key=UNNORM_KEY)
                adv_inputs = get_adv_inputs_from_attack_result(result)
                adv_pv = adv_inputs['pixel_values'].to(device=model_device, dtype=model_dtype)
                adv_ids = adv_inputs['input_ids'].to(model_device)
                adv_action = generate_action_from_inputs(adv_ids, adv_pv)
                raw_action = adv_action.copy()
                pgd_applied = 1; attacks_applied = 1
                perturbation_space = 'processor_pixel_values_linf'
            except Exception as e:
                infra_status = 'pgd_error: %s' % str(e)[:60]

        elif args.condition == 'random_linf':
            try:
                pv_clean = inp['pixel_values']
                random_seed_str = str(int(_attack_seed) + args.job_id)
                rand_gen = torch.Generator(device=pv_clean.device)
                rand_gen.manual_seed(int(random_seed_str))
                noise = (2 * torch.rand(pv_clean.shape, device=pv_clean.device,
                                       dtype=pv_clean.dtype, generator=rand_gen) - 1) * _eps_eff
                rand_pv = torch.clamp(pv_clean + noise, pv_clean - _eps_eff, pv_clean + _eps_eff)
                noise_linf = str(round(float(noise.abs().max().cpu()), 6))
                noise_l2 = str(round(float(torch.linalg.vector_norm(noise.reshape(-1)).cpu()), 6))
                inp_rand = {k: (rand_pv if k == 'pixel_values' else v) for k, v in inp.items()}
                gen2 = model.generate(**inp_rand, max_new_tokens=action_dim, do_sample=False,
                                       return_dict_in_generate=True, output_scores=True)
                rand_action = decode_tokens(gen2.sequences[0, -action_dim:].cpu().numpy())
                raw_action = rand_action.copy()
                attacks_applied = 1
                perturbation_space = 'random_linf_processor_pixel_values'
            except Exception as e:
                infra_status = 'random_error: %s' % str(e)[:60]

    # Binarize gripper for env
    env_action = raw_action.copy()
    env_action[-1] = -1.0 if float(raw_action[-1]) > 0.5 else 1.0
    env_action_6 = float(env_action[-1])
    is_open = int(env_action_6 < -0.5)

    obs, reward, done, info = env.step(env_action)
    decoded_open_bools.append(is_open)

    trace_rows.append({
        'step': step, 'in_window': int(in_window), 'attack_this_step': int(attack_this_step),
        'oracle_active': int(oracle_active),
        'env_action_6': round(env_action_6, 6), 'decoded_open_bool': is_open,
        'raw_action_6': round(float(raw_action[-1]), 6),
        'gripper_qpos': round(gripper_qpos, 8), 'arm_qpos_norm': round(arm_l2, 8),
        'pgd_applied': pgd_applied, 'attacks_applied': attacks_applied,
        'random_seed_str': random_seed_str, 'noise_linf': noise_linf, 'noise_l2': noise_l2,
        'perturbation_space': perturbation_space,
        'requested_task': args.task, 'actual_task_key': actual_task_key,
	        'actual_task_idx': cfg, 'actual_language': instruction,
	        'actual_bddl_file': task_obj.bddl_file,
	        'condition': args.condition, 'pair_id': pair_id,
        'attack_seed': args.attack_seed, 'env_seed': args.env_seed,
    })
    step += 1

env.close(); torch.cuda.empty_cache()

# Metrics — Phase1 convention: post-window after oracle duration
post_start = oracle_we; post_end = min(len(trace_rows), oracle_we + 40)
pre_qpos = np.array(qpos_history[:ws]) if ws > 0 else np.array([0.0])
post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([])
baseline_qpos = float(np.median(pre_qpos)) if len(pre_qpos) > 0 else 0.0

qpos_pos_peak = qpos_pos_area = qpos_neg_peak = qpos_neg_area = qpos_abs_peak = qpos_abs_area = 0.0
response_delay_pos = response_delay_neg = -1
if len(post_qpos) > 0:
    qpos_diff = post_qpos - baseline_qpos
    qpos_pos_peak = float(np.max(qpos_diff)); qpos_pos_area = float(np.sum(np.maximum(qpos_diff,0)))
    qpos_neg_peak = float(np.max(-qpos_diff)); qpos_neg_area = float(np.sum(np.maximum(-qpos_diff,0)))
    qpos_abs_peak = float(np.max(np.abs(qpos_diff))); qpos_abs_area = float(np.sum(np.abs(qpos_diff)))
    threshold = 0.005
    for i, q in enumerate(post_qpos):
        d = q - baseline_qpos
        if response_delay_pos < 0 and d > threshold: response_delay_pos = i
        if response_delay_neg < 0 and baseline_qpos - q > threshold: response_delay_neg = i

open_count = sum(1 for i in range(ws, min(we, len(decoded_open_bools))) if decoded_open_bools[i])
ws_steps = max(we - ws, 1)
streak = max_streak = 0
for i in range(ws, min(we, len(decoded_open_bools))):
    if decoded_open_bools[i]: streak += 1; max_streak = max(max_streak, streak)
    else: streak = 0

try:
    r = subprocess.run(['git','-C',REPO,'rev-parse','--short','HEAD'], capture_output=True, text=True, timeout=5)
    git_commit = r.stdout.strip() if r.returncode == 0 else 'unknown'
except: git_commit = 'unknown'

summary = {
    'job_id': args.job_id, 'pair_id': pair_id,
    'task': args.task, 'actual_task_key': actual_task_key,
	    'actual_task_idx': cfg, 'actual_language': instruction,
	    'actual_problem_folder': task_obj.problem_folder, 'actual_bddl_file': task_obj.bddl_file,
	    'state_id': args.state_id,
    'window_start': ws, 'window_end': we, 'open_duration': args.open_duration,
    'condition': args.condition, 'attack_seed': args.attack_seed, 'env_seed': args.env_seed,
    'eps_raw_pixels': args.eps_raw_pixels, 'pgd_steps': args.pgd_steps,
    'n_steps': step, 'n_window_steps': we - ws,
    'infra_status': infra_status, 'git_commit': git_commit,
    'runner_family': 'phase1_oracle_port',
    'adv_redecode_mode': 'model_generate_from_adv_inputs',
    'image_source': 'env.sim.render_rotate180',
    'prompt_mode': 'phase1_handrolled',
    'attack_objective': 'prefix_locked_gripper_open_margin',
    'qpos_baseline': round(baseline_qpos, 8),
    'qpos_pos_peak': round(qpos_pos_peak, 8), 'qpos_pos_area': round(qpos_pos_area, 8),
    'qpos_neg_peak': round(qpos_neg_peak, 8), 'qpos_neg_area': round(qpos_neg_area, 8),
    'qpos_abs_peak': round(qpos_abs_peak, 8), 'qpos_abs_area': round(qpos_abs_area, 8),
    'response_delay_pos': response_delay_pos, 'response_delay_neg': response_delay_neg,
    'decoded_open_count': open_count, 'max_open_streak': max_streak,
    'open_rate': round(open_count / ws_steps, 4),
    'mean_arm_qpos_norm_pre': round(float(np.mean(arm_l2_history[:ws])), 8) if ws > 0 and len(arm_l2_history) > ws else 0,
}

with open(out_json, 'w') as f: json.dump(summary, f)
if trace_rows:
    with open(out_trace, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader()
        for r in trace_rows: w.writerow(r)

print('[%s] Done: open=%d streak=%d pos=%.6f neg=%.6f abs=%.6f d_pos=%d infra=%s' % (
    datetime.now().strftime('%H:%M:%S'), open_count, max_streak,
    qpos_pos_area, qpos_neg_area, qpos_abs_area, response_delay_pos, infra_status))
