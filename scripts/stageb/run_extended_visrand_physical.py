#!/usr/bin/env python3
"""S8 Phase 2 extended VIS/RAND physical diagnostic — S6-attack-aligned + qpos.

Inherits the validated S6/S7 attack spec (TokenPrefixPGDAttacker, official prompt,
official image preprocess, seeded RAND generator) and adds physical qpos metrics
consistent with the S8 Phase 1 ORACLE runner.
"""
import csv, os, sys, argparse, json, hashlib, time, uuid, subprocess
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
# Phase 2 additions
ap.add_argument('--original_window_start', type=int, default=None)
ap.add_argument('--original_window_end', type=int, default=None)
ap.add_argument('--oracle_ref_L10_pos_area', type=float, default=0.0)
ap.add_argument('--length_mode', default='short')
args, _ = ap.parse_known_args()

if args.env_seed is None: args.env_seed = args.state_id
if args.original_window_start is None: args.original_window_start = args.window_start
if args.original_window_end is None: args.original_window_end = args.window_end

import torch; from PIL import Image

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE: print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

from gripper_attack.openvla_libero_exec_spec import (
    OFFICIAL_UNNORM_KEY_LIBERO_OBJECT as UNNORM_KEY,
    official_prompt,
    normalize_gripper_raw,
    raw_gripper_to_env_gripper,
    env_gripper_is_open,
    get_libero_image_official,
)
from transformers import AutoModelForVision2Seq, AutoProcessor
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result

print('[%s] Loading model...' % datetime.now().strftime('%H:%M:%S'))
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

# ── S6-validated attack config ──
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

# ── Spec-aligned helpers ──
def decode_tokens(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

def make_inputs(pil_image, instruction_text):
    text = official_prompt(instruction_text.lower())
    inp = processor(text, pil_image, return_tensors='pt')
    for k, v in list(inp.items()):
        if torch.is_floating_point(v):
            inp[k] = v.to(device=model_device, dtype=model_dtype)
        else:
            inp[k] = v.to(model_device)
    if not torch.all(inp['input_ids'][:, -1] == 29871):
        inp['input_ids'] = torch.cat((inp['input_ids'],
            torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)
    return inp

def decode_action(inp):
    with torch.inference_mode():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    tids = gen.sequences[0, -action_dim:].cpu().numpy()
    return decode_tokens(tids)

# ── Env ──
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,'salad_dressing':3,'bbq_sauce':4,'milk':5,'alphabet_soup':6,'tomato_sauce':7,'orange_juice':8}

ws = args.window_start; we = args.window_end
ow_start = args.original_window_start; ow_end = args.original_window_end
pair_id = args.pair_id or '%s_s%d_w%d_%d_%s_atk%d' % (args.task, args.state_id, ws, we, args.condition, args.attack_seed)
logical_pair_key = pair_id  # same
physical_pair_key = '%s_s%d_w%d_%d_L10' % (args.task, args.state_id, ow_start, ow_end)

os.makedirs(args.output_dir, exist_ok=True)
safe_pair = pair_id.replace('/', '_').replace('\\', '_')
out_json = os.path.join(args.output_dir, 'summary_%s_%s_job%d.json' % (safe_pair, args.condition.replace('_',''), args.job_id))
out_trace = os.path.join(args.output_dir, 'trace_%s_%s_job%d.csv' % (safe_pair, args.condition.replace('_',''), args.job_id))

print('[%s] %s s%d w[%d,%d] %s atk=%d L=%s' % (datetime.now().strftime('%H:%M:%S'), args.task, args.state_id, ws, we, args.condition, args.attack_seed, args.length_mode))

cfg = TASK_CFG.get(args.task)
if cfg is None: print('FATAL: unknown task'); sys.exit(1)

bm_dict = benchmark.get_benchmark_dict(); task_suite = bm_dict['libero_object']()
task_obj = task_suite.get_task(cfg); initial_states = task_suite.get_task_init_states(cfg)
if args.state_id >= len(initial_states): print('FATAL: state OOB'); sys.exit(1)
instruction = str(task_obj.language) if hasattr(task_obj,'language') else args.task.replace('_',' ')
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

infra_status = 'ok'
trace_rows = []
decoded_open_bools = []
qpos_history = []  # from env.sim.data.qpos[7] (Phase 1 consistency)
arm_qpos_history = []
current_step = 0; done = False

try:
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                             render_gpu_device_id=_render_gpu)
    env.seed(args.env_seed); obs = env.reset()
    env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(initial_states[args.state_id])

    while not done and current_step < min(we + 5, args.max_steps):
        # ── Official image preprocess ──
        img = get_libero_image_official(obs)
        pil = Image.fromarray(img.astype(np.uint8))
        inputs = make_inputs(pil, instruction)
        clean_pv = inputs['pixel_values']; clean_ids = inputs['input_ids']

        # Decode clean action
        clean_action = decode_action(inputs)
        clean_grip = raw_gripper_to_env_gripper(float(clean_action[-1]))

        # ── Qpos from env.sim (Phase 1 consistency) + obs ──
        try:
            qpos_full = env.sim.data.qpos.copy()
            gripper_qpos = float(qpos_full[7]) if len(qpos_full) > 7 else 0.0
            arm_qpos = qpos_full[:7]
        except:
            gripper_qpos = 0.0; arm_qpos = np.zeros(7)
        arm_norm = float(np.linalg.norm(arm_qpos[:3])) if len(arm_qpos) >= 3 else 0.0
        qpos_history.append(gripper_qpos); arm_qpos_history.append(arm_norm)

        # Also from obs for comparison
        gq = obs.get('robot0_gripper_qpos', np.zeros(2))
        obs_q0, obs_q1 = float(gq[0]), float(gq[1])

        in_window = 1 if ws <= current_step <= we else 0
        attack_this_step = in_window

        env_grip = clean_grip
        arm_l2 = 0.0
        pgd_applied = 0; attacks_applied = 0
        raw_action = clean_action.copy()
        random_seed_str = ''
        noise_linf = '0'; noise_l2 = '0'
        perturbation_space = 'none'

        if attack_this_step:
            if args.condition == 'vis_pgd':
                try:
                    result = attacker.attack(observation=pil, instruction=instruction.lower(),
                                              target_action=clean_action, unnorm_key=UNNORM_KEY)
                    adv_inputs = get_adv_inputs_from_attack_result(result)
                    adv_pv = adv_inputs['pixel_values'].to(device=model_device, dtype=model_dtype)
                    adv_ids = adv_inputs['input_ids'].to(model_device)
                    adv_action = decode_action({'input_ids': adv_ids, 'pixel_values': adv_pv})
                    raw_action = adv_action.copy()
                    env_grip = raw_gripper_to_env_gripper(float(adv_action[-1]))
                    arm_l2 = float(np.linalg.norm((adv_action[:6] - clean_action[:6]).reshape(-1)))
                    pgd_applied = 1; attacks_applied = 1
                    perturbation_space = 'processor_pixel_values_linf'
                except Exception as e:
                    env_grip = clean_grip
                    infra_status = 'pgd_error: %s' % str(e)[:60]

            elif args.condition == 'random_linf':
                try:
                    random_seed_str = str(int(_attack_seed) + args.job_id)
                    rand_gen = torch.Generator(device=clean_pv.device)
                    rand_gen.manual_seed(int(random_seed_str))
                    noise = (2 * torch.rand(clean_pv.shape, device=clean_pv.device,
                                           dtype=clean_pv.dtype, generator=rand_gen) - 1) * _eps_eff
                    rand_pv = torch.clamp(clean_pv + noise, clean_pv - _eps_eff, clean_pv + _eps_eff)
                    noise_linf = str(round(float(noise.abs().max().cpu()), 6))
                    noise_l2 = str(round(float(torch.linalg.vector_norm(noise.reshape(-1)).cpu()), 6))
                    rand_action = decode_action({'input_ids': clean_ids, 'pixel_values': rand_pv})
                    raw_action = rand_action.copy()
                    env_grip = raw_gripper_to_env_gripper(float(rand_action[-1]))
                    arm_l2 = float(np.linalg.norm((rand_action[:6] - clean_action[:6]).reshape(-1)))
                    attacks_applied = 1
                    perturbation_space = 'random_linf_processor_pixel_values'
                except Exception as e:
                    env_grip = clean_grip
                    infra_status = 'random_error: %s' % str(e)[:60]

        # Env action
        env_action_full = raw_action.copy()
        env_action_full[-1] = normalize_gripper_raw(float(raw_action[-1]), binarize=True)
        env_action_full[-1] = -env_action_full[-1]  # invert for LIBERO
        env_action_6 = float(env_action_full[-1])
        decoded_open_bool = int(env_gripper_is_open(env_action_6))

        obs, reward, done, info = env.step(env_action_full)
        decoded_open_bools.append(decoded_open_bool)

        trace_rows.append({
            'step': current_step,
            'in_window': in_window, 'attack_this_step': int(attack_this_step),
            'env_action_6': round(env_action_6, 6),
            'decoded_open_bool': decoded_open_bool,
            'raw_action_6': round(float(raw_action[-1]), 6),
            'gripper_qpos_sim': round(gripper_qpos, 8),
            'gripper_qpos_obs_q0': round(obs_q0, 8),
            'gripper_qpos_obs_q1': round(obs_q1, 8),
            'arm_qpos_norm': round(arm_norm, 8),
            'arm_action_l2': round(arm_l2, 8),
            'pgd_applied': pgd_applied,
            'attacks_applied': attacks_applied,
            'random_seed_str': random_seed_str,
            'noise_linf': noise_linf, 'noise_l2': noise_l2,
            'perturbation_space': perturbation_space,
            'condition': args.condition,
            'pair_id': pair_id,
            'attack_seed': args.attack_seed,
            'env_seed': args.env_seed,
        })
        current_step += 1

    env.close(); torch.cuda.empty_cache()

except Exception as e:
    infra_status = 'env_fatal: %s' % str(e)[:80]
    trace_rows = []; decoded_open_bools = []; qpos_history = []
    try: env.close()
    except: pass
    torch.cuda.empty_cache()

# ── Metrics ──
ws_idx = ws; we_idx = min(we, len(qpos_history))
post_start = we_idx; post_end = min(len(qpos_history), we_idx + 40)
pre_qpos = np.array(qpos_history[:ws_idx]) if ws_idx > 0 else np.array([0.0])
post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([])
baseline_qpos = float(np.median(pre_qpos)) if len(pre_qpos) > 0 else 0.0

qpos_pos_peak = qpos_pos_area = qpos_neg_peak = qpos_neg_area = qpos_abs_peak = qpos_abs_area = 0.0
response_delay_pos = response_delay_neg = -1

if len(post_qpos) > 0:
    qpos_diff = post_qpos - baseline_qpos
    qpos_pos_peak = float(np.max(qpos_diff))
    qpos_pos_area = float(np.sum(np.maximum(qpos_diff, 0)))
    qpos_neg_peak = float(np.max(-qpos_diff))
    qpos_neg_area = float(np.sum(np.maximum(-qpos_diff, 0)))
    qpos_abs_peak = float(np.max(np.abs(qpos_diff)))
    qpos_abs_area = float(np.sum(np.abs(qpos_diff)))
    threshold = 0.005
    for i, q in enumerate(post_qpos):
        d = q - baseline_qpos
        if response_delay_pos < 0 and d > threshold: response_delay_pos = i
        if response_delay_neg < 0 and baseline_qpos - q > threshold: response_delay_neg = i

open_count = sum(1 for i in range(ws_idx, min(we_idx, len(decoded_open_bools))) if decoded_open_bools[i])
orig_ws_steps = max(we_idx - ws_idx, 1)
streak = 0; max_streak = 0
for i in range(ws_idx, min(we_idx, len(decoded_open_bools))):
    if decoded_open_bools[i]: streak += 1; max_streak = max(max_streak, streak)
    else: streak = 0

mean_pre_arm = float(np.mean(arm_qpos_history[:ws_idx])) if ws_idx > 0 and len(arm_qpos_history) > ws_idx else 0.0

# Git info
try:
    r = subprocess.run(['git','-C',REPO,'rev-parse','--short','HEAD'], capture_output=True, text=True, timeout=5)
    git_commit = r.stdout.strip() if r.returncode == 0 else 'unknown'
except:
    git_commit = 'unknown'

summary = {
    'job_id': args.job_id, 'pair_id': pair_id, 'logical_pair_key': pair_id,
    'task': args.task, 'state_id': args.state_id,
    'window_start': ws, 'window_end': we,
    'original_window_start': ow_start, 'original_window_end': ow_end,
    'length_mode': args.length_mode,
    'physical_pair_key': physical_pair_key,
    'oracle_ref_L10_pos_area': args.oracle_ref_L10_pos_area,
    'condition': args.condition, 'attack_seed': args.attack_seed, 'env_seed': args.env_seed,
    'eps_raw_pixels': args.eps_raw_pixels, 'pgd_steps': args.pgd_steps,
    'n_steps': current_step, 'n_window_steps': we_idx - ws_idx,
    'infra_status': infra_status,
    'git_commit': git_commit,
    # Command
    'decoded_open_count': open_count, 'open_rate': round(open_count / orig_ws_steps, 4),
    'max_open_streak': max_streak,
    # Physical (Phase 1 consistency)
    'qpos_baseline': round(baseline_qpos, 8),
    'qpos_pos_peak': round(qpos_pos_peak, 8), 'qpos_pos_area': round(qpos_pos_area, 8),
    'qpos_neg_peak': round(qpos_neg_peak, 8), 'qpos_neg_area': round(qpos_neg_area, 8),
    'qpos_abs_peak': round(qpos_abs_peak, 8), 'qpos_abs_area': round(qpos_abs_area, 8),
    'response_delay_pos': response_delay_pos, 'response_delay_neg': response_delay_neg,
    'mean_arm_qpos_norm_pre': round(mean_pre_arm, 8),
    # Attack provenance
    'attack_objective': 'prefix_locked_gripper_open_margin',
    'random_linf_seed_rule': 'attack_seed + job_id',
    'runner': 's8_extended_visrand_physical_v2',
}

with open(out_json, 'w') as f: json.dump(summary, f)

if trace_rows:
    with open(out_trace, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader()
        for r in trace_rows: w.writerow(r)

print('[%s] Done: open=%d streak=%d pos_area=%.6f abs=%.6f d_pos=%d infra=%s' % (
    datetime.now().strftime('%H:%M:%S'), open_count, max_streak,
    qpos_pos_area, qpos_abs_area, response_delay_pos, infra_status))
