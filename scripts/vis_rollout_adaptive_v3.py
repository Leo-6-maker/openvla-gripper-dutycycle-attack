# -*- coding: utf-8 -*-
"""VIS Controlled Rollout Micro v3 — with online adaptive controller."""
import sys, os, csv, json, time, argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524'
sys.path.insert(0, f'{REPO}/src')
sys.path.insert(0, REPO)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
OUT_BASE = '/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601'
os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(f'{OUT_BASE}/tables', exist_ok=True)
os.makedirs(f'{OUT_BASE}/runs', exist_ok=True)

UNNORM_KEY = 'libero_object'
EPS = 4.0 / 255.0
STEPS = 20
STEP_SIZE = 1.0 / 255.0
ATTACK_OBJECTIVE = 'gripper_open_region_ce'

TASK_CONFIGS = {
    'cream_cheese': {
        'task_id': 1, 'task_name': 'cream_cheese',
        'instruction': 'pick up the cream cheese and place it in the basket',
        'perturb_start': 65, 'perturb_end': 75,
    },
    'tomato_sauce': {
        'task_id': 5, 'task_name': 'tomato_sauce',
        'instruction': 'pick up the tomato sauce and place it in the basket',
        'perturb_start': 128, 'perturb_end': 140,
    },
    'ketchup': {
        'task_id': 4, 'task_name': 'ketchup',
        'instruction': 'pick up the ketchup and place it in the basket',
        'perturb_start': 93, 'perturb_end': 103,
    },
    'salad_dressing': {
        'task_id': 2, 'task_name': 'salad_dressing',
        'instruction': 'put the salad dressing in the basket',
        'perturb_start': 88, 'perturb_end': 108,
    },
}
CONDITIONS = ['clean', 'vis_pgd', 'random_linf']

def prompt(instruction):
    return f"In: What action should the robot take to {instruction}?\nOut:"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=list(TASK_CONFIGS.keys()), required=True)
    ap.add_argument('--condition', choices=CONDITIONS, required=True)
    ap.add_argument('--gpu_pair', default='4,5')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--duration', type=int, default=0)
    ap.add_argument('--strategy', choices=['full','sparse'], default='full')
    ap.add_argument('--controller', choices=['fixed','open_streak_stop','qpos_safety_stop','streak_with_qpos_cap'], default='fixed')
    ap.add_argument('--K', type=int, default=0, help='OPEN streak threshold')
    ap.add_argument('--Q', type=float, default=0, help='qpos_delta threshold')
    ap.add_argument('--max_duration', type=int, default=0, help='max VIS duration')
    ap.add_argument('--dry_run', action='store_true')
    return ap.parse_args()

args = parse_args()
print(f'[0] VIS Rollout Micro: {args.task} / {args.condition} (seed={args.seed}) controller={args.controller}')
print(f'    GPU pair: {args.gpu_pair}')

# Load model
print('[1] Loading model...')
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='auto',
    max_memory={0: '9000MiB', 1: '9000MiB', 'cpu': '64GiB'}, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
device = next(model.parameters()).device
mdtype = next(model.parameters()).dtype
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC = np.array(model.bin_centers)
action_dim = int(model.get_action_dim(UNNORM_KEY))
stats = model.get_action_stats(UNNORM_KEY)
mask = np.array(stats.get('mask', np.ones_like(stats['q01'], dtype=bool)))
low = np.array(stats['q01'])
high = np.array(stats['q99'])
print(f'    device={device}, mdtype={mdtype}, action_dim={action_dim}')

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs

# Init LIBERO env
print('[2] Initializing LIBERO environment...')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
cfg = TASK_CONFIGS[args.task]
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_object']()
task = task_suite.get_task(cfg['task_id'])
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
initial_states = task_suite.get_task_init_states(cfg['task_id'])
state_id = 0
env_args = {
    'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
    'has_renderer': False, 'has_offscreen_renderer': True,
    'use_camera_obs': True, 'camera_names': ['agentview'], 'control_freq': 20,
    'render_gpu_device_id': int(args.gpu_pair.split(',')[0]),
}
env = OffScreenRenderEnv(**env_args)
env.seed(args.seed)

# Apply duration override
if args.duration > 0:
    cfg['perturb_end'] = cfg['perturb_start'] + args.duration - 1

def get_libero_image(obs, resize_size=224):
    img = obs["agentview_image"]; img = img[::-1, ::-1]
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    return np.array(img)

def decode_image(img_np, instruction):
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    text = prompt(str(instruction).lower())
    inputs = processor(text, pil_img, return_tensors='pt')
    inputs.pop('attention_mask', None)
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v): inputs[k] = v.to(device=device, dtype=mdtype)
        else: inputs[k] = v.to(device)
    if not torch.all(inputs['input_ids'][:, -1] == 29871):
        inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([[29871]], dtype=torch.long, device=device)), dim=1)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    disc = np.clip(VS - token_ids - 1, 0, len(BC) - 1)
    norm_actions = BC[disc].astype(np.float32)
    action = np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)
    return action, token_ids

def run_pgd_attack(img_np, instruction, clean_action, clean_gen, seed):
    target_action = np.asarray(clean_action, dtype=np.float32).copy(); target_action[-1] = 1.0
    attack_cfg = {
        'attack_optimizer': {
            'method': 'token_prefix_pgd', 'objective': ATTACK_OBJECTIVE,
            'epsilon': EPS, 'step_size': STEP_SIZE, 'num_steps': STEPS, 'random_start': False,
        }
    }
    attacker = TokenPrefixPGDAttacker(
        model, processor, attack_cfg, seed=seed,
        preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False, 'resize_size': 224}, device=device)
    attack_result = attacker.attack(observation=img_np, instruction=instruction,
        clean_action=clean_action, target_action=target_action,
        clean_model_output=clean_gen, unnorm_key=UNNORM_KEY)
    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
    adv_decoded = redecode_openvla_action_from_adv_inputs(
        model=model, processor=processor, adv_inputs=adv_inputs, instruction=instruction, unnorm_key=UNNORM_KEY)
    return np.asarray(adv_decoded.action, dtype=np.float32), adv_decoded.token_ids, attack_result, time.time()

def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize: action[..., -1] = np.sign(action[..., -1]); action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy(); action[..., -1] = -1.0 * action[..., -1]
    return action

# Run episode
obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
init_state = initial_states[state_id]; env.set_init_state(init_state)
num_steps_wait = 5
for _ in range(num_steps_wait): obs, _, _, _ = env.step(np.zeros(7))

max_steps = 300; trace_rows = []; success = False; done_any = False
rng = np.random.RandomState(args.seed + 100000)

dval = args.duration if args.duration > 0 else cfg['perturb_end'] - cfg['perturb_start'] + 1
ws = cfg['perturb_start']; we = cfg['perturb_end']

# Adaptive controller state
ctrl = {
    'mode': args.controller, 'K': args.K, 'Q': args.Q,
    'max_dur': args.max_duration if args.max_duration > 0 else 999,
    'active': True, 'stop_reason': 'none',
    'current_streak': 0, 'max_streak': 0, 'total_open': 0,
    'qpos_start': 0.0, 'qpos_delta_online': 0.0, 'attacks_applied': 0,
}
if args.controller != 'fixed' and args.condition == 'vis_pgd':
    print(f'    Adaptive controller: {args.controller} K={args.K} Q={args.Q} max_dur={ctrl["max_dur"]}')

print(f'[3] Running episode ({args.condition}), window=[{ws},{we}]')
t_start = time.time(); policy_step = 0; t = num_steps_wait

while t < max_steps + num_steps_wait:
    img_np = get_libero_image(obs, 224)
    eef_pos = obs['robot0_eef_pos'].copy()
    gripper_qpos = obs['robot0_gripper_qpos'].copy()
    in_window = cfg['perturb_start'] <= policy_step <= cfg['perturb_end']

    clean_grip = 0.0; adv_grip = 0.0; arm_l2 = 0.0; linf = 0.0; attack_dt = 0.0; token_flip = False

    if args.condition == 'clean' or not in_window:
        raw_action, _ = decode_image(img_np, cfg['instruction'])
        adv_grip = float(raw_action[-1]); clean_grip = adv_grip

    elif args.condition == 'vis_pgd':
        # Check if adaptive controller already stopped
        ctrl_stopped = (ctrl['mode'] != 'fixed' and ctrl['stop_reason'] != 'none')
        if ctrl_stopped:
            raw_action, _ = decode_image(img_np, cfg['instruction'])
            adv_grip = float(raw_action[-1]); clean_grip = adv_grip
        else:
            try:
                clean_action, clean_token_ids = decode_image(img_np, cfg['instruction'])
                clean_grip = float(clean_action[-1])
                if args.strategy == 'sparse' and abs(clean_grip) > 0.02:
                    raw_action = clean_action; adv_grip = clean_grip
                else:
                    pil_img = Image.fromarray(img_np.astype(np.uint8))
                    inputs = processor(prompt(str(cfg['instruction']).lower()), pil_img, return_tensors='pt')
                    inputs.pop('attention_mask', None)
                    for k, v in list(inputs.items()):
                        if torch.is_floating_point(v): inputs[k] = v.to(device=device, dtype=mdtype)
                        else: inputs[k] = v.to(device)
                    if not torch.all(inputs['input_ids'][:, -1] == 29871):
                        inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([[29871]], dtype=torch.long, device=device)), dim=1)
                    with torch.inference_mode():
                        clean_gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    adv_action, adv_token_ids, atk_result, _ = run_pgd_attack(img_np, cfg['instruction'], clean_action, clean_gen, args.seed + policy_step)
                    raw_action = adv_action; adv_grip = float(raw_action[-1])
                    arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
                    linf = float(atk_result.observation_perturb_linf)
                    token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])
                    # Update controller
                    if ctrl['mode'] != 'fixed':
                        ctrl['attacks_applied'] += 1
                        is_open = adv_grip > 0.5
                        ctrl['current_streak'] = ctrl['current_streak'] + 1 if is_open else 0
                        ctrl['max_streak'] = max(ctrl['max_streak'], ctrl['current_streak'])
                        ctrl['total_open'] += (1 if is_open else 0)
                        gq = float(gripper_qpos[0]) if len(gripper_qpos) > 0 else 0.0
                        if ctrl['attacks_applied'] == 1: ctrl['qpos_start'] = gq
                        ctrl['qpos_delta_online'] = abs(gq - ctrl['qpos_start'])
                        if ctrl['K'] > 0 and ctrl['current_streak'] >= ctrl['K']:
                            ctrl['stop_reason'] = 'streak_threshold'
                        elif ctrl['Q'] > 0 and ctrl['qpos_delta_online'] >= ctrl['Q']:
                            ctrl['stop_reason'] = 'qpos_threshold'
                        elif ctrl['attacks_applied'] >= ctrl['max_dur']:
                            ctrl['stop_reason'] = 'max_duration'
            except Exception as e:
                print(f'    PGD ERROR at step {policy_step}: {str(e)[:100]}')
                raw_action, _ = decode_image(img_np, cfg['instruction'])
                adv_grip = float(raw_action[-1]); clean_grip = adv_grip

    elif args.condition == 'random_linf':
        clean_action, clean_token_ids = decode_image(img_np, cfg['instruction'])
        img_f = img_np.astype(np.float32) / 255.0
        noise = rng.uniform(-EPS, EPS, img_f.shape).astype(np.float32)
        adv_img_np = (np.clip(img_f + noise, 0.0, 1.0) * 255).astype(np.uint8)
        adv_action, adv_token_ids = decode_image(adv_img_np, cfg['instruction'])
        raw_action = adv_action; adv_grip = float(raw_action[-1]); clean_grip = float(clean_action[-1])
        arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
        linf = float(np.abs(noise).max())
        token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])

    env_action = normalize_gripper_action(raw_action, binarize=True)
    env_action = invert_gripper_action(env_action)
    obs, reward, done, info = env.step(env_action)

    trace_rows.append({
        'task': args.task, 'condition': args.condition, 'seed': args.seed,
        'step': t, 'policy_step': policy_step, 'in_window': in_window,
        'raw_gripper': float(raw_action[-1]), 'env_gripper': float(env_action[-1]),
        'gripper_qpos': float(gripper_qpos[0]) if len(gripper_qpos) > 0 else 0,
        'clean_grip': clean_grip, 'adv_grip': adv_grip,
        'arm_l2': arm_l2, 'linf': linf, 'token_flip': token_flip, 'attack_dt': attack_dt,
        'eef_x': float(eef_pos[0]), 'eef_y': float(eef_pos[1]), 'eef_z': float(eef_pos[2]),
        'done': bool(done), 'reward': float(reward),
        'ctrl_mode': ctrl['mode'], 'ctrl_stop_reason': ctrl['stop_reason'],
        'ctrl_streak': ctrl['current_streak'], 'ctrl_max_streak': ctrl['max_streak'],
        'ctrl_qpos_delta': round(ctrl['qpos_delta_online'], 6), 'ctrl_attacks': ctrl['attacks_applied'],
    })
    if done: success = True; done_any = True; break
    t += 1; policy_step += 1
    if t >= max_steps + num_steps_wait - 1: done_any = True; break

total_dt = time.time() - t_start
window_rows = [r for r in trace_rows if r['in_window']]
n_flip = sum(1 for r in window_rows if r['token_flip'])
avg_al = np.mean([r['arm_l2'] for r in window_rows]) if window_rows else 0
print(f'[4] Episode finished: success={success}, steps={policy_step}, time={total_dt:.0f}s')
print(f'    Window: {len(window_rows)} steps, {n_flip} flips, avg armL2={avg_al:.4f}')
if ctrl['stop_reason'] != 'none':
    print(f'    Controller STOP: {ctrl["stop_reason"]} at attacks={ctrl["attacks_applied"]} streak={ctrl["max_streak"]} qpos_d={ctrl["qpos_delta_online"]:.5f}')

ts = time.strftime('%H%M%S')
ctrl_tag = f'_{ctrl["mode"]}_K{ctrl["K"]}_Q{ctrl["Q"]}_md{ctrl["max_dur"]}' if ctrl['mode'] != 'fixed' else ''
run_id = f'vis_{args.task}_s0_{args.condition}_{args.strategy}_d{dval}_w{ws}_{we}_seed{args.seed}{ctrl_tag}_{ts}'
csv_path = f'{OUT_BASE}/runs/{run_id}_trace.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys())); w.writeheader(); w.writerows(trace_rows)
summary = {
    'task': args.task, 'condition': args.condition, 'seed': args.seed, 'success': success,
    'total_steps': policy_step, 'window_start': ws, 'window_end': we, 'window_steps': len(window_rows),
    'window_token_flips': n_flip, 'avg_arm_l2': avg_al, 'total_dt_s': round(total_dt, 1),
    'controller': ctrl['mode'], 'K': ctrl['K'], 'Q': ctrl['Q'], 'max_dur': ctrl['max_dur'],
    'stop_reason': ctrl['stop_reason'], 'attacks_applied': ctrl['attacks_applied'],
    'max_open_streak': ctrl['max_streak'], 'qpos_delta_online': round(ctrl['qpos_delta_online'], 6),
}
print(json.dumps(summary))
print(f'Saved: {csv_path}')
