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
# Attack strength defaults (overridden by CLI / processor at runtime).
# eps_raw_pixels is the canonical budget: "8 raw pixel values Linf in [0,255]".
# Processor-space epsilon is computed as eps_raw / (255 * image_std_c) per channel.
EPS_RAW_PIXELS_DEFAULT = 8
STEPS_DEFAULT = 40
STEP_SIZE_RATIO = 8.0             # step_size = eps_processor / STEP_SIZE_RATIO
ATTACK_OBJECTIVE_DEFAULT = 'force_open_z_down_token_ce'
PGD_RESTARTS_DEFAULT = 3
Z_DOWN_WEIGHT_DEFAULT = 0.5
GRIPPER_OPEN_WEIGHT_DEFAULT = 1.0

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
    ap.add_argument('--controller', choices=['fixed','open_streak_stop','open_count_stop','qpos_safety_stop','min_hold_qpos_cap','streak_with_qpos_cap'], default='fixed')
    ap.add_argument('--K', type=int, default=0, help='OPEN streak/count threshold')
    ap.add_argument('--Q', type=float, default=0, help='qpos_delta threshold')
    ap.add_argument('--max_duration', type=int, default=0, help='max PGD attacks for adaptive controller')
    ap.add_argument('--min_attacks', type=int, default=0, help='min PGD attacks before allowing stop')
    # EPS: raw-pixel semantics (default) or processor-direct (legacy).
    ap.add_argument('--eps_raw_pixels', type=int, choices=[4, 8, 12, 16], default=None,
        help=f'raw RGB Linf budget in pixel values [0-255] (default: {EPS_RAW_PIXELS_DEFAULT})')
    ap.add_argument('--eps_processor_direct', type=float, default=None,
        help='use processor-space epsilon directly (legacy; bypasses raw-pixel conversion)')
    ap.add_argument('--pgd_steps', type=int, default=STEPS_DEFAULT, help=f'PGD iterations (default: {STEPS_DEFAULT})')
    ap.add_argument('--pgd_restarts', type=int, default=PGD_RESTARTS_DEFAULT, help=f'PGD random restarts (default: {PGD_RESTARTS_DEFAULT})')
    ap.add_argument('--objective', choices=['gripper_open_region_ce','force_open_z_down_token_ce',
        'force_open_region_z_down_ce',
        'force_gripper_open_token_ce','gripper_logit_margin_cw','targeted_directional_ce',
        'prefix_locked_gripper_open_region_ce','prefix_locked_gripper_open_margin',
        'gripper_open_expected_action'],
        default=ATTACK_OBJECTIVE_DEFAULT, help=f'attack objective (default: {ATTACK_OBJECTIVE_DEFAULT})')
    ap.add_argument('--z_down_weight', type=float, default=Z_DOWN_WEIGHT_DEFAULT,
        help=f'Z-down loss weight (default: {Z_DOWN_WEIGHT_DEFAULT})')
    ap.add_argument('--gripper_weight', type=float, default=GRIPPER_OPEN_WEIGHT_DEFAULT,
        help=f'gripper-open loss weight (default: {GRIPPER_OPEN_WEIGHT_DEFAULT})')
    ap.add_argument('--arm_preserve_weight', type=float, default=0.1,
        help='arm-preserve CE weight for prefix-locked objectives (default: 0.1)')
    ap.add_argument('--gripper_margin', type=float, default=5.0,
        help='gripper margin for prefix_locked_gripper_open_margin (default: 5.0)')
    ap.add_argument('--best_restart_metric', choices=['target_ce_final','gripper_open_prob_mass',
        'gripper_margin','decoded_gripper_open','composite'],
        default='gripper_open_prob_mass', help='metric for best-restart selection (default: gripper_open_prob_mass)')
    ap.add_argument('--dry_run', action='store_true')
    return ap.parse_args()

args = parse_args()
# Resolve EPS: raw-pixel semantics (default) or processor-direct (legacy flag).
EPS_RAW_PIXELS = args.eps_raw_pixels if args.eps_raw_pixels is not None else EPS_RAW_PIXELS_DEFAULT
EPS_PROCESSOR_DIRECT = args.eps_processor_direct
STEPS = args.pgd_steps
ATTACK_OBJECTIVE = args.objective
PGD_RESTARTS = args.pgd_restarts
Z_DOWN_WEIGHT = args.z_down_weight
GRIPPER_WEIGHT = args.gripper_weight
print(f'[0] VIS Rollout Micro: {args.task} / {args.condition} (seed={args.seed}) controller={args.controller}')
print(f'    GPU pair: {args.gpu_pair}')

# Load model — use --gpu_pair for device placement.
_gpu_ids = [int(x.strip()) for x in args.gpu_pair.split(',')]
print('[1] Loading model...')
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='auto',
    max_memory={_gpu_ids[0]: '9000MiB', _gpu_ids[1]: '9000MiB', 'cpu': '64GiB'}, trust_remote_code=True)
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

# ── EPS calibration from processor ──
# Read image normalization parameters to convert raw-pixel Linf → processor space.
if EPS_PROCESSOR_DIRECT is not None:
    EPS = float(EPS_PROCESSOR_DIRECT)
    EPS_SOURCE = 'eps_processor_direct'
    IMAGE_MEAN = None
    IMAGE_STD = None
    EPS_PER_CHANNEL = None
    EFFECTIVE_RAW_EPS = None
else:
    try:
        ip = processor.image_processor
        # PrismaticImageProcessor stores means/stds as lists-of-lists.
        # Use the LAST normalization (SigLIP) which is the final pre-model transform.
        all_means = getattr(ip, 'means', [[0.5, 0.5, 0.5]])
        all_stds = getattr(ip, 'stds', [[0.5, 0.5, 0.5]])
        IMAGE_MEAN = all_means[-1] if all_means else [0.5, 0.5, 0.5]
        IMAGE_STD = all_stds[-1] if all_stds else [0.5, 0.5, 0.5]
    except Exception:
        IMAGE_MEAN = [0.5, 0.5, 0.5]
        IMAGE_STD = [0.5, 0.5, 0.5]
    # Convert: delta_processor_c = delta_raw / (255 * std_c) for each channel.
    # Conservative choice: use the channel with the LARGEST std → smallest eps.
    # This ensures the most-constrained channel still respects the budget.
    EPS_PER_CHANNEL = [(EPS_RAW_PIXELS / 255.0) / float(s) for s in IMAGE_STD]
    EPS = min(EPS_PER_CHANNEL)
    EFFECTIVE_RAW_EPS = [EPS * 255.0 * float(s) for s in IMAGE_STD]
    EPS_SOURCE = 'eps_raw_pixels'
print(f'    EPS source: {EPS_SOURCE}')
print(f'    Image mean: {IMAGE_MEAN}')
print(f'    Image std:  {IMAGE_STD}')
if EPS_PER_CHANNEL is not None:
    print(f'    eps_raw_pixels: {EPS_RAW_PIXELS}')
    print(f'    eps_processor_per_channel: {[round(e, 6) for e in EPS_PER_CHANNEL]}')
    print(f'    eps_processor (min across channels): {EPS:.6f}')
    print(f'    effective_raw_eps_recovered: {[round(e, 4) for e in EFFECTIVE_RAW_EPS]}')
else:
    print(f'    eps_processor_direct: {EPS:.6f}')
STEP_SIZE = EPS / STEP_SIZE_RATIO
print(f'    steps={STEPS} step_size={STEP_SIZE:.6f} restarts={PGD_RESTARTS}')
print(f'    objective={ATTACK_OBJECTIVE} z_weight={Z_DOWN_WEIGHT} grip_weight={GRIPPER_WEIGHT}')

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

# Extend perturbation window to accommodate max_duration or explicit duration
if args.max_duration > 0:
    cfg['perturb_end'] = max(cfg['perturb_end'], cfg['perturb_start'] + args.max_duration - 1)
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
    target_action = np.asarray(clean_action, dtype=np.float32).copy()
    target_action[-1] = 1.0  # OPEN gripper (raw ~1.0 → most-open bin after normalization)
    if ATTACK_OBJECTIVE == 'force_open_z_down_token_ce':
        # DEPRECATED: target_action[-1]=1.0 may map to CLOSE in decoded-action space.
        # Use force_open_region_z_down_ce for corrected hybrid with OPEN-region gripper loss.
        target_action[2] = low[2]  # Z DOWN: minimum Z delta = most negative displacement
    if ATTACK_OBJECTIVE == 'force_open_region_z_down_ce':
        # Corrected hybrid: gripper uses corrected OPEN-region loss, Z uses CE toward down token.
        # target_action[-1] is set to 1.0 only for tokenization; actual loss uses region.
        target_action[2] = low[2]  # Z DOWN
    _PREFIX_LOCKED_SET = {'prefix_locked_gripper_open_region_ce', 'prefix_locked_gripper_open_margin', 'gripper_open_expected_action'}
    _GRIPPER_OBJ_SET = {'gripper_open_region_ce', 'force_open_z_down_token_ce'} | _PREFIX_LOCKED_SET
    base_random_start = (ATTACK_OBJECTIVE in _GRIPPER_OBJ_SET)
    best_result = None
    best_metric_val = float('-inf') if ATTACK_OBJECTIVE in _GRIPPER_OBJ_SET else float('inf')
    for restart in range(max(1, PGD_RESTARTS)):
        restart_seed = seed + restart * 1000
        attack_cfg = {
            'attack_optimizer': {
                'method': 'token_prefix_pgd', 'objective': ATTACK_OBJECTIVE,
                'epsilon': EPS, 'step_size': STEP_SIZE, 'num_steps': STEPS,
                'random_start': (base_random_start and restart > 0),
            }
        }
        if ATTACK_OBJECTIVE in {'force_open_z_down_token_ce', 'force_open_region_z_down_ce'}:
            attack_cfg['attack_optimizer']['loss_weights'] = {
                str(action_dim - 1): GRIPPER_WEIGHT,  # gripper dim
                '2': Z_DOWN_WEIGHT,                    # z dim
            }
        if ATTACK_OBJECTIVE in _PREFIX_LOCKED_SET:
            attack_cfg['attack_optimizer']['arm_preserve_weight'] = float(args.arm_preserve_weight)
            attack_cfg['attack_optimizer']['gripper_margin'] = float(args.gripper_margin)
            attack_cfg['attack_optimizer']['best_restart_metric'] = args.best_restart_metric
        attacker = TokenPrefixPGDAttacker(
            model, processor, attack_cfg, seed=restart_seed,
            preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False, 'resize_size': 224,
                               'postprocess_gripper': True}, device=device)
        attack_result = attacker.attack(observation=img_np, instruction=instruction,
            clean_action=clean_action, target_action=target_action,
            clean_model_output=clean_gen, unnorm_key=UNNORM_KEY)
        debug = attack_result.debug or {}
        # P1 FIX: restart selection uses actual generated output, not teacher-forced metrics.
        if ATTACK_OBJECTIVE in _GRIPPER_OBJ_SET:
            # Re-decode adversarial action to check actual generated gripper output
            try:
                from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs
                _adv_inputs = debug.get('adv_inputs')
                if _adv_inputs is not None:
                    _adv_decoded = redecode_openvla_action_from_adv_inputs(
                        model=model, processor=processor, adv_inputs=_adv_inputs,
                        instruction=str(instruction), unnorm_key=UNNORM_KEY)
                    _gen_action = np.asarray(_adv_decoded.action, dtype=np.float32)
                    _is_open = float(_gen_action[-1]) < 0.5
                    _nad_dof7 = abs(float(_gen_action[-1]) - float(clean_action[-1]))
                    _arm_l2 = float(np.linalg.norm(_gen_action[:6] - clean_action[:6]))
                    # Priority: true OPEN > highest NAD > highest teacher-forced open_prob
                    _tf_open = float(debug.get('gripper_open_prob_mass', 0.0) or 0.0)
                    _score = (1.0 if _is_open else 0.0) + 0.01 * _nad_dof7 + 0.0001 * _tf_open - 0.001 * _arm_l2
                    if _score > best_metric_val:
                        best_metric_val = _score
                        best_result = attack_result
                else:
                    # Fallback: no adv_inputs, use teacher-forced prob
                    _tf_open = float(debug.get('gripper_open_prob_mass', 0.0) or 0.0)
                    if _tf_open > best_metric_val:
                        best_metric_val = _tf_open
                        best_result = attack_result
            except Exception:
                _tf_open = float(debug.get('gripper_open_prob_mass', 0.0) or 0.0)
                if _tf_open > best_metric_val:
                    best_metric_val = _tf_open
                    best_result = attack_result
        else:
            loss = float(debug.get('target_ce_final', float('inf')))
            if loss < best_metric_val:
                best_metric_val = loss
                best_result = attack_result
        if PGD_RESTARTS <= 1:
            break
    adv_inputs = get_adv_inputs_from_attack_result(best_result)
    adv_decoded = redecode_openvla_action_from_adv_inputs(
        model=model, processor=processor, adv_inputs=adv_inputs, instruction=instruction, unnorm_key=UNNORM_KEY)
    return np.asarray(adv_decoded.action, dtype=np.float32), adv_decoded.token_ids, best_result

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
    'min_att': args.min_attacks if args.min_attacks > 0 else 0,
    'active': True, 'stop_reason': 'none',
    'current_streak': 0, 'max_streak': 0, 'total_open': 0,
    'qpos_start': 0.0, 'qpos_delta_online': 0.0, 'attacks_applied': 0,
    'qpos_pre': 0.0, 'qpos_post': 0.0,
}
if args.controller != 'fixed' and args.condition == 'vis_pgd':
    print(f'    Adaptive controller: {args.controller} K={args.K} Q={args.Q} max_dur={ctrl["max_dur"]}')

print(f'[3] Running episode ({args.condition}), window=[{ws},{we}]')
t_start = time.time(); policy_step = 0; t = num_steps_wait

while t < max_steps + num_steps_wait:
    img_np = get_libero_image(obs, 224)
    eef_pos = obs['robot0_eef_pos'].copy()
    gripper_qpos = obs['robot0_gripper_qpos'].copy()
    qpos_pre_step = float(gripper_qpos[0]) if len(gripper_qpos) > 0 else 0.0
    in_window = cfg['perturb_start'] <= policy_step <= cfg['perturb_end']

    clean_grip = 0.0; adv_grip = 0.0; arm_l2 = 0.0; linf = 0.0; attack_dt = 0.0; token_flip = False
    attack_attempted = False
    pgd_applied = False
    controller_stopped = (ctrl['mode'] != 'fixed' and ctrl['stop_reason'] != 'none')
    controller_active = (args.condition == 'vis_pgd' and in_window and not controller_stopped)
    effective_attack_step_idx = ''
    clean_action_vec = None  # will be set in each branch for NAD computation

    if args.condition == 'clean' or not in_window:
        raw_action, _ = decode_image(img_np, cfg['instruction'])
        adv_grip = float(raw_action[-1]); clean_grip = adv_grip
        clean_action_vec = raw_action.copy()

    elif args.condition == 'vis_pgd':
        # Check if adaptive controller already stopped
        ctrl_stopped = (ctrl['mode'] != 'fixed' and ctrl['stop_reason'] != 'none')
        controller_stopped = ctrl_stopped
        controller_active = not ctrl_stopped
        if ctrl_stopped:
            raw_action, _ = decode_image(img_np, cfg['instruction'])
            adv_grip = float(raw_action[-1]); clean_grip = adv_grip
            clean_action_vec = raw_action.copy()
        else:
            attack_attempted = True
            try:
                clean_action, clean_token_ids = decode_image(img_np, cfg['instruction'])
                clean_action_vec = clean_action.copy()
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
                    attack_t0 = time.time()
                    adv_action, adv_token_ids, atk_result = run_pgd_attack(img_np, cfg['instruction'], clean_action, clean_gen, args.seed + policy_step)
                    attack_dt = time.time() - attack_t0
                    pgd_applied = True
                    effective_attack_step_idx = ctrl['attacks_applied']
                    raw_action = adv_action; adv_grip = float(raw_action[-1])
                    arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
                    linf = float(atk_result.observation_perturb_linf)
                    token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])
                    # Update attack/controller audit state using causally available qpos.
                    ctrl['attacks_applied'] += 1
                    is_open = adv_grip > 0.5
                    ctrl['current_streak'] = ctrl['current_streak'] + 1 if is_open else 0
                    ctrl['max_streak'] = max(ctrl['max_streak'], ctrl['current_streak'])
                    ctrl['total_open'] += (1 if is_open else 0)
                    if ctrl['attacks_applied'] == 1:
                        ctrl['qpos_start'] = qpos_pre_step
                    ctrl['qpos_delta_online'] = abs(qpos_pre_step - ctrl['qpos_start'])
                    if ctrl['mode'] != 'fixed':
                        # Check stop conditions by mode
                        if ctrl['mode'] == 'min_hold_qpos_cap':
                            if ctrl['attacks_applied'] < ctrl['min_att']:
                                pass  # must continue
                            elif ctrl['Q'] > 0 and ctrl['qpos_delta_online'] >= ctrl['Q']:
                                ctrl['stop_reason'] = 'qpos_threshold'
                            elif ctrl['attacks_applied'] >= ctrl['max_dur']:
                                ctrl['stop_reason'] = 'max_duration'
                        elif ctrl['mode'] == 'open_streak_stop':
                            if ctrl['K'] > 0 and ctrl['current_streak'] >= ctrl['K']:
                                ctrl['stop_reason'] = 'streak_threshold'
                            elif ctrl['attacks_applied'] >= ctrl['max_dur']:
                                ctrl['stop_reason'] = 'max_duration'
                        elif ctrl['mode'] == 'open_count_stop':
                            if ctrl['K'] > 0 and ctrl['total_open'] >= ctrl['K']:
                                ctrl['stop_reason'] = 'open_count_threshold'
                            elif ctrl['attacks_applied'] >= ctrl['max_dur']:
                                ctrl['stop_reason'] = 'max_duration'
                        elif ctrl['Q'] > 0 and ctrl['qpos_delta_online'] >= ctrl['Q']:
                            ctrl['stop_reason'] = 'qpos_threshold'
                        elif ctrl['attacks_applied'] >= ctrl['max_dur']:
                            ctrl['stop_reason'] = 'max_duration'
                    controller_stopped = (ctrl['mode'] != 'fixed' and ctrl['stop_reason'] != 'none')
            except Exception as e:
                print(f'    PGD ERROR at step {policy_step}: {str(e)[:100]}')
                raw_action, _ = decode_image(img_np, cfg['instruction'])
                adv_grip = float(raw_action[-1]); clean_grip = adv_grip

    elif args.condition == 'random_linf':
        clean_action, clean_token_ids = decode_image(img_np, cfg['instruction'])
        clean_action_vec = clean_action.copy()
        img_f = img_np.astype(np.float32) / 255.0
        # Use raw-pixel Linf budget (matched to --eps_raw_pixels), NOT processor-space EPS.
        eps_raw_unit = EPS_RAW_PIXELS / 255.0
        noise = rng.uniform(-eps_raw_unit, eps_raw_unit, img_f.shape).astype(np.float32)
        adv_img_np = (np.clip(img_f + noise, 0.0, 1.0) * 255).astype(np.uint8)
        adv_action, adv_token_ids = decode_image(adv_img_np, cfg['instruction'])
        raw_action = adv_action; adv_grip = float(raw_action[-1]); clean_grip = float(clean_action[-1])
        arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
        linf = float(np.abs(noise).max())
        token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])

    env_action = normalize_gripper_action(raw_action, binarize=True)
    env_action = invert_gripper_action(env_action)
    obs, reward, done, info = env.step(env_action)
    gripper_qpos_post = obs['robot0_gripper_qpos'].copy()
    qpos_post_step = float(gripper_qpos_post[0]) if len(gripper_qpos_post) > 0 else 0.0

    clean_z_val = float(clean_action_vec[2]) if clean_action_vec is not None else 0.0
    adv_z_val = float(raw_action[2])
    nad_dof1_3 = float(np.linalg.norm(raw_action[:3] - clean_action_vec[:3])) if clean_action_vec is not None else 0.0
    trace_rows.append({
        'task': args.task, 'condition': args.condition, 'seed': args.seed,
        'step': t, 'policy_step': policy_step, 'in_window': in_window,
        'attack_attempted': attack_attempted, 'pgd_applied': pgd_applied,
        'controller_active': controller_active, 'controller_stopped': controller_stopped,
        'effective_attack_step_idx': effective_attack_step_idx,
        'raw_gripper': float(raw_action[-1]), 'env_gripper': float(env_action[-1]),
        'gripper_qpos': qpos_pre_step, 'qpos_pre_step': qpos_pre_step, 'qpos_post_step': qpos_post_step,
        'clean_grip': clean_grip, 'adv_grip': adv_grip,
        'clean_z': clean_z_val, 'adv_z': adv_z_val,
        'nad_dof7': abs(float(raw_action[-1]) - clean_grip),
        'nad_z': abs(adv_z_val - clean_z_val),
        'nad_dof1_3': nad_dof1_3,
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

# Post-loop: if controller was active but never stopped, mark force_window_end
if ctrl['mode'] != 'fixed' and ctrl['stop_reason'] == 'none' and ctrl['attacks_applied'] > 0:
    ctrl['stop_reason'] = 'force_window_end'

total_dt = time.time() - t_start
window_rows = [r for r in trace_rows if r['in_window']]
attacked_rows = [r for r in window_rows if r['pgd_applied']]
n_flip = sum(1 for r in window_rows if r['token_flip'])
attacked_flips = sum(1 for r in attacked_rows if r['token_flip'])
open_count_full_window = sum(1 for r in window_rows if r['adv_grip'] > 0.5)
open_count_attacked_steps = sum(1 for r in attacked_rows if r['adv_grip'] > 0.5)
qpos_delta_pre = 0.0
qpos_delta_post = 0.0
if attacked_rows:
    qpos_pre0 = attacked_rows[0]['qpos_pre_step']
    qpos_post0 = attacked_rows[0]['qpos_post_step']
    qpos_delta_pre = max(abs(r['qpos_pre_step'] - qpos_pre0) for r in attacked_rows)
    qpos_delta_post = max(abs(r['qpos_post_step'] - qpos_post0) for r in attacked_rows)
avg_al = np.mean([r['arm_l2'] for r in window_rows]) if window_rows else 0
print(f'[4] Episode finished: success={success}, steps={policy_step}, time={total_dt:.0f}s')
print(f'    Window: {len(window_rows)} steps, attacks={len(attacked_rows)}, flips={n_flip} full / {attacked_flips} attacked, avg armL2={avg_al:.4f}')
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
    'attacks_applied': len(attacked_rows),
    'token_flips_full_window': n_flip, 'token_flips_attacked_steps': attacked_flips,
    'open_count_full_window': open_count_full_window, 'open_count_attacked_steps': open_count_attacked_steps,
    'window_token_flips': n_flip, 'avg_arm_l2': avg_al, 'total_dt_s': round(total_dt, 1),
    'controller': ctrl['mode'], 'K': ctrl['K'], 'Q': ctrl['Q'], 'max_dur': ctrl['max_dur'],
    'stop_reason': ctrl['stop_reason'],
    'max_open_streak': ctrl['max_streak'], 'qpos_delta_online': round(ctrl['qpos_delta_online'], 6),
    'qpos_delta_pre': round(qpos_delta_pre, 6), 'qpos_delta_post': round(qpos_delta_post, 6),
    'official_success': None, 'cq_success': None, 'cq_failure': None, 'sr_cq_mismatch': None,
    'manual_audit_needed': None,
}
print(json.dumps(summary))
print(f'Saved: {csv_path}')
