#!/usr/bin/env python3
"""Active Probe v1: Temporal decoded gripper susceptibility probe.

Core difference from v0b:
- Decodes FULL action under probe perturbation, extracts gripper command
- Computes open_count, open_rate, longest_open_streak per condition
- Targeted vs random contrast for selectivity
- Separates command_susceptible label from physical_bridge label

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_active_probe_v1_temporal.py \
    --gpu_pair 0,1 --mode pilot --output_name pilot_v1
"""
import csv, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'
UNNORM_KEY = 'libero_object'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--mode', default='pilot', choices=['pilot', 'full31'])
ap.add_argument('--output_name', default='v1')
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--pgd_steps', type=int, default=3)
ap.add_argument('--n_probe_frames', type=int, default=10)
ap.add_argument('--shard', type=int, default=-1, help='0-indexed shard; -1 = all')
ap.add_argument('--shard_total', type=int, default=3)
args = ap.parse_args()

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]
log('GPU: physical=%s logical=%s render=%d' % (_VISIBLE, args.gpu_pair, _render_gpu))

# ── PILOT WINDOWS ───────────────────────────────────────────────
PILOT_WINDOWS = [
    # 4 physical_bridge positives
    ('ketchup', 0, 16, 33, 'positive', 'action_positive_physical_strong'),
    ('butter', 0, 29, 46, 'positive', 'action_positive_physical_strong'),
    ('alphabet_soup', 4, 4, 21, 'positive', 'claim_usable'),
    ('cream_cheese', 4, 28, 45, 'positive', 'claim_usable'),
    # 4 command-level negatives (vis_open=0, phys=0)
    ('alphabet_soup', 3, 21, 38, 'ignore', 'polluted_neg'),
    ('bbq_sauce', 0, 30, 47, 'ignore', 'polluted_neg'),
    ('ketchup', 4, 28, 45, 'negative', 'no_action_bridge'),
    ('milk', 8, 8, 25, 'negative', 'no_action_bridge'),
    # 4 clean/diverse negatives
    ('salad_dressing', 0, 31, 48, 'ignore', 'polluted_ctrl'),
    ('tomato_sauce', 3, 17, 34, 'negative', 'no_action_bridge_ctrl'),
    ('alphabet_soup', 8, 29, 46, 'ignore', 'polluted_ctrl2'),
    ('butter', 3, 29, 46, 'ignore', 'polluted_ctrl3'),
]

# ── Load model ──────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result

log('Loading model...')
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))
log('Model loaded, action_dim=%d' % action_dim)

# ── Action decode stats ─────────────────────────────────────────
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32)
HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

def decode_tokens_to_action(tids_1d):
    """Convert action token IDs to continuous 7-dim action (unnormalized)."""
    tids = np.asarray(tids_1d, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP) - 1)
    action = np.where(MK, 0.5 * (BC_NP[disc] + 1) * (HI - LO) + LO, BC_NP[disc])
    return action.astype(np.float32)

def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action

def get_env_gripper(action):
    """Full chain: unnormalized action → normalize → invert → gripper.
    Returns +1.0 = OPEN, -1.0 = CLOSE."""
    a = normalize_gripper_action(action.copy(), binarize=True)
    a = invert_gripper_action(a)
    return float(a[-1])

# ── Create attacker ──────────────────────────────────────────────
_eps_eff = args.eps_raw_pixels / 255.0
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
    model=model, processor=processor,
    config=attacker_config,
    seed=0, device='cuda:%d' % gpu_ids[0],
    preprocess_kwargs={'postprocess_gripper': True},
)
attacker._freeze_model()
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
log('Attacker ready on %s' % model_device)

# ── Prompt ───────────────────────────────────────────────────────
def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

# ── Core probe: decode gripper under perturbation ───────────────
def decode_gripper_from_inputs(inp):
    """Run model.generate with given inputs, return env_gripper (+1=OPEN, -1=CLOSE)."""
    with torch.inference_mode():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    tids = gen.sequences[0, -action_dim:].cpu().numpy()
    action = decode_tokens_to_action(tids)
    return get_env_gripper(action)

def random_linf_perturb(pv, epsilon):
    """Apply random Linf perturbation to pixel_values."""
    noise = (2 * torch.rand_like(pv) - 1) * epsilon
    pv_perturbed = pv + noise
    # Clamp to Linf ball around original
    pv_perturbed = torch.clamp(pv_perturbed, pv - epsilon, pv + epsilon)
    return pv_perturbed

def run_probe_at_frame(pil_image, instruction, eps_eff):
    """Run clean, targeted PGD, and random Linf decode at one frame.
    Returns dict with per-condition gripper values."""
    text = prompt_fn(instruction.lower())
    inputs = processor(text, pil_image, return_tensors='pt')
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v):
            inputs[k] = v.to(device=model_device, dtype=model_dtype)
        else:
            inputs[k] = v.to(model_device)
    # Ensure EOS token
    if not torch.all(inputs['input_ids'][:, -1] == 29871):
        inputs['input_ids'] = torch.cat((inputs['input_ids'],
            torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)

    clean_pv = inputs['pixel_values'].clone()
    clean_ids = inputs['input_ids'].clone()

    # 1) Clean decode
    clean_grip = decode_gripper_from_inputs({'input_ids': clean_ids, 'pixel_values': clean_pv})

    # 2) Targeted PGD decode
    targeted_grip = None; targeted_ok = False; targeted_err = ''
    try:
        # Decode clean action for target (needed by prefix_locked objective)
        with torch.inference_mode():
            gen = model.generate(input_ids=clean_ids, pixel_values=clean_pv,
                                max_new_tokens=action_dim, do_sample=False,
                                return_dict_in_generate=True, output_scores=False)
        clean_tids = gen.sequences[0, -action_dim:].cpu().numpy()
        clean_action = decode_tokens_to_action(clean_tids)

        result = attacker.attack(
            observation=pil_image,
            instruction=instruction.lower(),
            target_action=clean_action,
            unnorm_key=UNNORM_KEY,
        )
        adv_inputs = get_adv_inputs_from_attack_result(result)
        adv_pv = adv_inputs['pixel_values'].to(device=model_device, dtype=model_dtype)
        adv_ids = adv_inputs['input_ids'].to(device=model_device)
        targeted_grip = decode_gripper_from_inputs({'input_ids': adv_ids, 'pixel_values': adv_pv})
        targeted_ok = True
    except Exception as e:
        targeted_err = str(e)[:120]
        targeted_grip = float('nan')

    # 3) Random Linf decode
    rand_pv = random_linf_perturb(clean_pv.clone(), eps_eff)
    rand_grip = decode_gripper_from_inputs({'input_ids': clean_ids, 'pixel_values': rand_pv})

    return {
        'clean_grip': clean_grip,
        'targeted_grip': targeted_grip,
        'targeted_ok': targeted_ok,
        'targeted_err': targeted_err,
        'random_grip': rand_grip,
    }

from gripper_attack.gripper_semantics import env_gripper_is_open

# ── Compute streaks from a sequence of gripper values ────────────
def compute_gripper_stats(grip_values):
    """Given list of env gripper values (-1=OPEN, +1=CLOSE), compute stats."""
    opens = [1 if env_gripper_is_open(g) else 0 for g in grip_values]
    open_count = sum(opens)
    open_rate = open_count / max(len(opens), 1)
    streak = 0; max_streak = 0
    for o in opens:
        if o: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    return {
        'open_count': open_count,
        'open_rate': round(open_rate, 4),
        'longest_open_streak': max_streak,
    }

# ── TASK CFG ─────────────────────────────────────────────────────
TASK_CFG = {
    'ketchup': {'task_id': 0}, 'butter': {'task_id': 1},
    'cream_cheese': {'task_id': 2}, 'salad_dressing': {'task_id': 3},
    'bbq_sauce': {'task_id': 4}, 'milk': {'task_id': 5},
    'alphabet_soup': {'task_id': 6}, 'tomato_sauce': {'task_id': 7},
    'orange_juice': {'task_id': 8},
}

# ── Process candidates ───────────────────────────────────────────
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Load candidates based on mode
if args.mode == 'pilot':
    candidates = PILOT_WINDOWS
else:
    with open(SHARED + '/object_phase_response_labels_v2.csv') as f:
        candidates = [(r['task_key'].strip(), int(r['state_id']),
                       int(r['window_start']), int(r['window_end']),
                       r['label_status'].strip(), r['taxonomy'].strip())
                      for r in csv.DictReader(f)]

log('Processing %d candidates (mode=%s)' % (len(candidates), args.mode))

# Shard filtering
if args.shard >= 0:
    total = args.shard_total
    shard_size = (len(candidates) + total - 1) // total
    start = args.shard * shard_size
    end = min(start + shard_size, len(candidates))
    candidates = candidates[start:end]
    log('Shard %d/%d: range [%d:%d] = %d windows' % (args.shard, total, start, end, len(candidates)))

def make_inputs(pil_image, instruction_text):
    """Create model inputs dict from PIL image and instruction."""
    text = prompt_fn(instruction_text.lower())
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

window_rows = []
step_rows = []

for idx, c in enumerate(candidates):
    task, sid, ws, we, label_status, taxonomy = c
    label = label_status
    log('[%d/%d] %s s%d [%d,%d] %s' % (idx+1, len(candidates), task, sid, ws, we, label))

    cfg = TASK_CFG.get(task)
    if cfg is None: log('  SKIP: unknown task'); continue

    try:
        bm_dict = benchmark.get_benchmark_dict()
        task_suite = bm_dict['libero_object']()
        task_obj = task_suite.get_task(cfg['task_id'])
        initial_states = task_suite.get_task_init_states(cfg['task_id'])
        if int(sid) >= len(initial_states):
            log('  SKIP: state OOB'); continue
        instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else task.replace('_', ' ')
        bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 has_renderer=False, has_offscreen_renderer=True,
                                 use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                                 render_gpu_device_id=_render_gpu)
        env.seed(0); obs = env.reset()
        env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(initial_states[int(sid)])
    except Exception as e:
        log('  SKIP: env error %s' % str(e)[:80]); continue

    # Step to window_start
    current_step = 0; done = False
    while not done and current_step < ws:
        img = obs['agentview_image']
        pil = Image.fromarray(img.astype(np.uint8))
        inputs = make_inputs(pil, instruction)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=False)
        tids = gen.sequences[0, -action_dim:].cpu().numpy()
        action = decode_tokens_to_action(tids)
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    # Determine probe frames: evenly spaced across window
    window_len = we - ws + 1
    n_probe = min(args.n_probe_frames, window_len)
    if window_len <= n_probe:
        probe_frames = list(range(ws, we + 1))
    else:
        probe_frames = sorted(set([ws + int(i * (window_len - 1) / (n_probe - 1)) for i in range(n_probe)]))
    log('  probe frames: %s' % probe_frames[:5] + ('...' if len(probe_frames) > 5 else ''))

    # Collect per-frame probe results
    clean_grips = []; targeted_grips = []; random_grips = []
    frame_details = []

    frame_idx = 0
    while not done and current_step <= we + 1:
        img = obs['agentview_image']
        pil = Image.fromarray(img.astype(np.uint8))
        inputs = make_inputs(pil, instruction)

        if current_step in probe_frames:
            pr = run_probe_at_frame(pil, instruction, _eps_eff)

            clean_grips.append(pr['clean_grip'])
            targeted_grips.append(pr['targeted_grip'] if pr['targeted_ok'] else float('nan'))
            random_grips.append(pr['random_grip'])

            frame_details.append({
                'step': str(current_step),
                'clean_grip': str(pr['clean_grip']),
                'targeted_grip': str(pr['targeted_grip']) if pr['targeted_ok'] else 'nan',
                'targeted_ok': str(int(pr['targeted_ok'])),
                'targeted_err': pr['targeted_err'],
                'random_grip': str(pr['random_grip']),
            })
            frame_idx += 1

        # Step env with clean action to advance to next frame
        if current_step <= we:
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                     return_dict_in_generate=True, output_scores=False)
            tids = gen.sequences[0, -action_dim:].cpu().numpy()
            action = decode_tokens_to_action(tids)
            env_action = normalize_gripper_action(action.copy(), binarize=True)
            env_action = invert_gripper_action(env_action)
            obs, reward, done, info = env.step(env_action)
        current_step += 1

    env.close()

    if not clean_grips:
        log('  SKIP: no probe data'); continue

    # Compute stats
    clean_stats = compute_gripper_stats(clean_grips)
    targeted_valid = [g for g in targeted_grips if not np.isnan(g)]
    targeted_stats = compute_gripper_stats(targeted_valid) if targeted_valid else {'open_count': 0, 'open_rate': 0, 'longest_open_streak': 0}
    random_stats = compute_gripper_stats(random_grips)

    targeted_minus_random_open = targeted_stats['open_count'] - random_stats['open_count']
    targeted_minus_random_streak = targeted_stats['longest_open_streak'] - random_stats['longest_open_streak']

    # gripper_selectivity: how much gripper changes vs arm stability
    # Use std of targeted grippers vs clean as proxy
    clean_open_rate = clean_stats['open_rate']
    targeted_open_rate = targeted_stats['open_rate']
    gripper_delta = targeted_open_rate - clean_open_rate

    wr = {
        'task_key': task, 'state_id': str(sid),
        'window_start': str(ws), 'window_end': str(we),
        'label_status': label_status, 'taxonomy': taxonomy,
        'n_probe_frames': str(len(clean_grips)),
        # Clean baseline
        'clean_open_count': str(clean_stats['open_count']),
        'clean_open_rate': str(clean_stats['open_rate']),
        'clean_longest_open_streak': str(clean_stats['longest_open_streak']),
        # Targeted probe
        'targeted_open_count': str(targeted_stats['open_count']),
        'targeted_open_rate': str(targeted_stats['open_rate']),
        'targeted_longest_open_streak': str(targeted_stats['longest_open_streak']),
        # Random baseline
        'random_open_count': str(random_stats['open_count']),
        'random_open_rate': str(random_stats['open_rate']),
        'random_longest_open_streak': str(random_stats['longest_open_streak']),
        # Contrasts
        'targeted_minus_random_open_count': str(targeted_minus_random_open),
        'targeted_minus_random_streak': str(targeted_minus_random_streak),
        'gripper_delta_vs_clean': str(round(gripper_delta, 4)),
        # Probe errors
        'targeted_error_rate': str(round(sum(1 for g in targeted_grips if np.isnan(g)) / max(len(targeted_grips), 1), 4)),
    }
    window_rows.append(wr)

    # Write per-frame rows
    for fd in frame_details:
        fd['task_key'] = task; fd['state_id'] = str(sid)
        fd['window_start'] = str(ws); fd['window_end'] = str(we)
        fd['label_status'] = label_status; fd['taxonomy'] = taxonomy
        step_rows.append(fd)

    log('  clean: open=%d rate=%.2f streak=%d | targeted: open=%d rate=%.2f streak=%d | random: open=%d streak=%d | t-r: cnt=%d streak=%d' % (
        clean_stats['open_count'], clean_stats['open_rate'], clean_stats['longest_open_streak'],
        targeted_stats['open_count'], targeted_stats['open_rate'], targeted_stats['longest_open_streak'],
        random_stats['open_count'], random_stats['longest_open_streak'],
        targeted_minus_random_open, targeted_minus_random_streak))

# ── Write ────────────────────────────────────────────────────────
tag = args.output_name
if window_rows:
    wf = OUT_DIR + '/window_features_%s.csv' % tag
    with open(wf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader(); w.writerows(window_rows)
    log('Wrote %d windows to %s' % (len(window_rows), wf))

if step_rows:
    sf = OUT_DIR + '/step_features_%s.csv' % tag
    with open(sf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        w.writeheader(); w.writerows(step_rows)
    log('Wrote %d steps to %s' % (len(step_rows), sf))

log('DONE')
