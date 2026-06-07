#!/usr/bin/env python3
"""Active Probe v0: Lightweight model-level open-margin sensitivity probe.
For each candidate window, sample 3-5 steps and run FGSM1/PGD1/PGD3 probes.
Measures how much gripper open-margin increases under minimal adversarial perturbation.

P0 requirements (2026-06-06 code review):
  - GPU isolation: CUDA_VISIBLE_DEVICES=<physical>, --gpu_pair 0,1
  - NO import of vis_rollout_adaptive_v3 (top-level argparse side effects)
  - Correct LIBERO Object task mapping (matches vis_rollout_adaptive_v3.py)
  - Teacher-forced gripper token gradient objective
  - Raw RGB perturbation space (reprocess after perturbation)
  - All variables defined before use
"""
import csv, os, sys, time, argparse, glob
from datetime import datetime
import numpy as np
import torch

# ── GPU isolation guard ───────────────────────────────────────────
_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if _VISIBLE:
    # CUDA_VISIBLE_DEVICES is set — gpu_pair must be 0,1
    pass  # validated below after argparse
else:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set for GPU isolation.')
    print('Usage: CUDA_VISIBLE_DEVICES=2,6 python run_active_probe_v0.py --gpu_pair 0,1 ...')
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_DIR = '/data/liuyu/outputs/active_probe_v0_20260606'
UNNORM_KEY = 'libero_object'
SEMANTICS_VERSION = 'active_probe_v0_teacher_forced_gripper_20260606'
ACTION_TRANSFORM_VERSION = 'official_normalize_invert_v1'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

# ── CLI ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--shard_csv', required=True)
ap.add_argument('--shard_name', default='shard')
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--probe_steps', type=int, default=3)
args = ap.parse_args()

# GPU isolation hard guard
if _VISIBLE and args.gpu_pair != '0,1':
    log('FATAL: CUDA_VISIBLE_DEVICES=%s requires --gpu_pair 0,1, got %s' % (_VISIBLE, args.gpu_pair))
    sys.exit(1)

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
device_str = 'cuda:%d' % gpu_ids[0]
# Parse physical GPU IDs from CUDA_VISIBLE_DEVICES for EGL rendering
_physical_gpus = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical_gpus[0] if _physical_gpus else gpu_ids[0]
log('GPU: physical=%s, logical=%s, render=%d' % (_VISIBLE, args.gpu_pair, _render_gpu))

# ── Action transform (inline, official v1) ────────────────────────
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

# ── LIBERO Object task mapping (matches vis_rollout_adaptive_v3.py) ──
TASK_CFG = {
    'alphabet_soup':  {'task_id': 0},
    'cream_cheese':   {'task_id': 1},
    'salad_dressing': {'task_id': 2},
    'bbq_sauce':      {'task_id': 3},
    'ketchup':        {'task_id': 4},
    'tomato_sauce':   {'task_id': 5},
    'butter':         {'task_id': 6},
    'milk':           {'task_id': 7},
    'orange_juice':   {'task_id': 9},
}

# ── Load model ────────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor
log('Loading model on %s' % device_str)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '8000MiB', gpu_ids[1]: '8000MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()
model_device = next(model.parameters()).device
log('Model device: %s' % model_device)

action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
low = np.asarray(s['q01'], dtype=np.float32); high = np.asarray(s['q99'], dtype=np.float32)
mask_np = np.asarray(s.get('mask', np.ones_like(low, dtype=bool)), dtype=bool)
mdtype = next(model.parameters()).dtype
log('action_dim=%d vocab=%d dtype=%s' % (action_dim, VS, mdtype))

# ── Helpers ───────────────────────────────────────────────────────
from PIL import Image
from gripper_attack.uncertainty import extract_prefix_logits, softmax_np

def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

def preprocess_raw_rgb(img_rgb, instruction):
    """Preprocess raw RGB uint8 image [H,W,3] for OpenVLA."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    text = prompt_fn(str(instruction).lower())
    inputs = processor(text, pil, return_tensors='pt')
    inputs.pop('attention_mask', None)
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v):
            inputs[k] = v.to(device=model_device, dtype=mdtype)
        else:
            inputs[k] = v.to(model_device)
    if not torch.all(inputs['input_ids'][:, -1] == 29871):
        inputs['input_ids'] = torch.cat((inputs['input_ids'],
            torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)
    return inputs

def decode_action(token_ids):
    """Decode action token IDs to continuous action."""
    tids = token_ids.detach().cpu().numpy() if torch.is_tensor(token_ids) else np.asarray(token_ids)
    disc = np.clip(VS - tids - 1, 0, len(BC)-1)
    na = BC[disc].astype(np.float32)
    return np.where(mask_np, 0.5*(na+1)*(high-low)+low, na).astype(np.float32)

def clean_generate(img_rgb, instruction):
    """Generate clean action tokens and get generation object."""
    inputs = preprocess_raw_rgb(img_rgb, instruction)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
    action = decode_action(token_ids)
    # Extract gripper logits from generation scores
    logits_arr = extract_prefix_logits(gen, action_dim)
    gl = logits_arr[-1].copy() if logits_arr is not None and logits_arr.shape[0] > 0 else None
    return action, token_ids, gl, inputs

def compute_margin(gripper_logits):
    """open_logsumexp - close_logsumexp."""
    if gripper_logits is None: return 0.0
    gl = np.asarray(gripper_logits, dtype=np.float64)
    mid = len(gl)//2
    om = np.max(gl[:mid]) if mid>0 else -np.inf
    cm = np.max(gl[mid:]) if mid<len(gl) else -np.inf
    olse = om + np.log(np.sum(np.exp(gl[:mid]-om))) if mid>0 else -np.inf
    clse = cm + np.log(np.sum(np.exp(gl[mid:]-cm))) if mid<len(gl) else -np.inf
    return float(olse-clse) if np.isfinite(olse) and np.isfinite(clse) else 0.0

def compute_open_mass(gripper_logits):
    if gripper_logits is None: return 0.0
    gp = softmax_np(np.asarray(gripper_logits, dtype=np.float64))
    return float(np.sum(gp[:len(gp)//2]))

# ── Teacher-forced gripper logit objective ────────────────────────
def get_gripper_teacher_forced_logits(input_ids, pixel_values, clean_action_tokens):
    """Teacher-force: forward with prompt + action tokens before gripper dim.
    Returns gripper-dim next-token logits (pre-softmax)."""
    ids = input_ids.clone()
    # Append action tokens except the last (gripper) one
    prefix_tokens = clean_action_tokens[:-1]  # all except gripper
    prefix_t = torch.as_tensor(prefix_tokens, dtype=torch.long, device=ids.device).unsqueeze(0)
    full_ids = torch.cat([ids, prefix_t], dim=1)

    # Note: caller must ensure gradients are enabled (not inside torch.inference_mode)
    out = model(input_ids=full_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
    # Last position logits = next-token prediction for gripper dimension
    gripper_logits = out.logits[0, -1, :]  # [vocab_size]
    return gripper_logits

def open_margin_loss(gripper_logits):
    """Loss = -(open_logsumexp - close_logsumexp). Minimizing = maximizing open margin."""
    mid = gripper_logits.shape[0] // 2
    open_lse = torch.logsumexp(gripper_logits[:mid], dim=0)
    close_lse = torch.logsumexp(gripper_logits[mid:], dim=0)
    return -(open_lse - close_lse)

# ── Processor-space bounds ─────────────────────────────────────────
# Get image normalization params from processor
try:
    IMAGE_MEAN = np.asarray(processor.image_processor.image_mean, dtype=np.float32)
    IMAGE_STD = np.asarray(processor.image_processor.image_std, dtype=np.float32)
except:
    IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
log('Image norm: mean=%s std=%s' % (IMAGE_MEAN, IMAGE_STD))

# ── Probe: FGSM1 / PGD1 / PGD3 (processor-space differentiable) ──
def run_probe_steps(pixel_values, input_ids, clean_action_tokens, eps_raw, n_steps, probe_name):
    """Run PGD-n probe in processor-normalized pixel_values space.
    Gradient chain is continuous: pv -> model -> logits -> margin -> grad.
    No RGB/numpy/PIL conversions inside the PGD loop.

    Returns dict with probe_status, diagnostics, and decoded results.
    """
    result = {
        'probe_status': 'ok', 'probe_error': '',
        'grad_norm': 0.0, 'grad_abs_mean': 0.0,
        'clean_margin_tf': 0.0, 'adv_margin_tf': 0.0, 'margin_gain_tf': 0.0,
        'clean_margin_generate': 0.0, 'adv_margin_generate': 0.0,
        'token_flip': 0, 'open_mass_gain': 0.0,
    }

    n_channels = pixel_values.shape[1]  # actual number of channels (may be 3 or 6)
    pv_orig = pixel_values.clone().detach()
    ids = input_ids.clone().detach()

    # Free cached memory from generate() before enabling gradients
    torch.cuda.empty_cache()

    # Processor-space epsilon per channel — broadcast to [1, n_channels, 1, 1]
    _mean = np.tile(IMAGE_MEAN, n_channels // len(IMAGE_MEAN))[:n_channels] if n_channels > len(IMAGE_MEAN) else IMAGE_MEAN[:n_channels]
    _std = np.tile(IMAGE_STD, n_channels // len(IMAGE_STD))[:n_channels] if n_channels > len(IMAGE_STD) else IMAGE_STD[:n_channels]
    _eps = eps_raw / (255.0 * _std)
    _lo = (0.0 - _mean) / _std
    _hi = (1.0 - _mean) / _std
    eps_ch = torch.as_tensor(_eps, dtype=mdtype, device=model_device).view(1, n_channels, 1, 1)
    lower_ch = torch.as_tensor(_lo, dtype=mdtype, device=model_device).view(1, n_channels, 1, 1)
    upper_ch = torch.as_tensor(_hi, dtype=mdtype, device=model_device).view(1, n_channels, 1, 1)
    step_size = eps_ch / max(n_steps, 1) * 1.5

    # Random start within eps ball
    pv_adv = pv_orig + (torch.rand_like(pv_orig) * 2 - 1) * eps_ch * 0.01
    pv_adv = torch.clamp(pv_adv, lower_ch, upper_ch)

    # Get clean margin via teacher-force (before perturbation)
    with torch.no_grad():
        clean_gl_tf = get_gripper_teacher_forced_logits(ids, pv_orig, clean_action_tokens)
        result['clean_margin_tf'] = float(compute_margin(clean_gl_tf.cpu().numpy()))

    # PGD loop
    final_grad_norm = 0.0
    for step_i in range(n_steps):
        pv_adv = pv_adv.clone().detach().requires_grad_(True)

        try:
            gripper_logits = get_gripper_teacher_forced_logits(ids, pv_adv, clean_action_tokens)
            # Margin = open_logsumexp - close_logsumexp (higher = more OPEN)
            mid = gripper_logits.shape[0] // 2
            margin = torch.logsumexp(gripper_logits[:mid], dim=0) - torch.logsumexp(gripper_logits[mid:], dim=0)
            margin.backward()

            grad = pv_adv.grad
            if grad is None:
                result['probe_status'] = 'grad_missing'
                result['probe_error'] = 'pv_adv.grad is None at step %d' % step_i
                return result

            final_grad_norm = float(grad.norm().item())
            final_grad_abs = float(grad.abs().mean().item())

            with torch.no_grad():
                # Maximize margin: move in +grad direction
                pv_new = pv_adv + step_size * torch.sign(grad)
                eta = torch.clamp(pv_new - pv_orig, -eps_ch, eps_ch)
                pv_adv = torch.clamp(pv_orig + eta, lower_ch, upper_ch)

        except Exception as e:
            result['probe_status'] = 'probe_error'
            result['probe_error'] = '%s at step %d: %s' % (probe_name, step_i, str(e)[:100])
            return result

    result['grad_norm'] = round(final_grad_norm, 6)
    result['grad_abs_mean'] = round(final_grad_abs if final_grad_norm > 0 else 0.0, 8)

    # Get teacher-forced margin after perturbation
    with torch.no_grad():
        adv_gl_tf = get_gripper_teacher_forced_logits(ids, pv_adv.detach(), clean_action_tokens)
        result['adv_margin_tf'] = float(compute_margin(adv_gl_tf.cpu().numpy()))
        result['margin_gain_tf'] = round(result['adv_margin_tf'] - result['clean_margin_tf'], 6)

    # Decode from perturbed pixel_values via generate (full autoregressive)
    with torch.inference_mode():
        gen_adv = model.generate(input_ids=ids, pixel_values=pv_adv.detach(),
                                 max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        gen_clean = model.generate(input_ids=ids, pixel_values=pv_orig,
                                   max_new_tokens=action_dim, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True)

    adv_tids = gen_adv.sequences[0, -action_dim:].cpu().numpy()
    clean_tids_full = gen_clean.sequences[0, -action_dim:].cpu().numpy()

    # Token flip
    result['token_flip'] = int(clean_tids_full[-1]) != int(adv_tids[-1])

    # Margin from generated tokens
    logits_adv = extract_prefix_logits(gen_adv, action_dim)
    logits_clean = extract_prefix_logits(gen_clean, action_dim)
    gl_adv = logits_adv[-1].copy() if logits_adv is not None and logits_adv.shape[0] > 0 else None
    gl_clean = logits_clean[-1].copy() if logits_clean is not None and logits_clean.shape[0] > 0 else None

    result['clean_margin_generate'] = round(compute_margin(gl_clean), 6)
    result['adv_margin_generate'] = round(compute_margin(gl_adv), 6)
    result['open_mass_gain'] = round(compute_open_mass(gl_adv) - compute_open_mass(gl_clean), 8)

    return result

# ── Load candidates ───────────────────────────────────────────────
with open(args.shard_csv) as f:
    candidates = list(csv.DictReader(f))
log('Loaded %d candidates' % len(candidates))

# ── Probe each candidate ──────────────────────────────────────────
step_rows = []
window_rows = []
EPS_RAW = args.eps_raw_pixels

for idx, c in enumerate(candidates):
    task = c['task_key'].strip()
    sid = int(c['state_id'])
    ws = int(c['window_start'])
    we = int(c['window_end'])
    label = c.get('label_status','?')
    tax = c.get('taxonomy','?')

    log('[%d/%d] %s s%d [%d,%d] %s' % (idx+1, len(candidates), task, sid, ws, we, label))

    # ── Resolve task and create env ──────────────────────────────
    cfg = TASK_CFG.get(task)
    if cfg is None:
        log('  SKIP: unknown task %s' % task)
        continue

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict['libero_object']()
        task_obj = task_suite.get_task(cfg['task_id'])
        initial_states = task_suite.get_task_init_states(cfg['task_id'])
        if int(sid) >= len(initial_states):
            log('  SKIP: state_id %d OOB (max %d)' % (sid, len(initial_states)-1))
            continue

        # Get official instruction from task object
        instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else task.replace('_',' ')
        log('  instruction: %s' % instruction)

        bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
        env_args = {
            'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
            'has_renderer': False, 'has_offscreen_renderer': True,
            'use_camera_obs': True, 'camera_names': ['agentview'], 'control_freq': 20,
            'render_gpu_device_id': _render_gpu,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0); obs = env.reset()
        env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(initial_states[int(sid)])
    except Exception as e:
        log('  SKIP: env error %s' % str(e)[:100])
        continue

    # ── Step to window_start ─────────────────────────────────────
    current_step = 0
    done = False
    while not done and current_step < ws:
        img = obs['agentview_image']
        action, token_ids, _, inputs = clean_generate(img, instruction)
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    # ── Probe at sampled steps ───────────────────────────────────
    window_len = we - ws + 1
    n_probe = min(args.probe_steps, window_len)
    if n_probe <= 1:
        probe_steps = [ws]
    else:
        probe_steps = sorted(set([ws, (ws+we)//2, we]))[:n_probe]
    log('  probing at steps: %s (window [%d,%d])' % (probe_steps, ws, we))

    step_probe_results = []
    while not done and current_step <= max(probe_steps) + 1:
        img = obs['agentview_image']

        if current_step in probe_steps:
            # Clean decode — get pixel_values and input_ids for processor-space PGD
            clean_action, clean_tids, clean_gl, inputs = clean_generate(img, instruction)
            clean_margin = compute_margin(clean_gl)
            clean_open_mass = compute_open_mass(clean_gl)
            pv_clean = inputs['pixel_values']
            ids_clean = inputs['input_ids']

            probe_results = {}
            for pname, psteps in [('FGSM1',1),('PGD1',1),('PGD3',3)]:
                res = run_probe_steps(pv_clean, ids_clean, clean_tids, EPS_RAW, psteps, pname)
                probe_results[pname] = res

            # Extract diagnostics
            max_margin_gain_tf = max(probe_results[p]['margin_gain_tf'] for p in probe_results)
            max_grad_norm = max(probe_results[p]['grad_norm'] for p in probe_results)
            any_flip = any(probe_results[p]['token_flip'] for p in probe_results)
            any_error = any(probe_results[p]['probe_status'] != 'ok' for p in probe_results)

            step_probe_results.append({
                'step': current_step,
                'clean_margin_generate': clean_margin,
                'clean_open_mass': clean_open_mass,
                # FGSM1
                'fgsm1_grad_norm': probe_results['FGSM1']['grad_norm'],
                'fgsm1_margin_gain_tf': probe_results['FGSM1']['margin_gain_tf'],
                'fgsm1_adv_margin_generate': probe_results['FGSM1']['adv_margin_generate'],
                'fgsm1_open_mass_gain': probe_results['FGSM1']['open_mass_gain'],
                'fgsm1_token_flip': probe_results['FGSM1']['token_flip'],
                'fgsm1_probe_status': probe_results['FGSM1']['probe_status'],
                'fgsm1_probe_error': probe_results['FGSM1']['probe_error'],
                # PGD1
                'pgd1_grad_norm': probe_results['PGD1']['grad_norm'],
                'pgd1_margin_gain_tf': probe_results['PGD1']['margin_gain_tf'],
                'pgd1_adv_margin_generate': probe_results['PGD1']['adv_margin_generate'],
                'pgd1_open_mass_gain': probe_results['PGD1']['open_mass_gain'],
                'pgd1_token_flip': probe_results['PGD1']['token_flip'],
                'pgd1_probe_status': probe_results['PGD1']['probe_status'],
                'pgd1_probe_error': probe_results['PGD1']['probe_error'],
                # PGD3
                'pgd3_grad_norm': probe_results['PGD3']['grad_norm'],
                'pgd3_margin_gain_tf': probe_results['PGD3']['margin_gain_tf'],
                'pgd3_adv_margin_generate': probe_results['PGD3']['adv_margin_generate'],
                'pgd3_open_mass_gain': probe_results['PGD3']['open_mass_gain'],
                'pgd3_token_flip': probe_results['PGD3']['token_flip'],
                'pgd3_probe_status': probe_results['PGD3']['probe_status'],
                'pgd3_probe_error': probe_results['PGD3']['probe_error'],
                # Aggregates
                'max_margin_gain_tf': max_margin_gain_tf,
                'max_grad_norm': max_grad_norm,
                'any_token_flip': int(any_flip),
                'any_probe_error': int(any_error),
            })

        # Step env
        action, _, _, inputs = clean_generate(img, instruction)
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    env.close()

    if not step_probe_results:
        step_rows.append({
            'task_key': task, 'state_id': str(sid),
            'window_start': str(ws), 'window_end': str(we),
            'label_status': label, 'taxonomy': tax, 'step': str(-1),
            'probe_status': 'no_probe_steps',
        })
        continue

    # Per-step rows
    for sr in step_probe_results:
        step_rows.append({
            'task_key': task, 'state_id': str(sid),
            'window_start': str(ws), 'window_end': str(we),
            'label_status': label, 'taxonomy': tax,
            'probe_status': 'ok',
            **{k: str(v) for k, v in sr.items()},
        })

    # Window aggregate
    gains_fgsm_tf = [s['fgsm1_margin_gain_tf'] for s in step_probe_results]
    gains_pgd1_tf = [s['pgd1_margin_gain_tf'] for s in step_probe_results]
    gains_pgd3_tf = [s['pgd3_margin_gain_tf'] for s in step_probe_results]
    any_flip_window = any(s['any_token_flip'] for s in step_probe_results)
    any_error_window = any(s['any_probe_error'] for s in step_probe_results)
    clean_margins = [s['clean_margin_generate'] for s in step_probe_results]
    max_gain_any = max(max(gains_fgsm_tf), max(gains_pgd1_tf), max(gains_pgd3_tf))
    grad_norms = [s['max_grad_norm'] for s in step_probe_results]
    max_grad = max(grad_norms) if grad_norms else 0.0

    window_rows.append({
        'task_key': task, 'state_id': str(sid),
        'window_start': str(ws), 'window_end': str(we),
        'label_status': label, 'taxonomy': tax,
        'n_probe_steps': str(len(step_probe_results)),
        'clean_margin_mean': str(round(np.mean(clean_margins), 4)),
        'fgsm1_margin_gain_tf_max': str(round(max(gains_fgsm_tf), 4)),
        'fgsm1_margin_gain_tf_mean': str(round(np.mean(gains_fgsm_tf), 4)),
        'pgd1_margin_gain_tf_max': str(round(max(gains_pgd1_tf), 4)),
        'pgd1_margin_gain_tf_mean': str(round(np.mean(gains_pgd1_tf), 4)),
        'pgd3_margin_gain_tf_max': str(round(max(gains_pgd3_tf), 4)),
        'pgd3_margin_gain_tf_mean': str(round(np.mean(gains_pgd3_tf), 4)),
        'max_margin_gain_tf_any': str(round(max_gain_any, 4)),
        'max_grad_norm': str(round(max_grad, 4)),
        'any_token_flip': str(int(any_flip_window)),
        'any_probe_error': str(int(any_error_window)),
    })

    log('  tf_gains: FGSM1=%.4f PGD1=%.4f PGD3=%.4f max=%.4f grad=%.4f flip=%d err=%d' % (
        max(gains_fgsm_tf), max(gains_pgd1_tf), max(gains_pgd3_tf),
        max_gain_any, max_grad, int(any_flip_window), int(any_error_window)))

# ── Write outputs ─────────────────────────────────────────────────
step_csv = os.path.join(REPO, 'tables/active_probe_v0_step_features_%s.csv' % args.shard_name)
window_csv = os.path.join(REPO, 'tables/active_probe_v0_window_features_%s.csv' % args.shard_name)

if step_rows:
    with open(step_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        w.writeheader(); w.writerows(step_rows)
    log('Wrote %d step rows to %s' % (len(step_rows), step_csv))

if window_rows:
    with open(window_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader(); w.writerows(window_rows)
    log('Wrote %d window rows to %s' % (len(window_rows), window_csv))

log('=== Active Probe v0 %s complete: %d candidates, %d windows ===' % (
    args.shard_name, len(window_rows), len(window_rows)))
