#!/usr/bin/env python3
"""Active Probe v0b: Reuse TokenPrefixPGDAttacker core for correct open/close tokens,
processor-space PGD, model freezing, and budget projection.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_active_probe_v0b.py \
    --gpu_pair 0,1 --shard_csv tables/active_probe_v0_smoke.csv --shard_name smoke_v0b
"""
import csv, os, sys, time, argparse, glob
from datetime import datetime
import numpy as np
import torch

# ── GPU isolation ─────────────────────────────────────────────────
_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_DIR = '/data/liuyu/outputs/active_probe_v0b_20260606'
UNNORM_KEY = 'libero_object'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

# ── CLI ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--shard_csv', required=True)
ap.add_argument('--shard_name', default='shard')
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--probe_steps', type=int, default=3)
ap.add_argument('--pgd_steps', type=int, default=3)
args = ap.parse_args()

if _VISIBLE and args.gpu_pair != '0,1':
    log('FATAL: CUDA_VISIBLE_DEVICES=%s requires --gpu_pair 0,1' % _VISIBLE); sys.exit(1)

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]
log('GPU: physical=%s logical=%s render=%d' % (_VISIBLE, args.gpu_pair, _render_gpu))

# ── Load model & create attacker ──────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, CANONICAL_OPEN_SEMANTICS_VERSION

log('Loading model...')
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))

# Create attacker (reuses its freeze, region, PGD machinery)
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
# Manually freeze (attacker.attack does this but we access internals)
attacker._freeze_model()
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
log('Model device: %s, action_dim=%d' % (model_device, action_dim))

# Get correct open/close token regions
region = attacker.get_gripper_region_by_decoded_action(UNNORM_KEY, postprocess_gripper=True)
open_token_ids = region['open_token_ids']
close_token_ids = region['close_token_ids']
log('Open tokens: %d, Close tokens: %d, semantics: %s' % (
    region['open_count'], region['close_count'], region['canonical_semantics_version']))

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

# ── Standardized scoring functions ────────────────────────────────
def compute_open_score_from_logits(gripper_row_logits):
    """open_score = logsumexp(open_tokens) - max(non_open_tokens)."""
    open_lse = torch.logsumexp(gripper_row_logits[open_token_ids], dim=0)
    non_open_mask = torch.ones_like(gripper_row_logits, dtype=torch.bool)
    non_open_mask[open_token_ids] = False
    max_non_open = gripper_row_logits[non_open_mask].max()
    return open_lse - max_non_open

def compute_loss_score_from_logits(gripper_row_logits, margin=5.0):
    """loss_score = ReLU(max_non_open - logsumexp(open) + margin)."""
    open_lse = torch.logsumexp(gripper_row_logits[open_token_ids], dim=0)
    non_open_mask = torch.ones_like(gripper_row_logits, dtype=torch.bool)
    non_open_mask[open_token_ids] = False
    max_non_open = gripper_row_logits[non_open_mask].max()
    return torch.relu(max_non_open - open_lse + margin)

def get_gripper_row_from_full_ids(full_ids, pixel_values):
    """Run forward pass and return gripper-dim next-token logits."""
    with torch.no_grad():
        out = model(input_ids=full_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
    return out.logits[0, -1, :]  # [vocab_size]

# ── Probe function (reuses attacker internals) ────────────────────
def run_probe_v0b(pixel_values, input_ids, clean_action_tokens, n_steps, probe_name):
    """Run PGD-n probe using TokenPrefixPGDAttacker machinery.
    All in processor-space. No raw RGB conversion. No parameter gradients."""
    result = {
        'probe_status': 'ok', 'probe_error': '',
        'grad_norm': 0.0, 'grad_abs_mean': 0.0,
        'clean_open_score': 0.0, 'adv_open_score': 0.0, 'open_score_gain': 0.0,
        'clean_loss_score': 0.0, 'adv_loss_score': 0.0, 'loss_delta': 0.0,
        'clean_open_prob_mass': 0.0, 'adv_open_prob_mass': 0.0, 'open_prob_gain': 0.0,
        'token_flip': 0,
        'open_token_count': int(open_token_ids.numel()),
        'close_token_count': int(close_token_ids.numel()),
        'semantics_version': CANONICAL_OPEN_SEMANTICS_VERSION,
        'attacker_core_version': 'TokenPrefixPGDAttacker_v0b',
        'probe_direction': 'descent_loss',
    }

    # Freeze model (idempotent)
    attacker._freeze_model()

    pv_orig = pixel_values.clone().detach()
    ids = input_ids.clone().detach()
    model_dtype = next(model.parameters()).dtype

    # Construct labels for prefix-locked loss
    # target_ids: the CLEAN action tokens teach the model what to generate
    target_ids = torch.as_tensor(np.asarray(clean_action_tokens), dtype=torch.long, device=model_device)
    prompt_len = ids.shape[1]
    # full_ids = prompt + target action tokens
    full_ids = torch.cat([ids, target_ids.unsqueeze(0)], dim=1)
    # labels: prompt masked, action tokens as targets
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    x_orig = pv_orig.to(dtype=model_dtype)
    x_orig_model = x_orig.clone()

    # Compute clean scores using standardized functions
    with torch.no_grad():
        _loss_kwargs = {
            'objective': 'prefix_locked_gripper_open_margin',
            'num_action_tokens': int(target_ids.numel()),
            'region_token_ids': open_token_ids,
        }
        clean_gripper_row = get_gripper_row_from_full_ids(full_ids, x_orig_model)
        result['clean_open_score'] = round(float(compute_open_score_from_logits(clean_gripper_row).item()), 6)
        result['clean_loss_score'] = round(float(compute_loss_score_from_logits(clean_gripper_row).item()), 6)

        clean_audit = attacker._audit_logits(full_ids, labels, x_orig_model, target_ids, UNNORM_KEY,
                                              postprocess_gripper=True, region_token_ids=open_token_ids)
        result['clean_open_prob_mass'] = float(clean_audit.get('open_prob_mass', 0.0))

    # PGD: maximize margin (minimize loss = maximize open margin)
    eps_eff = attacker.epsilon
    step_sz = attacker.step_size
    if n_steps <= 1:
        step_sz = eps_eff  # FGSM-like single step

    # Random start within eps ball
    gen = torch.Generator(device=x_orig.device); gen.manual_seed(attacker.seed)
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps_eff, eps_eff, generator=gen) * 0.01
    adv = attacker._project_pixel_master(adv, x_orig)

    final_grad_norm = 0.0; final_grad_abs = 0.0
    for step_i in range(max(n_steps, 1)):
        adv = adv.detach().requires_grad_(True)
        adv_for_loss = attacker._cast_projected_pixel_values(adv, x_orig_model)
        loss = attacker._loss(full_ids, labels, adv_for_loss, **_loss_kwargs)
        try:
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        except Exception as e:
            result['probe_status'] = 'grad_error'
            result['probe_error'] = '%s step %d: %s' % (probe_name, step_i, str(e)[:100])
            del loss; torch.cuda.empty_cache()
            return result

        if grad is None:
            result['probe_status'] = 'grad_missing'
            result['probe_error'] = '%s step %d: grad is None' % (probe_name, step_i)
            return result

        final_grad_norm = float(grad.norm().item())
        final_grad_abs = float(grad.abs().mean().item())

        # Maximize margin: gradient ASCENT (loss = -margin, so -grad = +margin direction)
        # loss.backward gave us dL/dx. For margin maximization: x += step * sign(grad)
        # Wait — _loss returns total loss, and autograd.grad(loss, adv) gives d(loss)/d(adv)
        # The prefix_locked_gripper_open_margin loss is NEGATIVE of the margin we want to maximize
        # So: adv = adv - step * sign(grad)  (gradient descent on loss = ascent on margin)
        with torch.no_grad():
            adv_new = adv - step_sz * grad.sign()
            adv = attacker._project_pixel_master(adv_new, x_orig)
        del grad, loss
        torch.cuda.empty_cache()

    result['grad_norm'] = round(final_grad_norm, 6)
    result['grad_abs_mean'] = round(final_grad_abs, 8)

    # Audit after perturbation using standardized scores
    with torch.no_grad():
        adv_model = attacker._cast_projected_pixel_values(adv.detach(), x_orig_model)
        adv_gripper_row = get_gripper_row_from_full_ids(full_ids, adv_model)
        result['adv_open_score'] = round(float(compute_open_score_from_logits(adv_gripper_row).item()), 6)
        result['adv_loss_score'] = round(float(compute_loss_score_from_logits(adv_gripper_row).item()), 6)

        adv_audit = attacker._audit_logits(full_ids, labels, adv_model, target_ids, UNNORM_KEY,
                                            postprocess_gripper=True, region_token_ids=open_token_ids)
        result['adv_open_prob_mass'] = float(adv_audit.get('open_prob_mass', 0.0))

    result['open_score_gain'] = round(result['adv_open_score'] - result['clean_open_score'], 6)
    result['loss_delta'] = round(result['adv_loss_score'] - result['clean_loss_score'], 6)
    result['open_prob_gain'] = round(result['adv_open_prob_mass'] - result['clean_open_prob_mass'], 8)

    # Token flip: compare generated tokens from clean vs adv pixel_values
    with torch.inference_mode():
        gen_clean = model.generate(input_ids=ids, pixel_values=x_orig_model,
                                   max_new_tokens=action_dim, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True)
        gen_adv = model.generate(input_ids=ids, pixel_values=adv_model,
                                 max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
    clean_tids = gen_clean.sequences[0, -action_dim:].cpu().numpy()
    adv_tids = gen_adv.sequences[0, -action_dim:].cpu().numpy()
    result['token_flip'] = int(clean_tids[-1]) != int(adv_tids[-1])

    # Validate grad_norm
    if result['grad_norm'] == 0.0:
        result['probe_status'] = 'grad_invalid'
        if not result['probe_error']:
            result['probe_error'] = 'grad_norm is zero'

    return result

# ── Task config (matches vis_rollout_adaptive_v3.py) ─────────────
TASK_CFG = {
    'alphabet_soup': {'task_id': 0}, 'cream_cheese': {'task_id': 1},
    'salad_dressing': {'task_id': 2}, 'bbq_sauce': {'task_id': 3},
    'ketchup': {'task_id': 4}, 'tomato_sauce': {'task_id': 5},
    'butter': {'task_id': 6}, 'milk': {'task_id': 7},
    'orange_juice': {'task_id': 9},
}

# ── Prompt ────────────────────────────────────────────────────────
def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

# ── Process candidates ────────────────────────────────────────────
from PIL import Image
from gripper_attack.uncertainty import extract_prefix_logits

with open(args.shard_csv) as f:
    candidates = list(csv.DictReader(f))
log('Loaded %d candidates' % len(candidates))

step_rows = []; window_rows = []
EPS_RAW = args.eps_raw_pixels

for idx, c in enumerate(candidates):
    task = c['task_key'].strip(); sid = int(c['state_id'])
    ws = int(c['window_start']); we = int(c['window_end'])
    label = c.get('label_status','?'); tax = c.get('taxonomy','?')
    log('[%d/%d] %s s%d [%d,%d] %s' % (idx+1, len(candidates), task, sid, ws, we, label))

    cfg = TASK_CFG.get(task)
    if cfg is None: log('  SKIP: unknown task'); continue

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        bm_dict = benchmark.get_benchmark_dict()
        task_suite = bm_dict['libero_object']()
        task_obj = task_suite.get_task(cfg['task_id'])
        initial_states = task_suite.get_task_init_states(cfg['task_id'])
        if int(sid) >= len(initial_states):
            log('  SKIP: state OOB'); continue
        instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else task.replace('_',' ')
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
        text = prompt_fn(instruction.lower())
        inputs = processor(text, pil, return_tensors='pt')
        for k, v in list(inputs.items()):
            if torch.is_floating_point(v): inputs[k] = v.to(device=model_device, dtype=model_dtype)
            else: inputs[k] = v.to(model_device)
        if not torch.all(inputs['input_ids'][:, -1] == 29871):
            inputs['input_ids'] = torch.cat((inputs['input_ids'],
                torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        tids = gen.sequences[0, -action_dim:].cpu().numpy()
        VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        BC = np.asarray(model.bin_centers, dtype=np.float32)
        s = model.get_action_stats(UNNORM_KEY)
        lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
        mk = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
        disc = np.clip(VS - tids - 1, 0, len(BC)-1)
        action = np.where(mk, 0.5*(BC[disc].astype(np.float32)+1)*(hi-lo)+lo, BC[disc].astype(np.float32)).astype(np.float32)
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    # Probe at sampled steps
    window_len = we - ws + 1
    n_probe = min(args.probe_steps, window_len)
    probe_steps = sorted(set([ws, (ws+we)//2, we]))[:max(n_probe,1)]
    log('  probing at: %s' % probe_steps)

    step_probe_results = []
    while not done and current_step <= max(probe_steps) + 1:
        img = obs['agentview_image']
        if current_step in probe_steps:
            pil = Image.fromarray(img.astype(np.uint8))
            text = prompt_fn(instruction.lower())
            inputs = processor(text, pil, return_tensors='pt')
            for k, v in list(inputs.items()):
                if torch.is_floating_point(v): inputs[k] = v.to(device=model_device, dtype=model_dtype)
                else: inputs[k] = v.to(model_device)
            if not torch.all(inputs['input_ids'][:, -1] == 29871):
                inputs['input_ids'] = torch.cat((inputs['input_ids'],
                    torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)

            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                     return_dict_in_generate=True, output_scores=True)
            clean_tids = gen.sequences[0, -action_dim:].cpu().numpy()

            pv = inputs['pixel_values']; ids = inputs['input_ids']

            probe_results = {}
            for pname, psteps in [('FGSM1',1),('PGD1',1),('PGD3',3)]:
                res = run_probe_v0b(pv, ids, clean_tids, psteps, pname)
                probe_results[pname] = res

            max_open_score_gain = max(probe_results[p]['open_score_gain'] for p in probe_results)
            max_grad = max(probe_results[p]['grad_norm'] for p in probe_results)
            any_flip = any(probe_results[p]['token_flip'] for p in probe_results)
            any_error = any(probe_results[p]['probe_status'] != 'ok' for p in probe_results)

            sr = {'step': current_step}
            for pname in ['FGSM1','PGD1','PGD3']:
                r = probe_results[pname]
                for k in ['grad_norm','open_score_gain','loss_delta','open_prob_gain','token_flip','probe_status','probe_error']:
                    sr['%s_%s' % (pname.lower(), k)] = str(r.get(k, ''))
            sr['max_open_score_gain'] = str(max_open_score_gain)
            sr['max_grad_norm'] = str(max_grad)
            sr['any_token_flip'] = str(int(any_flip))
            sr['any_probe_error'] = str(int(any_error))
            sr['open_token_count'] = str(probe_results['FGSM1']['open_token_count'])
            sr['close_token_count'] = str(probe_results['FGSM1']['close_token_count'])
            step_probe_results.append(sr)

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        tids = gen.sequences[0, -action_dim:].cpu().numpy()
        disc = np.clip(VS - tids - 1, 0, len(BC)-1)
        action = np.where(mk, 0.5*(BC[disc].astype(np.float32)+1)*(hi-lo)+lo, BC[disc].astype(np.float32)).astype(np.float32)
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    env.close()

    if not step_probe_results: continue

    for sr in step_probe_results:
        sr['task_key'] = task; sr['state_id'] = str(sid)
        sr['window_start'] = str(ws); sr['window_end'] = str(we)
        sr['label_status'] = label; sr['taxonomy'] = tax
        step_rows.append(sr)

    gains = [float(s['max_open_score_gain']) for s in step_probe_results]
    grads = [float(s['max_grad_norm']) for s in step_probe_results]
    window_rows.append({
        'task_key': task, 'state_id': str(sid), 'window_start': str(ws), 'window_end': str(we),
        'label_status': label, 'taxonomy': tax,
        'n_probe_steps': str(len(step_probe_results)),
        'max_open_score_gain': str(round(max(gains), 6)),
        'mean_open_score_gain': str(round(np.mean(gains), 6)),
        'max_grad_norm': str(round(max(grads), 6)),
        'any_token_flip': str(int(any_flip)),
        'any_probe_error': str(int(any_error)),
    })
    log('  open_gain: max=%.6f mean=%.6f grad=%.6f flip=%d err=%d' % (
        max(gains), np.mean(gains), max(grads), int(any_flip), int(any_error)))

# ── Write ─────────────────────────────────────────────────────────
step_csv = os.path.join(REPO, 'tables/active_probe_v0b_step_features_%s.csv' % args.shard_name)
win_csv = os.path.join(REPO, 'tables/active_probe_v0b_window_features_%s.csv' % args.shard_name)
if step_rows:
    with open(step_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys())); w.writeheader(); w.writerows(step_rows)
    log('Wrote %d step rows to %s' % (len(step_rows), step_csv))
if window_rows:
    with open(win_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys())); w.writeheader(); w.writerows(window_rows)
    log('Wrote %d window rows to %s' % (len(window_rows), win_csv))
log('=== Active Probe v0b %s complete: %d windows ===' % (args.shard_name, len(window_rows)))
