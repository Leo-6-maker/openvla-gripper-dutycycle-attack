#!/usr/bin/env python3
"""Sign convention audit: test both gradient directions on one candidate.
Determines whether descent or ascent increases open_score.
"""
import csv, os, sys, time
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE: print('FATAL: CUDA_VISIBLE_DEVICES required'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
OUT_DIR = '/data/liuyu/outputs/active_probe_v0b_20260606'
UNNORM_KEY = 'libero_object'; os.makedirs(OUT_DIR, exist_ok=True)

GPU_PAIR = '0,1'  # logical
EPS_RAW = 6.0; N_PROBE_STEPS = 3
gpu_ids = [0, 1]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else 0

from transformers import AutoModelForVision2Seq, AutoProcessor
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, CANONICAL_OPEN_SEMANTICS_VERSION

print('Loading model...')
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={0: '10500MiB', 1: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))

_eps_eff = EPS_RAW / 255.0
attacker = TokenPrefixPGDAttacker(
    model=model, processor=processor,
    config={'epsilon': _eps_eff, 'step_size': _eps_eff / 3 * 1.5, 'num_steps': 3,
            'random_start': True, 'objective': 'prefix_locked_gripper_open_margin',
            'arm_preserve_weight': 0.5, 'gripper_margin': 5.0},
    seed=0, device='cuda:0', preprocess_kwargs={'postprocess_gripper': True})
attacker._freeze_model()
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

region = attacker.get_gripper_region_by_decoded_action(UNNORM_KEY, postprocess_gripper=True)
open_token_ids = region['open_token_ids']
close_token_ids = region['close_token_ids']
print('Open tokens: %d, Close tokens: %d' % (region['open_count'], region['close_count']))

# Helpers
from PIL import Image
def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

# Standardized scoring functions
def compute_open_score(gripper_row_logits):
    """open_score = logsumexp(open_tokens) - max(non_open_tokens). Higher = more OPEN."""
    open_lse = torch.logsumexp(gripper_row_logits[open_token_ids], dim=0)
    # non-open = close + boundary; use max for robustness
    non_open_mask = torch.ones_like(gripper_row_logits, dtype=torch.bool)
    non_open_mask[open_token_ids] = False
    max_non_open = gripper_row_logits[non_open_mask].max()
    return open_lse - max_non_open

def compute_loss_score(gripper_row_logits, margin=5.0):
    """loss_score = ReLU(max_non_open - logsumexp(open) + margin). Lower = better (less loss)."""
    open_lse = torch.logsumexp(gripper_row_logits[open_token_ids], dim=0)
    non_open_mask = torch.ones_like(gripper_row_logits, dtype=torch.bool)
    non_open_mask[open_token_ids] = False
    max_non_open = gripper_row_logits[non_open_mask].max()
    return torch.relu(max_non_open - open_lse + margin)

# Action transform
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

# ── Run ketchup s0 to step 16 (smoke-proven probe step) ─────────
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm_dict = benchmark.get_benchmark_dict()
task_suite = bm_dict['libero_object']()
task_obj = task_suite.get_task(4)  # ketchup
initial_states = task_suite.get_task_init_states(4)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                         has_renderer=False, has_offscreen_renderer=True,
                         use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                         render_gpu_device_id=_render_gpu)
env.seed(0); obs = env.reset()
env.sim.data.qvel[:] = 0; env.sim.forward()
env.set_init_state(initial_states[0])

instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else 'pick up the ketchup and place it in the basket'
print('Instruction:', instruction)

# Step to step 16 (first probe step)
current_step = 0; done = False
while not done and current_step < 16:
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

print('At step %d, getting probe image...' % current_step)

# ── Get clean inputs at this step ─────────────────────────────────
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
clean_tids = gen.sequences[0, -action_dim:].cpu().numpy()

pv = inputs['pixel_values']; ids = inputs['input_ids']
target_ids = torch.as_tensor(np.asarray(clean_tids), dtype=torch.long, device=model_device)
prompt_len = ids.shape[1]
full_ids = torch.cat([ids, target_ids.unsqueeze(0)], dim=1)
labels = full_ids.clone(); labels[:, :prompt_len] = -100

x_orig = pv.to(dtype=model_dtype)
x_orig_model = x_orig.clone()
eps_eff = attacker.epsilon
step_sz = eps_eff / 3 * 1.5

# ── Get clean scores ──────────────────────────────────────────────
attacker._freeze_model()
with torch.no_grad():
    _lk = {'objective': 'prefix_locked_gripper_open_margin', 'num_action_tokens': int(target_ids.numel()), 'region_token_ids': open_token_ids}
    _clean_loss_out = attacker._loss(full_ids, labels, x_orig_model, **_lk)
    clean_audit = attacker._audit_logits(full_ids, labels, x_orig_model, target_ids, UNNORM_KEY,
                                          postprocess_gripper=True, region_token_ids=open_token_ids)
    # Also compute open_score directly
    _out = model(input_ids=full_ids, pixel_values=x_orig_model, use_cache=False, return_dict=True)
    _gripper_row = _out.logits[0, -1, :]
    clean_open_score = float(compute_open_score(_gripper_row).item())
    clean_loss_score = float(compute_loss_score(_gripper_row).item())

print('\nClean: open_score=%.4f loss_score=%.4f' % (clean_open_score, clean_loss_score))

# ── Test both directions ──────────────────────────────────────────
results = []
for direction_name, sign in [('A_descent_loss', -1.0), ('B_ascent_loss', +1.0)]:
    print('\n=== Direction: %s (sign=%.0f) ===' % (direction_name, sign))

    # Random start
    gen_rng = torch.Generator(device=x_orig.device); gen_rng.manual_seed(0)
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps_eff, eps_eff, generator=gen_rng) * 0.01
    adv = attacker._project_pixel_master(adv, x_orig)

    final_grad_norm = 0.0
    for step_i in range(3):
        adv = adv.detach().requires_grad_(True)
        adv_for_loss = attacker._cast_projected_pixel_values(adv, x_orig_model)
        loss = attacker._loss(full_ids, labels, adv_for_loss, **_lk)
        try:
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        except Exception as e:
            print('  GRAD ERROR: %s' % e); break
        if grad is None: print('  GRAD None'); break
        final_grad_norm = float(grad.norm().item())

        with torch.no_grad():
            adv_new = adv + sign * step_sz * grad.sign()  # sign = +1 for ascent, -1 for descent
            adv = attacker._project_pixel_master(adv_new, x_orig)
        del grad, loss; torch.cuda.empty_cache()

    # Audit after perturbation
    with torch.no_grad():
        adv_model = attacker._cast_projected_pixel_values(adv.detach(), x_orig_model)
        adv_audit = attacker._audit_logits(full_ids, labels, adv_model, target_ids, UNNORM_KEY,
                                            postprocess_gripper=True, region_token_ids=open_token_ids)
        _out_adv = model(input_ids=full_ids, pixel_values=adv_model, use_cache=False, return_dict=True)
        _gr_adv = _out_adv.logits[0, -1, :]
        adv_open_score = float(compute_open_score(_gr_adv).item())
        adv_loss_score = float(compute_loss_score(_gr_adv).item())

    open_score_gain = adv_open_score - clean_open_score
    loss_delta = adv_loss_score - clean_loss_score
    open_prob_gain = float(adv_audit.get('open_prob_mass', 0.0)) - float(clean_audit.get('open_prob_mass', 0.0))

    print('  grad_norm=%.4f' % final_grad_norm)
    print('  clean_open_score=%.4f  adv_open_score=%.4f  open_score_gain=%.4f' % (clean_open_score, adv_open_score, open_score_gain))
    print('  clean_loss_score=%.4f  adv_loss_score=%.4f  loss_delta=%.4f' % (clean_loss_score, adv_loss_score, loss_delta))
    print('  open_prob_gain=%.6f' % open_prob_gain)

    # Token flip check
    with torch.inference_mode():
        gc = model.generate(input_ids=ids, pixel_values=x_orig_model, max_new_tokens=action_dim,
                            do_sample=False, return_dict_in_generate=True, output_scores=True)
        ga = model.generate(input_ids=ids, pixel_values=adv_model, max_new_tokens=action_dim,
                            do_sample=False, return_dict_in_generate=True, output_scores=True)
    ct = gc.sequences[0, -action_dim:].cpu().numpy()
    at = ga.sequences[0, -action_dim:].cpu().numpy()
    token_flip = int(ct[-1]) != int(at[-1])
    print('  token_flip=%d (clean=%d adv=%d)' % (token_flip, ct[-1], at[-1]))

    valid = open_score_gain > 0 and loss_delta < 0
    print('  VALID (gain>0, delta<0): %s' % ('YES' if valid else 'NO'))

    results.append({
        'direction': direction_name, 'sign': str(sign),
        'grad_norm': str(round(final_grad_norm, 4)),
        'clean_open_score': str(round(clean_open_score, 4)),
        'adv_open_score': str(round(adv_open_score, 4)),
        'open_score_gain': str(round(open_score_gain, 4)),
        'clean_loss_score': str(round(clean_loss_score, 4)),
        'adv_loss_score': str(round(adv_loss_score, 4)),
        'loss_delta': str(round(loss_delta, 4)),
        'open_prob_gain': str(round(open_prob_gain, 8)),
        'token_flip': str(token_flip),
        'valid': str(valid),
    })

env.close()

# ── Decision ──────────────────────────────────────────────────────
print('\n=== DECISION ===')
for r in results:
    print('%s: open_gain=%s loss_delta=%s valid=%s' % (
        r['direction'], r['open_score_gain'], r['loss_delta'], r['valid']))

valid_results = [r for r in results if r['valid'] == 'True']
if valid_results:
    print('\nCORRECT DIRECTION: %s' % valid_results[0]['direction'])
else:
    print('\nNO DIRECTION GIVES VALID RESULT — OBJECTIVE_SIGN_BLOCKED')

# Write audit CSV
out_csv = os.path.join(REPO, 'tables/active_probe_v0b_sign_audit_step_features.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print('Wrote %s' % out_csv)
