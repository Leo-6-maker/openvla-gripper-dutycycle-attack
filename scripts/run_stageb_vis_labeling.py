#!/usr/bin/env python3
"""Stage-B VIS labeling: run VIS PGD20 + matched random Linf for a single window job.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_stageb_vis_labeling.py \
    --gpu_pair 0,1 --task ketchup --state-id 0 --window_start 16 --window_end 33 \
    --condition vis_pgd --job_id 0 --output_dir /path/to/out
"""
import csv, os, sys, argparse, json
from datetime import datetime
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--task', required=True)
ap.add_argument('--state-id', type=int, required=True)
ap.add_argument('--window_start', type=int, required=True)
ap.add_argument('--window_end', type=int, required=True)
ap.add_argument('--condition', choices=['vis_pgd', 'random_linf'], required=True)
ap.add_argument('--job_id', type=int, default=-1)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--max_steps', type=int, default=400)
args = ap.parse_args()

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]
if _physical and _physical == [0, 1]:
    gpu_ids = [1, 0]; _render_gpu = 1
log('GPU: physical=%s render=%d' % (_VISIBLE, _render_gpu))

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
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
log('Model loaded, action_dim=%d, device=%s' % (action_dim, model_device))

VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32)
HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

def decode_tokens_to_action(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP) - 1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

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
    model=model, processor=processor, config=attacker_config, seed=0,
    device='cuda:%d' % gpu_ids[0], preprocess_kwargs={'postprocess_gripper': True})
attacker._freeze_model()

def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

def make_inputs(pil_image, instruction):
    text = prompt_fn(instruction.lower())
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
    return decode_tokens_to_action(tids)

TASK_CFG = {
    'ketchup': 0, 'butter': 1, 'cream_cheese': 2, 'salad_dressing': 3,
    'bbq_sauce': 4, 'milk': 5, 'alphabet_soup': 6, 'tomato_sauce': 7, 'orange_juice': 8,
}

# ── Run ──────────────────────────────────────────────────────────
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ws = args.window_start; we = args.window_end
cid = '%s_s%d_w%d_%d' % (args.task, args.state_id, ws, we)
log('%s [%d,%d] %s' % (cid, ws, we, args.condition))

cfg = TASK_CFG.get(args.task)
if cfg is None: log('FATAL: unknown task'); sys.exit(1)

infra_status = 'ok'; provenance = 'clean'
trace_rows = []; decoded_grips = []; qpos_deltas = []
clean_actions = []

try:
    bm_dict = benchmark.get_benchmark_dict()
    task_suite = bm_dict['libero_object']()
    task_obj = task_suite.get_task(cfg)
    initial_states = task_suite.get_task_init_states(cfg)
    if args.state_id >= len(initial_states):
        log('FATAL: state OOB'); sys.exit(1)
    instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else args.task.replace('_', ' ')
    bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                             render_gpu_device_id=_render_gpu)
    env.seed(0); obs = env.reset()
    env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(initial_states[args.state_id])
except Exception as e:
    log('INFRA: env init error: %s' % str(e)[:80])
    infra_status = 'env_init_fail'
    env = None

if env is not None:
    current_step = 0; done = False; qpos_before = 0; width_before = 0
    try:
        while not done and current_step < min(we + 5, args.max_steps):
            img = obs['agentview_image']
            pil = Image.fromarray(img.astype(np.uint8))
            inputs = make_inputs(pil, instruction)
            clean_pv = inputs['pixel_values']; clean_ids = inputs['input_ids']

            # Decode clean action
            clean_action = decode_action(inputs)
            clean_grip = get_env_gripper(clean_action)
            clean_actions.append(clean_action)

            # Gripper state
            
            # Use obs robot0_gripper_qpos (correct joint indices)
            gq = obs.get('robot0_gripper_qpos', np.zeros(2))
            gripper_qpos = float((abs(gq[0]) + abs(gq[1])) / 2.0)
            try:
                eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_center')]
            except:
                eef_pos = np.zeros(3)

            in_window = 1 if ws <= current_step <= we else 0
            attack_this_step = in_window and args.condition != 'clean'

            env_grip = clean_grip; arm_l2 = 0.0; pgd_applied = 0; attacks_applied = 0

            if attack_this_step:
                if args.condition == 'vis_pgd':
                    try:
                        qpos_before = gripper_qpos
                        width_before = float(abs(obs.get("robot0_gripper_qpos", [0,0])[0]) + abs(obs.get("robot0_gripper_qpos", [0,0])[1])) / 2.0
                        result = attacker.attack(observation=pil, instruction=instruction.lower(),
                                                  target_action=clean_action, unnorm_key=UNNORM_KEY)
                        adv_inputs = get_adv_inputs_from_attack_result(result)
                        adv_pv = adv_inputs['pixel_values'].to(device=model_device, dtype=model_dtype)
                        adv_ids = adv_inputs['input_ids'].to(model_device)
                        adv_action = decode_action({'input_ids': adv_ids, 'pixel_values': adv_pv})
                        env_grip = get_env_gripper(adv_action)
                        arm_l2 = float(np.linalg.norm((adv_action[:6] - clean_action[:6]).reshape(-1)))
                        pgd_applied = 1
                        attacks_applied = 1
                        # Use adversarial action for env step
                        env_action_full = normalize_gripper_action(adv_action.copy(), binarize=True)
                        env_action_full = invert_gripper_action(env_action_full)
                    except Exception as e:
                        env_grip = clean_grip
                        infra_status = 'pgd_error: %s' % str(e)[:60]
                        env_action_full = normalize_gripper_action(clean_action.copy(), binarize=True)
                        env_action_full = invert_gripper_action(env_action_full)

                elif args.condition == 'random_linf':
                    try:
                        qpos_before = gripper_qpos
                        noise = (2 * torch.rand_like(clean_pv) - 1) * _eps_eff
                        rand_pv = torch.clamp(clean_pv + noise, clean_pv - _eps_eff, clean_pv + _eps_eff)
                        rand_action = decode_action({'input_ids': clean_ids, 'pixel_values': rand_pv})
                        env_grip = get_env_gripper(rand_action)
                        arm_l2 = float(np.linalg.norm((rand_action[:6] - clean_action[:6]).reshape(-1)))
                        env_action_full = normalize_gripper_action(rand_action.copy(), binarize=True)
                        env_action_full = invert_gripper_action(env_action_full)
                    except Exception as e:
                        env_grip = clean_grip
                        infra_status = 'random_error: %s' % str(e)[:60]
                        env_action_full = normalize_gripper_action(clean_action.copy(), binarize=True)
                        env_action_full = invert_gripper_action(env_action_full)
                else:
                    env_action_full = normalize_gripper_action(clean_action.copy(), binarize=True)
                    env_action_full = invert_gripper_action(env_action_full)
            else:
                env_action_full = normalize_gripper_action(clean_action.copy(), binarize=True)
                env_action_full = invert_gripper_action(env_action_full)

            if in_window:
                decoded_grips.append(env_grip)
                
                gq_after = obs.get('robot0_gripper_qpos', np.zeros(2))
                qpos_after = float((abs(gq_after[0]) + abs(gq_after[1])) / 2.0)
                qpos_deltas.append(qpos_after - qpos_before if attack_this_step else 0.0)

            trace_rows.append({
                'step': str(current_step), 'in_window': str(in_window),
                'attack_this_step': str(int(attack_this_step)),
                'env_grip': str(round(env_grip, 1)), 'arm_l2': str(round(arm_l2, 6)),
                'pgd_applied': str(pgd_applied), 'attacks_applied': str(attacks_applied),
                'gripper_qpos': str(round(gripper_qpos, 6)),
                'done': str(int(done)),
            })

            obs, reward, done, info = env.step(env_action_full)
            current_step += 1

    except Exception as e:
        infra_status = 'runtime_error: %s' % str(e)[:80]
        log('INFRA: %s' % infra_status)

    env.close()

# ── Compute summary metrics ──────────────────────────────────────
open_count = sum(1 for g in decoded_grips if g > 0)
streak = 0; max_streak = 0
for g in decoded_grips:
    if g > 0: streak += 1; max_streak = max(max_streak, streak)
    else: streak = 0
total_qpos_delta = float(np.sum(np.abs(qpos_deltas))) if qpos_deltas else 0.0
mean_arm_l2 = float(np.mean([float(r['arm_l2']) for r in trace_rows if r['in_window'] == '1'])) if trace_rows else 0.0
success = 1 if done else 0

# ── Write outputs ────────────────────────────────────────────────
tag = '%s_%s_job%d' % (args.task, args.condition, args.job_id)
os.makedirs(args.output_dir, exist_ok=True)

# Summary JSON
summary = {
    'job_id': args.job_id,
    'task_key': args.task, 'state_id': str(args.state_id),
    'window_start': ws, 'window_end': we,
    'condition': args.condition,
    'infra_status': infra_status, 'provenance_status': provenance,
    'decoded_open_count': open_count, 'decoded_longest_open_streak': max_streak,
    'qpos_delta': round(total_qpos_delta, 6), 'mean_arm_l2': round(mean_arm_l2, 6),
    'success': success, 'n_window_steps': len(decoded_grips),
    'n_total_steps': current_step,
}
with open(os.path.join(args.output_dir, 'summary_%s.json' % tag), 'w') as f:
    json.dump(summary, f)

# Trace CSV
if trace_rows:
    with open(os.path.join(args.output_dir, 'trace_%s.csv' % tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)

log('Done: open=%d streak=%d qpos_delta=%.6f arm_l2=%.6f infra=%s' % (
    open_count, max_streak, total_qpos_delta, mean_arm_l2, infra_status))
