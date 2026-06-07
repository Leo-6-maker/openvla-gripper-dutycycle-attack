#!/usr/bin/env python3
"""Stage-B VIS labeling v1.1 — aligned with OpenVLA-LIBERO executable spec.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_stageb_vis_labeling.py \
    --gpu_pair 0,1 --task ketchup --state-id 0 --window_start 16 --window_end 33 \
    --condition vis_pgd --job_id 0 --output_dir /path/to/out
"""
import csv, os, sys, argparse, json, subprocess, uuid
from datetime import datetime
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

# ── Spec module ────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

from gripper_attack.openvla_libero_exec_spec import (
    OPENVLA_LIBERO_EXEC_SPEC_VERSION,
    OFFICIAL_UNNORM_KEY_LIBERO_OBJECT,
    official_prompt,
    normalize_gripper_raw,
    raw_gripper_to_env_gripper,
    env_gripper_is_open, env_gripper_is_close,
    raw_gripper_is_open,
    get_libero_image_official,
)

UNNORM_KEY = OFFICIAL_UNNORM_KEY_LIBERO_OBJECT
TRACE_VERSION = 'corrected_stageb_v1_1'
RUNNER_VERSION = 'stageb_vis_labeling_v1_1_spec_aligned_20260607'
OPEN_CONVENTION = 'env_action_6_lt_neg_0p5_means_OPEN'


def _get_git_info():
    """Return (git_commit_short, git_dirty_bool)."""
    try:
        r = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        commit = r.stdout.strip() if r.returncode == 0 else 'unknown'
        r2 = subprocess.run(['git', '-C', REPO, 'diff-index', '--quiet', 'HEAD', '--'],
                            capture_output=True, timeout=5)
        dirty = '1' if r2.returncode != 0 else '0'
        return commit, dirty
    except Exception:
        return 'unknown', '1'


GIT_COMMIT, GIT_DIRTY = _get_git_info()

# ── CLI ────────────────────────────────────────────────────────────
def log(msg):
    print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))


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
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--image_preprocess', choices=['official_rot180', 'legacy_no_rotation'],
                default='official_rot180')
args = ap.parse_args()

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]
if _physical and _physical == [0, 1]:
    gpu_ids = [1, 0]; _render_gpu = 1
log('GPU: physical=%s render=%d' % (_VISIBLE, _render_gpu))
log('spec=%s trace=%s runner=%s' % (OPENVLA_LIBERO_EXEC_SPEC_VERSION, TRACE_VERSION, RUNNER_VERSION))

# ── Load model ──────────────────────────────────────────────────────
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
    """Normalize gripper dimension to [-1, +1] with sign binarization."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = normalize_gripper_raw(float(action[..., -1]), binarize=binarize)
    return action


def invert_gripper_action(action):
    """Invert gripper sign for LIBERO env (-1=OPEN, +1=CLOSE)."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


# ── Create attacker ─────────────────────────────────────────────────
_eps_eff = args.eps_raw_pixels / 255.0

# ── v1.1 runtime constants ─────────────────────────────────────────
PROMPT_STYLE = 'official_in_out'
IMAGE_PREPROCESS_STYLE = 'official_rot180_only' if args.image_preprocess == 'official_rot180' else 'legacy_direct_agentview_no_rotation'
EPS_PROCESSOR = _eps_eff
EPS_RAW_PIXELS_COMPAT = args.eps_raw_pixels

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
    model=model, processor=processor, config=attacker_config, seed=args.seed,
    device='cuda:%d' % gpu_ids[0], preprocess_kwargs={'postprocess_gripper': True})
attacker._freeze_model()


def make_inputs(pil_image, instruction):
    text = official_prompt(instruction.lower())
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

# ── Run ─────────────────────────────────────────────────────────────
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ws = args.window_start; we = args.window_end
pair_id = str(uuid.uuid4())[:8]
cid = '%s_s%d_w%d_%d' % (args.task, args.state_id, ws, we)
log('%s [%d,%d] %s pair=%s' % (cid, ws, we, args.condition, pair_id))

cfg = TASK_CFG.get(args.task)
if cfg is None:
    log('FATAL: unknown task'); sys.exit(1)


infra_status = 'ok'; provenance = 'clean'
trace_rows = []; decoded_grips = []; qpos_deltas_shifted = []
current_step = 0; done = False

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
    env.seed(args.seed); obs = env.reset()
    env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(initial_states[args.state_id])
except Exception as e:
    log('INFRA: env init error: %s' % str(e)[:80])
    infra_status = 'env_init_fail'
    env = None

if env is not None:
    rng = np.random.RandomState(args.seed + args.job_id)
    try:
        while not done and current_step < min(we + 5, args.max_steps):
            # ── Official image preprocessing ──
            if args.image_preprocess == 'official_rot180':
                img = get_libero_image_official(obs)
            else:
                img = obs['agentview_image']
            pil = Image.fromarray(img.astype(np.uint8))
            inputs = make_inputs(pil, instruction)
            clean_pv = inputs['pixel_values']; clean_ids = inputs['input_ids']

            # Decode clean action
            clean_action = decode_action(inputs)
            clean_grip = raw_gripper_to_env_gripper(float(clean_action[-1]))

            # ── Gripper qpos from obs ──
            gq = obs.get('robot0_gripper_qpos', np.zeros(2))
            q0, q1 = float(gq[0]), float(gq[1])
            qpos_abs_sum = abs(q0) + abs(q1)
            qpos_abs_mean = qpos_abs_sum / 2.0

            in_window = 1 if ws <= current_step <= we else 0
            attack_this_step = in_window and args.condition != 'clean'

            env_grip = clean_grip
            arm_l2 = 0.0
            pgd_applied = 0
            attacks_applied = 0
            raw_action = clean_action.copy()
            random_seed_str = ''
            noise_linf = '0'
            noise_l2 = '0'
            perturbation_space = 'none'

            if attack_this_step:
                if args.condition == 'vis_pgd':
                    try:
                        qpos_before = qpos_abs_sum
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
                        qpos_before = qpos_abs_sum
                        random_seed_str = str(args.seed + args.job_id)
                        noise = (2 * torch.rand_like(clean_pv) - 1) * _eps_eff
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

            # Compute env action
            env_action_full = normalize_gripper_action(raw_action.copy(), binarize=True)
            env_action_full = invert_gripper_action(env_action_full)
            env_action_6 = float(env_action_full[6])

            # ── decoded_open_bool from spec function ──
            decoded_open_bool = str(int(env_gripper_is_open(env_action_6)))

            if in_window:
                decoded_grips.append(env_action_6)

            # ── Full trace row ──
            row = {
                'step': str(current_step),
                'row_id': '%s_%s_%d' % (args.task, pair_id, current_step),
                'in_window': str(in_window),
                'attack_this_step': str(int(attack_this_step)),
                'pair_id': pair_id,
                'condition': args.condition,
                'task_key': args.task,
                'state_id': str(args.state_id),
                'seed': str(args.seed),
                'window_start': str(ws),
                'window_end': str(we),
                'done': str(int(done)),
                'pgd_applied': str(pgd_applied),
                'attacks_applied': str(attacks_applied),
                # env actions
                'env_action_0': str(round(float(env_action_full[0]), 6)),
                'env_action_1': str(round(float(env_action_full[1]), 6)),
                'env_action_2': str(round(float(env_action_full[2]), 6)),
                'env_action_3': str(round(float(env_action_full[3]), 6)),
                'env_action_4': str(round(float(env_action_full[4]), 6)),
                'env_action_5': str(round(float(env_action_full[5]), 6)),
                'env_action_6': str(round(env_action_6, 6)),
                'env_grip': str(round(env_grip, 1)),
                'decoded_open_bool': decoded_open_bool,
                'open_convention': OPEN_CONVENTION,
                # raw actions
                'raw_action_0': str(round(float(raw_action[0]), 6)),
                'raw_action_1': str(round(float(raw_action[1]), 6)),
                'raw_action_2': str(round(float(raw_action[2]), 6)),
                'raw_action_3': str(round(float(raw_action[3]), 6)),
                'raw_action_4': str(round(float(raw_action[4]), 6)),
                'raw_action_5': str(round(float(raw_action[5]), 6)),
                'raw_action_6': str(round(float(raw_action[6]), 6)),
                # qpos
                'obs_gripper_qpos_0': str(round(q0, 6)),
                'obs_gripper_qpos_1': str(round(q1, 6)),
                'obs_gripper_qpos_abs_sum': str(round(qpos_abs_sum, 6)),
                'obs_gripper_qpos_abs_mean': str(round(qpos_abs_mean, 6)),
                'qpos_source': 'obs_robot0_gripper_qpos',
                # arm
                'arm_l2': str(round(arm_l2, 6)),
                'gripper_qpos': str(round(qpos_abs_mean, 6)),
                # random Linf metadata
                'random_seed': random_seed_str,
                'perturbation_space': perturbation_space,
                'random_noise_linf': noise_linf,
                'random_noise_l2': noise_l2,
                'eps_processor': str(round(EPS_PROCESSOR, 6)),
                'eps_raw_pixels_name_deprecated_or_compat': str(EPS_RAW_PIXELS_COMPAT),
                # provenance
                'trace_version': TRACE_VERSION,
                'runner_version': RUNNER_VERSION,
                'exec_spec_version': OPENVLA_LIBERO_EXEC_SPEC_VERSION,
                'git_commit': GIT_COMMIT,
                'git_dirty': GIT_DIRTY,
                'prompt_style': PROMPT_STYLE,
                'image_preprocess_style': IMAGE_PREPROCESS_STYLE,
                'unnorm_key': UNNORM_KEY,
            }
            trace_rows.append(row)

            # ── Step env ──
            obs, reward, done, info = env.step(env_action_full)

            # Measure qpos AFTER env.step for shifted delta
            if in_window and attack_this_step:
                gq_after = obs.get('robot0_gripper_qpos', np.zeros(2))
                q0a, q1a = float(gq_after[0]), float(gq_after[1])
                qpos_after = abs(q0a) + abs(q1a)
                qpos_deltas_shifted.append(qpos_after - qpos_before)

            current_step += 1

    except Exception as e:
        infra_status = 'runtime_error: %s' % str(e)[:80]
        log('INFRA: %s' % infra_status)

    env.close()

# ── Compute summary metrics using spec functions ────────────────────
open_count = sum(1 for g in decoded_grips if env_gripper_is_open(g))
streak = 0; max_streak = 0
for g in decoded_grips:
    if env_gripper_is_open(g):
        streak += 1; max_streak = max(max_streak, streak)
    else:
        streak = 0
total_qpos_delta = float(np.sum(np.abs(qpos_deltas_shifted))) if qpos_deltas_shifted else 0.0
mean_arm_l2 = float(np.mean([float(r['arm_l2']) for r in trace_rows if r['in_window'] == '1'])) if trace_rows else 0.0
success = 1 if done else 0

# ── Trace column order (deterministic, spec-aligned) ────────────────
TRACE_COLUMNS = [
    'step', 'row_id', 'in_window', 'attack_this_step',
    'pair_id', 'condition', 'task_key', 'state_id', 'seed',
    'window_start', 'window_end', 'done',
    'pgd_applied', 'attacks_applied',
    'env_action_0', 'env_action_1', 'env_action_2', 'env_action_3',
    'env_action_4', 'env_action_5', 'env_action_6', 'env_grip',
    'decoded_open_bool', 'open_convention',
    'raw_action_0', 'raw_action_1', 'raw_action_2', 'raw_action_3',
    'raw_action_4', 'raw_action_5', 'raw_action_6',
    'obs_gripper_qpos_0', 'obs_gripper_qpos_1',
    'obs_gripper_qpos_abs_sum', 'obs_gripper_qpos_abs_mean',
    'qpos_source', 'arm_l2', 'gripper_qpos',
    'random_seed', 'perturbation_space',
    'random_noise_linf', 'random_noise_l2',
    'eps_processor', 'eps_raw_pixels_name_deprecated_or_compat',
    'trace_version', 'runner_version', 'exec_spec_version',
    'git_commit', 'git_dirty',
    'prompt_style', 'image_preprocess_style', 'unnorm_key',
]

# ── Write outputs ───────────────────────────────────────────────────
tag = '%s_%s_job%d' % (args.task, args.condition, args.job_id)
os.makedirs(args.output_dir, exist_ok=True)

# Summary JSON
summary = {
    'job_id': args.job_id,
    'pair_id': pair_id,
    'task_key': args.task, 'state_id': str(args.state_id),
    'window_start': ws, 'window_end': we,
    'condition': args.condition,
    'seed': args.seed,
    'infra_status': infra_status, 'provenance_status': provenance,
    'decoded_open_count': open_count, 'decoded_longest_open_streak': max_streak,
    'open_convention': OPEN_CONVENTION,
    'qpos_delta': round(total_qpos_delta, 6), 'mean_arm_l2': round(mean_arm_l2, 6),
    'success': success, 'n_window_steps': len(decoded_grips),
    'n_total_steps': current_step,
    'trace_version': TRACE_VERSION,
    'runner_version': RUNNER_VERSION,
    'exec_spec_version': OPENVLA_LIBERO_EXEC_SPEC_VERSION,
    'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY,
    'prompt_style': PROMPT_STYLE,
    'image_preprocess_style': IMAGE_PREPROCESS_STYLE,
    'unnorm_key': UNNORM_KEY,
    'open_convention': OPEN_CONVENTION,
}
with open(os.path.join(args.output_dir, 'summary_%s.json' % tag), 'w') as f:
    json.dump(summary, f)

# Trace CSV
if trace_rows:
    with open(os.path.join(args.output_dir, 'trace_%s.csv' % tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS, extrasaction='ignore')
        w.writeheader(); w.writerows(trace_rows)

log('Done: open=%d streak=%d qpos_delta=%.6f arm_l2=%.6f infra=%s' % (
    open_count, max_streak, total_qpos_delta, mean_arm_l2, infra_status))
