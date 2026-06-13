#!/usr/bin/env python3
"""V6 Online-Trigger Layer3 Runner.
Replaces fixed-window attack with online critical-CLOSE event detection.
Preprocessing EXACTLY matches V4 runner (decode_with_scores, enabled=True, K_trigger).
Three conditions: clean_observer, online_random_linf, online_vis_pgd.
Trigger: first pre-success clean CLOSE onset (RULE_TRIGGER_MVP).
Event budget: H=5 steps, B=3 max perturbed frames, eps=6, PGD=20.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")

# ── Args ──
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--condition', required=True,
                choices=['clean_observer', 'online_random_linf', 'online_vis_pgd'])
ap.add_argument('--attack_seed', type=int, default=99)
ap.add_argument('--random_control_seed', type=str, default='')
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=int, default=6)
ap.add_argument('--model_path',
                default='/data/aviary/models/openvla/openvla-7b-finetuned-libero-object')
ap.add_argument('--render_gpu_device_id', type=int, default=0)
ap.add_argument('--model_gpu_device_id', type=int, default=-1)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--job_id', type=str, default='0')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_steps_override', type=int, default=280)
ap.add_argument('--success_metric', default='check_success')
ap.add_argument('--num_steps_wait', type=int, default=10)
ap.add_argument('--event_horizon', type=int, default=5)
ap.add_argument('--max_perturb_frames', type=int, default=3)
args = ap.parse_args()

# ── V4-aligned model load ──
print('[%s] V6 loading model...' % time.strftime('%H:%M:%S'), flush=True)


def load_model_s20d(model_path, model_gpu_device_id=-1):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {
            "device_map": {"": int(model_gpu_device_id)},
            "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"},
        }
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw)
    dev = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v
                break
            if isinstance(v, int):
                dev = "cuda:%d" % v
                break
    return model, processor, dev


model, processor, device = load_model_s20d(
    args.model_path, model_gpu_device_id=int(args.model_gpu_device_id))
model_dtype = torch.bfloat16
unnorm_key = 'libero_object'
K_trigger = 8  # V4 default
action_dim = int(model.get_action_dim(unnorm_key))
assert action_dim == 7, f"Unexpected action_dim={action_dim}"
print('[%s] Model loaded. device=%s action_dim=%d' % (
    time.strftime('%H:%M:%S'), device, action_dim), flush=True)

# ── V4 imports ──
from v4_run_eval_openvla import (
    decode_with_scores, postprocess_openvla_action_for_libero, prompt,
    prepare_openvla_image, _model_float_dtype)

# ── Env ──
from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()
TASK_IDX = {
    'ketchup': 4, 'tomato_sauce': 5, 'milk': 7, 'butter': 6,
    'cream_cheese': 1, 'salad_dressing': 2, 'bbq_sauce': 3,
    'alphabet_soup': 0, 'orange_juice': 9, 'chocolate_pudding': 8,
}
task_idx = TASK_IDX[args.task]
task_obj = task_suite.get_task(task_idx)
init_states = task_suite.get_task_init_states(task_idx)
bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task_obj.problem_folder, task_obj.bddl_file)
instruction_raw = task_obj.language
instruction = prompt(instruction_raw)  # V4 prompt wrapping

env, obs = build_v4_exact_env(
    bddl_file, int(args.render_gpu_device_id),
    args.max_steps_override, args.num_steps_wait)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, args.num_steps_wait)

max_steps = args.max_steps_override

# ── Attack setup ──
eps_norm = args.eps_raw_pixels / 255.0
attacker = None
attacker_config = {}
if args.condition == 'online_vis_pgd':
    from gripper_attack.attack_adapter import (
        OpenVLAVisualAttacker)
    attacker_config = {
        'method': 'token_prefix_pgd',
        'epsilon': eps_norm,
        'alpha': eps_norm / args.pgd_steps * 2.5,
        'num_iter': args.pgd_steps,
        'token_label_source': 'prefix_locked_gripper_open_margin',
        'target_token_margin': 5,
        'K_trigger': K_trigger,
        'use_restart': True,
        'num_restarts': 1,
        'random_start': True,
        'target_return_first': False,
    }
    attacker = OpenVLAVisualAttacker(
        model, processor, attacker_config, device=device)
    print('[%s] V6 TokenPrefixPGD attacker ready' %
          time.strftime('%H:%M:%S'), flush=True)


def physical_gripper_state(env, obs):
    try:
        return env.env.robots[0].controller.gripper_state
    except Exception:
        return {}


# ── Helpers ──
def fmt_float(x, nd=6):
    if x in ("", None):
        return ""
    try:
        return round(float(x), nd)
    except Exception:
        return ""


# ── Online trigger state ──
trigger_found = False
trigger_step = -1
event_start = -1
event_end = -1
perturb_frame_count = 0
naturally_open_skip = 0
clean_close_history = []
eligible_close_opportunities = 0
close_streak = 0
prev_clean_open = True

# ── Episode state ──
success_primary = False
success_done_any = False
success_check_any = False
success_step_primary = -1
done_step_any = -1
total_decoded_open = 0
max_open_streak = 0
current_streak = 0
C2O_count = 0
attacked_close_count = 0
trace_rows = []
step = 0

while step < max_steps:
    if 'agentview_image' not in obs:
        infra_status = f"missing camera at step {step}"
        break

    img_uint8 = obs['agentview_image']

    # ── V4 EXACT clean decode ──
    clean_action, prefix_logits, Tclean, gen_out = decode_with_scores(
        model, processor, device,
        img_uint8, instruction, unnorm_key, K_trigger,
        libero_official_preprocess=False,
        libero_preprocess_backend='official_pil_lanczos',
        center_crop=True,
        resize_size=224,
        drop_attention_mask=True,
    )
    clean_env_action = postprocess_openvla_action_for_libero(
        clean_action, enabled=True)
    clean_gripper_raw = float(clean_action[-1])
    clean_gripper_env = float(clean_env_action[-1])
    clean_close = clean_gripper_raw < 0.5  # V4: raw<0.5 → CLOSE
    clean_open = not clean_close

    clean_close_history.append(clean_close)
    if clean_close:
        close_streak += 1
    else:
        close_streak = 0
    close_onset = clean_close and prev_clean_open
    prev_clean_open = clean_open

    # ── Online trigger check (RULE_TRIGGER_MVP) ──
    if (not trigger_found and clean_close and close_onset
            and not success_primary):
        trigger_found = True
        trigger_step = step
        event_start = step
        event_end = min(step + args.event_horizon, max_steps)
        print('[%s] TRIGGER step=%d onset=%d streak=%d' % (
            time.strftime('%H:%M:%S'), step,
            int(close_onset), close_streak), flush=True)

    # ── Qpos before step ──
    gripper_phys_before = physical_gripper_state(env, obs)
    gripper_qpos_before = float(
        np.sum(gripper_phys_before.get('qpos', [0.0])))

    # ── Attack decision ──
    in_event = trigger_found and event_start <= step < event_end
    attack_this_step = False
    pgd_applied = 0
    perturbation_space = 'none'
    env_action = clean_env_action.copy()
    executed_action = clean_action.copy()
    infra_status = 'ok'

    if in_event and clean_close and perturb_frame_count < args.max_perturb_frames:
        attack_this_step = True
        eligible_close_opportunities += 1

        if args.condition == 'online_vis_pgd' and attacker is not None:
            try:
                attack_result = attacker.attack(
                    img_uint8, instruction,
                    clean_action, clean_action, gen_out,
                    unnorm_key=unnorm_key)
                if attack_result is None:
                    raise RuntimeError(
                        "V6 HARD FAIL: attack_result is None")
                if attack_result.x_adv is None:
                    raise RuntimeError(
                        "V6 HARD FAIL: x_adv is None in attack_result")
                adv_action, _, _, _ = decode_with_scores(
                    model, processor, device,
                    attack_result.x_adv, instruction, unnorm_key, K_trigger,
                    libero_official_preprocess=False,
                    libero_preprocess_backend='official_pil_lanczos',
                    center_crop=True, resize_size=224,
                    drop_attention_mask=True)
                adv_env_action = postprocess_openvla_action_for_libero(
                    adv_action, enabled=True)
                env_action = adv_env_action
                executed_action = adv_action
                pgd_applied = 1
                perturbation_space = 'online_token_prefix_pgd_v6'
                attacked_close_count += 1
            except Exception as e:
                infra_status = 'vis_error: %s' % str(e)[:80]
                raise

        elif args.condition == 'online_random_linf':
            try:
                rand_gen = np.random.RandomState(
                    int(args.random_control_seed or args.attack_seed) + step)
                rand_action = clean_action.copy()
                rand_action[-1] += rand_gen.uniform(
                    -eps_norm, eps_norm) * 2.0
                rand_env_action = rand_action.copy()
                rand_env_action[-1] = (
                    -1.0 if rand_action[-1] > 0.5
                    else (1.0 if rand_action[-1] < -0.5 else 0.0))
                env_action = rand_env_action
                executed_action = rand_action
                pgd_applied = 2
                perturbation_space = 'online_random_linf_v6'
                attacked_close_count += 1
            except Exception as e:
                infra_status = 'rand_error: %s' % str(e)[:80]

        perturb_frame_count += 1
    elif in_event and not clean_close:
        naturally_open_skip += 1

    # ── Env step ──
    eef_before = (env.env.robots[0]._hand_pos
                  if hasattr(env.env.robots[0], '_hand_pos') else None)
    obs, reward, done, info = env.step(env_action)

    gripper_phys_after = physical_gripper_state(env, obs)
    gripper_qpos_after = float(
        np.sum(gripper_phys_after.get('qpos', [0.0])))
    is_open = int(env_action[-1] < -0.5)
    c2o_this_step = int(clean_close and is_open)
    C2O_count += c2o_this_step

    total_decoded_open += is_open
    current_streak = current_streak + 1 if is_open else 0
    max_open_streak = max(max_open_streak, current_streak)

    success_done = bool(done)
    success_check = bool(env.check_success())
    success_primary_now = (
        success_done if args.success_metric == 'done'
        else success_check)
    if success_primary_now and not success_primary:
        success_primary = True
        success_step_primary = step
    if success_done and not success_done_any:
        success_done_any = True
        done_step_any = step
    if success_check and not success_check_any:
        success_check_any = True

    # ── Trace ──
    trace_rows.append({
        'step': step, 'state_id': args.state_id, 'task': args.task,
        'condition': args.condition,
        'in_event': int(in_event),
        'attack_this_step': int(attack_this_step),
        'clean_gripper_env': round(clean_gripper_env, 6),
        'executed_gripper_env': round(float(env_action[-1]), 6),
        'decoded_open_bool': is_open,
        'gripper_qpos_before': round(gripper_qpos_before, 6),
        'gripper_qpos_after': round(gripper_qpos_after, 6),
        'physical_gripper_opening_delta': round(
            gripper_qpos_after - gripper_qpos_before, 6),
        'eef_x': round(float(eef_before[0]), 6) if eef_before is not None else '',
        'eef_y': round(float(eef_before[1]), 6) if eef_before is not None else '',
        'eef_z': round(float(eef_before[2]), 6) if eef_before is not None else '',
        'eef_z_after': '',
        'obj_x': '', 'obj_y': '', 'obj_z': '', 'obj_z_after': '',
        'target_object_name': '',
        'pgd_applied': pgd_applied,
        'perturbation_space': perturbation_space,
        'success_done': int(success_done),
        'success_check': int(success_check),
        'success_primary': int(success_primary_now),
        'attack_seed': args.attack_seed, 'job_id': args.job_id,
        'infra_status': infra_status,
        'window_start': '', 'window_end': '',
        'trigger_found': int(trigger_found),
        'trigger_step': trigger_step,
        'clean_close': int(clean_close),
        'close_onset': int(close_onset),
        'close_streak': close_streak,
        'c2o_this_step': c2o_this_step,
    })

    step += 1
    if success_primary or done:
        break

env.close()

# ── Summary ──
safe_tag = '%s_s%d_v6_%s_seed%d' % (
    args.task, args.state_id, args.condition, args.attack_seed)
summary = {
    'runner_family': 's20d_v6_online_trigger_l3',
    'vis_runner_version': 'v6_online_trigger',
    'task': args.task, 'state_id': args.state_id,
    'condition': args.condition,
    'attack_seed': args.attack_seed, 'job_id': args.job_id,
    'trigger_found': trigger_found, 'trigger_step': trigger_step,
    'close_onset': int(close_onset),
    'close_streak_at_trigger': close_streak,
    'eligible_close_opportunities': eligible_close_opportunities,
    'perturb_frame_count': perturb_frame_count,
    'naturally_open_skip': naturally_open_skip,
    'C2O_count': C2O_count,
    'attacked_close_count': attacked_close_count,
    'decoded_open_count': total_decoded_open,
    'max_open_streak': max_open_streak,
    'n_steps': step,
    'max_steps': max_steps,
    'success_primary': success_primary,
    'success_done_any': success_done_any,
    'success_check_any': success_check_any,
    'timeout': not done and not success_primary,
    'infra_status': 'ok',
    'event_horizon': args.event_horizon,
    'eps_raw_pixels': args.eps_raw_pixels,
    'pgd_steps': args.pgd_steps,
    'model_path': args.model_path,
    'decode_path': 'v4_decode_with_scores',
    'preprocess_path': 'v4_prepare_openvla_image_official_pil_lanczos_center_crop_224',
    'postprocess_path': 'v4_postprocess_openvla_action_for_libero_enabled_true',
    'attention_mask': 'v4_drop',
    'eos_token': 'v4_29871_insertion',
    'trigger_method': 'RULE_TRIGGER_MVP_clean_close_onset',
}
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(
        args.output_dir,
        'summary_%s_job%s.json' % (safe_tag, args.job_id)), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
if trace_rows:
    with open(os.path.join(
            args.output_dir,
            'trace_%s_job%s.csv' % (safe_tag, args.job_id)),
            'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader()
        w.writerows(trace_rows)

print('[%s] V6 DONE. trigger=%s step=%d C2O=%d attacked=%d/%d opens=%d/%d' % (
    time.strftime('%H:%M:%S'), trigger_found, trigger_step, C2O_count,
    attacked_close_count, eligible_close_opportunities,
    total_decoded_open, step), flush=True)
