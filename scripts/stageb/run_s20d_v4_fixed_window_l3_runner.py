#!/usr/bin/env python3
"""S20d: V4-based official-eval-aligned fixed-window Level-3 runner.
Imports critical functions from v4_run_eval_openvla.py to preserve exact V4 behavior.
Phase 1: clean-only clone. Phase 3 adds fixed-window attack.
"""
from __future__ import annotations
import os, sys, argparse, json, csv, time
from pathlib import Path
import numpy as np, torch
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from v4_run_eval_openvla import (
    load_model, decode_with_scores, postprocess_openvla_action_for_libero,
    physical_gripper_state, prompt,
)

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True, choices=[
    'ketchup','tomato_sauce','milk','butter','cream_cheese','salad_dressing',
    'bbq_sauce','alphabet_soup','orange_juice','chocolate_pudding'])
ap.add_argument('--state_ids', default='0', help='comma-separated state ids')
ap.add_argument('--condition', choices=['clean','vis_pgd','random_linf'], default='clean')
ap.add_argument('--window_start', type=int, default=0)
ap.add_argument('--window_end', type=int, default=10)
ap.add_argument('--attack_seed', type=int, default=0)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--random_control_seed', type=int, default=None)
ap.add_argument('--max_steps_override', type=int, default=280)
ap.add_argument('--num_steps_wait', type=int, default=10)
ap.add_argument('--success_metric', choices=['done','check_success'], default='done')
ap.add_argument('--output_dir', required=True)
ap.add_argument('--save_video_dir', default='')
ap.add_argument('--job_id', type=int, default=0)
ap.add_argument('--pair_id', default='')
ap.add_argument('--model_path', default='/data/aviary/models/openvla/openvla-7b-finetuned-libero-object')
ap.add_argument('--model_gpu_device_id', type=int, default=-1)
ap.add_argument('--render_gpu_device_id', type=int, default=0)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

_eps_eff = args.eps_raw_pixels / 255.0
state_ids = [int(x.strip()) for x in args.state_ids.split(',') if x.strip()]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)

os.environ.setdefault("OPENVLA_RENDER_LOCAL_DEVICE", str(args.render_gpu_device_id))

# ── Model loading (EXACT V4 path) ──
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model from {args.model_path}", flush=True)
model, processor, device = load_model(args.model_path, model_gpu_device_id=args.model_gpu_device_id)
model_dtype = next(model.parameters()).dtype
print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded on {device} dtype={model_dtype}", flush=True)

# ── LIBERO env setup (V4 pattern) ──
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_IDX = {
    'ketchup': 4, 'tomato_sauce': 5, 'milk': 7, 'butter': 6,
    'cream_cheese': 1, 'salad_dressing': 2, 'bbq_sauce': 3,
    'alphabet_soup': 0, 'orange_juice': 9, 'chocolate_pudding': 8,
}

bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()
task_idx = TASK_IDX[args.task]
task_obj = task_suite.get_task(task_idx)
init_states = task_suite.get_task_init_states(task_idx)
instruction = task_obj.language
bddl_file = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

unnorm_key = 'libero_object'
K_trigger = 8  # match V4 default uncertainty K

action_dim = int(model.get_action_dim(unnorm_key))

os.makedirs(args.output_dir, exist_ok=True)
video_enabled = bool(args.save_video_dir)
if video_enabled:
    os.makedirs(args.save_video_dir, exist_ok=True)

# ── Attack setup (Phase 3, safe import) ──
attacker = None
if args.condition in ('vis_pgd',):
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    attacker_config = {
        'epsilon': _eps_eff, 'step_size': _eps_eff / max(args.pgd_steps, 1) * 1.5,
        'num_steps': args.pgd_steps, 'random_start': True,
        'objective': 'prefix_locked_gripper_open_margin',
        'arm_preserve_weight': 0.5, 'gripper_margin': 5.0,
    }
    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor, config={'attack_optimizer': attacker_config,
            'directional_target': {'direction_id': 'gripper_open', 'dims': list(range(action_dim))},
            'uncertainty': {'K_trigger': K_trigger}},
        direction_spec={'g_hat': np.zeros(action_dim, dtype=np.float32), 'dims': list(range(action_dim))},
        seed=args.attack_seed,
        preprocess_kwargs={'libero_official_preprocess': False,
                          'libero_preprocess_backend': 'official_pil_lanczos',
                          'center_crop': True, 'resize_size': 224,
                          'postprocess_gripper': True},
        device=device)

# ── Run each state ──
for sid in state_ids:
    if sid >= len(init_states):
        print(f'state_id {sid} out of range (max {len(init_states)-1})')
        continue

    safe_tag = f"{args.task}_s{sid}_w{args.window_start}_{args.window_end}_s20d_{args.condition}_seed{args.attack_seed}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe_tag}", flush=True)

    # V4 env construction
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256, camera_widths=256,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=['agentview'],
        control_freq=20,
        render_gpu_device_id=args.render_gpu_device_id,
        horizon=args.max_steps_override + args.num_steps_wait)

    # V4: env.seed(0) hardcoded
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[sid])

    # V4 wait steps
    if args.num_steps_wait > 0:
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)

    trace_rows = []
    qpos_history = []
    success_done_any = False; success_check_any = False
    success_step_primary = -1; done_step_any = -1
    infra_status = 'ok'
    ws = args.window_start; we = args.window_end
    max_steps = args.max_steps_override

    for step in range(max_steps):
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
        clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)

        # Qpos before step (V4 pattern via physical_gripper_state)
        gripper_phys_before = physical_gripper_state(env, obs)
        gripper_qpos_before = float(np.sum(gripper_phys_before.get('qpos', [0.0])))
        qpos_history.append(gripper_qpos_before)

        # ── Fixed-window attack injection ──
        in_window = ws <= step < we
        attack_this_step = in_window and args.condition != 'clean'

        pgd_applied = 0
        random_seed_str = ''; random_seed_mode = 'n/a'
        perturbation_space = 'none'
        env_action = clean_env_action.copy()
        executed_action = clean_action.copy()

        if attack_this_step:
            if args.condition == 'vis_pgd' and attacker is not None:
                try:
                    # Use V4 attacker approach: attacker.attack returns AttackResult
                    # We then decode through V4's decode_with_scores using the adversarial image
                    attack_result = attacker.attack(
                        img_uint8, instruction,
                        clean_action, None, gen_out,
                        unnorm_key=unnorm_key)
                    if attack_result is not None and attack_result.x_adv is not None:
                        adv_img = attack_result.x_adv
                        adv_action, _, _, _ = decode_with_scores(
                            model, processor, device,
                            adv_img, instruction, unnorm_key, K_trigger,
                            libero_official_preprocess=False,
                            libero_preprocess_backend='official_pil_lanczos',
                            center_crop=True, resize_size=224,
                            drop_attention_mask=True)
                        adv_env_action = postprocess_openvla_action_for_libero(adv_action, enabled=True)
                        env_action = adv_env_action
                        executed_action = adv_action
                        pgd_applied = 1
                        perturbation_space = 'vis_pgd_v4_decode'
                except Exception as e:
                    infra_status = f'pgd_error: {str(e)[:80]}'

            elif args.condition == 'random_linf':
                try:
                    # Apply Linf noise in pixel_values space, then decode through V4 path
                    from v4_run_eval_openvla import _model_float_dtype
                    from gripper_attack.openvla_preprocess import prepare_openvla_image

                    # Prepare image exactly as V4 does
                    prep_img = prepare_openvla_image(
                        img_uint8,
                        libero_official_preprocess=False,
                        center_crop=True,
                        resize_size=224,
                        libero_preprocess_backend='official_pil_lanczos')
                    processor_prompt = prompt(instruction.lower())
                    inp = processor(processor_prompt, prep_img, return_tensors='pt')
                    inp.pop('attention_mask', None)

                    pv_clean = inp['pixel_values'].to(device=device, dtype=model_dtype)
                    input_ids_val = inp['input_ids'].to(device=device)

                    # EOS token insertion (V4 pattern)
                    if not torch.all(input_ids_val[:, -1] == 29871):
                        input_ids_val = torch.cat(
                            (input_ids_val,
                             torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids_val.device)),
                            dim=1)

                    # Determine seed for reproducibility
                    if args.random_control_seed is not None:
                        random_seed_str = str(args.random_control_seed)
                        random_seed_mode = 'explicit_random_control_seed'
                    else:
                        random_seed_str = str(args.attack_seed + args.job_id)
                        random_seed_mode = 'legacy_attack_seed_plus_job_id'

                    rand_gen = torch.Generator(device=pv_clean.device)
                    rand_gen.manual_seed(int(random_seed_str))
                    noise = (2 * torch.rand(pv_clean.shape, device=pv_clean.device,
                                           dtype=pv_clean.dtype,
                                           generator=rand_gen) - 1) * _eps_eff
                    rand_pv = torch.clamp(pv_clean + noise, 0.0, 1.0)

                    action_dim_val = int(model.get_action_dim(unnorm_key))
                    with torch.inference_mode():
                        gen_rand = model.generate(
                            input_ids=input_ids_val,
                            pixel_values=rand_pv,
                            max_new_tokens=action_dim_val,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=False)
                    rand_tids = gen_rand.sequences[0, -action_dim_val:].cpu().numpy()

                    # V4 decode from token IDs
                    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
                    discretized = np.clip(vocab_size - rand_tids - 1, 0, model.bin_centers.shape[0] - 1)
                    norm_actions = model.bin_centers[discretized]
                    action_stats = model.get_action_stats(unnorm_key)
                    mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
                    high, low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
                    rand_action = np.where(mask,
                                           0.5 * (norm_actions + 1) * (high - low) + low,
                                           norm_actions).astype(np.float32)
                    rand_env_action = postprocess_openvla_action_for_libero(rand_action, enabled=True)
                    env_action = rand_env_action
                    executed_action = rand_action
                    perturbation_space = 'random_linf_v4'

                except Exception as e:
                    infra_status = f'random_error: {str(e)[:80]}'

        # LibERO OPEN command convention: env_action[-1] < -0.5 means OPEN
        is_open = int(env_action[-1] < -0.5)

        # ── Step environment ──
        obs, reward, done, info = env.step(env_action)

        # Qpos after step
        gripper_phys_after = physical_gripper_state(env, obs)
        gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))

        # V4 success tracking
        success_check = bool(env.check_success())
        success_done = bool(done)
        success_primary = success_done if args.success_metric == 'done' else success_check

        if success_done and not success_done_any:
            success_done_any = True; done_step_any = step
        if success_check and not success_check_any:
            success_check_any = True
        if success_primary and success_step_primary < 0:
            success_step_primary = step

        # Video frame
        video_frame_path = ''
        if video_enabled:
            from PIL import Image
            pil_img = Image.fromarray(img_uint8)
            video_frame_path = os.path.join(args.save_video_dir, f'frame_{step:06d}.png')
            pil_img.save(video_frame_path)

        trace_rows.append({
            'step': step,
            'state_id': sid,
            'task': args.task,
            'condition': args.condition,
            'in_window': int(in_window),
            'attack_this_step': int(attack_this_step),
            'clean_gripper_env': round(float(clean_env_action[-1]), 6),
            'executed_gripper_env': round(float(env_action[-1]), 6),
            'decoded_open_bool': is_open,
            'gripper_qpos_before': round(gripper_qpos_before, 8),
            'gripper_qpos_after': round(gripper_qpos_after, 8),
            'physical_gripper_opening_delta': round(gripper_qpos_after - gripper_qpos_before, 8),
            'pgd_applied': pgd_applied,
            'random_seed_str': random_seed_str,
            'random_seed_mode': random_seed_mode,
            'perturbation_space': perturbation_space,
            'success_done': int(success_done),
            'success_check': int(success_check),
            'success_primary': int(success_primary),
            'reward': round(float(reward), 6) if isinstance(reward, (int, float)) else 0,
            'attack_seed': args.attack_seed,
            'job_id': args.job_id,
            'infra_status': infra_status,
            'window_start': ws, 'window_end': we,
        })

        if success_primary or done:
            break

    env.close()
    torch.cuda.empty_cache()

    # ── Window metrics ──
    open_count = sum(1 for i in range(ws, min(we, len(trace_rows)))
                     if trace_rows[i].get('decoded_open_bool', 0))
    streak = max_streak = 0
    for i in range(ws, min(we, len(trace_rows))):
        if trace_rows[i].get('decoded_open_bool', 0):
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0

    pre_qpos = np.array(qpos_history[:ws]) if ws > 0 else np.array([0.0])
    baseline_qpos = float(np.median(pre_qpos)) if len(pre_qpos) > 0 else 0.0
    post_start = we
    post_end = min(len(qpos_history), we + 40)
    post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([])
    qpos_pos_area = float(np.sum(np.maximum(post_qpos - baseline_qpos, 0))) if len(post_qpos) > 0 else 0.0

    n_steps = len(trace_rows)

    summary = {
        'runner_family': 's20d_v4_fixed_window_l3',
        'job_id': args.job_id,
        'task': args.task,
        'state_id': sid,
        'condition': args.condition,
        'window_start': ws, 'window_end': we,
        'attack_seed': args.attack_seed,
        'random_control_seed': args.random_control_seed,
        'n_steps': n_steps,
        'max_steps': max_steps,
        'num_steps_wait': args.num_steps_wait,
        'success_primary': success_primary,
        'success_primary_metric': args.success_metric,
        'success_done_any': success_done_any,
        'success_check_any': success_check_any,
        'success_step_primary': success_step_primary,
        'done_step': done_step_any,
        'timeout': n_steps >= max_steps and not success_primary,
        'decoded_open_count': open_count,
        'max_open_streak': max_streak,
        'qpos_pos_area': round(qpos_pos_area, 8),
        'qpos_baseline': round(baseline_qpos, 8),
        'infra_status': infra_status,
        'video_dir': args.save_video_dir,
        'model_path': args.model_path,
        'decode_path': 'v4_decode_with_scores',
        'postprocess_path': 'v4_postprocess_openvla_action_for_libero',
        'image_preprocess': 'v4_prepare_openvla_image_official_pil_lanczos_center_crop_224',
        'eos_token': 'v4_29871_insertion',
        'attention_mask': 'v4_drop',
        'dtype': str(model_dtype),
    }

    out_json = os.path.join(args.output_dir,
                            f'summary_{safe_tag}_job{args.job_id}.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    if trace_rows:
        out_trace = os.path.join(args.output_dir,
                                 f'trace_{safe_tag}_job{args.job_id}.csv')
        with open(out_trace, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            for r in trace_rows:
                w.writerow(r)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done: state={sid} "
          f"steps={n_steps} primary_success={success_primary}@step{success_step_primary} "
          f"done={success_done_any} check_success={success_check_any} "
          f"open={open_count} streak={max_streak} infra={infra_status}", flush=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] S20d runner done", flush=True)
