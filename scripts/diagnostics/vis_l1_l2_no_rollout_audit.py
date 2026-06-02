#!/usr/bin/env python3
"""No-rollout action audit for L1+L2 VIS attack-strength upgrade.

Runs TokenPrefixPGD on verified contact/carry/pre-place frames across multiple
objectives and epsilon budgets. Does NOT step a LIBERO environment.

Outputs:
  tables/vis_l1_l2_no_rollout_metrics.csv
  reports/VIS_L1_L2_NO_ROLLOUT_AUDIT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'

TASK_INSTRUCTIONS = {
    'cream_cheese': 'pick up the cream cheese and place it in the basket',
    'salad_dressing': 'put the salad dressing in the basket',
    'ketchup': 'pick up the ketchup and place it in the basket',
    'tomato_sauce': 'pick up the tomato sauce and place it in the basket',
}

CSV_FIELDS = [
    'task', 'frame_label', 'objective', 'eps_raw_pixels', 'eps_processor',
    'steps', 'z_weight', 'grip_weight',
    'clean_gripper_token', 'adv_gripper_token', 'gripper_token_flipped',
    'clean_gripper_action', 'adv_gripper_action',
    'clean_z_action', 'adv_z_action',
    'nad_dof7', 'nad_z', 'nad_dof1_3', 'arm_l2',
    'target_ce_initial', 'target_ce_final',
    'open_bin_prob_mass_before', 'open_bin_prob_mass_after',
    'perturbation_linf_processor', 'perturbation_linf_raw',
    'pgd_restarts', 'attack_runtime_sec', 'error',
]


def prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def load_model(gpu_pair: str = '0,1'):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    gpu_ids = [int(x.strip()) for x in gpu_pair.split(',')]
    max_mem = {int(gpu_ids[0]): '9000MiB', int(gpu_ids[1]): '9000MiB', 'cpu': '64GiB'}
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map='auto',
        max_memory=max_mem, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor


def decode_clean_action(model, processor, img_np, instruction, device, mdtype, VS, BC, mask, low, high, action_dim):
    import torch
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    text = prompt(str(instruction).lower())
    inputs = processor(text, pil_img, return_tensors='pt')
    inputs.pop('attention_mask', None)
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v):
            inputs[k] = v.to(device=device, dtype=mdtype)
        else:
            inputs[k] = v.to(device)
    if not torch.all(inputs['input_ids'][:, -1] == 29871):
        inputs['input_ids'] = torch.cat(
            (inputs['input_ids'], torch.tensor([[29871]], dtype=torch.long, device=device)), dim=1)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    disc = np.clip(VS - token_ids - 1, 0, len(BC) - 1)
    norm_actions = BC[disc].astype(np.float32)
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)
    return action, token_ids, gen


def get_processor_stats(processor):
    """Extract image normalization stats from processor."""
    try:
        ip = processor.image_processor
        all_stds = getattr(ip, 'stds', [[0.5, 0.5, 0.5]])
        all_means = getattr(ip, 'means', [[0.5, 0.5, 0.5]])
        return all_means[-1], all_stds[-1]
    except Exception:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def eps_raw_to_processor(eps_raw_pixels, image_std):
    """Convert raw pixel Linf budget to processor-space epsilon."""
    eps_per_channel = [(eps_raw_pixels / 255.0) / float(s) for s in image_std]
    return eps_per_channel, min(eps_per_channel)


def run_one_attack(model, processor, img_np, instruction, clean_action, clean_gen,
                   clean_token_ids,  # P0-3: explicitly pass clean tokens from decode
                   objective, eps_proc, steps, step_size, z_weight, grip_weight,
                   action_dim, low, seed, device, pgd_restarts=1):
    """Run TokenPrefixPGD on one frame (with optional restarts), return metrics dict.

    P0-3: clean_token_ids comes from the caller's clean decode, NOT from adv_decoded.
    P0-5: pgd_restarts runs N PGD trajectories and picks the best by target_ce_final.
    """
    import torch
    t0 = time.time()
    error = ''
    result_row = {}
    try:
        target_action = np.asarray(clean_action, dtype=np.float32).copy()
        target_action[-1] = 1.0  # OPEN
        if objective == 'force_open_z_down_token_ce':
            target_action[2] = low[2]  # Z DOWN

        # P0-5: PGD restarts — try N random starts, keep best by final CE loss.
        best_result = None
        best_loss = float('inf')
        for restart in range(max(1, int(pgd_restarts))):
            restart_seed = int(seed) + restart * 1000
            attack_cfg = {
                'attack_optimizer': {
                    'method': 'token_prefix_pgd', 'objective': objective,
                    'epsilon': eps_proc, 'step_size': step_size, 'num_steps': steps,
                    'random_start': (restart > 0),  # first run from zero, restarts from random
                }
            }
            if objective == 'force_open_z_down_token_ce':
                attack_cfg['attack_optimizer']['loss_weights'] = {
                    str(action_dim - 1): grip_weight,
                    '2': z_weight,
                }

            attacker = TokenPrefixPGDAttacker(
                model, processor, attack_cfg, seed=restart_seed,
                preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False,
                                   'resize_size': 224, 'postprocess_gripper': True},
                device=device)

            attack_result = attacker.attack(
                observation=img_np, instruction=instruction,
                clean_action=clean_action, target_action=target_action,
                clean_model_output=clean_gen, unnorm_key=UNNORM_KEY)

            loss = (attack_result.debug or {}).get('target_ce_final', float('inf'))
            if isinstance(loss, (int, float)) and loss < best_loss:
                best_loss = loss
                best_result = attack_result

        attack_result = best_result
        adv_inputs = get_adv_inputs_from_attack_result(attack_result)
        adv_decoded = redecode_openvla_action_from_adv_inputs(
            model=model, processor=processor, adv_inputs=adv_inputs,
            instruction=instruction, unnorm_key=UNNORM_KEY)

        adv_action = np.asarray(adv_decoded.action, dtype=np.float32)
        adv_tokens = adv_decoded.token_ids
        debug = attack_result.debug or {}
        clean_audit = (debug.get('clean_logit_audit') or {}).get('action_token_logit_audit', [])
        adv_audit = (debug.get('adv_logit_audit') or {}).get('action_token_logit_audit', [])

        gripper_clean = clean_audit[-1] if clean_audit and len(clean_audit) > 0 else {}
        gripper_adv = adv_audit[-1] if adv_audit and len(adv_audit) > 0 else {}

        # P0-3: use caller-provided clean_token_ids, NOT adv_decoded.clean_token_ids.
        clean_grip_token = int(clean_token_ids[-1])
        adv_grip_token = int(adv_tokens[-1]) if adv_tokens is not None and len(adv_tokens) > 0 else -1

        linf_proc = float(debug.get('pixel_budget_adv_inputs_linf', 0.0))
        image_std = getattr(processor.image_processor, 'stds', [[0.5, 0.5, 0.5]])[-1]
        linf_raw = linf_proc * 255.0 * float(image_std[0])

        result_row = {
            'clean_gripper_token': clean_grip_token,
            'adv_gripper_token': adv_grip_token,
            'gripper_token_flipped': clean_grip_token != adv_grip_token,
            'clean_gripper_action': float(clean_action[-1]),
            'adv_gripper_action': float(adv_action[-1]),
            'clean_z_action': float(clean_action[2]),
            'adv_z_action': float(adv_action[2]),
            'nad_dof7': abs(float(adv_action[-1]) - float(clean_action[-1])),
            'nad_z': abs(float(adv_action[2]) - float(clean_action[2])),
            'nad_dof1_3': float(np.linalg.norm(adv_action[:3] - clean_action[:3])),
            'arm_l2': float(np.linalg.norm(adv_action[:6] - clean_action[:6])),
            'target_ce_initial': debug.get('target_ce_initial', None),
            'target_ce_final': debug.get('target_ce_final', None),
            'open_bin_prob_mass_before': gripper_clean.get('open_bin_prob_mass', None),
            'open_bin_prob_mass_after': gripper_adv.get('open_bin_prob_mass', None),
            'perturbation_linf_processor': linf_proc,
            'perturbation_linf_raw': round(linf_raw, 4),
            'pgd_restarts': int(pgd_restarts),
        }
    except Exception as e:
        error = str(e)[:200]

    result_row['attack_runtime_sec'] = round(time.time() - t0, 3)
    result_row['error'] = error
    return result_row


def load_frames_from_clean_policy_rollout(task_name, model, processor, device, mdtype,
                                           VS, BC, mask, low, high, action_dim, env_gpu=0):
    """Run a clean policy rollout and collect frames at every step.

    P0-4: This replaces the zero-action loader. Frames are collected during actual
    task execution with the model's clean policy. Each frame is tagged with its
    decoded gripper action and step index for phase identification (contact/carry/pre-place).
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    TASK_IDS = {
        'cream_cheese': 1, 'tomato_sauce': 5,
        'ketchup': 4, 'salad_dressing': 2,
    }
    task_id = TASK_IDS[task_name]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(task_id)

    env_args = {
        'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
        'has_renderer': False, 'has_offscreen_renderer': True,
        'use_camera_obs': True, 'camera_names': ['agentview'],
        'control_freq': 20, 'render_gpu_device_id': int(env_gpu),
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    init_state = initial_states[0]
    env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    frames = []
    instruction = TASK_INSTRUCTIONS[task_name]
    max_steps = 300
    for step in range(max_steps):
        img_np_raw = obs['agentview_image']
        # Apply the same preprocessing as vis_rollout_adaptive_v3
        img = img_np_raw[::-1, ::-1]
        img = Image.fromarray(img).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        img_np = np.array(img)

        # Decode clean action
        action, token_ids = decode_clean_action(
            model, processor, img_np, instruction, device, mdtype,
            VS, BC, mask, low, high, action_dim)

        # Normalize + invert gripper for env step (same as rollout script)
        env_action = action.copy()
        env_action[-1] = 2.0 * env_action[-1] - 1.0
        env_action[-1] = np.sign(env_action[-1])
        env_action[-1] = 1.0 if env_action[-1] == 0 else env_action[-1]
        env_action[-1] = -1.0 * env_action[-1]

        obs, reward, done, info = env.step(env_action)

        frames.append({
            'task': task_name,
            'frame_label': f'{task_name}_step{step:03d}',
            'step': step,
            'image': img_np,
            'instruction': instruction,
            'clean_action': action,
            'clean_token_ids': token_ids,
            'decoded_gripper': float(action[-1]),
            'env_gripper': float(env_action[-1]),
            'gripper_qpos': float(obs['robot0_gripper_qpos'][0]) if len(obs['robot0_gripper_qpos']) > 0 else 0.0,
            'done': bool(done),
        })
        if done:
            break
    env.close()
    return frames


def load_frames_from_task_zero_action(task_name, env_gpu=0):
    """DEPRECATED: smoke-test only — collects frames with zero action (robot not moving).

    This does NOT produce real contact/carry/pre-place frames. Use
    load_frames_from_clean_policy_rollout for any gating decision.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    TASK_IDS = {
        'cream_cheese': 1, 'tomato_sauce': 5,
        'ketchup': 4, 'salad_dressing': 2,
    }
    task_id = TASK_IDS[task_name]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(task_id)

    env_args = {
        'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
        'has_renderer': False, 'has_offscreen_renderer': True,
        'use_camera_obs': True, 'camera_names': ['agentview'],
        'control_freq': 20, 'render_gpu_device_id': int(env_gpu),
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    init_state = initial_states[0]
    env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    frames = []
    instruction = TASK_INSTRUCTIONS[task_name]
    for step in range(300):
        img = obs['agentview_image']
        img = img[::-1, ::-1]
        img = Image.fromarray(img).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        img_np = np.array(img)
        frames.append({
            'task': task_name,
            'frame_label': f'{task_name}_zero_step{step:03d}',
            'step': step,
            'image': img_np,
            'instruction': instruction,
            'clean_action': None,
            'clean_token_ids': None,
        })
        obs, _, done, _ = env.step(np.zeros(7))
        if done:
            break
    env.close()
    return frames


def main():
    ap = argparse.ArgumentParser(description='VIS L1+L2 No-Rollout Action Audit')
    ap.add_argument('--task', choices=list(TASK_INSTRUCTIONS.keys()), nargs='+',
                    default=['cream_cheese', 'salad_dressing', 'ketchup'],
                    help='tasks to audit')
    ap.add_argument('--objectives', nargs='+',
                    default=['gripper_open_region_ce', 'force_open_z_down_token_ce'])
    ap.add_argument('--eps_raw_list', type=int, nargs='+', default=[4, 8, 12])
    ap.add_argument('--steps_list', type=int, nargs='+', default=[20, 40])
    ap.add_argument('--z_weights', type=float, nargs='+', default=[0.25, 0.5])
    ap.add_argument('--gpu_pair', default='0,1')
    ap.add_argument('--frame_source', choices=['clean_policy_rollout', 'zero_action'],
                    default='clean_policy_rollout',
                    help='clean_policy_rollout (default) = real task frames; zero_action = smoke test only')
    ap.add_argument('--pgd_restarts', type=int, default=3,
                    help='PGD random restarts for audit (default: 3)')
    ap.add_argument('--frame_step_stride', type=int, default=5,
                    help='sample every N steps (default: 5)')
    ap.add_argument('--max_frames_per_task', type=int, default=20,
                    help='max frames per task')
    ap.add_argument('--output_dir', default=str(REPO_ROOT / 'tables'))
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    import torch
    print('=== VIS L1+L2 No-Rollout Action Audit ===')
    print(f'Tasks: {args.task}')
    print(f'Frame source: {args.frame_source}')
    print(f'Objectives: {args.objectives}')
    print(f'EPS raw: {args.eps_raw_list}')
    print(f'Steps: {args.steps_list}')
    print(f'Z weights: {args.z_weights}')
    print(f'PGD restarts: {args.pgd_restarts}')

    if args.dry_run:
        total = len(args.task) * len(args.objectives) * len(args.eps_raw_list) * len(args.steps_list) * len(args.z_weights) * args.max_frames_per_task
        print(f'[DRY RUN] Would run {total} attack combinations')
        return

    model, processor = load_model(args.gpu_pair)
    device = str(next(model.parameters()).device)
    mdtype = next(model.parameters()).dtype
    VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    BC = np.array(model.bin_centers)
    action_dim = int(model.get_action_dim(UNNORM_KEY))
    stats = model.get_action_stats(UNNORM_KEY)
    mask = np.array(stats.get('mask', np.ones_like(stats['q01'], dtype=bool)))
    low = np.array(stats['q01'])
    high = np.array(stats['q99'])
    image_mean, image_std = get_processor_stats(processor)

    print(f'device={device} dtype={mdtype} action_dim={action_dim}')
    print(f'image_mean={image_mean} image_std={image_std}')

    env_gpu = int(args.gpu_pair.split(',')[0])
    all_rows = []
    for task_name in args.task:
        print(f'\n--- {task_name} ---')
        if args.frame_source == 'clean_policy_rollout':
            # P0-4: run clean policy rollout to get real contact/carry/pre-place frames
            frames = load_frames_from_clean_policy_rollout(
                task_name, model, processor, device, mdtype,
                VS, BC, mask, low, high, action_dim, env_gpu=env_gpu)
            # Use pre-computed clean_action/token_ids from the rollout
            use_precomputed_clean = True
        else:
            frames = load_frames_from_task_zero_action(task_name, env_gpu=env_gpu)
            use_precomputed_clean = False
            print('  WARNING: zero_action frames — smoke test only, NOT for gate decisions')

        # Sample frames: skip early setup steps, sample evenly
        sampled = frames[30::args.frame_step_stride][:args.max_frames_per_task]
        print(f'  {len(frames)} total frames, {len(sampled)} sampled')

        for frame_data in sampled:
            img_np = frame_data['image']
            instruction = frame_data['instruction']

            # P0-3 + P0-4: use pre-computed clean actions from rollout, or decode fresh
            if use_precomputed_clean and frame_data.get('clean_action') is not None:
                clean_action = frame_data['clean_action']
                clean_tokens = frame_data['clean_token_ids']
                # Need clean_gen for attack — decode it (lightweight: model.generate)
                clean_action_redecoded, _, clean_gen = decode_clean_action(
                    model, processor, img_np, instruction, device, mdtype,
                    VS, BC, mask, low, high, action_dim)
            else:
                clean_action, clean_tokens, clean_gen = decode_clean_action(
                    model, processor, img_np, instruction, device, mdtype,
                    VS, BC, mask, low, high, action_dim)

            for objective in args.objectives:
                for eps_raw in args.eps_raw_list:
                    eps_per_ch, eps_proc = eps_raw_to_processor(eps_raw, image_std)
                    for steps in args.steps_list:
                        step_size = eps_proc / 8.0
                        z_weights_iter = args.z_weights if objective == 'force_open_z_down_token_ce' else [0.0]
                        for z_w in z_weights_iter:
                            grip_w = 1.0
                            # P0-3: pass clean_token_ids explicitly
                            # P0-5: pass pgd_restarts
                            result = run_one_attack(
                                model, processor, img_np, instruction,
                                clean_action, clean_gen, clean_tokens,
                                objective, eps_proc, steps, step_size, z_w, grip_w,
                                action_dim, low, seed=42, device=device,
                                pgd_restarts=args.pgd_restarts)

                            row = {
                                'task': task_name,
                                'frame_label': frame_data['frame_label'],
                                'objective': objective,
                                'eps_raw_pixels': eps_raw,
                                'eps_processor': round(eps_proc, 6),
                                'steps': steps,
                                'z_weight': z_w,
                                'grip_weight': grip_w,
                            }
                            row.update(result)
                            all_rows.append(row)

                            flipped = 'FLIP' if row.get('gripper_token_flipped') else 'noop'
                            print(f'  {frame_data["frame_label"]} obj={objective} eps={eps_raw} steps={steps} '
                                  f'z_w={z_w} | grip_flip={flipped} nad_z={row["nad_z"]:.4f} '
                                  f'armL2={row["arm_l2"]:.4f} linf_raw={row["perturbation_linf_raw"]} '
                                  f't={row["attack_runtime_sec"]:.1f}s')

    # Write CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'vis_l1_l2_no_rollout_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} rows to {csv_path}')

    # Summary
    for objective in args.objectives:
        obj_rows = [r for r in all_rows if r['objective'] == objective and not r.get('error')]
        flips = sum(1 for r in obj_rows if r.get('gripper_token_flipped'))
        avg_nad_z = np.mean([r['nad_z'] for r in obj_rows]) if obj_rows else 0
        avg_nad_dof7 = np.mean([r['nad_dof7'] for r in obj_rows]) if obj_rows else 0
        print(f'{objective}: {flips}/{len(obj_rows)} token flips '
              f'({100*flips/max(len(obj_rows),1):.1f}%) '
              f'avg NAD_Z={avg_nad_z:.4f} avg NAD_DoF7={avg_nad_dof7:.4f}')


if __name__ == '__main__':
    main()
