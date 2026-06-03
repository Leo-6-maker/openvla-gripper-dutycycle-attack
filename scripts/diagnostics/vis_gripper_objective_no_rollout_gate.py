#!/usr/bin/env python3
"""No-rollout gate for gripper-specific objectives on selected frames.

Supports --frame_steps to target specific clean-policy frames (e.g. step030).
Reuses model loading, clean rollout, PGD attack, and redecode from vis_l1_l2_no_rollout_audit.

Output: tables/vis_gripper_objective_no_rollout_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs
from gripper_attack.gripper_semantics import raw_gripper_is_open

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'

TASK_INSTRUCTIONS = {
    'cream_cheese': 'pick up the cream cheese and place it in the basket',
    'salad_dressing': 'put the salad dressing in the basket',
    'ketchup': 'pick up the ketchup and place it in the basket',
    'tomato_sauce': 'pick up the tomato sauce and place it in the basket',
}

TASK_IDS = {'cream_cheese': 1, 'salad_dressing': 2, 'ketchup': 4, 'tomato_sauce': 5}

ALL_OBJECTIVES = [
    'gripper_open_region_ce',
    'prefix_locked_gripper_open_region_ce',
    'prefix_locked_gripper_open_margin',
    'gripper_open_expected_action',
    'force_open_z_down_token_ce',
]

CSV_FIELDS = [
    'task', 'frame_label', 'step', 'objective', 'eps_raw_pixels', 'eps_processor',
    'steps', 'pgd_restarts', 'arm_preserve_weight', 'gripper_margin',
    'clean_gripper_token', 'adv_gripper_token', 'gripper_token_flipped',
    'clean_gripper_action', 'adv_gripper_action', 'gripper_delta',
    'nad_dof7', 'nad_z', 'nad_dof1_3', 'arm_l2',
    'open_region_prob_mass_before', 'open_region_prob_mass_after',
    'gripper_margin_before', 'gripper_margin_after',
    'target_ce_initial', 'target_ce_final',
    'perturbation_linf_processor', 'perturbation_linf_raw',
    'attack_runtime_sec', 'error',
]


def prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def load_model(gpu_pair: str = '0,1'):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    gpu_ids = [int(x.strip()) for x in gpu_pair.split(',')]
    max_mem = {gpu_ids[0]: '9000MiB', gpu_ids[1]: '9000MiB', 'cpu': '64GiB'}
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map='auto',
        max_memory=max_mem, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor


def get_processor_stats(processor):
    try:
        ip = processor.image_processor
        all_stds = getattr(ip, 'stds', [[0.5, 0.5, 0.5]])
        all_means = getattr(ip, 'means', [[0.5, 0.5, 0.5]])
        return all_means[-1], all_stds[-1]
    except Exception:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def eps_raw_to_processor(eps_raw_pixels, image_std):
    eps_per_channel = [(eps_raw_pixels / 255.0) / float(s) for s in image_std]
    return eps_per_channel, min(eps_per_channel)


def collect_clean_frames(task_name, model, processor, device, mdtype,
                          VS, BC, mask, low, high, action_dim, target_steps, env_gpu=0):
    """Run clean policy rollout, return only frames at target_steps."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

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
    env.set_init_state(initial_states[0])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    import torch
    instruction = TASK_INSTRUCTIONS[task_name]
    target_set = set(int(s) for s in target_steps)
    max_needed = max(target_set) + 1 if target_set else 300
    collected = {}
    step = 0
    while step < max_needed and step < 300:
        img = obs['agentview_image'][::-1, ::-1]
        img = Image.fromarray(img).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        img_np = np.array(img)

        qpos = float(obs['robot0_gripper_qpos'][0]) if len(obs['robot0_gripper_qpos']) > 0 else 0.0

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

        if step in target_set:
            collected[step] = {
                'task': task_name,
                'frame_label': f'{task_name}_step{step:03d}',
                'step': step,
                'image': img_np,
                'instruction': instruction,
                'clean_action': action,
                'clean_token_ids': token_ids,
                'clean_gen': gen,
                'clean_gripper_action': float(action[-1]),
                'gripper_qpos': qpos,
            }
            print(f'    collected step {step}: grip_act={float(action[-1]):.4f} qpos={qpos:.5f}')
            if len(collected) >= len(target_set):
                break

        env_action = action.copy()
        env_action[-1] = 2.0 * env_action[-1] - 1.0
        env_action[-1] = np.sign(env_action[-1])
        env_action[-1] = 1.0 if env_action[-1] == 0 else env_action[-1]
        env_action[-1] = -1.0 * env_action[-1]
        obs, reward, done, info = env.step(env_action)
        if done:
            break
        step += 1
    env.close()
    return list(collected.values())


def run_one_attack_with_restarts(model, processor, frame_data, objective,
                                  eps_proc, steps, step_size, pgd_restarts,
                                  arm_preserve_weight, gripper_margin,
                                  action_dim, low, image_std, device, mdtype):
    """Run TokenPrefixPGD with restarts on one frame. Returns metrics dict."""
    import torch
    t0 = time.time()
    error = ''
    result_row = {}
    img_np = frame_data['image']
    instruction = frame_data['instruction']
    clean_action = frame_data['clean_action']
    clean_gen = frame_data['clean_gen']
    clean_token_ids = frame_data['clean_token_ids']

    try:
        target_action = np.asarray(clean_action, dtype=np.float32).copy()
        target_action[-1] = 1.0  # OPEN gripper
        if objective == 'force_open_z_down_token_ce':
            target_action[2] = low[2]  # Z DOWN

        is_prefix_locked = objective.startswith('prefix_locked_') or objective == 'gripper_open_expected_action'
        is_gripper_obj = is_prefix_locked or objective in {'gripper_open_region_ce', 'force_open_z_down_token_ce'}

        best_result = None
        best_score = float('-inf')
        for restart in range(max(1, int(pgd_restarts))):
            restart_seed = 42 + restart * 1000
            attack_cfg = {
                'attack_optimizer': {
                    'method': 'token_prefix_pgd', 'objective': objective,
                    'epsilon': eps_proc, 'step_size': step_size, 'num_steps': steps,
                    'random_start': (restart > 0),
                }
            }
            if objective == 'force_open_z_down_token_ce':
                attack_cfg['attack_optimizer']['loss_weights'] = {
                    str(action_dim - 1): 1.0, '2': 0.5}
            if is_prefix_locked:
                attack_cfg['attack_optimizer']['arm_preserve_weight'] = float(arm_preserve_weight)
                attack_cfg['attack_optimizer']['gripper_margin'] = float(gripper_margin)
                attack_cfg['attack_optimizer']['best_restart_metric'] = 'gripper_open_prob_mass'

            attacker = TokenPrefixPGDAttacker(
                model, processor, attack_cfg, seed=restart_seed,
                preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False,
                                   'resize_size': 224, 'postprocess_gripper': True},
                device=device)

            attack_result = attacker.attack(
                observation=img_np, instruction=instruction,
                clean_action=clean_action, target_action=target_action,
                clean_model_output=clean_gen, unnorm_key=UNNORM_KEY)

            debug = attack_result.debug or {}
            score = float(debug.get('gripper_open_prob_mass',
                         debug.get('open_region_prob_mass_after', 0.0)) or 0.0)
            if score > best_score:
                best_score = score
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
        g_clean = clean_audit[-1] if clean_audit else {}
        g_adv = adv_audit[-1] if adv_audit else {}

        clean_grip_token = int(clean_token_ids[-1])
        adv_grip_token = int(adv_tokens[-1]) if adv_tokens is not None and len(adv_tokens) > 0 else -1

        linf_proc = float(debug.get('pixel_budget_adv_inputs_linf', 0.0))
        linf_raw = linf_proc * 255.0 * float(image_std[0])

        open_mass_before = g_clean.get('open_bin_prob_mass', 0.0) or 0.0
        open_mass_after = g_adv.get('open_bin_prob_mass', 0.0) or 0.0
        margin_before = float(open_mass_before) - float(g_clean.get('close_bin_prob_mass', 0.0) or 0.0)
        margin_after = float(open_mass_after) - float(g_adv.get('close_bin_prob_mass', 0.0) or 0.0)

        result_row = {
            'clean_gripper_token': clean_grip_token,
            'adv_gripper_token': adv_grip_token,
            'gripper_token_flipped': clean_grip_token != adv_grip_token,
            'clean_gripper_action': float(clean_action[-1]),
            'adv_gripper_action': float(adv_action[-1]),
            'gripper_delta': float(adv_action[-1]) - float(clean_action[-1]),
            'nad_dof7': abs(float(adv_action[-1]) - float(clean_action[-1])),
            'nad_z': abs(float(adv_action[2]) - float(clean_action[2])),
            'nad_dof1_3': float(np.linalg.norm(adv_action[:3] - clean_action[:3])),
            'arm_l2': float(np.linalg.norm(adv_action[:6] - clean_action[:6])),
            'open_region_prob_mass_before': round(float(open_mass_before), 6),
            'open_region_prob_mass_after': round(float(open_mass_after), 6),
            'gripper_margin_before': round(float(margin_before), 6),
            'gripper_margin_after': round(float(margin_after), 6),
            'target_ce_initial': debug.get('target_ce_initial', None),
            'target_ce_final': debug.get('target_ce_final', None),
            'perturbation_linf_processor': linf_proc,
            'perturbation_linf_raw': round(linf_raw, 4),
        }
    except Exception as e:
        error = str(e)[:200]

    result_row['attack_runtime_sec'] = round(time.time() - t0, 3)
    result_row['error'] = error
    return result_row


def main():
    ap = argparse.ArgumentParser(description='VIS Gripper Objective No-Rollout Gate')
    ap.add_argument('--task', choices=list(TASK_INSTRUCTIONS.keys()), nargs='+',
                    default=['cream_cheese'])
    ap.add_argument('--frame_steps', type=int, nargs='+', default=[30],
                    help='specific clean-policy steps to attack (default: 30)')
    ap.add_argument('--objectives', nargs='+', default=['gripper_open_region_ce',
        'prefix_locked_gripper_open_region_ce', 'prefix_locked_gripper_open_margin',
        'gripper_open_expected_action'])
    ap.add_argument('--eps_raw_list', type=int, nargs='+', default=[8])
    ap.add_argument('--steps_list', type=int, nargs='+', default=[40])
    ap.add_argument('--pgd_restarts', type=int, default=3)
    ap.add_argument('--arm_preserve_weight', type=float, default=0.1)
    ap.add_argument('--gripper_margin', type=float, default=5.0)
    ap.add_argument('--gpu_pair', default='0,1')
    ap.add_argument('--output_dir', default=str(REPO_ROOT / 'tables'))
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        total = len(args.task) * len(args.frame_steps) * len(args.objectives) * len(args.eps_raw_list) * len(args.steps_list)
        print(f'[DRY RUN] Would run {total} attack combinations on {args.frame_steps}')
        return

    import torch
    print('=== VIS Gripper Objective No-Rollout Gate ===')
    print(f'Tasks: {args.task}')
    print(f'Target steps: {args.frame_steps}')
    print(f'Objectives: {args.objectives}')
    print(f'EPS raw: {args.eps_raw_list}')
    print(f'Steps: {args.steps_list}')
    print(f'PGD restarts: {args.pgd_restarts}')
    print(f'Arm preserve weight: {args.arm_preserve_weight}')
    print(f'Gripper margin: {args.gripper_margin}')

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
        print(f'\n=== {task_name} ===')
        frames = collect_clean_frames(
            task_name, model, processor, device, mdtype,
            VS, BC, mask, low, high, action_dim,
            args.frame_steps, env_gpu=env_gpu)
        print(f'  Collected {len(frames)}/{len(args.frame_steps)} frames')

        for frame_data in frames:
            for objective in args.objectives:
                for eps_raw in args.eps_raw_list:
                    eps_per_ch, eps_proc = eps_raw_to_processor(eps_raw, image_std)
                    for steps in args.steps_list:
                        step_size = eps_proc / 8.0
                        result = run_one_attack_with_restarts(
                            model, processor, frame_data, objective,
                            eps_proc, steps, step_size, args.pgd_restarts,
                            args.arm_preserve_weight, args.gripper_margin,
                            action_dim, low, image_std, device, mdtype)

                        row = {
                            'task': task_name,
                            'frame_label': frame_data['frame_label'],
                            'step': frame_data['step'],
                            'objective': objective,
                            'eps_raw_pixels': eps_raw,
                            'eps_processor': round(eps_proc, 6),
                            'steps': steps,
                            'pgd_restarts': args.pgd_restarts,
                            'arm_preserve_weight': args.arm_preserve_weight,
                            'gripper_margin': args.gripper_margin,
                        }
                        row.update(result)
                        all_rows.append(row)

                        flipped = 'FLIP' if row.get('gripper_token_flipped') else 'noop'
                        _adv_grip = row.get('adv_gripper_action', 1.0)
                        is_open_mark = 'OPEN' if row.get('gripper_is_open', raw_gripper_is_open(float(_adv_grip))) else 'noop'
                        print(f'  {frame_data["frame_label"]} obj={objective} eps={eps_raw} '
                              f'| {flipped} {is_open_mark} nad_dof7={row.get("nad_dof7", 0):.4f} '
                              f'armL2={row.get("arm_l2", 0):.4f} open_after={row.get("open_region_prob_mass_after", 0)} '
                              f't={row.get("attack_runtime_sec", 0):.1f}s')

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'vis_gripper_objective_no_rollout_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} rows to {csv_path}')

    for obj in args.objectives:
        obj_rows = [r for r in all_rows if r['objective'] == obj and not r.get('error')]
        flips = sum(1 for r in obj_rows if r.get('gripper_token_flipped'))
        avg_open = np.mean([r['open_region_prob_mass_after'] for r in obj_rows]) if obj_rows else 0
        avg_arm = np.mean([r['arm_l2'] for r in obj_rows]) if obj_rows else 0
        print(f'{obj}: {flips}/{len(obj_rows)} flips avg_open_after={avg_open:.4f} avg_armL2={avg_arm:.4f}')


if __name__ == '__main__':
    main()
