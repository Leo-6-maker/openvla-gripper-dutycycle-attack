#!/usr/bin/env python3
"""Gripper margin scan: audit clean-policy rollout frames for gripper vulnerability.

For each task, runs a clean policy rollout and records per-step:
- clean_gripper_token, clean_gripper_action
- open_region_prob_mass, non_open_max_logit
- gripper_margin_to_open (= logsumexp(open) - max(non_open))
- qpos
- ProprioNoStep proxy (if available)

Output: tables/vis_gripper_margin_scan.csv
        reports/VIS_GRIPPER_MARGIN_SCAN.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'

TASK_CONFIGS = {
    'cream_cheese': {'task_id': 1, 'instruction': 'pick up the cream cheese and place it in the basket'},
    'salad_dressing': {'task_id': 2, 'instruction': 'put the salad dressing in the basket'},
    'ketchup': {'task_id': 4, 'instruction': 'pick up the ketchup and place it in the basket'},
    'tomato_sauce': {'task_id': 5, 'instruction': 'pick up the tomato sauce and place it in the basket'},
}

CSV_FIELDS = [
    'task', 'step', 'gripper_qpos',
    'clean_gripper_token', 'clean_gripper_action',
    'open_region_prob_mass', 'non_open_max_logit',
    'gripper_margin_to_open', 'open_region_logsumexp',
    'top1_token_id', 'top1_is_open',
    'env_gripper', 'done',
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


def main():
    ap = argparse.ArgumentParser(description='VIS Gripper Margin Scan')
    ap.add_argument('--task', choices=list(TASK_CONFIGS.keys()), nargs='+',
                    default=['cream_cheese', 'salad_dressing', 'ketchup', 'tomato_sauce'])
    ap.add_argument('--gpu_pair', default='0,1')
    ap.add_argument('--output_dir', default=str(REPO_ROOT / 'tables'))
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        print(f'[DRY RUN] Would scan: {args.task}')
        return

    import torch
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

    # Get OPEN region token IDs
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
    _dummy_cfg = {'attack_optimizer': {'method': 'token_prefix_pgd', 'objective': 'gripper_open_region_ce'}}
    _attacker = TokenPrefixPGDAttacker(
        model, processor, _dummy_cfg, seed=0,
        preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False,
                           'resize_size': 224, 'postprocess_gripper': True}, device=device)
    open_token_ids = _attacker.action_bins_for_env_sign(
        action_dim - 1, 'negative', UNNORM_KEY, postprocess_gripper=True)
    open_token_set = set(int(x) for x in open_token_ids.detach().cpu().tolist())
    print(f'OPEN region: {len(open_token_set)} tokens (range {min(open_token_set)}-{max(open_token_set)})')

    # LIBERO env
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    all_rows = []
    for task_name in args.task:
        cfg = TASK_CONFIGS[task_name]
        print(f'\n=== {task_name} ===')
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict['libero_object']()
        task = task_suite.get_task(cfg['task_id'])
        bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        initial_states = task_suite.get_task_init_states(cfg['task_id'])

        env_args = {
            'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
            'has_renderer': False, 'has_offscreen_renderer': True,
            'use_camera_obs': True, 'camera_names': ['agentview'],
            'control_freq': 20, 'render_gpu_device_id': int(args.gpu_pair.split(',')[0]),
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        obs = env.reset()
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        env.set_init_state(initial_states[0])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        instruction = cfg['instruction']
        max_steps = 300
        t0 = time.time()

        for step in range(max_steps):
            img = obs['agentview_image'][::-1, ::-1]
            img = Image.fromarray(img).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            img_np = np.array(img)

            qpos = float(obs['robot0_gripper_qpos'][0]) if len(obs['robot0_gripper_qpos']) > 0 else 0.0

            # Decode clean action with logits
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

            # Extract gripper-token logits from scores (last token = gripper)
            gripper_scores = gen.scores[-1][0].float()  # vocab logits for last generated token
            gripper_probs = torch.softmax(gripper_scores, dim=-1)

            open_token_tensor = open_token_ids.to(gripper_probs.device)
            open_prob_mass = float(gripper_probs[open_token_tensor].sum().cpu())
            open_logsumexp = float(torch.logsumexp(gripper_scores[open_token_tensor], dim=0).cpu())

            # Max logit among non-open tokens
            non_open_mask = torch.ones_like(gripper_scores, dtype=torch.bool)
            non_open_mask[open_token_tensor] = False
            non_open_max_logit = float(gripper_scores[non_open_mask].max().cpu())

            gripper_margin = open_logsumexp - non_open_max_logit

            top1_token = int(torch.argmax(gripper_scores).cpu().item())
            top1_is_open = top1_token in open_token_set

            clean_grip_token = int(token_ids[-1])
            clean_grip_action = float(action[-1])

            # Step env
            env_action = action.copy()
            env_action[-1] = 2.0 * env_action[-1] - 1.0
            env_action[-1] = np.sign(env_action[-1])
            env_action[-1] = 1.0 if env_action[-1] == 0 else env_action[-1]
            env_action[-1] = -1.0 * env_action[-1]
            obs, reward, done, info = env.step(env_action)

            row = {
                'task': task_name,
                'step': step,
                'gripper_qpos': qpos,
                'clean_gripper_token': clean_grip_token,
                'clean_gripper_action': clean_grip_action,
                'open_region_prob_mass': round(open_prob_mass, 6),
                'non_open_max_logit': round(non_open_max_logit, 4),
                'gripper_margin_to_open': round(gripper_margin, 4),
                'open_region_logsumexp': round(open_logsumexp, 4),
                'top1_token_id': top1_token,
                'top1_is_open': top1_is_open,
                'env_gripper': float(env_action[-1]),
                'done': bool(done),
            }
            all_rows.append(row)

            if step % 30 == 0:
                print(f'  step {step}: open_prob={open_prob_mass:.4f} margin={gripper_margin:.2f} '
                      f'top1_open={top1_is_open} grip_act={clean_grip_action:.4f}')

            if done:
                break

        elapsed = time.time() - t0
        print(f'  {step + 1} frames in {elapsed:.0f}s ({elapsed / max(step, 1):.2f}s/step)')
        env.close()

    # Write CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'vis_gripper_margin_scan.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} rows to {csv_path}')

    # Per-task summary
    for task_name in args.task:
        task_rows = [r for r in all_rows if r['task'] == task_name]
        margins = [r['gripper_margin_to_open'] for r in task_rows]
        open_probs = [r['open_region_prob_mass'] for r in task_rows]
        top1_opens = sum(1 for r in task_rows if r['top1_is_open'])
        low_margin_frames = sorted(task_rows, key=lambda r: r['gripper_margin_to_open'])[:10]
        print(f'\n{task_name}: {len(task_rows)} frames')
        print(f'  avg margin={np.mean(margins):.2f}  avg open_prob={np.mean(open_probs):.4f}')
        print(f'  top1_is_open: {top1_opens}/{len(task_rows)} ({100*top1_opens/max(len(task_rows),1):.1f}%)')
        print(f'  lowest-margin frames:')
        for r in low_margin_frames[:5]:
            print(f'    step={r["step"]:3d} margin={r["gripper_margin_to_open"]:.2f} '
                  f'open_prob={r["open_region_prob_mass"]:.4f} qpos={r["gripper_qpos"]:.5f}')


if __name__ == '__main__':
    main()
