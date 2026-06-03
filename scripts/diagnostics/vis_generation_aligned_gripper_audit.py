#!/usr/bin/env python3
"""Generation-aligned gripper audit for P0 corrected-smoke diagnosis.

Problem: teacher-forced _audit_logits shows corrected_open_prob_mass=1.0, but
actual autoregressive generation still produces CLOSE gripper token.

This script records three views per objective on cream_cheese_step030:
  A. Clean generation (model.generate with clean image)
  B. Teacher-forced audit (_audit_logits on adv pixel_values with clean prefix)
  C. Actual adv generation (model.generate with adv pixel_values)

Outputs:
  tables/vis_generation_aligned_gripper_audit.csv
  reports/VIS_GENERATION_ALIGNED_GRIPPER_AUDIT.md
"""
from __future__ import annotations

import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.gripper_semantics import raw_gripper_is_open

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'

TASK_INSTRUCTIONS = {
    'cream_cheese': 'pick up the cream cheese and place it in the basket',
}
TASK_IDS = {'cream_cheese': 1}

CSV_FIELDS = [
    'objective', 'view', 'restart',
    'generated_tokens', 'decoded_actions',
    'gripper_token', 'gripper_action', 'gripper_is_open',
    'arm_tokens_changed', 'arm_l2',
    'corrected_open_prob_mass', 'gripper_logit_margin',
    'open_region_logsumexp', 'non_open_max_logit',
    'label_positions_used', 'gripper_row_index',
    'target_ce_final',
    'note',
]


def prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def load_model(gpu_pair: str = '0,1'):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    gpu_ids = [int(x) for x in gpu_pair.split(',')]
    max_mem = {gpu_ids[0]: '9000MiB', gpu_ids[1]: '9000MiB', 'cpu': '64GiB'}
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map='auto',
        max_memory=max_mem, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objectives', nargs='+',
                    default=['gripper_open_region_ce', 'prefix_locked_gripper_open_region_ce',
                             'prefix_locked_gripper_open_margin', 'gripper_open_expected_action'])
    ap.add_argument('--eps_raw', type=int, default=8)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--pgd_restarts', type=int, default=3)
    ap.add_argument('--gpu_pair', default='0,1')
    ap.add_argument('--output_dir', default=str(REPO_ROOT / 'tables'))
    args = ap.parse_args()

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

    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
    from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs

    image_mean, image_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    try:
        ip = processor.image_processor
        all_stds = getattr(ip, 'stds', [[0.5, 0.5, 0.5]])
        image_std = all_stds[-1]
    except Exception:
        pass

    eps_proc = min((args.eps_raw / 255.0) / float(s) for s in image_std)
    step_size = eps_proc / 8.0

    print(f'device={device} action_dim={action_dim}')
    print(f'eps_raw={args.eps_raw} eps_proc={eps_proc:.6f} steps={args.steps}')
    print(f'VS={VS} bins={len(BC)}')

    # Collect cream_cheese step030
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_name = 'cream_cheese'
    cfg = TASK_IDS[task_name]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    task = task_suite.get_task(cfg)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(cfg)

    env_args = {
        'bddl_file_name': bddl, 'camera_heights': 256, 'camera_widths': 256,
        'has_renderer': False, 'has_offscreen_renderer': True,
        'use_camera_obs': True, 'camera_names': ['agentview'],
        'control_freq': 20, 'render_gpu_device_id': int(args.gpu_pair.split(',')[0]),
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0); obs = env.reset()
    env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(initial_states[0])
    for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))

    instruction = TASK_INSTRUCTIONS[task_name]
    target_step = 30
    for step in range(target_step + 1):
        img = obs['agentview_image'][::-1, ::-1]
        img = Image.fromarray(img).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        img_np = np.array(img)

        pil_img = Image.fromarray(img_np.astype(np.uint8))
        text = prompt(str(instruction).lower())
        inputs = processor(text, pil_img, return_tensors='pt')
        inputs.pop('attention_mask', None)
        for k, v in list(inputs.items()):
            if torch.is_floating_point(v): inputs[k] = v.to(device=device, dtype=mdtype)
            else: inputs[k] = v.to(device)
        if not torch.all(inputs['input_ids'][:, -1] == 29871):
            inputs['input_ids'] = torch.cat(
                (inputs['input_ids'], torch.tensor([[29871]], dtype=torch.long, device=device)), dim=1)

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
        disc = np.clip(VS - token_ids - 1, 0, len(BC) - 1)
        norm_actions = BC[disc].astype(np.float32)
        action = np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)

        if step < target_step:
            env_action = action.copy()
            env_action[-1] = 2.0*env_action[-1]-1.0; env_action[-1] = np.sign(env_action[-1])
            env_action[-1] = 1.0 if env_action[-1]==0 else env_action[-1]; env_action[-1] = -1.0*env_action[-1]
            obs, _, done, _ = env.step(env_action)

    env.close()

    # View A: clean generation
    clean_tokens = token_ids.copy()
    clean_action = action.copy()
    print(f'\n=== View A: Clean Generation ===')
    print(f'  tokens: {clean_tokens.tolist()}')
    print(f'  actions: {[float(f"{a:.4f}") for a in clean_action]}')
    print(f'  gripper: token={clean_tokens[-1]} action={clean_action[-1]:.4f}')

    all_rows = []
    clean_row = {
        'objective': 'clean', 'view': 'A_generated', 'restart': 0,
        'generated_tokens': str(clean_tokens.tolist()),
        'decoded_actions': str([float(f"{a:.4f}") for a in clean_action]),
        'gripper_token': int(clean_tokens[-1]),
        'gripper_action': float(clean_action[-1]),
        'gripper_is_open': raw_gripper_is_open(float(clean_action[-1])),
        'arm_tokens_changed': 0, 'arm_l2': 0.0,
        'note': 'clean baseline generation',
    }
    all_rows.append(clean_row)

    target_action = np.asarray(clean_action, dtype=np.float32).copy()
    target_action[-1] = 1.0  # tokenization only; corrected region used for loss

    for objective in args.objectives:
        print(f'\n=== Objective: {objective} ===')
        is_prefix = objective.startswith('prefix_locked_') or objective == 'gripper_open_expected_action'

        for restart in range(args.pgd_restarts):
            restart_seed = 42 + restart * 1000
            attack_cfg = {
                'attack_optimizer': {
                    'method': 'token_prefix_pgd', 'objective': objective,
                    'epsilon': eps_proc, 'step_size': step_size, 'num_steps': args.steps,
                    'random_start': (restart > 0),
                }
            }
            if is_prefix:
                attack_cfg['attack_optimizer']['arm_preserve_weight'] = 0.1
                attack_cfg['attack_optimizer']['gripper_margin'] = 5.0

            attacker = TokenPrefixPGDAttacker(
                model, processor, attack_cfg, seed=restart_seed,
                preprocess_kwargs={'libero_official_preprocess': False, 'center_crop': False,
                                   'resize_size': 224, 'postprocess_gripper': True},
                device=device)

            attack_result = attacker.attack(
                observation=img_np, instruction=instruction,
                clean_action=clean_action, target_action=target_action,
                clean_model_output=gen, unnorm_key=UNNORM_KEY)

            debug = attack_result.debug or {}
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)

            # View B: teacher-forced audit
            clean_audit = (debug.get('clean_logit_audit') or {}).get('action_token_logit_audit', [])
            adv_audit = (debug.get('adv_logit_audit') or {}).get('action_token_logit_audit', [])
            g_clean = clean_audit[-1] if clean_audit else {}
            g_adv = adv_audit[-1] if adv_audit else {}

            tf_open_mass = g_adv.get('open_bin_prob_mass', 0.0) or 0.0
            tf_margin = float(g_adv.get('open_region_logsumexp', -100) or -100) - float(g_adv.get('non_open_max_logit', 0) or 0)

            # View C: actual adv generation
            adv_pixel_values = adv_inputs['pixel_values']
            adv_input_ids = adv_inputs['input_ids']
            with torch.inference_mode():
                adv_gen = model.generate(
                    input_ids=adv_input_ids, pixel_values=adv_pixel_values,
                    max_new_tokens=action_dim, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
            adv_tokens = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
            adv_disc = np.clip(VS - adv_tokens - 1, 0, len(BC)-1)
            adv_norm = BC[adv_disc].astype(np.float32)
            adv_action = np.where(mask, 0.5*(adv_norm+1)*(high-low)+low, adv_norm).astype(np.float32)

            arm_changed = sum(1 for i in range(6) if int(clean_tokens[i]) != int(adv_tokens[i]))
            arm_l2_gen = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
            grip_is_open = raw_gripper_is_open(float(adv_action[-1]))

            print(f'  restart={restart}: gen_grip_token={adv_tokens[-1]} gen_grip_action={adv_action[-1]:.4f} '
                  f'is_open={grip_is_open} arm_changed={arm_changed} armL2={arm_l2_gen:.4f} '
                  f'tf_open_mass={tf_open_mass:.4f}')

            for view_name, tokens, action_val in [
                ('B_teacher_forced', clean_tokens, float(clean_action[-1])),
                ('C_generated', adv_tokens, float(adv_action[-1]))]:
                row = {
                    'objective': objective, 'view': view_name, 'restart': restart,
                    'generated_tokens': str(tokens.tolist()),
                    'decoded_actions': str([float(f"{a:.4f}") for a in (clean_action if view_name == 'B_teacher_forced' else adv_action)]),
                    'gripper_token': int(tokens[-1]),
                    'gripper_action': float(adv_action[-1] if view_name == 'C_generated' else clean_action[-1]),
                    'gripper_is_open': grip_is_open if view_name == 'C_generated' else False,
                    'arm_tokens_changed': arm_changed if view_name == 'C_generated' else 0,
                    'arm_l2': arm_l2_gen if view_name == 'C_generated' else 0.0,
                    'corrected_open_prob_mass': round(float(tf_open_mass), 6) if view_name == 'B_teacher_forced' else None,
                    'gripper_logit_margin': round(float(tf_margin), 4) if view_name == 'B_teacher_forced' else None,
                    'open_region_logsumexp': g_adv.get('open_region_logsumexp') if view_name == 'B_teacher_forced' else None,
                    'non_open_max_logit': g_adv.get('non_open_max_logit') if view_name == 'B_teacher_forced' else None,
                    'target_ce_final': debug.get('target_ce_final') if view_name == 'B_teacher_forced' else None,
                    'note': 'teacher-forced: model sees clean prefix tokens' if view_name == 'B_teacher_forced' else 'autoregressive: model generates each token',
                }
                all_rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'vis_generation_aligned_gripper_audit.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} rows to {csv_path}')


if __name__ == '__main__':
    main()
