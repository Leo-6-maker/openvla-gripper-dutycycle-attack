#!/usr/bin/env python3
"""S20D v5: TRUE TokenPrefixPGD fixed-window runner.
Fixes S20D v4 bug: v4 vis_pgd used fallback visual_linf_noise_adapter.
v5 enforces method=token_prefix_pgd, passes target_action=clean_action,
consumes debug["adv_inputs"] for re-decode, and hard-fails on fallback."""
from __future__ import annotations
import os, sys, argparse, json, csv, time
from pathlib import Path
import numpy as np, torch
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from v4_run_eval_openvla import (
    decode_with_scores, postprocess_openvla_action_for_libero)

TARGET_OBJECT_GUESS = {
    'ketchup': 'ketchup_1', 'tomato_sauce': 'tomato_sauce_1', 'milk': 'milk_1',
    'butter': 'butter_1', 'cream_cheese': 'cream_cheese_1', 'salad_dressing': 'salad_dressing_1',
    'bbq_sauce': 'bbq_sauce_1', 'alphabet_soup': 'alphabet_soup_1',
    'orange_juice': 'orange_juice_1', 'chocolate_pudding': 'chocolate_pudding_1',
}

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True, choices=[
    'alphabet_soup','bbq_sauce','butter','chocolate_pudding','cream_cheese',
    'ketchup','milk','orange_juice','salad_dressing','tomato_sauce'])
ap.add_argument('--state_ids', default='0,1,2,3,4,5,6,7,8,9')
ap.add_argument('--condition', choices=['clean','vis_pgd','random_linf'], default='clean')
ap.add_argument('--window_start', type=int, default=0)
ap.add_argument('--window_end', type=int, default=10)
ap.add_argument('--attack_seed', type=int, default=0)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--random_control_seed', type=int, default=0)
ap.add_argument('--success_metric', default='check_success')
ap.add_argument('--num_steps_wait', type=int, default=10)
ap.add_argument('--model_path', default='/data/aviary/models/openvla/openvla-7b-finetuned-libero-object')
ap.add_argument('--max_steps_override', type=int, default=280)
ap.add_argument('--render_gpu_device_id', type=str, default='0')
ap.add_argument('--model_gpu_device_id', type=str, default='-1')
ap.add_argument('--output_dir', required=True)
ap.add_argument('--save_video_dir', default='')
ap.add_argument('--job_id', default='')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

# ── Model load (same as S20D v4) ──
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
        extra_kw = {"device_map": {"": int(model_gpu_device_id)},
                     "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"}}
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw)
    dev = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v; break
            if isinstance(v, int):
                dev = f"cuda:{v}"; break
    print(f"[model] loaded path={model_path} device={dev} attn={attn_impl} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    return model, processor, dev

os.environ['LIBERO_LOGLEVEL'] = 'ERROR'
print(datetime.now().strftime('[%H:%M:%S]'), 'Loading model from', args.model_path)
model, processor, device = load_model_s20d(
    args.model_path, model_gpu_device_id=int(args.model_gpu_device_id))
model_dtype = torch.bfloat16
K_trigger = 8  # match V4 default uncertainty K
unnorm_key = "libero_object"
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7, f"Unexpected action_dim={action_dim}"
print(datetime.now().strftime('[%H:%M:%S]'), 'Model loaded on %s dtype=%s' % (device, model_dtype))

# ── Env setup ──
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()
TASK_IDX = {'ketchup': 4, 'tomato_sauce': 5, 'milk': 7, 'butter': 6, 'cream_cheese': 1, 'salad_dressing': 2, 'bbq_sauce': 3, 'alphabet_soup': 0, 'orange_juice': 9, 'chocolate_pudding': 8}
task_idx = TASK_IDX[args.task]
task_obj = task_suite.get_task(task_idx)
init_states = task_suite.get_task_init_states(task_idx)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

# ── Shared helpers ──
def decode_action_from_token_ids(model, token_ids, unnorm_key):
    """V4 token-to-action decode, shared by random_linf and token_pgd."""
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    action_stats = model.get_action_stats(unnorm_key)
    mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
    high, low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions)
    return action.astype(np.float32)

def generate_from_adv_inputs(adv_inputs, device, model_dtype, action_dim):
    """Generate action token ids from adversarial processor inputs."""
    input_ids = adv_inputs["input_ids"].to(device)
    pixel_values = adv_inputs["pixel_values"].to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=False)
    return gen.sequences[0, -action_dim:].detach().cpu().numpy(), gen

def physical_gripper_state(env, obs):
    try:
        return env.env.robots[0].controller.gripper_state
    except:
        return {}

# ── Attack setup (v5: FIXED) ──
eps_norm = args.eps_raw_pixels / 255.0  # P0-F: global scope
attacker = None
attacker_config = {}
v5_attack_telemetry = {}

if args.condition == 'vis_pgd':
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker, get_adv_inputs_from_attack_result

    attacker_config = {
        'method': 'token_prefix_pgd',  # ← BUGFIX 1: explicitly set method
        'epsilon': eps_norm,
        'step_size': eps_norm / max(args.pgd_steps, 1) * 1.5,
        'num_steps': args.pgd_steps,
        'random_start': True,
        'objective': 'prefix_locked_gripper_open_margin',
        'arm_preserve_weight': 0.5,
        'gripper_margin': 5.0,
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

    # Hard-fail if wrong method
    if attacker.method not in {'token_prefix_pgd', 'openvla_token_prefix_pgd', 'visual_token_prefix_pgd'}:
        raise RuntimeError(
            'V5 HARD FAIL: condition=vis_pgd but attacker.method=%s (expected token_prefix_pgd). '
            'Refusing to run fallback visual_linf_noise_adapter.' % attacker.method)
    print(datetime.now().strftime('[%H:%M:%S]'), 'v5 TokenPrefixPGD attacker ready. method=%s objective=%s' %
          (attacker.method, attacker_config.get('objective','?')))

# Video
video_enabled = bool(args.save_video_dir)
if video_enabled:
    os.makedirs(args.save_video_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# ── Per-state loop ──
state_ids = [int(x.strip()) for x in args.state_ids.split(',')]
for sid in state_ids:
    if sid >= len(init_states):
        print('state_id %d out of range' % sid); continue

    safe_tag = f"{args.task}_s{sid}_w{args.window_start}_{args.window_end}_s20d_{args.condition}_seed{args.attack_seed}"
    env = OffScreenRenderEnv(robots=['Panda'], bddl_file_name=bddl_file,
        has_renderer=False, has_offscreen_renderer=True, render_gpu_device_id=int(args.render_gpu_device_id),
        use_camera_obs=True, control_freq=20, camera_heights=224, camera_widths=224)
    try: env.env.sim.model.vis.global_.offload = 0
    except: pass
    obs = env.reset()
    obs = env.set_init_state(init_states[sid])

    # V4-aligned dummy wait (P0 fix: was missing)
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(dummy_action)

    max_steps = args.max_steps_override
    ws, we = max(0, min(args.window_start, max_steps)), max(0, min(args.window_end, max_steps))

    success_primary = False; success_done_any = False; success_check_any = False
    success_step_primary = -1; done_step_any = -1
    infra_status = 'ok'; step = 0
    trace_rows = []
    qpos_history = []
    target_object_name = TARGET_OBJECT_GUESS.get(args.task, 'akita_black_bowl_1')
    total_decoded_open = 0; max_streak = 0; current_streak = 0

    # v5 attack telemetry (per-state)
    v5_telemetry = {
        'attack_method': attacker.method if attacker else 'n/a',
        'attacker_config_method': attacker_config.get('method','n/a') if attacker else 'n/a',
        'attack_objective': attacker_config.get('objective','n/a') if attacker else 'n/a',
        'token_label_source': 'not_available',
        'target_action_source': 'clean_action' if args.condition == 'vis_pgd' else 'n/a',
        'target_ce_initial': -1.0, 'target_ce_final': -1.0, 'loss_decrease': -1.0,
        'open_region_prob_mass_after': -1.0, 'close_bin_prob_mass_after': -1.0,
        'gripper_prob_mass_margin_after': -1.0, 'gripper_logit_margin_after': -1.0,
        'corrected_open_token_count': -1,
        'region_mapping_status': 'not_available',
        'arm_preserve_weight': attacker_config.get('arm_preserve_weight', -1) if attacker else -1,
        'gripper_margin_param': attacker_config.get('gripper_margin', -1) if attacker else -1,
        'pixel_budget_adv_inputs_linf': -1.0,
        'pixel_budget_master_linf': -1.0,
        'adv_decode_path': 'token_pgd_adv_inputs_generate' if args.condition == 'vis_pgd' else 'n/a',
        'used_adv_inputs': False,
        'used_x_adv': False,
        'fallback_adapter_used': False,
        'pgd_applied': 0,
    }

    while step < max_steps:
        obs['agentview_image']
        # render handled by obs access

        img_uint8 = obs['agentview_image']

        # ── V4 clean decode ──
        clean_action, prefix_logits, Tclean, gen_out = decode_with_scores(
            model, processor, device, img_uint8, instruction, unnorm_key, K_trigger,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
        clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)

        gripper_phys_before = physical_gripper_state(env, obs)
        gripper_qpos_before = float(np.sum(gripper_phys_before.get('qpos', [0.0])))
        qpos_history.append(gripper_qpos_before)

        # ── Window attack injection ──
        in_window = ws <= step < we
        attack_this_step = in_window and args.condition != 'clean'

        pgd_applied = 0; perturbation_space = 'none'
        env_action = clean_env_action.copy()
        executed_action = clean_action.copy()

        if attack_this_step:
            if args.condition == 'vis_pgd' and attacker is not None:
                try:
                    # ── v5 FIXED attack call ──
                    # BUGFIX 2: pass target_action=clean_action (not None)
                    attack_result = attacker.attack(
                        img_uint8, instruction,
                        clean_action,
                        clean_action,  # ← target_action for prefix_locked
                        gen_out,
                        unnorm_key=unnorm_key)

                    if attack_result is not None:
                        # BUGFIX 3: consume debug["adv_inputs"], not x_adv
                        adv_inputs = get_adv_inputs_from_attack_result(attack_result)

                        if adv_inputs is not None and adv_inputs.get("input_ids") is not None:
                            token_ids, gen = generate_from_adv_inputs(adv_inputs, device, model_dtype, action_dim)
                            adv_action = decode_action_from_token_ids(model, token_ids, unnorm_key)
                            adv_env_action = postprocess_openvla_action_for_libero(adv_action, enabled=True)
                            env_action = adv_env_action
                            executed_action = adv_action
                            pgd_applied = 1
                            perturbation_space = 'token_prefix_pgd_adv_inputs_v5'

                            v5_telemetry['used_adv_inputs'] = True
                            v5_telemetry['used_x_adv'] = False
                            v5_telemetry['pgd_applied'] = 1

                            # Extract debug telemetry
                            dbg = getattr(attack_result, 'debug', {}) or {}
                            v5_telemetry['attack_method'] = getattr(attack_result, 'attack_method', 'unknown')
                            v5_telemetry['token_label_source'] = str(dbg.get('token_label_source', 'not_available'))
                            v5_telemetry['target_ce_initial'] = float(dbg.get('target_ce_initial', -1) or -1)
                            v5_telemetry['target_ce_final'] = float(dbg.get('target_ce_final', -1) or -1)
                            v5_telemetry['loss_decrease'] = float(dbg.get('loss_decrease', 0) or 0)
                            v5_telemetry['open_region_prob_mass_after'] = float(dbg.get('open_region_prob_mass_after', -1) or -1)
                            v5_telemetry['close_bin_prob_mass_after'] = float(dbg.get('close_bin_prob_mass_after', -1) or -1)
                            v5_telemetry['gripper_prob_mass_margin_after'] = float(dbg.get('gripper_prob_mass_margin_after', -1) or -1)
                            v5_telemetry['gripper_logit_margin_after'] = float(dbg.get('gripper_logit_margin_after', -1) or -1)
                            v5_telemetry['corrected_open_token_count'] = int(dbg.get('corrected_open_token_count', -1) or -1)
                            v5_telemetry['region_mapping_status'] = str(dbg.get('region_mapping_status', 'not_available'))
                            v5_telemetry['pixel_budget_adv_inputs_linf'] = float(dbg.get('pixel_budget_adv_inputs_linf', -1) or -1)
                            v5_telemetry['pixel_budget_master_linf'] = float(dbg.get('pixel_budget_master_linf', -1) or -1)
                            v5_telemetry['fallback_adapter_used'] = bool(dbg.get('fallback_adapter_used', False))
                        else:
                            infra_status = 'v5_token_pgd_no_adv_inputs'
                            raise RuntimeError('V5 HARD FAIL: adv_inputs missing for token_pgd')
                    else:
                        raise RuntimeError("V5 HARD FAIL: attack_result is None for token_pgd")
                except Exception as e:
                                        raise

            elif args.condition == 'random_linf':
                try:
                    from gripper_attack.openvla_preprocess import prepare_openvla_image
                    rng = np.random.RandomState(args.random_control_seed)
                    noise_pattern = rng.choice([-1, 1], size=(224,224,3)).astype(np.float32)
                    pixel_values = prepare_openvla_image(
                        img_uint8, libero_official_preprocess=False,
                        libero_preprocess_backend='official_pil_lanczos',
                        center_crop=True, resize_size=224)
                    pixel_values_np = pixel_values.cpu().numpy() if hasattr(pixel_values, 'cpu') else np.array(pixel_values)
                    perturbed = np.clip(pixel_values_np + eps_norm * noise_pattern, 0.0, 1.0)
                    perturbed_t = torch.from_numpy(perturbed).to(device=device, dtype=model_dtype)
                    input_ids = torch.tensor([[model.processor.tokenizer(instruction, return_tensors='pt').input_ids[0].tolist()]], device=device)
                    with torch.inference_mode():
                        gen_rand = model.generate(input_ids=input_ids, pixel_values=perturbed_t,
                            max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=False)
                    rand_tids = gen_rand.sequences[0, -action_dim:].detach().cpu().numpy()
                    rand_action = decode_action_from_token_ids(model, rand_tids, unnorm_key)
                    rand_env_action = postprocess_openvla_action_for_libero(rand_action, enabled=True)
                    env_action = rand_env_action; executed_action = rand_action
                    pgd_applied = 0; perturbation_space = 'random_linf_v5_decode'
                except Exception as e:
                    infra_status = 'rand_error: %s' % str(e)[:80]

        eef_before = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        obj_before_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None
        obj_before = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None
        obs, reward, done, info = env.step(env_action)

        gripper_phys_after = physical_gripper_state(env, obs)
        gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))
        eef_after = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], "_hand_pos") else None
        obj_after = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None
        is_open = 1 if env_action[-1] < -0.5 else 0

        total_decoded_open += is_open
        if is_open: current_streak += 1
        else: current_streak = 0
        max_streak = max(max_streak, current_streak)


        success_done = bool(done)
        success_check = bool(env.check_success())
        success_primary_now = success_done if args.success_metric == 'done' else success_check
        if success_primary_now and not success_primary:
            success_primary = True; success_step_primary = step
        if success_done and not success_done_any: success_done_any = True; done_step_any = step
        if success_check and not success_check_any: success_check_any = True

        if video_enabled:
            from PIL import Image
            pil_img = Image.fromarray(img_uint8)
            pil_img.save(os.path.join(args.save_video_dir, f'frame_{step:06d}.png'))

        rng_val = getattr(args, 'random_control_seed', 0)
        trace_rows.append({
            'step': step, 'state_id': sid, 'task': args.task, 'condition': args.condition,
            'in_window': int(in_window), 'attack_this_step': int(attack_this_step),
            'clean_gripper_env': round(float(clean_env_action[-1]), 6),
            'executed_gripper_env': round(float(env_action[-1]), 6),
            'decoded_open_bool': is_open,
            'gripper_qpos_before': round(gripper_qpos_before, 8),
            'gripper_qpos_after': round(gripper_qpos_after, 8),
            'physical_gripper_opening_delta': round(gripper_qpos_after - gripper_qpos_before, 8),
            'eef_x': round(float(eef_before[0]), 6) if eef_before is not None else '',
            'eef_y': round(float(eef_before[1]), 6) if eef_before is not None else '',
            'eef_z': round(float(eef_before[2]), 6) if eef_before is not None else '',
            'eef_z_after': round(float(eef_after[2]), 6) if eef_after is not None else '',
            'obj_x': round(float(obj_before[0]), 6) if obj_before is not None else '',
            'obj_y': round(float(obj_before[1]), 6) if obj_before is not None else '',
            'obj_z': round(float(obj_before[2]), 6) if obj_before is not None else '',
            'obj_z_after': round(float(obj_after[2]), 6) if obj_after is not None else '',
            'target_object_name': target_object_name or '',
            'pgd_applied': pgd_applied, 'perturbation_space': perturbation_space,
            'success_done': int(success_done), 'success_check': int(success_check),
            'success_primary': int(success_primary_now),
            'attack_seed': args.attack_seed, 'job_id': args.job_id,
            'infra_status': infra_status, 'window_start': ws, 'window_end': we,
            # v5 per-step telemetry
            'attack_method': v5_telemetry.get('attack_method', '') if pgd_applied else '',
            'token_label_source': v5_telemetry.get('token_label_source', '') if pgd_applied else '',
            'target_ce_initial': round(v5_telemetry['target_ce_initial'], 6),
            'target_ce_final': round(v5_telemetry['target_ce_final'], 6),
            'loss_decrease': round(v5_telemetry['loss_decrease'], 6),
            'gripper_logit_margin_after': round(v5_telemetry['gripper_logit_margin_after'], 6),
            'open_region_prob_mass_after': round(v5_telemetry['open_region_prob_mass_after'], 6),
            'close_bin_prob_mass_after': round(v5_telemetry['close_bin_prob_mass_after'], 6),
            'corrected_open_token_count': v5_telemetry.get('corrected_open_token_count', '') if pgd_applied else '',
            'pixel_budget_adv_inputs_linf': round(v5_telemetry['pixel_budget_adv_inputs_linf'], 8),
            'adv_decode_path': v5_telemetry.get('adv_decode_path', '') if pgd_applied else '',
            'used_adv_inputs': v5_telemetry.get('used_adv_inputs', '') if pgd_applied else '',
            'fallback_adapter_used': v5_telemetry.get('fallback_adapter_used', False) if pgd_applied else False,
            'adv_gripper_raw': round(float(executed_action[-1]), 6) if pgd_applied else '',
            'adv_env_gripper': round(float(env_action[-1]), 6) if pgd_applied else '',
            'adv_open_bool': is_open if pgd_applied else '',
        })

        if success_primary or done: break
        step += 1

    env.close(); torch.cuda.empty_cache()

    # ── Summary ──
    open_count = sum(1 for i in range(ws, min(we, len(trace_rows))) if trace_rows[i].get('decoded_open_bool', 0))
    wqpos = np.array(qpos_history[ws:min(we, len(qpos_history))]) if ws < len(qpos_history) else np.array([])
    baseline_qpos = float(np.mean(wqpos)) if len(wqpos) > 0 else 0.0
    post_start = min(len(qpos_history), we + 40)
    post_qpos = np.array(qpos_history[we:post_start]) if we < len(qpos_history) else np.array([])
    qpos_pos_area = float(np.sum(np.maximum(post_qpos - baseline_qpos, 0))) if len(post_qpos) > 0 else 0.0

    summary = {
        'runner_family': 's20d_v5_token_pgd_fixed_window_l3',
        'job_id': args.job_id, 'task': args.task, 'state_id': sid,
        'condition': args.condition, 'window_start': ws, 'window_end': we,
        'attack_seed': args.attack_seed, 'random_control_seed': args.random_control_seed,
        'n_steps': len(trace_rows), 'max_steps': max_steps, 'num_steps_wait': args.num_steps_wait,
        'success_primary': success_primary, 'success_primary_metric': args.success_metric,
        'success_done_any': success_done_any, 'success_check_any': success_check_any,
        'success_step_primary': success_step_primary, 'done_step': done_step_any,
        'timeout': len(trace_rows) >= max_steps and not success_primary,
        'decoded_open_count': open_count, 'max_open_streak': max_streak,
        'qpos_pos_area': round(qpos_pos_area, 8), 'qpos_baseline': round(baseline_qpos, 8),
        'infra_status': infra_status, 'video_dir': args.save_video_dir,
        'model_path': args.model_path,
        'decode_path': 'v4_decode_with_scores',
        'postprocess_path': 'v4_postprocess_openvla_action_for_libero',
        'image_preprocess': 'v4_prepare_openvla_image_official_pil_lanczos_center_crop_224',
        'eos_token': 'v4_29871_insertion', 'attention_mask': 'v4_drop',
        'dtype': str(model_dtype),
        # v5 telemetry
        'vis_runner_version': 'v5_token_pgd',
        'attack_method': v5_telemetry.get('attack_method', '') if pgd_applied else '',
        'attacker_config_method': v5_telemetry['attacker_config_method'],
        'attack_objective': v5_telemetry['attack_objective'],
        'token_label_source': v5_telemetry.get('token_label_source', '') if pgd_applied else '',
        'target_action_source': v5_telemetry['target_action_source'],
        'target_ce_initial': round(v5_telemetry['target_ce_initial'], 6),
        'target_ce_final': round(v5_telemetry['target_ce_final'], 6),
        'loss_decrease': round(v5_telemetry['loss_decrease'], 6),
        'gripper_logit_margin_after': round(v5_telemetry['gripper_logit_margin_after'], 6),
        'open_region_prob_mass_after': round(v5_telemetry['open_region_prob_mass_after'], 6),
        'close_bin_prob_mass_after': round(v5_telemetry['close_bin_prob_mass_after'], 6),
        'gripper_prob_mass_margin_after': round(v5_telemetry['gripper_prob_mass_margin_after'], 6),
        'corrected_open_token_count': v5_telemetry.get('corrected_open_token_count', '') if pgd_applied else '',
        'region_mapping_status': v5_telemetry['region_mapping_status'],
        'pixel_budget_adv_inputs_linf': round(v5_telemetry['pixel_budget_adv_inputs_linf'], 8),
        'pixel_budget_master_linf': round(v5_telemetry['pixel_budget_master_linf'], 8),
        'adv_decode_path': v5_telemetry.get('adv_decode_path', '') if pgd_applied else '',
        'used_adv_inputs': v5_telemetry.get('used_adv_inputs', '') if pgd_applied else '',
        'used_x_adv': v5_telemetry['used_x_adv'],
        'fallback_adapter_used': v5_telemetry['fallback_adapter_used'],
        'v5_pgd_applied': v5_telemetry['pgd_applied'],
    }

    out_json = os.path.join(args.output_dir, f'summary_{safe_tag}_job{args.job_id}.json')
    with open(out_json, 'w') as f: json.dump(summary, f, indent=2)

    if trace_rows:
        out_trace = os.path.join(args.output_dir, f'trace_{safe_tag}_job{args.job_id}.csv')
        with open(out_trace, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader(); w.writerows(trace_rows)

    print(datetime.now().strftime('[%H:%M:%S]'), 'Done: state=%d steps=%d primary_success=%s@step%s done=%s check=%s open=%d streak=%d infra=%s' %
          (sid, len(trace_rows), success_primary, success_step_primary, success_done_any, success_check_any, open_count, max_streak, infra_status))

print(datetime.now().strftime('[%H:%M:%S]'), 'S20d v5 TokenPGD runner done')
