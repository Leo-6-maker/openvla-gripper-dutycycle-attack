#!/usr/bin/env python3
"""V6 Online-Trigger Layer3 Runner (P0-fixed).
- P0-1: VIS uses get_adv_inputs_from_attack_result() + V5 re-decode path
- P0-2: Attacker config matches V5 exactly (step_size, num_steps, objective, seed, preprocess_kwargs)
- P0-3: instruction = raw task language (decode_with_scores applies prompt internally)
- P0-4: RAND perturbs pixel_values in visual space (L∞, eps=6/255), same as VIS input space
- P0-5: env factory committed alongside
- P1-1: Import physical_gripper_state from v4_run_eval_openvla
- P1-2: clean_observer does NOT increment perturb_frame_count
- P1-3: Summary records trigger-moment values
- P1-4: infra_status recorded per-episode, not hardcoded
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

# ── Runner self-provenance ──
import hashlib as _hl
_runner_sha256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
del _hl

# ── Args ──
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--condition', required=True,
                choices=['clean_observer', 'online_random_linf', 'online_vis_pgd'])
ap.add_argument('--attack_seed', type=int, default=99)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=int, default=6)
ap.add_argument('--attack_objective', default='prefix_locked_gripper_top1_open_vs_close_execspec_v2',
                choices=['prefix_locked_gripper_top1_open_vs_close_execspec_v2',
                         'autoregressive_prefix_gripper_open_execspec_v3'])
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
                dev = v; break
            if isinstance(v, int):
                dev = "cuda:%d" % v; break
    return model, processor, dev


model, processor, device = load_model_s20d(
    args.model_path, model_gpu_device_id=int(args.model_gpu_device_id))
model_dtype = torch.bfloat16
unnorm_key = 'libero_object'
K_trigger = 8
action_dim = int(model.get_action_dim(unnorm_key))
assert action_dim == 7, f"Unexpected action_dim={action_dim}"
print('[%s] Model loaded. device=%s' % (time.strftime('%H:%M:%S'), device), flush=True)

# ── Actual import path provenance (prevents stale module shadowing) ──
import hashlib as _hl
import gripper_attack.attack_adapter as _aa
import gripper_attack.gripper_semantics as _gs
import gripper_attack.openvla_libero_exec_spec as _spec

_actual_adapter_path = str(Path(_aa.__file__).resolve())
_actual_semantics_path = str(Path(_gs.__file__).resolve())
_actual_spec_path = str(Path(_spec.__file__).resolve())

_expected_src = str((REPO_ROOT / "src" / "gripper_attack").resolve())
assert _actual_adapter_path.startswith(_expected_src), \
    f"adapter loaded from {_actual_adapter_path}, expected under {_expected_src}"
assert _actual_semantics_path.startswith(_expected_src), \
    f"semantics loaded from {_actual_semantics_path}, expected under {_expected_src}"
assert _actual_spec_path.startswith(_expected_src), \
    f"exec_spec loaded from {_actual_spec_path}, expected under {_expected_src}"

# Function identity: all modules must share the same canonical function objects
assert _aa.raw_gripper_is_open is _gs.raw_gripper_is_open is _spec.raw_gripper_is_open, \
    "raw_gripper_is_open function identity mismatch across modules"
assert _aa.raw_gripper_is_close is _gs.raw_gripper_is_close is _spec.raw_gripper_is_close, \
    "raw_gripper_is_close function identity mismatch across modules"

_adapter_sha256 = _hl.sha256(open(_actual_adapter_path, 'rb').read()).hexdigest()
_semantics_sha256 = _hl.sha256(open(_actual_semantics_path, 'rb').read()).hexdigest()
_spec_sha256 = _hl.sha256(open(_actual_spec_path, 'rb').read()).hexdigest()
_function_identity_ok = True
del _hl, _aa, _gs, _spec

# ── V4 imports ──
from v4_run_eval_openvla import (
    prompt,
    decode_with_scores, postprocess_openvla_action_for_libero,
    physical_gripper_state)

# ── Env ──
from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait)
from libero.libero import benchmark, get_libero_path

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
# P0-3: RAW instruction — decode_with_scores applies prompt() internally
instruction = task_obj.language

env, obs = build_v4_exact_env(
    bddl_file, int(args.render_gpu_device_id),
    args.max_steps_override, args.num_steps_wait)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, args.num_steps_wait)

max_steps = args.max_steps_override

# ── Attack setup (P0-2: V5-aligned config, P0-1: adv_inputs path) ──
eps_norm = args.eps_raw_pixels / 255.0
attacker = None
if args.condition == 'online_vis_pgd':
    from gripper_attack.attack_adapter import (
        OpenVLAVisualAttacker, get_adv_inputs_from_attack_result)

    attacker_config = {
        'method': 'token_prefix_pgd',
        'epsilon': eps_norm,
        'step_size': eps_norm / max(args.pgd_steps, 1) * 1.5,
        'num_steps': args.pgd_steps,
        'random_start': True,
        'objective': args.attack_objective,
        'arm_preserve_weight': 0.5,
        'gripper_margin': 0.5,
    }
    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor,
        config={
            'attack_optimizer': attacker_config,
            'directional_target': {
                'direction_id': 'gripper_open',
                'dims': list(range(action_dim)),
            },
            'uncertainty': {'K_trigger': K_trigger},
        },
        direction_spec={
            'g_hat': np.zeros(action_dim, dtype=np.float32),
            'dims': list(range(action_dim)),
        },
        seed=args.attack_seed,
        preprocess_kwargs={
            'libero_official_preprocess': False,
            'libero_preprocess_backend': 'official_pil_lanczos',
            'center_crop': True,
            'resize_size': 224,
            'postprocess_gripper': True,
        },
        device=device)
    print('[%s] V6 TokenPrefixPGD attacker ready (seed=%d, PGD=%d)' %
          (time.strftime('%H:%M:%S'), args.attack_seed, args.pgd_steps),
          flush=True)

# ── Helpers ──
def decode_action_from_token_ids(token_ids):
    """V5 token→action decoder (used for VIS re-decode)."""
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)


def generate_from_adv_inputs(adv_inputs, device, model_dtype, action_dim):
    """V5 re-decode from adversarial processor inputs."""
    input_ids = adv_inputs["input_ids"].to(device)
    pixel_values = adv_inputs["pixel_values"].to(
        device=device, dtype=model_dtype)
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=False)
    return gen.sequences[0, -action_dim:].detach().cpu().numpy(), gen


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
# P1-3: save trigger-moment values
trigger_close_onset = False
trigger_close_streak = 0
trigger_clean_gripper_raw = 0.0
trigger_clean_gripper_env = 0.0

event_start = -1
event_end = -1
perturb_frame_count = 0
# Wave 0 attack telemetry
adv_inputs_used = False
fallback_detected = False
autoregressive_raw = None
naturally_open_skip = 0
eligible_close_opportunities = 0
close_streak = 0
close_opportunity_idx = 0
open_count_B3 = 0
b3_opportunities_available = 0
qpos_abs_at_trigger = None
qpos_abs_peak_post_trigger = None
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
C2O_native_count = 0
C2O_clip_count = 0
attacked_close_count = 0
episode_infra_status = 'ok'
trace_rows = []
step = 0

done = False

while step < max_steps:
    if 'agentview_image' not in obs:
        episode_infra_status = "missing_camera"
        break

    img_uint8 = obs['agentview_image']

    # ── V4 EXACT clean decode (P0-3: raw instruction, decode_with_scores wraps prompt) ──
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
    clean_close = clean_gripper_raw < 0.5
    clean_open = clean_gripper_raw > 0.5
    clean_boundary = not clean_close and not clean_open

    if clean_close:
        close_streak += 1
    else:
        close_streak = 0
    close_onset = clean_close and prev_clean_open
    prev_clean_open = clean_open

    # ── Online trigger (RULE_TRIGGER_MVP: first pre-success clean CLOSE onset) ──
    if (not trigger_found and clean_close and close_onset
            and not success_primary):
        trigger_found = True
        trigger_step = step
        # P1-3: save trigger-moment values
        trigger_close_onset = True
        trigger_close_streak = close_streak
        trigger_clean_gripper_raw = clean_gripper_raw
        trigger_clean_gripper_env = clean_gripper_env
        event_start = step
        event_end = min(step + args.event_horizon, max_steps)
        print('[%s] TRIGGER step=%d onset=%d streak=%d clean_grip=%.3f' % (
            time.strftime('%H:%M:%S'), step,
            int(close_onset), close_streak, clean_gripper_raw),
            flush=True)

    # ── Qpos before ──
    gripper_phys_before = physical_gripper_state(env, obs)
    gripper_qpos_before = float(
        np.sum(gripper_phys_before.get('qpos', [0.0])))
    qpos_abs_before = float(gripper_phys_before.get('qpos_abs_sum', 0) or 0.0)
    gripper_qpos_vector_before = [float(v) for v in gripper_phys_before.get('qpos', [])]

    # ── Attack decision ──
    in_event = trigger_found and event_start <= step < event_end
    attack_this_step = False
    pgd_applied = 0
    perturbation_space = 'none'
    within_first_B3 = False  # set below if in-event clean_close
    # Reset per-step telemetry
    adv_inputs_used = False
    fallback_detected = False
    autoregressive_raw = None
    step_result_method = ''
    step_actual_linf = ''
    step_fallback_reason = ''
    step_any_token_clipped = False
    step_gripper_token_id = ''
    step_gripper_disc_before = ''
    step_gripper_disc_after = ''
    step_gripper_clipped = False
    step_gripper_region = ''
    result_objective = ''
    env_action = clean_env_action.copy()
    executed_action = clean_action.copy()
    infra_status = 'ok'
    is_attack_condition = args.condition in (
        'online_vis_pgd', 'online_random_linf')

    if in_event and clean_close and perturb_frame_count < args.max_perturb_frames:
        eligible_close_opportunities += 1
        close_opportunity_idx += 1
        within_first_B3 = close_opportunity_idx <= 3
        if within_first_B3:
            b3_opportunities_available += 1

        if is_attack_condition:
            attack_this_step = True

        if args.condition == 'online_vis_pgd' and attacker is not None:
            try:
                # P0-1: use get_adv_inputs_from_attack_result, NOT x_adv
                attack_result = attacker.attack(
                    img_uint8, instruction,
                    clean_action, clean_action, gen_out,
                    unnorm_key=unnorm_key)
                if attack_result is None:
                    raise RuntimeError(
                        "V6 HARD FAIL: attack_result is None")

                # Wave 0: hard-fail on wrong method (uses attacker.method, not .config)
                if attacker.method != 'token_prefix_pgd':
                    raise RuntimeError(
                        f"V6 HARD FAIL: attacker method={attacker.method}, expected token_prefix_pgd")

                # Wave 0: hard-fail on wrong result method
                result_method = getattr(attack_result, 'attack_method', 'none')
                if result_method == 'visual_linf_noise_adapter' or 'fallback' in str(result_method).lower():
                    raise RuntimeError(
                        f"V6 HARD FAIL: attack_result.attack_method={result_method} (fallback/noise adapter)")
                if not str(result_method).startswith('token_prefix_pgd'):
                    raise RuntimeError(
                        f"V6 HARD FAIL: attack_result.attack_method={result_method}, expected token_prefix_pgd_*")

                # Wave 0: hard-fail on fallback in debug
                debug_info = getattr(attack_result, 'debug', None) or {}
                if debug_info.get('fallback', False) or debug_info.get('fallback_reason', ''):
                    raise RuntimeError(
                        f"V6 HARD FAIL: fallback detected: {debug_info.get('fallback_reason', 'unknown')}")

                adv_inputs = get_adv_inputs_from_attack_result(
                    attack_result)
                if (adv_inputs is None
                        or adv_inputs.get("input_ids") is None):
                    raise RuntimeError(
                        "V6 HARD FAIL: adv_inputs missing")

                adv_inputs_used = True

                # P0-2: hard-assert actual perturbation budget
                _BUDGET_ATOL = 5e-5  # ~0.013/255, float32 PGD accumulation tolerance
                actual_linf = float(debug_info.get('pixel_budget_adv_inputs_linf', 999))
                master_linf = float(debug_info.get('pixel_budget_master_linf', 999))
                if actual_linf > eps_norm + _BUDGET_ATOL:
                    raise RuntimeError(f"V6 HARD FAIL: actual Linf {actual_linf:.8f} > eps_norm {eps_norm:.8f} + {_BUDGET_ATOL}")
                if master_linf > eps_norm + _BUDGET_ATOL:
                    raise RuntimeError(f"V6 HARD FAIL: master Linf {master_linf:.8f} > eps_norm {eps_norm:.8f} + {_BUDGET_ATOL}")
                # Consistency: observation_perturb_linf must match pixel_budget_adv_inputs_linf
                obs_perturb_linf = float(getattr(attack_result, 'observation_perturb_linf', 999))
                if abs(obs_perturb_linf - actual_linf) > 1e-8:
                    raise RuntimeError(f"V6 HARD FAIL: Linf mismatch obs={obs_perturb_linf:.8f} vs adv_inputs={actual_linf:.8f}")
                result_pgd_steps = int(getattr(attack_result, 'num_attack_steps', 0))
                if result_pgd_steps != args.pgd_steps:
                    raise RuntimeError(f"V6 HARD FAIL: PGD steps {result_pgd_steps} != {args.pgd_steps}")
                # P0-2: exact objective match
                EXPECTED_OBJECTIVE = str(args.attack_objective)
                result_objective = str(debug_info.get('attack_objective', ''))
                if result_objective != EXPECTED_OBJECTIVE:
                    raise RuntimeError(f"V6 HARD FAIL: objective={result_objective}, expected={EXPECTED_OBJECTIVE}")
                if getattr(attack_result, 'x_adv', None) is not None:
                    raise RuntimeError("V6 HARD FAIL: x_adv is not None (should use adv_inputs)")

                # Save actual telemetry from attack_result
                step_result_method = result_method
                step_actual_linf = actual_linf
                step_fallback_reason = str(debug_info.get('fallback_reason', ''))

                # Official OpenVLA decoder: token_ids → np.clip → action
                # No hard-fail for out-of-native-range tokens — official pipeline clips them.
                token_ids, _ = generate_from_adv_inputs(
                    adv_inputs, device, model_dtype, action_dim)
                n_bins = int(model.bin_centers.shape[0])
                vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
                generated_tokens = [int(t) for t in token_ids[-action_dim:]]
                gripper_token_id = int(generated_tokens[-1])

                # Clip telemetry (observation only, no control flow)
                disc_before = [vocab_eff - tid - 1 for tid in generated_tokens]
                disc_after = [max(0, min(n_bins - 1, d)) for d in disc_before]
                any_clipped = any(d != da for d, da in zip(disc_before, disc_after))
                gripper_disc_before = vocab_eff - gripper_token_id - 1
                gripper_disc_after = max(0, min(n_bins - 1, gripper_disc_before))
                gripper_clipped = gripper_disc_before != gripper_disc_after

                # Official decode (with np.clip inside decode_action_from_token_ids)
                adv_action = decode_action_from_token_ids(token_ids)
                autoregressive_raw = adv_action
                gripper_raw = float(adv_action[-1])
                if gripper_raw > 0.5:
                    gripper_region = 'OPEN'
                elif gripper_raw < 0.5:
                    gripper_region = 'CLOSE'
                else:
                    gripper_region = 'EXACT_BOUNDARY'

                # Clip telemetry to step variables
                step_gripper_token_id = gripper_token_id
                step_gripper_disc_before = gripper_disc_before
                step_gripper_disc_after = gripper_disc_after
                step_gripper_clipped = gripper_clipped
                step_any_token_clipped = any_clipped
                step_gripper_region = gripper_region

                adv_env_action = postprocess_openvla_action_for_libero(
                    adv_action, enabled=True)
                env_action = adv_env_action
                executed_action = adv_action
                pgd_applied = 1
                perturbation_space = 'online_token_prefix_pgd_v6'
                attacked_close_count += 1
                perturb_frame_count += 1
            except Exception as e:
                episode_infra_status = 'vis_error: %s' % str(e)[:80]
                raise

        elif args.condition == 'online_random_linf':
            try:
                # P0-4: visual RAND — perturb pixel_values, re-decode
                # Apply L∞ noise in pixel_values space (same as VIS perturbation space)
                rand_noise = torch.empty_like(
                    gen_out.sequences)  # placeholder — we need pixel_values
                # Re-run processor to get clean pixel_values
                from v4_run_eval_openvla import prepare_openvla_image
                v4_image = prepare_openvla_image(
                    img_uint8, libero_official_preprocess=False,
                    center_crop=True, resize_size=224,
                    libero_preprocess_backend='official_pil_lanczos')
                rand_inputs = processor(
                    text=prompt(str(instruction).lower()), images=v4_image,
                    return_tensors='pt')
                rand_inputs.pop("attention_mask", None)
                in_ids = rand_inputs.get("input_ids")
                if in_ids is not None and not torch.all(
                        in_ids[:, -1] == 29871):
                    rand_inputs["input_ids"] = torch.cat((
                        in_ids, torch.unsqueeze(
                            torch.tensor([29871]).long(), dim=0).to(
                            in_ids.device)), dim=1)
                rand_inputs = {
                    k: v.to(device=device,
                            dtype=model_dtype
                            if v.dtype in (torch.float32, torch.bfloat16)
                            else v.dtype)
                    for k, v in rand_inputs.items()}
                # P0-3: RAND budget audit with BF16 cast correction
                _BUDGET_ATOL = 5e-5  # ~0.013/255, float32 PGD accumulation tolerance
                pv0 = rand_inputs["pixel_values"].detach()
                pv0_fp32 = pv0.float()
                g_rand = torch.Generator(device=pv0.device)
                g_rand.manual_seed(args.attack_seed + step)
                noise = torch.empty(pv0.shape, device=pv0.device, dtype=torch.float32).uniform_(-eps_norm, eps_norm, generator=g_rand)
                rand_fp32 = torch.maximum(torch.minimum(pv0_fp32 + noise, pv0_fp32 + eps_norm), pv0_fp32 - eps_norm)
                rand_model = rand_fp32.to(dtype=model_dtype)
                # Correct BF16 rounding overshoot
                over_budget = (rand_model.float() - pv0_fp32).abs() > eps_norm + _BUDGET_ATOL
                if over_budget.any():
                    rand_model = torch.where(over_budget, pv0, rand_model)
                # Hard-assert RAND budget
                rand_linf = float((rand_model.float() - pv0_fp32).abs().max().cpu())
                if rand_linf > eps_norm + _BUDGET_ATOL:
                    raise RuntimeError(f"V6 HARD FAIL: RAND Linf {rand_linf:.8f} > eps_norm {eps_norm:.8f}")
                step_actual_linf = rand_linf
                rand_inputs["pixel_values"] = rand_model

                # Re-decode
                with torch.inference_mode():
                    rand_gen = model.generate(
                        **rand_inputs, max_new_tokens=action_dim,
                        do_sample=False, return_dict_in_generate=True,
                        output_scores=False)
                rand_tids = rand_gen.sequences[
                    0, -action_dim:].detach().cpu().numpy()

                # Official OpenVLA decoder: token_ids → np.clip → action
                vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
                n_bins = int(model.bin_centers.shape[0])
                rand_generated = [int(t) for t in rand_tids]
                rand_gripper_tid = int(rand_generated[-1])

                # Clip telemetry (observation only)
                rand_disc_before = [vocab_eff - tid - 1 for tid in rand_generated]
                rand_disc_after = [max(0, min(n_bins - 1, d)) for d in rand_disc_before]
                rand_any_clipped = any(d != da for d, da in zip(rand_disc_before, rand_disc_after))
                rand_gripper_disc_before = vocab_eff - rand_gripper_tid - 1
                rand_gripper_disc_after = max(0, min(n_bins - 1, rand_gripper_disc_before))
                rand_gripper_clipped = rand_gripper_disc_before != rand_gripper_disc_after

                rand_action = decode_action_from_token_ids(rand_tids)
                rand_env_action = postprocess_openvla_action_for_libero(
                    rand_action, enabled=True)
                env_action = rand_env_action
                executed_action = rand_action
                pgd_applied = 2
                perturbation_space = 'online_random_linf_pixel_v6'
                attacked_close_count += 1
                perturb_frame_count += 1

                # RAND audit telemetry
                step_result_method = 'random_linf_pixel_values'
                step_actual_linf = rand_linf
                step_gripper_token_id = rand_gripper_tid
                step_gripper_disc_before = rand_gripper_disc_before
                step_gripper_disc_after = rand_gripper_disc_after
                step_gripper_clipped = rand_gripper_clipped
                step_any_token_clipped = rand_any_clipped
                rand_gripper_raw = float(rand_action[-1])
                if rand_gripper_raw > 0.5:
                    step_gripper_region = 'OPEN'
                elif rand_gripper_raw < 0.5:
                    step_gripper_region = 'CLOSE'
                else:
                    step_gripper_region = 'EXACT_BOUNDARY'
                autoregressive_raw = rand_action
            except Exception as e:
                episode_infra_status = 'rand_error: %s' % str(e)[:80]
                raise

    elif in_event and not clean_close:
        naturally_open_skip += 1

    # ── Env step ──
    eef_before = (env.env.robots[0]._hand_pos
                  if hasattr(env.env.robots[0], '_hand_pos') else None)
    obs, reward, done, info = env.step(env_action)

    gripper_phys_after = physical_gripper_state(env, obs)
    gripper_qpos_after = float(
        np.sum(gripper_phys_after.get('qpos', [0.0])))
    qpos_abs_after = float(gripper_phys_after.get('qpos_abs_sum', 0) or 0.0)
    gripper_qpos_vector_after = [float(v) for v in gripper_phys_after.get('qpos', [])]
    qpos_opening_delta = gripper_qpos_before - gripper_qpos_after  # positive=opening
    qpos_abs_delta = abs(qpos_abs_after - qpos_abs_before)
    is_open = int(env_action[-1] < -0.5)

    # C2O metrics: official execution is PRIMARY
    executed_raw_gripper = float(executed_action[-1])
    executed_env_gripper = float(env_action[-1])
    c2o_official = int(clean_close and executed_env_gripper < -0.5)
    c2o_native_open = int(clean_close and executed_raw_gripper > 0.5 and executed_env_gripper < -0.5)
    c2o_clip_mediated = int(c2o_official and bool(step_gripper_clipped))
    c2o_this_step = c2o_official
    C2O_count += c2o_official
    C2O_native_count += c2o_native_open
    C2O_clip_count += c2o_clip_mediated
    if c2o_official and within_first_B3:
        open_count_B3 += 1

    # Physical tracking post-trigger (fixed window: trigger+7)
    if trigger_found and qpos_abs_at_trigger is None:
        qpos_abs_at_trigger = qpos_abs_before
    if trigger_found and step <= trigger_step + 7:
        if qpos_abs_peak_post_trigger is None or qpos_abs_after > qpos_abs_peak_post_trigger:
            qpos_abs_peak_post_trigger = qpos_abs_after

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
        'close_opportunity_idx': close_opportunity_idx if in_event and clean_close else '',
        'within_first_B3': int(within_first_B3),
        'gripper_qpos_before': round(gripper_qpos_before, 6),
        'gripper_qpos_after': round(gripper_qpos_after, 6),
        'qpos_abs_before': qpos_abs_before,
        'qpos_abs_after': qpos_abs_after,
        'qpos_abs_delta': qpos_abs_delta,
        'gripper_qpos_vector_before': gripper_qpos_vector_before,
        'gripper_qpos_vector_after': gripper_qpos_vector_after,
        'physical_gripper_opening_delta': round(
            gripper_qpos_before - gripper_qpos_after, 6),
        'gripper_qpos_delta': round(
            gripper_qpos_after - gripper_qpos_before, 6),
        'eef_x': round(float(eef_before[0]), 6)
        if eef_before is not None else '',
        'eef_y': round(float(eef_before[1]), 6)
        if eef_before is not None else '',
        'eef_z': round(float(eef_before[2]), 6)
        if eef_before is not None else '',
        'eef_z_after': '',
        'obj_x': '', 'obj_y': '', 'obj_z': '', 'obj_z_after': '',
        'target_object_name': '',
        'pgd_applied': pgd_applied,
        'perturbation_space': perturbation_space,
        'success_done': int(success_done),
        'success_check': int(success_check),
        'success_primary': int(success_primary_now),
        'attack_seed': args.attack_seed, 'job_id': args.job_id,
        'infra_status': episode_infra_status,
        'window_start': '', 'window_end': '',
        'trigger_found': int(trigger_found),
        'trigger_step': trigger_step,
        'clean_close': int(clean_close),
        'close_onset': int(close_onset),
        'close_streak': close_streak,
        'c2o_this_step': c2o_this_step,
        # Wave 0 audit-ready per-step telemetry
        'step_attack_method': step_result_method if (attack_this_step and is_attack_condition) else ('random_linf_pixel_values' if (args.condition == 'online_random_linf' and attack_this_step) else ''),
        'step_attack_objective': result_objective if (attack_this_step and is_attack_condition) else '',
        'step_adv_inputs_used': int(adv_inputs_used) if (attack_this_step and is_attack_condition) else '',
        'step_fallback_detected': int(bool(step_fallback_reason)) if (attack_this_step and is_attack_condition) else '',
        'step_fallback_reason': step_fallback_reason if args.condition == 'online_vis_pgd' else '',
        'step_actual_linf': step_actual_linf if (attack_this_step and is_attack_condition) else '',
        'step_any_token_clipped': int(step_any_token_clipped) if (attack_this_step and is_attack_condition) else '',
        'step_gripper_token_id': step_gripper_token_id if (attack_this_step and is_attack_condition) else '',
        'step_gripper_disc_before': step_gripper_disc_before if (attack_this_step and is_attack_condition) else '',
        'step_gripper_disc_after': step_gripper_disc_after if (attack_this_step and is_attack_condition) else '',
        'step_gripper_clipped': int(step_gripper_clipped) if (attack_this_step and is_attack_condition) else '',
        'step_gripper_region': step_gripper_region if (attack_this_step and is_attack_condition) else '',
        'step_clean_gripper_raw': round(clean_gripper_raw, 6),
        'step_executed_gripper_raw': round(float(autoregressive_raw[-1]), 6) if (attack_this_step and is_attack_condition and autoregressive_raw is not None) else '',
        'c2o_official': c2o_official,
        'c2o_native_open': c2o_native_open,
        'c2o_clip_mediated': c2o_clip_mediated,
    })

    step += 1
    if success_primary or done:
        break

env.close()

# ── Summary (P1-3: trigger-moment values, P1-4: real infra_status) ──
safe_tag = '%s_s%d_v6_%s_seed%d' % (
    args.task, args.state_id, args.condition, args.attack_seed)
summary = {
    'runner_family': 's20d_v6_online_trigger_l3',
    'vis_runner_version': 'v6_online_trigger_execspec_v2_auditready',
    'runner_sha256': _runner_sha256,
    'adapter_sha256': _adapter_sha256,
    'semantics_sha256': _semantics_sha256,
    'exec_spec_sha256': _spec_sha256,
    'adapter_path': _actual_adapter_path,
    'semantics_path': _actual_semantics_path,
    'exec_spec_path': _actual_spec_path,
    'function_identity_ok': _function_identity_ok,
    # Condition-aware attack provenance
    'attack_method': 'token_prefix_pgd' if args.condition == 'online_vis_pgd' else ('random_linf_pixel_values' if args.condition == 'online_random_linf' else 'none'),
    'attack_objective': args.attack_objective if args.condition == 'online_vis_pgd' else ('none' if args.condition == 'clean_observer' else 'random_linf'),
    'attack_margin': 0.5 if args.condition == 'online_vis_pgd' else '',
    'attack_pgd_steps': args.pgd_steps if args.condition == 'online_vis_pgd' else 0,
    'attack_eps_raw_pixels': args.eps_raw_pixels if args.condition in ('online_vis_pgd', 'online_random_linf') else 0,
    'task': args.task, 'state_id': args.state_id,
    'condition': args.condition,
    'attack_seed': args.attack_seed, 'job_id': args.job_id,
    'trigger_found': trigger_found,
    'trigger_step': trigger_step,
    'trigger_close_onset': trigger_close_onset,
    'trigger_close_streak': trigger_close_streak,
    'trigger_clean_gripper_raw': trigger_clean_gripper_raw,
    'trigger_clean_gripper_env': trigger_clean_gripper_env,
    'eligible_close_opportunities': eligible_close_opportunities,
    'perturb_frame_count': perturb_frame_count,
    'naturally_open_skip': naturally_open_skip,
    'C2O_count': C2O_count,
    'C2O_native_open_count': C2O_native_count,
    'C2O_clip_mediated_count': C2O_clip_count,
    'open_count_B3': open_count_B3,
    'b3_opportunities_available': b3_opportunities_available,
    'open_duty_B3': round(open_count_B3 / max(b3_opportunities_available, 1), 4),
    'qpos_abs_at_trigger': qpos_abs_at_trigger if qpos_abs_at_trigger is not None else '',
    'qpos_abs_peak_post_trigger': qpos_abs_peak_post_trigger if qpos_abs_peak_post_trigger is not None else '',
    'qpos_abs_peak_delta': round(float(qpos_abs_peak_post_trigger or 0) - float(qpos_abs_at_trigger or 0), 8) if qpos_abs_at_trigger is not None else '',
    'attacked_close_count': attacked_close_count,
    'decoded_open_count': total_decoded_open,
    'max_open_streak': max_open_streak,
    'n_steps': step,
        'n_pgd_applied_steps': sum(1 for r in trace_rows if r.get('pgd_applied', 0)),
        'success_step_primary': success_step_primary,
        'done_step_any': done_step_any,
    'max_steps': max_steps,
    'success_primary': success_primary,
    'success_done_any': success_done_any,
    'success_check_any': success_check_any,
    'timeout': not done and not success_primary,
    'infra_status': episode_infra_status,
    'event_horizon': args.event_horizon,
    'eps_raw_pixels': args.eps_raw_pixels,
    'pgd_steps': args.pgd_steps,
    'model_path': args.model_path,
    'decode_path': 'v4_decode_with_scores',
    'preprocess_path': 'v4_center_crop_224_official_pil_lanczos',
    'postprocess_path': 'v4_enabled_true',
    'attention_mask': 'v4_drop',
    'eos_token': 'v4_29871',
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

print('[%s] V6 DONE. trigger=%s step=%d C2O=%d attacked=%d/%d '
      'opens=%d/%d infra=%s' % (
          time.strftime('%H:%M:%S'), trigger_found, trigger_step,
          C2O_count, attacked_close_count, eligible_close_opportunities,
          total_decoded_open, step, episode_infra_status), flush=True)
