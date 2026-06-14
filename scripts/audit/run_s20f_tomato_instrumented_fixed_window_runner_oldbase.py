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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from v4_run_eval_openvla import (
    decode_with_scores, postprocess_openvla_action_for_libero,
    physical_gripper_state, prompt,
)
def load_model_s20d(model_path, model_gpu_device_id=-1):
    """V4 load_model with use_fast=True to avoid protobuf dependency (missing in env)."""
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

os.environ.setdefault("OPENVLA_RENDER_LOCAL_DEVICE", str(args.render_gpu_device_id))

# ── Model loading (EXACT V4 path) ──
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model from {args.model_path}", flush=True)
model, processor, device = load_model_s20d(args.model_path, model_gpu_device_id=args.model_gpu_device_id)
model_dtype = next(model.parameters()).dtype
print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded on {device} dtype={model_dtype}", flush=True)

# ── LIBERO env setup (V4 pattern) ──
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.grasp import eef_pos, object_pos

# Best-effort target object name resolution (non-critical: falls back to None)
TARGET_OBJECT_GUESS = {
    'ketchup': 'ketchup_green_bottle_1',
    'tomato_sauce': 'tomato_sauce_bottle_1',
    'milk': 'milk_carton_1',
    'butter': 'butter_box_1',
    'cream_cheese': 'cream_cheese_box_1',
    'salad_dressing': 'salad_dressing_bottle_1',
    'bbq_sauce': 'bbq_sauce_bottle_1',
    'alphabet_soup': 'alphabet_soup_can_1',
    'orange_juice': 'orange_juice_carton_1',
    'chocolate_pudding': 'chocolate_pudding_box_1',
}

def get_object_pose_safe(env, obj_name):
    """Non-fatal object pose query. Returns None on failure."""
    try:
        return object_pos(env, obj_name)
    except Exception:
        return None

def resolve_target_object(env, task_name):
    """Try to find the target object in the env. Falls back to guess table."""
    candidate = TARGET_OBJECT_GUESS.get(task_name, '')
    if candidate:
        try:
            _ = env.sim.model.body_name2id(candidate)
            return candidate
        except Exception:
            pass
    # Scan all bodies for likely target (not robot, table, basket)
    try:
        for i in range(env.sim.model.nbody):
            try:
                name = env.sim.model.body_id2name(i)
                ln = name.lower()
                if any(kw in ln for kw in ['robot', 'table', 'basket', 'bin', 'world', 'floor']):
                    continue
                if any(kw in ln for kw in ['bottle', 'box', 'carton', 'can', 'bowl', 'cream', 'butter',
                                            'ketchup', 'tomato', 'milk', 'chocolate', 'pudding',
                                            'sauce', 'soup', 'salad', 'juice', 'cheese', 'dressing',
                                            'bbq', 'orange', 'alphabet']):
                    return name
            except Exception:
                continue
    except Exception:
        pass
    return None

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


def raw_gripper_to_env(raw):
    return -1.0 if float(raw) > 0.5 else 1.0


def raw_gripper_class(raw):
    raw = float(raw)
    if abs(raw - 0.5) <= 1e-9:
        return "NATIVE_BOUNDARY"
    if raw > 0.5:
        return "NATIVE_OPEN"
    return "NATIVE_CLOSE"


def extract_exact_tokens(gen):
    if gen is None or not hasattr(gen, "sequences"):
        raise ValueError("generation output missing sequences")
    values = [int(x) for x in gen.sequences[0].detach().cpu().tolist()]
    if hasattr(gen, "prompt_len"):
        new_tokens = values[int(gen.prompt_len):]
    else:
        new_tokens = values[-int(action_dim):]
    if len(new_tokens) != int(action_dim):
        raise ValueError("expected %d new tokens, got %d" % (int(action_dim), len(new_tokens)))
    return new_tokens


def classify_token(token_id, vocab_eff, n_bins, stats):
    token_id = int(token_id)
    disc_before = int(vocab_eff - token_id - 1)
    disc_after = max(0, min(int(n_bins) - 1, disc_before))
    clipped = disc_before != disc_after
    center = float(model.bin_centers[disc_after])
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi = np.asarray(stats["q99"], dtype=np.float32)
    lo = np.asarray(stats["q01"], dtype=np.float32)
    gdim = len(hi) - 1
    if bool(mask[gdim]):
        raw = float(0.5 * (center + 1.0) * (hi[gdim] - lo[gdim]) + lo[gdim])
    else:
        raw = float(center)
    env = raw_gripper_to_env(raw)
    if clipped:
        cls = "CLIP_MEDIATED_OPEN" if env < -0.5 else "CLIP_MEDIATED_CLOSE"
    else:
        cls = raw_gripper_class(raw)
    return {
        "token_id": token_id,
        "disc_before": disc_before,
        "disc_after": disc_after,
        "clipped": bool(clipped),
        "decoded_raw_gripper": round(raw, 8),
        "executed_env_gripper": round(env, 6),
        "execution_class": cls,
    }


def score_row_audit(row, emitted_token, vocab_eff, n_bins, stats):
    if hasattr(row, "detach"):
        row_cpu = row.detach().float().cpu()
    else:
        row_cpu = torch.as_tensor(row).float().cpu()
    topk = torch.topk(row_cpu, k=min(2, int(row_cpu.numel())))
    top_tokens = [int(x) for x in topk.indices.tolist()]
    top_scores = [float(x) for x in topk.values.tolist()]
    best = {
        "open": {"token": None, "score": None},
        "close": {"token": None, "score": None},
        "boundary": {"token": None, "score": None},
    }
    native_start = max(0, int(vocab_eff) - int(n_bins))
    native_end = int(vocab_eff)
    for tid in range(native_start, native_end):
        info = classify_token(tid, vocab_eff, n_bins, stats)
        cls = info["execution_class"]
        key = None
        if cls == "NATIVE_OPEN":
            key = "open"
        elif cls == "NATIVE_CLOSE":
            key = "close"
        elif cls == "NATIVE_BOUNDARY":
            key = "boundary"
        if key is None:
            continue
        score = float(row_cpu[tid])
        if best[key]["score"] is None or score > float(best[key]["score"]):
            best[key] = {"token": int(tid), "score": score}
    top1 = top_tokens[0]
    audit = {
        "top1_token": top1,
        "top1_score": top_scores[0],
        "top2_token": top_tokens[1] if len(top_tokens) > 1 else None,
        "top2_score": top_scores[1] if len(top_scores) > 1 else None,
        "top1_minus_top2_gap": (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else None,
        "best_native_open_token": best["open"]["token"],
        "best_native_open_score": best["open"]["score"],
        "best_native_close_token": best["close"]["token"],
        "best_native_close_score": best["close"]["score"],
        "best_native_boundary_token": best["boundary"]["token"],
        "best_native_boundary_score": best["boundary"]["score"],
        "emitted_token": int(emitted_token),
        "generation_score_argmax": top1,
        "processed_score_argmax_token": top1,
        "argmax_matches_emitted": int(top1) == int(emitted_token),
    }
    for tid in (31744, 31872):
        audit["score_token_%d" % tid] = float(row_cpu[tid]) if tid < row_cpu.numel() else None
    return audit


def generation_token_telemetry(gen, *, label, emitted_action):
    """Flatten exact action-token and final-score telemetry for trace CSV rows."""
    prefix = f"{label}_"
    out = {
        prefix + "generated_action_token_ids": "",
        prefix + "generated_arm_prefix_token_ids": "",
        prefix + "generated_gripper_token_id": "",
        prefix + "token_execution_class": "",
        prefix + "token_decoded_raw_gripper": "",
        prefix + "token_executed_env_gripper": "",
        prefix + "generation_score_argmax": "",
        prefix + "generation_score_invariant_ok": "",
        prefix + "generation_score_invariant_reason": "",
        prefix + "generation_score_audit_json": "",
    }
    try:
        tokens = extract_exact_tokens(gen)
        gripper_token = int(tokens[-1])
        vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        n_bins = int(model.bin_centers.shape[0])
        stats = model.get_action_stats(unnorm_key)
        exec_info = classify_token(gripper_token, vocab_eff, n_bins, stats)
        audit = {}
        if getattr(gen, "scores", None):
            audit = score_row_audit(gen.scores[-1][0], gripper_token, vocab_eff, n_bins, stats)
            ok = int(audit.get("generation_score_argmax")) == int(gripper_token)
            reason = "" if ok else "GENERATE_SCORE_ARGMAX_MISMATCH"
        else:
            ok, reason = False, "GENERATION_SCORES_MISSING"
        out.update({
            prefix + "generated_action_token_ids": json.dumps([int(x) for x in tokens]),
            prefix + "generated_arm_prefix_token_ids": json.dumps([int(x) for x in tokens[:6]]),
            prefix + "generated_gripper_token_id": gripper_token,
            prefix + "token_execution_class": exec_info.get("execution_class", ""),
            prefix + "token_decoded_raw_gripper": exec_info.get("decoded_raw_gripper", ""),
            prefix + "token_executed_env_gripper": exec_info.get("executed_env_gripper", ""),
            prefix + "generation_score_argmax": audit.get("generation_score_argmax", ""),
            prefix + "generation_score_invariant_ok": int(bool(ok)),
            prefix + "generation_score_invariant_reason": reason,
            prefix + "generation_score_audit_json": json.dumps(audit, sort_keys=True),
        })
    except Exception as exc:
        out[prefix + "generation_score_invariant_ok"] = 0
        out[prefix + "generation_score_invariant_reason"] = (
            "TOKEN_TELEMETRY_ERROR:" + str(exc)[:160])
    return out

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

    target_object_name = resolve_target_object(env, args.task)
    if target_object_name is None:
        target_object_name = TARGET_OBJECT_GUESS.get(args.task, 'akita_black_bowl_1')

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
        clean_token_telemetry = generation_token_telemetry(
            gen_out, label="clean", emitted_action=clean_action)
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
        executed_gen_out = gen_out

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
                        adv_action, _, _, adv_gen = decode_with_scores(
                            model, processor, device,
                            adv_img, instruction, unnorm_key, K_trigger,
                            libero_official_preprocess=False,
                            libero_preprocess_backend='official_pil_lanczos',
                            center_crop=True, resize_size=224,
                            drop_attention_mask=True)
                        adv_env_action = postprocess_openvla_action_for_libero(adv_action, enabled=True)
                        env_action = adv_env_action
                        executed_action = adv_action
                        executed_gen_out = adv_gen
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
                            output_scores=True)
                    try:
                        gen_rand.prompt_input_ids = input_ids_val.detach().cpu()
                        gen_rand.prompt_len = int(input_ids_val.shape[1])
                    except Exception:
                        pass
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
                    executed_gen_out = gen_rand
                    perturbation_space = 'random_linf_v4'

                except Exception as e:
                    infra_status = f'random_error: {str(e)[:80]}'

        # LibERO OPEN command convention: env_action[-1] < -0.5 means OPEN
        is_open = int(env_action[-1] < -0.5)
        executed_token_telemetry = generation_token_telemetry(
            executed_gen_out, label="executed", emitted_action=executed_action)

        # ── Non-invasive pose logging (before step) ──
        eef_before = eef_pos(env) if eef_pos is not None else None
        obj_before = get_object_pose_safe(env, target_object_name)

        # ── Step environment ──
        obs, reward, done, info = env.step(env_action)

        # Qpos after step
        gripper_phys_after = physical_gripper_state(env, obs)
        gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))

        # ── Non-invasive pose logging (after step) ──
        eef_after = eef_pos(env) if eef_pos is not None else None
        obj_after = get_object_pose_safe(env, target_object_name)

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

        row = {
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
            'eef_x': round(float(eef_before[0]), 6) if eef_before is not None else '',
            'eef_y': round(float(eef_before[1]), 6) if eef_before is not None else '',
            'eef_z': round(float(eef_before[2]), 6) if eef_before is not None else '',
            'eef_z_after': round(float(eef_after[2]), 6) if eef_after is not None else '',
            'obj_x': round(float(obj_before[0]), 6) if obj_before is not None else '',
            'obj_y': round(float(obj_before[1]), 6) if obj_before is not None else '',
            'obj_z': round(float(obj_before[2]), 6) if obj_before is not None else '',
            'obj_z_after': round(float(obj_after[2]), 6) if obj_after is not None else '',
            'target_object_name': target_object_name or '',
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
        }
        row.update(clean_token_telemetry)
        row.update(executed_token_telemetry)
        trace_rows.append(row)

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
