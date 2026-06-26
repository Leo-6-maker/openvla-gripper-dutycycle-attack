#!/usr/bin/env python3
"""
SC5 MLP-triggered VIS bridge — TELEMETRY V2.

Key additions vs v1:
  1. Per-step: clean/adv policy/exec actions, token IDs, timing breakdown, gripper/eef state
  2. Summary: model SHA, bridge SHA, git HEAD, config hash, objective, timing policy
  3. COMPLETE.json atomic protocol (replaces .done)
  4. ArmLock NAD-compatible telemetry (adv action saved before lock)
  5. Video manifest with SHA256

Protocol frozen: epsilon=6/255, PGD=20, K=10, target=31744, upstream_tf_jpeg, prev_delta, no fallback.
"""
import argparse, copy, csv, hashlib, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object")
EPSILON = 0.023529411764705882; TARGET_TOKEN = 31744; ARM_GATE = 5; PGD_STEPS = 20; K = 10

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["CLEAN","TRUE_T10","RAND_T10","SHUFFLED_T10"])
ap.add_argument("--state_id", type=int, required=True)
ap.add_argument("--anchor", type=int, required=True, help="Teacher anchor (AUDIT ONLY)")
ap.add_argument("--seed_id", type=int, required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--render_gpu", type=int, required=True)
ap.add_argument("--mlp_path", default="outputs/sc5_canonical_eng/sc5_mlp_s2.pt")
ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")
ap.add_argument("--save_video", action="store_true", default=False)
ap.add_argument("--source_commit", default="", help="Git commit SHA (required when --save_video)")
ap.add_argument("--video_fps", type=int, default=10)
ap.add_argument("--frame_stride", type=int, default=2)
ap.add_argument("--libero_preprocess_backend", default="upstream_tf_jpeg",
                choices=["upstream_tf_jpeg", "project_pil_lanczos", "none"])
ap.add_argument("--attack_objective", default="autoregressive_prefix_gripper_target_token_logratio_arm_v3")
ap.add_argument("--arm_lock", action="store_true", default=False)
ap.add_argument("--keep_running", action="store_true", default=False)
ap.add_argument("--trigger_step_override", type=int, default=-1)
ap.add_argument("--eval_seed", type=int, default=-1, help="Environment seed metadata; current fixed protocol requires 0")
args = ap.parse_args()

if args.save_video and not args.source_commit:
    raise ValueError("--source_commit is required when --save_video")

# ── Provenance ──
import subprocess
_git_head = ""
try:
    _git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
except Exception:
    pass
_bridge_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_eval_seed = args.eval_seed if args.eval_seed >= 0 else 0
if args.eval_seed not in (-1, 0):
    raise ValueError(f"--eval_seed must be -1 or 0 in current fixed-env protocol, got {args.eval_seed}")
_timing_policy = "student"
_trigger_source = "mlp_detector"
if args.trigger_step_override >= 0:
    if "early" in str(args.output_dir).lower():
        _timing_policy = "early_shift"
    elif "random" in str(args.output_dir).lower():
        _timing_policy = "random_time"
    _trigger_source = "override"

# ── Backend ──
from gripper_attack.openvla_preprocess import resolve_backend
PREPROCESS_BACKEND = resolve_backend(args.libero_preprocess_backend)
USES_JPEG_ROUNDTRIP = (PREPROCESS_BACKEND == "upstream_tf_jpeg")
print(f"Preprocess: requested={args.libero_preprocess_backend} resolved={PREPROCESS_BACKEND}")

STATE_ID = args.state_id; ANCHOR = args.anchor; IS_ATTACK = args.condition != "CLEAN"
IS_RAND = "RAND" in args.condition; IS_SHUFFLED = "SHUFFLED" in args.condition
ATTACK_FRAMES = K if IS_ATTACK else 0

# ── Model ──
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
_dtype_name = os.environ.get("OPENVLA_DTYPE", "bfloat16")
_attn_name = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager")
_dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
if _dtype_name not in _dtype_map:
    raise RuntimeError("OPENVLA_DTYPE must be bfloat16 or float32, got: %s" % _dtype_name)
_torch_dtype = _dtype_map[_dtype_name]
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=_torch_dtype,
    low_cpu_mem_usage=True, device_map="cuda:0", attn_implementation=_attn_name)
model_dtype = next(model.parameters()).dtype
device = "cuda:0"
_actual_dtype_str = str(model_dtype).replace("torch.", "")
_actual_attn = getattr(model.config, "_attn_implementation", "unknown")
print("Model: device=%s dtype=%s attn=%s" % (device, _actual_dtype_str, _actual_attn))
action_dim = int(model.get_action_dim("libero_object"))

# Compute victim model SHA from checkpoint
import hashlib as _hl
_model_sha = _hl.sha256()
for _pth in sorted(Path(MODEL_PATH).rglob("*.safetensors")):
    _model_sha.update(str(_pth.relative_to(MODEL_PATH)).encode())
    with open(_pth, "rb") as _f:
        while True:
            _chunk = _f.read(1 << 20)
            if not _chunk: break
            _model_sha.update(_chunk)
_victim_model_sha = _model_sha.hexdigest()[:16]
print("Victim model SHA16: %s" % _victim_model_sha)

# ── MLP detector ──
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES
detector = SC5DetectorRuntime(args.mlp_path, tau_corridor=0.3, tau_release=0.3, guard=5)
print("MLP detector: dataset_sha256=%s" % detector.dataset_sha256[:16])
_detector_sha = detector.checkpoint_sha256[:16]
_config_sha = _hl.sha256(json.dumps({
    "epsilon": EPSILON, "pgd_steps": PGD_STEPS, "K": K, "target": TARGET_TOKEN,
    "backend": PREPROCESS_BACKEND, "arm_lock": args.arm_lock,
    "objective": args.attack_objective, "timing": _timing_policy,
}, sort_keys=True).encode()).hexdigest()[:16]

# ── Video ──
_video_raw_frames = []
if args.save_video:
    print("Video: ENABLED (fps=%d, stride=%d)" % (args.video_fps, args.frame_stride))

# ── Attacker ──
attacker = None
if IS_ATTACK and not IS_RAND:
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    from gripper_attack.route_contract import UNTARGETED_OBJECTIVES
    _is_untargeted = args.attack_objective in UNTARGETED_OBJECTIVES
    opt = {"method": "token_prefix_pgd", "objective": args.attack_objective,
           "epsilon": EPSILON, "num_steps": PGD_STEPS,
           "step_size": EPSILON * 0.075, "random_start": True, "prefix_refresh_interval": 1,
           "surrogate_score_path": "cached_autoregressive_generate_v1",
           "strict_route": not _is_untargeted, "allow_fallback": False, "temporal_init": "prev_delta"}
    if not _is_untargeted:
        opt["target_token_id"] = TARGET_TOKEN
        opt["target_execution_class"] = "CLIP_MEDIATED_OPEN"
        opt["gripper_margin"] = 5.0
        opt["arm_preserve_weight"] = 0.5
        opt["arm_gate_min_match_count"] = ARM_GATE
    if IS_SHUFFLED:
        opt["gradient_transform"] = "permute"
        opt["gradient_transform_seed"] = args.seed_id + 100000
    attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
        seed=args.seed_id, preprocess_kwargs={"libero_preprocess_backend": PREPROCESS_BACKEND, "center_crop": True, "resize_size": 224}, device=device)

# ── Env ──
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.v3_generation_parity import extract_exact_new_tokens
from libero.libero import benchmark, get_libero_path

TASK_IDX = args.task_idx
bm = benchmark.get_benchmark_dict(); suite = bm["libero_object"]()
task_obj = suite.get_task(TASK_IDX); init_states = suite.get_task_init_states(TASK_IDX)
bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[STATE_ID])
env, obs = apply_dummy_wait(env, obs, 10)

_task_name = task_obj.name
_obj_key = _task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")
obj_sid = env.sim.model.site_name2id(f"{_obj_key}_1_default_site")
obj_z0 = float(env.sim.data.site_xpos[obj_sid][2])

# ── Streaming adapter ──
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
_streamer = SC5StreamingFeatureAdapterV2()
_mlp_emit = -1
_eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
_prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
_invalid_steps = 0; _first_valid_step = -1

telemetry = []; attack_count = 0; prev_delta_flags = []

for step in range(400):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos",[])) > 0 else float("nan")
    q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos",[])) > 1 else float("nan")
    qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
    gripper_width = abs(q7) + abs(q8) if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))

    # ── EEF velocity (causal backward difference) ──
    eef_valid = np.all(np.isfinite([eef_x, eef_y, eef_z]))
    if _prev_eef is not None and eef_valid:
        eef_vx = eef_x - _prev_eef[0]; eef_vy = eef_y - _prev_eef[1]; eef_vz = eef_z - _prev_eef[2]
    else:
        eef_vx = float("nan"); eef_vy = float("nan"); eef_vz = float("nan")
    if eef_valid:
        _prev_eef = (eef_x, eef_y, eef_z)

    # ── Clean decode ──
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(model, processor, device, raw, instruction, "libero_object", 8,
        libero_preprocess_backend=PREPROCESS_BACKEND,
        center_crop=True, resize_size=224, drop_attention_mask=True)
    t1 = time.perf_counter()
    clean_policy_action = np.asarray(action, dtype=np.float32)
    raw_grip = float(action[-1]); env_grip_c = -1.0 if raw_grip > 0.5 else 1.0
    clean_env_action = postprocess_openvla_action_for_libero(clean_policy_action, enabled=True)

    # ── MLP ONLINE TRIGGER ──
    _feat_valid = False; _feat_error = ""; _feat_25d = {}
    _det_state = detector.state; _det_cp = None; _det_rp = None; _det_pp = None

    if not detector.emitted:
        gripper_ok = not (np.isnan(q7) or np.isnan(q8) or np.isnan(qpos_sum))
        gripper_w = abs(q7)+abs(q8) if gripper_ok else float("nan")
        gripper_q = float(qpos_sum) if gripper_ok else float("nan")

        try:
            _res = _streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=env_grip_c,
                gripper_qpos=gripper_q, gripper_opening_proxy=gripper_w,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
                action_dx=float(action[0]), action_dy=float(action[1]),
                action_dz=float(action[2]), action_gripper=raw_grip)
        except ValueError as e:
            _res = {"valid": False, "error": "step_sequence:%s" % str(e)[:80]}
        except Exception as e:
            _res = {"valid": False, "error": "streamer_error:%s" % type(e).__name__}

        _feat_valid = _res.get("valid", False)
        _feat_error = _res.get("error", "")
        if _feat_valid:
            _feat_25d = dict(_res["features"])
            if _first_valid_step < 0:
                _first_valid_step = step
        else:
            _invalid_steps += 1

        if _feat_valid:
            _decision = detector.update(_res["features"], step)
            _det_state = _decision["state"]
            _det_cp = _decision.get("corridor_p")
            _det_rp = _decision.get("release_p")
            _det_pp = _decision.get("pred_phase")
            if _decision["emitted"]:
                _mlp_emit = _decision["emit_step"]

    # === ATTACK ===
    attack_this = False; prev_flag = False
    _trigger_step = args.trigger_step_override if args.trigger_step_override >= 0 else _mlp_emit
    _effective_trigger = _trigger_step

    adv_policy_before_lock = None
    adv_policy_after_lock = None
    clean_tokens_list = None
    adv_tokens_list = None
    pgd_opt_ms = 0.0
    adv_decode_ms = 0.0
    arm_lock_ms = 0.0
    # Track attack index (0-based within attack window)
    attack_idx = attack_count
    # Total step timing
    clean_fwd_ms = round((t1 - t0) * 1000, 3)
    total_ms = clean_fwd_ms

    if IS_ATTACK and _trigger_step >= 0 and step >= _trigger_step and attack_count < ATTACK_FRAMES:
        _clean_candidate = clean_policy_action.copy()
        t_atk_start = time.perf_counter()

        if IS_RAND:
            from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
            proc_image = prepare_openvla_image_for_attack(raw,
                libero_preprocess_backend=PREPROCESS_BACKEND, center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
            inputs.pop("attention_mask", None)
            iids = inputs["input_ids"].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            x = inputs["pixel_values"].to(device=device, dtype=model_dtype)
            delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=args.seed_id+100000+attack_count, dtype=torch.float32, device=x.device)
            proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
            adv_pv = proj.detach().to(dtype=model_dtype)
            t_after_sample = time.perf_counter()
            pgd_opt_ms = round((t_after_sample - t_atk_start) * 1000, 3)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=adv_pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            t_after_decode = time.perf_counter()
            adv_decode_ms = round((t_after_decode - t_after_sample) * 1000, 3)
            adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
        else:
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
            clean_action_np = np.asarray(action, dtype=np.float32)
            proc_image = prepare_openvla_image_for_attack(raw,
                libero_preprocess_backend=PREPROCESS_BACKEND, center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
            inputs.pop("attention_mask", None)
            iids = inputs["input_ids"].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            pv = inputs["pixel_values"].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            clean_tokens_list = [int(t) for t in clean_tokens]
            # Attack
            clean_gen = type("CleanGen", (), {})()
            clean_gen.sequences = torch.tensor([iids[0].detach().cpu().tolist() + [int(t) for t in clean_tokens]], dtype=torch.long, device=device)
            clean_gen.scores = []
            attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen, unnorm_key="libero_object")
            t_after_pgd = time.perf_counter()
            pgd_opt_ms = round((t_after_pgd - t_atk_start) * 1000, 3)
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)
            adv_pv = adv_inputs["pixel_values"]
            with torch.inference_mode():
                go_adv = model.generate(input_ids=iids, pixel_values=adv_pv.to(device=device, dtype=model_dtype), max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            t_after_adv = time.perf_counter()
            adv_decode_ms = round((t_after_adv - t_after_pgd) * 1000, 3)
            adv_tokens = extract_exact_new_tokens(go_adv.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            prev_flag = attack_result.debug.get("temporal_prev_delta_used", False) if hasattr(attack_result, "debug") else False
            adv_tokens_list = [int(t) for t in adv_tokens]

        # Decode action from tokens
        grip = int(adv_tokens[-1])
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        disc = np.clip(vocab_size - np.array([int(t) for t in adv_tokens]) - 1, 0, model.bin_centers.shape[0]-1)
        na = model.bin_centers[disc]
        s = model.get_action_stats("libero_object")
        lo = np.asarray(s["q01"], dtype=np.float32); hi = np.asarray(s["q99"], dtype=np.float32)
        mk = np.asarray(s.get("mask", np.ones_like(lo, dtype=bool)), dtype=bool)
        attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
        adv_policy_before_lock = attack_action.copy()

        # Arm Execution Lock
        if args.arm_lock:
            arm_lock_ms_start = time.perf_counter()
            _arm_error = np.max(np.abs(attack_action[0:6] - clean_action_np[0:6]))
            attack_action[0:6] = clean_action_np[0:6].copy()
            arm_lock_ms = round((time.perf_counter() - arm_lock_ms_start) * 1000, 6)
            if _arm_error > 0:
                print(f"ArmLock step={step}: max_abs_arm_error={_arm_error:.2e}")

        executed_env_action = postprocess_openvla_action_for_libero(attack_action, enabled=True)
        adv_policy_after_lock = attack_action.copy()

        raw_grip = float(attack_action[-1]); env_grip = float(executed_env_action[-1])
        attack_this = True; attack_count += 1
        prev_delta_flags.append(prev_flag)

        t_end = time.perf_counter()
        total_ms = round((t_end - t0) * 1000, 3)
        env_action_final = executed_env_action
    else:
        env_action_final = clean_env_action
        total_ms = round((time.perf_counter() - t0) * 1000, 3)

    # ── Per-step telemetry (v2) ──
    _tel = {
        # Identity
        "step": step,
        "task": _task_name,
        "state_id": STATE_ID,
        "perturbation_seed": args.seed_id,
        "eval_seed": _eval_seed,
        "condition": args.condition,
        "objective_id": args.attack_objective,
        "timing_policy": _timing_policy,
        "trigger_source": _trigger_source,
        # Detector
        "teacher_anchor": ANCHOR,
        "detector_emit_step": _mlp_emit,
        "trigger_step_override": args.trigger_step_override,
        "effective_trigger_step": _effective_trigger,
        "attack_this": attack_this,
        "attack_index": attack_idx if attack_this else -1,
        # Policy actions (7D)
        "clean_policy_action_7d": json.dumps([float(x) for x in clean_policy_action]),
        "adv_policy_action_7d_before_lock": json.dumps([float(x) for x in adv_policy_before_lock]) if attack_this else "",
        "executed_policy_action_7d_after_lock": json.dumps([float(x) for x in adv_policy_after_lock]) if attack_this else "",
        # Environment actions (7D)
        "clean_env_action_7d": json.dumps([float(x) for x in clean_env_action]),
        "executed_env_action_7d": json.dumps([float(x) for x in env_action_final]),
        # Token IDs
        "clean_token_ids_7d": json.dumps(clean_tokens_list) if clean_tokens_list else "",
        "adv_token_ids_7d": json.dumps(adv_tokens_list) if adv_tokens_list else "",
        "target_token": TARGET_TOKEN if IS_ATTACK else -1,
        # Gripper
        "gripper_qpos_left": q7,
        "gripper_qpos_right": q8,
        "gripper_qpos_sum": qpos_sum,
        "gripper_width": gripper_width,
        "raw_gripper": raw_grip,
        "env_gripper": float(env_action_final[-1]),
        # EEF
        "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
        # Object
        "object_x": obj_x, "object_y": obj_y, "object_z": obj_z,
        "object_eef_distance": eef_obj_dist,
        # Timing
        "clean_forward_ms": clean_fwd_ms,
        "pgd_optimization_ms": pgd_opt_ms,
        "adv_decode_ms": adv_decode_ms,
        "arm_lock_ms": arm_lock_ms,
        "total_step_ms": total_ms,
        # Attack metadata
        "attack_count": attack_count,
        "adv_token": int(adv_tokens[-1]) if attack_this and not IS_RAND else ("" if not attack_this else "RAND"),
        "prev_delta_used": prev_flag,
        # Detector
        "feat_valid": _feat_valid,
        "feat_error": _feat_error,
        "detector_state": _det_state,
        "corridor_p": _det_cp,
        "release_p": _det_rp,
        "pred_phase": _det_pp,
        # Legacy compat
        "model_ms": round(total_ms, 2),
        "qpos_source": "q7+q8_sum",
        "raw_action_7d": json.dumps([float(x) for x in action]),
        "env_action_7d": json.dumps([float(x) for x in env_action_final]),
        "clean_action_7d": json.dumps([float(x) for x in clean_policy_action]),
    }
    if _feat_valid:
        for fn in SC5_FEATURES:
            _tel["f_"+fn] = _feat_25d.get(fn, float("nan"))
    telemetry.append(_tel)

    obs, _, done, _ = env.step(env_action_final)
    if args.save_video and step % args.frame_stride == 0:
        try:
            _raw = obs.get("agentview_image", None)
            if _raw is not None:
                _video_raw_frames.append(np.asarray(copy.deepcopy(_raw)))
        except Exception:
            pass
    if done and not args.keep_running: break

success = bool(env.check_success()) if hasattr(env, "check_success") else False
env.close()

# ── Metrics ──
atk_rows = [r for r in telemetry if r["attack_this"] == True]
n_atk = len(atk_rows)
n_open_token = sum(1 for r in atk_rows if str(r.get("adv_token","")) != "" and str(r["adv_token"]) == str(TARGET_TOKEN))
n_env_open = sum(1 for r in atk_rows if float(r["env_gripper"]) < 0)
n_arm_ok = 0  # arm_duty removed — adv_arm never populated in telemetry v2

# ── Video ──
_video_manifest = {}
if args.save_video and _video_raw_frames:
    try:
        from imageio.v2 import mimwrite as _mimwrite
        out_vdir = Path(args.output_dir); out_vdir.mkdir(parents=True, exist_ok=True)
        _raw_path = out_vdir / "rollout_raw.mp4"
        _mimwrite(str(_raw_path), [np.asarray(f) for f in _video_raw_frames],
                  fps=args.video_fps, codec="libx264", quality=None,
                  output_params=["-preset", "fast", "-crf", "30"])
        print("Video saved: %s (%d frames)" % (_raw_path, len(_video_raw_frames)))
        _video_sha = _hl.sha256()
        with open(_raw_path, "rb") as _vf:
            while True:
                _chunk = _vf.read(1 << 20)
                if not _chunk: break
                _video_sha.update(_chunk)
        _video_manifest = {
            "raw_video_path": str(_raw_path),
            "frame_count": len(_video_raw_frames),
            "fps": args.video_fps,
            "stride": args.frame_stride,
            "source_commit": args.source_commit,
            "video_sha256": _video_sha.hexdigest(),
        }
    except Exception as _ve:
        print("Video encoding failed: %s" % _ve)

# ── Summary (v2) ──
summary = {
    # Identity
    "condition": args.condition,
    "state_id": STATE_ID,
    "task_name": _task_name,
    "task_idx": TASK_IDX,
    "perturbation_seed": args.seed_id,
    "eval_seed": _eval_seed,
    "objective_id": args.attack_objective,
    "arm_lock": args.arm_lock,
    "timing_policy": _timing_policy,
    "trigger_source": _trigger_source,
    # Detector
    "teacher_anchor": ANCHOR if ANCHOR > 0 else None,
    "teacher_anchor_valid": ANCHOR > 0,
    "mlp_emit_step": _mlp_emit,
    "mlp_triggered": detector.emitted,
    "effective_trigger_step": _trigger_step if _trigger_step >= 0 else -1,
    "anchor_error": (_mlp_emit - ANCHOR) if _mlp_emit >= 0 and ANCHOR > 0 else None,
    "invalid_feature_steps": _invalid_steps,
    "first_valid_step": _first_valid_step,
    "manual_anchor_used": False,
    "privileged_detector_input_used": False,
    "effective_env_seed": 0,
    "env_seed_applied": True,
    # Protocol
    "epsilon": EPSILON,
    "pgd_steps": PGD_STEPS,
    "K": K,
    "target_token": TARGET_TOKEN,
    # Provenance
    "victim_model_sha16": _victim_model_sha,
    "detector_checkpoint_sha256": detector.checkpoint_sha256,
    "detector_dataset_sha256": detector.dataset_sha256,
    "bridge_sha256": _bridge_sha,
    "git_head": _git_head,
    "config_sha16": _config_sha,
    "preprocess_backend_requested": args.libero_preprocess_backend,
    "preprocess_backend_resolved": PREPROCESS_BACKEND,
    "preprocess_uses_jpeg_roundtrip": USES_JPEG_ROUNDTRIP,
    # Dtype
    "requested_dtype": _dtype_name,
    "actual_dtype": _actual_dtype_str,
    "requested_attn": _attn_name,
    "actual_attn": _actual_attn,
    # Results
    "n_steps": len(telemetry),
    "attack_frames": n_atk,
    "open_tokens": n_open_token,
    "env_open_frames": n_env_open,
    "token_open_duty": round(n_open_token/n_atk, 3) if n_atk>0 else 0,
    "arm_duty": None, "arm_duty_available": False,
    "env_open_duty": round(n_env_open/n_atk, 3) if n_atk>0 else 0,
    "prev_delta_flags": prev_delta_flags,
    "task_success": success,
}
if _video_manifest:
    summary["video"] = _video_manifest

# ── Write outputs ──
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

# CSV telemetry — write all fields, dynamic from first row
_tel_fields = list(telemetry[0].keys()) if telemetry else []
with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=_tel_fields); w.writeheader(); w.writerows(telemetry)

# Episode summary
with open(out / "episode_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# COMPLETE.json — atomic protocol
_complete = {"status": "COMPLETE", "timestamp": time.time(), "exit_code": 0,
             "bridge_sha256": _bridge_sha, "task_success": success,
             "attack_frames": n_atk, "n_steps": len(telemetry)}
_complete_tmp = out / "COMPLETE.json.tmp"
_f = open(_complete_tmp, "w")
json.dump(_complete, _f)
_f.flush()
os.fsync(_f.fileno())
_f.close()
_complete_tmp.rename(out / "COMPLETE.json")

# Legacy .done for backward compat
(out / ".done").touch()

print("%s s%d teacher=%d emit=%d err=%d: steps=%d atk=%d tok=%.2f env=%.2f arm=%.2f succ=%s%s" % (
    args.condition, STATE_ID, ANCHOR, _mlp_emit,
    (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else -1,
    len(telemetry), n_atk, summary["token_open_duty"], summary["env_open_duty"],
    "N/A", success, " [VIDEO]" if args.save_video else ""))
