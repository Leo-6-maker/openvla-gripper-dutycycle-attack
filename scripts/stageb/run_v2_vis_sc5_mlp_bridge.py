#!/usr/bin/env python3
"""SC5 MLP-triggered VIS bridge — identical to v2 bridge, replaces fixed anchor with learned detector.

CHANGES vs run_v2_vis_sc5_bridge.py (only 3):
  1. MLP detector loading (after model)
  2. Online 25D feature extraction (before trigger in loop)
  3. Trigger condition: step >= _mlp_emit (was: step >= ANCHOR)

ALL VIS attack code preserved exactly. ALL telemetry preserved exactly.
"""
import argparse, csv, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

EPSILON = 0.023529411764705882; TARGET_TOKEN = 31744; ARM_GATE = 5; PGD_STEPS = 20; K = 10

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["CLEAN","TRUE_T10","RAND_T10","SHUFFLED_T10","VIS","RAND","SHUFFLED"])
ap.add_argument("--suite", default="libero_object", choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
ap.add_argument("--model_path", default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
ap.add_argument("--unnorm_key", default="")
ap.add_argument("--state_id", type=int, required=True)
ap.add_argument("--anchor", type=int, required=True, help="Teacher anchor (AUDIT ONLY, not used for trigger)")
ap.add_argument("--seed_id", type=int, required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--render_gpu", type=int, required=True)
ap.add_argument("--mlp_path", default="outputs/sc5_canonical_eng/sc5_mlp_s2.pt")
ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")
ap.add_argument("--max_steps", type=int, default=400)
ap.add_argument("--write_video", action="store_true")
args = ap.parse_args()

def write_video(path, video_frames, fps=10):
    if not video_frames:
        return "NO_FRAMES"
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return "IMAGEIO_UNAVAILABLE:%s" % type(exc).__name__
    imageio.mimwrite(path, video_frames, fps=fps)
    return "WROTE"

CONDITION = {"VIS": "TRUE_T10", "RAND": "RAND_T10", "SHUFFLED": "SHUFFLED_T10"}.get(args.condition, args.condition)
SUITE_KEY = args.unnorm_key or args.suite
MODEL_PATH = args.model_path
STATE_ID = args.state_id; ANCHOR = args.anchor; IS_ATTACK = CONDITION != "CLEAN"
IS_RAND = "RAND" in CONDITION; IS_SHUFFLED = "SHUFFLED" in CONDITION
ATTACK_FRAMES = K if IS_ATTACK else 0

# ── OpenVLA model (identical to v2 bridge) ──
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map="auto",
    max_memory={idx: "10000MiB" for idx in range(visible)} | {"cpu": "128GiB"}, attn_implementation="eager")
model_dtype = next(model.parameters()).dtype
device = "cuda:0"
for v in model.hf_device_map.values():
    if isinstance(v, int): device = "cuda:%d" % v; break
action_dim = int(model.get_action_dim(SUITE_KEY))
print("Model on %s" % device)

# ── MLP detector (NEW — only change from v2 bridge) ──
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES
_ckpt_for_thresholds = torch.load(args.mlp_path, map_location="cpu", weights_only=False)
_tau_c = float(_ckpt_for_thresholds.get("selected_tau_corridor", 0.3))
_tau_r = float(_ckpt_for_thresholds.get("selected_tau_release", 0.3))
detector = SC5DetectorRuntime(
    args.mlp_path,
    tau_corridor=_tau_c,
    tau_release=_tau_r,
    guard=5,
    allowed_split_modes=("frozen", "provisional_cross_suite_frozen"),
)
print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])

# ── Persistent attacker (identical to v2 bridge) ──
attacker = None
if IS_ATTACK and not IS_RAND:
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
           "target_token_id": TARGET_TOKEN, "epsilon": EPSILON, "num_steps": PGD_STEPS,
           "step_size": EPSILON * 0.075, "random_start": True, "prefix_refresh_interval": 1,
           "surrogate_score_path": "cached_autoregressive_generate_v1",
           "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": ARM_GATE,
           "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
           "target_execution_class": "CLIP_MEDIATED_OPEN"}
    if IS_SHUFFLED: opt["gradient_transform"] = "permute"; opt["gradient_transform_seed"] = args.seed_id + 100000
    attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
        seed=args.seed_id, preprocess_kwargs={"libero_official_preprocess": False,
            "libero_preprocess_backend": "official_pil_lanczos", "center_crop": True, "resize_size": 224}, device=device)

# ── Env (identical to v2 bridge) ──
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.v3_generation_parity import extract_exact_new_tokens
from libero.libero import benchmark, get_libero_path

TASK_IDX = args.task_idx
bm = benchmark.get_benchmark_dict(); suite = bm[args.suite]()
task_obj = suite.get_task(TASK_IDX); init_states = suite.get_task_init_states(TASK_IDX)
bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[STATE_ID])
env, obs = apply_dummy_wait(env, obs, 10)

_task_name = task_obj.name
obj_sid = None
obj_z0 = float("nan")
for _candidate_site in ("akita_black_bowl_1_default_site", "wine_bottle_1_default_site", "book_1_default_site"):
    try:
        obj_sid = env.sim.model.site_name2id(_candidate_site)
        obj_z0 = float(env.sim.data.site_xpos[obj_sid][2])
        break
    except Exception:
        pass

# ── Online streaming adapter (NEW) ──
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
_streamer = SC5StreamingFeatureAdapterV2()
_mlp_emit = -1
# Initialize _prev_eef from env after dummy-wait (enables valid velocity at step 0)
_eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
_prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
_invalid_steps = 0; _first_valid_step = -1  # for parity diagnosis

telemetry = []; attack_count = 0; prev_delta_flags = []; frames = []

for step in range(args.max_steps):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()
    if args.write_video:
        frames.append(raw.copy())

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos",[])) > 0 else float("nan")
    q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos",[])) > 1 else float("nan")
    qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    if obj_sid is not None:
        obj_xyz = env.sim.data.site_xpos[obj_sid]
        obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
        eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))
    else:
        obj_x = obj_y = obj_z = eef_obj_dist = float("nan")

    # Clean decode (identical to v2 bridge)
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(model, processor, device, raw, instruction, SUITE_KEY, 8,
        libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
        center_crop=True, resize_size=224, drop_attention_mask=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
    clean_env_action_final = np.asarray(env_action_final, dtype=np.float32).copy()

    # ── MLP ONLINE TRIGGER (causal eef velocity, always advances streamer) ──
    _feat_valid = False; _feat_error = ""; _feat_25d = {}
    _det_state = detector.state; _det_cp = None; _det_rp = None; _det_pp = None

    if not detector.emitted:
        # Causal EEF velocity: backward difference, only when position valid
        eef_valid = np.all(np.isfinite([eef_x, eef_y, eef_z]))
        if _prev_eef is not None and eef_valid:
            _vx = eef_x - _prev_eef[0]; _vy = eef_y - _prev_eef[1]; _vz = eef_z - _prev_eef[2]
        else:
            _vx = float("nan"); _vy = float("nan"); _vz = float("nan")
        if eef_valid:
            _prev_eef = (eef_x, eef_y, eef_z)

        gripper_ok = not (np.isnan(q7) or np.isnan(q8) or np.isnan(qpos_sum))
        gripper_w = abs(q7)+abs(q8) if gripper_ok else float("nan")
        gripper_q = float(qpos_sum) if gripper_ok else float("nan")

        # Always call streamer (preserves step sequence)
        try:
            _res = _streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=gripper_q, gripper_opening_proxy=gripper_w,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=_vx, eef_vy=_vy, eef_vz=_vz,
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

    # === VIS ATTACK (IDENTICAL to v2 bridge, only trigger condition changed) ===
    attack_this = False; adv_token = None; adv_arm = 0; prev_flag = False
    if IS_ATTACK and _mlp_emit >= 0 and step >= _mlp_emit and attack_count < ATTACK_FRAMES:
        clean_tokens_for_arm = None
        if IS_RAND:
            from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
            proc_image = prepare_openvla_image_for_attack(raw, libero_official_preprocess=False,
                libero_preprocess_backend="official_pil_lanczos", center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
            inputs.pop("attention_mask", None)
            iids = inputs["input_ids"].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            x = inputs["pixel_values"].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                go_clean = model.generate(input_ids=iids, pixel_values=x, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            clean_tokens_for_arm = extract_exact_new_tokens(go_clean.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=args.seed_id+100000+attack_count, dtype=torch.float32, device=x.device)
            proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
            adv_pv = proj.detach().to(dtype=model_dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=adv_pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
        else:
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
            from gripper_attack.route_contract import validate_true_pgd_attack_result
            clean_action_np = np.asarray(action, dtype=np.float32)
            proc_image = prepare_openvla_image_for_attack(raw, libero_official_preprocess=False,
                libero_preprocess_backend="official_pil_lanczos", center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
            inputs.pop("attention_mask", None)
            iids = inputs["input_ids"].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            pv = inputs["pixel_values"].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            clean_tokens_for_arm = clean_tokens
            clean_gen = type("CleanGen", (), {})()
            clean_gen.sequences = torch.tensor([iids[0].detach().cpu().tolist() + [int(t) for t in clean_tokens]], dtype=torch.long, device=device)
            clean_gen.scores = []
            attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen, unnorm_key=SUITE_KEY)
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)
            adv_pv = adv_inputs["pixel_values"]
            with torch.inference_mode():
                go_adv = model.generate(input_ids=iids, pixel_values=adv_pv.to(device=device, dtype=model_dtype), max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            adv_tokens = extract_exact_new_tokens(go_adv.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            prev_flag = attack_result.debug.get("temporal_prev_delta_used", False) if hasattr(attack_result, "debug") else False

        grip = int(adv_tokens[-1])
        if clean_tokens_for_arm is not None:
            adv_arm = sum(1 for a, b in zip([int(t) for t in adv_tokens[:6]], [int(t) for t in clean_tokens_for_arm[:6]]) if a == b)
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        disc = np.clip(vocab_size - np.array([int(t) for t in adv_tokens]) - 1, 0, model.bin_centers.shape[0]-1)
        na = model.bin_centers[disc]
        s = model.get_action_stats(SUITE_KEY)
        lo = np.asarray(s["q01"], dtype=np.float32); hi = np.asarray(s["q99"], dtype=np.float32)
        mk = np.asarray(s.get("mask", np.ones_like(lo, dtype=bool)), dtype=bool)
        attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
        attack_env_action_final = postprocess_openvla_action_for_libero(attack_action, enabled=True)
        env_action_final = clean_env_action_final.copy()
        env_action_final[-1] = float(attack_env_action_final[-1])
        raw_grip = float(attack_action[-1]); env_grip = float(env_action_final[-1])
        attack_this = True; attack_count += 1
        prev_delta_flags.append(prev_flag)
        adv_token = grip

    t_vla = time.perf_counter() - t0

    _tel = {"step": step, "condition": CONDITION, "suite": args.suite, "unnorm_key": SUITE_KEY, "task_idx": TASK_IDX, "anchor": ANCHOR,
        "mlp_emit": _mlp_emit, "raw_gripper": raw_grip, "env_gripper": env_grip,
        "qpos_sum": qpos_sum, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "obj_x": obj_x, "obj_y": obj_y, "obj_z": obj_z, "eef_obj_dist": eef_obj_dist,
        "attack_count": attack_count, "attack_this": attack_this,
        "adv_token": adv_token if adv_token else "", "adv_arm": adv_arm if attack_this else "",
        "prev_delta_used": prev_flag, "model_ms": round(t_vla*1000, 2),
        "feat_valid": _feat_valid, "feat_error": _feat_error,
        "detector_state": _det_state, "corridor_p": _det_cp, "release_p": _det_rp,
        "pred_phase": _det_pp, "qpos_source": "q7+q8_sum"}
    if _feat_valid:
        for fn in SC5_FEATURES:
            _tel["f_"+fn] = _feat_25d.get(fn, float("nan"))
    telemetry.append(_tel)

    obs, _, done, _ = env.step(env_action_final)
    if done: break

success = bool(env.check_success()) if hasattr(env, "check_success") else False
env.close()

# Metrics (identical to v2 bridge)
atk_rows = [r for r in telemetry if r["attack_this"] == True]
n_atk = len(atk_rows)
n_open_token = sum(1 for r in atk_rows if str(r.get("adv_token","")) != "" and int(r["adv_token"]) == TARGET_TOKEN)
n_env_open = sum(1 for r in atk_rows if float(r["env_gripper"]) < 0)
n_arm_ok = sum(1 for r in atk_rows if str(r.get("adv_arm","")) != "" and int(r["adv_arm"]) >= ARM_GATE)

summary = {"condition": CONDITION, "requested_condition": args.condition,
    "suite": args.suite, "task_idx": TASK_IDX, "unnorm_key": SUITE_KEY,
    "state_id": STATE_ID, "teacher_anchor": ANCHOR,
    "mlp_emit_step": _mlp_emit, "mlp_triggered": detector.emitted,
    "anchor_error": (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else None,
    "invalid_feature_steps": _invalid_steps, "first_valid_step": _first_valid_step,
    "checkpoint_sha256": detector.checkpoint_sha256,
    "dataset_sha256": detector.dataset_sha256,
    "tau_corridor": _tau_c, "tau_release": _tau_r,
    "manual_anchor_used": False, "privileged_detector_input_used": False,
    "arm_action_preservation_mode": "execute_clean_arm_with_attacked_gripper",
    "n_steps": len(telemetry), "attack_frames": n_atk,
    "open_tokens": n_open_token, "arm_ok_frames": n_arm_ok, "env_open_frames": n_env_open,
    "token_open_duty": round(n_open_token/n_atk,3) if n_atk>0 else 0,
    "arm_duty": round(n_arm_ok/n_atk,3) if n_atk>0 else 0,
    "env_open_duty": round(n_env_open/n_atk,3) if n_atk>0 else 0,
    "prev_delta_flags": prev_delta_flags, "task_success": success}

out = Path(args.output_dir)
if out.exists() and any(out.iterdir()):
    raise SystemExit("output_dir exists and is non-empty: %s" % out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
if args.write_video:
    summary["raw_video_status"] = write_video(out / "rollout_raw.mp4", frames)
    overlay = []
    for idx, frame in enumerate(frames):
        arr = frame.copy()
        if idx < len(telemetry):
            row = telemetry[idx]
            if row.get("attack_this") is True:
                arr[:8, :, 0] = 255
                arr[:8, :, 1] = 220
                arr[:8, :, 2] = 0
            if int(row.get("mlp_emit", -1) or -1) == int(row.get("step", idx)):
                arr[10:18, :, 0] = 255
                arr[10:18, :, 1] = 0
                arr[10:18, :, 2] = 255
        overlay.append(arr)
    summary["overlay_video_status"] = write_video(out / "rollout_overlay.mp4", overlay)
with open(out / "episode_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("%s s%d teacher=%d emit=%d err=%d: steps=%d atk=%d tok=%.2f env=%.2f arm=%.2f succ=%s" % (
    CONDITION, STATE_ID, ANCHOR, _mlp_emit,
    (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else -1,
    len(telemetry), n_atk, summary["token_open_duty"], summary["env_open_duty"],
    summary["arm_duty"], success))
