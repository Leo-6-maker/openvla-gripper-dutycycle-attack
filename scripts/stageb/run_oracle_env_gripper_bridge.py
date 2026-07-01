#!/usr/bin/env python3
"""COMMAND_OPEN_ORACLE bridge — privileged environment-level gripper OPEN override.

Distinct from VIS attack bridge:
  - No PGD perturbation; no attacker loaded
  - Clean decode at every step
  - At trigger_step, env_action[-1] forced to OPEN (-1.0 in LIBERO space) for K steps
  - Telemetry format identical to v2 VIS bridge for downstream compatibility
"""
import argparse, copy, csv, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path("/mnt/sdc/dty_user/openvla_attack")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object")
K = 10  # attack window length (matched to VIS protocol)

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["CLEAN","TRUE_T10","COMMAND_OPEN_ORACLE"])
ap.add_argument("--state_id", type=int, required=True)
ap.add_argument("--anchor", type=int, required=True, help="Teacher anchor (telemetry only)")
ap.add_argument("--seed_id", type=int, required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--render_gpu", type=int, required=True)
ap.add_argument("--mlp_path", default="outputs/sc5_canonical_eng/sc5_mlp_s2.pt")
ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index")
ap.add_argument("--libero_preprocess_backend", default="upstream_tf_jpeg",
                choices=["upstream_tf_jpeg", "project_pil_lanczos", "none"])
ap.add_argument("--trigger_step_override", type=int, default=-1,
                help="If >= 0, overrides MLP emit for trigger")
args = ap.parse_args()

from gripper_attack.openvla_preprocess import resolve_backend
PREPROCESS_BACKEND = resolve_backend(args.libero_preprocess_backend)
USES_JPEG_ROUNDTRIP = (PREPROCESS_BACKEND == "upstream_tf_jpeg")

STATE_ID = args.state_id; ANCHOR = args.anchor
IS_ORACLE = args.condition == "COMMAND_OPEN_ORACLE"
IS_ATTACK = args.condition != "CLEAN"
ATTACK_FRAMES = K if IS_ATTACK else 0

# ── OpenVLA model ──
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
print("Model on %s dtype=%s attn=%s" % (device, _actual_dtype_str, _actual_attn))
action_dim = int(model.get_action_dim("libero_object"))

# ── MLP detector (for telemetry parity only; not used for trigger) ──
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES
detector = SC5DetectorRuntime(args.mlp_path, tau_corridor=0.3, tau_release=0.3, guard=5)
print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])

# ── Env ──
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
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

# ── Streaming adapter (for telemetry parity) ──
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
_streamer = SC5StreamingFeatureAdapterV2()
_mlp_emit = -1
_eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
_prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
_invalid_steps = 0; _first_valid_step = -1

telemetry = []; attack_count = 0

for step in range(400):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos",[])) > 0 else float("nan")
    q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos",[])) > 1 else float("nan")
    qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))

    # Clean decode
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(model, processor, device, raw, instruction, "libero_object", 8,
        libero_preprocess_backend=PREPROCESS_BACKEND,
        center_crop=True, resize_size=224, drop_attention_mask=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # MLP streaming (telemetry parity, not used for trigger when override is set)
    _feat_valid = False; _feat_error = ""; _feat_25d = {}
    _det_state = detector.state; _det_cp = None; _det_rp = None; _det_pp = None

    if not detector.emitted:
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

    # === ORACLE OVERRIDE: force gripper OPEN at environment level ===
    attack_this = False; oracle_forced = False
    _trigger_step = args.trigger_step_override if args.trigger_step_override >= 0 else _mlp_emit
    _clean_candidate = np.asarray(action, dtype=np.float32)
    _env_before_override = float(env_action_final[-1])

    if IS_ORACLE and _trigger_step >= 0 and step >= _trigger_step and attack_count < ATTACK_FRAMES:
        env_action_final = np.asarray(env_action_final, dtype=np.float32).copy()
        env_action_final[-1] = np.clip(-1.0, -1.0, 1.0)  # -1.0 = OPEN in LIBERO postprocessed space
        oracle_forced = True
        attack_this = True
        attack_count += 1
        raw_grip = 1.0  # OPEN token equivalent
        env_grip = float(env_action_final[-1])

    t_vla = time.perf_counter() - t0

    _clean_env = postprocess_openvla_action_for_libero(_clean_candidate.copy(), enabled=True)
    _tel = {"step": step, "condition": args.condition, "anchor": ANCHOR,
        "mlp_emit": _mlp_emit, "raw_gripper": raw_grip, "env_gripper": env_grip,
        "qpos_sum": qpos_sum, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "obj_x": obj_x, "obj_y": obj_y, "obj_z": obj_z, "eef_obj_dist": eef_obj_dist,
        "attack_count": attack_count, "attack_this": attack_this,
        "adv_token": 31744 if oracle_forced else "", "adv_arm": "",
        "prev_delta_used": False, "model_ms": round(t_vla*1000, 2),
        "feat_valid": _feat_valid, "feat_error": _feat_error,
        "detector_state": _det_state, "corridor_p": _det_cp, "release_p": _det_rp,
        "pred_phase": _det_pp, "qpos_source": "q7+q8_sum",
        "oracle_env_override_active": oracle_forced,
        "oracle_env_action_before_override": _env_before_override,
        "oracle_env_action_after_override": float(env_action_final[-1]) if oracle_forced else float(env_action_final[-1]),
        "raw_action_7d": json.dumps([float(x) for x in action]),
        "env_action_7d": json.dumps([float(x) for x in env_action_final]),
        "clean_action_7d": json.dumps([float(x) for x in _clean_candidate]),
        "clean_env_action_7d": json.dumps([float(x) for x in _clean_env])}
    if _feat_valid:
        for fn in SC5_FEATURES:
            _tel["f_"+fn] = _feat_25d.get(fn, float("nan"))
    telemetry.append(_tel)

    obs, _, done, _ = env.step(env_action_final)
    if done: break

success = bool(env.check_success()) if hasattr(env, "check_success") else False
env.close()

atk_rows = [r for r in telemetry if r["attack_this"] == True]
n_atk = len(atk_rows)
n_env_open = sum(1 for r in atk_rows if r.get("oracle_env_override_active", False))

summary = {"condition": args.condition, "state_id": STATE_ID, "teacher_anchor": ANCHOR,
    "mlp_emit_step": _mlp_emit, "mlp_triggered": detector.emitted,
    "anchor_error": (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else None,
    "invalid_feature_steps": _invalid_steps, "first_valid_step": _first_valid_step,
    "checkpoint_sha256": detector.checkpoint_sha256,
    "dataset_sha256": detector.dataset_sha256,
    "manual_anchor_used": False, "privileged_detector_input_used": False,
    "n_steps": len(telemetry), "attack_frames": n_atk,
    "open_tokens": n_atk, "arm_ok_frames": n_atk, "env_open_frames": n_env_open,
    "token_open_duty": 1.0 if n_atk > 0 else 0, "env_open_duty": round(n_env_open/n_atk, 3) if n_atk > 0 else 0,
    "arm_duty": 1.0, "prev_delta_flags": [], "task_success": success,
    "oracle_protocol": "env_gripper_force_open_continuous",
    "requested_dtype": _dtype_name, "actual_dtype": _actual_dtype_str,
    "requested_attn": _attn_name, "actual_attn": _actual_attn,
    "preprocess_backend_requested": args.libero_preprocess_backend,
    "preprocess_backend_resolved": PREPROCESS_BACKEND,
    "preprocess_uses_jpeg_roundtrip": USES_JPEG_ROUNDTRIP}

out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
with open(out / "episode_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("%s s%d teacher=%d emit=%d: steps=%d atk=%d env_open=%d succ=%s" % (
    args.condition, STATE_ID, ANCHOR, _mlp_emit,
    len(telemetry), n_atk, n_env_open, success))
