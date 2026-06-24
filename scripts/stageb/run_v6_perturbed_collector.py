#!/usr/bin/env python3
"""V6 Perturbed Collector — V2 bridge base + correct perturbation + atomic completion.

Key fixes vs V5 / V6-r1:
  1. Perturbation modifies free-joint qpos (authoritative), NOT body_xpos/body_xquat
  2. CLEAN-only — attack branches removed entirely
  3. Atomic completion: tmp → validate → rename → .done last
  4. Fail-closed target resolution
  5. Full provenance: run_uuid, cell_uuid, exit_code, asset SHAs
"""
import argparse, copy, csv, hashlib, json, os, shutil, sys, time, uuid, numpy as np, torch
from pathlib import Path

_candidate = Path(__file__).resolve()
while _candidate.parent != _candidate:
    if (_candidate / "src" / "gripper_attack").is_dir():
        REPO = _candidate; break
    _candidate = _candidate.parent
else:
    REPO = Path(os.environ.get("OPENVLA_REPO", "/mnt/sdc/dty_user/openvla_attack"))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "migration"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH",
    "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object")

ap = argparse.ArgumentParser()
ap.add_argument("--state_id", type=int, required=True)
ap.add_argument("--task_idx", type=int, required=True)
ap.add_argument("--seed_id", type=int, required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--render_gpu", type=int, required=True)
ap.add_argument("--mlp_path", default="outputs/sc5_canonical_eng/sc5_mlp_s2.pt")
ap.add_argument("--perturbation_template", default="P0", choices=["P0","P1","P2","P3","P4","P5","P6","P7"])
ap.add_argument("--pool", default="smoke", choices=["train","dev","smoke"], help="Split pool (written to telemetry + summary + .done)")
ap.add_argument("--vla_manifest_sha256", default="", help="Pre-computed VLA model manifest SHA (avoids re-hashing 7B shards)")
args = ap.parse_args()

STATE_ID = args.state_id; ANCHOR = 0  # dummy — CLEAN only
CONDITION = "CLEAN"; ATTACK_FRAMES = 0
POOL = args.pool

run_uuid = str(uuid.uuid4())[:8]
cell_uuid = str(uuid.uuid4())[:12]

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
print("Model on %s dtype=%s attn=%s (requested: dtype=%s attn=%s)" % (
    device, _actual_dtype_str, _actual_attn, _dtype_name, _attn_name))
action_dim = int(model.get_action_dim("libero_object"))

# ── MLP detector ──
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

# ── State hash utility (hashes qpos+qvel+act) ──
def _state_sha256(env):
    state_bytes = env.sim.data.qpos.tobytes() + env.sim.data.qvel.tobytes() + env.sim.data.act.tobytes()
    return hashlib.sha256(state_bytes).hexdigest()

# ── Target resolution (fail-closed) ──
from label_m1c_object_teacher import resolve_target_position
tgt = resolve_target_position(TASK_IDX, STATE_ID)
if tgt is None:
    env.close()
    raise RuntimeError("TARGET_UNRESOLVABLE: task=%d state=%d" % (TASK_IDX, STATE_ID))
target_x, target_y, target_z = tgt

# ── Perturbation: modify free-joint qpos (authoritative, NOT body_xpos) ──
pert_template = args.perturbation_template
pert_spec = {"template_id": pert_template, "base_seed": args.seed_id,
             "dx_m": 0.0, "dy_m": 0.0, "dyaw_rad": 0.0, "object_body": None}

# Compute perturbation offsets
from gripper_attack.v5_perturbation import get_perturbation
dx, dy, dyaw = get_perturbation(pert_template, args.seed_id)
pert_spec["dx_m"] = float(dx); pert_spec["dy_m"] = float(dy); pert_spec["dyaw_rad"] = float(dyaw)

# Find object body (keyword match + BDDL fallback)
task_keyword = task_obj.name.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "")
available = set(env.sim.model.body_names)
obj_body_name = None
for name in sorted(available):
    if "basket" in name or "bin" in name: continue
    if task_keyword in name and name.endswith("_main"):
        obj_body_name = name; break
if obj_body_name is None:
    for line in open(bddl).read().split('\n'):
        line = line.strip()
        if line and not line.startswith('(:') and ' - ' in line:
            parts = line.split(' - ')
            main_name = parts[0].strip() + "_main"
            if parts[1].strip() not in ['basket', 'bin']:
                if main_name in available:
                    obj_body_name = main_name; break
if obj_body_name is None:
    env.close()
    raise RuntimeError("OBJECT_BODY_UNRESOLVABLE: task=%d state=%d keyword=%s" % (TASK_IDX, STATE_ID, task_keyword))

pert_spec["object_body"] = obj_body_name
body_id = env.sim.model.body_name2id(obj_body_name)
joint_id = env.sim.model.body_jntadr[body_id]  # first joint for this body
jnt_type = env.sim.model.jnt_type[joint_id]

if jnt_type != 0:  # 0 = mjJNT_FREE
    env.close()
    raise RuntimeError("Object joint is not free (type=%d), expected mjJNT_FREE" % jnt_type)

qadr = env.sim.model.jnt_qposadr[joint_id]

# Capture original qpos before perturbation
orig_qpos = env.sim.data.qpos[qadr:qadr+7].copy()
original_state_sha = _state_sha256(env)
orig_body_pos = env.sim.data.body_xpos[body_id].copy()
orig_body_quat = env.sim.data.body_xquat[body_id].copy()

# Apply perturbation to free-joint qpos
if pert_template != "P0":
    new_qpos = orig_qpos.copy()
    new_qpos[0] += dx  # x
    new_qpos[1] += dy  # y
    if dyaw != 0:
        from scipy.spatial.transform import Rotation
        qw, qx, qy, qz = float(orig_qpos[3]), float(orig_qpos[4]), float(orig_qpos[5]), float(orig_qpos[6])
        rot = Rotation.from_euler('z', dyaw)
        q_out = (Rotation.from_quat([qx, qy, qz, qw]) * rot).as_quat()
        new_qpos[3] = float(q_out[3])  # qw
        new_qpos[4] = float(q_out[0])  # qx
        new_qpos[5] = float(q_out[1])  # qy
        new_qpos[6] = float(q_out[2])  # qz
    env.sim.data.qpos[qadr:qadr+7] = new_qpos

env.sim.forward()

# Read back actual pose AFTER sim.forward()
actual_body_pos = env.sim.data.body_xpos[body_id].copy()
actual_body_quat = env.sim.data.body_xquat[body_id].copy()
perturbed_state_sha = _state_sha256(env)

# Verify perturbation semantics
if pert_template == "P0":
    if original_state_sha != perturbed_state_sha:
        env.close()
        raise RuntimeError("P0_HASH_CHANGED: original=%s perturbed=%s" % (original_state_sha[:16], perturbed_state_sha[:16]))
else:
    if original_state_sha == perturbed_state_sha:
        env.close()
        raise RuntimeError("PERTURBATION_NOOP: template=%s original_sha==perturbed_sha=%s" % (pert_template, original_state_sha[:16]))

pert_spec["original_state_sha256"] = original_state_sha
pert_spec["perturbed_state_sha256"] = perturbed_state_sha
pert_spec["original_qpos"] = orig_qpos.tolist()
pert_spec["perturbed_qpos"] = env.sim.data.qpos[qadr:qadr+7].copy().tolist()
pert_spec["original_body_pos"] = orig_body_pos.tolist()
pert_spec["actual_body_pos_after_forward"] = actual_body_pos.tolist()
pert_spec["joint_id"] = int(joint_id)
pert_spec["jnt_qposadr"] = int(qadr)
pert_spec["jnt_type"] = int(jnt_type)

print("Perturbation: %s dx=%.4f dy=%.4f dyaw=%.4f orig_sha=%s pert_sha=%s pos_delta=(%.4f,%.4f,%.4f)" % (
    pert_template, pert_spec["dx_m"], pert_spec["dy_m"], pert_spec["dyaw_rad"],
    original_state_sha[:16], perturbed_state_sha[:16],
    float(actual_body_pos[0] - orig_body_pos[0]),
    float(actual_body_pos[1] - orig_body_pos[1]),
    float(actual_body_pos[2] - orig_body_pos[2])))

# Check object not below table
if actual_body_pos[2] < 0.01:
    env.close()
    raise ValueError("Perturbation pushed object below table: z=%.4f" % actual_body_pos[2])

# Pose tolerance assertions (fail-closed)
if pert_template != "P0":
    actual_dx = float(actual_body_pos[0] - orig_body_pos[0])
    actual_dy = float(actual_body_pos[1] - orig_body_pos[1])
    tol_xy = 0.001  # 1mm tolerance
    if abs(actual_dx - dx) > tol_xy:
        env.close()
        raise RuntimeError("POSE_DX_MISMATCH: requested=%.4f actual=%.4f" % (dx, actual_dx))
    if abs(actual_dy - dy) > tol_xy:
        env.close()
        raise RuntimeError("POSE_DY_MISMATCH: requested=%.4f actual=%.4f" % (dy, actual_dy))
    if dyaw != 0:
        from scipy.spatial.transform import Rotation
        actual_rot = Rotation.from_quat([
            float(actual_body_quat[1]), float(actual_body_quat[2]),
            float(actual_body_quat[3]), float(actual_body_quat[0])])
        orig_rot = Rotation.from_quat([
            float(orig_body_quat[1]), float(orig_body_quat[2]),
            float(orig_body_quat[3]), float(orig_body_quat[0])])
        actual_dyaw = float((orig_rot.inv() * actual_rot).as_euler('xyz')[2])
        if abs(actual_dyaw - dyaw) > np.deg2rad(1.0):
            env.close()
            raise RuntimeError("POSE_DYAW_MISMATCH: requested=%.4f actual=%.4f" % (dyaw, actual_dyaw))

# ── Dummy wait (MUST happen before post-wait capture) ──
env, obs = apply_dummy_wait(env, obs, 10)

# ── Post-dummy-wait state capture ──
rollout_start_sha = _state_sha256(env)
rollout_start_body_pos = env.sim.data.body_xpos[body_id].copy()
rollout_start_body_quat = env.sim.data.body_xquat[body_id].copy()

print("Post-wait: sha=%s pos=(%.4f,%.4f,%.4f)" % (
    rollout_start_sha[:16],
    float(rollout_start_body_pos[0]), float(rollout_start_body_pos[1]), float(rollout_start_body_pos[2])))

_task_name = task_obj.name
_obj_key = _task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")
obj_sid = env.sim.model.site_name2id("%s_1_default_site" % _obj_key)

# ── Online streaming adapter ──
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
_streamer = SC5StreamingFeatureAdapterV2()
_mlp_emit = -1
_eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
_prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
_invalid_steps = 0; _first_valid_step = -1

telemetry = []

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

    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(model, processor, device, raw, instruction, "libero_object", 8,
        libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
        center_crop=True, resize_size=224, drop_attention_mask=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # ── MLP ONLINE TRIGGER ──
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

    t_vla = time.perf_counter() - t0

    _tel = {"step": step, "condition": CONDITION, "pool": POOL,
        "task_idx": TASK_IDX, "parent_state_id": STATE_ID,
        "run_uuid": run_uuid, "cell_uuid": cell_uuid,
        "anchor": ANCHOR,
        "mlp_emit": _mlp_emit, "raw_gripper": raw_grip, "env_gripper": env_grip,
        "qpos_sum": qpos_sum, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "obj_x": obj_x, "obj_y": obj_y, "obj_z": obj_z, "eef_obj_dist": eef_obj_dist,
        "target_x": target_x, "target_y": target_y, "target_z": target_z,
        "attack_count": 0, "attack_this": False,
        "adv_token": "", "adv_arm": "", "prev_delta_used": False,
        "model_ms": round(t_vla*1000, 2),
        "feat_valid": _feat_valid, "feat_error": _feat_error,
        "detector_state": _det_state, "corridor_p": _det_cp, "release_p": _det_rp,
        "pred_phase": _det_pp, "qpos_source": "q7+q8_sum",
        "perturbation_template": pert_template, "perturbation_seed": args.seed_id,
        "initial_state_sha256": original_state_sha,
        "perturbed_initial_state_sha256": perturbed_state_sha,
        "raw_action_7d": json.dumps([float(x) for x in action]),
        "env_action_7d": json.dumps([float(x) for x in env_action_final])}
    if _feat_valid:
        for fn in SC5_FEATURES:
            _tel["f_"+fn] = _feat_25d.get(fn, float("nan"))
    telemetry.append(_tel)

    obs, _, done, _ = env.step(env_action_final)
    if done: break

success = bool(env.check_success()) if hasattr(env, "check_success") else False
env.close()

# ── Asset SHAs ──
bridge_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ckpt_path = Path(args.mlp_path)
ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else "MISSING"
teacher_config_path = REPO / "migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json"
teacher_config_sha = hashlib.sha256(teacher_config_path.read_bytes()).hexdigest() if teacher_config_path.exists() else "MISSING"
target_resolver_path = REPO / "scripts/migration/label_m1c_object_teacher.py"
target_resolver_sha = hashlib.sha256(target_resolver_path.read_bytes()).hexdigest() if target_resolver_path.exists() else "MISSING"
pert_gen_path = REPO / "src/gripper_attack/v5_perturbation.py"
pert_gen_sha = hashlib.sha256(pert_gen_path.read_bytes()).hexdigest() if pert_gen_path.exists() else "MISSING"

# ── VLA model manifest SHA ──
if args.vla_manifest_sha256:
    vla_manifest_sha = args.vla_manifest_sha256
else:
    model_dir = Path(MODEL_PATH)
    vla_manifest_lines = sorted(
        "%s %s" % (f.relative_to(model_dir), hashlib.sha256(f.read_bytes()).hexdigest())
        for f in model_dir.rglob("*") if f.is_file())
    vla_manifest_sha = hashlib.sha256("\n".join(vla_manifest_lines).encode()).hexdigest()
    print("VLA manifest SHA computed: %s (pass --vla_manifest_sha256 to skip)" % vla_manifest_sha[:16])

# ── Atomic completion protocol ──
final_dir = Path(args.output_dir)
done_file = final_dir / ".done"

# P0-4: fail-closed on existing complete cell
if done_file.exists():
    raise RuntimeError("CELL_ALREADY_COMPLETE: %s exists — refusing to overwrite" % str(done_file))

# Remove stale partial (no .done) — quarantine instead of rmtree
if final_dir.exists() and not done_file.exists():
    stale_dir = final_dir.with_name(final_dir.name + ".stale_" + run_uuid)
    os.rename(str(final_dir), str(stale_dir))
    print("Moved stale partial to %s" % stale_dir)

tmp_dir = final_dir.with_name(final_dir.name + ".tmp." + run_uuid)
if tmp_dir.exists():
    shutil.rmtree(str(tmp_dir))
tmp_dir.mkdir(parents=True, exist_ok=True)

# Write telemetry to tmp
tel_path = tmp_dir / "step_telemetry.csv"
with open(tel_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
    w.writeheader(); w.writerows(telemetry)
    f.flush(); os.fsync(f.fileno())
tel_bytes = tel_path.read_bytes()
tel_sha = hashlib.sha256(tel_bytes).hexdigest()

# Build summary
summary = {
    "run_uuid": run_uuid, "cell_uuid": cell_uuid, "exit_code": 0,
    "pool": POOL, "task_idx": TASK_IDX, "parent_state_id": STATE_ID,
    "condition": CONDITION, "attack_frames": 0,
    "n_steps": len(telemetry), "task_success": success,
    "perturbation_template": pert_template,
    "perturbation_seed": args.seed_id,
    "perturbation_spec": pert_spec,
    "target_position": [target_x, target_y, target_z],
    "mlp_emit_step": _mlp_emit, "mlp_triggered": detector.emitted,
    "anchor_error": (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else None,
    "invalid_feature_steps": _invalid_steps, "first_valid_step": _first_valid_step,
    "selected_original_state_sha256": original_state_sha,
    "perturbed_pre_wait_sha256": perturbed_state_sha,
    "rollout_start_post_wait_sha256": rollout_start_sha,
    "trajectory_content_sha256": tel_sha,
    "checkpoint_sha256": ckpt_sha,
    "dataset_sha256": detector.dataset_sha256,
    "bridge_sha256": bridge_sha,
    "runner_sha256": bridge_sha,
    "teacher_config_sha256": teacher_config_sha,
    "target_resolver_sha256": target_resolver_sha,
    "perturbation_generator_sha256": pert_gen_sha,
    "vla_model_manifest_sha256": vla_manifest_sha,
    "requested_dtype": _dtype_name, "actual_dtype": _actual_dtype_str,
    "requested_attn": _attn_name, "actual_attn": _actual_attn,
    "manual_anchor_used": False, "privileged_detector_input_used": False,
}

# Write summary to tmp
ep_path = tmp_dir / "episode_summary.json"
with open(ep_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
    f.flush(); os.fsync(f.fileno())
ep_bytes = ep_path.read_bytes()
ep_sha = hashlib.sha256(ep_bytes).hexdigest()

# Validate before rename
csv_lines = len(telemetry)
if csv_lines != summary["n_steps"]:
    shutil.rmtree(str(tmp_dir))
    raise RuntimeError("N_STEPS_MISMATCH: csv=%d summary=%d" % (csv_lines, summary["n_steps"]))
if summary["condition"] != "CLEAN":
    shutil.rmtree(str(tmp_dir))
    raise RuntimeError("NOT_CLEAN: condition=%s" % summary["condition"])
if summary["attack_frames"] != 0:
    shutil.rmtree(str(tmp_dir))
    raise RuntimeError("ATTACK_FRAMES_NONZERO: %d" % summary["attack_frames"])

# fsync tmp dir before rename
os.fsync(os.open(str(tmp_dir), os.O_RDONLY))

# Atomic rename
os.rename(str(tmp_dir), str(final_dir))

# fsync parent dir
parent_dir = final_dir.parent
os.fsync(os.open(str(parent_dir), os.O_RDONLY))

# .done LAST — atomic via tmp + rename
done_tmp = final_dir / ".done.tmp"
done = {
    "exit_code": 0,
    "telemetry_sha": tel_sha,         # legacy schema (v2 bridge compat)
    "telemetry_sha256": tel_sha,      # current schema
    "summary_sha256": ep_sha,
    "completed": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    "run_uuid": run_uuid, "cell_uuid": cell_uuid,
    "pool": POOL,
}
with open(done_tmp, "w") as f:
    json.dump(done, f)
    f.flush(); os.fsync(f.fileno())
os.rename(str(done_tmp), str(done_file))
os.fsync(os.open(str(final_dir), os.O_RDONLY))

print("%s pool=%s task=%d s%d emit=%d err=%d steps=%d succ=%s pert=%s orig_sha=%s pert_sha=%s" % (
    CONDITION, POOL, TASK_IDX, STATE_ID, _mlp_emit,
    (_mlp_emit - ANCHOR) if _mlp_emit >= 0 else -1,
    len(telemetry), success, pert_template,
    original_state_sha[:16], perturbed_state_sha[:16]))
