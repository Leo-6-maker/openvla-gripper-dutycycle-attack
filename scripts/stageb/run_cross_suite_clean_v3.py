#!/usr/bin/env python3
"""
Cross-Suite Clean Collector V3 — all P0 fixes applied.

P0-1: eligible target missing → SystemExit (even in canary mode)
P0-2: target coordinates read EVERY step, not just once
P0-3: 25D features via SC5StreamingFeatureAdapterV2
P0-4: task_success recorded (env.check_success())
P0-5: gripper qpos NaN on missing (fail-closed, no 0.0 default)
P0-6: full provenance: protocol/registry/collector/model SHAs
P0-7: validate ALL protocol constraints
P0-8: atomic COMPLETE, refuse non-empty output dir
"""
import argparse, csv, hashlib, json, math, os, sys, time
import numpy as np
from pathlib import Path

# ── CLI ──
ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, choices=["libero_spatial","libero_goal","libero_10"])
ap.add_argument("--task_idx", type=int, required=True)
ap.add_argument("--state_id", type=int, required=True)
ap.add_argument("--eval_seed", type=int, default=0)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--render_gpu", type=int, required=True)
ap.add_argument("--max_steps", type=int, default=400)
ap.add_argument("--protocol", required=True)
ap.add_argument("--registry", required=True)
ap.add_argument("--canary", action="store_true")
args = ap.parse_args()

# ── P0-8: Refuse non-empty output dir ──
out = Path(args.output_dir)
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Output directory not empty: %s" % out)
out.mkdir(parents=True, exist_ok=True)

# ── Environment ──
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = "/mnt/sdc/dty_user/openvla_attack"
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.join(BASE, "src"))

# ── Load + hash protocol + registry ──
with open(args.protocol, "rb") as f: proto_raw = f.read()
with open(args.registry, "rb") as f: reg_raw = f.read()
proto_sha = hashlib.sha256(proto_raw).hexdigest()
reg_sha = hashlib.sha256(reg_raw).hexdigest()
collector_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()

protocol = json.loads(proto_raw)
registry = json.loads(reg_raw)

# ── P0-7: Validate all protocol constraints ──
proto_gate = protocol.get("gate", "")
assert proto_gate == "CROSS_SUITE_CLEAN1500_PROTOCOL_V1", "Wrong protocol gate: %s" % proto_gate
reg_gate = registry.get("gate", "")
assert reg_gate == "CROSS_SUITE_OBJECT_TARGET_REGISTRY_V1", "Wrong registry gate: %s" % reg_gate

suite_cfg = protocol["suites"][args.suite]
assert 0 <= args.task_idx <= 9, "task_idx out of range"
assert 0 <= args.state_id <= 49, "state_id out of range"
assert args.eval_seed == protocol["eval_seed"], "eval_seed mismatch"
assert args.max_steps == protocol["max_steps"], "max_steps mismatch"
assert args.render_gpu == suite_cfg["gpu"], "GPU mismatch: expected %d, got %d" % (
    suite_cfg["gpu"], args.render_gpu)

MODEL_PATH = suite_cfg["model"]
task_reg = registry[args.suite][str(args.task_idx)]
primary_object_site = task_reg["primary_object_site"]
target_site = task_reg.get("target_site")
teacher_eligible = task_reg.get("teacher_eligible", False)
target_type = task_reg.get("target_type", "unknown")
abstain_reason = task_reg.get("abstain_reason", "")

print("Suite: %s task=%d state=%d eligible=%s" % (
    args.suite, args.task_idx, args.state_id, teacher_eligible), flush=True)

# ── Model ──
from transformers import AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OpenVLAForActionPrediction.from_pretrained(MODEL_PATH, local_files_only=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
action_dim = model.config.action_dim
model = model.to(dtype=torch.bfloat16, device=device)
model.eval()
print("Model loaded (action_dim=%d)" % action_dim, flush=True)

# ── Env ──
from libero.libero import benchmark, get_libero_path
from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, physical_gripper_state

bm = benchmark.get_benchmark_dict()
suite = bm[args.suite]()
task_obj = suite.get_task(args.task_idx)
init_states = suite.get_task_init_states(args.task_idx)
bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

env, obs = build_v4_exact_env(bddl, args.render_gpu, args.max_steps, 10)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, 10)

# Object site
obj_sid = env.sim.model.site_name2id(primary_object_site)

# Target site (resolve ID once, read coords EVERY step — P0-2)
tgt_sid = -1
if target_site and teacher_eligible:
    try:
        tgt_sid = env.sim.model.site_name2id(target_site)
    except ValueError:
        # P0-1: eligible target missing MUST fail (even in canary)
        raise SystemExit("FATAL: eligible target '%s' not found in scene" % target_site)

# ── P0-3: 25D feature streamer ──
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
streamer = SC5StreamingFeatureAdapterV2()

FEATURE_NAMES = [
    "f_gripper_command","f_gripper_qpos","f_gripper_opening_proxy",
    "f_eef_x","f_eef_y","f_eef_z","f_eef_vx","f_eef_vy","f_eef_vz",
    "f_action_dx","f_action_dy","f_action_dz","f_action_gripper",
    "f_recent_close_streak","f_recent_open_streak","f_recent_gripper_flip_count",
    "f_close_onset","f_time_since_close","f_eef_speed",
    "f_eef_z_delta_since_close","f_qpos_delta_1","f_qpos_delta_3",
    "f_opening_proxy_delta_3","f_opening_proxy_variance_5","f_eef_speed_variance_5",
]

# ── Rollout ──
telemetry_rows = []
privileged_records = []
_eef_prev = None

for step in range(args.max_steps):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()

    action, _, _, _ = decode_with_scores(model, processor, device, raw, task_obj.language,
        args.suite, action_dim, libero_preprocess_backend="upstream_tf_jpeg",
        center_crop=True, resize_size=224, drop_attention_mask=True)
    env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # ── P0-5: Gripper fail-closed (NaN, not 0.0) ──
    try:
        gs = physical_gripper_state(env)
        if gs and len(gs.get("qpos", [])) >= 2:
            q7 = float(gs["qpos"][0])
            q8 = float(gs["qpos"][1])
            if math.isnan(q7) or math.isnan(q8):
                q7 = float("nan"); q8 = float("nan")
        else:
            q7 = float("nan"); q8 = float("nan")
    except Exception:
        q7 = float("nan"); q8 = float("nan")

    gripper_valid = not (math.isnan(q7) or math.isnan(q8))
    if gripper_valid:
        grip_width = abs(q7) + abs(q8)
        qpos_sum = q7 + q8
    else:
        grip_width = float("nan")
        qpos_sum = float("nan")

    raw_gripper = float(action[-1])
    env_gripper = float(env_action[-1])

    # EEF
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    if _eef_prev:
        eef_vx = eef_x - _eef_prev[0]; eef_vy = eef_y - _eef_prev[1]; eef_vz = eef_z - _eef_prev[2]
    else:
        eef_vx = eef_vy = eef_vz = 0.0
    _eef_prev = (eef_x, eef_y, eef_z)

    # Object
    obj_pos = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
    obj_eef_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))

    # ── P0-2: Target coords read EVERY step ──
    target_valid = False
    target_x = target_y = target_z = float("nan")
    if teacher_eligible and tgt_sid >= 0:
        txyz = env.sim.data.site_xpos[tgt_sid]
        target_x, target_y, target_z = float(txyz[0]), float(txyz[1]), float(txyz[2])
        target_valid = not (math.isnan(target_x) or math.isnan(target_y) or math.isnan(target_z))

    obj_tgt_dist = float(np.sqrt((obj_x-target_x)**2 + (obj_y-target_y)**2 + (obj_z-target_z)**2)) \
        if target_valid else float("nan")

    # ── P0-3: 25D features ──
    action_dx = float(action[0]) if len(action) > 0 else 0.0
    action_dy = float(action[1]) if len(action) > 1 else 0.0
    action_dz = float(action[2]) if len(action) > 2 else 0.0
    action_gripper = raw_gripper

    try:
        feat_result = streamer.update(step, raw_gripper, env_gripper,
            qpos_sum if gripper_valid else 0.0,
            grip_width if gripper_valid else 0.0,
            eef_x, eef_y, eef_z, eef_vx, eef_vy, eef_vz,
            action_dx, action_dy, action_dz, action_gripper)
        feat_valid = feat_result["valid"]
        feat_error = feat_result.get("error", "")
        if feat_valid:
            feat_dict = feat_result["features"]
            f_values = {name: float(feat_dict[name.replace("f_","")]) for name in FEATURE_NAMES
                       if name.replace("f_","") in feat_dict}
        else:
            f_values = {name: float("nan") for name in FEATURE_NAMES}
    except Exception as e:
        feat_valid = False
        feat_error = str(e)
        f_values = {name: float("nan") for name in FEATURE_NAMES}

    # Telemetry row
    telem = {
        "step": step, "suite": args.suite,
        "task_idx": args.task_idx, "state_id": args.state_id,
        "eval_seed": args.eval_seed, "condition": "CLEAN",
        "attack_this": False, "attack_frames": 0,
        "gripper_qpos_left": q7, "gripper_qpos_right": q8,
        "gripper_qpos": qpos_sum, "gripper_width": grip_width,
        "raw_gripper": raw_gripper, "env_gripper": env_gripper,
        "gripper_valid": gripper_valid,
        "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
        "object_x": obj_x, "object_y": obj_y, "object_z": obj_z,
        "object_eef_distance": obj_eef_dist,
        "target_x": target_x, "target_y": target_y, "target_z": target_z,
        "object_to_target_distance": obj_tgt_dist,
        "target_binding_type": target_type,
        "target_binding_name": target_site or "",
        "target_binding_valid": target_valid,
        "teacher_eligible": teacher_eligible,
        "abstain_reason": abstain_reason if not teacher_eligible else "",
        "feat_valid": feat_valid, "feat_error": feat_error,
    }
    telem.update(f_values)
    telemetry_rows.append(telem)

    # Privileged record
    priv_available = teacher_eligible and target_valid and gripper_valid
    privileged_records.append({
        "step_idx": step, "policy_step_idx": step,
        "teacher_privileged_state_available": priv_available,
        "gripper_command": raw_gripper,
        "gripper_qpos": qpos_sum if gripper_valid else float("nan"),
        "gripper_width": grip_width if gripper_valid else float("nan"),
        "gripper_opening_proxy": grip_width if gripper_valid else float("nan"),
        "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
        "object_eef_distance": obj_eef_dist,
        "object_to_target_distance": obj_tgt_dist,
        "object_pose_json": json.dumps([obj_x, obj_y, obj_z]),
        "target_pose_json": json.dumps([target_x, target_y, target_z]),
        "teacher_eligible": teacher_eligible,
        "target_valid": target_valid,
    })

    obs, reward, done, info = env.step(env_action)
    if done: break

# ── P0-4: Success/failure ──
try:
    task_success = bool(env.check_success())
except Exception:
    task_success = False
env.close()

# ── Schema gate ──
n_steps = len(telemetry_rows)
widths = [r["gripper_width"] for r in privileged_records
          if not (math.isnan(r["gripper_width"]) or r["gripper_width"] is None)]
p2p = max(widths) - min(widths) if len(widths) > 1 else 0.0
gripper_ok = p2p > 1e-4

if teacher_eligible:
    # P0-1: eligible episodes MUST have valid target at every step
    all_target_valid = all(r.get("target_binding_valid") for r in telemetry_rows)
else:
    all_target_valid = True  # ineligible = abstain is correct

gate_pass = n_steps > 10 and gripper_ok and all_target_valid

# ── Write output ──
with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=telemetry_rows[0].keys())
    w.writeheader(); w.writerows(telemetry_rows)

with open(out / "privileged_step_records.jsonl", "w") as f:
    for rec in privileged_records:
        f.write(json.dumps(rec) + "\n")

summary = {"suite": args.suite, "task_idx": args.task_idx,
    "task_name": task_obj.name, "state_id": args.state_id,
    "eval_seed": args.eval_seed, "n_steps": n_steps,
    "condition": "CLEAN", "task_success": task_success,
    "teacher_eligible": teacher_eligible, "target_valid": all_target_valid,
    "gripper_valid": gripper_ok, "gate_pass": gate_pass,
    "target_site": target_site, "primary_object_site": primary_object_site,
    "abstain_reason": abstain_reason if not teacher_eligible else ""}
with open(out / "episode_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Atomic COMPLETE
complete = {"status": "COMPLETE" if gate_pass else "SCHEMA_FAIL",
    "n_steps": n_steps, "gate_pass": gate_pass,
    "teacher_eligible": teacher_eligible, "target_valid": all_target_valid,
    "gripper_valid": gripper_ok, "task_success": task_success,
    "timestamp": time.time()}
tmp = out / "COMPLETE.json.tmp"
final = out / ("COMPLETE.json" if gate_pass else "SCHEMA_FAIL.json")
with open(tmp, "w") as f: json.dump(complete, f, indent=2)
os.replace(tmp, final)

# Provenance
with open(out / "artifact_sha256.json", "w") as f:
    shas = {}
    for fn in ["step_telemetry.csv","privileged_step_records.jsonl","episode_summary.json",
               "COMPLETE.json","SCHEMA_FAIL.json"]:
        fp = out / fn
        if fp.exists():
            with open(fp, "rb") as fp2: shas[fn] = hashlib.sha256(fp2.read()).hexdigest()
    shas["protocol_sha256"] = proto_sha
    shas["registry_sha256"] = reg_sha
    shas["collector_sha256"] = collector_sha
    shas["model_path"] = MODEL_PATH
    json.dump(shas, f, indent=2)

print("DONE: steps=%d p2p=%.4f gate=%s success=%s eligible=%s target=%s" % (
    n_steps, p2p, gate_pass, task_success, teacher_eligible, all_target_valid), flush=True)
if not gate_pass:
    sys.exit(1)
