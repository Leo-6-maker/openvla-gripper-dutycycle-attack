#!/usr/bin/env python3
"""
Cross-Suite Clean Collector V2 — suite-aware target resolver.
Uses per-task target registry, NOT hardcoded basket_1_default_site.

Canary mode: --canary flag limits to 1 episode for schema validation.
Full mode: task_start..task_end × state_start..state_end for one suite.
"""
import argparse, csv, json, os, sys, time, hashlib
import numpy as np
from pathlib import Path

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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = "/mnt/sdc/dty_user/openvla_attack"
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.join(BASE, "src"))

# ── Load protocol + registry ──
with open(args.protocol) as f: protocol = json.load(f)
with open(args.registry) as f: registry = json.load(f)

suite_cfg = protocol["suites"][args.suite]
MODEL_PATH = suite_cfg["model"]
task_reg = registry[args.suite][str(args.task_idx)]

primary_object_site = task_reg["primary_object_site"]
target_site = task_reg.get("target_site")
teacher_eligible = task_reg.get("teacher_eligible", False)
target_type = task_reg.get("target_type", "unknown")
abstain_reason = task_reg.get("abstain_reason", "")

print("Suite: %s task=%d state=%d" % (args.suite, args.task_idx, args.state_id), flush=True)
print("Object: %s  Target: %s  Eligible: %s" % (primary_object_site, target_site, teacher_eligible), flush=True)
if not teacher_eligible:
    print("ABSTAIN: %s" % abstain_reason, flush=True)

# ── Model ──
from transformers import AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OpenVLAForActionPrediction.from_pretrained(MODEL_PATH, local_files_only=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = model.to(dtype=torch.bfloat16, device=device)
model.eval()
print("Model loaded", flush=True)

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

# Object site lookup
obj_sid = env.sim.model.site_name2id(primary_object_site)

# Target site lookup (fail-closed)
target_valid = False
target_x = target_y = target_z = float("nan")
if target_site and teacher_eligible:
    try:
        tgt_sid = env.sim.model.site_name2id(target_site)
        txyz = env.sim.data.site_xpos[tgt_sid].copy()
        target_x, target_y, target_z = float(txyz[0]), float(txyz[1]), float(txyz[2])
        target_valid = True
    except ValueError:
        print("FATAL: target site '%s' not found in scene (eligible task)" % target_site, flush=True)
        if not args.canary:
            sys.exit(1)
        target_valid = False

# ── Rollout ──
telemetry_rows = []
privileged_records = []
_eef_prev = None

for step in range(args.max_steps):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()

    action, _, _, _ = decode_with_scores(model, processor, device, raw, task_obj.language,
        args.suite, 8, libero_preprocess_backend="upstream_tf_jpeg",
        center_crop=True, resize_size=224, drop_attention_mask=True)
    env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # Privileged state
    gs = physical_gripper_state(env)
    q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos", [])) > 0 else 0.0
    q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos", [])) > 1 else 0.0
    grip_width = abs(q7) + abs(q8)
    raw_gripper = float(action[-1])
    env_gripper = float(env_action[-1])
    qpos_sum = q7 + q8

    # EEF
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    eef_vx = eef_x - _eef_prev[0] if _eef_prev else 0.0
    eef_vy = eef_y - _eef_prev[1] if _eef_prev else 0.0
    eef_vz = eef_z - _eef_prev[2] if _eef_prev else 0.0
    _eef_prev = (eef_x, eef_y, eef_z)

    # Object
    obj_pos = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
    obj_eef_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))

    # Target distance
    if target_valid:
        obj_tgt_dist = float(np.sqrt((obj_x-target_x)**2 + (obj_y-target_y)**2 + (obj_z-target_z)**2))
    else:
        obj_tgt_dist = float("nan")

    # Attack markers (always false for CLEAN)
    attack_this = False
    attack_frames = 0

    # Telemetry
    telemetry_rows.append({
        "step": step, "suite": args.suite, "task_idx": args.task_idx,
        "state_id": args.state_id, "eval_seed": args.eval_seed,
        "condition": "CLEAN", "attack_this": attack_this, "attack_frames": attack_frames,
        "gripper_qpos_left": q7, "gripper_qpos_right": q8,
        "gripper_qpos_sum": qpos_sum, "gripper_qpos": qpos_sum,
        "gripper_width": grip_width, "raw_gripper": raw_gripper,
        "env_gripper": env_gripper,
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
    })

    # Privileged record
    priv_rec = {
        "step_idx": step, "policy_step_idx": step,
        "teacher_privileged_state_available": teacher_eligible and target_valid,
        "gripper_command": raw_gripper,
        "gripper_qpos": qpos_sum,
        "gripper_width": grip_width,
        "gripper_opening_proxy": grip_width,
        "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
        "object_eef_distance": obj_eef_dist,
        "object_to_target_distance": obj_tgt_dist,
        "object_pose_json": json.dumps([obj_x, obj_y, obj_z]),
        "target_pose_json": json.dumps([target_x, target_y, target_z]),
        "teacher_eligible": teacher_eligible,
        "target_valid": target_valid,
    }
    # If not eligible, mark privileged state as unavailable for teacher
    if not teacher_eligible or not target_valid:
        priv_rec["teacher_privileged_state_available"] = False
    privileged_records.append(priv_rec)

    obs, reward, done, info = env.step(env_action)
    if done: break

env.close()

# ── Write outputs ──
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=telemetry_rows[0].keys())
    w.writeheader(); w.writerows(telemetry_rows)

with open(out / "privileged_step_records.jsonl", "w") as f:
    for rec in privileged_records:
        f.write(json.dumps(rec) + "\n")

with open(out / "episode_summary.json", "w") as f:
    json.dump({"suite": args.suite, "task_idx": args.task_idx,
        "task_name": task_obj.name, "state_id": args.state_id,
        "eval_seed": args.eval_seed, "n_steps": len(telemetry_rows),
        "condition": "CLEAN", "teacher_eligible": teacher_eligible,
        "target_valid": target_valid, "target_site": target_site,
        "primary_object_site": primary_object_site}, f, indent=2)

# Schema gate
widths = [r["gripper_width"] for r in privileged_records]
p2p = max(widths) - min(widths) if widths else 0
gate_pass = p2p > 1e-4 and len(telemetry_rows) > 10

complete = {"status": "COMPLETE" if gate_pass else "SCHEMA_FAIL",
    "n_steps": len(telemetry_rows), "opening_proxy_p2p": p2p,
    "target_valid": target_valid, "teacher_eligible": teacher_eligible,
    "timestamp": time.time()}
with open(out / "COMPLETE.json" if gate_pass else "SCHEMA_FAIL.json", "w") as f:
    json.dump(complete, f, indent=2)

print("DONE: steps=%d p2p=%.4f gate=%s" % (len(telemetry_rows), p2p, gate_pass), flush=True)
