#!/usr/bin/env python3
"""Butter_s11 clean artifact-rich canary — minimal mod of v1 CLEAN_D5 runner.

Adds: object_pose, target_pose, eef_obj_dist, obj_target_dist, RGB frames.
Reuses: v1 model loading, D5 detector, decode, env, replay.
"""
import csv, hashlib, io, json, os, sys, time, numpy as np, torch
from pathlib import Path

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
D5_CKPT = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"
D5_CFG = "/data/liuyu/outputs/d5_training/d5_frozen_config.json"

OUTPUT_DIR = "/data/liuyu/outputs/v2_butter_s11_canary"
RENDER_GPU = 5
TASK_IDX = 6; STATE_ID = 11; NUM_WAIT = 10; MAX_STEPS = 400

# --- Load model (identical to v1 runner) ---
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map="auto", max_memory={idx: "10000MiB" for idx in range(visible)} | {"cpu": "128GiB"},
    attn_implementation="eager")
model_dtype = next(model.parameters()).dtype
device = "cuda:0"
for v in model.hf_device_map.values():
    if isinstance(v, int): device = "cuda:%d" % v; break
action_dim = int(model.get_action_dim("libero_object"))
print("Model on %s" % device)

# --- D5 shadow (identical to v1) ---
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
detector = D5FrozenOnlineDetectorV1(D5_CKPT, D5_CFG)
detector.reset()

# --- Replay Butter_s11 (identical to v1) ---
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm["libero_object"]()
task_obj = suite.get_task(TASK_IDX); init_states = suite.get_task_init_states(TASK_IDX)
bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, RENDER_GPU, MAX_STEPS, NUM_WAIT)
obs = env.set_init_state(init_states[STATE_ID])
env, obs = apply_dummy_wait(env, obs, NUM_WAIT)

obj_sid = env.sim.model.site_name2id("butter_1_default_site")
target_sid = env.sim.model.site_name2id("basket_1_default_site")

out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
frame_dir = out / "frames"; frame_dir.mkdir(exist_ok=True)

telemetry = []
d5_emit_step = -1

for step in range(MAX_STEPS):
    if "agentview_image" not in obs: break
    raw = np.asarray(obs["agentview_image"]).copy()

    # Physical gripper (identical to v1)
    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos", [])) > 0 else float("nan")
    q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos", [])) > 1 else float("nan")
    qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")

    # EEF (identical to v1)
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

    # --- NEW: Object/target privileged telemetry ---
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    target_xyz = env.sim.data.site_xpos[target_sid]
    target_x, target_y, target_z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))
    obj_target_dist = float(np.sqrt((obj_x-target_x)**2 + (obj_y-target_y)**2 + (obj_z-target_z)**2))

    # EEF velocity (3-step)
    eef_vx = eef_vy = eef_vz = 0.0
    if len(telemetry) >= 3:
        p3 = telemetry[-3]
        dt = max(1, step - int(p3["step"]))
        eef_vx = (eef_x - float(p3["eef_x"])) / dt
        eef_vy = (eef_y - float(p3["eef_y"])) / dt
        eef_vz = (eef_z - float(p3["eef_z"])) / dt

    # Decode action (identical to v1)
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(
        model, processor, device, raw, instruction, "libero_object", 8,
        libero_official_preprocess=False, libero_preprocess_backend="official_pil_lanczos",
        center_crop=True, resize_size=224, drop_attention_mask=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
    t_vla = time.perf_counter() - t0

    # D5 shadow (identical to v1, including emit recording)
    detector.update(step, raw_grip, env_grip, qpos_sum if not np.isnan(qpos_sum) else float("nan"),
                    eef_x, eef_y, eef_z, 1 if raw_grip > 0.5 else 0,
                    raw_valid=True, env_valid=True, qpos_valid=not np.isnan(qpos_sum), eef_valid=True)
    d5_score = detector.audit_records[-1].get("score", 0) if detector.audit_records else 0
    if d5_emit_step < 0 and detector.emit_step >= 0:
        d5_emit_step = detector.emit_step

    # RGB frame
    from PIL import Image
    fpath = str(frame_dir / "step_%04d.png" % step)
    Image.fromarray(raw).save(fpath)

    is_wait = step < NUM_WAIT
    telemetry.append({
        "step": step, "policy_step_idx": -1 if is_wait else step - NUM_WAIT,
        "phase": "wait" if is_wait else "policy",
        "raw_gripper": raw_grip, "env_gripper": env_grip,
        "q7": q7, "q8": q8, "qpos_sum": qpos_sum,
        "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
        "obj_x": obj_x, "obj_y": obj_y, "obj_z": obj_z,
        "target_x": target_x, "target_y": target_y, "target_z": target_z,
        "eef_obj_dist": eef_obj_dist, "obj_target_dist": obj_target_dist,
        "d5_emit_step": d5_emit_step, "d5_score": d5_score,
        "model_ms": round(t_vla * 1000, 2),
        "rgb_path": fpath,
    })

    obs, _, done, _ = env.step(env_action)
    if done: break

success = bool(env.check_success()) if hasattr(env, "check_success") else False
env.close()

# --- Write outputs ---
with open(out / "step_telemetry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
    w.writeheader(); w.writerows(telemetry)

with open(out / "step_records.jsonl", "w") as f:
    for r in telemetry:
        rec = {
            "step_idx": r["step"], "policy_step_idx": r["policy_step_idx"],
            "phase": r["phase"], "image_path": r["rgb_path"], "image_path_available": True,
            "gripper_command": r["raw_gripper"], "gripper_qpos": r["qpos_sum"],
            "gripper_width": r["qpos_sum"],
            "eef_x": r["eef_x"], "eef_y": r["eef_y"], "eef_z": r["eef_z"],
            "eef_vx": r["eef_vx"], "eef_vy": r["eef_vy"], "eef_vz": r["eef_vz"],
            "object_pose_json": json.dumps([r["obj_x"], r["obj_y"], r["obj_z"], 0, 0, 1.0, 0]),
            "target_pose_json": json.dumps([r["target_x"], r["target_y"], r["target_z"]]),
            "object_to_target_distance": r["obj_target_dist"],
            "object_eef_distance": r["eef_obj_dist"],
            "teacher_privileged_state_available": r["phase"] == "policy",
            "d5_score": r["d5_score"], "d5_emit_step": r["d5_emit_step"],
            "reward": 0.0, "done": False, "success_so_far": False,
        }
        f.write(json.dumps(rec) + "\n")

summary = {
    "parent": "butter_s11", "state_id": STATE_ID,
    "n_steps": len(telemetry), "d5_emit_step": d5_emit_step,
    "task_success": success,
    "step10_eef_x": telemetry[10]["eef_x"] if len(telemetry) > 10 else None,
    "step10_eef_y": telemetry[10]["eef_y"] if len(telemetry) > 10 else None,
    "step10_eef_z": telemetry[10]["eef_z"] if len(telemetry) > 10 else None,
}
with open(out / "canary_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("Butter_s11 canary: steps=%d success=%s d5_emit=%d" % (
    len(telemetry), success, d5_emit_step))
