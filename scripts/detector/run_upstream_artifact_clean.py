#!/usr/bin/env python3
"""Upstream artifact-rich clean collector — records per-step features for SC5 corpus.

Non-interference: runs exactly the same closed-loop as run_upstream_clean30.py,
producing the same actions, but additionally records privileged state and 25D features.

Usage:
  python launch_upstream_collector.py --profile fp32_upstream --corpus_plan plan.json ...
"""
import os, sys, json, csv, hashlib, argparse, time, math
import numpy as np
from PIL import Image

assert "MUJOCO_GL" in os.environ, "MUJOCO_GL not set"
assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES not set"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from gripper_attack.openvla_preprocess import prepare_openvla_image, resolve_backend

SC5_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

CSV_FIELDS = [
    "episode_key", "profile", "task_idx", "init_idx", "init_state_sha", "step_idx",
    # Privileged state
    "object_pose_x", "object_pose_y", "object_pose_z",
    "target_pose_x", "target_pose_y", "target_pose_z",
    "object_to_target_distance", "object_eef_distance",
    "teacher_privileged_state_available",
    # Environment state
    "reward", "done", "check_success", "termination",
    # 25D features
] + SC5_FEATURES


def sha256_hex(data) -> str:
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha256(data).hexdigest()


def _compute_25d_features(history, step, raw_gripper, env_gripper, gripper_qpos_vec,
                          gripper_opening_proxy, eef_pos, eef_vel, action_vec):
    """Compute 25D causal features from step history (no future peeking)."""
    f = {}

    # 13D direct/vector
    qpos_scalar = float(gripper_qpos_vec[0]) if hasattr(gripper_qpos_vec, '__len__') else float(gripper_qpos_vec)
    f["gripper_command"] = float(raw_gripper)
    f["gripper_qpos"] = float(qpos_scalar)
    f["gripper_opening_proxy"] = float(gripper_opening_proxy)
    f["eef_x"] = float(eef_pos[0])
    f["eef_y"] = float(eef_pos[1])
    f["eef_z"] = float(eef_pos[2])
    f["eef_vx"] = float(eef_vel[0]) if len(eef_vel) > 0 else 0.0
    f["eef_vy"] = float(eef_vel[1]) if len(eef_vel) > 1 else 0.0
    f["eef_vz"] = float(eef_vel[2]) if len(eef_vel) > 2 else 0.0
    f["action_dx"] = float(action_vec[0])
    f["action_dy"] = float(action_vec[1])
    f["action_dz"] = float(action_vec[2])
    f["action_gripper"] = float(action_vec[6])

    # Causal derived features
    close_intent = raw_gripper <= 0.5
    prev_close = history[-1].get("close_intent", False) if history else False
    streak = history[-1].get("close_streak", 0) if history else 0
    open_streak = history[-1].get("open_streak", 0) if history else 0
    flips = history[-1].get("flip_count", 0) if history else 0
    last_close = history[-1].get("last_close_step", -1) if history else -1
    onset_detected = history[-1].get("onset_detected", False) if history else False

    new_streak = streak + 1 if close_intent else 0
    new_open_streak = open_streak + 1 if not close_intent else 0
    new_flips = flips
    new_last_close = last_close
    new_onset = onset_detected

    if prev_close is not None:
        if close_intent and not prev_close:
            new_onset = True
            new_last_close = step
            if step > 0:
                new_flips += 1
        if not close_intent and prev_close:
            new_flips += 1

    f["recent_close_streak"] = float(min(new_streak, 999))
    f["recent_open_streak"] = float(min(new_open_streak, 999))
    f["recent_gripper_flip_count"] = float(new_flips)
    f["close_onset"] = 1.0 if (new_onset and new_streak == 1) else 0.0
    f["time_since_close"] = float(step - new_last_close) if new_last_close >= 0 else float(step + 1)

    eef_speed = math.sqrt(f["eef_vx"]**2 + f["eef_vy"]**2 + f["eef_vz"]**2)
    f["eef_speed"] = float(eef_speed)

    ref_z = history[new_last_close].get("eef_z", f["eef_z"]) if new_last_close >= 0 and new_last_close < len(history) else f["eef_z"]
    f["eef_z_delta_since_close"] = float(f["eef_z"] - ref_z) if new_last_close >= 0 else 0.0

    # qpos deltas
    prev_qpos = [history[-i].get("gripper_qpos", qpos_scalar) for i in range(1, 4) if len(history) >= i]
    f["qpos_delta_1"] = float(f["gripper_qpos"] - prev_qpos[0]) if len(prev_qpos) >= 1 else 0.0
    f["qpos_delta_3"] = float(f["gripper_qpos"] - prev_qpos[-1]) if len(prev_qpos) >= 3 else 0.0

    # Opening proxy delta/variance
    prev_ops = [history[-i].get("gripper_opening_proxy", gripper_opening_proxy) for i in range(1, 6) if len(history) >= i]
    f["opening_proxy_delta_3"] = float(f["gripper_opening_proxy"] - prev_ops[-1]) if len(prev_ops) >= 3 else 0.0
    if len(prev_ops) >= 5:
        ops_vals = prev_ops[:5] + [f["gripper_opening_proxy"]]
        f["opening_proxy_variance_5"] = float(np.var(ops_vals))
    else:
        f["opening_proxy_variance_5"] = 0.0

    # EEF speed variance
    prev_speeds = [history[-i].get("eef_speed", eef_speed) for i in range(1, 6) if len(history) >= i]
    if len(prev_speeds) >= 5:
        sp_vals = prev_speeds[:5] + [eef_speed]
        f["eef_speed_variance_5"] = float(np.var(sp_vals))
    else:
        f["eef_speed_variance_5"] = 0.0

    # Update history
    hist_entry = {
        "close_intent": close_intent, "close_streak": new_streak,
        "open_streak": new_open_streak, "flip_count": new_flips,
        "last_close_step": new_last_close, "onset_detected": new_onset,
        "eef_z": f["eef_z"], "gripper_qpos": qpos_scalar,
        "gripper_opening_proxy": gripper_opening_proxy, "eef_speed": eef_speed,
    }
    history.append(hist_entry)

    return f


def run_episode_artifact(model, proc, task_suite, ti, ii, seed, max_steps, wait_steps,
                         preprocess_backend, profile_name):
    """Run one episode and collect per-step artifacts. Non-interference guaranteed."""
    DEV = next(model.parameters()).device
    stats = model.get_action_stats("libero_spatial")

    task = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[ii])
    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    init_state_sha = sha256_hex(init_states[ii])
    label = "task%d_init%d" % (ti, ii)

    rows = []
    feature_history = []
    done, success, error = False, False, None
    prev_env_grip = None
    t0 = time.time()

    for step in range(max_steps):
        raw = obs["agentview_image"]
        processed = prepare_openvla_image(raw, libero_preprocess_backend=preprocess_backend,
                                          center_crop=True, resize_size=224)

        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        # Gripper post-process (identical to clean30 runner)
        raw_g = act[6]
        norm_g = (raw_g * 2) - 1
        bin_g = 1.0 if norm_g >= 0 else -1.0
        env_g = -bin_g
        env_act = np.zeros(7)
        env_act[:6] = act[:6]
        env_act[6] = env_g

        # Privileged state extract
        eef_pos = obs.get("robot0_eef_pos", [float("nan")]*3)
        eef_vel = obs.get("robot0_eef_vel", [0.0]*3)
        gripper_qpos = obs.get("robot0_gripper_qpos", [float("nan")])
        obj_pos = obs.get("object_pos", [float("nan")]*3) if "object_pos" in obs else [float("nan")]*3
        obj_quat = obs.get("object_quat", [float("nan")]*4) if "object_quat" in obs else [float("nan")]*4
        tgt_pos = obs.get("target_pos", [float("nan")]*3) if "target_pos" in obs else [float("nan")]*3
        obj_tgt_dist = float(np.linalg.norm(np.array(obj_pos[:3]) - np.array(tgt_pos[:3])))
        obj_eef_dist = float(np.linalg.norm(np.array(obj_pos[:3]) - np.array(eef_pos)))
        has_privileged = not (np.any(np.isnan(obj_pos)) or np.any(np.isnan(tgt_pos)))

        # Gripper opening proxy (from qpos)
        qpos_scalar = float(gripper_qpos[0]) if hasattr(gripper_qpos, '__len__') else float(gripper_qpos)
        grip_proxy = max(0.0, min(1.0, 1.0 - qpos_scalar))

        # Compute 25D features
        features_25d = _compute_25d_features(
            feature_history, step, raw_g, env_g, gripper_qpos, grip_proxy, eef_pos, eef_vel, act
        )

        # Step environment
        try:
            obs, rew, done, info = env.step(env_act.tolist())
        except Exception as e:
            error = str(e)
            break

        # Success check
        chk_success = False
        try:
            chk_success = bool(env.check_success())
        except Exception:
            pass
        if chk_success:
            done, success = True, True

        # Build row
        row = {
            "episode_key": label, "profile": profile_name,
            "task_idx": ti, "init_idx": ii, "init_state_sha": init_state_sha,
            "step_idx": step,
            "object_pose_x": float(obj_pos[0]), "object_pose_y": float(obj_pos[1]),
            "object_pose_z": float(obj_pos[2]),
            "target_pose_x": float(tgt_pos[0]), "target_pose_y": float(tgt_pos[1]),
            "target_pose_z": float(tgt_pos[2]),
            "object_to_target_distance": float(obj_tgt_dist),
            "object_eef_distance": float(obj_eef_dist),
            "teacher_privileged_state_available": str(has_privileged),
            "reward": str(rew), "done": str(done),
            "check_success": str(chk_success),
            "termination": "success" if success else ("error" if error else "timeout"),
        }
        for fn in SC5_FEATURES:
            row[fn] = str(features_25d.get(fn, "nan"))

        rows.append(row)
        if done:
            break

    env.close()
    dt = time.time() - t0
    return {
        "label": label, "task_idx": ti, "init_idx": ii,
        "steps": len(rows), "success": success,
        "termination": "success" if success else ("error" if error else "timeout"),
        "error": error, "duration_s": round(dt, 1),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_plan", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    parser.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    parser.add_argument("--profile_name", default="fp32_eager")
    parser.add_argument("--preprocess_backend", default="upstream_tf_jpeg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    DTYPE = torch.float32 if args.dtype == "float32" else torch.bfloat16
    backend = resolve_backend(args.preprocess_backend)

    with open(args.corpus_plan) as f:
        plan = json.load(f)
    episodes = plan["episodes"]
    print("Corpus plan: %d episodes  dtype=%s  attn=%s  backend=%s" % (
        len(episodes), args.dtype, args.attn, backend))

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True,
        low_cpu_mem_usage=True)
    proc = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    if args.attn == "flash_attention_2":
        assert actual_attn == "flash_attention_2", "FATAL: flash_attention_2 not active"
    print("Devices: %s  actual_attn: %s  VRAM: %.1f GiB" % (
        sorted(set(str(p.device) for p in model.parameters())),
        actual_attn, torch.cuda.max_memory_allocated() / 1024**3))

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    results = []

    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = "task%d_init%d" % (ti, ii)
        ep_dir = os.path.join(args.output_dir, label)
        done_file = os.path.join(ep_dir, ".done")

        if args.resume and os.path.exists(done_file):
            print("[%d/%d] %s SKIP" % (i + 1, len(episodes), label))
            continue

        os.makedirs(ep_dir, exist_ok=True)
        print("[%d/%d] %s" % (i + 1, len(episodes), label), end=" ", flush=True)

        res, rows = run_episode_artifact(model, proc, task_suite, ti, ii, args.seed,
                                         args.max_steps, 10, backend, args.profile_name)
        results.append(res)
        all_rows.extend(rows)

        json.dump(res, open(os.path.join(ep_dir, "result.json"), "w"), indent=2)
        with open(os.path.join(ep_dir, "trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        with open(done_file, "w") as f:
            f.write("")

        print("OK steps=%d succ=%d" % (res["steps"], res["success"]))

    succ = sum(1 for r in results if r["success"])
    print("Done: %d/%d (%.1f%%)" % (succ, len(results), 100 * succ / max(1, len(results))))

    # Write aggregate CSV
    csv_path = os.path.join(args.output_dir, "corpus_flat.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)

    manifest = {
        "runner": os.path.basename(__file__),
        "profile": args.profile_name, "dtype": args.dtype, "attn": args.attn,
        "actual_attn": actual_attn, "backend": backend,
        "total": len(results), "success": succ,
        "corpus_csv": csv_path, "corpus_csv_sha": sha256_hex(open(csv_path, "rb").read()) if all_rows else "N/A",
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
