#!/usr/bin/env python3
"""Upstream artifact-rich clean collector — canonical implementation.

Uses shared canonical modules for 25D feature extraction, gripper semantics,
and schema adaptation. No local feature re-implementation.

Non-interference: identical closed-loop policy to run_upstream_clean30.py.
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
from gripper_attack.sc5_streaming_features_v2 import (
    SC5StreamingFeatureAdapterV2, FEATURE_NAMES,
)
from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper

# Shard support
try:
    with open(os.path.join(REPO_ROOT, "configs", "detector",
                           "libero_spatial_object_target_binding.json")) as _bf:
        OBJECT_TARGET_BINDING = json.load(_bf)
except Exception:
    OBJECT_TARGET_BINDING = None


def sha256_hex(data) -> str:
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    elif isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(str(data).encode()).hexdigest()


def _qpos_to_scalar(qpos):
    """Extract scalar gripper qpos from MuJoCo vector."""
    if isinstance(qpos, (list, np.ndarray)):
        arr = np.asarray(qpos, dtype=np.float64).ravel()
        if len(arr) >= 2:
            return float(np.sum(np.abs(arr)))
        return float(arr[0]) if len(arr) > 0 else float("nan")
    return float(qpos)


def _qpos_to_opening_proxy(qpos):
    """Gripper opening proxy from canonical semantics.
    Physical OPEN = increasing abs_sum; CLOSED ≈ 0.
    Normalised proxy: min(1.0, max(0.0, abs_sum / expected_open_max)).
    """
    scalar = _qpos_to_scalar(qpos)
    expected_open_max = 0.06
    return float(min(1.0, max(0.0, scalar / max(expected_open_max, 1e-9))))


def _read_mujoco_pose(env, body_name):
    """Read body position from MuJoCo sim. Returns [x,y,z] or [nan]*3."""
    try:
        sim = env.sim
        body_id = sim.model.body_name2id(body_name)
        pos = sim.data.body_xpos[body_id].copy()
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        return [float("nan")] * 3


def _verify_binding(env, task_idx):
    """Verify object/target MuJoCo binding. Returns (obj_name, tgt_name, ok)."""
    if OBJECT_TARGET_BINDING is None:
        return "unknown", "unknown", False
    tasks = OBJECT_TARGET_BINDING.get("tasks", [])
    if task_idx >= len(tasks):
        return "unknown", "unknown", False
    spec = tasks[task_idx]
    obj_name = spec.get("object", "akita_black_bowl_1")
    tgt_name = spec.get("target", "plate_region")
    try:
        sim = env.sim
        obj_id = sim.model.body_name2id(obj_name)
        tgt_id = sim.model.site_name2id(tgt_name) if hasattr(sim.model, 'site_name2id') else None
        if tgt_id is None:
            tgt_id = sim.model.body_name2id(tgt_name)
        ok = obj_id >= 0 and tgt_id >= 0
    except Exception:
        ok = False
    return obj_name, tgt_name, ok


def run_episode(model, proc, task_suite, ti, ii, seed, max_steps, wait_steps,
                preprocess_backend, profile_name, run_attrs):
    """Run one episode with artifact collection. Non-interference: same policy."""
    DEV = next(model.parameters()).device

    task = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name
    init_state_sha = sha256_hex(init_states[ii])
    label = "task%d_init%d" % (ti, ii)

    # MuJoCo binding verification
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[ii])
    obj_name, tgt_name, binding_ok = _verify_binding(env, ti)
    if not binding_ok:
        env.close()
        raise RuntimeError(f"Binding check failed for {label}: obj={obj_name} tgt={tgt_name}")

    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # Canonical feature adapter
    adapter = SC5StreamingFeatureAdapterV2()
    adapter.reset()

    rows = []
    done, success, error = False, False, None
    prev_env_grip = None
    t0 = time.time()

    for step in range(max_steps):
        raw_img = obs["agentview_image"]
        processed = prepare_openvla_image(raw_img, libero_preprocess_backend=preprocess_backend,
                                          center_crop=True, resize_size=224)

        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        raw_g = float(act[6])
        norm_g = (raw_g * 2.0) - 1.0
        bin_g = 1.0 if norm_g >= 0 else -1.0
        env_g = -bin_g
        env_act = np.zeros(7)
        env_act[:6] = act[:6]
        env_act[6] = env_g

        # --- Privileged state from MuJoCo ---
        obj_pose = _read_mujoco_pose(env, obj_name)
        tgt_pose = _read_mujoco_pose(env, tgt_name)

        eef_pos = obs.get("robot0_eef_pos", [float("nan")]*3)
        eef_pos_arr = np.asarray(eef_pos, dtype=np.float64).ravel()
        has_eef = len(eef_pos_arr) >= 3 and not np.any(np.isnan(eef_pos_arr[:3]))

        # EEF velocity: try direct obs key, otherwise causal backward diff
        eef_vel = None
        if "robot0_eef_vel" in obs:
            eef_vel = obs["robot0_eef_vel"]
        elif "robot0_eef_vel_lin" in obs:
            eef_vel = obs["robot0_eef_vel_lin"]
        # Causal backward difference from adapter history if available
        if eef_vel is None:
            if len(adapter.history) > 0 and adapter.history[-1].get("valid"):
                prev = adapter.history[-1]
                if step == 0:
                    eef_vel = np.array([float("nan")]*3)
                else:
                    eef_vel = np.array([eef_pos_arr[0] - prev["eef_x"],
                                       eef_pos_arr[1] - prev["eef_y"],
                                       eef_pos_arr[2] - prev["eef_z"]])
            else:
                eef_vel = np.array([float("nan")]*3)
        eef_vel_arr = np.asarray(eef_vel, dtype=np.float64).ravel()

        qpos_raw = obs.get("robot0_gripper_qpos", np.array([float("nan")]))
        qpos_scalar = _qpos_to_scalar(qpos_raw)
        opening_proxy = _qpos_to_opening_proxy(qpos_raw)

        obj_tgt_dist = float(np.linalg.norm(np.array(obj_pose) - np.array(tgt_pose)))
        obj_eef_dist = float(np.linalg.norm(np.array(obj_pose) - eef_pos_arr[:3]))
        has_priv = (not np.any(np.isnan(obj_pose))) and (not np.any(np.isnan(tgt_pose)))

        # Canonical 25D features via adapter
        feat_out = adapter.update(
            step_id=step,
            raw_gripper=raw_g, env_gripper=env_g,
            gripper_qpos=qpos_scalar, gripper_opening_proxy=opening_proxy,
            eef_x=float(eef_pos_arr[0]), eef_y=float(eef_pos_arr[1]),
            eef_z=float(eef_pos_arr[2]),
            eef_vx=float(eef_vel_arr[0]) if len(eef_vel_arr) > 0 else float("nan"),
            eef_vy=float(eef_vel_arr[1]) if len(eef_vel_arr) > 1 else float("nan"),
            eef_vz=float(eef_vel_arr[2]) if len(eef_vel_arr) > 2 else float("nan"),
            action_dx=float(act[0]), action_dy=float(act[1]),
            action_dz=float(act[2]), action_gripper=float(act[6]),
        )

        # Step environment
        try:
            obs, rew, done, info = env.step(env_act.tolist())
        except Exception as e:
            error = str(e)
            break

        chk_success = False
        try:
            chk_success = bool(env.check_success())
        except Exception:
            pass
        if chk_success:
            done, success = True, True

        # Build artifact row
        row = {
            "episode_key": label, "profile": profile_name,
            "task_idx": ti, "init_idx": ii,
            "init_state_sha": init_state_sha, "step_idx": step,
            "object_pose_x": float(obj_pose[0]), "object_pose_y": float(obj_pose[1]),
            "object_pose_z": float(obj_pose[2]),
            "target_pose_x": float(tgt_pose[0]), "target_pose_y": float(tgt_pose[1]),
            "target_pose_z": float(tgt_pose[2]),
            "object_to_target_distance": float(obj_tgt_dist),
            "object_eef_distance": float(obj_eef_dist),
            "teacher_privileged_state_available": str(has_priv),
            "reward": str(rew), "done": str(done),
            "check_success": str(chk_success),
            "termination": "success" if success else ("error" if error else "timeout"),
            "feature_valid": str(feat_out.get("valid", False)),
            "feature_error": str(feat_out.get("error", "")),
            "raw_gripper": str(raw_g), "env_gripper": str(env_g),
            "qpos_scalar": str(qpos_scalar), "opening_proxy": str(opening_proxy),
            "binding_ok": str(binding_ok), "object_name": obj_name, "target_name": tgt_name,
        }
        if feat_out.get("valid") and feat_out.get("features"):
            for fn in FEATURE_NAMES:
                row[fn] = str(feat_out["features"].get(fn, "nan"))
        else:
            for fn in FEATURE_NAMES:
                row[fn] = "nan"

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
        "binding_ok": binding_ok, "object_name": obj_name,
    }, rows, adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_manifest", required=True,
                        help="Frozen shard JSON (list of episodes)")
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

    with open(args.episode_manifest) as f:
        shard = json.load(f)
    episodes = shard.get("episodes", shard if isinstance(shard, list) else [])
    print("Shard: %s — %d episodes  dtype=%s  attn=%s  backend=%s" % (
        shard.get("plan_name", "unnamed"), len(episodes), args.dtype, args.attn, backend))

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

    # Run attrs for resume integrity
    runner_sha = sha256_hex(open(__file__, "rb").read())
    shard_sha = sha256_hex(open(args.episode_manifest, "rb").read())
    run_attrs = {
        "runner_sha": runner_sha, "shard_sha": shard_sha,
        "backend": backend, "dtype": args.dtype, "attn": args.attn,
        "seed": args.seed, "max_steps": args.max_steps,
        "profile_name": args.profile_name,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = "task%d_init%d" % (ti, ii)
        ep_dir = os.path.join(args.output_dir, label)
        done_file = os.path.join(ep_dir, ".done")
        manifest_file = os.path.join(ep_dir, "episode_manifest.json")

        if args.resume and os.path.exists(done_file):
            if os.path.exists(manifest_file):
                stored = json.load(open(manifest_file))
                ok = all(str(stored.get(k)) == str(run_attrs.get(k))
                        for k in ["runner_sha", "shard_sha", "backend",
                                  "dtype", "attn", "seed", "max_steps"])
                if ok:
                    print("[%d/%d] %s SKIP" % (i + 1, len(episodes), label))
                    if os.path.exists(os.path.join(ep_dir, "result.json")):
                        results.append(json.load(open(os.path.join(ep_dir, "result.json"))))
                    continue
                else:
                    print("[%d/%d] %s INTEGRITY_FAIL — re-running" % (i + 1, len(episodes), label))
                    os.remove(done_file)

        os.makedirs(ep_dir, exist_ok=True)
        print("[%d/%d] %s" % (i + 1, len(episodes), label), end=" ", flush=True)

        try:
            res, rows, adapter = run_episode(
                model, proc, task_suite, ti, ii, args.seed, args.max_steps, 10,
                backend, args.profile_name, run_attrs)
        except RuntimeError as e:
            print("BINDING_FAIL: %s" % str(e))
            sys.exit(1)

        res["runner_sha"] = runner_sha
        results.append(res)

        # Atomic save: episode_manifest, result, trace, .done (last)
        ep_manifest = dict(run_attrs, label=label, task_idx=ti, init_idx=ii,
                          steps=res["steps"], success=res["success"],
                          binding_ok=res.get("binding_ok", False))
        json.dump(ep_manifest, open(manifest_file, "w"), indent=2)
        json.dump(res, open(os.path.join(ep_dir, "result.json"), "w"), indent=2)

        fieldnames = ["episode_key", "profile", "task_idx", "init_idx",
                      "init_state_sha", "step_idx",
                      "object_pose_x", "object_pose_y", "object_pose_z",
                      "target_pose_x", "target_pose_y", "target_pose_z",
                      "object_to_target_distance", "object_eef_distance",
                      "teacher_privileged_state_available",
                      "reward", "done", "check_success", "termination",
                      "feature_valid", "feature_error",
                      "raw_gripper", "env_gripper", "qpos_scalar", "opening_proxy",
                      "binding_ok", "object_name", "target_name"] + FEATURE_NAMES
        with open(os.path.join(ep_dir, "trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        with open(done_file, "w") as f:
            f.write("")

        print("OK steps=%d succ=%d binding_ok=%s" % (res["steps"], res["success"], res.get("binding_ok", "?")))

    succ = sum(1 for r in results if r["success"])
    print("Done: %d/%d (%.1f%%)" % (succ, len(results), 100 * succ / max(1, len(results))))

    manifest = {
        "runner": os.path.basename(__file__), "runner_sha": runner_sha,
        "shard_sha": shard_sha, "profile": args.profile_name,
        "dtype": args.dtype, "attn": args.attn, "actual_attn": actual_attn,
        "backend": backend, "total": len(results), "success": succ,
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
