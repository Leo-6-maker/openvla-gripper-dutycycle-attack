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


def _get_obs_keys(obs):
    """Discover object and target observation keys from LIBERO obs dict.
    Object is always akita_black_bowl_1; target is plate_1.
    Returns (obj_pos_key, tgt_pos_key)."""
    for prefix in ["akita_black_bowl_1", "akita_black_bowl"]:
        pk = prefix + "_pos"
        if pk in obs:
            obj_key = pk
            break
    else:
        obj_key = None

    for prefix in ["plate_1", "plate"]:
        pk = prefix + "_pos"
        if pk in obs:
            tgt_key = pk
            break
    else:
        tgt_key = None

    return obj_key, tgt_key


def _verify_binding(obs):
    """Verify object/target keys exist in observation. Returns (obj_key, tgt_key, ok)."""
    obj_key, tgt_key = _get_obs_keys(obs)
    ok = obj_key is not None and tgt_key is not None
    return obj_key or "unknown", tgt_key or "unknown", ok


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

    # MuJoCo binding verification via obs keys
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[ii])
    obj_key, tgt_key, binding_ok = _verify_binding(obs)
    if not binding_ok:
        env.close()
        raise RuntimeError(f"Binding check failed for {label}: obj_key={obj_key} tgt_key={tgt_key}")

    # Track last two EEF positions during wait steps for step-0 velocity
    wait_eef_positions = []
    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        eef_p = obs.get("robot0_eef_pos", [0.0]*3)
        wait_eef_positions.append(np.asarray(eef_p, dtype=np.float64).ravel()[:3].copy())
        if len(wait_eef_positions) > 2:
            wait_eef_positions = wait_eef_positions[-2:]

    # Canonical feature adapter
    adapter = SC5StreamingFeatureAdapterV2()
    adapter.reset()

    rows = []
    eef_pos_history = []  # for causal backward-diff velocity
    done, success, error = False, False, None
    prev_env_grip = None
    t0 = time.time()

    for step in range(max_steps):
        raw_img = obs["agentview_image"]
        processed = prepare_openvla_image(raw_img, libero_preprocess_backend=preprocess_backend,
                                          center_crop=True, resize_size=224)
        processed_sha = sha256_hex(np.array(processed))

        prompt = "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        inputs = proc(prompt, processed, return_tensors="pt")
        ids = inputs["input_ids"].to(device=DEV)
        px = inputs["pixel_values"].to(dtype=model.dtype, device=DEV)
        ids_sha = sha256_hex(ids)
        pixel_sha = sha256_hex(px)

        result = model.predict_action(input_ids=ids, pixel_values=px,
                                      unnorm_key="libero_spatial", do_sample=False)
        act = np.array(result).flatten() if not isinstance(result, np.ndarray) else result.flatten()

        raw_g = float(act[6])
        env_g = raw_gripper_to_env_gripper(raw_g)
        env_act = np.zeros(7)
        env_act[:6] = act[:6]
        env_act[6] = env_g

        # --- Privileged state from obs keys ---
        obj_pose_raw = obs.get(obj_key, [float("nan")]*3)
        tgt_pose_raw = obs.get(tgt_key, [float("nan")]*3)
        obj_pose = np.asarray(obj_pose_raw, dtype=np.float64).ravel()[:3]
        tgt_pose = np.asarray(tgt_pose_raw, dtype=np.float64).ravel()[:3]

        eef_pos = obs.get("robot0_eef_pos", [float("nan")]*3)
        eef_pos_arr = np.asarray(eef_pos, dtype=np.float64).ravel()

        # EEF velocity: causal backward difference from collector's own position history.
        # LIBERO does not expose robot0_eef_vel; compute from sequential eef_pos diffs.
        # Step 0 uses wait-step positions (final_wait - penultimate_wait). No zero-fill.
        if step == 0 and len(wait_eef_positions) >= 2:
            prev_pos = wait_eef_positions[-2]
            cur_pos = wait_eef_positions[-1]
            eef_vx = float(cur_pos[0] - prev_pos[0])
            eef_vy = float(cur_pos[1] - prev_pos[1])
            eef_vz = float(cur_pos[2] - prev_pos[2])
        elif step == 0:
            eef_vx, eef_vy, eef_vz = 0.0, 0.0, 0.0
        else:
            prev_pos = eef_pos_history[-1]
            eef_vx = float(eef_pos_arr[0] - prev_pos[0])
            eef_vy = float(eef_pos_arr[1] - prev_pos[1])
            eef_vz = float(eef_pos_arr[2] - prev_pos[2])
        eef_pos_history.append((float(eef_pos_arr[0]), float(eef_pos_arr[1]), float(eef_pos_arr[2])))

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
            eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
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
            "binding_ok": str(binding_ok), "object_key": obj_key, "target_key": tgt_key,
            "raw_action_json": json.dumps(act.tolist()),
            "raw_action_sha256": sha256_hex(act.tobytes()),
            "env_action_json": json.dumps(env_act.tolist()),
            "env_action_sha256": sha256_hex(env_act.tobytes()),
            "processed_image_sha256": processed_sha,
            "processor_pixel_sha256": pixel_sha,
            "input_ids_sha256": ids_sha,
        }
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
        "binding_ok": binding_ok, "object_key": obj_key, "target_key": tgt_key,
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

    # Hash supporting modules
    preprocess_path = os.path.join(REPO_ROOT, "src", "gripper_attack", "openvla_preprocess.py")
    features_path = os.path.join(REPO_ROOT, "src", "gripper_attack", "sc5_streaming_features_v2.py")
    exec_spec_path = os.path.join(REPO_ROOT, "src", "gripper_attack", "openvla_libero_exec_spec.py")
    binding_path = os.path.join(REPO_ROOT, "configs", "detector", "libero_spatial_object_target_binding.json")

    run_attrs = {
        "runner_sha": runner_sha, "shard_sha": shard_sha,
        "preprocess_sha": sha256_hex(open(preprocess_path, "rb").read()) if os.path.exists(preprocess_path) else "MISSING",
        "feature_extractor_sha": sha256_hex(open(features_path, "rb").read()) if os.path.exists(features_path) else "MISSING",
        "gripper_exec_spec_sha": sha256_hex(open(exec_spec_path, "rb").read()) if os.path.exists(exec_spec_path) else "MISSING",
        "binding_config_sha": sha256_hex(open(binding_path, "rb").read()) if os.path.exists(binding_path) else "MISSING",
        "backend": backend, "dtype": args.dtype, "attn": args.attn,
        "requested_attn": args.attn, "actual_attn": actual_attn,
        "seed": args.seed, "max_steps": args.max_steps,
        "profile_name": args.profile_name,
        "model_path": args.model_path,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, ep in enumerate(episodes):
        ti, ii = ep["task_idx"], ep["init_idx"]
        label = "task%d_init%d" % (ti, ii)
        ep_dir = os.path.join(args.output_dir, label)
        done_file = os.path.join(ep_dir, ".done")
        manifest_file = os.path.join(ep_dir, "episode_manifest.json")

        # Resume with fail-closed integrity check
        skip = False
        if args.resume and os.path.exists(done_file):
            if os.path.exists(manifest_file):
                stored = json.load(open(manifest_file))
                mismatch = []
                for k in ["runner_sha", "shard_sha", "preprocess_sha", "feature_extractor_sha",
                          "gripper_exec_spec_sha", "binding_config_sha",
                          "backend", "dtype", "requested_attn", "actual_attn",
                          "seed", "max_steps", "profile_name"]:
                    sv = str(stored.get(k, ""))
                    rv = str(run_attrs.get(k, ""))
                    if sv != rv:
                        mismatch.append(k)
                if not mismatch:
                    print("[%d/%d] %s SKIP" % (i + 1, len(episodes), label))
                    if os.path.exists(os.path.join(ep_dir, "result.json")):
                        results.append(json.load(open(os.path.join(ep_dir, "result.json"))))
                    skip = True
                else:
                    print("[%d/%d] %s INTEGRITY_MISMATCH: %s" % (i + 1, len(episodes), label, ",".join(mismatch)))
                    print("  EXIT: cannot re-run with different contract. Remove .done manually if intentional.")
                    sys.exit(1)
            else:
                print("[%d/%d] %s RESUME_FAIL: .done exists but no episode_manifest.json" % (i + 1, len(episodes), label))
                sys.exit(1)
        if skip:
            continue

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

        # Atomic save: write .tmp, os.replace, .done last
        ep_manifest = dict(run_attrs, label=label, task_idx=ti, init_idx=ii,
                          steps=res["steps"], success=res["success"],
                          binding_ok=res.get("binding_ok", False))

        result_tmp = os.path.join(ep_dir, "result.json.tmp")
        trace_tmp = os.path.join(ep_dir, "trace.csv.tmp")
        manifest_tmp = os.path.join(ep_dir, "episode_manifest.json.tmp")

        json.dump(ep_manifest, open(manifest_tmp, "w"), indent=2)
        json.dump(res, open(result_tmp, "w"), indent=2)

        fieldnames = ["episode_key", "profile", "task_idx", "init_idx",
                      "init_state_sha", "step_idx",
                      "object_pose_x", "object_pose_y", "object_pose_z",
                      "target_pose_x", "target_pose_y", "target_pose_z",
                      "object_to_target_distance", "object_eef_distance",
                      "teacher_privileged_state_available",
                      "reward", "done", "check_success", "termination",
                      "feature_valid", "feature_error",
                      "raw_gripper", "env_gripper", "qpos_scalar", "opening_proxy",
                      "binding_ok", "object_key", "target_key",
                      "raw_action_json", "raw_action_sha256",
                      "env_action_json", "env_action_sha256",
                      "processed_image_sha256", "processor_pixel_sha256",
                      "input_ids_sha256"] + FEATURE_NAMES
        with open(trace_tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        os.replace(manifest_tmp, manifest_file)
        os.replace(result_tmp, os.path.join(ep_dir, "result.json"))
        os.replace(trace_tmp, os.path.join(ep_dir, "trace.csv"))
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
