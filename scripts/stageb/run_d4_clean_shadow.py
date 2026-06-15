#!/usr/bin/env python3
"""D4.3: Single-episode clean-shadow / clean-reference runner.

Read-only ProductionStreamingDetector attached to the exact existing
OpenVLA/LIBERO clean runner. No action modification. No attack. No perturbation.

Two modes:
  --mode shadow   Detector enabled (read-only), records all detector outputs
  --mode reference Detector completely disabled, records identical schema with
                  detector fields set to DISABLED

Per-step:
  1. Obtain clean OpenVLA model action.
  2. Hash raw action bytes (pre).
  3. Derive detector scalar inputs from COPIES (shadow mode only).
  4. Call ProductionStreamingDetector (shadow mode only).
  5. Re-hash original action (must match pre; if not → ACTION_IDENTITY_FAIL, ABORT).
  6. Postprocess action as normal.
  7. Step environment.
  8. Record all metadata.

Fail-closed: validity computed from real field checks, NOT hardcoded True.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Pipeline root — override via env for deployment
PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features
from run_l12_e4c2b_repair import sha256_file

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
FROZEN_TAU = 0.236312

ALL_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice",
]

TASK_IDX = {
    "ketchup": 4, "tomato_sauce": 5, "milk": 7, "butter": 6,
    "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "alphabet_soup": 0, "orange_juice": 9, "chocolate_pudding": 8,
}

TARGET_OBJECT_GUESS = {
    "ketchup": "ketchup_green_bottle_1",
    "tomato_sauce": "tomato_sauce_bottle_1", "milk": "milk_carton_1",
    "butter": "butter_box_1", "cream_cheese": "cream_cheese_box_1",
    "salad_dressing": "salad_dressing_bottle_1", "bbq_sauce": "bbq_sauce_bottle_1",
    "alphabet_soup": "alphabet_soup_can_1", "orange_juice": "orange_juice_carton_1",
    "chocolate_pudding": "chocolate_pudding_box_1",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr) -> str:
    return sha256_bytes(np.asarray(arr, dtype=np.float32).tobytes())


# ── Live validity checking ──

def _is_finite_float(val) -> bool:
    if val is None: return False
    if isinstance(val, bool): return False
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return False
        return True
    return False


def _check_live_validity(raw_action, env_action, qpos_val, eef_before):
    """Compute validity flags from real observation + action data.

    Returns dict of validity booleans and diagnostic strings.
    """
    raw_gripper = float(raw_action[-1])
    env_gripper = float(env_action[-1])

    raw_valid = _is_finite_float(raw_gripper) and 0.0 <= raw_gripper <= 1.0
    env_valid = _is_finite_float(env_gripper)
    qpos_valid = _is_finite_float(qpos_val)

    eef_ok = True
    if eef_before is None:
        eef_ok = False
    else:
        for i, v in enumerate(eef_before):
            if not _is_finite_float(v):
                eef_ok = False

    # Convention check: raw > 0.5 <=> env < -0.5 (OPEN)
    #                   raw < 0.5 <=> env > +0.5 (CLOSE)
    convention_ok = True
    decoded_open = 0
    if raw_valid and env_valid:
        if raw_gripper > 0.5 and env_gripper < -0.5:
            decoded_open = 1
        elif raw_gripper < 0.5 and env_gripper > 0.5:
            decoded_open = 0
        else:
            # Convention violation: raw and env don't agree
            convention_ok = False
    elif raw_valid:
        # Only raw available — derive from raw
        decoded_open = 1 if raw_gripper > 0.5 else 0
    elif env_valid:
        # Only env available — derive from env
        decoded_open = 1 if env_gripper < -0.5 else 0

    # decoded_open must be 0 or 1
    decoded_open_ok = decoded_open in (0, 1)

    # Gripper semantics valid if we have consistent raw+env or at least one
    semantics_ok = raw_valid or env_valid

    return {
        "raw_valid": raw_valid,
        "env_valid": env_valid,
        "qpos_valid": qpos_valid,
        "eef_valid": eef_ok,
        "convention_ok": convention_ok,
        "decoded_open": decoded_open,
        "decoded_open_ok": decoded_open_ok,
        "semantics_ok": semantics_ok,
        "all_valid": (raw_valid and env_valid and qpos_valid and eef_ok
                      and convention_ok and decoded_open_ok),
    }


# ── Episode runner ──

def run_episode(args, task, state_id, detector, model, processor, device_ov,
                model_dtype, unnorm_key, K_trigger, action_dim, attempt_id,
                episode_dir, sentinel_path):
    """Run one clean episode.

    Args:
        detector: ProductionStreamingDetector or None (reference mode).
        attempt_id: integer attempt number (1-indexed).
        episode_dir: Path to write per-episode artifacts.
        sentinel_path: Path to atomic sentinel file.

    Returns:
        dict with episode results, or None on fatal infra failure before
        first model action.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gripper_attack.grasp import eef_pos
    from v4_run_eval_openvla import (
        decode_with_scores, postprocess_openvla_action_for_libero,
        physical_gripper_state,
    )

    is_reference = (detector is None)

    bm = benchmark.get_benchmark_dict()
    task_suite = bm["libero_object"]()
    task_idx = TASK_IDX[task]
    task_obj = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    instruction = task_obj.language
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file,
    )

    max_steps = args.max_steps_override
    num_steps_wait = args.num_steps_wait

    if state_id >= len(init_states):
        return {"fatal": True, "reason": f"state_id {state_id} out of range"}

    # ── Env setup ──
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256, camera_widths=256,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview"],
        control_freq=20,
        render_gpu_device_id=args.render_gpu_device_id,
        horizon=max_steps + num_steps_wait,
    )
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[state_id])

    if num_steps_wait > 0:
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)

    # ── Atomic one-shot sentinel ──
    sentinel_data = (
        f"task={task}|state_id={state_id}|attempt={attempt_id}|"
        f"mode={'reference' if is_reference else 'shadow'}|"
        f"timestamp={datetime.now(timezone.utc).isoformat()}"
    )
    if sentinel_path.exists():
        return {"fatal": True, "reason": f"Sentinel already exists: {sentinel_path}"}
    sentinel_path.write_text(sentinel_data)

    # ── Reset detector (shadow mode only) ──
    if not is_reference:
        detector.reset()

    # ── Episode state ──
    step_trace = []
    detector_candidates = []
    action_identity = []
    latency_records = []

    success_done_any = False
    success_check_any = False
    success_step_primary = -1
    infra_status = "ok"
    detector_exception = False
    action_identity_fail = False

    # Identical-action tracking for reference/shadow comparison
    raw_action_sequence_sha = []
    env_action_sequence_sha = []
    obs_sequence_sha = []

    for step_idx in range(max_steps):
        if "agentview_image" not in obs:
            infra_status = f"missing camera at step {step_idx}"
            break

        img_uint8 = obs["agentview_image"]

        # ── Step 1: Get clean action ──
        t0 = time.perf_counter()
        clean_action, _, _, _ = decode_with_scores(
            model, processor, device_ov,
            img_uint8, instruction, unnorm_key, K_trigger,
            libero_official_preprocess=False,
            libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        t_model = time.perf_counter() - t0

        clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)

        # ── Step 2: Hash raw action (pre-detector) ──
        raw_action_hash_pre = sha256_array(clean_action)
        env_action_hash_pre_postprocess = sha256_array(clean_env_action)

        # ── Live validity check ──
        gripper_phys = physical_gripper_state(env, obs)
        qpos_val = float(np.sum(gripper_phys.get("qpos", [0.0])))
        eef_before = eef_pos(env)

        v = _check_live_validity(clean_action, clean_env_action, qpos_val, eef_before)
        raw_gripper_val = float(clean_action[-1])
        env_gripper_val = float(clean_env_action[-1])
        eef_x = float(eef_before[0]) if eef_before is not None else 0.0
        eef_y = float(eef_before[1]) if eef_before is not None else 0.0
        eef_z = float(eef_before[2]) if eef_before is not None else 0.0

        # ── Step 3-4: Detector (shadow mode only) ──
        det_result = None
        t_det = 0.0
        if not is_reference:
            t_det_start = time.perf_counter()
            try:
                det_result = detector.update(
                    step_idx, raw_gripper_val, env_gripper_val, qpos_val,
                    eef_x, eef_y, eef_z, v["decoded_open"],
                    raw_valid=v["raw_valid"], env_valid=v["env_valid"],
                    qpos_valid=v["qpos_valid"], eef_valid=v["eef_valid"],
                    gripper_semantics_valid=v["semantics_ok"],
                )
            except Exception as e:
                detector_exception = True
                infra_status = f"detector_exception: {str(e)[:120]}"
            t_det = time.perf_counter() - t_det_start

        # ── Step 5: Re-hash raw action (post-detector) ──
        raw_action_hash_post = sha256_array(clean_action)

        # ── Step 5b: HARD GATE — action identity ──
        if raw_action_hash_pre != raw_action_hash_post:
            action_identity_fail = True
            infra_status = "ACTION_IDENTITY_FAIL"
            # Record but DO NOT execute
            action_identity.append({
                "step": step_idx,
                "action_hash_pre": raw_action_hash_pre,
                "action_hash_post": raw_action_hash_post,
                "action_identical": 0,
                "env_action_hash": env_action_hash_pre_postprocess,
            })
            break  # Abort episode immediately

        # ── Step 6-7: Execute unchanged action ──
        t_env_start = time.perf_counter()
        obs, reward, done, info = env.step(clean_env_action)
        t_env = time.perf_counter() - t_env_start

        # Observation hash
        obs_hash = sha256_bytes(obs["agentview_image"].tobytes()) if "agentview_image" in obs else ""

        # Sequence hashes for reference/shadow comparison
        raw_action_sequence_sha.append(raw_action_hash_pre)
        env_action_sequence_sha.append(env_action_hash_pre_postprocess)
        obs_sequence_sha.append(obs_hash)

        # Qpos after
        gripper_phys_after = physical_gripper_state(env, obs)
        qpos_after = float(np.sum(gripper_phys_after.get("qpos", [0.0])))

        # Success tracking
        success_check = bool(env.check_success())
        success_done = bool(done)
        success_primary = success_done if args.success_metric == "done" else success_check
        if success_done and not success_done_any:
            success_done_any = True
        if success_check and not success_check_any:
            success_check_any = True
        if success_primary and success_step_primary < 0:
            success_step_primary = step_idx

        # ── Record step trace ──
        trace_row = {
            "step": step_idx, "task": task, "state_id": state_id,
            "raw_gripper": round(raw_gripper_val, 6),
            "env_gripper": round(env_gripper_val, 6),
            "gripper_qpos_before": round(qpos_val, 8),
            "gripper_qpos_after": round(qpos_after, 8),
            "eef_x": round(eef_x, 6), "eef_y": round(eef_y, 6),
            "eef_z": round(eef_z, 6),
            "decoded_open": v["decoded_open"],
            "raw_valid": int(v["raw_valid"]), "env_valid": int(v["env_valid"]),
            "qpos_valid": int(v["qpos_valid"]), "eef_valid": int(v["eef_valid"]),
            "convention_ok": int(v["convention_ok"]),
            "success_done": int(success_done),
            "success_check": int(success_check),
        }
        step_trace.append(trace_row)

        # ── Record action identity ──
        action_identity.append({
            "step": step_idx,
            "action_hash_pre": raw_action_hash_pre,
            "action_hash_post": raw_action_hash_post,
            "action_identical": 1,
            "env_action_hash": env_action_hash_pre_postprocess,
            "obs_hash": obs_hash,
        })

        # ── Record latency ──
        latency_records.append({
            "step": step_idx,
            "model_inference_us": round(t_model * 1_000_000),
            "detector_update_us": round(t_det * 1_000_000) if not is_reference else "DISABLED",
            "env_step_us": round(t_env * 1_000_000),
        })

        # ── Record detector candidate (shadow mode only) ──
        if det_result is not None:
            cand_row = {
                "step": step_idx, "task": task, "state_id": state_id,
                "score": det_result["score"],
                "abstain": det_result["abstain"],
                "abstained": int(det_result["abstained"]),
            }
            for i, fn in enumerate(FEATURE_NAMES):
                cand_row[f"feat_{fn}"] = det_result["features"].get(fn, "")
                cand_row[f"norm_{fn}"] = det_result["normalized_features"][i]
            detector_candidates.append(cand_row)

        if success_primary or done:
            break

    env.close()
    torch.cuda.empty_cache()

    n_steps = len(step_trace)

    # ── Detector emission (shadow mode only) ──
    if is_reference:
        emit_step = "DISABLED"
        n_cands = "DISABLED"
    else:
        emit_step = detector.emit_step
        n_cands = len(detector.candidate_features)

    # ── Episode manifest ──
    episode_manifest = {
        "task": task,
        "state_id": state_id,
        "mode": "reference" if is_reference else "shadow",
        "attempt_id": attempt_id,
        "n_steps": n_steps,
        "max_steps": max_steps,
        "success_primary": int(success_primary) if success_step_primary >= 0 else 0,
        "success_step_primary": success_step_primary,
        "success_done_any": int(success_done_any),
        "success_check_any": int(success_check_any),
        "timeout": n_steps >= max_steps and not success_primary,
        "infra_status": infra_status,
        "detector_exception": detector_exception,
        "action_identity_fail": action_identity_fail,
        "detector_emit_step": emit_step,
        "detector_n_candidates": n_cands,
        "total_time_sec": round(time.time(), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Sequence hashes for comparison
        "raw_action_sequence_sha256": sha256_bytes(
            "|".join(raw_action_sequence_sha).encode()),
        "env_action_sequence_sha256": sha256_bytes(
            "|".join(env_action_sequence_sha).encode()),
        "obs_sequence_sha256": sha256_bytes(
            "|".join(obs_sequence_sha).encode()),
    }

    # ── Write per-episode artifacts ──
    episode_dir.mkdir(parents=True, exist_ok=False)  # must not already exist

    with open(episode_dir / "episode_manifest.json", "w") as f:
        json.dump(episode_manifest, f, indent=2)

    if step_trace:
        with open(episode_dir / "step_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(step_trace[0].keys()))
            w.writeheader(); w.writerows(step_trace)

    if detector_candidates:
        with open(episode_dir / "detector_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detector_candidates[0].keys()))
            w.writeheader(); w.writerows(detector_candidates)

    with open(episode_dir / "detector_emission.json", "w") as f:
        json.dump({
            "emit_step": emit_step if isinstance(emit_step, int) else -1,
            "n_candidates": n_cands if isinstance(n_cands, int) else 0,
            "detector_enabled": not is_reference,
            "candidate_steps": ([cf[0] for cf in detector.candidate_features]
                                if not is_reference else []),
        }, f, indent=2)

    if action_identity:
        with open(episode_dir / "action_identity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(action_identity[0].keys()))
            w.writeheader(); w.writerows(action_identity)

    if latency_records:
        with open(episode_dir / "latency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(latency_records[0].keys()))
            w.writeheader(); w.writerows(latency_records)

    # Provenance
    git_head = os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown"
    with open(episode_dir / "provenance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["git_HEAD", git_head])
        w.writerow(["checkpoint_sha", FROZEN_CHECKPOINT_SHA])
        w.writerow(["threshold", str(FROZEN_TAU)])
        w.writerow(["runner_sha", sha256_file(__file__)])
        w.writerow(["task", task])
        w.writerow(["state_id", str(state_id)])
        w.writerow(["mode", "reference" if is_reference else "shadow"])
        w.writerow(["attempt_id", str(attempt_id)])
        w.writerow(["timestamp", datetime.now(timezone.utc).isoformat()])
        w.writerow(["sentinel_sha", sha256_file(str(sentinel_path))])

    # Artifact hashes
    artifact_names = [
        "episode_manifest.json", "step_trace.csv", "detector_candidates.csv",
        "detector_emission.json", "action_identity.csv", "latency.csv",
        "provenance.csv",
    ]
    with open(episode_dir / "artifact_hashes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        for an in artifact_names:
            ap = episode_dir / an
            if ap.exists():
                w.writerow([an, sha256_file(str(ap))])

    # Placeholder teacher_sidecar (PENDING — not yet implemented for live mode)
    with open(episode_dir / "teacher_sidecar.json", "w") as f:
        json.dump({
            "status": "PENDING_SIDECAR",
            "note": "Teacher-P sidecar requires post-hoc RC1a remap; not yet live",
            "task": task, "state_id": state_id,
        }, f, indent=2)

    return episode_manifest


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--state-id", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["reference", "shadow"], default="shadow")
    ap.add_argument("--attempt-id", type=int, default=1,
                    help="Attempt number (1-based)")
    ap.add_argument("--model-path",
                    default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--render-gpu-device-id", type=int, default=0)
    ap.add_argument("--model-gpu-device-id", type=int, default=-1)
    ap.add_argument("--max-steps-override", type=int, default=280)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--success-metric", choices=["done", "check_success"],
                    default="check_success")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    is_reference = (args.mode == "reference")

    # ── Output directory: must be provided by orchestrator; we create episode_dir ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    safe_tag = f"{args.task}_s{args.state_id}_{args.mode}_attempt{args.attempt_id}"
    episode_dir = out / safe_tag
    sentinel_path = episode_dir / "SENTINEL.txt"

    # ── Verify checkpoint ──
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA, (
        f"Checkpoint SHA mismatch: got {actual_ckpt[:16]}..."
    )
    print(f"Checkpoint: {actual_ckpt[:16]}... VERIFIED")

    # ── Load detector (shadow mode only) ──
    detector = None
    if not is_reference:
        device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        means = ckpt["normalization"]["means"]
        stdevs = ckpt["normalization"]["stdevs"]
        impute = ckpt["normalization"]["impute"]

        detector_model = CandidateRanker(n_features=16).to(device)
        detector_model.load_state_dict(ckpt["model_state"])
        detector_model.eval()

        from gripper_attack.production_detector import ProductionStreamingDetector
        detector = ProductionStreamingDetector(
            detector_model, means, stdevs, impute,
            threshold=FROZEN_TAU, device=str(device),
        )
        print(f"Detector loaded (threshold={FROZEN_TAU})")
    else:
        print("Detector DISABLED (reference mode)")

    # ── Load OpenVLA model ──
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OPENVLA_RENDER_LOCAL_DEVICE", str(args.render_gpu_device_id))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading OpenVLA from {args.model_path}")

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True, use_fast=True,
    )
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(args.model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {
            "device_map": {"": int(args.model_gpu_device_id)},
            "max_memory": {int(args.model_gpu_device_id): mm, "cpu": "128GiB"},
        }
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw,
    )
    device_ov = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v_device in model.hf_device_map.values():
            if isinstance(v_device, str) and v_device.startswith("cuda"):
                device_ov = v_device; break
            if isinstance(v_device, int):
                device_ov = f"cuda:{v_device}"; break
    model_dtype = next(model.parameters()).dtype
    print(f"OpenVLA loaded on {device_ov} dtype={model_dtype}")

    unnorm_key = "libero_object"
    K_trigger = 8
    action_dim = int(model.get_action_dim(unnorm_key))

    # ── Run episode ──
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {safe_tag}")
    result = run_episode(
        args, args.task, args.state_id, detector,
        model, processor, device_ov, model_dtype,
        unnorm_key, K_trigger, action_dim,
        args.attempt_id, episode_dir, sentinel_path,
    )

    if result is None:
        print("FATAL: episode returned None")
        sys.exit(1)

    if result.get("fatal"):
        print(f"FATAL: {result.get('reason', 'unknown')}")
        sys.exit(1)

    print(f"  steps={result['n_steps']} success={result['success_primary']} "
          f"mode={result['mode']} emit={result['detector_emit_step']} "
          f"identity_fail={result['action_identity_fail']} "
          f"infra={result['infra_status']}")
    print(f"Output: {episode_dir}")


if __name__ == "__main__":
    main()
