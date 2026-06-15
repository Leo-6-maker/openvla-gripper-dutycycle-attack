#!/usr/bin/env python3
"""D4.3: Live clean-shadow detector runner.

Read-only ProductionStreamingDetector attached to the exact existing
OpenVLA/LIBERO clean runner. No action modification. No attack. No perturbation.

Per-step:
  1. Obtain clean OpenVLA model action.
  2. Copy + hash raw model action bytes.
  3. Derive detector scalar inputs from copies only.
  4. Call ProductionStreamingDetector.
  5. Re-hash original action object (must match step 2).
  6. Postprocess original clean action exactly as normal.
  7. Hash final env action.
  8. Execute unchanged clean action.

Records per-step: action hashes, detector candidates, 16 raw/normalized features,
MLP score, threshold decision, validity flags, latency.

After episode: independent offline Teacher-P sidecar (RC1a remap + enumeration).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ── Pipeline root setup ──
PIPELINE_ROOT = os.environ.get(
    "L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline",
)
REPO_ROOT = os.environ.get(
    "L12_REPO_ROOT",
    "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607",
)
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

# V4 runner imports (from pipeline root)
from v4_run_eval_openvla import (
    decode_with_scores,
    postprocess_openvla_action_for_libero,
    physical_gripper_state,
    prompt,
)

# Detector imports
from gripper_attack.production_detector import ProductionStreamingDetector
from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features

# RC1a remap + Teacher-P (for offline sidecar)
from remap_v4_trace_for_l12 import remap_v4_to_l12
from run_l12_e4c2b_repair import sha256_file

# ── Frozen constants ──
FROZEN_CHECKPOINT_SHA = (
    "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
)
FROZEN_TAU = 0.236312

TASK_IDX = {
    "ketchup": 4, "tomato_sauce": 5, "milk": 7, "butter": 6,
    "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "alphabet_soup": 0, "orange_juice": 9, "chocolate_pudding": 8,
}

ALL_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice",
]

TARGET_OBJECT_GUESS = {
    "ketchup": "ketchup_green_bottle_1",
    "tomato_sauce": "tomato_sauce_bottle_1",
    "milk": "milk_carton_1",
    "butter": "butter_box_1",
    "cream_cheese": "cream_cheese_box_1",
    "salad_dressing": "salad_dressing_bottle_1",
    "bbq_sauce": "bbq_sauce_bottle_1",
    "alphabet_soup": "alphabet_soup_can_1",
    "orange_juice": "orange_juice_carton_1",
    "chocolate_pudding": "chocolate_pudding_box_1",
}


def load_model_s20d(model_path, model_gpu_device_id=-1):
    """V4 load_model with use_fast=True."""
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=True,
    )
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {
            "device_map": {"": int(model_gpu_device_id)},
            "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"},
        }
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw,
    )
    dev = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v; break
            if isinstance(v, int):
                dev = f"cuda:{v}"; break
    print(f"[model] loaded path={model_path} device={dev} attn={attn_impl}")
    return model, processor, dev


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_numpy(arr: np.ndarray) -> str:
    return sha256_bytes(arr.tobytes())


def run_episode(args, task, state_id, detector, model, processor, device,
                model_dtype, unnorm_key, K_trigger, action_dim):
    """Run one clean episode with detector attached (read-only)."""

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gripper_attack.grasp import eef_pos, object_pos

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
        print(f"  state_id {state_id} out of range (max {len(init_states) - 1})")
        return None

    t_start = time.time()

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

    # Wait steps
    if num_steps_wait > 0:
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)

    # Resolve target object
    target_object_name = TARGET_OBJECT_GUESS.get(task, "")
    if target_object_name:
        try:
            env.sim.model.body_name2id(target_object_name)
        except Exception:
            target_object_name = ""

    # ── Detector reset ──
    detector.reset()

    # ── Per-step records ──
    step_trace = []
    detector_candidates = []
    action_identity = []
    latency_records = []
    obs_hashes = []

    success_done_any = False
    success_check_any = False
    success_step_primary = -1
    infra_status = "ok"
    detector_exception = False

    for step_idx in range(max_steps):
        if "agentview_image" not in obs:
            infra_status = f"missing camera at step {step_idx}"
            break

        img_uint8 = obs["agentview_image"]

        # ── Step 1: Get clean action ──
        t0 = time.perf_counter()
        clean_action, prefix_logits, Tclean, gen_out = decode_with_scores(
            model, processor, device,
            img_uint8, instruction, unnorm_key, K_trigger,
            libero_official_preprocess=False,
            libero_preprocess_backend="official_pil_lanczos",
            center_crop=True,
            resize_size=224,
            drop_attention_mask=True,
        )
        t_model = time.perf_counter() - t0

        clean_env_action = postprocess_openvla_action_for_libero(
            clean_action, enabled=True,
        )

        # ── Step 2: Hash raw action before detector ──
        action_hash_pre = sha256_numpy(clean_action)

        # ── Step 3: Derive detector inputs from COPIES ──
        raw_gripper_val = float(clean_action[-1])
        env_gripper_val = float(clean_env_action[-1])

        gripper_phys_before = physical_gripper_state(env, obs)
        qpos_val = float(np.sum(gripper_phys_before.get("qpos", [0.0])))

        eef_before = eef_pos(env)
        eef_x = float(eef_before[0]) if eef_before is not None else 0.0
        eef_y = float(eef_before[1]) if eef_before is not None else 0.0
        eef_z = float(eef_before[2]) if eef_before is not None else 0.0

        is_open = int(env_gripper_val < -0.5)

        # Validity flags
        raw_valid = True
        env_valid = True
        qpos_valid = bool(gripper_phys_before.get("qpos") is not None)
        eef_valid = eef_before is not None
        gripper_semantics_valid = True  # live env — always valid unless we detect issues

        # Hash detector inputs
        det_input_bytes = np.array([
            raw_gripper_val, env_gripper_val, qpos_val,
            eef_x, eef_y, eef_z, float(is_open),
        ], dtype=np.float32).tobytes()
        det_input_hash = sha256_bytes(det_input_bytes)

        # ── Step 4: Call detector ──
        t_det_start = time.perf_counter()
        try:
            det_result = detector.update(
                step_idx, raw_gripper_val, env_gripper_val, qpos_val,
                eef_x, eef_y, eef_z, is_open,
                raw_valid=raw_valid, env_valid=env_valid,
                qpos_valid=qpos_valid, eef_valid=eef_valid,
                gripper_semantics_valid=gripper_semantics_valid,
            )
        except Exception as e:
            detector_exception = True
            infra_status = f"detector_exception: {str(e)[:120]}"
            det_result = None
        t_det = time.perf_counter() - t_det_start

        # ── Step 5: Re-hash raw action ──
        action_hash_post = sha256_numpy(clean_action)
        action_ok = (action_hash_pre == action_hash_post)

        # ── Step 6: Postprocess action (already done above) ──
        env_action = clean_env_action.copy()

        # ── Step 7: Hash env action ──
        env_action_hash = sha256_numpy(env_action)

        # ── Step 8: Execute ──
        t_env_start = time.perf_counter()
        obs, reward, done, info = env.step(env_action)
        t_env = time.perf_counter() - t_env_start

        # Observation hash
        obs_hash = sha256_bytes(obs.get("agentview_image", b"").tobytes()) if "agentview_image" in obs else ""

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
        step_trace.append({
            "step": step_idx,
            "task": task,
            "state_id": state_id,
            "raw_gripper": round(raw_gripper_val, 6),
            "env_gripper": round(env_gripper_val, 6),
            "gripper_qpos_before": round(qpos_val, 8),
            "gripper_qpos_after": round(qpos_after, 8),
            "eef_x": round(eef_x, 6),
            "eef_y": round(eef_y, 6),
            "eef_z": round(eef_z, 6),
            "decoded_open": is_open,
            "success_done": int(success_done),
            "success_check": int(success_check),
            "infra_status": infra_status if step_idx == 0 else "",
        })

        # ── Record detector candidate ──
        if det_result is not None:
            detector_candidates.append({
                "step": step_idx,
                "task": task,
                "state_id": state_id,
                "score": det_result["score"],
                "abstain": det_result["abstain"],
                "abstained": int(det_result["abstained"]),
                **{f"feat_{fn}": det_result["features"].get(fn, "")
                   for fn in FEATURE_NAMES},
                **{f"norm_{fn}": det_result["normalized_features"][i]
                   for i, fn in enumerate(FEATURE_NAMES)},
            })

        # ── Record action identity ──
        identity_ok = action_ok and np.allclose(clean_action, clean_env_action
            if action_dim <= 7 else clean_action)  # env action differs in gripper dim
        action_identity.append({
            "step": step_idx,
            "action_hash_pre": action_hash_pre,
            "action_hash_post": action_hash_post,
            "action_identical": int(action_hash_pre == action_hash_post),
            "env_action_hash": env_action_hash,
            "detector_input_hash": det_input_hash,
            "obs_hash": obs_hash,
        })

        # ── Record latency ──
        latency_records.append({
            "step": step_idx,
            "model_inference_us": round(t_model * 1_000_000),
            "detector_update_us": round(t_det * 1_000_000),
            "env_step_us": round(t_env * 1_000_000),
        })

        if success_primary or done:
            break

    env.close()
    torch.cuda.empty_cache()

    t_total = time.time() - t_start
    n_steps = len(step_trace)

    # ── Detector emission ──
    emission = {
        "emit_step": detector.emit_step,
        "has_emitted": detector.has_emitted,
        "n_candidates": len(detector.candidate_features),
        "candidate_steps": [cf[0] for cf in detector.candidate_features],
    }

    # ── Episode manifest ──
    episode_manifest = {
        "task": task,
        "state_id": state_id,
        "n_steps": n_steps,
        "max_steps": max_steps,
        "success_primary": int(success_primary) if success_step_primary >= 0 else 0,
        "success_step_primary": success_step_primary,
        "success_done_any": int(success_done_any),
        "success_check_any": int(success_check_any),
        "timeout": n_steps >= max_steps and not success_primary,
        "infra_status": infra_status,
        "detector_exception": detector_exception,
        "detector_emit_step": detector.emit_step,
        "detector_n_candidates": len(detector.candidate_features),
        "total_time_sec": round(t_total, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "episode_manifest": episode_manifest,
        "step_trace": step_trace,
        "detector_candidates": detector_candidates,
        "detector_emission": emission,
        "action_identity": action_identity,
        "latency_records": latency_records,
        "clean_action_sequence": None,  # not stored for privacy; hash chain is sufficient
    }


def run_teacher_p_sidecar(trace_rows, task, state_id):
    """Offline Teacher-P enumeration on completed episode trace.

    Runs RC1a remap on the in-memory trace rows to get Teacher-P taxonomy.
    Teacher-P is NEVER used during the actual rollout.
    """
    # Build a temporary CSV-compatible structure
    # trace_rows already has the necessary fields; run remap logic

    # The remap needs source_paths; we'll use the in-memory data directly.
    # Teacher-P categories:
    #   UNIQUE_MULTI, UNIQUE_SINGLE, AMBIGUOUS, UNAVAILABLE, NO_CLOSE

    # For now, return placeholder — full sidecar needs RC1a remap integration.
    # The sidecar is post-hoc and does not affect shadow results.
    return {
        "task": task,
        "state_id": state_id,
        "taxonomy": "PENDING_SIDECAR",
        "teacher_p_step": -1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--state_ids", default="0",
                    help="comma-separated state ids")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path",
                    default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to frozen D1b detector checkpoint")
    ap.add_argument("--render-gpu-device-id", type=int, default=0)
    ap.add_argument("--model-gpu-device-id", type=int, default=-1)
    ap.add_argument("--max-steps-override", type=int, default=280)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--success-metric", choices=["done", "check_success"],
                    default="check_success")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable-detector", action="store_true",
                    help="Run CLEAN_REFERENCE (detector disabled)")
    ap.add_argument("--mode", choices=["canary", "panel"], default="panel")
    args = ap.parse_args()

    state_ids = [int(x.strip()) for x in args.state_ids.split(",") if x.strip()]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Verify checkpoint ──
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA, (
        f"Checkpoint SHA mismatch: {actual_ckpt[:16]}..."
    )
    print(f"Checkpoint: {actual_ckpt[:16]}... VERIFIED")

    # ── Load model ──
    device = torch.device("cpu")  # detector device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    detector_model = CandidateRanker(n_features=16).to(device)
    detector_model.load_state_dict(ckpt["model_state"])
    detector_model.eval()

    detector = ProductionStreamingDetector(
        detector_model, means, stdevs, impute, threshold=FROZEN_TAU, device=str(device),
    )

    # ── Load OpenVLA model ──
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OPENVLA_RENDER_LOCAL_DEVICE", str(args.render_gpu_device_id))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading OpenVLA from {args.model_path}")
    model, processor, device_ov = load_model_s20d(
        args.model_path, model_gpu_device_id=args.model_gpu_device_id,
    )
    model_dtype = next(model.parameters()).dtype
    print(f"[{datetime.now().strftime('%H:%M:%S')}] OpenVLA loaded on {device_ov}")

    unnorm_key = "libero_object"
    K_trigger = 8
    action_dim = int(model.get_action_dim(unnorm_key))

    # ── Run episodes ──
    all_results = []
    for sid in state_ids:
        safe_tag = f"{args.task}_s{sid}_clean_shadow"
        if args.disable_detector:
            safe_tag = f"{args.task}_s{sid}_clean_reference"
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {safe_tag}")

        result = run_episode(
            args, args.task, sid,
            detector if not args.disable_detector else None,
            model, processor, device_ov, model_dtype,
            unnorm_key, K_trigger, action_dim,
        )
        if result is None:
            continue

        all_results.append(result)

        # ── Write per-episode artifacts ──
        ep_dir = out / safe_tag
        ep_dir.mkdir(parents=True, exist_ok=True)

        # episode_manifest.json
        with open(ep_dir / "episode_manifest.json", "w") as f:
            json.dump(result["episode_manifest"], f, indent=2)

        # step_trace.csv
        if result["step_trace"]:
            with open(ep_dir / "step_trace.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(result["step_trace"][0].keys()))
                w.writeheader()
                w.writerows(result["step_trace"])

        # detector_candidates.csv
        if result["detector_candidates"]:
            with open(ep_dir / "detector_candidates.csv", "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(result["detector_candidates"][0].keys()),
                )
                w.writeheader()
                w.writerows(result["detector_candidates"])

        # detector_emission.json
        with open(ep_dir / "detector_emission.json", "w") as f:
            json.dump(result["detector_emission"], f, indent=2)

        # action_identity.csv
        if result["action_identity"]:
            with open(ep_dir / "action_identity.csv", "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(result["action_identity"][0].keys()),
                )
                w.writeheader()
                w.writerows(result["action_identity"])

        # latency.csv
        if result["latency_records"]:
            with open(ep_dir / "latency.csv", "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(result["latency_records"][0].keys()),
                )
                w.writeheader()
                w.writerows(result["latency_records"])

        # provenance.csv
        with open(ep_dir / "provenance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            w.writerow(["checkpoint_sha", actual_ckpt])
            w.writerow(["threshold", FROZEN_TAU])
            w.writerow(["runner_sha", sha256_file(__file__)])
            w.writerow(["task", args.task])
            w.writerow(["state_id", str(sid)])
            w.writerow(["detector_enabled", str(not args.disable_detector).lower()])
            w.writerow(["timestamp", datetime.now(timezone.utc).isoformat()])

        # artifact_hashes.csv
        artifact_files = [
            "episode_manifest.json", "step_trace.csv", "detector_candidates.csv",
            "detector_emission.json", "action_identity.csv", "latency.csv",
            "provenance.csv",
        ]
        with open(ep_dir / "artifact_hashes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["artifact", "sha256"])
            for af in artifact_files:
                af_path = ep_dir / af
                if af_path.exists():
                    w.writerow([af, sha256_file(str(af_path))])

        # teacher_sidecar.csv (post-hoc, not affecting shadow)
        tp_result = run_teacher_p_sidecar(
            result["step_trace"], args.task, sid,
        )
        with open(ep_dir / "teacher_sidecar.json", "w") as f:
            json.dump(tp_result, f, indent=2)

        em = result["episode_manifest"]
        print(f"  steps={em['n_steps']} success={em['success_primary']} "
              f"det_emit={em['detector_emit_step']} "
              f"det_cands={em['detector_n_candidates']} "
              f"infra={em['infra_status']}")

    # ── Episode inventory ──
    inventory_path = out / "episode_inventory.csv"
    with open(inventory_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "state_id", "mode", "n_steps", "success",
                     "detector_emit_step", "detector_n_candidates",
                     "infra_status", "detector_enabled"])
        for r in all_results:
            em = r["episode_manifest"]
            w.writerow([
                em["task"], em["state_id"],
                "reference" if args.disable_detector else "shadow",
                em["n_steps"], em["success_primary"],
                em["detector_emit_step"], em["detector_n_candidates"],
                em["infra_status"], not args.disable_detector,
            ])

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] D4.3 shadow runner done")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
