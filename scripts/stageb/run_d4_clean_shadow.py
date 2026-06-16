#!/usr/bin/env python3
"""D4.3: Single-episode clean-shadow / clean-reference runner.

Read-only ProductionStreamingDetector. No action modification.

Atomic phase markers (written to attempt directory):
  ATTEMPT_STARTED       — attempt directory created
  MODEL_LOADED          — OpenVLA model + detector loaded
  FIRST_ACTION_GENERATED — first model action produced (retry boundary)

Retry: only allowed if FIRST_ACTION_GENERATED does not exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr) -> str:
    return sha256_bytes(np.asarray(arr, dtype=np.float32).tobytes())


def write_phase_marker(episode_dir: Path, marker_name: str, extra: dict = None):
    """Write an atomic phase marker file."""
    data = {
        "marker": marker_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    marker_path = episode_dir / f"{marker_name}.json"
    tmp = str(marker_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(marker_path))  # atomic on POSIX


def phase_marker_exists(episode_dir: Path, marker_name: str) -> bool:
    return (episode_dir / f"{marker_name}.json").exists()


# ── Live validity ──

def _isfinite(val) -> bool:
    if val is None: return False
    if isinstance(val, bool): return False
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return True
    return False


def _check_live_validity(raw_action, env_action, qpos_raw, eef_before):
    """Compute validity flags from real observation + action data.

    Fail-closed: any missing/invalid field → invalid. No silent defaults.
    """
    raw_gripper = float(raw_action[-1])
    env_gripper = float(env_action[-1])

    raw_valid = _isfinite(raw_gripper) and 0.0 <= raw_gripper <= 1.0
    env_valid = _isfinite(env_gripper)

    # Qpos: validate raw array before summing
    if qpos_raw is not None and hasattr(qpos_raw, '__len__') and len(qpos_raw) > 0:
        qpos_valid = all(_isfinite(float(v)) for v in qpos_raw)
        qpos_val = float(np.sum(qpos_raw)) if qpos_valid else float("nan")
    else:
        qpos_valid = False
        qpos_val = float("nan")

    # EEF
    eef_valid = False
    eef_x = eef_y = eef_z = float("nan")
    if eef_before is not None and hasattr(eef_before, '__len__') and len(eef_before) >= 3:
        if all(_isfinite(float(v)) for v in eef_before[:3]):
            eef_valid = True
            eef_x = float(eef_before[0])
            eef_y = float(eef_before[1])
            eef_z = float(eef_before[2])

    # Convention: raw > 0.5 ⇔ env < -0.5; raw < 0.5 ⇔ env > 0.5
    convention_ok = False
    decoded_open = -1
    if raw_valid and env_valid:
        if raw_gripper > 0.5 and env_gripper < -0.5:
            convention_ok = True; decoded_open = 1
        elif raw_gripper < 0.5 and env_gripper > 0.5:
            convention_ok = True; decoded_open = 0

    decoded_open_ok = (decoded_open in (0, 1))

    # Gripper semantics: ALL of raw, env, convention, decoded_open must be valid
    semantics_ok = (
        raw_valid and env_valid and convention_ok and decoded_open_ok
    )

    return {
        "raw_valid": raw_valid, "env_valid": env_valid,
        "qpos_valid": qpos_valid, "eef_valid": eef_valid,
        "convention_ok": convention_ok,
        "decoded_open": decoded_open if decoded_open_ok else -1,
        "decoded_open_ok": decoded_open_ok,
        "semantics_ok": semantics_ok,
        "raw_gripper": raw_gripper, "env_gripper": env_gripper,
        "qpos_val": qpos_val, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        "any_invalid": not (raw_valid and env_valid and qpos_valid and eef_valid
                            and convention_ok and decoded_open_ok),
    }


# ── Episode runner ──

def run_episode(args, task, state_id, detector, model, processor, device_ov,
                model_dtype, unnorm_key, K_trigger, action_dim, attempt_id,
                episode_dir):
    """Run one clean episode.

    episode_dir MUST already exist (created by caller with ATTEMPT_STARTED marker).

    Returns dict with episode results.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gripper_attack.grasp import eef_pos, object_pos
    from v4_run_eval_openvla import (
        decode_with_scores, postprocess_openvla_action_for_libero,
        physical_gripper_state,
    )

    # ── Privileged sidecar: object mapping ──
    OBJECT_MAP = {
        "alphabet_soup": "alphabet_soup_1_main",
        "cream_cheese": "cream_cheese_1_main",
        "salad_dressing": "salad_dressing_1_main",
        "bbq_sauce": "bbq_sauce_1_main",
        "ketchup": "ketchup_1_main",
        "tomato_sauce": "tomato_sauce_1_main",
        "butter": "butter_1_main",
        "milk": "milk_1_main",
        "chocolate_pudding": "chocolate_pudding_1_main",
        "orange_juice": "orange_juice_1_main",
    }

    def get_object_pose_safe(env, obj_name):
        try:
            return object_pos(env, obj_name)
        except Exception:
            return None

    def resolve_object_name(env, task_name):
        candidate = OBJECT_MAP.get(task_name, "")
        if candidate:
            try:
                env.sim.model.body_name2id(candidate)
                return candidate, True
            except Exception:
                pass
        return "", False

    is_reference = (detector is None)
    max_steps = args.max_steps_override
    num_steps_wait = args.num_steps_wait

    # ── Env setup ──
    bm = benchmark.get_benchmark_dict()
    task_suite = bm["libero_object"]()
    task_idx = TASK_IDX[task]
    task_obj = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    instruction = task_obj.language
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file,
    )

    if state_id >= len(init_states):
        return {"fatal": True, "reason": f"state_id {state_id} out of range"}

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

    # ── Resolve privileged object (after env init, before episode loop) ──
    target_object_name = ""
    object_lookup_ok = True
    obj_init_x = None; obj_init_y = None; obj_init_z = None
    if args.enable_privileged_sidecar:
        target_object_name, object_lookup_ok = resolve_object_name(env, task)
        if not object_lookup_ok:
            infra_status = f"PRIVILEGED_OBJECT_LOOKUP_FAIL:{task}"
        else:
            obj_init = get_object_pose_safe(env, target_object_name)
            if obj_init is not None:
                obj_init_x = float(obj_init[0])
                obj_init_y = float(obj_init[1])
                obj_init_z = float(obj_init[2])
            else:
                obj_init_x = None; obj_init_y = None; obj_init_z = None

    # ── Reset detector ──
    if not is_reference:
        detector.reset()
        pre_reset_state = {
            "next_expected_step": detector._next_expected_step,
            "history_len": len(detector.history),
            "emit_step": detector.emit_step,
            "candidate_count": len(detector.candidate_features),
        }

    # ── Episode state ──
    step_trace = []
    detector_candidates = []
    action_identity = []
    latency_records = []
    invalid_field_steps = []

    success_done_any = False
    success_check_any = False
    success_step_primary = -1
    done_step_any = -1
    infra_status = "ok"
    detector_exception = False
    action_identity_fail = False

    raw_action_sequence_sha = []
    env_action_sequence_sha = []
    obs_sequence_sha = []

    first_action_generated = False

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

        # ── FIRST_ACTION_GENERATED marker ──
        if not first_action_generated:
            write_phase_marker(episode_dir, "FIRST_ACTION_GENERATED", {
                "step": step_idx, "mode": "reference" if is_reference else "shadow",
                "attempt_id": attempt_id,
            })
            first_action_generated = True

        clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)

        # ── Step 2: Hash raw action ──
        raw_action_hash_pre = sha256_array(clean_action)
        env_action_hash_pre = sha256_array(clean_env_action)

        # ── PRE-ACTION readings (before env.step) ──
        gripper_phys = physical_gripper_state(env, obs)
        qpos_raw = gripper_phys.get("qpos") if gripper_phys else None
        eef_before = eef_pos(env)

        v = _check_live_validity(clean_action, clean_env_action, qpos_raw, eef_before)

        # Privileged sidecar PRE (must be BEFORE env.step)
        priv_pre = {}
        if args.enable_privileged_sidecar:
            priv_valid = object_lookup_ok
            priv_reason = "" if priv_valid else "object_lookup_fail"
            obj_pre = get_object_pose_safe(env, target_object_name) if priv_valid else None
            if obj_pre is None and priv_valid:
                priv_valid = False
                priv_reason = "object_pose_read_fail"
            priv_pre = {
                "eef_pre_x": round(float(eef_before[0]), 6) if eef_before is not None else "",
                "eef_pre_y": round(float(eef_before[1]), 6) if eef_before is not None else "",
                "eef_pre_z": round(float(eef_before[2]), 6) if eef_before is not None else "",
                "obj_pre_x": round(float(obj_pre[0]), 6) if obj_pre is not None else "",
                "obj_pre_y": round(float(obj_pre[1]), 6) if obj_pre is not None else "",
                "obj_pre_z": round(float(obj_pre[2]), 6) if obj_pre is not None else "",
                "eef_to_obj_pre": round(float(np.linalg.norm(
                    np.array(eef_before) - np.array(obj_pre))), 6)
                    if eef_before is not None and obj_pre is not None else "",
                "privileged_valid": int(priv_valid),
                "privileged_failure_reason": priv_reason if not priv_valid else "",
            }

        # ── Step 3-4: Detector ──
        det_result = None
        t_det = 0.0
        if not is_reference:
            t_det_start = time.perf_counter()
            try:
                det_result = detector.update(
                    step_idx,
                    v["raw_gripper"], v["env_gripper"], v["qpos_val"],
                    v["eef_x"], v["eef_y"], v["eef_z"],
                    v["decoded_open"],
                    raw_valid=v["raw_valid"], env_valid=v["env_valid"],
                    qpos_valid=v["qpos_valid"], eef_valid=v["eef_valid"],
                    gripper_semantics_valid=v["semantics_ok"],
                )
            except Exception as e:
                detector_exception = True
                infra_status = f"detector_exception: {str(e)[:120]}"
            t_det = time.perf_counter() - t_det_start

        # ── Step 5: Re-hash ──
        raw_action_hash_post = sha256_array(clean_action)

        # ── HARD GATE: action identity ──
        if raw_action_hash_pre != raw_action_hash_post:
            action_identity_fail = True
            infra_status = "ACTION_IDENTITY_FAIL"
            action_identity.append({
                "step": step_idx,
                "action_hash_pre": raw_action_hash_pre,
                "action_hash_post": raw_action_hash_post,
                "action_identical": 0,
                "env_action_hash": env_action_hash_pre,
            })
            break  # ABORT — do NOT execute

        # ── Step 6-7: Execute ──
        t_env_start = time.perf_counter()
        obs, reward, done, info = env.step(clean_env_action)
        t_env = time.perf_counter() - t_env_start

        obs_hash = sha256_bytes(obs["agentview_image"].tobytes()) if "agentview_image" in obs else ""

        raw_action_sequence_sha.append(raw_action_hash_pre)
        env_action_sequence_sha.append(env_action_hash_pre)
        obs_sequence_sha.append(obs_hash)

        # Qpos after
        gripper_phys_after = physical_gripper_state(env, obs)
        qpos_after_raw = gripper_phys_after.get("qpos") if gripper_phys_after else None
        if qpos_after_raw is not None and hasattr(qpos_after_raw, '__len__') and len(qpos_after_raw) > 0:
            qpos_after = float(np.sum(qpos_after_raw))
        else:
            qpos_after = float("nan")

        # Success
        success_check = bool(env.check_success())
        success_done = bool(done)
        success_primary = success_done if args.success_metric == "done" else success_check
        if success_done and not success_done_any:
            success_done_any = True; done_step_any = step_idx
        if success_check and not success_check_any:
            success_check_any = True
        if success_primary and success_step_primary < 0:
            success_step_primary = step_idx

        # ── Record step trace ──
        trace_row = {
            "step": step_idx, "task": task, "state_id": state_id,
            "raw_gripper": round(v["raw_gripper"], 6),
            "env_gripper": round(v["env_gripper"], 6),
            "gripper_qpos_before": round(v["qpos_val"], 8) if v["qpos_valid"] else "",
            "gripper_qpos_after": round(qpos_after, 8),
            "eef_x": round(v["eef_x"], 6) if v["eef_valid"] else "",
            "eef_y": round(v["eef_y"], 6) if v["eef_valid"] else "",
            "eef_z": round(v["eef_z"], 6) if v["eef_valid"] else "",
            "decoded_open": v["decoded_open"],
            "raw_valid": int(v["raw_valid"]), "env_valid": int(v["env_valid"]),
            "qpos_valid": int(v["qpos_valid"]), "eef_valid": int(v["eef_valid"]),
            "convention_ok": int(v["convention_ok"]),
            "semantics_ok": int(v["semantics_ok"]),
            "success_done": int(success_done),
            "success_check": int(success_check),
        }

        # ── POST-ACTION privileged sidecar (after env.step) ──
        if args.enable_privileged_sidecar:
            # Merge PRE fields from pre-read above
            trace_row.update(priv_pre)
            trace_row["target_object_name"] = target_object_name
            trace_row["obj_init_x"] = round(obj_init_x, 6) if obj_init_x is not None else ""
            trace_row["obj_init_y"] = round(obj_init_y, 6) if obj_init_y is not None else ""
            trace_row["obj_init_z"] = round(obj_init_z, 6) if obj_init_z is not None else ""

            # POST-action readings (after env.step)
            obj_post = get_object_pose_safe(env, target_object_name) if priv_pre.get("privileged_valid", 0) else None
            eef_post = eef_pos(env)
            trace_row["eef_post_x"] = round(float(eef_post[0]), 6) if eef_post is not None else ""
            trace_row["eef_post_y"] = round(float(eef_post[1]), 6) if eef_post is not None else ""
            trace_row["eef_post_z"] = round(float(eef_post[2]), 6) if eef_post is not None else ""
            trace_row["obj_post_x"] = round(float(obj_post[0]), 6) if obj_post is not None else ""
            trace_row["obj_post_y"] = round(float(obj_post[1]), 6) if obj_post is not None else ""
            trace_row["obj_post_z"] = round(float(obj_post[2]), 6) if obj_post is not None else ""
            if obj_post is not None and eef_post is not None:
                trace_row["eef_to_obj_post"] = round(float(np.linalg.norm(
                    np.array(eef_post) - np.array(obj_post))), 6)
            else:
                trace_row["eef_to_obj_post"] = ""
            if obj_post is not None and obj_init_z is not None:
                trace_row["obj_z_delta_post"] = round(float(obj_post[2]) - obj_init_z, 6)
            else:
                trace_row["obj_z_delta_post"] = ""

        step_trace.append(trace_row)

        if v["any_invalid"]:
            invalid_field_steps.append(step_idx)

        # Action identity
        action_identity.append({
            "step": step_idx,
            "action_hash_pre": raw_action_hash_pre,
            "action_hash_post": raw_action_hash_post,
            "action_identical": 1,
            "env_action_hash": env_action_hash_pre,
            "obs_hash": obs_hash,
        })

        # Latency
        latency_records.append({
            "step": step_idx,
            "model_inference_us": round(t_model * 1_000_000),
            "detector_update_us": round(t_det * 1_000_000) if not is_reference else "DISABLED",
            "env_step_us": round(t_env * 1_000_000),
        })

        # Detector candidate
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

    # Detector post-reset state
    if not is_reference:
        post_ep_state = {
            "next_expected_step": detector._next_expected_step,
            "history_len": len(detector.history),
            "emit_step": detector.emit_step,
            "candidate_count": len(detector.candidate_features),
        }
    else:
        pre_reset_state = {}
        post_ep_state = {}

    # ── Episode manifest ──
    episode_manifest = {
        "task": task, "state_id": state_id,
        "mode": "reference" if is_reference else "shadow",
        "attempt_id": attempt_id,
        "n_steps": n_steps, "max_steps": max_steps,
        "success_primary": int(success_primary) if success_step_primary >= 0 else 0,
        "success_step_primary": success_step_primary,
        "success_done_any": int(success_done_any),
        "success_check_any": int(success_check_any),
        "done_step": done_step_any,
        "timeout": n_steps >= max_steps and not success_primary,
        "infra_status": infra_status,
        "detector_exception": detector_exception,
        "action_identity_fail": action_identity_fail,
        "invalid_field_steps": invalid_field_steps,
        "n_invalid_field_steps": len(invalid_field_steps),
        "detector_emit_step": (detector.emit_step if not is_reference else "DISABLED"),
        "detector_n_candidates": (len(detector.candidate_features) if not is_reference else "DISABLED"),
        "detector_pre_reset": pre_reset_state if not is_reference else {},
        "detector_post_episode": post_ep_state if not is_reference else {},
        "first_action_generated": first_action_generated,
        "total_time_sec": round(time.time(), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_action_sequence_sha256": sha256_bytes("|".join(raw_action_sequence_sha).encode()),
        "env_action_sequence_sha256": sha256_bytes("|".join(env_action_sequence_sha).encode()),
        "obs_sequence_sha256": sha256_bytes("|".join(obs_sequence_sha).encode()),
    }

    # ── Write per-episode artifacts ──
    if not step_trace:
        return {"fatal": True, "reason": "Zero steps recorded"}

    with open(episode_dir / "step_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_trace[0].keys()))
        w.writeheader(); w.writerows(step_trace)

    if detector_candidates:
        with open(episode_dir / "detector_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detector_candidates[0].keys()))
            w.writeheader(); w.writerows(detector_candidates)
    else:
        # Write empty candidate file for schema consistency
        empty_fields = ["step", "task", "state_id", "score", "abstain", "abstained"]
        with open(episode_dir / "detector_candidates.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(empty_fields)

    with open(episode_dir / "detector_emission.json", "w") as f:
        json.dump({
            "emit_step": detector.emit_step if not is_reference else -1,
            "n_candidates": len(detector.candidate_features) if not is_reference else 0,
            "detector_enabled": not is_reference,
        }, f, indent=2)

    with open(episode_dir / "action_identity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(action_identity[0].keys()))
        w.writeheader(); w.writerows(action_identity)

    with open(episode_dir / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(latency_records[0].keys()))
        w.writeheader(); w.writerows(latency_records)

    # Provenance
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip() or "unknown"
    git_status_out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
    ).stdout
    git_clean = (git_status_out.strip() == "")
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()

    with open(episode_dir / "provenance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["git_HEAD", git_head])
        w.writerow(["git_clean", str(git_clean)])
        w.writerow(["branch", subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True,
        ).stdout.strip()])
        w.writerow(["checkpoint_sha", FROZEN_CHECKPOINT_SHA])
        w.writerow(["threshold", str(FROZEN_TAU)])
        w.writerow(["runner_sha", sha256_file(__file__)])
        w.writerow(["task", task])
        w.writerow(["state_id", str(state_id)])
        w.writerow(["mode", "reference" if is_reference else "shadow"])
        w.writerow(["attempt_id", str(attempt_id)])
        w.writerow(["cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "")])
        w.writerow(["nvidia_smi", nvidia_smi.replace("\n", " | ")])
        w.writerow(["command", " ".join(sys.argv)])
        w.writerow(["timestamp", datetime.now(timezone.utc).isoformat()])
        w.writerow(["phase_markers", "ATTEMPT_STARTED MODEL_LOADED FIRST_ACTION_GENERATED"])

    # Recursive artifact hashes (write after all other files, hash them all)
    artifact_names = [
        "ATTEMPT_STARTED.json", "MODEL_LOADED.json", "FIRST_ACTION_GENERATED.json",
        "step_trace.csv", "detector_candidates.csv", "detector_emission.json",
        "action_identity.csv", "latency.csv", "provenance.csv",
    ]
    hash_rows = []
    for an in artifact_names:
        ap = episode_dir / an
        if ap.exists():
            hash_rows.append([an, sha256_file(str(ap))])
    # Teacher sidecar
    ts_path = episode_dir / "teacher_sidecar.json"
    if args.enable_privileged_sidecar:
        priv_valid_steps = sum(1 for r in step_trace if r.get("privileged_valid") == 1)
        priv_invalid_steps = sum(1 for r in step_trace if r.get("privileged_valid") == 0)
        failure_reasons = sorted(set(
            r.get("privileged_failure_reason", "") for r in step_trace
            if r.get("privileged_failure_reason", "")
        ))
        ts_body = {
            "status": "COMPLETE",
            "task": task, "state_id": state_id,
            "sidecar_enabled": True,
            "target_object_name": target_object_name,
            "object_lookup_ok": object_lookup_ok,
            "obj_init_x": round(obj_init_x, 6) if obj_init_x is not None else None,
            "obj_init_y": round(obj_init_y, 6) if obj_init_y is not None else None,
            "obj_init_z": round(obj_init_z, 6) if obj_init_z is not None else None,
            "n_steps_total": len(step_trace),
            "privileged_valid_steps": priv_valid_steps,
            "privileged_invalid_steps": priv_invalid_steps,
            "privileged_valid": 1 if priv_invalid_steps == 0 and priv_valid_steps == len(step_trace) else 0,
            "privileged_failure_reasons": failure_reasons,
        }
    else:
        ts_body = {
            "status": "DISABLED",
            "task": task, "state_id": state_id,
            "sidecar_enabled": False,
        }
    with open(ts_path, "w") as f:
        json.dump(ts_body, f, indent=2)
    hash_rows.append(["teacher_sidecar.json", sha256_file(str(ts_path))])
    # Manifest itself
    with open(episode_dir / "episode_manifest.json", "w") as f:
        json.dump(episode_manifest, f, indent=2)
    hash_rows.append(["episode_manifest.json", sha256_file(str(episode_dir / "episode_manifest.json"))])

    with open(episode_dir / "artifact_hashes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        w.writerows(hash_rows)

    return episode_manifest


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--state-id", type=int, required=True)
    ap.add_argument("--episode-dir", required=True,
                    help="Exact episode output directory (runner creates this)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["reference", "shadow"], default="shadow")
    ap.add_argument("--attempt-id", type=int, default=1)
    ap.add_argument("--model-path",
                    default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--render-gpu-device-id", type=int, default=0)
    ap.add_argument("--model-gpu-device-id", type=int, default=-1)
    ap.add_argument("--max-steps-override", type=int, default=280)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--success-metric", choices=["done", "check_success"],
                    default="check_success")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-privileged-sidecar", action="store_true",
                    help="Record object pose and EEF-object distance (read-only)")
    # Provenance verification (fail-closed)
    ap.add_argument("--expected-git-head", default="",
                    help="If set, fail if git HEAD doesn't match")
    ap.add_argument("--expected-branch", default="",
                    help="If set, fail if branch doesn't match")
    ap.add_argument("--require-clean-worktree", action="store_true",
                    help="If set, fail if worktree is dirty")
    args = ap.parse_args()

    is_reference = (args.mode == "reference")
    episode_dir = Path(args.episode_dir)
    safe_tag = (
        f"{args.task}_s{args.state_id}_{args.mode}_attempt{args.attempt_id}"
    )
    assert episode_dir.name == safe_tag, (
        f"FATAL: episode_dir name {episode_dir.name} != safe_tag {safe_tag}"
    )

    # ── Phase 1: Create attempt directory atomically (runner OWNS this) ──
    try:
        episode_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"FATAL: Attempt directory already exists: {episode_dir}")
        sys.exit(1)

    write_phase_marker(episode_dir, "ATTEMPT_STARTED", {
        "task": args.task, "state_id": args.state_id,
        "mode": args.mode, "attempt_id": args.attempt_id,
    })

    # ── Provenance hard gates (MANDATORY when expected values provided) ──
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    if not git_head:
        print("FATAL: Could not determine git HEAD")
        sys.exit(1)

    git_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()

    # Use git status --porcelain to catch tracked AND untracked changes
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
    ).stdout
    git_clean = (git_status.strip() == "")

    if args.expected_git_head:
        assert git_head == args.expected_git_head, (
            f"FATAL: Git HEAD mismatch: got {git_head[:16]}..., "
            f"expected {args.expected_git_head[:16]}..."
        )
    if args.expected_branch:
        assert git_branch == args.expected_branch, (
            f"FATAL: Branch mismatch: got {git_branch}, "
            f"expected {args.expected_branch}"
        )
    if args.require_clean_worktree:
        assert git_clean, (
            f"FATAL: Worktree is not clean. git status --porcelain:\n{git_status[:500]}"
        )

    print(f"Git: HEAD={git_head[:16]}... branch={git_branch} clean={git_clean}")

    # ── Verify checkpoint ──
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA, (
        f"Checkpoint SHA mismatch: got {actual_ckpt[:16]}..."
    )
    print(f"Checkpoint: {actual_ckpt[:16]}... VERIFIED")

    # ── Load detector (shadow only) ──
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

    # ── Load OpenVLA ──
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

    write_phase_marker(episode_dir, "MODEL_LOADED", {
        "model_path": args.model_path, "device": device_ov,
        "dtype": str(model_dtype),
    })

    unnorm_key = "libero_object"
    K_trigger = 8
    action_dim = int(model.get_action_dim(unnorm_key))

    # ── Run episode ──
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {safe_tag}")
    result = run_episode(
        args, args.task, args.state_id, detector,
        model, processor, device_ov, model_dtype,
        unnorm_key, K_trigger, action_dim,
        args.attempt_id, episode_dir,
    )

    if result is None or result.get("fatal"):
        print(f"FATAL: {result.get('reason', 'unknown') if result else 'None'}")
        # Write failure manifest
        with open(episode_dir / "episode_manifest.json", "w") as f:
            json.dump({
                "fatal": True,
                "reason": result.get("reason", "unknown") if result else "None",
                "task": args.task, "state_id": args.state_id,
                "mode": args.mode, "attempt_id": args.attempt_id,
                "first_action_generated": phase_marker_exists(episode_dir, "FIRST_ACTION_GENERATED"),
            }, f, indent=2)
        sys.exit(1)

    print(f"  steps={result['n_steps']} success={result['success_primary']} "
          f"mode={result['mode']} emit={result['detector_emit_step']} "
          f"identity_fail={result['action_identity_fail']} "
          f"invalid_fields={result['n_invalid_field_steps']} "
          f"infra={result['infra_status']}")
    print(f"Output: {episode_dir}")


if __name__ == "__main__":
    main()
