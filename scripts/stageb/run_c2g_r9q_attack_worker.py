#!/usr/bin/env python3
"""Persistent R9Q attack worker: one suite model, many matched cells.

The worker is intentionally independent of the frozen D7 runner. It consumes
only the source-only R9Q manifest and the clean C2g bundle. Attack outcomes are
written as evidence and are never used to choose timing, thresholds, or rows.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS
from scripts.stageb.run_c2g_clean_window_vis_pgd import (
    build_attacker,
    clean_rgb,
    git_provenance,
    selected_model_hashes,
)
from gripper_attack.c2g_clean_window_runtime import C2gCleanWindowRuntime, sha256_file
from gripper_attack.c2g_gripper_critical_window_detector import FixedBurstTriggerScheduler

SUITES = {"libero_object", "libero_spatial", "libero_goal", "libero_10"}
CONDITIONS = {"CLEAN", "R9Q_DETECTOR_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"}
EXPECTED_FEATURE_NAMES = (
    "gripper_command", "gripper_qpos", "gripper_opening_proxy", "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz", "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count", "close_onset",
    "time_since_close", "eef_speed", "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _status(path: Path, **fields: Any) -> None:
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(fields, timestamp=time.time(), pid=os.getpid())
    write_json(path, current)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_thresholds(bundle: Path) -> dict[str, float | int]:
    raw = _load_json(bundle / "detector_config.json")
    if isinstance(raw.get("selected_config"), dict):
        raw = raw["selected_config"]
    if isinstance(raw.get("thresholds"), dict):
        raw = raw["thresholds"]
    required = ("tau_critical", "tau_release", "tau_ground", "persistence_window", "persistence_required")
    if not all(key in raw for key in required):
        raise ValueError(f"detector config is missing selected thresholds: {required}")
    result: dict[str, float | int] = {
        "tau_critical": float(raw["tau_critical"]),
        "tau_release": float(raw["tau_release"]),
        "tau_ground": float(raw["tau_ground"]),
        "persistence_window": int(raw["persistence_window"]),
        "persistence_required": int(raw["persistence_required"]),
    }
    if result["persistence_window"] != 3 or result["persistence_required"] != 2:
        raise ValueError("R9Q deployment requires the frozen 2-of-3 persistence contract")
    return result


def _parse_parent_key(value: str) -> tuple[str, int, int]:
    parts = str(value).split("/")
    if len(parts) < 3:
        raise ValueError(f"invalid parent_key: {value}")
    return parts[0], int(parts[1].replace("task_", "")), int(parts[2].replace("state_", ""))


def _int(row: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if row.get(name) not in (None, ""):
            return int(row[name])
    return default


def _build_args() -> argparse.Namespace:
    return argparse.Namespace(
        epsilon=6.0 / 255.0,
        step_size=(6.0 / 255.0) * 0.075,
        pgd_steps=20,
        random_start=True,
        temporal_init="prev_delta",
        gripper_margin=5.0,
        arm_preserve_weight=0.5,
        preprocess_backend="official_pil_lanczos",
        resize_size=224,
        device="cuda:0",
    )


def _canonical_success(env: Any, reward: Any, info: Any) -> tuple[bool, bool | None, bool, dict[str, Any]]:
    check = None
    if hasattr(env, "check_success"):
        check = bool(env.check_success())
    info = info if isinstance(info, dict) else {}
    info_fields = {
        "info_success": info.get("success"),
        "info_task_success": info.get("task_success"),
        "info_is_success": info.get("is_success"),
        "reward_after_step": float(reward) if isinstance(reward, (int, float, np.number)) else None,
    }
    observed = bool(check) if check is not None else False
    return observed, (bool(check) if check is not None else None), check is not None, info_fields


def _write_episode(output: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> None:
    write_json(output / "episode_metadata.json", metadata)
    with (output / "step_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    write_json(output / "artifact_sha256.json", {
        "metadata_sha256": sha256_file(output / "episode_metadata.json"),
        "step_records_sha256": sha256_file(output / "step_records.jsonl"),
    })


def run_cell(
    *,
    row: dict[str, Any],
    output_root: Path,
    model: Any,
    processor: Any,
    runtime: C2gCleanWindowRuntime,
    model_path: Path,
    unnorm_key: str,
    device: str,
    expected_commit: str,
    args: argparse.Namespace,
) -> bool:
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from v4_run_eval_openvla import decode_prepared_inputs_with_scores, decode_with_scores, physical_gripper_state, postprocess_openvla_action_for_libero
    from gripper_attack.attack_adapter import get_adv_inputs_from_attack_result

    parent_key = str(row["parent_key"])
    condition = str(row["condition"])
    suite, task_index, state_id = _parse_parent_key(parent_key)
    max_steps = _int(row, "max_steps", default=300)
    burst_length = _int(row, "burst_length", default=10)
    planned_start = _int(row, "planned_start_step", default=-1)
    objective_seed = _int(row, "objective_seed", default=42)
    output = output_root / "cells" / suite / parent_key / condition
    if output.exists() and (output / "episode_metadata.json").is_file():
        raise RuntimeError(f"refusing to overwrite existing cell: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records: list[dict[str, Any]] = []
    env = None
    success: bool | None = None
    runtime_valid = False
    error: dict[str, Any] | None = None
    task_language = ""
    termination_reason = "RUNTIME_ERROR"
    attack_count = 0
    first_attack_step: int | None = None
    detector_trigger_step: int | None = None
    try:
        runtime.reset()
        suite_object = benchmark.get_benchmark_dict()[suite]()
        task = suite_object.get_task(task_index)
        init_states = suite_object.get_task_init_states(task_index)
        if state_id < 0 or state_id >= len(init_states):
            raise IndexError(f"state_id {state_id} outside init-state count {len(init_states)}")
        task_language = str(getattr(task, "language", "") or getattr(task, "name", "") or "").strip()
        if not task_language:
            task_language = Path(task.bddl_file).stem.replace("_", " ")
        bddl_path = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        physical_gpu = int(os.environ.get("C2G_PHYSICAL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0])
        env, obs = build_v4_exact_env(bddl_path, physical_gpu, max_steps, 10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)
        streamer = SC5StreamingFeatureAdapterV2()
        eef_site = env.sim.model.site_name2id("gripper0_grip_site")
        previous_eef: np.ndarray | None = None
        attacker = None
        if condition in {"R9Q_DETECTOR_T10", "RAND_T10"}:
            attacker = build_attacker(model, processor, runtime, args, control=False, seed=objective_seed)

        for step in range(max_steps):
            rgb = clean_rgb(obs, step)
            gripper_state = physical_gripper_state(env, obs)
            qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
            qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
            opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
            clean_action, _, clean_decode_seconds, clean_generation = decode_with_scores(
                model, processor, device, rgb, task_language, unnorm_key, 8,
                libero_official_preprocess=False, libero_preprocess_backend=args.preprocess_backend,
                center_crop=True, resize_size=args.resize_size, drop_attention_mask=True,
            )
            if not getattr(clean_generation, "scores", None):
                raise RuntimeError("clean generation lacks token logits")
            clean_action = np.asarray(clean_action, dtype=np.float32)
            clean_gripper_logits = clean_generation.scores[-1][0].detach()
            clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)
            eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32)
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
            previous_eef = eef.copy()
            stream = streamer.update(
                step_id=step, raw_gripper=float(clean_action[-1]), env_gripper=float(clean_env_action[-1]),
                gripper_qpos=qpos_sum, gripper_opening_proxy=opening,
                eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                action_dx=float(clean_env_action[0]), action_dy=float(clean_env_action[1]),
                action_dz=float(clean_env_action[2]), action_gripper=float(clean_action[-1]),
            )
            if not stream.get("valid"):
                raise RuntimeError(f"invalid 25D stream at step {step}")
            features = [float(stream["features"].get(name, 0.0) or 0.0) for name in EXPECTED_FEATURE_NAMES]
            detector = runtime.predict(
                features_25d=features, rgb=rgb, task_language=task_language,
                clean_gripper_logits=clean_gripper_logits,
            )
            decision = detector["decision"]
            if decision.trigger_started and detector_trigger_step is None:
                detector_trigger_step = step
            if condition == "CLEAN":
                attack_this = False
            elif condition in {"R9Q_DETECTOR_T10", "COMMAND_OPEN_ORACLE"}:
                attack_this = bool(decision.attack_active)
            elif condition == "RAND_T10":
                attack_this = bool(planned_start <= step < planned_start + burst_length)
            else:
                raise ValueError(f"unsupported condition: {condition}")

            executed_action = clean_action.copy()
            executed_env_action = clean_env_action.copy()
            attack_debug: dict[str, Any] = {}
            attack_seconds = 0.0
            if attack_this:
                if first_attack_step is None:
                    first_attack_step = step
                if condition == "COMMAND_OPEN_ORACLE":
                    executed_action[-1] = 1.0
                    executed_env_action = postprocess_openvla_action_for_libero(executed_action, enabled=True)
                    attack_debug = {"payload_mode": "DIRECT_COMMAND_OPEN", "command_raw_gripper": 1.0}
                else:
                    if attacker is None:
                        raise RuntimeError("VIS attack condition has no attacker")
                    attack_start = time.perf_counter()
                    result = attacker.attack(
                        rgb, task_language, clean_action, clean_action, clean_generation, unnorm_key=unnorm_key,
                    )
                    adversarial_inputs = get_adv_inputs_from_attack_result(result)
                    executed_action, _, _, _ = decode_prepared_inputs_with_scores(
                        model, device, adversarial_inputs, unnorm_key, 8,
                    )
                    executed_action = np.asarray(executed_action, dtype=np.float32)
                    executed_env_action = postprocess_openvla_action_for_libero(executed_action, enabled=True)
                    attack_seconds = time.perf_counter() - attack_start
                    attack_debug = dict(result.debug or {}) if isinstance(result.debug, dict) else {}
                attack_count += 1

            obs, reward, done, info = env.step(executed_env_action)
            observed, check_success, check_available, info_fields = _canonical_success(env, reward, info)
            if not check_available:
                raise RuntimeError("env.check_success() is unavailable; canonical success is unknown")
            records.append({
                "step": step,
                "condition": condition,
                "clean_raw_action": clean_action.astype(float).tolist(),
                "executed_raw_action": executed_action.astype(float).tolist(),
                "clean_env_action": np.asarray(clean_env_action).astype(float).tolist(),
                "executed_env_action": np.asarray(executed_env_action).astype(float).tolist(),
                "action_delta": (executed_action - clean_action).astype(float).tolist(),
                "clean_gripper_raw": float(clean_action[-1]),
                "executed_gripper_raw": float(executed_action[-1]),
                "clean_gripper_env": float(clean_env_action[-1]),
                "executed_gripper_env": float(executed_env_action[-1]),
                "gripper_action_delta": float(executed_env_action[-1] - clean_env_action[-1]),
                "attack_delivered": bool(attack_this),
                "attack_index": attack_count - 1 if attack_this else None,
                "detector_ready": bool(detector["ready"]),
                "detector_outputs": detector["outputs"],
                "detector_susceptibility_gate": bool(detector["susceptibility_gate"]),
                "detector_susceptibility_gate_enabled": bool(detector["susceptibility_gate_enabled"]),
                "detector_effective_valid": bool(detector["effective_valid"]),
                "detector_policy_summary": detector["policy"],
                "detector_trigger_started": bool(decision.trigger_started),
                "detector_attack_active": bool(decision.attack_active),
                "scheduler_state": str(decision.state),
                "planned_start_step": planned_start,
                "gripper_qpos_sum": qpos_sum,
                "gripper_opening_proxy": opening,
                "runtime_valid_before_step": True,
                "reward_after_step": info_fields["reward_after_step"],
                "done_after_step": bool(done),
                "env_check_success_after_step": check_success,
                "info_success_after_step": info_fields["info_success"],
                "info_task_success_after_step": info_fields["info_task_success"],
                "info_is_success_after_step": info_fields["info_is_success"],
                "canonical_success_after_step": bool(observed),
                "success_check_available": bool(check_available),
                "clean_decode_seconds": float(clean_decode_seconds),
                "attack_seconds": float(attack_seconds),
                "attack_debug": {
                    key: value for key, value in attack_debug.items()
                    if key.endswith("sha256") or key in {
                        "gradient_transform", "gradient_transform_seed", "num_loss_forwards",
                        "num_backwards", "pixel_budget_linf", "pixel_budget_delta0_adv_inputs_linf",
                    }
                },
            })
            success = bool(observed)
            if observed:
                termination_reason = "ENV_CHECK_SUCCESS"
                break
            if done:
                termination_reason = "DONE_WITHOUT_SUCCESS"
                break
        else:
            termination_reason = "MAX_POLICY_STEPS"
        runtime_valid = bool(records and all(record["runtime_valid_before_step"] for record in records) and success is not None)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error_message": str(exc)}
        runtime_valid = False
        success = None
        termination_reason = "RUNTIME_ERROR"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    provenance = git_provenance(expected_commit)
    metadata = {
        "protocol_name": "C2G_R9Q_MATCHED_ATTACK",
        "protocol_version": "2026-07-13.v1",
        "attack_space": "VIS_PGD_OR_DIRECT_COMMAND",
        "parent_key": parent_key,
        "condition": condition,
        "suite": suite,
        "task_index": task_index,
        "state_id": state_id,
        "runtime_valid": runtime_valid,
        "success": success,
        "error": error,
        "termination_reason": termination_reason,
        "total_steps": len(records),
        "attack_delivery_count": attack_count,
        "expected_attack_frames": 0 if condition == "CLEAN" else burst_length,
        "first_attack_step": first_attack_step,
        "detector_trigger_step": detector_trigger_step,
        "planned_start_step": planned_start,
        "burst_length": burst_length,
        "objective_seed": objective_seed,
        "task_language": task_language,
        "policy_model_path": str(model_path),
        "policy_model_file_hashes": selected_model_hashes(model_path),
        "unnorm_key": unnorm_key,
        "detector_checkpoint_sha256": row["detector_checkpoint_sha256"],
        "detector_config_sha256": row["detector_config_sha256"],
        "detector_checkpoint_schema": getattr(runtime, "checkpoint_schema_version", ""),
        "normalization_sha256": getattr(runtime, "normalization_sha256", None),
        "susceptibility_gate_enabled": bool(getattr(runtime, "susceptibility_gate_enabled", True)),
        "runtime_gate_heads": ["critical_window", "release_safe", "grounding_confidence"],
        "manifest_sha256": row.get("source_parent_manifest_sha256"),
        "worker_id": os.environ.get("C2G_WORKER_ID", ""),
        "physical_gpu": int(os.environ.get("C2G_PHYSICAL_GPU", "-1")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "runtime_seconds": time.time() - started,
        "attack_outcomes_used_for_selection": False,
        **provenance,
    }
    _write_episode(output, metadata, records)
    return runtime_valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector-bundle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--model-load-lock-file", required=True)
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args(argv)
    manifest = Path(args.manifest).resolve()
    bundle = Path(args.detector_bundle).resolve()
    output_root = Path(args.output_root).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("worker manifest is empty")
    suites = {str(row.get("suite", "")) for row in rows}
    if len(suites) != 1 or next(iter(suites)) not in SUITES:
        raise SystemExit(f"worker must contain one suite, got {sorted(suites)}")
    suite = next(iter(suites))
    if any(str(row.get("assigned_worker_id")) != args.worker_id for row in rows):
        raise SystemExit("worker manifest assignment mismatch")
    if any(str(row.get("condition")) not in CONDITIONS for row in rows):
        raise SystemExit("unsupported condition in worker manifest")
    os.environ["C2G_WORKER_ID"] = args.worker_id
    os.environ["C2G_PHYSICAL_GPU"] = str(args.physical_gpu)
    _status(Path(args.status_file), worker_id=args.worker_id, physical_gpu=args.physical_gpu,
            suite=suite, phase="WAITING_MODEL_LOAD_LOCK", total_cell_count=len(rows), completed_cell_count=0,
            failed_cell_count=0)
    lock_path = Path(args.model_load_lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        _status(Path(args.status_file), phase="LOADING_MODEL")
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelClass
        model_path = Path(SUITE_MODELS[suite]).resolve()
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        model = AutoModelClass.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0",
        ).eval()
        norm_stats = getattr(model, "norm_stats", {})
        unnorm_key = suite if suite in norm_stats else f"{suite}_no_noops"
        if unnorm_key not in norm_stats:
            raise RuntimeError(f"cannot resolve {suite} unnorm_key from norm_stats.keys()={sorted(norm_stats)}")
        checkpoint = bundle / "checkpoint.pt"
        expected_checkpoint = str(rows[0]["detector_checkpoint_sha256"])
        if sha256_file(checkpoint) != expected_checkpoint:
            raise RuntimeError("detector bundle checkpoint SHA does not match manifest")
        thresholds = _selected_thresholds(bundle)
        runtime = C2gCleanWindowRuntime(
            checkpoint, openvla_model=model, openvla_processor=processor,
            unnorm_key=unnorm_key, device="cuda:0", burst_length=10,
            normalization_path=bundle / "normalization.json",
            susceptibility_gate_enabled=False,
        )
        runtime.scheduler = FixedBurstTriggerScheduler(
            burst_length=10,
            tau_critical=float(thresholds["tau_critical"]),
            tau_release=float(thresholds["tau_release"]),
            tau_ground=float(thresholds["tau_ground"]),
            persistence_window=int(thresholds["persistence_window"]),
            persistence_required=int(thresholds["persistence_required"]),
            one_shot=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _status(Path(args.status_file), phase="MODEL_READY", unnorm_key=unnorm_key,
                checkpoint_sha256=sha256_file(checkpoint), model_path=str(model_path))
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    _status(Path(args.status_file), phase="RUNNING_CELLS")
    failed = 0
    for index, row in enumerate(rows):
        _status(Path(args.status_file), phase="RUNNING_CELLS", current_parent_key=row["parent_key"],
                current_condition=row["condition"], completed_cell_count=index, failed_cell_count=failed)
        try:
            valid = run_cell(
                row=row, output_root=output_root, model=model, processor=processor, runtime=runtime,
                model_path=model_path, unnorm_key=unnorm_key, device="cuda:0",
                expected_commit=args.expected_git_commit, args=_build_args(),
            )
            if not valid:
                failed += 1
        except Exception as exc:
            failed += 1
            _status(Path(args.status_file), last_error=f"{type(exc).__name__}: {exc}")
    _status(Path(args.status_file), phase="PASS" if failed == 0 else "FAILED",
            completed_cell_count=len(rows), failed_cell_count=failed, current_parent_key=None)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
