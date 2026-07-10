#!/usr/bin/env python3
"""Run the end-to-end clean-window Detector-v2 + visual PGD pipeline.

Execution order at every step:
  clean observation -> clean OpenVLA decode -> clean policy/proprio detector input
  -> detector or preregistered random-time scheduler -> fixed B-frame visual attack
  -> adversarial re-decode -> LIBERO env.step.

The runner supports the frozen five-condition matched-load matrix.  It never uses
attacked outcomes to choose detector timing or thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
from gripper_attack.c2g_clean_window_runtime import C2gCleanWindowRuntime, sha256_file
from gripper_attack.c2g_matched_load_manifest import CORE_CONDITIONS

PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"
DETECTOR_CONDITIONS = {"DET_GRIPPER_VIS_PGD", "DET_RANDOM_VIS_ATTACK"}
RANDOM_TIME_CONDITIONS = {"RANDTIME_GRIPPER_VIS_PGD", "RANDTIME_RANDOM_VIS_ATTACK"}
CONTROL_CONDITIONS = {"DET_RANDOM_VIS_ATTACK", "RANDTIME_RANDOM_VIS_ATTACK"}


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git_provenance(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"expected commit {expected_commit}, got {commit}")
    if status.strip():
        raise RuntimeError("refusing online run from dirty worktree")
    return {"git_commit": commit, "git_clean": True}


def parse_parent_key(value: str) -> tuple[str, int, int]:
    parts = value.split("/")
    if len(parts) < 3:
        raise ValueError("parent-key must begin suite/task_N/state_N")
    suite = parts[0]
    task_index = int(parts[1].replace("task_", ""))
    state_id = int(parts[2].replace("state_", ""))
    return suite, task_index, state_id


def visible_gpu_id() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def selected_model_hashes(model_path: Path) -> dict[str, str]:
    names = {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "processor_config.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "processing_prismatic.py", "configuration_prismatic.py",
        "modeling_prismatic.py",
    }
    return {
        path.name: sha256_file(path)
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.name in names
    }


def build_attacker(
    model: Any,
    processor: Any,
    runtime: C2gCleanWindowRuntime,
    args: argparse.Namespace,
    *,
    control: bool,
    seed: int,
) -> TokenPrefixPGDAttacker:
    token_map = runtime.token_semantics["token_action_map"]
    open_ids = runtime.token_semantics["open_token_ids"]
    target_token = max(open_ids, key=lambda token: (float(token_map[int(token)]), -int(token)))
    optimizer = {
        "method": "token_prefix_pgd",
        "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "target_token_id": int(target_token),
        "target_execution_class": "CLIP_MEDIATED_OPEN",
        "epsilon": float(args.epsilon),
        "step_size": float(args.step_size),
        "num_steps": int(args.pgd_steps),
        "random_start": bool(args.random_start),
        "prefix_refresh_interval": 1,
        "surrogate_score_path": "cached_autoregressive_generate_v1",
        "gripper_margin": float(args.gripper_margin),
        "arm_preserve_weight": float(args.arm_preserve_weight),
        "strict_route": True,
        "allow_fallback": False,
        "temporal_init": str(args.temporal_init),
        "gradient_transform": "permute" if control else "none",
        "gradient_transform_seed": int(seed),
    }
    return TokenPrefixPGDAttacker(
        model=model,
        processor=processor,
        config={"attack_optimizer": optimizer},
        seed=int(seed),
        preprocess_kwargs={
            "libero_official_preprocess": False,
            "libero_preprocess_backend": args.preprocess_backend,
            "center_crop": True,
            "resize_size": int(args.resize_size),
            "postprocess_gripper": True,
        },
        device=args.device,
    )


def clean_rgb(obs: MappingLike, step: int) -> np.ndarray:
    if "agentview_image" not in obs:
        raise RuntimeError(f"missing agentview_image at step {step}")
    image = np.asarray(obs["agentview_image"])
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if np.nanmax(image) <= 1.0 else image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] < 3 or image.size == 0 or np.max(image[..., :3]) < 5:
        raise RuntimeError(f"blank or malformed RGB at step {step}")
    return image[..., :3].copy()


# MappingLike avoids importing typing.Mapping at runtime in old server environments.
MappingLike = Dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--condition", required=True, choices=CORE_CONDITIONS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--policy-model-manifest", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--planned-start-step", type=int, default=-1)
    parser.add_argument("--objective-seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=6.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=(6.0 / 255.0) * 0.075)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-init", default="prev_delta")
    parser.add_argument("--gripper-margin", type=float, default=5.0)
    parser.add_argument("--arm-preserve-weight", type=float, default=0.5)
    parser.add_argument("--preprocess-backend", default="official_pil_lanczos")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--minimum-open-minus-close-log-mass", type=float, default=-8.0)
    parser.add_argument("--minimum-entropy", type=float, default=0.0)
    args = parser.parse_args(argv)

    suite, task_index, state_id = parse_parent_key(args.parent_key)
    if args.condition in RANDOM_TIME_CONDITIONS and args.planned_start_step < 0:
        raise ValueError("random-time conditions require --planned-start-step")
    if args.condition in DETECTOR_CONDITIONS and args.planned_start_step >= 0:
        raise ValueError("detector-timing conditions must not supply planned start")
    output = Path(args.output_dir).resolve() / args.parent_key / args.condition
    output.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(args.expected_git_commit)
    started = time.time()
    records: list[dict[str, Any]] = []
    env = None

    try:
        from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelClass
        from libero.libero import benchmark, get_libero_path
        from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
        from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
        from v4_run_eval_openvla import (
            decode_prepared_inputs_with_scores,
            decode_with_scores,
            physical_gripper_state,
            postprocess_openvla_action_for_libero,
        )

        model_path = Path(args.model_path or SUITE_MODELS[suite]).resolve()
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        model = AutoModelClass.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=args.device,
        ).eval()
        unnorm_key = suite if suite in getattr(model, "norm_stats", {}) else f"{suite}_no_noops"
        if unnorm_key not in getattr(model, "norm_stats", {}):
            raise RuntimeError(f"cannot resolve unnorm_key for {suite}")
        runtime = C2gCleanWindowRuntime(
            args.checkpoint,
            openvla_model=model,
            openvla_processor=processor,
            unnorm_key=unnorm_key,
            device=args.device,
            burst_length=args.burst_length,
            minimum_open_minus_close_log_mass=args.minimum_open_minus_close_log_mass,
            minimum_entropy=args.minimum_entropy,
        )
        control = args.condition in CONTROL_CONDITIONS
        attacker = None
        if args.condition != "CLEAN":
            attacker = build_attacker(
                model,
                processor,
                runtime,
                args,
                control=control,
                seed=args.objective_seed,
            )

        suite_object = benchmark.get_benchmark_dict()[suite]()
        task = suite_object.get_task(task_index)
        init_states = suite_object.get_task_init_states(task_index)
        if state_id < 0 or state_id >= len(init_states):
            raise IndexError("state_id outside task init-state range")
        task_language = str(getattr(task, "language", "") or task.name or "").strip()
        if not task_language:
            task_language = str(Path(task.bddl_file).stem).replace("_", " ")
        bddl_path = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        env, obs = build_v4_exact_env(bddl_path, visible_gpu_id(), args.max_steps, 10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)
        eef_site = env.sim.model.site_name2id("gripper0_grip_site")
        streamer = SC5StreamingFeatureAdapterV2()
        previous_eef = None
        attack_count = 0
        first_attack_step = None
        success = False

        for step in range(args.max_steps):
            rgb = clean_rgb(obs, step)
            gripper_state = physical_gripper_state(env, obs)
            qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
            qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
            opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
            clean_action, _, clean_decode_seconds, clean_generation = decode_with_scores(
                model,
                processor,
                args.device,
                rgb,
                task_language,
                unnorm_key,
                8,
                libero_official_preprocess=False,
                libero_preprocess_backend=args.preprocess_backend,
                center_crop=True,
                resize_size=args.resize_size,
                drop_attention_mask=True,
            )
            if not getattr(clean_generation, "scores", None):
                raise RuntimeError("clean generation lacks token logits")
            clean_gripper_logits = clean_generation.scores[-1][0].detach()
            clean_action = np.asarray(clean_action, dtype=np.float32)
            clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)
            eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32)
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
            previous_eef = eef.copy()
            stream = streamer.update(
                step_id=step,
                raw_gripper=float(clean_action[-1]),
                env_gripper=float(clean_env_action[-1]),
                gripper_qpos=qpos_sum,
                gripper_opening_proxy=opening,
                eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                action_dx=float(clean_env_action[0]), action_dy=float(clean_env_action[1]),
                action_dz=float(clean_env_action[2]), action_gripper=float(clean_action[-1]),
            )
            if not stream.get("valid"):
                raise RuntimeError(f"invalid 25D stream at step {step}")
            canonical_names = list(stream["features"].keys())
            expected_names = [
                "gripper_command", "gripper_qpos", "gripper_opening_proxy",
                "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
                "action_dx", "action_dy", "action_dz", "action_gripper",
                "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
                "close_onset", "time_since_close", "eef_speed", "eef_z_delta_since_close",
                "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
                "opening_proxy_variance_5", "eef_speed_variance_5",
            ]
            features_25d = [float(stream["features"].get(name, 0.0) or 0.0) for name in expected_names]
            detector_result = runtime.predict(
                features_25d=features_25d,
                rgb=rgb,
                task_language=task_language,
                clean_gripper_logits=clean_gripper_logits,
            )

            if args.condition == "CLEAN":
                attack_this_step = False
            elif args.condition in DETECTOR_CONDITIONS:
                attack_this_step = bool(detector_result["decision"].attack_active)
            else:
                attack_this_step = bool(
                    args.planned_start_step <= step < args.planned_start_step + args.burst_length
                )
            executed_action = clean_action.copy()
            executed_env_action = clean_env_action.copy()
            attack_result = None
            adv_generation = None
            attack_seconds = 0.0
            if attack_this_step:
                if attacker is None:
                    raise RuntimeError("attack condition has no attacker")
                if first_attack_step is None:
                    first_attack_step = step
                t_attack = time.perf_counter()
                attack_result = attacker.attack(
                    rgb,
                    task_language,
                    clean_action,
                    clean_action,
                    clean_generation,
                    unnorm_key=unnorm_key,
                )
                adversarial_inputs = get_adv_inputs_from_attack_result(attack_result)
                executed_action, _, _, adv_generation = decode_prepared_inputs_with_scores(
                    model,
                    args.device,
                    adversarial_inputs,
                    unnorm_key,
                    8,
                )
                executed_action = np.asarray(executed_action, dtype=np.float32)
                executed_env_action = postprocess_openvla_action_for_libero(executed_action, enabled=True)
                attack_seconds = time.perf_counter() - t_attack
                attack_count += 1

            debug = attack_result.debug if attack_result is not None and isinstance(attack_result.debug, dict) else {}
            record = {
                "step": step,
                "condition": args.condition,
                "detector_ready": bool(detector_result["ready"]),
                "detector_outputs": detector_result["outputs"],
                "clean_policy": detector_result["policy"],
                "susceptibility_gate": bool(detector_result["susceptibility_gate"]),
                "scheduler_state": str(detector_result["decision"].state),
                "trigger_started": bool(detector_result["decision"].trigger_started),
                "detector_attack_active": bool(detector_result["decision"].attack_active),
                "attack_delivered": bool(attack_this_step),
                "attack_index": attack_count - 1 if attack_this_step else None,
                "clean_raw_action": clean_action.astype(float).tolist(),
                "executed_raw_action": executed_action.astype(float).tolist(),
                "clean_env_action": clean_env_action.astype(float).tolist(),
                "executed_env_action": executed_env_action.astype(float).tolist(),
                "clean_gripper_raw": float(clean_action[-1]),
                "executed_gripper_raw": float(executed_action[-1]),
                "clean_gripper_env": float(clean_env_action[-1]),
                "executed_gripper_env": float(executed_env_action[-1]),
                "arm_action_delta_l2": float(np.linalg.norm(executed_env_action[:-1] - clean_env_action[:-1])),
                "gripper_action_delta": float(executed_env_action[-1] - clean_env_action[-1]),
                "gripper_qpos_sum": qpos_sum,
                "gripper_opening_proxy": opening,
                "num_loss_forwards": int(debug.get("num_loss_forwards", 0)),
                "num_backwards": int(debug.get("num_backwards", 0)),
                "num_adv_decodes": 1 if attack_this_step else 0,
                "observation_perturb_linf": float(getattr(attack_result, "observation_perturb_linf", 0.0) or 0.0),
                "observation_perturb_l2": float(getattr(attack_result, "observation_perturb_l2", 0.0) or 0.0),
                "clean_decode_seconds": clean_decode_seconds,
                "attack_seconds": attack_seconds,
                "success": False,
            }
            records.append(record)
            obs, reward, done, info = env.step(executed_env_action)
            success = bool((info or {}).get("success", False) or reward > 0.5)
            if success:
                records[-1]["success"] = True
            if done:
                break

        metadata = {
            "protocol_name": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "parent_key": args.parent_key,
            "condition": args.condition,
            "suite": suite,
            "task_index": task_index,
            "state_id": state_id,
            "success": success,
            "runtime_valid": True,
            "total_steps": len(records),
            "attack_delivery_count": attack_count,
            "first_attack_step": first_attack_step,
            "expected_attack_frames": 0 if args.condition == "CLEAN" else args.burst_length,
            "planned_start_step": args.planned_start_step if args.condition in RANDOM_TIME_CONDITIONS else None,
            "objective_family": "NONE" if args.condition == "CLEAN" else (
                "SHUFFLED_GRIPPER_GRADIENT" if control else "GRIPPER_TARGETED_VIS_PGD"
            ),
            "attack_load": {
                "burst_length": args.burst_length,
                "epsilon": args.epsilon,
                "step_size": args.step_size,
                "pgd_steps": args.pgd_steps,
                "random_start": args.random_start,
                "temporal_init": args.temporal_init,
                "preprocess_backend": args.preprocess_backend,
                "resize_size": args.resize_size,
            },
            "detector_checkpoint": str(Path(args.checkpoint).resolve()),
            "detector_checkpoint_sha256": sha256_file(Path(args.checkpoint).resolve()),
            "token_semantics": {
                "open_count": len(runtime.token_semantics["open_token_ids"]),
                "close_count": len(runtime.token_semantics["close_token_ids"]),
                "sha256": runtime.token_semantics["token_semantics_sha256"],
            },
            "policy_model_path": str(model_path),
            "policy_model_file_hashes": selected_model_hashes(model_path),
            "task_language": task_language,
            "task_bddl": bddl_path,
            "objective_seed": args.objective_seed,
            "runtime_seconds": time.time() - started,
            **provenance,
        }
        exit_code = 0
    except Exception as exc:
        metadata = {
            "protocol_name": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "parent_key": args.parent_key,
            "condition": args.condition,
            "suite": suite,
            "task_index": task_index,
            "state_id": state_id,
            "success": None,
            "runtime_valid": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "total_steps": len(records),
            "runtime_seconds": time.time() - started,
            **provenance,
        }
        exit_code = 1
    finally:
        write_json(output / "episode_metadata.json", metadata)
        with (output / "step_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
