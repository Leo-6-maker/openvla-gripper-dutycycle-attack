#!/usr/bin/env python3
"""Collect the authorized R8T train-only Teacher-v2 canary.

Unlike the legacy Clean2000 collector, this collector records complete raw and
applied 7D actions plus enough environment/controller/task provenance to make the
new trajectories independently auditable.  It remains CLEAN-only and never reads
or creates attacked outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.collect_c2g_clean_window_rollouts import (
    contact_pairs,
    entity_position,
    git_provenance,
    normalized_rgb,
    policy_features,
    read_manifest,
    save_rgb,
    sha256_file,
    target_support_contact,
)
from scripts.stageb.collect_c2g_clean_window_rollouts_event_v2 import (
    _binding_by_index,
    _event_with_binding,
    _single_binding_event,
    entity_joint_scalar_with_hint,
)
from scripts.stageb.verify_c2g_suite_model_map_strict import verify as verify_model_map
from gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from gripper_attack.c2g_clean_event_tracking import (
    goal_event_bindings,
    joint_hint_from_interaction_site,
    select_active_goal_event,
)
from gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from gripper_attack.c2g_clean_window_runtime import derive_gripper_token_semantics
from gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets

COLLECTION_SCHEMA = "c2g.r8t.teacher_v2_canary_collection.2026-07-11.v1"
TEACHER_SCHEMA = "c2g.teacher_v2.raw_privileged_evidence.2026-07-11.v1"
EVENT_TRACKING_SCHEMA = "c2g.clean_goal_event_tracking.2026-07-11.v1"
ACTION_ORDER = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def stable_episode_seed(base_seed: int, parent_key: str) -> int:
    digest = hashlib.sha256(f"R8T|{base_seed}|{parent_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def array_sha256(value: Any) -> tuple[str, list[int], str]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest(), list(array.shape), str(array.dtype)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        try:
            module = importlib.import_module(name)
            return str(getattr(module, "__version__", "LOCAL_SOURCE"))
        except Exception:
            return "NOT_INSTALLED"


def git_commit_for_path(path: Path) -> str:
    current = path.resolve()
    for root in (current, *current.parents):
        if (root / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip()
            except Exception:
                return "UNRESOLVED"
    return "UNRESOLVED"


def runtime_provenance() -> dict[str, Any]:
    values = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "libero": package_version("libero"),
        "robosuite": package_version("robosuite"),
        "mujoco": package_version("mujoco"),
        "mujoco_py": package_version("mujoco_py"),
    }
    try:
        import libero
        values["libero_source_path"] = str(Path(libero.__file__).resolve())
        values["libero_git_commit"] = git_commit_for_path(Path(libero.__file__).resolve())
    except Exception:
        values["libero_source_path"] = "UNRESOLVED"
        values["libero_git_commit"] = "UNRESOLVED"
    return values


def jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(child, depth + 1) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child, depth + 1) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def controller_provenance(env: Any) -> dict[str, Any]:
    roots = [env, getattr(env, "env", None), getattr(env, "unwrapped", None)]
    for root in roots:
        if root is None:
            continue
        robots = getattr(root, "robots", None)
        if not robots:
            continue
        robot = robots[0]
        controller = getattr(robot, "controller", None)
        output: dict[str, Any] = {
            "robot_class": type(robot).__name__,
            "controller_class": type(controller).__name__ if controller is not None else "UNRESOLVED",
        }
        for name in ("control_freq", "controller_config", "controller_configs", "control_dim", "action_dim"):
            value = getattr(root, name, None)
            if value is None:
                value = getattr(robot, name, None)
            if value is None and controller is not None:
                value = getattr(controller, name, None)
            if value is not None:
                output[name] = jsonable(value)
        if controller is not None:
            for name in ("input_type", "output_min", "output_max", "kp", "damping_ratio"):
                value = getattr(controller, name, None)
                if value is not None:
                    output[name] = jsonable(value)
        return output
    return {"controller_class": "UNRESOLVED"}


def finite_action(value: Any, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"{name} must be a finite 7D vector, got {action.shape}")
    return action


def selected_model_hashes(model_path: Path) -> dict[str, str]:
    allowed_prefixes = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    return {
        path.name: sha256_file(path)
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.name in allowed_prefixes
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260711)
    parser.add_argument("--near-target-threshold", type=float, default=0.08)
    parser.add_argument("--relative-lift-threshold", type=float, default=0.015)
    parser.add_argument("--progress-threshold", type=float, default=0.01)
    parser.add_argument("--fixture-motion-threshold", type=float, default=0.005)
    args = parser.parse_args(argv)

    provenance = git_provenance(args.expected_git_commit)
    model_verification = verify_model_map(
        args.suite_model_map.resolve(),
        args.suite_model_report.resolve(),
        args.goal_model_manifest.resolve(),
    )
    args.model_verification_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.model_verification_report, model_verification)
    model_map = json.loads(args.suite_model_map.read_text(encoding="utf-8"))
    episodes = read_manifest(args.manifest.resolve())
    if not episodes:
        raise RuntimeError("empty R8T canary manifest")
    suites = {str(row.get("suite", "")) for row in episodes}
    if len(suites) != 1 or not suites.issubset(SUITES):
        raise ValueError("R8T shard must contain exactly one valid suite")
    if any(row.get("cohort") != "DETECTOR_TRAIN" or row.get("split") != "train" for row in episodes):
        raise ValueError("R8T collector accepts train-only parents")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    from scripts.stageb.c2f_libero_openvla_adapter import _resolve_task_language
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from v4_run_eval_openvla import (
        decode_with_scores,
        physical_gripper_state,
        postprocess_openvla_action_for_libero,
    )

    runtime = runtime_provenance()
    suite = next(iter(suites))
    model_path = Path(str(model_map[suite])).resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
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
        raise RuntimeError(f"cannot resolve unnorm key for {suite}")
    semantics = derive_gripper_token_semantics(model, unnorm_key)

    results: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []
    suite_obj = benchmark.get_benchmark_dict()[suite]()

    for episode_index, episode in enumerate(episodes):
        task_index = int(episode["task_index"])
        state_id = int(episode["state_id"])
        parent_key = str(episode["parent_key"])
        seed = stable_episode_seed(args.base_seed, parent_key)
        set_deterministic_seeds(seed)
        episode_dir = output_root / "episodes" / suite / parent_key
        if episode_dir.exists():
            raise FileExistsError(episode_dir)
        episode_dir.mkdir(parents=True)
        rgb_dir = episode_dir / "rgb"
        step_path = episode_dir / "step_records.jsonl"
        metadata_path = episode_dir / "episode_metadata.json"
        env = None
        rows_written = 0
        started = time.time()
        last_info: dict[str, Any] = {}
        try:
            task = suite_obj.get_task(task_index)
            states = suite_obj.get_task_init_states(task_index)
            if state_id < 0 or state_id >= len(states):
                raise IndexError("state_id outside official init-state range")
            init_state = states[state_id]
            init_sha, init_shape, init_dtype = array_sha256(init_state)
            task_language, language_source = _resolve_task_language(task, episode)
            bddl_path = (Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).resolve()
            structured = parse_bddl_task_metadata(bddl_path)
            task_metadata = {
                **structured,
                "task_language": task_language,
                "suite": suite,
                "task_index": task_index,
                "episode_key": parent_key,
                "parent_key": parent_key,
                "gripper_command_semantics": "raw_openvla_threshold_0p5_close_below",
            }
            resolution = resolve_task_targets(task_metadata)
            mechanism = infer_clean_mechanism_type(task_metadata, resolution=resolution)
            bindings = goal_event_bindings(resolution)
            if mechanism != "unsupported_or_unknown" and not bindings:
                raise RuntimeError("eligible mechanism has no structured goal-event binding")
            binding_by_index = _binding_by_index(bindings)
            task_metadata.update(
                mechanism_type=mechanism,
                structured_goal_metadata={
                    "target_objects": list(resolution.resolved_target_objects),
                    "target_receptacles": list(resolution.resolved_receptacles),
                    "target_sites": list(resolution.resolved_sites),
                    "target_fixtures": list(resolution.resolved_manipulable_entities),
                    "target_destinations": list(resolution.resolved_destination_entities),
                    "goal_bindings": [list(value) for value in resolution.goal_bindings],
                },
                goal_event_bindings=[binding.to_dict() for binding in bindings],
                event_tracking_schema=EVENT_TRACKING_SCHEMA,
                teacher_schema_version=TEACHER_SCHEMA,
            )

            visible = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            env, obs = build_v4_exact_env(str(bddl_path), visible, args.max_steps, args.dummy_wait)
            obs = env.set_init_state(init_state)
            env, obs = apply_dummy_wait(env, obs, args.dummy_wait)
            controller = controller_provenance(env)
            streamer = SC5StreamingFeatureAdapterV2()
            eef_site = env.sim.model.site_name2id("gripper0_grip_site")
            previous_eef = None
            target_objects = set(resolution.resolved_target_objects)
            target_manipulable = set(resolution.resolved_manipulable_entities)
            region_owner_by_site = dict(task_metadata.get("region_owner_by_site", {}))
            baseline_z: dict[int, float] = {}
            initial_distance: dict[int, float] = {}
            initial_joint: dict[int, float] = {}
            last_active_index: int | None = None
            for binding in bindings:
                target_position = entity_position(env, binding.target_entity)
                destination_position = entity_position(env, binding.destination_entity) if binding.destination_entity else None
                if target_position is not None and destination_position is not None:
                    initial_distance[binding.subgoal_index] = float(np.linalg.norm(target_position - destination_position))
                hint = joint_hint_from_interaction_site(binding.target_entity, binding.interaction_site)
                joint = entity_joint_scalar_with_hint(env, binding.target_entity, hint)
                if joint is not None:
                    initial_joint[binding.subgoal_index] = joint

            with step_path.open("w", encoding="utf-8") as handle:
                for step in range(args.max_steps):
                    rgb = normalized_rgb(obs, step)
                    gripper_state = physical_gripper_state(env, obs)
                    qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
                    qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
                    opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
                    decoded, _, _, generation = decode_with_scores(
                        model, processor, args.device, rgb, task_language, unnorm_key, 8,
                        libero_official_preprocess=False,
                        libero_preprocess_backend="official_pil_lanczos",
                        center_crop=True,
                        resize_size=224,
                        drop_attention_mask=True,
                    )
                    if not getattr(generation, "scores", None):
                        raise RuntimeError("clean generation lacks score tensors")
                    raw_action = finite_action(decoded, "clean_action_raw_7d")
                    applied_action = finite_action(
                        postprocess_openvla_action_for_libero(raw_action, enabled=True),
                        "applied_action_7d",
                    )
                    logits = generation.scores[-1][0].detach()
                    policy = policy_features(logits, semantics)
                    top_k = min(16, int(logits.numel()))
                    top_values, top_ids = torch.topk(logits.float(), k=top_k)
                    eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32).copy()
                    velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
                    previous_eef = eef.copy()
                    stream = streamer.update(
                        step_id=step,
                        raw_gripper=float(raw_action[-1]),
                        env_gripper=float(applied_action[-1]),
                        gripper_qpos=qpos_sum,
                        gripper_opening_proxy=opening,
                        eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                        eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                        action_dx=float(applied_action[0]), action_dy=float(applied_action[1]),
                        action_dz=float(applied_action[2]), action_gripper=float(raw_action[-1]),
                    )
                    features = list(stream["features"].values())
                    if len(features) != 25 or not np.isfinite(np.asarray(features, dtype=np.float32)).all():
                        raise RuntimeError(f"invalid 25D features at step {step}")

                    pairs = contact_pairs(env)
                    event = select_active_goal_event(
                        pairs,
                        bindings,
                        manipulable_targets=sorted(target_manipulable),
                        finger_aliases=task_metadata.get("finger_aliases"),
                    )
                    if event["active_target_known"]:
                        last_active_index = int(event["active_subgoal_index"])
                    elif len(bindings) == 1:
                        event = _single_binding_event(bindings[0])
                    elif last_active_index is not None:
                        event = _event_with_binding(event, binding_by_index[last_active_index])
                        event["active_target_reason"] = "RETAINED_LAST_CONTACTED_SUBGOAL"
                    active_binding = (
                        binding_by_index.get(int(event["active_subgoal_index"]))
                        if event.get("active_target_known") and event.get("active_subgoal_index") is not None
                        else None
                    )
                    clean_close = float(raw_action[-1]) < 0.5
                    target_position = destination_position = None
                    relative_lift = progress = fixture_motion = None
                    target_contact = bilateral_contact = False
                    near_target = supported = release_safe = None
                    manipulation_active = constrained_active = None
                    if active_binding is not None:
                        target_contact = active_binding.target_entity in set(event.get("contacted_goal_targets", []))
                        bilateral_contact = active_binding.target_entity in set(event.get("bilateral_goal_targets", []))
                        target_position = entity_position(env, active_binding.target_entity)
                        destination_position = entity_position(env, active_binding.destination_entity) if active_binding.destination_entity else None
                        if target_position is not None and destination_position is not None:
                            current_distance = float(np.linalg.norm(target_position - destination_position))
                            initial_distance.setdefault(active_binding.subgoal_index, current_distance)
                            progress = initial_distance[active_binding.subgoal_index] - current_distance
                            near_target = current_distance <= args.near_target_threshold
                        if clean_close and bilateral_contact and target_position is not None and active_binding.target_entity in target_objects:
                            baseline_z.setdefault(active_binding.subgoal_index, float(target_position[2]))
                        if target_position is not None and active_binding.subgoal_index in baseline_z:
                            relative_lift = float(target_position[2]) - baseline_z[active_binding.subgoal_index]
                        hint = joint_hint_from_interaction_site(active_binding.target_entity, active_binding.interaction_site)
                        joint = entity_joint_scalar_with_hint(env, active_binding.target_entity, hint)
                        if joint is not None:
                            initial_joint.setdefault(active_binding.subgoal_index, joint)
                            fixture_motion = abs(joint - initial_joint[active_binding.subgoal_index])
                            constrained_active = fixture_motion >= args.fixture_motion_threshold
                        if any(value is not None for value in (relative_lift, progress, fixture_motion)):
                            manipulation_active = bool(
                                (relative_lift is not None and relative_lift >= args.relative_lift_threshold)
                                or (progress is not None and progress >= args.progress_threshold)
                                or (fixture_motion is not None and fixture_motion >= args.fixture_motion_threshold)
                            )
                        if active_binding.destination_entity:
                            support_entities = [active_binding.destination_entity]
                            owner = region_owner_by_site.get(active_binding.destination_entity)
                            if owner:
                                support_entities.append(owner)
                            supported = target_support_contact(pairs, active_binding.target_entity, support_entities)
                            if near_target is not None:
                                release_safe = bool(near_target and supported)
                        elif fixture_motion is not None:
                            release_safe = bool(
                                fixture_motion >= args.fixture_motion_threshold and not target_contact
                            )

                    row = {
                        "step": step,
                        "teacher_schema_version": TEACHER_SCHEMA,
                        "rgb_path": f"rgb/frame_{step:06d}.png",
                        "task_language": task_language,
                        "features_25d": [float(value) for value in features],
                        "clean_policy_intent_9d": policy,
                        **{name: policy[index] for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)},
                        "clean_action_raw_7d": raw_action.astype(float).tolist(),
                        "applied_action_7d": applied_action.astype(float).tolist(),
                        "action_order": list(ACTION_ORDER),
                        "clean_action_token_top_ids": top_ids.detach().cpu().tolist(),
                        "clean_action_token_top_logits": top_values.detach().cpu().tolist(),
                        "clean_gripper_command": float(raw_action[-1]),
                        "clean_close_intent": clean_close,
                        "mujoco_contact_pairs": pairs,
                        **event,
                        "object_relative_lift": relative_lift,
                        "target_distance_decrease": progress,
                        "constrained_manipulation_active": constrained_active,
                        "manipulation_progress_active": manipulation_active,
                        "near_target": near_target,
                        "supported_at_target": supported,
                        "release_safe": release_safe,
                        "target_object_position": None if target_position is None else target_position.astype(float).tolist(),
                        "target_destination_position": None if destination_position is None else destination_position.astype(float).tolist(),
                        "fixture_joint_motion": fixture_motion,
                        "active_target_contact": target_contact,
                        "active_target_bilateral_contact": bilateral_contact,
                    }
                    save_rgb(rgb_dir / f"frame_{step:06d}.png", rgb)
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    rows_written += 1
                    obs, _, done, info = env.step(applied_action)
                    last_info = dict(info or {})
                    if release_safe:
                        last_active_index = None
                    if done:
                        break

            metadata = {
                "schema": COLLECTION_SCHEMA,
                **task_metadata,
                "state_id": state_id,
                "cohort": episode["cohort"],
                "split": episode["split"],
                "task_language_source": language_source,
                "resolution": resolution.to_dict(),
                "model_path": str(model_path),
                "model_selected_hashes": selected_model_hashes(model_path),
                "suite_model_map": str(args.suite_model_map.resolve()),
                "suite_model_map_sha256": sha256_file(args.suite_model_map.resolve()),
                "suite_model_report": str(args.suite_model_report.resolve()),
                "suite_model_report_sha256": sha256_file(args.suite_model_report.resolve()),
                "goal_model_manifest": str(args.goal_model_manifest.resolve()),
                "goal_model_manifest_sha256": sha256_file(args.goal_model_manifest.resolve()),
                "model_verification_report": str(args.model_verification_report.resolve()),
                "model_verification_report_sha256": sha256_file(args.model_verification_report.resolve()),
                "unnorm_key": unnorm_key,
                "token_semantics_sha256": semantics["token_semantics_sha256"],
                "open_token_ids": list(semantics["open_token_ids"]),
                "close_token_ids": list(semantics["close_token_ids"]),
                "raw_action_order": list(ACTION_ORDER),
                "applied_action_order": list(ACTION_ORDER),
                "action_semantics": {
                    "raw": "OpenVLA unnormalized 7D delta-pose plus gripper",
                    "applied": "postprocess_openvla_action_for_libero(raw_action, enabled=True)",
                    "order": list(ACTION_ORDER),
                },
                "controller_config": controller,
                "runtime_versions": runtime,
                "bddl_file": str(bddl_path),
                "bddl_sha256": sha256_file(bddl_path),
                "official_init_state_sha256": init_sha,
                "official_init_state_shape": init_shape,
                "official_init_state_dtype": init_dtype,
                "replay_seed": seed,
                "base_seed": args.base_seed,
                "max_steps": args.max_steps,
                "dummy_wait": args.dummy_wait,
                "thresholds": {
                    "near_target": args.near_target_threshold,
                    "relative_lift": args.relative_lift_threshold,
                    "progress": args.progress_threshold,
                    "fixture_motion": args.fixture_motion_threshold,
                },
                "runtime_valid": True,
                "n_steps": rows_written,
                "clean_success_observed": bool(
                    last_info.get("success", False)
                    or last_info.get("task_success", False)
                    or last_info.get("is_success", False)
                ),
                "runtime_seconds": time.time() - started,
                "condition": "CLEAN",
                "student_allowed_modalities": [
                    "rgb", "task_language", "features_25d", "clean_policy_intent_9d"
                ],
                "student_forbidden_modalities": [
                    "mujoco_contact_pairs", "active_target_entity", "active_subgoal_index",
                    "object_relative_lift", "target_distance_decrease",
                    "target_object_position", "target_destination_position", "release_safe",
                    "attack_outcome", "post_intervention",
                ],
                **provenance,
            }
            write_json(metadata_path, metadata)
            results.append({
                "parent_key": parent_key,
                "suite": suite,
                "task_index": task_index,
                "state_id": state_id,
                "n_steps": rows_written,
                "clean_success_observed": metadata["clean_success_observed"],
                "status": "PASS",
            })
            for artifact in (metadata_path, step_path):
                artifact_entries.append({
                    "path": artifact.relative_to(output_root).as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                })
        except Exception as exc:
            failure = {
                "schema": COLLECTION_SCHEMA,
                "parent_key": parent_key,
                "suite": suite,
                "task_index": task_index,
                "state_id": state_id,
                "runtime_valid": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "n_steps": rows_written,
                **provenance,
            }
            write_json(metadata_path, failure)
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        print(f"[{episode_index + 1}/{len(episodes)}] {parent_key}: {rows_written} rows", flush=True)

    artifact_entries.sort(key=lambda row: row["path"])
    artifact_manifest = output_root / "c2g_r8t_collection_artifacts.jsonl"
    artifact_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifact_entries),
        encoding="utf-8",
    )
    report = {
        "schema": COLLECTION_SCHEMA,
        "status": "PASS_C2G_R8T_TEACHER_V2_CANARY_COLLECTION",
        "suite": suite,
        "episode_count": len(results),
        "runtime_valid_episode_count": len(results),
        "results": results,
        "artifact_manifest": str(artifact_manifest),
        "artifact_manifest_sha256": sha256_file(artifact_manifest),
        "model_verification_report": str(args.model_verification_report.resolve()),
        "model_verification_report_sha256": sha256_file(args.model_verification_report.resolve()),
        "openvla_clean_inference_runs": len(results),
        "libero_clean_rollouts": len(results),
        "attacks_launched": 0,
        "attack_outcomes_read": False,
        **provenance,
    }
    report_path = output_root / "c2g_r8t_collection_report.json"
    write_json(report_path, report)
    print(json.dumps({**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
