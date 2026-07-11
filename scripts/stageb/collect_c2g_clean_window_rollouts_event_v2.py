#!/usr/bin/env python3
"""Collect clean Detector-v2 rollouts with per-step active goal-event tracking.

This release collector supersedes the legacy single ``primary_target`` shortcut. It
binds each official BDDL goal predicate to its manipulated target and destination,
selects the current target from clean finger contacts, and maintains independent
progress baselines for every target. Ambiguous multi-target steps remain null/unknown.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.stageb.collect_c2g_clean_window_rollouts import (
    COLLECTION_SCHEMA,
    contact_pairs,
    entity_joint_scalar,
    entity_position,
    git_provenance,
    normalized_rgb,
    policy_features,
    read_manifest,
    save_rgb,
    sha256_file,
    target_support_contact,
    visible_gpu_id,
    write_json,
    model_names,
)
from gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from gripper_attack.c2g_clean_event_tracking import (
    GoalEventBinding,
    goal_event_bindings,
    joint_hint_from_interaction_site,
    select_active_goal_event,
)
from gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from gripper_attack.c2g_clean_window_runtime import derive_gripper_token_semantics
from gripper_attack.c2g_teacher_v2_contact_identity import canonicalize_mujoco_name
from gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets


REPO = Path(__file__).resolve().parents[2]
EVENT_TRACKING_SCHEMA = "c2g.clean_goal_event_tracking.2026-07-11.v1"


def _canonical_tokens(value: str) -> set[str]:
    return {token for token in canonicalize_mujoco_name(value).split("_") if token}


def entity_joint_scalar_with_hint(env: Any, entity: str, hint: str = "") -> float | None:
    """Read one fixture joint, preferring a region-derived selector.

    Official cabinet sites such as ``wooden_cabinet_1_middle_region`` correspond to
    joints such as ``wooden_cabinet_1_middle_level``. Averaging every cabinet joint
    would erase the active drawer motion, so a nonempty selector must match the joint
    token set before the conservative entity-wide fallback is used.
    """

    names = model_names(env.sim.model, "joint")
    target = canonicalize_mujoco_name(entity)
    candidates = [
        index
        for index, name in enumerate(names)
        if canonicalize_mujoco_name(name) == target
        or canonicalize_mujoco_name(name).startswith(target + "_")
    ]
    if hint:
        hint_tokens = _canonical_tokens(hint)
        hinted = [
            index for index in candidates
            if hint_tokens and hint_tokens.issubset(_canonical_tokens(names[index]))
        ]
        if hinted:
            candidates = hinted
        elif len(candidates) > 1:
            return None
    if not candidates:
        return entity_joint_scalar(env, entity)
    values: list[float] = []
    for joint_id in candidates:
        try:
            address = int(env.sim.model.jnt_qposadr[joint_id])
            values.append(float(env.sim.data.qpos[address]))
        except Exception:
            continue
    return float(np.mean(values)) if values else None


def _binding_by_index(bindings: Sequence[GoalEventBinding]) -> dict[int, GoalEventBinding]:
    output = {int(binding.subgoal_index): binding for binding in bindings}
    if len(output) != len(bindings):
        raise ValueError("duplicate goal-event subgoal index")
    return output


def _single_binding_event(binding: GoalEventBinding) -> dict[str, Any]:
    return {
        "active_target_known": True,
        "active_target_entity": binding.target_entity,
        "active_subgoal_index": int(binding.subgoal_index),
        "active_operator": binding.operator,
        "active_destination_entity": binding.destination_entity or None,
        "active_interaction_site": binding.interaction_site or None,
        "active_target_bilateral_contact": False,
        "active_target_reason": "SINGLE_STRUCTURED_GOAL_TARGET",
        "contacted_goal_targets": [],
        "bilateral_goal_targets": [],
        "per_target_contact_reason": {},
    }


def _event_with_binding(event: Mapping[str, Any], binding: GoalEventBinding) -> dict[str, Any]:
    output = dict(event)
    output.update(
        active_target_known=True,
        active_target_entity=binding.target_entity,
        active_subgoal_index=int(binding.subgoal_index),
        active_operator=binding.operator,
        active_destination_entity=binding.destination_entity or None,
        active_interaction_site=binding.interaction_site or None,
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--model-path-template", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--near-target-threshold", type=float, default=0.08)
    parser.add_argument("--relative-lift-threshold", type=float, default=0.015)
    parser.add_argument("--progress-threshold", type=float, default=0.01)
    parser.add_argument("--fixture-motion-threshold", type=float, default=0.005)
    parser.add_argument("--suite", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    provenance = git_provenance(args.expected_git_commit)
    episodes = read_manifest(args.manifest.resolve())
    if args.suite:
        episodes = [row for row in episodes if str(row.get("suite")) == args.suite]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError("no episodes selected")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS, _resolve_task_language
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from v4_run_eval_openvla import decode_with_scores, physical_gripper_state, postprocess_openvla_action_for_libero

    model_cache: dict[str, tuple[Any, Any, str, dict[str, Any], Path]] = {}
    results: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for episode_index, episode in enumerate(episodes):
        suite = str(episode["suite"])
        task_index = int(episode["task_index"])
        state_id = int(episode.get("state_id", 0))
        parent_key = str(episode.get("parent_key") or f"{suite}/task_{task_index}/state_{state_id}")
        episode_dir = output_root / "episodes" / suite / parent_key
        if episode_dir.exists() and not args.overwrite:
            raise FileExistsError(f"existing episode directory: {episode_dir}")
        if episode_dir.exists():
            import shutil
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)
        rgb_dir = episode_dir / "rgb"
        step_path = episode_dir / "step_records.jsonl"
        metadata_path = episode_dir / "episode_metadata.json"
        env = None
        started = time.time()
        rows_written = 0
        last_info: dict[str, Any] = {}

        try:
            if suite not in model_cache:
                model_path = Path(
                    args.model_path_template.format(suite=suite)
                    if args.model_path_template else SUITE_MODELS[suite]
                ).resolve()
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
                model_cache[suite] = (model, processor, unnorm_key, semantics, model_path)
            model, processor, unnorm_key, semantics, model_path = model_cache[suite]

            suite_obj = benchmark.get_benchmark_dict()[suite]()
            task = suite_obj.get_task(task_index)
            states = suite_obj.get_task_init_states(task_index)
            if state_id < 0 or state_id >= len(states):
                raise IndexError("state_id outside task init-state range")
            task_language, language_source = _resolve_task_language(task, episode)
            bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
            task_metadata["mechanism_type"] = mechanism
            task_metadata["structured_goal_metadata"] = {
                "target_objects": list(resolution.resolved_target_objects),
                "target_receptacles": list(resolution.resolved_receptacles),
                "target_sites": list(resolution.resolved_sites),
                "target_fixtures": list(resolution.resolved_manipulable_entities),
                "target_destinations": list(resolution.resolved_destination_entities),
                "goal_bindings": [list(value) for value in resolution.goal_bindings],
            }
            task_metadata["goal_event_bindings"] = [binding.to_dict() for binding in bindings]
            task_metadata["event_tracking_schema"] = EVENT_TRACKING_SCHEMA

            env, obs = build_v4_exact_env(
                str(bddl_path), visible_gpu_id(), args.max_steps, args.dummy_wait
            )
            obs = env.set_init_state(states[state_id])
            env, obs = apply_dummy_wait(env, obs, args.dummy_wait)
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
                destination_position = (
                    entity_position(env, binding.destination_entity)
                    if binding.destination_entity else None
                )
                if target_position is not None and destination_position is not None:
                    initial_distance[binding.subgoal_index] = float(
                        np.linalg.norm(target_position - destination_position)
                    )
                hint = joint_hint_from_interaction_site(
                    binding.target_entity, binding.interaction_site
                )
                joint = entity_joint_scalar_with_hint(env, binding.target_entity, hint)
                if joint is not None:
                    initial_joint[binding.subgoal_index] = joint

            with step_path.open("w", encoding="utf-8") as step_handle:
                for step in range(args.max_steps):
                    rgb = normalized_rgb(obs, step)
                    gripper_state = physical_gripper_state(env, obs)
                    qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
                    qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
                    opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
                    action, _, _, generation = decode_with_scores(
                        model,
                        processor,
                        args.device,
                        rgb,
                        task_language,
                        unnorm_key,
                        8,
                        libero_official_preprocess=False,
                        libero_preprocess_backend="official_pil_lanczos",
                        center_crop=True,
                        resize_size=224,
                        drop_attention_mask=True,
                    )
                    if not getattr(generation, "scores", None):
                        raise RuntimeError("clean generation lacks score tensors")
                    action = np.asarray(action, dtype=np.float32)
                    env_action = postprocess_openvla_action_for_libero(action, enabled=True)
                    logits = generation.scores[-1][0].detach()
                    policy = policy_features(logits, semantics)
                    eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32)
                    velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
                    previous_eef = eef.copy()
                    stream = streamer.update(
                        step_id=step,
                        raw_gripper=float(action[-1]),
                        env_gripper=float(env_action[-1]),
                        gripper_qpos=qpos_sum,
                        gripper_opening_proxy=opening,
                        eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                        eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                        action_dx=float(env_action[0]), action_dy=float(env_action[1]),
                        action_dz=float(env_action[2]), action_gripper=float(action[-1]),
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
                        if event.get("active_target_known")
                        and event.get("active_subgoal_index") is not None
                        else None
                    )
                    clean_close = float(action[-1]) < 0.5
                    target_position = None
                    destination_position = None
                    current_distance = None
                    relative_lift = None
                    progress = None
                    fixture_motion = None
                    target_contact = False
                    bilateral_contact = False
                    near_target: bool | None = None
                    supported: bool | None = None
                    release_safe: bool | None = None
                    manipulation_active: bool | None = None
                    constrained_active: bool | None = None

                    if active_binding is not None:
                        target_contact = active_binding.target_entity in set(
                            event.get("contacted_goal_targets", [])
                        )
                        bilateral_contact = active_binding.target_entity in set(
                            event.get("bilateral_goal_targets", [])
                        )
                        target_position = entity_position(env, active_binding.target_entity)
                        destination_position = (
                            entity_position(env, active_binding.destination_entity)
                            if active_binding.destination_entity else None
                        )
                        if target_position is not None and destination_position is not None:
                            current_distance = float(
                                np.linalg.norm(target_position - destination_position)
                            )
                            if active_binding.subgoal_index not in initial_distance:
                                initial_distance[active_binding.subgoal_index] = current_distance
                            progress = initial_distance[active_binding.subgoal_index] - current_distance
                            near_target = current_distance <= args.near_target_threshold

                        if (
                            clean_close
                            and bilateral_contact
                            and target_position is not None
                            and active_binding.target_entity in target_objects
                            and active_binding.subgoal_index not in baseline_z
                        ):
                            baseline_z[active_binding.subgoal_index] = float(target_position[2])
                        if (
                            target_position is not None
                            and active_binding.subgoal_index in baseline_z
                        ):
                            relative_lift = float(target_position[2]) - baseline_z[active_binding.subgoal_index]

                        hint = joint_hint_from_interaction_site(
                            active_binding.target_entity, active_binding.interaction_site
                        )
                        joint = entity_joint_scalar_with_hint(
                            env, active_binding.target_entity, hint
                        )
                        if joint is not None:
                            if active_binding.subgoal_index not in initial_joint:
                                initial_joint[active_binding.subgoal_index] = joint
                            fixture_motion = abs(
                                joint - initial_joint[active_binding.subgoal_index]
                            )
                            constrained_active = fixture_motion >= args.fixture_motion_threshold

                        known_progress_values = [
                            relative_lift is not None,
                            progress is not None,
                            fixture_motion is not None,
                        ]
                        if any(known_progress_values):
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
                            supported = target_support_contact(
                                pairs, active_binding.target_entity, support_entities
                            )
                            if near_target is not None:
                                release_safe = bool(near_target and supported)
                        elif fixture_motion is not None:
                            # For articulated tasks, active contact + joint motion is the
                            # vulnerable manipulation phase. Releasing is marked safe only
                            # after measurable goal motion and loss of current target contact.
                            release_safe = bool(
                                fixture_motion >= args.fixture_motion_threshold
                                and not target_contact
                            )

                    # Do not serialize false evidence when the active target or metric is
                    # unresolved. Null is required so Teacher-v2 masks the row.
                    row = {
                        "step": step,
                        "rgb_path": f"rgb/frame_{step:06d}.png",
                        "task_language": task_language,
                        "features_25d": [float(value) for value in features],
                        "clean_policy_intent_9d": policy,
                        **{name: policy[index] for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)},
                        "clean_gripper_command": float(action[-1]),
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
                    step_handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    rows_written += 1
                    obs, reward, done, info = env.step(env_action)
                    last_info = dict(info or {})
                    if release_safe:
                        last_active_index = None
                    if done:
                        break

            metadata = {
                "schema": COLLECTION_SCHEMA,
                **task_metadata,
                "state_id": state_id,
                "task_language_source": language_source,
                "resolution": resolution.to_dict(),
                "model_path": str(model_path),
                "model_selected_hashes": {
                    path.name: sha256_file(path)
                    for path in sorted(model_path.iterdir())
                    if path.is_file()
                    and path.name in {
                        "config.json",
                        "model.safetensors.index.json",
                        "processor_config.json",
                        "tokenizer.json",
                    }
                },
                "unnorm_key": unnorm_key,
                "token_semantics_sha256": semantics["token_semantics_sha256"],
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
                    "mujoco_contact_pairs",
                    "active_target_entity",
                    "active_subgoal_index",
                    "object_relative_lift",
                    "target_distance_decrease",
                    "target_object_position",
                    "target_destination_position",
                    "release_safe",
                    "attack_outcome",
                    "post_intervention",
                ],
                **provenance,
            }
            write_json(metadata_path, metadata)
            results.append(
                {
                    "parent_key": parent_key,
                    "suite": suite,
                    "task_index": task_index,
                    "state_id": state_id,
                    "n_steps": rows_written,
                    "mechanism_type": mechanism,
                    "goal_event_count": len(bindings),
                    "status": "PASS",
                }
            )
            for artifact in (metadata_path, step_path):
                manifest_entries.append(
                    {
                        "path": artifact.relative_to(output_root).as_posix(),
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                )
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
            results.append({**failure, "status": "HOLD"})
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        print(f"[{episode_index + 1}/{len(episodes)}] {parent_key}: {rows_written} rows", flush=True)

    manifest_path = output_root / "c2g_clean_collection_input_manifest.jsonl"
    manifest_entries = sorted(manifest_entries, key=lambda row: row["path"])
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_entries),
        encoding="utf-8",
    )
    report = {
        "gate": "C2G_CLEAN_WINDOW_COLLECTION",
        "status": "PASS_CLEAN_COLLECTION",
        "schema": COLLECTION_SCHEMA,
        "event_tracking_schema": EVENT_TRACKING_SCHEMA,
        "episode_count": len(results),
        "results": results,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "openvla_clean_inference_runs": len(results),
        "libero_clean_rollouts": len(results),
        "attacks_launched": 0,
        "attack_outcomes_read": False,
        **provenance,
    }
    write_json(output_root / "c2g_clean_collection_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
