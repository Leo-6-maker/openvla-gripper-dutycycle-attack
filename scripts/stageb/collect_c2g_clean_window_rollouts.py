#!/usr/bin/env python3
"""Collect observation-rich clean rollouts for C2g Detector-v2 training.

This is a CLEAN-only collector.  It records deployment-visible student inputs and
clean privileged Teacher-v2 evidence, but never runs a visual attack and never
reads attacked outcomes.  Outputs are directly consumable by
materialize_c2g_clean_window_dataset.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES, summarize_clean_gripper_logits
from gripper_attack.c2g_clean_window_runtime import derive_gripper_token_semantics
from gripper_attack.c2g_teacher_v2_contact_identity import canonicalize_mujoco_name
from gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets

COLLECTION_SCHEMA = "c2g.clean_window_collection.2026-07-10.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("episodes", value) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain a list of objects")
    return [dict(row) for row in rows]


def git_provenance(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"expected commit {expected_commit}, got {commit}")
    if status.strip():
        raise RuntimeError("collector requires a clean worktree")
    return {"git_commit": commit, "git_clean": True}


def visible_gpu_id() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def model_names(model: Any, kind: str) -> list[str]:
    count_name = {"body": "nbody", "site": "nsite", "joint": "njnt", "geom": "ngeom"}[kind]
    id_name = f"{kind}_id2name"
    count = int(getattr(model, count_name, 0))
    fn = getattr(model, id_name, None)
    if fn is None:
        values = getattr(model, f"{kind}_names", ()) or ()
        return [str(value or "") for value in values]
    return [str(fn(index) or "") for index in range(count)]


def contact_pairs(env: Any) -> list[list[str]]:
    model = env.sim.model
    data = env.sim.data
    output: list[list[str]] = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        left = str(model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}")
        right = str(model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}")
        output.append([left, right])
    return output


def _canonical_candidates(names: Sequence[str], entity: str) -> list[int]:
    target = canonicalize_mujoco_name(entity)
    exact = [index for index, name in enumerate(names) if canonicalize_mujoco_name(name) == target]
    if exact:
        return exact
    return [
        index for index, name in enumerate(names)
        if canonicalize_mujoco_name(name).startswith(target + "_")
        or target.startswith(canonicalize_mujoco_name(name) + "_")
    ]


def entity_position(env: Any, entity: str) -> np.ndarray | None:
    model, data = env.sim.model, env.sim.data
    site_names = model_names(model, "site")
    candidates = _canonical_candidates(site_names, entity)
    if len(candidates) == 1:
        return np.asarray(data.site_xpos[candidates[0]], dtype=np.float32).copy()
    body_names = model_names(model, "body")
    candidates = _canonical_candidates(body_names, entity)
    if len(candidates) == 1:
        return np.asarray(data.body_xpos[candidates[0]], dtype=np.float32).copy()
    return None


def entity_joint_scalar(env: Any, entity: str) -> float | None:
    names = model_names(env.sim.model, "joint")
    candidates = _canonical_candidates(names, entity)
    if not candidates:
        return None
    values: list[float] = []
    for joint_id in candidates:
        try:
            address = int(env.sim.model.jnt_qposadr[joint_id])
            values.append(float(env.sim.data.qpos[address]))
        except Exception:
            continue
    return float(np.mean(values)) if values else None


def target_support_contact(pairs: Sequence[Sequence[str]], target: str, destinations: Sequence[str]) -> bool:
    target_name = canonicalize_mujoco_name(target)
    destination_names = [canonicalize_mujoco_name(value) for value in destinations]
    for first, second in pairs:
        a, b = canonicalize_mujoco_name(first), canonicalize_mujoco_name(second)
        a_target = a == target_name or a.startswith(target_name + "_")
        b_target = b == target_name or b.startswith(target_name + "_")
        a_destination = any(a == value or a.startswith(value + "_") for value in destination_names)
        b_destination = any(b == value or b.startswith(value + "_") for value in destination_names)
        if (a_target and b_destination) or (b_target and a_destination):
            return True
    return False


def policy_features(logits: torch.Tensor, semantics: Mapping[str, Any]) -> list[float]:
    summary = summarize_clean_gripper_logits(
        logits,
        open_token_ids=semantics["open_token_ids"],
        close_token_ids=semantics["close_token_ids"],
    )
    return [float(summary[name].detach().cpu()) for name in CLEAN_POLICY_FEATURE_NAMES]


def save_rgb(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def normalized_rgb(obs: Mapping[str, Any], step: int) -> np.ndarray:
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
        raise RuntimeError(f"blank/malformed RGB at step {step}")
    return image[..., :3].copy()


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
                processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
                model = AutoModelClass.from_pretrained(
                    str(model_path), trust_remote_code=True, local_files_only=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device,
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
            }
            resolution = resolve_task_targets(task_metadata)
            mechanism = infer_clean_mechanism_type(task_metadata, resolution=resolution)
            task_metadata["mechanism_type"] = mechanism
            task_metadata["structured_goal_metadata"] = {
                "target_objects": list(resolution.resolved_target_objects),
                "target_receptacles": list(resolution.resolved_receptacles),
                "target_sites": list(resolution.resolved_sites),
                "target_fixtures": list(resolution.resolved_manipulable_entities),
            }

            env, obs = build_v4_exact_env(str(bddl_path), visible_gpu_id(), args.max_steps, args.dummy_wait)
            obs = env.set_init_state(states[state_id])
            env, obs = apply_dummy_wait(env, obs, args.dummy_wait)
            streamer = SC5StreamingFeatureAdapterV2()
            eef_site = env.sim.model.site_name2id("gripper0_grip_site")
            previous_eef = None
            target_entities = list(resolution.resolved_target_objects or resolution.resolved_manipulable_entities)
            destination_entities = list(resolution.resolved_receptacles) + list(resolution.resolved_sites)
            primary_target = target_entities[0] if len(target_entities) == 1 else ""
            primary_destination = destination_entities[0] if len(destination_entities) == 1 else ""
            grasp_baseline_z: float | None = None
            initial_target_distance: float | None = None
            initial_fixture_joint: float | None = entity_joint_scalar(env, primary_target) if primary_target else None

            with step_path.open("w", encoding="utf-8") as step_handle:
                for step in range(args.max_steps):
                    rgb = normalized_rgb(obs, step)
                    gripper_state = physical_gripper_state(env, obs)
                    qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
                    qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
                    opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
                    action, _, _, generation = decode_with_scores(
                        model, processor, args.device, rgb, task_language, unnorm_key, 8,
                        libero_official_preprocess=False,
                        libero_preprocess_backend="official_pil_lanczos",
                        center_crop=True, resize_size=224, drop_attention_mask=True,
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
                        step_id=step, raw_gripper=float(action[-1]), env_gripper=float(env_action[-1]),
                        gripper_qpos=qpos_sum, gripper_opening_proxy=opening,
                        eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                        eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                        action_dx=float(env_action[0]), action_dy=float(env_action[1]),
                        action_dz=float(env_action[2]), action_gripper=float(action[-1]),
                    )
                    features = list(stream["features"].values())
                    if len(features) != 25 or not np.isfinite(np.asarray(features, dtype=np.float32)).all():
                        raise RuntimeError(f"invalid 25D features at step {step}")
                    pairs = contact_pairs(env)
                    target_position = entity_position(env, primary_target) if primary_target else None
                    destination_position = entity_position(env, primary_destination) if primary_destination else None
                    current_distance = None
                    if target_position is not None and destination_position is not None:
                        current_distance = float(np.linalg.norm(target_position - destination_position))
                    if initial_target_distance is None and current_distance is not None:
                        initial_target_distance = current_distance
                    clean_close = float(action[-1]) < 0.5
                    target_contact = any(
                        canonicalize_mujoco_name(primary_target) in {
                            canonicalize_mujoco_name(first), canonicalize_mujoco_name(second)
                        }
                        or canonicalize_mujoco_name(first).startswith(canonicalize_mujoco_name(primary_target) + "_")
                        or canonicalize_mujoco_name(second).startswith(canonicalize_mujoco_name(primary_target) + "_")
                        for first, second in pairs
                    ) if primary_target else False
                    if clean_close and target_contact and target_position is not None and grasp_baseline_z is None:
                        grasp_baseline_z = float(target_position[2])
                    relative_lift = (
                        float(target_position[2]) - grasp_baseline_z
                        if target_position is not None and grasp_baseline_z is not None else None
                    )
                    progress = (
                        initial_target_distance - current_distance
                        if initial_target_distance is not None and current_distance is not None else None
                    )
                    fixture_joint = entity_joint_scalar(env, primary_target) if primary_target else None
                    fixture_motion = (
                        abs(fixture_joint - initial_fixture_joint)
                        if fixture_joint is not None and initial_fixture_joint is not None else None
                    )
                    manipulation_active = bool(
                        (relative_lift is not None and relative_lift >= args.relative_lift_threshold)
                        or (progress is not None and progress >= args.progress_threshold)
                        or (fixture_motion is not None and fixture_motion >= args.fixture_motion_threshold)
                    )
                    near_target = bool(current_distance is not None and current_distance <= args.near_target_threshold)
                    supported = bool(primary_target and destination_entities and target_support_contact(pairs, primary_target, destination_entities))
                    rgb_relative = f"rgb/frame_{step:06d}.png"
                    save_rgb(rgb_dir / f"frame_{step:06d}.png", rgb)
                    row = {
                        "step": step,
                        "rgb_path": rgb_relative,
                        "task_language": task_language,
                        "features_25d": [float(value) for value in features],
                        "clean_policy_intent_9d": policy,
                        **{name: policy[index] for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)},
                        "clean_gripper_command": float(action[-1]),
                        "clean_close_intent": clean_close,
                        "mujoco_contact_pairs": pairs,
                        "object_relative_lift": relative_lift,
                        "target_distance_decrease": progress,
                        "constrained_manipulation_active": bool(fixture_motion is not None and fixture_motion >= args.fixture_motion_threshold),
                        "manipulation_progress_active": manipulation_active,
                        "near_target": near_target,
                        "supported_at_target": supported,
                        "release_safe": bool(near_target and supported),
                        "target_object_position": None if target_position is None else target_position.astype(float).tolist(),
                        "target_destination_position": None if destination_position is None else destination_position.astype(float).tolist(),
                        "fixture_joint_motion": fixture_motion,
                    }
                    step_handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    rows_written += 1
                    obs, reward, done, info = env.step(env_action)
                    last_info = dict(info or {})
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
                    if path.is_file() and path.name in {"config.json", "model.safetensors.index.json", "processor_config.json", "tokenizer.json"}
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
                "student_allowed_modalities": ["rgb", "task_language", "features_25d", "clean_policy_intent_9d"],
                "student_forbidden_modalities": [
                    "mujoco_contact_pairs", "object_relative_lift", "target_distance_decrease",
                    "target_object_position", "target_destination_position", "release_safe",
                    "attack_outcome", "post_intervention",
                ],
                **provenance,
            }
            write_json(metadata_path, metadata)
            results.append({"parent_key": parent_key, "suite": suite, "task_index": task_index, "state_id": state_id, "n_steps": rows_written, "status": "PASS"})
            for artifact in (metadata_path, step_path):
                manifest_entries.append({
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
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_entries), encoding="utf-8")
    report = {
        "gate": "C2G_CLEAN_WINDOW_COLLECTION",
        "status": "PASS_CLEAN_COLLECTION",
        "schema": COLLECTION_SCHEMA,
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
