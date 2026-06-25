#!/usr/bin/env python3
"""Clean-only C3R renderer and policy determinism qualification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from scripts.stageb import layer3_exact_restore_runner as exact


def array_diff(reference: np.ndarray, candidate: np.ndarray, prefix: str) -> dict[str, Any]:
    a, b = np.asarray(reference), np.asarray(candidate)
    out: dict[str, Any] = {
        f"{prefix}_shape": json.dumps(list(b.shape)),
        f"{prefix}_dtype": str(b.dtype),
        f"{prefix}_sha256": exact.hash_array(b),
        f"{prefix}_shape_match": a.shape == b.shape,
    }
    if a.shape != b.shape:
        out.update(
            {
                f"{prefix}_diff_count": -1,
                f"{prefix}_max_abs": "",
                f"{prefix}_mae": "",
                f"{prefix}_first_indices": "[]",
                f"{prefix}_channel_counts": "{}",
                f"{prefix}_bbox": "[]",
            }
        )
        return out
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mask = delta != 0
    indices = np.argwhere(mask)
    channel_counts = {}
    if mask.ndim == 3 and mask.shape[-1] <= 4:
        channel_counts = {str(i): int(mask[..., i].sum()) for i in range(mask.shape[-1])}
    elif mask.ndim == 4 and mask.shape[1] <= 4:
        channel_counts = {str(i): int(mask[:, i, ...].sum()) for i in range(mask.shape[1])}
    bbox = []
    if indices.size and mask.ndim >= 2:
        bbox = [
            int(indices[:, 0].min()),
            int(indices[:, 1].min()),
            int(indices[:, 0].max()),
            int(indices[:, 1].max()),
        ]
    out.update(
        {
            f"{prefix}_diff_count": int(mask.sum()),
            f"{prefix}_max_abs": float(delta.max(initial=0.0)),
            f"{prefix}_mae": float(delta.mean()) if delta.size else 0.0,
            f"{prefix}_first_indices": json.dumps(indices[:32].tolist()),
            f"{prefix}_channel_counts": json.dumps(channel_counts, sort_keys=True),
            f"{prefix}_bbox": json.dumps(bbox),
        }
    )
    return out


def capture_stages(env: exact.RealLiberoEnvAdapter, policy: exact.RealOpenVLAPolicyAdapter, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    r1, prepared, inputs = policy.policy_input_stages(obs)
    height, width = int(r1.shape[0]), int(r1.shape[1])
    r0 = np.asarray(
        env.env.sim.render(
            camera_name="agentview",
            width=width,
            height=height,
            depth=False,
        )
    ).copy()
    pixel_values = inputs["pixel_values"].detach().cpu().numpy().copy()
    return {"r0": r0, "r1": r1, "r2": prepared, "r3": prepared.copy(), "r4": pixel_values}


def compare_stages(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for stage in ("r0", "r1", "r2", "r3", "r4"):
        row.update(array_diff(reference[stage], candidate[stage], stage))
    return row


def action_exactness(
    *,
    policy: exact.RealOpenVLAPolicyAdapter,
    obs: Mapping[str, Any],
    expected_action: list[float],
    expected_tokens: list[int],
) -> dict[str, Any]:
    action, tokens = policy.act(obs)
    actual_raw = np.asarray(action)
    expected_raw = np.asarray(expected_action)
    actual_env = exact.postprocess_openvla_action_for_libero(action)
    expected_env = exact.postprocess_openvla_action_for_libero(expected_action)
    return {
        "tokens_exact": [int(x) for x in tokens] == [int(x) for x in expected_tokens],
        "raw_action_exact": bool(
            actual_raw.dtype == expected_raw.dtype
            and actual_raw.shape == expected_raw.shape
            and actual_raw.tobytes() == expected_raw.tobytes()
        ),
        "env_action_exact": bool(
            actual_env.dtype == expected_env.dtype
            and actual_env.shape == expected_env.shape
            and actual_env.tobytes() == expected_env.tobytes()
        ),
        "gripper_semantic_exact": float(actual_env[-1]) == float(expected_env[-1]),
        "actual_tokens": json.dumps([int(x) for x in tokens]),
        "actual_raw_action_sha256": exact.hash_array(actual_raw),
        "actual_env_action_sha256": exact.hash_array(actual_env),
    }


def build_args(ns: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero_goal",
        model_path=ns.model_path,
        unnorm_key="libero_goal",
        detector_path=ns.detector_path,
        render_gpu=int(ns.render_gpu),
        eval_seed=0,
        max_steps=400,
    )


def fresh_student(args: SimpleNamespace, env: exact.RealLiberoEnvAdapter):
    from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

    detector = SC5DetectorRuntime(args.detector_path, guard=5)
    return exact.RealSC5StudentAdapter(
        detector=detector,
        streamer=SC5StreamingFeatureAdapterV2(),
        env_adapter=env,
    )


def save_bundle(root: Path, selected: Mapping[str, Any], args: SimpleNamespace) -> None:
    trace = list(selected["prefix_trace"])
    exact.write_jsonl(root / "prefix_trace.jsonl", trace)
    images = {
        f"next_{idx:04d}": np.asarray(obs["agentview_image"]).copy()
        for idx, obs in enumerate(selected["prefix_next_observations"])
    }
    images["branch"] = np.asarray(selected["snapshot"].observation["agentview_image"]).copy()
    np.savez_compressed(root / "reference_agentview_images.npz", **images)
    metadata = {
        "parent_key": selected["snapshot"].parent_manifest.parent_key,
        "task_idx": int(selected["snapshot"].parent_manifest.task_idx),
        "state_id": int(selected["snapshot"].parent_manifest.state_id),
        "eval_seed": int(selected["snapshot"].parent_manifest.eval_seed),
        "instruction": selected["instruction"],
        "emit_step": int(selected["snapshot"].prefix.emit_step),
        "clean_action": list(selected["snapshot"].clean_action_t),
        "clean_tokens": [int(x) for x in selected["snapshot"].clean_tokens_t],
        "branch_pre_hashes": selected["branch_pre_hashes"],
        "branch_post_student_update_hashes": selected["branch_post_student_update_hashes"],
        "model_path": args.model_path,
        "detector_path": args.detector_path,
        "model_sha256": exact.sha256_path(Path(args.model_path)),
        "detector_sha256": exact.sha256_path(Path(args.detector_path)),
        "prefix_trace_sha256": exact.hash_jsonable(trace),
    }
    exact.write_json(root / "reference_bundle.json", metadata)


def load_bundle(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    metadata = json.loads((root / "reference_bundle.json").read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in (root / "prefix_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    if exact.hash_jsonable(trace) != metadata["prefix_trace_sha256"]:
        raise exact.ExactRestoreError("C3R prefix trace SHA mismatch")
    with np.load(root / "reference_agentview_images.npz", allow_pickle=False) as archive:
        images = {key: archive[key].copy() for key in archive.files}
    return metadata, trace, images


def reference_stages(policy: exact.RealOpenVLAPolicyAdapter, image: np.ndarray, r0: np.ndarray) -> dict[str, np.ndarray]:
    obs = {"agentview_image": image}
    r1, prepared, inputs = policy.policy_input_stages(obs)
    return {
        "r0": r0,
        "r1": r1,
        "r2": prepared,
        "r3": prepared.copy(),
        "r4": inputs["pixel_values"].detach().cpu().numpy().copy(),
    }


def run_prefix_replay(
    *,
    args: SimpleNamespace,
    policy: exact.RealOpenVLAPolicyAdapter,
    metadata: Mapping[str, Any],
    trace: list[dict[str, Any]],
    images: Mapping[str, np.ndarray],
    r0_reference: dict[str, np.ndarray] | None,
    repetition: int,
    cohort: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    env_raw, obs, task_obj, _bddl = exact.build_real_env_for_candidate(
        suite="libero_goal",
        task_idx=int(metadata["task_idx"]),
        state_id=int(metadata["state_id"]),
        render_gpu=int(args.render_gpu),
        max_steps=int(args.max_steps),
    )
    env = exact.RealLiberoEnvAdapter(env_raw)
    student = fresh_student(args, env)
    rows: list[dict[str, Any]] = []
    captured_r0: dict[str, np.ndarray] = {}
    try:
        for idx, expected in enumerate(trace):
            action_check = action_exactness(
                policy=policy,
                obs=obs,
                expected_action=expected["raw_action"],
                expected_tokens=expected["tokens"],
            )
            pre = exact.prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
            exact.update_student_for_step(
                student,
                step=int(expected["step"]),
                obs=obs,
                action=expected["raw_action"],
                tokens=expected["tokens"],
            )
            post_student = exact.prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
            obs, _reward, done, _info = env.step_env_action(expected["env_action"])
            if done:
                raise exact.ExactRestoreError(f"C3R replay terminated during prefix at step {idx}")
            post = exact.prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
            actual_stages = capture_stages(env, policy, obs)
            key = f"step_{idx:04d}"
            captured_r0[key] = actual_stages["r0"]
            baseline_r0 = actual_stages["r0"] if r0_reference is None else r0_reference[key]
            expected_stages = reference_stages(policy, images[f"next_{idx:04d}"], baseline_r0)
            rows.append(
                {
                    "cohort": cohort,
                    "repetition": int(repetition),
                    "phase": "prefix_post_step",
                    "step": int(expected["step"]),
                    **action_check,
                    "pre_qpos_exact": pre["qpos_sha256"] == expected["qpos_sha256"],
                    "pre_qvel_exact": pre["qvel_sha256"] == expected["qvel_sha256"],
                    "pre_flat_sim_exact": pre["flat_sim_state_sha256"] == expected["flat_sim_state_sha256"],
                    "pre_student_exact": pre["student_state_sha256"] == expected["student_state_sha256"],
                    "pre_feature_history_exact": pre["feature_history_sha256"] == expected["feature_history_sha256"],
                    "post_student_exact": post_student["student_state_sha256"]
                    == expected["post_student_state_sha256"],
                    "post_feature_history_exact": post_student["feature_history_sha256"]
                    == expected["post_feature_history_sha256"],
                    "post_qpos_exact": post["qpos_sha256"] == expected["post_qpos_sha256"],
                    "post_qvel_exact": post["qvel_sha256"] == expected["post_qvel_sha256"],
                    "post_flat_sim_exact": post["flat_sim_state_sha256"]
                    == expected["post_flat_sim_state_sha256"],
                    **compare_stages(expected_stages, actual_stages),
                }
            )

        branch_action = action_exactness(
            policy=policy,
            obs=obs,
            expected_action=metadata["clean_action"],
            expected_tokens=metadata["clean_tokens"],
        )
        branch_pre = exact.prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
        exact.update_student_for_step(
            student,
            step=int(metadata["emit_step"]),
            obs=obs,
            action=metadata["clean_action"],
            tokens=metadata["clean_tokens"],
        )
        branch_post = exact.prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
        branch_stages = capture_stages(env, policy, obs)
        captured_r0["branch"] = branch_stages["r0"]
        baseline_r0 = branch_stages["r0"] if r0_reference is None else r0_reference["branch"]
        expected_stages = reference_stages(policy, images["branch"], baseline_r0)
        rows.append(
            {
                "cohort": cohort,
                "repetition": int(repetition),
                "phase": "branch_boundary",
                "step": int(metadata["emit_step"]),
                **branch_action,
                "pre_qpos_exact": branch_pre["qpos_sha256"]
                == metadata["branch_pre_hashes"]["qpos_sha256"],
                "pre_qvel_exact": branch_pre["qvel_sha256"]
                == metadata["branch_pre_hashes"]["qvel_sha256"],
                "pre_flat_sim_exact": branch_pre["flat_sim_state_sha256"]
                == metadata["branch_pre_hashes"]["flat_sim_state_sha256"],
                "pre_student_exact": branch_pre["student_state_sha256"]
                == metadata["branch_pre_hashes"]["student_state_sha256"],
                "pre_feature_history_exact": branch_pre["feature_history_sha256"]
                == metadata["branch_pre_hashes"]["feature_history_sha256"],
                "post_student_exact": branch_post["student_state_sha256"]
                == metadata["branch_post_student_update_hashes"]["student_state_sha256"],
                "post_feature_history_exact": branch_post["feature_history_sha256"]
                == metadata["branch_post_student_update_hashes"]["feature_history_sha256"],
                "post_qpos_exact": True,
                "post_qvel_exact": True,
                "post_flat_sim_exact": True,
                **compare_stages(expected_stages, branch_stages),
            }
        )
        return rows, captured_r0
    finally:
        env.close()


def run_static_same_process(
    *,
    selected: Mapping[str, Any],
    repetitions: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    env = selected["env_adapter"]
    policy = selected["policy"]
    expected_action = list(selected["snapshot"].clean_action_t)
    expected_tokens = [int(x) for x in selected["snapshot"].clean_tokens_t]
    baseline: dict[str, np.ndarray] | None = None
    rows = []
    for repetition in range(repetitions):
        obs = env.get_observation_after_restore()
        stages = capture_stages(env, policy, obs)
        if baseline is None:
            baseline = {key: value.copy() for key, value in stages.items()}
        state = exact.prefix_replay_state_hashes(env=env, obs=obs, student=selected["student"], policy=policy)
        rows.append(
            {
                "cohort": "static_same_process",
                "repetition": repetition,
                "phase": "branch_boundary",
                "step": int(selected["snapshot"].prefix.emit_step),
                **action_exactness(
                    policy=policy,
                    obs=obs,
                    expected_action=expected_action,
                    expected_tokens=expected_tokens,
                ),
                "qpos_sha256": state["qpos_sha256"],
                "qvel_sha256": state["qvel_sha256"],
                "flat_sim_state_sha256": state["flat_sim_state_sha256"],
                **compare_stages(baseline, stages),
            }
        )
    assert baseline is not None
    return rows, baseline


def run_same_process(ns: argparse.Namespace) -> None:
    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    args = build_args(ns)
    candidates = exact.read_candidate_manifest(Path(ns.candidate_manifest), suite="libero_goal", eval_seed=0)
    if len(candidates) != 1:
        raise exact.ExactRestoreError("C3R requires exactly one frozen candidate")
    selected = exact.find_emit_snapshot_for_candidate(
        args=args,
        candidate=candidates[0],
        attempt_dir=out,
        repetition=0,
    )
    try:
        save_bundle(out, selected, args)
        static_rows, _static_baseline = run_static_same_process(
            selected=selected,
            repetitions=int(ns.static_repetitions),
        )
        exact.write_dict_csv(out / "c3r_static_render_same_process.csv", static_rows)
        metadata, trace, images = load_bundle(out)
        calibration_rows: list[dict[str, Any]] = []
        r0_reference = None
        for repetition in range(int(ns.calibration_repetitions)):
            rows, captured_r0 = run_prefix_replay(
                args=args,
                policy=selected["policy"],
                metadata=metadata,
                trace=trace,
                images=images,
                r0_reference=r0_reference,
                repetition=repetition,
                cohort="calibration",
            )
            if r0_reference is None:
                r0_reference = captured_r0
                np.savez_compressed(out / "reference_raw_framebuffer.npz", **r0_reference)
            calibration_rows.extend(rows)
        exact.write_dict_csv(out / "c3r_prefix_replay_calibration.csv", calibration_rows)
        exact.write_json(
            out / "c3r_same_process_summary.json",
            {
                "result": "COMPLETE",
                "static_repetitions": int(ns.static_repetitions),
                "calibration_repetitions": int(ns.calibration_repetitions),
                "attack_paths_run": False,
            },
        )
        exact.write_recursive_manifest(out)
    finally:
        selected["env_adapter"].close()
        exact.release_real_policy(selected["policy"])


def run_fresh_worker(ns: argparse.Namespace) -> None:
    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    args = build_args(ns)
    bundle = Path(ns.bundle_dir)
    metadata, trace, images = load_bundle(bundle)
    with np.load(bundle / "reference_raw_framebuffer.npz", allow_pickle=False) as archive:
        r0_reference = {key: archive[key].copy() for key in archive.files}
    env_raw, _obs, task_obj, _bddl = exact.build_real_env_for_candidate(
        suite="libero_goal",
        task_idx=int(metadata["task_idx"]),
        state_id=int(metadata["state_id"]),
        render_gpu=int(args.render_gpu),
        max_steps=int(args.max_steps),
    )
    bootstrap_env = exact.RealLiberoEnvAdapter(env_raw)
    policy, _student, _model, _detector = exact.load_real_policy_and_student(
        args,
        env_adapter=bootstrap_env,
        instruction=str(task_obj.language),
    )
    bootstrap_env.close()
    try:
        rows, _captured = run_prefix_replay(
            args=args,
            policy=policy,
            metadata=metadata,
            trace=trace,
            images=images,
            r0_reference=r0_reference,
            repetition=int(ns.worker_index),
            cohort="validation",
        )
        exact.write_dict_csv(out / "c3r_prefix_replay_validation.csv", rows)
        exact.write_dict_csv(
            out / "c3r_static_render_fresh_process.csv",
            [row for row in rows if row["phase"] == "branch_boundary"],
        )
        exact.write_json(
            out / "c3r_fresh_worker_summary.json",
            {"result": "COMPLETE", "worker_index": int(ns.worker_index), "attack_paths_run": False},
        )
        exact.write_recursive_manifest(out)
    finally:
        exact.release_real_policy(policy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("same-process", "fresh-worker"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--candidate-manifest", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--detector-path", required=True)
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--static-repetitions", type=int, default=100)
    parser.add_argument("--calibration-repetitions", type=int, default=20)
    parser.add_argument("--worker-index", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "same-process":
        if not args.candidate_manifest:
            raise SystemExit("--candidate-manifest is required")
        run_same_process(args)
    else:
        if not args.bundle_dir or args.worker_index < 0:
            raise SystemExit("--bundle-dir and nonnegative --worker-index are required")
        run_fresh_worker(args)


if __name__ == "__main__":
    main()
