#!/usr/bin/env python3
"""Materialize a train-ready C2g clean-window dataset from clean rollout artifacts.

The materializer is deterministic and fail-closed.  It reads only clean episode
metadata, clean step records, and RGB frames.  Clean Teacher-v2 labels are built
from the privileged fields already captured in those clean records; attack or
post-intervention fields are rejected by the label builder.

The output NPZ contains causal fixed-length sequences, per-head targets/masks,
episode identities, deterministic splits, and provenance hashes.  OpenVLA/SigLIP
embedding extraction is reused from the mature C2f materializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from tools.multisuite_detector.c2g_clean_dataset_adapter import (
    MODEL_TARGET_MAP,
    derive_episode_fully_known_negative,
    teacher_row_to_model_targets,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)
from tools.multisuite_detector.materialize_c2f_frozen_embeddings import Embedder

DATASET_SCHEMA_VERSION = "c2g.clean_window_dataset.2026-07-10.v1"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HEADS = tuple(MODEL_TARGET_MAP) + ("grounding_confidence",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def ordered_unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if len(set(steps)) != len(steps):
        duplicates = sorted(step for step in set(steps) if steps.count(step) > 1)
        raise ValueError(f"duplicate step ids: {duplicates[:20]}")
    return ordered


def discover_episodes(root: Path) -> list[tuple[Path, Path]]:
    candidates = sorted(root.rglob("step_records.jsonl"))
    pairs: list[tuple[Path, Path]] = []
    for step_path in candidates:
        metadata_path = step_path.with_name("episode_metadata.json")
        if metadata_path.is_file():
            pairs.append((metadata_path, step_path))
    if not pairs:
        raise RuntimeError(f"no episode_metadata.json + step_records.jsonl pairs under {root}")
    return pairs


def _finite_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have length {length}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def policy_vector(row: Mapping[str, Any]) -> np.ndarray:
    for key in ("clean_policy_intent_9d", "clean_policy_features", "policy_intent"):
        if key in row and row[key] is not None:
            return _finite_vector(row[key], length=len(CLEAN_POLICY_FEATURE_NAMES), name=key)
    if all(name in row and row[name] is not None for name in CLEAN_POLICY_FEATURE_NAMES):
        return _finite_vector(
            [row[name] for name in CLEAN_POLICY_FEATURE_NAMES],
            length=len(CLEAN_POLICY_FEATURE_NAMES),
            name="named clean policy features",
        )
    raise KeyError("clean policy-intent features are absent")


def episode_identity(metadata: Mapping[str, Any], metadata_path: Path) -> tuple[str, str, int]:
    suite = str(metadata.get("suite", ""))
    task_index = int(metadata.get("task_index", metadata.get("task_id", -1)))
    if suite not in SUITES or task_index < 0:
        raise ValueError(f"invalid suite/task identity in {metadata_path}")
    key = str(metadata.get("episode_key") or metadata.get("parent_key") or metadata_path.parent.as_posix())
    return key, suite, task_index


def stable_bucket(text: str, seed: int, modulo: int = 10000) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def split_for_episode(
    episode_key: str,
    suite: str,
    task_index: int,
    *,
    mode: str,
    seed: int,
    held_out_task: str,
    held_out_suite: str,
    val_fraction: float,
    test_fraction: float,
) -> str:
    identity = f"{suite}:{task_index}"
    bucket = stable_bucket(episode_key, seed) / 10000.0
    if mode == "within_task":
        if bucket < test_fraction:
            return "test"
        if bucket < test_fraction + val_fraction:
            return "val"
        return "train"
    if mode == "leave_one_task_out":
        if not held_out_task:
            raise ValueError("--held-out-task is required for leave_one_task_out")
        if identity == held_out_task:
            return "test"
        return "val" if bucket < val_fraction else "train"
    if mode == "leave_one_suite_out":
        if held_out_suite not in SUITES:
            raise ValueError("--held-out-suite must name a LIBERO suite")
        if suite == held_out_suite:
            return "test"
        return "val" if bucket < val_fraction else "train"
    raise ValueError(f"unknown split mode: {mode}")


def _relative_rgb_path(episode_dir: Path, row: Mapping[str, Any]) -> Path:
    value = str(row.get("rgb_path", "")).strip()
    if not value:
        raise KeyError("rgb_path is required for visual materialization")
    path = (episode_dir / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sequence_labels(
    labels: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = {name: [] for name in HEADS}
    masks = {name: [] for name in HEADS}
    for row in labels[start:end]:
        converted = teacher_row_to_model_targets(row)
        for name in HEADS:
            targets[name].append(float(converted["targets"][name]))
            masks[name].append(bool(converted["masks"][name]))
    return (
        {name: np.asarray(values, dtype=np.float32) for name, values in targets.items()},
        {name: np.asarray(values, dtype=np.bool_) for name, values in masks.items()},
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.window <= 0:
        raise ValueError("window must be positive")
    if not (0.0 <= args.val_fraction < 1.0 and 0.0 <= args.test_fraction < 1.0):
        raise ValueError("split fractions must be in [0,1)")
    if args.val_fraction + args.test_fraction >= 1.0 and args.split_mode == "within_task":
        raise ValueError("within-task val+test fractions must be < 1")

    embedder = Embedder(
        args.backend,
        args.device,
        args.model_name,
        args.embedding_dim,
        openvla_model_path=args.openvla_model_path,
    )
    thresholds = CleanTeacherThresholds(
        burst_length=args.burst_length,
        contact_persistence_steps=args.contact_persistence_steps,
        relative_lift_threshold=args.relative_lift_threshold,
        target_progress_threshold=args.target_progress_threshold,
        grounding_confidence_threshold=args.grounding_confidence_threshold,
    )

    x_proprio: list[np.ndarray] = []
    x_policy: list[np.ndarray] = []
    x_visual: list[np.ndarray] = []
    x_language: list[np.ndarray] = []
    targets: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}
    suite_rows: list[str] = []
    task_rows: list[int] = []
    episode_rows: list[str] = []
    step_rows: list[int] = []
    split_rows: list[str] = []
    fully_negative_rows: list[bool] = []
    sample_weights: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    text_cache: dict[str, np.ndarray] = {}
    image_cache: dict[str, np.ndarray] = {}
    episode_count = 0

    pairs = discover_episodes(root)
    if args.max_episodes > 0:
        pairs = pairs[: args.max_episodes]

    for metadata_path, step_path in pairs:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            rows = ordered_unique_rows(read_jsonl(step_path))
            episode_key, suite, task_index = episode_identity(metadata, metadata_path)
            labels = build_clean_teacher_episode(rows, metadata, thresholds=thresholds)
            if len(labels) != len(rows):
                raise RuntimeError("teacher label count differs from source step count")
            episode_negative = derive_episode_fully_known_negative(labels).get(episode_key, False)
            split = split_for_episode(
                episode_key,
                suite,
                task_index,
                mode=args.split_mode,
                seed=args.seed,
                held_out_task=args.held_out_task,
                held_out_suite=args.held_out_suite,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
            language_text = str(metadata.get("task_language") or rows[0].get("task_language", "")).strip()
            if not language_text:
                raise ValueError("task language is empty")
            if language_text not in text_cache:
                text_cache[language_text] = embedder.encode_text(language_text).astype(np.float32)
            language_embedding = text_cache[language_text]

            proprio = np.asarray([row["features_25d"] for row in rows], dtype=np.float32)
            if proprio.ndim != 2 or proprio.shape[1] != 25 or not np.isfinite(proprio).all():
                raise ValueError("features_25d must form a finite [time,25] array")
            if args.use_policy_intent:
                policy = np.asarray([policy_vector(row) for row in rows], dtype=np.float32)
            else:
                policy = np.zeros((len(rows), len(CLEAN_POLICY_FEATURE_NAMES)), dtype=np.float32)

            if len(rows) < args.window:
                continue
            episode_dir = step_path.parent
            for end in range(args.window - 1, len(rows)):
                start = end - args.window + 1
                label_targets, label_masks = _sequence_labels(labels, start, end + 1)
                if args.drop_all_unknown and not label_masks["critical_window"].any():
                    continue
                current = rows[end]
                if args.use_visual:
                    rgb_path = _relative_rgb_path(episode_dir, current)
                    cache_key = rgb_path.as_posix()
                    if cache_key not in image_cache:
                        image_cache[cache_key] = embedder.encode_image(rgb_path).astype(np.float32)
                    visual_embedding = image_cache[cache_key]
                else:
                    visual_embedding = np.zeros((args.embedding_dim,), dtype=np.float32)

                x_proprio.append(proprio[start : end + 1])
                x_policy.append(policy[start : end + 1])
                x_visual.append(visual_embedding)
                x_language.append(language_embedding)
                for name in HEADS:
                    targets[name].append(label_targets[name])
                    masks[name].append(label_masks[name])
                suite_rows.append(suite)
                task_rows.append(task_index)
                episode_rows.append(episode_key)
                step_rows.append(int(current["step"]))
                split_rows.append(split)
                fully_negative_rows.append(bool(episode_negative))
                known = label_masks["critical_window"].astype(np.float32)
                positive = label_targets["critical_window"] > 0.5
                weight = np.where(positive, args.positive_weight, 1.0).astype(np.float32) * known
                sample_weights.append(weight)
            episode_count += 1

            for artifact in (metadata_path, step_path):
                manifest.append(
                    {
                        "path": artifact.relative_to(root).as_posix(),
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                )
        except Exception as exc:
            errors.append({"path": str(step_path), "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise

    if errors and args.require_zero_errors:
        raise RuntimeError(f"materialization encountered {len(errors)} episode errors; first={errors[0]}")
    if not x_proprio:
        raise RuntimeError("no trainable windows were materialized")

    dataset_path = output_dir / f"c2g_clean_window_w{args.window:02d}_{args.backend}_{args.split_mode}.npz"
    payload: dict[str, Any] = {
        "schema_version": np.asarray(DATASET_SCHEMA_VERSION),
        "X_proprio": np.asarray(x_proprio, dtype=np.float32),
        "X_policy": np.asarray(x_policy, dtype=np.float32),
        "X_visual": np.asarray(x_visual, dtype=np.float16 if args.backend == "openvla_siglip" else np.float32),
        "X_language": np.asarray(x_language, dtype=np.float16 if args.backend == "openvla_siglip" else np.float32),
        "suite": np.asarray(suite_rows),
        "task_index": np.asarray(task_rows, dtype=np.int64),
        "episode_key": np.asarray(episode_rows),
        "step": np.asarray(step_rows, dtype=np.int64),
        "split": np.asarray(split_rows),
        "episode_fully_known_negative": np.asarray(fully_negative_rows, dtype=np.bool_),
        "sample_weight": np.asarray(sample_weights, dtype=np.float32),
        "feature_names_policy": np.asarray(CLEAN_POLICY_FEATURE_NAMES),
    }
    for name in HEADS:
        payload[f"y_{name}"] = np.asarray(targets[name], dtype=np.float32)
        payload[f"m_{name}"] = np.asarray(masks[name], dtype=np.bool_)
    np.savez_compressed(dataset_path, **payload)

    canonical_manifest = sorted(manifest, key=lambda item: item["path"])
    manifest_path = output_dir / "c2g_clean_window_input_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in canonical_manifest),
        encoding="utf-8",
    )
    errors_path = output_dir / "c2g_clean_window_materialization_errors.jsonl"
    errors_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in errors),
        encoding="utf-8",
    )
    split_counts = {name: int(sum(value == name for value in split_rows)) for name in ("train", "val", "test")}
    known_count = int(np.asarray(masks["critical_window"], dtype=np.bool_).sum())
    positive_count = int(
        (np.asarray(targets["critical_window"]) * np.asarray(masks["critical_window"], dtype=np.float32)).sum()
    )
    report = {
        "gate": "C2G_CLEAN_WINDOW_DATASET_MATERIALIZATION",
        "status": "PASS_MATERIALIZED" if not errors else "PASS_WITH_RECORDED_ERRORS",
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "error_ledger_path": str(errors_path),
        "n_windows": len(x_proprio),
        "n_episodes_processed": episode_count,
        "n_episode_errors": len(errors),
        "window": args.window,
        "burst_length": args.burst_length,
        "split_mode": args.split_mode,
        "held_out_task": args.held_out_task,
        "held_out_suite": args.held_out_suite,
        "split_counts": split_counts,
        "known_critical_labels": known_count,
        "positive_critical_labels": positive_count,
        "visual_dim": int(np.asarray(x_visual).shape[-1]),
        "language_dim": int(np.asarray(x_language).shape[-1]),
        "policy_intent_dim": len(CLEAN_POLICY_FEATURE_NAMES),
        "backend": args.backend,
        "openvla_model_path": args.openvla_model_path if args.backend == "openvla_siglip" else "",
        "use_visual": bool(args.use_visual),
        "use_policy_intent": bool(args.use_policy_intent),
        "seed": args.seed,
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "boundaries": {
            "clean_only": True,
            "attack_outcome_read": False,
            "counterfactual_read": False,
            "libero_rollouts_launched": 0,
            "attacks_launched": 0,
        },
    }
    write_json(output_dir / "c2g_clean_window_materialization_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--contact-persistence-steps", type=int, default=2)
    parser.add_argument("--relative-lift-threshold", type=float, default=0.015)
    parser.add_argument("--target-progress-threshold", type=float, default=0.01)
    parser.add_argument("--grounding-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--backend", choices=("stats", "clip", "openvla_siglip"), default="stats")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--openvla-model-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--use-visual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-policy-intent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-all-unknown", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--split-mode", choices=("within_task", "leave_one_task_out", "leave_one_suite_out"), default="within_task")
    parser.add_argument("--held-out-task", default="", help="suite:task_index for leave_one_task_out")
    parser.add_argument("--held-out-suite", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--require-zero-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = materialize(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
