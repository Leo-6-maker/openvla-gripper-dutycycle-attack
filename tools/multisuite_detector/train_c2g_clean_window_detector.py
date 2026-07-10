#!/usr/bin/env python3
"""Train, evaluate, calibrate, and export the C2g clean-window detector.

This program consumes only the clean dataset produced by
materialize_c2g_clean_window_dataset.py.  Model selection and threshold
calibration use clean labels; no attacked rollout result is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, optim

from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    clean_window_loss,
)
from tools.multisuite_detector.materialize_c2g_clean_window_dataset import (
    DATASET_SCHEMA_VERSION,
    HEADS,
)

CHECKPOINT_SCHEMA_VERSION = "c2g.clean_window_checkpoint.2026-07-10.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def batch_indices(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    schema = str(data.get("schema_version", ""))
    if schema != DATASET_SCHEMA_VERSION:
        raise ValueError(f"dataset schema mismatch: {schema!r}")
    required = {
        "X_proprio", "X_policy", "X_visual", "X_language", "suite", "task_index",
        "episode_key", "step", "split", "episode_fully_known_negative", "sample_weight",
    }
    for name in HEADS:
        required.add(f"y_{name}")
        required.add(f"m_{name}")
    missing = sorted(required - set(data))
    if missing:
        raise KeyError("dataset missing fields: " + ", ".join(missing))
    n = int(data["X_proprio"].shape[0])
    if n <= 0 or any(int(data[key].shape[0]) != n for key in required if key != "schema_version"):
        raise ValueError("dataset arrays have inconsistent sample cardinality")
    if data["X_proprio"].ndim != 3 or data["X_proprio"].shape[-1] != 25:
        raise ValueError("X_proprio must be [sample,time,25]")
    if data["X_policy"].ndim != 3:
        raise ValueError("X_policy must be [sample,time,policy_dim]")
    if data["sample_weight"].shape != data["X_proprio"].shape[:2]:
        raise ValueError("sample_weight must be [sample,time]")
    for name in HEADS:
        expected = data["X_proprio"].shape[:2]
        if data[f"y_{name}"].shape != expected or data[f"m_{name}"].shape != expected:
            raise ValueError(f"{name} target/mask must be [sample,time]")
    for key in ("X_proprio", "X_policy", "X_visual", "X_language", "sample_weight"):
        if not np.isfinite(data[key].astype(np.float32)).all():
            raise ValueError(f"{key} contains non-finite values")
    return data


def tensor_batch(data: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    def tensor(key: str, dtype: torch.dtype | None = None) -> Tensor:
        value = torch.from_numpy(data[key][indices])
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value.to(device)

    targets = {name: tensor(f"y_{name}", torch.float32) for name in HEADS}
    masks = {name: tensor(f"m_{name}").bool() for name in HEADS}
    masks["episode_fully_known_negative"] = tensor("episode_fully_known_negative").bool()
    return {
        "proprio": tensor("X_proprio", torch.float32),
        "policy": tensor("X_policy", torch.float32),
        "visual": tensor("X_visual", torch.float32),
        "language": tensor("X_language", torch.float32),
        "targets": targets,
        "masks": masks,
        "sample_weight": tensor("sample_weight", torch.float32),
    }


def forward_batch(
    model: C2gGripperCriticalWindowDetector,
    batch: Mapping[str, Any],
    *,
    use_policy_intent: bool,
    use_visual: bool,
    return_sequence: bool,
) -> dict[str, Tensor]:
    return model(
        batch["proprio"],
        batch["language"],
        policy_intent=batch["policy"] if use_policy_intent else None,
        siglip_visual=batch["visual"] if use_visual else None,
        return_sequence=return_sequence,
    )


def collect_current_probabilities(
    model: C2gGripperCriticalWindowDetector,
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_policy_intent: bool,
    use_visual: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    probabilities = {name: [] for name in HEADS}
    with torch.no_grad():
        for batch_index in batch_indices(indices, batch_size):
            batch = tensor_batch(data, batch_index, device)
            outputs = forward_batch(
                model,
                batch,
                use_policy_intent=use_policy_intent,
                use_visual=use_visual,
                return_sequence=False,
            )
            for name in HEADS:
                probabilities[name].append(torch.sigmoid(outputs[name]).cpu().numpy())
    return {
        name: np.concatenate(values).astype(np.float64) if values else np.empty((0,), dtype=np.float64)
        for name, values in probabilities.items()
    }


def binary_metrics(predicted: np.ndarray, target: np.ndarray, known: np.ndarray) -> dict[str, float | int]:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    known = known.astype(bool)
    p = predicted[known]
    y = target[known]
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    tn = int(np.sum(~p & ~y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn),
    }


def evaluate_thresholds(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    tau_critical: float,
    tau_release: float,
    tau_ground: float,
    persistence_window: int = 3,
    persistence_required: int = 2,
) -> dict[str, Any]:
    current_target = data["y_critical_window"][indices, -1] > 0.5
    current_known = data["m_critical_window"][indices, -1].astype(bool)
    release_target = data["y_release_safe"][indices, -1] > 0.5
    gate = (
        (probabilities["critical_window"] >= tau_critical)
        & (probabilities["release_safe"] < tau_release)
        & (probabilities["grounding_confidence"] >= tau_ground)
    )
    window = binary_metrics(gate, current_target, current_known)

    episode_key = data["episode_key"][indices].astype(str)
    step = data["step"][indices].astype(np.int64)
    suites = data["suite"][indices].astype(str)
    order = np.lexsort((step, episode_key))
    episode_stats: list[dict[str, Any]] = []
    for key in sorted(set(episode_key.tolist())):
        local = order[episode_key[order] == key]
        if local.size == 0:
            continue
        local = local[np.argsort(step[local])]
        local_gate = gate[local]
        persistent = np.zeros_like(local_gate, dtype=bool)
        for pos in range(len(local_gate)):
            left = max(0, pos - persistence_window + 1)
            if pos - left + 1 == persistence_window:
                persistent[pos] = int(local_gate[left : pos + 1].sum()) >= persistence_required
        y = current_target[local]
        known = current_known[local]
        safe = release_target[local] & data["m_release_safe"][indices[local], -1].astype(bool)
        trigger_positions = np.flatnonzero(persistent)
        positive_positions = np.flatnonzero(y & known)
        first_trigger = int(step[local[trigger_positions[0]]]) if trigger_positions.size else None
        first_positive = int(step[local[positive_positions[0]]]) if positive_positions.size else None
        fully_negative = bool(known.all() and not bool((y & known).any()))
        episode_stats.append(
            {
                "episode_key": key,
                "suite": str(suites[local[0]]),
                "triggered": bool(trigger_positions.size),
                "positive": bool(positive_positions.size),
                "fully_known_negative": fully_negative,
                "release_safe_trigger": bool(np.any(persistent & safe)),
                "first_trigger": first_trigger,
                "first_positive": first_positive,
                "delay": None if first_trigger is None or first_positive is None else first_trigger - first_positive,
            }
        )
    positive_episodes = [row for row in episode_stats if row["positive"]]
    negative_episodes = [row for row in episode_stats if row["fully_known_negative"]]
    triggered_positive = sum(row["triggered"] for row in positive_episodes)
    triggered_negative = sum(row["triggered"] for row in negative_episodes)
    safe_triggers = sum(row["release_safe_trigger"] for row in episode_stats)
    delays = [row["delay"] for row in positive_episodes if row["delay"] is not None]
    per_suite: dict[str, dict[str, float | int]] = {}
    for suite in sorted(set(row["suite"] for row in episode_stats)):
        rows = [row for row in episode_stats if row["suite"] == suite]
        pos = [row for row in rows if row["positive"]]
        neg = [row for row in rows if row["fully_known_negative"]]
        per_suite[suite] = {
            "episode_count": len(rows),
            "positive_episode_recall": sum(row["triggered"] for row in pos) / max(1, len(pos)),
            "negative_episode_any_trigger_rate": sum(row["triggered"] for row in neg) / max(1, len(neg)),
        }
    return {
        "thresholds": {
            "tau_critical": tau_critical,
            "tau_release": tau_release,
            "tau_ground": tau_ground,
            "persistence_window": persistence_window,
            "persistence_required": persistence_required,
        },
        "window": window,
        "episode_count": len(episode_stats),
        "positive_episode_count": len(positive_episodes),
        "fully_known_negative_episode_count": len(negative_episodes),
        "positive_episode_recall": triggered_positive / max(1, len(positive_episodes)),
        "negative_episode_any_trigger_rate": triggered_negative / max(1, len(negative_episodes)),
        "release_safe_episode_trigger_rate": safe_triggers / max(1, len(episode_stats)),
        "first_trigger_delay_mean": float(np.mean(delays)) if delays else None,
        "first_trigger_delay_p50": float(np.median(delays)) if delays else None,
        "per_suite": per_suite,
    }


def threshold_sweep(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    max_episode_fp: float,
    max_release_safe_trigger: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for tau_critical in (0.30, 0.40, 0.50, 0.60, 0.70):
        for tau_release in (0.30, 0.40, 0.50, 0.60):
            for tau_ground in (0.30, 0.50, 0.70):
                metrics = evaluate_thresholds(
                    data,
                    indices,
                    probabilities,
                    tau_critical=tau_critical,
                    tau_release=tau_release,
                    tau_ground=tau_ground,
                )
                results.append(metrics)
    feasible = [
        row for row in results
        if row["negative_episode_any_trigger_rate"] <= max_episode_fp
        and row["release_safe_episode_trigger_rate"] <= max_release_safe_trigger
    ]
    pool = feasible or results
    best = max(
        pool,
        key=lambda row: (
            row["positive_episode_recall"],
            row["window"]["f1"],
            -row["negative_episode_any_trigger_rate"],
            -row["release_safe_episode_trigger_rate"],
        ),
    )
    return {"best": best, "feasible_count": len(feasible), "evaluated_count": len(results)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--start-weight", type=float, default=0.4)
    parser.add_argument("--active-weight", type=float, default=0.2)
    parser.add_argument("--early-weight", type=float, default=0.25)
    parser.add_argument("--miss-weight", type=float, default=0.5)
    parser.add_argument("--negative-episode-weight", type=float, default=0.5)
    parser.add_argument("--release-safe-episode-weight", type=float, default=0.5)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--use-policy-intent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-visual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-language-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-fp", type=float, default=0.10)
    parser.add_argument("--max-release-safe-trigger", type=float, default=0.05)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    split = data["split"].astype(str)
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    if train_indices.size == 0 or val_indices.size == 0:
        raise RuntimeError("dataset requires nonempty train and val splits")

    config = C2gDetectorConfig(
        visual_dim=int(data["X_visual"].shape[-1]),
        language_dim=int(data["X_language"].shape[-1]),
        policy_intent_dim=int(data["X_policy"].shape[-1]),
        hidden=args.hidden,
        dropout=args.dropout,
        patch_dim=None,
        use_policy_intent=args.use_policy_intent,
        use_visual=args.use_visual,
        use_language_conditioning=args.use_language_conditioning,
    )
    model = C2gGripperCriticalWindowDetector(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: dict[str, Tensor] | None = None
    best_score = float("-inf")
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_indices)
        epoch_losses: list[float] = []
        for indices in batch_indices(train_indices, args.batch_size):
            batch = tensor_batch(data, indices, device)
            outputs = forward_batch(
                model,
                batch,
                use_policy_intent=args.use_policy_intent,
                use_visual=args.use_visual,
                return_sequence=True,
            )
            losses = clean_window_loss(
                outputs,
                batch["targets"],
                batch["masks"],
                sample_weight=batch["sample_weight"],
                auxiliary_weight=args.auxiliary_weight,
                start_weight=args.start_weight,
                active_weight=args.active_weight,
                early_weight=args.early_weight,
                miss_weight=args.miss_weight,
                negative_episode_weight=args.negative_episode_weight,
                release_safe_episode_weight=args.release_safe_episode_weight,
                smoothness_weight=args.smoothness_weight,
                include_episode_losses=True,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(losses["total"].detach().cpu()))

        val_probabilities = collect_current_probabilities(
            model,
            data,
            val_indices,
            batch_size=args.batch_size,
            device=device,
            use_policy_intent=args.use_policy_intent,
            use_visual=args.use_visual,
        )
        provisional = evaluate_thresholds(
            data,
            val_indices,
            val_probabilities,
            tau_critical=0.5,
            tau_release=0.5,
            tau_ground=0.5,
        )
        score = (
            provisional["positive_episode_recall"]
            + provisional["window"]["f1"]
            - provisional["negative_episode_any_trigger_rate"]
            - provisional["release_safe_episode_trigger_rate"]
        )
        if score > best_score:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "selection_score": float(score),
            "val_positive_episode_recall": provisional["positive_episode_recall"],
            "val_negative_episode_any_trigger_rate": provisional["negative_episode_any_trigger_rate"],
            "val_release_safe_episode_trigger_rate": provisional["release_safe_episode_trigger_rate"],
            "val_window_f1": provisional["window"]["f1"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_probabilities = collect_current_probabilities(
        model,
        data,
        val_indices,
        batch_size=args.batch_size,
        device=device,
        use_policy_intent=args.use_policy_intent,
        use_visual=args.use_visual,
    )
    sweep = threshold_sweep(
        data,
        val_indices,
        val_probabilities,
        max_episode_fp=args.max_episode_fp,
        max_release_safe_trigger=args.max_release_safe_trigger,
    )
    thresholds = sweep["best"]["thresholds"]
    test_metrics: dict[str, Any] = {}
    if test_indices.size:
        test_probabilities = collect_current_probabilities(
            model,
            data,
            test_indices,
            batch_size=args.batch_size,
            device=device,
            use_policy_intent=args.use_policy_intent,
            use_visual=args.use_visual,
        )
        test_metrics = evaluate_thresholds(
            data,
            test_indices,
            test_probabilities,
            tau_critical=float(thresholds["tau_critical"]),
            tau_release=float(thresholds["tau_release"]),
            tau_ground=float(thresholds["tau_ground"]),
        )

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": best_state,
        "model_config": {
            "visual_dim": config.visual_dim,
            "language_dim": config.language_dim,
            "policy_intent_dim": config.policy_intent_dim,
            "hidden": config.hidden,
            "dropout": config.dropout,
            "patch_dim": config.patch_dim,
            "use_policy_intent": config.use_policy_intent,
            "use_visual": config.use_visual,
            "use_language_conditioning": config.use_language_conditioning,
        },
        "window": int(data["X_proprio"].shape[1]),
        "thresholds": thresholds,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": args.git_commit,
        "history": history,
        "validation": sweep["best"],
        "test": test_metrics,
        "feature_contract": {
            "proprio_dim": 25,
            "policy_feature_names": [str(value) for value in data.get("feature_names_policy", [])],
            "visual_dim": int(data["X_visual"].shape[-1]),
            "language_dim": int(data["X_language"].shape[-1]),
            "clean_only": True,
        },
    }
    checkpoint_path = output_dir / "c2g_clean_window_detector.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "gate": "C2G_CLEAN_WINDOW_TRAINING",
        "status": "PASS_TRAINED_NEEDS_INDEPENDENT_AUDIT",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "train_samples": int(train_indices.size),
        "val_samples": int(val_indices.size),
        "test_samples": int(test_indices.size),
        "model_config": checkpoint["model_config"],
        "threshold_calibration": sweep,
        "test_metrics": test_metrics,
        "best_score": best_score,
        "runtime_seconds": time.time() - started,
        "git_commit": args.git_commit,
        "boundaries": {
            "clean_only_training": True,
            "attack_outcomes_read": False,
            "libero_rollouts_launched": 0,
            "online_attacks_launched": 0,
            "d7_table1_modified": False,
        },
    }
    write_json(output_dir / "c2g_clean_window_training_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
