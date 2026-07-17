#!/usr/bin/env python3
"""Build one sealed, held-out FIT prediction bundle.

This module is intentionally independent of CAL/CHECK and attack code.  It
only turns a formally trained fold candidate into per-step records for the
pre-registered FIT viability evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from gripper_attack.b3_formal import B3Normalization, B3OfficialStatefulGRU, B3_HEADS
from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.b3_v3_dataset import B3Episode


def _normalise(value: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=value.dtype, device=value.device)
    std_t = torch.tensor(std, dtype=value.dtype, device=value.device)
    return (value - mean_t) / std_t


def build_prediction_records(
    model: B3OfficialStatefulGRU,
    episodes: Iterable[B3Episode],
    normalization: B3Normalization,
    *,
    checkpoint_sha256: str,
    fold_id: int,
    seed: int,
    variant: str,
    provisional_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if len(checkpoint_sha256) != 64 or variant != model.config.variant:
        raise ValueError("prediction bundle checkpoint/variant binding is invalid")
    if not 0.0 < provisional_threshold < 1.0:
        raise ValueError("provisional threshold must be in (0, 1)")
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode in sorted(episodes, key=lambda item: item.canonical_parent_key):
            hidden = None
            for step in range(len(episode.features_25d)):
                x25 = _normalise(episode.features_25d[step], normalization.mean_25d, normalization.std_25d)
                x9 = None
                if model.config.variant == "B3_25D9D":
                    if episode.features_9d is None:
                        raise ValueError("B3_25D9D validation episode has no 9D input")
                    x9 = _normalise(episode.features_9d[step], normalization.mean_9d, normalization.std_9d)
                logits, hidden = model.step(x25, x9, hidden, episode.valid_mask[step:step + 1])
                probabilities = {head: float(torch.sigmoid(logits[f"{head}_logit"])[0]) for head in B3_HEADS}
                known = bool(episode.known_masks["retention_continuation_t10"][step])
                event_id = int(episode.event_ids[step]) if step < len(episode.event_ids) else -1
                close_streak = float(episode.features_25d[step, 13])
                time_since_close = float(episode.features_25d[step, 17])
                records.append({
                    "schema": "B3_OFFICIAL_V3_FIT_PREDICTION_RECORD_V1",
                    "canonical_parent_key": episode.canonical_parent_key,
                    "suite": episode.suite,
                    "task_idx": episode.task_idx,
                    "state_id": episode.state_id,
                    "split": episode.split,
                    "step": step,
                    "event_id": event_id,
                    "event_ordinal": event_id,
                    "target_t10_known": known,
                    "target_t10": bool(episode.targets["retention_continuation_t10"][step]) if known else False,
                    "pred_emit": probabilities["retention_continuation_t10"] >= provisional_threshold,
                    "release_imminent": probabilities["release_imminent"] >= provisional_threshold,
                    "retention_continuation_t10_probability": probabilities["retention_continuation_t10"],
                    "release_imminent_probability": probabilities["release_imminent"],
                    "grasp_support_probability": probabilities["grasp_support"],
                    "retention_active_probability": probabilities["retention_active"],
                    "logits": {head: float(logits[f"{head}_logit"][0]) for head in B3_HEADS},
                    "recent_close_streak": close_streak,
                    "time_since_close": time_since_close,
                    "fold_id": fold_id,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint_sha256": checkpoint_sha256,
                    "attack_enabled": False,
                    "teacher_inputs_consumed": False,
                })
    return records


def write_prediction_bundle(output_root: Path, records: list[dict[str, Any]], *, fold_id: int, seed: int, variant: str, checkpoint_sha256: str, validation_identity_sha256: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if not records:
        raise ValueError("prediction bundle cannot be empty")
    identities = sorted({str(row["canonical_parent_key"]) for row in records})
    if len(identities) != 200:
        raise ValueError(f"validation prediction bundle must contain 200 identities, got {len(identities)}")
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=True)
        records_path = staging / "prediction_records.jsonl"
        records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        manifest = {
            "schema": "B3_OFFICIAL_V3_FIT_PREDICTION_BUNDLE_V1",
            "fold_id": fold_id,
            "seed": seed,
            "variant": variant,
            "checkpoint_sha256": checkpoint_sha256,
            "validation_identity_count": len(identities),
            "validation_identity_sha256": validation_identity_sha256,
            "record_count": len(records),
            "provisional_threshold": 0.5,
            "attack_enabled": False,
            "teacher_inputs_consumed": False,
            "effectiveness_metrics_produced": False,
        }
        manifest_path = staging / "prediction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (manifest_path.with_name(manifest_path.name + ".sha256")).write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
        seal_directory(staging)
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_prediction_bundle(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verify_sealed_directory(root)
    manifest = json.loads((root / "prediction_manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (root / "prediction_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest.get("schema") != "B3_OFFICIAL_V3_FIT_PREDICTION_BUNDLE_V1" or manifest.get("attack_enabled") is not False or manifest.get("teacher_inputs_consumed") is not False:
        raise ValueError("prediction bundle boundary failed")
    if len({row.get("canonical_parent_key") for row in records}) != 200:
        raise ValueError("prediction bundle validation identity count mismatch")
    return manifest, records


__all__ = ["build_prediction_records", "write_prediction_bundle", "load_prediction_bundle"]
