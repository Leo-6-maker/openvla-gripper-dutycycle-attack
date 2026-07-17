#!/usr/bin/env python3
"""Build one sealed, held-out FIT prediction bundle.

This module is intentionally independent of CAL/CHECK and attack code.  It
only turns a formally trained fold candidate into per-step records for the
pre-registered FIT viability evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from gripper_attack.b3_formal import B3Normalization, B3OfficialStatefulGRU, B3_HEADS
from gripper_attack.b3_formal import json_sha
from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.b3_v3_dataset import B3Episode


PREDICTION_SOURCE_BINDING_NAMES = (
    "registry_root_sha256",
    "s1_root_sha256",
    "fold_bundle_sha256",
    "checkpoint_bundle_sha256",
    "normalization_bundle_sha256",
    "normalization_sha256",
    "normalization_file_sha256",
    "authorization_payload_sha256",
    "runner_binding_sha256",
)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


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


def write_prediction_bundle(
    output_root: Path,
    records: list[dict[str, Any]],
    *,
    fold_id: int,
    seed: int,
    variant: str,
    checkpoint_sha256: str,
    validation_identity_sha256: str,
    source_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if not records:
        raise ValueError("prediction bundle cannot be empty")
    identities = sorted({str(row["canonical_parent_key"]) for row in records})
    if len(identities) != 200:
        raise ValueError(f"validation prediction bundle must contain 200 identities, got {len(identities)}")
    actual_identity_sha256 = json_sha(identities)
    if validation_identity_sha256 != actual_identity_sha256:
        raise ValueError("validation identity SHA does not match the actual prediction identities")
    for row in records:
        if (
            row.get("fold_id") != fold_id
            or row.get("seed") != seed
            or row.get("variant") != variant
            or row.get("checkpoint_sha256") != checkpoint_sha256
            or row.get("attack_enabled") is not False
            or row.get("teacher_inputs_consumed") is not False
        ):
            raise ValueError("prediction record coordinate/source binding mismatch")
    if source_bindings is not None:
        missing = [name for name in PREDICTION_SOURCE_BINDING_NAMES if name not in source_bindings]
        if missing or any(not _is_sha(source_bindings[name]) for name in PREDICTION_SOURCE_BINDING_NAMES):
            raise ValueError(f"prediction source binding is incomplete: {missing}")
        policy_root = source_bindings.get("policy_intent_root_sha256")
        if variant == "B3_25D" and policy_root is not None:
            raise ValueError("B3_25D prediction cannot consume a 9D root")
        if variant == "B3_25D9D" and not _is_sha(policy_root):
            raise ValueError("B3_25D9D prediction must bind a 9D root")
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
            "validation_identity_sha256": actual_identity_sha256,
            "validation_identities": identities,
            "source_bindings": dict(source_bindings or {}),
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
    identities = sorted({str(row.get("canonical_parent_key", "")) for row in records})
    if len(identities) != 200 or manifest.get("validation_identity_count") != 200:
        raise ValueError("prediction bundle validation identity count mismatch")
    if manifest.get("validation_identities") != identities or manifest.get("validation_identity_sha256") != json_sha(identities):
        raise ValueError("prediction bundle validation identity set mismatch")
    for row in records:
        if row.get("fold_id") != manifest.get("fold_id") or row.get("seed") != manifest.get("seed") or row.get("variant") != manifest.get("variant") or row.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise ValueError("prediction record coordinate/checkpoint mismatch")
        if row.get("attack_enabled") is not False or row.get("teacher_inputs_consumed") is not False:
            raise ValueError("prediction record boundary failed")
    bindings = manifest.get("source_bindings")
    if bindings:
        missing = [name for name in PREDICTION_SOURCE_BINDING_NAMES if name not in bindings]
        if missing or any(not _is_sha(bindings[name]) for name in PREDICTION_SOURCE_BINDING_NAMES):
            raise ValueError(f"prediction source binding is incomplete: {missing}")
        policy_root = bindings.get("policy_intent_root_sha256")
        if manifest.get("variant") == "B3_25D" and policy_root is not None:
            raise ValueError("B3_25D prediction contains a 9D source binding")
        if manifest.get("variant") == "B3_25D9D" and not _is_sha(policy_root):
            raise ValueError("B3_25D9D prediction is missing its 9D source binding")
    return manifest, records


__all__ = ["PREDICTION_SOURCE_BINDING_NAMES", "build_prediction_records", "write_prediction_bundle", "load_prediction_bundle"]
