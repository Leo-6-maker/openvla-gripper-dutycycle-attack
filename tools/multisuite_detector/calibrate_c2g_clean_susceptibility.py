#!/usr/bin/env python3
"""Calibrate Detector-v2 clean policy susceptibility without attacked outcomes.

The calibration uses only validation-split clean policy-intent features and known
clean Teacher-v2 critical-window labels.  It patches the exported checkpoint with
frozen close-intent, log-mass-margin, and entropy gates, then updates the training
report with the new checkpoint hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from src.gripper_attack.c2g_clean_window_runtime import CHECKPOINT_SCHEMA_VERSION
from tools.multisuite_detector.train_c2g_clean_window_detector import load_dataset

CALIBRATION_SCHEMA = "c2g.clean_susceptibility_calibration.2026-07-10.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate(
    data: Mapping[str, np.ndarray],
    *,
    split_name: str = "val",
    positive_retention: float = 0.80,
    require_top1_close: bool = True,
) -> dict[str, Any]:
    if not 0.0 < positive_retention <= 1.0:
        raise ValueError("positive_retention must be in (0,1]")
    names = [str(value) for value in data.get("feature_names_policy", [])]
    if names != list(CLEAN_POLICY_FEATURE_NAMES):
        raise ValueError("policy feature order differs from the frozen clean feature contract")
    indices = np.flatnonzero(data["split"].astype(str) == split_name)
    if indices.size == 0:
        raise ValueError(f"split {split_name} is empty")
    current_policy = data["X_policy"][indices, -1].astype(np.float64)
    current_target = data["y_critical_window"][indices, -1] > 0.5
    current_known = data["m_critical_window"][indices, -1].astype(bool)
    margin_index = names.index("clean_open_minus_close_log_mass")
    entropy_index = names.index("clean_action_token_entropy_normalized")
    close_index = names.index("clean_top1_is_close")
    positive = current_known & current_target
    close = current_policy[:, close_index] >= 0.5
    calibration_mask = positive & close if require_top1_close else positive
    if int(calibration_mask.sum()) <= 0:
        raise ValueError("validation split has no known positive clean-close rows for susceptibility calibration")
    quantile = max(0.0, 1.0 - positive_retention)
    margins = current_policy[calibration_mask, margin_index]
    entropies = current_policy[calibration_mask, entropy_index]
    minimum_margin = float(np.quantile(margins, quantile, method="lower"))
    minimum_entropy = float(np.quantile(entropies, quantile, method="lower"))
    gate = (
        (close if require_top1_close else np.ones_like(close, dtype=bool))
        & (current_policy[:, margin_index] >= minimum_margin)
        & (current_policy[:, entropy_index] >= minimum_entropy)
    )
    retained_positive = int(np.sum(gate & positive))
    total_positive = int(np.sum(positive))
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "split": split_name,
        "positive_retention_target": positive_retention,
        "require_clean_close": bool(require_top1_close),
        "minimum_open_minus_close_log_mass": minimum_margin,
        "minimum_entropy": minimum_entropy,
        "calibration_positive_close_count": int(calibration_mask.sum()),
        "known_positive_count": total_positive,
        "retained_known_positive_count": retained_positive,
        "retained_known_positive_fraction": retained_positive / max(1, total_positive),
        "uses_attack_outcomes": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--positive-retention", type=float, default=0.80)
    parser.add_argument("--require-clean-close", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    checkpoint_path = args.checkpoint.resolve()
    training_report_path = args.training_report.resolve()
    data = load_dataset(dataset)
    calibration = calibrate(
        data,
        split_name=args.split,
        positive_retention=args.positive_retention,
        require_top1_close=args.require_clean_close,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if str(checkpoint.get("schema_version", "")) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema is not the clean-window deployment schema")
    if checkpoint.get("dataset_sha256") != sha256_file(dataset):
        raise ValueError("checkpoint dataset hash does not match calibration dataset")
    checkpoint["susceptibility"] = calibration
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    training_report["checkpoint_sha256"] = checkpoint_sha
    training_report["clean_susceptibility_calibration"] = calibration
    training_report_path.write_text(
        json.dumps(training_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report = {
        "gate": "C2G_CLEAN_SUSCEPTIBILITY_CALIBRATION",
        "status": "PASS_C2G_CLEAN_SUSCEPTIBILITY_CALIBRATED",
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "training_report": str(training_report_path),
        "training_report_sha256": sha256_file(training_report_path),
        "calibration": calibration,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
