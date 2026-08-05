#!/usr/bin/env python3
"""Fail-closed transfer/parity audit for the frozen Stage 2 R2 detector."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "detector_v5"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from d8_train_core import apply_normalization, create_model, load_checkpoint
from run_detector_clean_freeze import cache_effective_rows, load_cache, load_clean_event_groups
from run_detector_stage2_r2 import detailed_candidate_metrics, scheduler_trace
from stage3a_runtime import sha256_file


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_scores(checkpoint: Path, rows: list[dict[str, Any]], norm: dict[str, Any]) -> list[float]:
    model = create_model(seed=20260717).to("cpu")
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    output: list[float] = []
    batch_size = 8192
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = np.asarray(
                [row["features_25d_raw"] for row in rows[start : start + batch_size]], dtype=np.float32
            )
            if batch.shape[1:] != (25,) or not np.isfinite(batch).all():
                raise RuntimeError("Cache A contains a malformed/non-finite 25D row")
            logits = model(apply_normalization(torch.from_numpy(batch), norm)).detach().cpu().numpy()
            if logits.ndim != 1 or not np.isfinite(logits).all():
                raise RuntimeError("final checkpoint produced malformed/non-finite logits")
            output.extend(float(value) for value in logits.tolist())
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-cache-seal", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema": "D8_STAGE3A_FINAL_CHECKPOINT_TRANSFER_AUDIT_V1",
        "status": "RUNNING",
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "scheduler_freeze_is_embedded_in_receipt": True,
    }
    try:
        stage2_root = args.stage2_root.resolve(strict=True)
        checkpoint = args.checkpoint.resolve(strict=True)
        receipt_path = args.freeze_receipt.resolve(strict=True)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("source_commit") != args.expected_source_commit or receipt.get("source_tree") != args.expected_source_tree:
            raise RuntimeError("Stage 2 R2 receipt source commit/tree mismatch")
        checkpoint_sha = sha256_file(checkpoint)
        receipt_sha = sha256_file(receipt_path)
        if checkpoint_sha != args.expected_checkpoint_sha256.lower():
            raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
        if receipt.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("receipt/checkpoint SHA mismatch")
        scheduler = receipt.get("scheduler")
        if not isinstance(scheduler, dict):
            raise RuntimeError("missing embedded scheduler freeze")
        candidate = {
            key: scheduler[key]
            for key in ("threshold", "persistence", "hysteresis", "cooldown")
        }
        provenance = receipt.get("provenance") or {}
        cache_root = Path(provenance["cache_root"]).resolve(strict=True)
        rows, cache_manifest, cache_seal = load_cache(cache_root, args.expected_cache_seal)
        effective = cache_effective_rows(rows)
        if not effective:
            raise RuntimeError("Cache A has no effective rows")
        norm = None
        model = create_model(seed=20260717).to("cpu")
        checkpoint_data = load_checkpoint(checkpoint, model, map_location="cpu")
        norm = checkpoint_data.get("normalization")
        if not isinstance(norm, dict) or norm.get("schema") != "D8_NORMALIZATION_V2" or norm.get("feature_dim") != 25:
            raise RuntimeError("checkpoint normalization binding mismatch")
        mean = np.asarray(norm.get("mean"), dtype=np.float64)
        std = np.asarray(norm.get("std"), dtype=np.float64)
        if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise RuntimeError("checkpoint normalization is invalid")

        scores_a = _load_scores(checkpoint, effective, norm)
        scores_b = _load_scores(checkpoint, effective, norm)
        score_deterministic = bool(np.array_equal(np.asarray(scores_a), np.asarray(scores_b)))
        if not score_deterministic:
            raise RuntimeError("final checkpoint score replay is not deterministic")

        aggregate = [
            dict(row, target=float(row["physical_target"]), score=float(score))
            for row, score in zip(effective, scores_a)
        ]
        event_groups, event_binding = load_clean_event_groups(
            Path(provenance["sidecar_root"]), Path(provenance["teacher_root"]), rows
        )
        metrics_a, traces_a = detailed_candidate_metrics(aggregate, event_groups, candidate)
        metrics_b, traces_b = detailed_candidate_metrics(aggregate, event_groups, candidate)
        scheduler_deterministic = bool(
            json.dumps(traces_a, sort_keys=True, separators=(",", ":"))
            == json.dumps(traces_b, sort_keys=True, separators=(",", ":"))
        )
        if not scheduler_deterministic:
            raise RuntimeError("frozen scheduler replay is not deterministic")
        required_metrics = {
            key: metrics_a.get(key)
            for key in (
                "false_onset_episode_rate",
                "negative_active_step_rate",
                "active_overlap_event_recall",
                "median_first_activation_delay",
            )
        }
        gate_pass = (
            float(required_metrics["false_onset_episode_rate"]) <= 0.10
            and float(required_metrics["negative_active_step_rate"]) <= 0.05
        )
        report.update(
            {
                "status": "PASS" if gate_pass else "FINAL_CHECKPOINT_TRANSFER_FAIL",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "scheduler_freeze_path": str(receipt_path),
                "scheduler_freeze_sha256": receipt_sha,
                "checkpoint_schema": checkpoint_data.get("schema"),
                "model_architecture": {"input": 25, "hidden": 32, "hidden2": 16, "output": 1},
                "normalization_schema": norm.get("schema"),
                "feature_dim": int(norm.get("feature_dim")),
                "score_deterministic": score_deterministic,
                "scheduler_deterministic": scheduler_deterministic,
                "all_scores_finite": bool(np.isfinite(np.asarray(scores_a)).all()),
                "scheduler": candidate,
                "metrics": required_metrics,
                "full_metrics": metrics_a,
                "cache_root": str(cache_root),
                "cache_seal": cache_seal,
                "event_binding": event_binding,
                "source_commit": args.expected_source_commit,
                "source_tree": args.expected_source_tree,
                "stage2_root": str(stage2_root),
                "stage2_root_modified": False,
                "transfer_gate": "PASS" if gate_pass else "FAIL",
            }
        )
    except Exception as exc:
        report.update({"status": "FINAL_CHECKPOINT_TRANSFER_FAIL", "error": f"{type(exc).__name__}: {exc}"})
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "FINAL_CHECKPOINT_TRANSFER_AUDIT.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output_root / "FINAL_CHECKPOINT_TRANSFER_AUDIT.json")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

