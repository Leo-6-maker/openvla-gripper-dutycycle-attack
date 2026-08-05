#!/usr/bin/env python3
"""Read-only comparison of the frozen R2 OOF and final-checkpoint scorers."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "fec", ROOT / "scripts" / "detector_v5"):
    if str(path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(path))

from audit_stage2_r2_transfer import _load_scores
from d8_train_core import create_model
from run_detector_clean_freeze import (
    cache_effective_rows,
    load_cache,
    load_clean_event_groups,
    load_oof,
)
from run_detector_stage2_r2 import build_aggregate_rows, detailed_candidate_metrics
from stage3a_runtime import load_frozen_checkpoint, sha256_file


SEEDS = (20260720, 20260721, 20260722, 20260723, 20260724,
         20260725, 20260726, 20260727, 20260728, 20260729)
PRE_EVENT_WINDOW = 2
POST_EVENT_TOLERANCE = 2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeError("correlation inputs are not aligned")
    return float(np.corrcoef(left, right)[0, 1])


def score_summary(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    def stats(values: np.ndarray) -> dict[str, Any]:
        if not len(values):
            return {"n": 0}
        return {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    above = scores > threshold
    return {
        "all": stats(scores),
        "negative_target": stats(scores[labels == 0]),
        "positive_target": stats(scores[labels == 1]),
        "threshold": float(threshold),
        "threshold_percentile_leq": float(np.mean(scores <= threshold) * 100.0),
        "threshold_fraction_above": float(np.mean(above)),
        "threshold_count_above": int(np.sum(above)),
    }


def protected_steps(groups: Iterable[Mapping[str, Any]]) -> set[int]:
    protected: set[int] = set()
    for group in groups:
        fragments = [(int(start), int(end)) for start, end in group["fragment_ranges"]]
        start = min(item[0] for item in fragments)
        end = max(item[1] for item in fragments)
        protected.update(
            range(start - PRE_EVENT_WINDOW, end + POST_EVENT_TOLERANCE + 1)
        )
    return protected


def false_onset_detail(trace: list[dict[str, Any]], groups: Iterable[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    protected = protected_steps(groups)
    false_rows = [
        row for row in trace
        if row.get("emission") and int(row["step"]) not in protected
    ]
    margins = [float(row["score"]) - threshold for row in false_rows]
    return {
        "false_onset": bool(false_rows),
        "false_onset_count": len(false_rows),
        "first_false_onset_step": int(false_rows[0]["step"]) if false_rows else None,
        "first_false_onset_margin": margins[0] if margins else None,
        "max_false_onset_margin": max(margins) if margins else None,
        "false_latched_steps": sum(
            int(row.get("latched_active", False)) for row in false_rows
        ),
    }


def episode_rows(
    oof_traces: Mapping[str, list[dict[str, Any]]],
    final_traces: Mapping[str, list[dict[str, Any]]],
    event_groups: Mapping[str, list[dict[str, Any]]],
    oof_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    oof_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    final_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        oof_by_episode[str(row["episode_id"])].append(row)
    for row in final_rows:
        final_by_episode[str(row["episode_id"])].append(row)
    output = []
    for episode_id in sorted(oof_by_episode):
        oof_scores = np.asarray([float(row["score"]) for row in oof_by_episode[episode_id]])
        final_scores = np.asarray([float(row["score"]) for row in final_by_episode[episode_id]])
        if len(oof_scores) != len(final_scores):
            raise RuntimeError(f"OOF/final episode length mismatch: {episode_id}")
        oof_false = false_onset_detail(oof_traces[episode_id], event_groups.get(episode_id, []), threshold)
        final_false = false_onset_detail(final_traces[episode_id], event_groups.get(episode_id, []), threshold)
        category = f"OOF_{'FALSE' if oof_false['false_onset'] else 'TRUE'}_FINAL_{'FALSE' if final_false['false_onset'] else 'TRUE'}"
        output.append({
            "episode_id": episode_id,
            "suite": episode_id.split("/", 1)[0],
            "step_count": int(len(oof_scores)),
            "category": category,
            "oof": {**oof_false, "mean_logit": float(np.mean(oof_scores)), "max_logit": float(np.max(oof_scores))},
            "final": {**final_false, "mean_logit": float(np.mean(final_scores)), "max_logit": float(np.max(final_scores))},
            "score_shift": {
                "mean_logit_delta_final_minus_oof": float(np.mean(final_scores - oof_scores)),
                "max_logit_delta_final_minus_oof": float(np.max(final_scores - oof_scores)),
                "min_logit_delta_final_minus_oof": float(np.min(final_scores - oof_scores)),
                "mean_absolute_logit_delta": float(np.mean(np.abs(final_scores - oof_scores))),
            },
        })
    return output


def suite_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[row["suite"]].append(row)
    return {
        suite: {
            "episode_count": len(rows),
            "oof_false_onset_episode_count": sum(row["oof"]["false_onset"] for row in rows),
            "final_false_onset_episode_count": sum(row["final"]["false_onset"] for row in rows),
            "new_final_false_onset_episode_count": sum(row["category"] == "OOF_TRUE_FINAL_FALSE" for row in rows),
            "oof_false_onset_episode_rate": sum(row["oof"]["false_onset"] for row in rows) / len(rows),
            "final_false_onset_episode_rate": sum(row["final"]["false_onset"] for row in rows) / len(rows),
            "mean_logit_delta_final_minus_oof": statistics.mean(row["score_shift"]["mean_logit_delta_final_minus_oof"] for row in rows),
            "mean_absolute_logit_delta": statistics.mean(row["score_shift"]["mean_absolute_logit_delta"] for row in rows),
        }
        for suite, rows in sorted(grouped.items())
    }


def seal_directory(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage2_root", "formal_root", "cache_root", "sidecar_root", "teacher_root", "checkpoint", "freeze_receipt", "transfer_report", "output_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--expected-stage2-source-commit", required=True)
    parser.add_argument("--expected-stage2-source-tree", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-cache-seal", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    stage2_root = args.stage2_root.resolve(strict=True)
    formal_root = args.formal_root.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    freeze_receipt = args.freeze_receipt.resolve(strict=True)
    transfer_report_path = args.transfer_report.resolve(strict=True)

    transfer = read_json(transfer_report_path)
    receipt = read_json(freeze_receipt)
    if transfer.get("status") != "FINAL_CHECKPOINT_TRANSFER_FAIL":
        raise RuntimeError("discrepancy audit is only for the recorded transfer failure")
    if receipt.get("source_commit") != args.expected_stage2_source_commit or receipt.get("source_tree") != args.expected_stage2_source_tree:
        raise RuntimeError("Stage 2 receipt source binding mismatch")
    checkpoint_sha = sha256_file(checkpoint)
    receipt_sha = sha256_file(freeze_receipt)
    if checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    if receipt.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("receipt/checkpoint binding mismatch")
    if transfer.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("transfer report/checkpoint binding mismatch")

    cache_rows, _, cache_seal = load_cache(cache_root, args.expected_cache_seal)
    effective = cache_effective_rows(cache_rows)
    seed_scores, oof_meta = load_oof(
        formal_root, cache_rows, args.expected_stage2_source_commit, args.expected_stage2_source_tree
    )
    oof_rows = build_aggregate_rows(cache_rows, seed_scores)
    model = create_model(seed=20260717).to("cpu")
    checkpoint_data = load_frozen_checkpoint(checkpoint, model)
    norm = checkpoint_data.get("normalization")
    if not isinstance(norm, dict) or norm.get("schema") != "D8_NORMALIZATION_V2" or norm.get("feature_dim") != 25:
        raise RuntimeError("final checkpoint normalization binding mismatch")
    final_scores = _load_scores(checkpoint, effective, norm)
    final_rows = [dict(row, target=float(row["physical_target"]), score=float(score)) for row, score in zip(effective, final_scores)]
    oof_by_key = {(str(row["episode_id"]), int(row["step"])): row for row in oof_rows}
    final_by_key = {(str(row["episode_id"]), int(row["step"])): row for row in final_rows}
    if set(oof_by_key) != set(final_by_key) or len(oof_by_key) != len(effective):
        raise RuntimeError("OOF/final identity closure mismatch")
    keys = sorted(oof_by_key)
    oof_rows = [oof_by_key[key] for key in keys]
    final_rows = [final_by_key[key] for key in keys]
    event_groups, event_binding = load_clean_event_groups(sidecar_root, teacher_root, cache_rows)
    candidate = {key: receipt["scheduler"][key] for key in ("threshold", "persistence", "hysteresis", "cooldown")}
    oof_metrics, oof_traces = detailed_candidate_metrics(oof_rows, event_groups, candidate)
    final_metrics, final_traces = detailed_candidate_metrics(final_rows, event_groups, candidate)
    oof_array = np.asarray([float(row["score"]) for row in oof_rows], dtype=np.float64)
    final_array = np.asarray([float(row["score"]) for row in final_rows], dtype=np.float64)
    labels = np.asarray([int(row["target"]) for row in oof_rows], dtype=np.int64)
    if len(oof_array) != len(final_array) or not np.isfinite(oof_array).all() or not np.isfinite(final_array).all():
        raise RuntimeError("OOF/final score alignment or finiteness failed")
    episodes = episode_rows(oof_traces, final_traces, event_groups, oof_rows, final_rows, float(candidate["threshold"]))
    category_counts = dict(sorted(Counter(row["category"] for row in episodes).items()))
    oof_provenance = read_json(formal_root / "EXECUTION_RECEIPT.json").get("provenance", {})
    report = {
        "schema": "D8_STAGE2_R2_TRANSFER_DISCREPANCY_V1",
        "status": "TRANSFER_DISCREPANCY_AUDIT_COMPLETE",
        "scientific_transfer_gate": "FAIL_PRESERVED",
        "authorization_changed": False,
        "inputs_read_only": True,
        "checkpoint_sha256": checkpoint_sha,
        "scheduler_freeze_sha256": receipt_sha,
        "stage2_root": str(stage2_root),
        "formal_oof_root": str(formal_root),
        "cache_root": str(cache_root),
        "cache_seal": cache_seal["sha256sums_sha256"],
        "formal_oof_seal": oof_meta["formal_seal"]["sha256sums_sha256"],
        "formal_oof_prediction_file_count": oof_meta["prediction_file_count"],
        "effective_identity_count": len(effective),
        "effective_episode_count": len(episodes),
        "stage2_source_commit": args.expected_stage2_source_commit,
        "stage2_source_tree": args.expected_stage2_source_tree,
        "oof_training_source_commit": oof_provenance.get("source_commit"),
        "oof_training_source_tree": oof_provenance.get("source_tree"),
        "oof_training_source_matches_stage2_source": (
            oof_provenance.get("source_commit") == args.expected_stage2_source_commit
            and oof_provenance.get("source_tree") == args.expected_stage2_source_tree
        ),
        "scheduler": candidate,
        "oof_metrics": oof_metrics,
        "final_checkpoint_metrics": final_metrics,
        "metric_delta_final_minus_oof": {
            key: (None if oof_metrics.get(key) is None or final_metrics.get(key) is None else float(final_metrics[key]) - float(oof_metrics[key]))
            for key in ("false_onset_episode_count", "false_onset_episode_rate", "negative_active_step_rate", "active_overlap_event_recall", "median_first_activation_delay")
        },
        "score_system_comparison": {
            "pearson": correlation(oof_array, final_array),
            "spearman": correlation(rank_values(oof_array), rank_values(final_array)),
            "mean_logit_delta_final_minus_oof": float(np.mean(final_array - oof_array)),
            "std_logit_delta_final_minus_oof": float(np.std(final_array - oof_array, ddof=1)),
            "oof": score_summary(oof_array, labels, float(candidate["threshold"])),
            "final_checkpoint": score_summary(final_array, labels, float(candidate["threshold"])),
        },
        "category_counts": category_counts,
        "new_final_false_onset_episodes": [row for row in episodes if row["category"] == "OOF_TRUE_FINAL_FALSE"],
        "suite_breakdown": suite_summary(episodes),
        "episodes": episodes,
        "event_binding": event_binding,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "stage2_root_modified": False,
    }
    output_root.mkdir(parents=True)
    report_path = output_root / "FINAL_CHECKPOINT_TRANSFER_DISCREPANCY.json"
    atomic_json(report_path, report)
    artifact_sha = sha256_file(report_path)
    output_root_seal = seal_directory(output_root)
    print(json.dumps({"status": report["status"], "artifact_sha256": artifact_sha, "output_root_seal": output_root_seal, "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
