"""Causal, one-shot FIT evaluator for V5-A development checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from gripper_attack.b3_formal import json_sha
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, seal_directory, sha256_file
from gripper_attack.v5_dataset import aggregate_retrospective_window_scores, causal_window_anchor_scores, classify_v5_episode_windows, load_fit_registry, load_v5_episodes
from gripper_attack.v5_protocol import V5ModelContract
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _window_for_step(episode: Any, step: int) -> str | None:
    containing = [window for window in episode.windows if step in window.step_indices]
    if containing:
        return containing[0].window_id
    prior = [window for window in episode.windows if window.start <= step]
    return max(prior, key=lambda window: window.start).window_id if prior else None


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    contract = V5ModelContract("V5_A_PROPRIO")
    model = CausalMultimodalVulnerabilityRanker(contract)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    device = torch.device(args.device)
    model.to(device).eval()
    rows = load_fit_registry(args.registry_csv.resolve())
    fold = load_fit_fold_bundle(args.fold_root.resolve())
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == args.fold_id)
    by_key = {row["canonical_parent_key"]: row for row in rows}
    identities = list(fold_row["validation_identities"])
    episodes = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in identities])
    mean = checkpoint["normalization_mean_25d"].to(device)
    std = checkpoint["normalization_std_25d"].to(device)
    prediction_records: list[dict[str, Any]] = []
    scheduler_records: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    for episode in episodes:
        x = ((episode.features_25d.to(device) - mean) / std).unsqueeze(0)
        output = model.forward_sequence(x, valid_mask=episode.valid_mask.to(device).unsqueeze(0))
        utility = torch.sigmoid(output["utility_logit"][0]).cpu()
        release = torch.sigmoid(output["release_logit"][0]).cpu()
        regrasp = torch.sigmoid(output["regrasp_logit"][0]).cpu()
        scheduler = V5OneShotScheduler(V5SchedulerConfig(uncertainty_veto_enabled=False))
        emitted: list[dict[str, Any]] = []
        for step in range(len(episode.features_25d)):
            result = scheduler.update(
                step=step,
                candidate_close=bool(episode.candidate_close[step]),
                valid=bool(episode.valid_mask[step]),
                utility_probability=float(utility[step]),
                release_probability=float(release[step]),
                regrasp_probability=float(regrasp[step]),
                uncertainty_probability=0.0,
            )
            scheduler_records.append({"canonical_parent_key": episode.canonical_parent_key, **result})
            prediction_records.append({
                "canonical_parent_key": episode.canonical_parent_key,
                "step": step,
                "utility_probability": float(utility[step]),
                "release_probability": float(release[step]),
                "regrasp_probability": float(regrasp[step]),
                "candidate_close": bool(episode.candidate_close[step]),
                "student_valid": bool(episode.valid_mask[step]),
                "scheduler_emit": bool(result["emit"]),
            })
            if result["emit"]:
                emitted.append({"step": step, "window_id": _window_for_step(episode, step), "result": result})
        retro_scores, retro_rows = aggregate_retrospective_window_scores(output["utility_logit"][0].cpu(), episode)
        causal_scores, causal_rows = causal_window_anchor_scores(output["utility_logit"][0].cpu(), episode)
        tiers = [int(row["utility_tier"]) for row in causal_rows if row["utility_tier"] is not None]
        category = classify_v5_episode_windows(episode.windows)
        best_tier = None
        selected_tier = None
        if causal_rows:
            best_index = int(torch.argmax(causal_scores).item())
            selected_tier = causal_rows[best_index]["utility_tier"]
            best_tier = max(tiers) if tiers else None
        episode_metrics.append({
            "canonical_parent_key": episode.canonical_parent_key,
            "category": category,
            "retrospective_window_count": len(retro_rows),
            "causal_window_count": len(causal_rows),
            "retrospective_best_tier": max((row["utility_tier"] for row in retro_rows if row["utility_tier"] is not None), default=None),
            "causal_selected_tier": selected_tier,
            "causal_best_teacher_tier": best_tier,
            "causal_top1_hit": bool(selected_tier is not None and best_tier is not None and int(selected_tier) == int(best_tier)),
            "emit_count": len(emitted),
            "emit_step": emitted[0]["step"] if emitted else None,
            "selected_window_id": emitted[0]["window_id"] if emitted else None,
            "one_shot_compliant": len(emitted) <= 1,
        })
    true_mixed = [row for row in episode_metrics if row["category"] == "TRUE_MIXED"]
    pure_negative = [row for row in episode_metrics if row["category"] == "PURE_NEGATIVE"]
    summary = {
        "schema": "DETECTOR_V5_CAUSAL_ONLINE_EVALUATION_V2",
        "fold_id": args.fold_id,
        "validation_identity_count": len(episodes),
        "validation_identity_sha256": json_sha(identities),
        "true_mixed_episode_count": len(true_mixed),
        "true_mixed_top1_hit_count": sum(row["causal_top1_hit"] for row in true_mixed),
        "true_mixed_top1_hit_rate": (sum(row["causal_top1_hit"] for row in true_mixed) / len(true_mixed)) if true_mixed else None,
        "pure_negative_episode_count": len(pure_negative),
        "pure_negative_abstain_count": sum(row["emit_count"] == 0 for row in pure_negative),
        "pure_negative_abstain_rate": (sum(row["emit_count"] == 0 for row in pure_negative) / len(pure_negative)) if pure_negative else None,
        "one_shot_compliance": all(row["one_shot_compliant"] for row in episode_metrics),
        "total_emits": sum(row["emit_count"] for row in episode_metrics),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
    }
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        for name, records in (("prediction_records.jsonl", prediction_records), ("scheduler_records.jsonl", scheduler_records), ("episode_metrics.jsonl", episode_metrics)):
            (staging / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        (staging / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({"schema": "DETECTOR_V5_CAUSAL_ONLINE_BUNDLE_V2", "summary_sha256": sha256_file(staging / "evaluation_summary.json"), "protected_splits_read": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(4), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
