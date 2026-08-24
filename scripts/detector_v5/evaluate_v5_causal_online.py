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
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.v5_dataset import aggregate_retrospective_window_scores, causal_window_anchor_scores, classify_v5_episode_windows, load_fit_registry, load_policy_intent_root, load_v5_episodes
from gripper_attack.v5_protocol import V5ModelContract, variant_uses_intent
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _window_for_step(episode: Any, step: int) -> str | None:
    containing = [window for window in episode.windows if window.rankable and step in window.step_indices]
    if containing:
        return containing[0].window_id
    return None


def _scheduler_replay(
    episode: Any,
    utility: torch.Tensor,
    release: torch.Tensor,
    regrasp: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    scheduler = V5OneShotScheduler(V5SchedulerConfig(utility_threshold=threshold, uncertainty_veto_enabled=False))
    emitted_step: int | None = None
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
        if result["emit"] and emitted_step is None:
            emitted_step = step
    selected_window_id = _window_for_step(episode, emitted_step) if emitted_step is not None else None
    selected_window = next((window for window in episode.windows if window.window_id == selected_window_id), None)
    category = classify_v5_episode_windows(episode.windows)
    positive_tiers = [int(window.utility_tier) for window in episode.windows if window.rankable and window.utility_tier is not None and int(window.utility_tier) >= 2]
    selected_tier = None if selected_window is None else int(selected_window.utility_tier)
    return {
        "canonical_parent_key": episode.canonical_parent_key,
        "suite": episode.suite,
        "task_idx": episode.task_idx,
        "category": category,
        "emit": emitted_step is not None,
        "emit_step": emitted_step,
        "selected_window_id": selected_window_id,
        "selected_tier": selected_tier,
        "selected_highest_tier": bool(selected_tier is not None and positive_tiers and selected_tier == max(positive_tiers)),
        "selected_tier_ge2": bool(selected_tier is not None and selected_tier >= 2),
        "outside_rankable_emit": bool(emitted_step is not None and selected_window is None),
        "release_trigger": bool(emitted_step is not None and episode.release_imminent[emitted_step]),
        "regrasp_trigger": bool(emitted_step is not None and episode.regrasp_or_unstable[emitted_step]),
        "one_shot_compliant": scheduler.emitted is True if emitted_step is not None else True,
    }


def _threshold_results(
    episodes: list[Any],
    replay_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        _scheduler_replay(episode, utility, release, regrasp, threshold)
        for episode, (utility, release, regrasp) in zip(episodes, replay_inputs)
    ]


def _threshold_sweep(episodes: list[Any], replay_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(5, 96, 5):
        threshold = index / 100.0
        results = _threshold_results(episodes, replay_inputs, threshold)
        positive = [row for row in results if row["category"] in ("TRUE_MIXED", "POSITIVE_ONLY")]
        mixed = [row for row in results if row["category"] == "TRUE_MIXED"]
        pure_negative = [row for row in results if row["category"] == "PURE_NEGATIVE"]
        no_candidate = [row for row in results if row["category"] == "NO_CANDIDATE"]
        selected = [row for row in results if row["emit"] and row["selected_window_id"] is not None]
        rows.append({
            "threshold": threshold,
            "positive_episode_count": len(positive),
            "critical_window_recall": (sum(row["selected_tier_ge2"] for row in positive) / len(positive)) if positive else None,
            "mixed_episode_count": len(mixed),
            "mixed_correct_selection": (sum(row["selected_highest_tier"] for row in mixed) / len(mixed)) if mixed else None,
            "pure_negative_episode_count": len(pure_negative),
            "pure_negative_abstention": (sum(not row["emit"] for row in pure_negative) / len(pure_negative)) if pure_negative else None,
            "no_candidate_episode_count": len(no_candidate),
            "no_candidate_abstention": (sum(not row["emit"] for row in no_candidate) / len(no_candidate)) if no_candidate else None,
            "scheduler_selected_highest_tier_count": sum(row["selected_highest_tier"] for row in results),
            "scheduler_selected_tier_ge2_count": sum(row["selected_tier_ge2"] for row in results),
            "scheduler_selected_tier_ge2_precision": (sum(row["selected_tier_ge2"] for row in selected) / len(selected)) if selected else None,
            "total_emits": sum(row["emit"] for row in results),
            "release_trigger_count": sum(row["release_trigger"] for row in results),
            "regrasp_trigger_count": sum(row["regrasp_trigger"] for row in results),
            "outside_rankable_emit_count": sum(row["outside_rankable_emit"] for row in results),
            "one_shot_compliance": all(row["one_shot_compliant"] for row in results),
        })
    return rows


def _select_working_point(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("critical_window_recall") is not None
        and float(row["critical_window_recall"]) >= 0.95
    ]
    if not eligible:
        return {
            "status": "HOLD",
            "selected_threshold": None,
            "reason": "NO_THRESHOLD_MEETS_CRITICAL_WINDOW_RECALL_0.95",
        }
    selected = max(eligible, key=lambda row: float(row["threshold"]))
    return {
        "status": "PASS",
        "selected_threshold": float(selected["threshold"]),
        "reason": "MAXIMUM_THRESHOLD_WITH_CRITICAL_WINDOW_RECALL_GTE_0.95",
        "selected_row": selected,
    }


def _macro_rate(rows: list[dict[str, Any]], metric: str, group_key: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    values = [sum(bool(row[metric]) for row in group) / len(group) for group in grouped.values() if group]
    return sum(values) / len(values) if values else None


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    contract_value = checkpoint.get("model_contract", {})
    if not isinstance(contract_value, dict) or not isinstance(contract_value.get("variant"), str):
        raise ValueError("checkpoint does not contain a V5 model contract")
    contract = V5ModelContract(str(contract_value["variant"]), visual_dim=int(contract_value.get("visual_dim", 0)))
    model = CausalMultimodalVulnerabilityRanker(contract)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    verify_sealed_directory(args.s1_root.resolve())
    verify_sealed_directory(args.teacher_root.resolve())
    policy_index = None
    policy_meta = None
    if variant_uses_intent(contract.variant):
        if args.policy_intent_root is None:
            raise ValueError("V5-B checkpoint evaluation requires --policy-intent-root")
        policy_index, policy_meta = load_policy_intent_root(args.policy_intent_root)
    elif args.policy_intent_root is not None:
        raise ValueError("V5-A evaluator must not consume policy-intent root")
    rows = load_fit_registry(args.registry_csv.resolve())
    fold = load_fit_fold_bundle(args.fold_root.resolve())
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == args.fold_id)
    by_key = {row["canonical_parent_key"]: row for row in rows}
    identities = list(fold_row["validation_identities"])
    episodes = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in identities], policy_index=policy_index)
    mean = checkpoint["normalization_mean_25d"].to(device)
    std = checkpoint["normalization_std_25d"].to(device)
    intent_mean = checkpoint.get("normalization_mean_9d")
    intent_std = checkpoint.get("normalization_std_9d")
    if variant_uses_intent(contract.variant):
        if not isinstance(intent_mean, torch.Tensor) or not isinstance(intent_std, torch.Tensor):
            raise ValueError("V5-B checkpoint is missing 9D normalization")
        intent_mean = intent_mean.to(device)
        intent_std = intent_std.to(device)
    prediction_records: list[dict[str, Any]] = []
    scheduler_records: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    replay_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for episode in episodes:
        x = ((episode.features_25d.to(device) - mean) / std).unsqueeze(0)
        intent = None
        if variant_uses_intent(contract.variant):
            assert isinstance(intent_mean, torch.Tensor) and isinstance(intent_std, torch.Tensor)
            intent = ((episode.policy_intent_9d.to(device) - intent_mean) / intent_std).unsqueeze(0)
        output = model.forward_sequence(x, intent=intent, valid_mask=episode.valid_mask.to(device).unsqueeze(0))
        utility = torch.sigmoid(output["utility_logit"][0]).cpu()
        release = torch.sigmoid(output["release_logit"][0]).cpu()
        regrasp = torch.sigmoid(output["regrasp_logit"][0]).cpu()
        replay_inputs.append((utility, release, regrasp))
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
                "raw_quality_emit": bool(float(utility[step]) >= 0.5),
                "candidate_gated_emit": bool(
                    episode.valid_mask[step]
                    and episode.candidate_close[step]
                    and float(utility[step]) >= 0.5
                ),
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
                "selected_window_tier": next((int(window.utility_tier) for window in episode.windows if emitted and window.window_id == emitted[0]["window_id"]), None),
                "release_trigger": bool(emitted and episode.release_imminent[emitted[0]["step"]]),
                "regrasp_trigger": bool(emitted and episode.regrasp_or_unstable[emitted[0]["step"]]),
                "one_shot_compliant": len(emitted) <= 1,
        })
    true_mixed = [row for row in episode_metrics if row["category"] == "TRUE_MIXED"]
    pure_negative = [row for row in episode_metrics if row["category"] == "PURE_NEGATIVE"]
    threshold_sweep = _threshold_sweep(episodes, replay_inputs)
    working_point = _select_working_point(threshold_sweep)
    selected_threshold = working_point["selected_threshold"]
    selected_results = [] if selected_threshold is None else _threshold_results(episodes, replay_inputs, float(selected_threshold))
    selected_positive = [row for row in selected_results if row["category"] in ("TRUE_MIXED", "POSITIVE_ONLY")]
    selected_mixed = [row for row in selected_results if row["category"] == "TRUE_MIXED"]
    selected_pure_negative = [row for row in selected_results if row["category"] == "PURE_NEGATIVE"]
    selected_no_candidate = [row for row in selected_results if row["category"] == "NO_CANDIDATE"]
    selected_emits = [row for row in selected_results if row["emit"]]
    selected_rankable_emits = [row for row in selected_emits if row["selected_window_id"] is not None]
    exact_ids = {
        "scheduler_selected_highest_tier": sorted(row["canonical_parent_key"] for row in selected_results if row["selected_highest_tier"]),
        "scheduler_selected_tier_ge2": sorted(row["canonical_parent_key"] for row in selected_results if row["selected_tier_ge2"]),
        "pure_negative": sorted(row["canonical_parent_key"] for row in selected_pure_negative),
        "outside_rankable": sorted(row["canonical_parent_key"] for row in selected_results if row["outside_rankable_emit"]),
        "release_trigger": sorted(row["canonical_parent_key"] for row in selected_results if row["release_trigger"]),
        "regrasp_trigger": sorted(row["canonical_parent_key"] for row in selected_results if row["regrasp_trigger"]),
    }
    summary = {
        "schema": "DETECTOR_V5_CAUSAL_ONLINE_EVALUATION_V3",
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
        "candidate": contract.variant,
        "policy_intent_consumed": variant_uses_intent(contract.variant),
        "policy_intent_root_sha256s_sha256": None if policy_meta is None else policy_meta["policy_root_sha256s_sha256"],
        "scheduler_selected_tier_ge2_count": sum(row["selected_window_tier"] is not None and row["selected_window_tier"] >= 2 for row in episode_metrics),
        "scheduler_release_trigger_count": sum(row["release_trigger"] for row in episode_metrics),
        "scheduler_regrasp_trigger_count": sum(row["regrasp_trigger"] for row in episode_metrics),
        "scheduler_outside_rankable_emit_count": sum(row["emit_count"] > 0 and row["selected_window_id"] is None for row in episode_metrics),
        "threshold_working_point_rule": "maximum_threshold_with_critical_window_recall_gte_0.95",
        "working_point_status": working_point["status"],
        "selected_threshold": selected_threshold,
        "critical_window_recall": (
            sum(row["selected_tier_ge2"] for row in selected_positive) / len(selected_positive)
            if selected_positive else None
        ),
        "mixed_scheduler_correct_selection": (
            sum(row["selected_highest_tier"] for row in selected_mixed) / len(selected_mixed)
            if selected_mixed else None
        ),
        "selected_tier_ge2_precision": (
            sum(row["selected_tier_ge2"] for row in selected_rankable_emits) / len(selected_rankable_emits)
            if selected_rankable_emits else None
        ),
        "pure_negative_abstention": (
            sum(not row["emit"] for row in selected_pure_negative) / len(selected_pure_negative)
            if selected_pure_negative else None
        ),
        "no_candidate_abstention": (
            sum(not row["emit"] for row in selected_no_candidate) / len(selected_no_candidate)
            if selected_no_candidate else None
        ),
        "selected_total_emits": len(selected_emits),
        "selected_outside_rankable_emits": sum(row["outside_rankable_emit"] for row in selected_results),
        "selected_release_trigger_count": sum(row["release_trigger"] for row in selected_results),
        "selected_regrasp_trigger_count": sum(row["regrasp_trigger"] for row in selected_results),
        "selected_one_shot_compliance": all(row["one_shot_compliant"] for row in selected_results),
        "suite_macro_critical_window_recall": _macro_rate(selected_positive, "selected_tier_ge2", "suite"),
        "task_macro_critical_window_recall": _macro_rate(selected_positive, "selected_tier_ge2", "task_idx"),
        "exact_identities": exact_ids,
        "causal_anchor_top1": {
            "count": sum(row["causal_top1_hit"] for row in true_mixed),
            "denominator": len(true_mixed),
            "rate": (sum(row["causal_top1_hit"] for row in true_mixed) / len(true_mixed)) if true_mixed else None,
        },
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
        (staging / "threshold_sweep.json").write_text(json.dumps(threshold_sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({"schema": "DETECTOR_V5_CAUSAL_ONLINE_BUNDLE_V3", "summary_sha256": sha256_file(staging / "evaluation_summary.json"), "threshold_sweep_sha256": sha256_file(staging / "threshold_sweep.json"), "candidate": contract.variant, "policy_intent_root_sha256s_sha256": None if policy_meta is None else policy_meta["policy_root_sha256s_sha256"], "protected_splits_read": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
