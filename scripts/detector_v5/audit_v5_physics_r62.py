#!/usr/bin/env python3
"""CPU-only R6.2 audit for causal Physics targets and training geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.v5_dataset import load_fit_registry, load_policy_intent_root, load_v5_episodes


ANCHOR_DWELL = 10
THRESHOLD = 0.5


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_rows(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / "labels" / str(row["suite"]) / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}" / "physics_teacher_v21.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _jsonl(path)


def _segments(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _label_rows(root, row):
        if bool(item.get("candidate_close")):
            grouped[str(item["window_id"])].append(item)
    result: list[dict[str, Any]] = []
    for window_id, members in sorted(grouped.items()):
        members.sort(key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in members]
        if steps != list(range(steps[0], steps[-1] + 1)):
            raise ValueError(f"non-contiguous Physics segment: {row['canonical_parent_key']}:{window_id}")
        anchor = steps[0] + ANCHOR_DWELL - 1
        anchor_row = next((item for item in members if int(item["step"]) == anchor), None)
        before = [item for item in members if int(item["step"]) <= anchor]
        tiers = [int(item["utility_tier"]) for item in members if item.get("utility_tier") is not None]
        before_tiers = [int(item["utility_tier"]) for item in before if item.get("utility_tier") is not None]
        def onset(predicate: Any) -> int | None:
            return next((int(item["step"]) for item in members if predicate(item)), None)
        result.append({
            "canonical_parent_key": row["canonical_parent_key"],
            "suite": row["suite"],
            "task_idx": int(row["task_idx"]),
            "state_id": int(row["state_id"]),
            "window_id": window_id,
            "segment_start": steps[0],
            "segment_end": steps[-1],
            "segment_length": len(steps),
            "decision_anchor": anchor if anchor_row is not None else None,
            "tier_at_anchor": None if anchor_row is None or anchor_row.get("utility_tier") is None else int(anchor_row["utility_tier"]),
            "max_tier_up_to_anchor": max(before_tiers) if before_tiers else None,
            "final_segment_max_tier": max(tiers) if tiers else None,
            "tier2_onset_step": onset(lambda item: item.get("utility_tier") is not None and int(item["utility_tier"]) >= 2),
            "tier3_onset_step": onset(lambda item: item.get("utility_tier") is not None and int(item["utility_tier"]) >= 3),
            "lift_onset_step": onset(lambda item: float(item.get("lift_score", 0.0)) >= 0.50),
            "stable_grasp_onset": onset(lambda item: float(item.get("stable_grasp_score", 0.0)) >= 0.50),
            "support_removed_onset": onset(lambda item: float(item.get("support_removed", 0.0)) >= 1.0),
            "target_progress_known_at_anchor": bool(anchor_row and anchor_row.get("target_progress_known")),
            "causal_trigger_eligible_at_anchor": bool(anchor_row and anchor_row.get("causal_trigger_eligible")),
            "anchor_local_positive": bool(before_tiers and max(before_tiers) >= 2),
            "anchor_local_negative": bool(before_tiers and max(before_tiers) <= 1),
            "future_promoted_positive": bool(tiers and max(tiers) >= 2 and (not before_tiers or max(before_tiers) < 2)),
            "component_valid_mask_at_anchor": {} if anchor_row is None else dict(anchor_row.get("component_valid_mask", {})),
            "anchor_components": {} if anchor_row is None else {
                "stable_grasp_score": float(anchor_row.get("stable_grasp_score", 0.0)),
                "relative_pose_stability": float(anchor_row.get("relative_pose_stability", 0.0)),
                "object_eef_comotion_score": float(anchor_row.get("object_eef_comotion_score", 0.0)),
                "lift_score": float(anchor_row.get("lift_score", 0.0)),
                "support_removed": float(anchor_row.get("support_removed", 0.0)),
                "target_progress": float(anchor_row.get("target_progress", 0.0)) if anchor_row.get("target_progress_known") else 0.0,
                "gripper_contact_score": float(anchor_row.get("gripper_contact_score", 0.0)),
                "release_risk": float(anchor_row.get("release_risk", 0.0)),
                "regrasp_or_instability_risk": float(anchor_row.get("regrasp_or_instability_risk", 0.0)),
            },
        })
    return result


def _category(segments: list[dict[str, Any]]) -> str:
    positive = any(item["anchor_local_positive"] for item in segments)
    negative = any(item["anchor_local_negative"] for item in segments)
    if positive and negative:
        return "TRUE_MIXED"
    if positive:
        return "POSITIVE_ONLY"
    if negative:
        return "PURE_NEGATIVE"
    return "NO_CANDIDATE"


def _sha_identity(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _subset(train_keys: list[str], segments_by_key: dict[str, list[dict[str, Any]]], rows_by_key: dict[str, dict[str, Any]]) -> list[str]:
    by_task: dict[tuple[str, int], list[str]] = defaultdict(list)
    for key in train_keys:
        row = rows_by_key[key]
        by_task[(str(row["suite"]), int(row["task_idx"]))].append(key)
    selected: list[str] = []
    for task, keys in sorted(by_task.items()):
        ranked = sorted(keys, key=_sha_identity)
        positive = [key for key in ranked if _category(segments_by_key[key]) in {"TRUE_MIXED", "POSITIVE_ONLY"}]
        negative = [key for key in ranked if _category(segments_by_key[key]) == "PURE_NEGATIVE"]
        picks: list[str] = []
        for pool in (positive, negative, ranked):
            for key in pool:
                if key not in picks:
                    picks.append(key)
                    break
        for key in ranked:
            if len(picks) >= 4:
                break
            if key not in picks:
                picks.append(key)
        selected.extend(picks[:4])
    if len(selected) != 160 or len(set(selected)) != 160:
        raise ValueError(f"balanced subset closure failed: {len(selected)}")
    return sorted(selected)


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std, torch.cat((mean, std))


def _fit_linear(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260717)
    train_x, eval_x, norm = _standardize(train_x, eval_x)
    model = nn.Linear(train_x.shape[1], 1)
    positive = float(train_y.sum())
    negative = float(len(train_y) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(250):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x).squeeze(-1), train_y)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(model(eval_x).squeeze(-1)), norm


def _ranking_metrics(rows: list[dict[str, Any]], scores: torch.Tensor) -> dict[str, Any]:
    scored = [dict(row, score=float(score)) for row, score in zip(rows, scores)]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_key[row["canonical_parent_key"]].append(row)
    positive = [row for row in scored if row["anchor_local_positive"]]
    negative = [row for row in scored if row["anchor_local_negative"]]
    mixed = [values for values in by_key.values() if any(item["anchor_local_positive"] for item in values) and any(item["anchor_local_negative"] for item in values)]
    pure = [values for values in by_key.values() if values and all(item["anchor_local_negative"] for item in values)]
    top1 = [max(values, key=lambda item: item["score"])["anchor_local_positive"] for values in mixed]
    top2 = [any(item["anchor_local_positive"] for item in sorted(values, key=lambda item: item["score"], reverse=True)[:2]) for values in mixed]
    pairwise = sum(float(pos["score"] > neg["score"]) + 0.5 * float(pos["score"] == neg["score"]) for pos in positive for neg in negative)
    pair_count = len(positive) * len(negative)
    return {
        "segment_count": len(scored),
        "positive_segment_count": len(positive),
        "negative_segment_count": len(negative),
        "positive_recall_at_0.5": sum(row["score"] >= THRESHOLD for row in positive) / len(positive) if positive else None,
        "true_mixed_episode_count": len(mixed),
        "true_mixed_top1": sum(top1) / len(top1) if top1 else None,
        "true_mixed_top2_recall": sum(top2) / len(top2) if top2 else None,
        "pure_negative_episode_count": len(pure),
        "pure_negative_abstention_at_0.5": sum(max(item["score"] for item in values) < THRESHOLD for values in pure) / len(pure) if pure else None,
        "pairwise_positive_over_negative": pairwise / pair_count if pair_count else None,
        "score_min": min((row["score"] for row in scored), default=None),
        "score_max": max((row["score"] for row in scored), default=None),
    }


def _baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[row["canonical_parent_key"]].append(row)
    methods = {
        "earliest_causal_segment": lambda values: min(values, key=lambda item: item["segment_start"]),
        "latest_causal_segment": lambda values: max(values, key=lambda item: item["segment_start"]),
        "longest_causal_segment": lambda values: max(values, key=lambda item: (item["segment_length"], -item["segment_start"])),
    }
    output: dict[str, Any] = {}
    for name, pick in methods.items():
        mixed = [values for values in by_key.values() if any(item["anchor_local_positive"] for item in values) and any(item["anchor_local_negative"] for item in values)]
        pure = [values for values in by_key.values() if values and all(item["anchor_local_negative"] for item in values)]
        selected = [pick(values) for values in by_key.values() if values]
        output[name] = {
            "true_mixed_top1": sum(bool(pick(values)["anchor_local_positive"]) for values in mixed) / len(mixed) if mixed else None,
            "pure_negative_abstention": 0.0,
            "selected_positive_rate": sum(bool(item["anchor_local_positive"]) for item in selected) / len(selected) if selected else None,
            "mixed_denominator": len(mixed),
            "pure_negative_denominator": len(pure),
        }
    return output


def _shallow_bounds(train_segments: list[dict[str, Any]], validation_segments: list[dict[str, Any]], feature_rows: dict[str, list[dict[str, Any]]], episodes_by_key: dict[str, Any], policy_by_key: dict[str, list[dict[str, Any]]] | None) -> dict[str, Any]:
    def build(rows: list[dict[str, Any]], kind: str) -> torch.Tensor:
        values: list[list[float]] = []
        for row in rows:
            anchor = int(row["decision_anchor"])
            if kind == "25d":
                values.append([float(value) for value in episodes_by_key[row["canonical_parent_key"]].features_25d[anchor]])
            elif kind == "policy":
                if policy_by_key is None:
                    raise ValueError("policy features requested without policy root")
                values.append([float(value) for value in policy_by_key[row["canonical_parent_key"]][anchor]["clean_policy_intent_9d"]])
            elif kind == "25d_policy":
                if policy_by_key is None:
                    raise ValueError("policy features requested without policy root")
                values.append(
                    [float(value) for value in episodes_by_key[row["canonical_parent_key"]].features_25d[anchor]]
                    + [float(value) for value in policy_by_key[row["canonical_parent_key"]][anchor]["clean_policy_intent_9d"]]
                )
            elif kind == "privileged":
                values.append([float(value) for value in row["anchor_components"].values()])
        return torch.tensor(values, dtype=torch.float32)

    train_y = torch.tensor([float(row["anchor_local_positive"]) for row in train_segments], dtype=torch.float32)
    eval_y = torch.tensor([float(row["anchor_local_positive"]) for row in validation_segments], dtype=torch.float32)
    output: dict[str, Any] = {}
    for name in ("25d", "25d_policy", "privileged"):
        if name == "25d_policy" and policy_by_key is None:
            output[name] = {"status": "HOLD_POLICY_ROOT_NOT_PROVIDED"}
            continue
        train_x = build(train_segments, name)
        eval_x = build(validation_segments, name)
        scores, norm = _fit_linear(train_x, train_y, eval_x)
        output[name] = {
            "status": "PASS",
            "features": int(train_x.shape[1]),
            "train_positive_fraction": float(train_y.mean()),
            "validation_positive_fraction": float(eval_y.mean()),
            "metrics": _ranking_metrics(validation_segments, scores),
            "normalization_train_only": True,
            "normalization_shape": list(norm.shape),
            "privileged_future_evidence": name == "privileged",
        }
    return output


def _write_sealed(root: Path, payloads: dict[str, str]) -> None:
    if root.exists():
        raise FileExistsError(root)
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        for name, value in payloads.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value, encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--old-subset-identities", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.registry_root, args.s1_root, args.teacher_root, args.fold_root]
    if args.policy_root:
        roots.append(args.policy_root)
    for root in roots:
        verify_sealed_directory(root.resolve())
    rows = load_fit_registry(args.registry_csv.resolve())
    rows_by_key = {row["canonical_parent_key"]: row for row in rows}
    policy_index = None
    policy_meta = None
    if args.policy_root:
        policy_index, policy_meta = load_policy_intent_root(args.policy_root.resolve())
    episodes = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), rows, policy_index=policy_index)
    episodes_by_key = {episode.canonical_parent_key: episode for episode in episodes}
    segments_by_key = {key: _segments(args.teacher_root.resolve(), row) for key, row in rows_by_key.items()}
    fold_manifest = _json(args.fold_root.resolve() / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json")
    fold = next(item for item in fold_manifest["folds"] if int(item["fold_id"]) == 0)
    train_keys = list(fold["train_identities"])
    validation_keys = list(fold["validation_identities"])
    old_subset_value = _json(args.old_subset_identities.resolve())
    old_subset = old_subset_value.get("identities", old_subset_value) if isinstance(old_subset_value, dict) else old_subset_value
    balanced = _subset(train_keys, segments_by_key, rows_by_key)
    balanced_payload = {
        "schema": "OFFICIAL_V3_DETECTOR_V5_PHYSICS_V21_CATEGORY_BALANCED_SUBSET_V1",
        "selection": "per-task deterministic hash order; positive-containing and pure-negative preferred when available",
        "seed": 20260717,
        "fold_id": 0,
        "identity_count": len(balanced),
        "identities": balanced,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    _write_sealed(args.subset_root.resolve(), {
        "identities.json": json.dumps(balanced_payload, indent=2, sort_keys=True) + "\n",
        "input_binding.json": json.dumps({
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root.resolve() / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root.resolve() / "SHA256SUMS"),
            "source_identities_sha256": sha256_file(args.fold_root.resolve() / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json"),
            "protected_splits_read": [],
        }, indent=2, sort_keys=True) + "\n",
    })
    all_segment_rows = [segment for key in sorted(segments_by_key) for segment in segments_by_key[key]]
    train_segments = [segment for key in train_keys for segment in segments_by_key[key] if segment["decision_anchor"] is not None]
    validation_segments = [segment for key in validation_keys for segment in segments_by_key[key] if segment["decision_anchor"] is not None]
    old_segments = [segment for key in old_subset for segment in segments_by_key.get(key, [])]
    balanced_segments = [segment for key in balanced for segment in segments_by_key[key]]
    category_sets = {
        "fit_train_600": train_keys,
        "fit_validation_200": validation_keys,
        "old_smoke_train": list(old_subset),
        "new_balanced_train_160": balanced,
    }
    episode_geometry = {}
    for name, keys in category_sets.items():
        selected = [segments_by_key[key] for key in keys]
        episode_geometry[name] = {
            "identity_count": len(keys),
            "category_counts": dict(Counter(_category(segments) for segments in selected)),
            "final_tier_segment_counts": dict(Counter(int(segment["final_segment_max_tier"]) for segments in selected for segment in segments if segment["final_segment_max_tier"] is not None)),
            "anchor_tier_segment_counts": dict(Counter(int(segment["max_tier_up_to_anchor"]) for segments in selected for segment in segments if segment["max_tier_up_to_anchor"] is not None)),
            "tier3_containing_episode_count": sum(any(int(segment["final_segment_max_tier"] or -1) >= 3 for segment in segments) for segments in selected),
            "segment_count": sum(len(segments) for segments in selected),
            "anchor_segment_count": sum(sum(segment["decision_anchor"] is not None for segment in segments) for segments in selected),
        }
    causal = {
        "schema": "DETECTOR_V5_PHYSICS_R62_CAUSAL_TARGET_AUDIT_V1",
        "identity_count": len(rows),
        "segment_count": len(all_segment_rows),
        "segments_with_anchor": sum(segment["decision_anchor"] is not None for segment in all_segment_rows),
        "future_promoted_positive_count": sum(segment["future_promoted_positive"] for segment in all_segment_rows),
        "future_promoted_positive_rate": sum(segment["future_promoted_positive"] for segment in all_segment_rows) / len(all_segment_rows),
        "anchor_tier_differs_from_final_count": sum(segment["decision_anchor"] is not None and segment["tier_at_anchor"] != segment["final_segment_max_tier"] for segment in all_segment_rows),
        "anchor_tier_differs_from_final_rate": sum(segment["decision_anchor"] is not None and segment["tier_at_anchor"] != segment["final_segment_max_tier"] for segment in all_segment_rows) / max(1, sum(segment["decision_anchor"] is not None for segment in all_segment_rows)),
        "tier2_or_3_onset_after_anchor_count": sum(segment["decision_anchor"] is not None and segment["tier2_onset_step"] is not None and segment["tier2_onset_step"] > segment["decision_anchor"] for segment in all_segment_rows),
        "tier2_or_3_onset_after_anchor_rate": sum(segment["decision_anchor"] is not None and segment["tier2_onset_step"] is not None and segment["tier2_onset_step"] > segment["decision_anchor"] for segment in all_segment_rows) / max(1, sum(segment["decision_anchor"] is not None for segment in all_segment_rows)),
        "anchor_local_category_counts": dict(Counter(_category(segments_by_key[key]) for key in rows_by_key)),
        "final_loader_category_counts": dict(Counter("POSITIVE" if any(window.utility_tier is not None and int(window.utility_tier) >= 2 for window in episodes_by_key[key].windows) and not all(window.utility_tier is not None and int(window.utility_tier) >= 2 for window in episodes_by_key[key].windows) else "OTHER" for key in rows_by_key)),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    bounds = _shallow_bounds(train_segments, validation_segments, {key: segments_by_key[key] for key in rows_by_key}, episodes_by_key, policy_index)
    payloads = {
        "causal_target_summary.json": json.dumps(causal, indent=2, sort_keys=True) + "\n",
        "causal_target_rows.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_segment_rows),
        "episode_geometry.json": json.dumps(episode_geometry, indent=2, sort_keys=True) + "\n",
        "shallow_bounds.json": json.dumps(bounds, indent=2, sort_keys=True) + "\n",
        "baseline_results.json": json.dumps(_baseline_metrics(validation_segments), indent=2, sort_keys=True) + "\n",
        "balanced_subset_summary.json": json.dumps({
            "identity_count": len(balanced),
            "identity_sha256": sha256_file(args.subset_root.resolve() / "identities.json"),
            "category_counts": dict(Counter(_category(segments_by_key[key]) for key in balanced)),
            "per_task_counts": dict(Counter(f"{rows_by_key[key]['suite']}/task_{int(rows_by_key[key]['task_idx']):02d}" for key in balanced)),
        }, indent=2, sort_keys=True) + "\n",
        "input_binding.json": json.dumps({
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "registry_root_sha256s_sha256": sha256_file(args.registry_root.resolve() / "SHA256SUMS"),
            "s1_root_sha256s_sha256": sha256_file(args.s1_root.resolve() / "SHA256SUMS"),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root.resolve() / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root.resolve() / "SHA256SUMS"),
            "policy_root_sha256s_sha256": None if policy_meta is None else policy_meta["policy_root_sha256s_sha256"],
            "old_subset_sha256": sha256_file(args.old_subset_identities.resolve()),
            "balanced_subset_sha256": sha256_file(args.subset_root.resolve() / "identities.json"),
            "protected_splits_read": [],
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }, indent=2, sort_keys=True) + "\n",
    }
    _write_sealed(args.output_root.resolve(), payloads)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "subset_root": str(args.subset_root.resolve()), "identity_count": len(rows), "segment_count": len(all_segment_rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
