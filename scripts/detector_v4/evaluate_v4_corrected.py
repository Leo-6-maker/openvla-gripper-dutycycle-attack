#!/usr/bin/env python3
"""Phase/window-aware FIT validation for corrected V4 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from gripper_attack.v4_contract import FIT_STATES, identity_sha, sha256_file, verify_checksum_manifest
from gripper_attack.v4_dataset import load_v4_episode, select_fold_episodes
from gripper_attack.v4_formal import V4Normalization, V4StatefulQualityGRU


def _seal(root: Path) -> None:
    payloads = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}), key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(p)}  {str(p.relative_to(root)).replace(os.sep, '/')}\n" for p in payloads), encoding="utf-8")
    value = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{value}  SHA256SUMS\n", encoding="utf-8")


def _load_fold_validation(fold_root: Path, fold_id: int) -> list[str]:
    verify_checksum_manifest(fold_root)
    paths = [fold_root / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json", fold_root / "fold_manifest.json", fold_root / "manifest.json"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise ValueError("fold manifest not found")
    value = json.loads(path.read_text(encoding="utf-8"))
    row = next((item for item in value.get("folds", []) if int(item.get("fold_id", -1)) == fold_id), None)
    if row is None:
        raise ValueError("fold missing")
    identities = sorted(set(row.get("validation_identities", [])))
    if len(identities) != 200:
        raise ValueError("validation identity count is not 200")
    return identities


def select_working_point(threshold_metrics: list[dict[str, Any]], decision_config: dict[str, Any]) -> dict[str, Any]:
    """Select the pre-recorded working point without inspecting future splits."""
    rule = decision_config.get("working_point_rule")
    if rule != "maximum_threshold_with_valid_event_hit_gte_0.95":
        raise ValueError(f"unsupported working-point rule: {rule}")
    minimum = float(decision_config.get("valid_event_hit_minimum", 0.95))
    eligible = [row for row in threshold_metrics if float(row["valid_event_hit"]) >= minimum]
    if not eligible:
        return {
            "status": "HOLD",
            "reason": "NO_VALID_THRESHOLD",
            "minimum_valid_event_hit": minimum,
            "threshold": None,
            "rule": rule,
        }
    selected = max(eligible, key=lambda row: float(row["threshold"]))
    return {
        "status": "PASS",
        "reason": "VALID_THRESHOLD_SELECTED",
        "minimum_valid_event_hit": minimum,
        "threshold": float(selected["threshold"]),
        "rule": rule,
        "selected_metrics": selected,
    }


def _episode_records(
    ep, probs: torch.Tensor, fold_id: int, seed: int, view: str,
    working_point_threshold: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for step in range(ep.n_steps):
        raw_emit = None if working_point_threshold is None else bool(probs[step] >= working_point_threshold)
        candidate_gated_emit = None if raw_emit is None else bool(
            ep.student_valid_mask[step] and ep.candidate_close[step] and raw_emit
        )
        rows.append({
            "schema": "DETECTOR_V4_PREDICTION_STEP_V2",
            "canonical_parent_key": ep.canonical_parent_key,
            "suite": ep.suite, "task_idx": ep.task_idx, "state_id": ep.state_id,
            "step": step, "fold_id": fold_id, "seed": seed, "view": view,
            "quality_probability": float(probs[step]),
            "student_valid": bool(ep.student_valid_mask[step]),
            "candidate_close": bool(ep.candidate_close[step]),
            "raw_quality_emit": raw_emit,
            "candidate_gated_emit": candidate_gated_emit,
            "working_point_threshold": working_point_threshold,
            "event_id": int(ep.event_id[step]),
            "phase_id": int(ep.phase_id[step]),
            "window_id": int(ep.window_id[step]),
            "quality_target": None if ep.quality_target[step] < 0 else float(ep.quality_target[step]),
            "quality_supervision": bool(ep.quality_supervision_mask[step]),
            "release_target": None if ep.release_target[step] < 0 else float(ep.release_target[step]),
            "release_supervision": bool(ep.release_supervision_mask[step]),
        })
    return rows


def _metrics(episodes, probabilities: dict[str, torch.Tensor], threshold: float) -> dict[str, Any]:
    valid_events = []
    later_events = []
    invalid_windows = []
    release_steps = 0
    release_overlap = 0
    pure_negative = []
    mixed = []
    suite_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ep in episodes:
        prob = probabilities[ep.canonical_parent_key]
        emit = ep.student_valid_mask & ep.candidate_close & (prob >= threshold)
        qmask = ep.quality_supervision_mask
        positive = qmask & (ep.quality_target > 0.5) & (ep.event_id >= 0)
        negative = qmask & (ep.quality_target < 0.5) & (ep.event_id >= 0)
        pos_ids = sorted(set(int(v) for v in ep.event_id[positive].tolist()))
        neg_ids = sorted(set(int(v) for v in ep.event_id[negative].tolist()))
        for event_id in pos_ids:
            mask = positive & (ep.event_id == event_id)
            hit = bool(emit[mask].any())
            valid_events.append(hit)
            if event_id >= 1:
                later_events.append(hit)
            suite_counts[ep.suite][0] += 1
            suite_counts[ep.suite][1] += int(hit)
            task_key = f"{ep.suite}/task_{ep.task_idx:02d}"
            task_counts[task_key][0] += 1
            task_counts[task_key][1] += int(hit)
        for event_id in neg_ids:
            mask = negative & (ep.event_id == event_id)
            invalid_windows.append(bool(emit[mask].any()))
        if not pos_ids and neg_ids:
            pure_negative.append(bool(emit[negative].any()))
        if pos_ids and neg_ids:
            mixed.append(bool(emit[negative].any()))
        rel = ep.release_supervision_mask & (ep.release_target > 0.5)
        release_steps += int(rel.sum())
        release_overlap += int((emit & rel).sum())
    denom = lambda n: max(1, n)
    suite_macro = sum(v[1] / denom(v[0]) for v in suite_counts.values()) / denom(len(suite_counts))
    task_macro = sum(v[1] / denom(v[0]) for v in task_counts.values()) / denom(len(task_counts))
    return {
        "threshold": threshold,
        "valid_event_hit": sum(valid_events) / denom(len(valid_events)),
        "valid_event_hit_n": sum(valid_events), "valid_event_count": len(valid_events),
        "later_event_hit": sum(later_events) / denom(len(later_events)),
        "later_event_hit_n": sum(later_events), "later_event_count": len(later_events),
        "invalid_window_any_emit": sum(invalid_windows) / denom(len(invalid_windows)),
        "invalid_window_emit_n": sum(invalid_windows), "invalid_window_count": len(invalid_windows),
        "pure_negative_any_emit": sum(pure_negative) / denom(len(pure_negative)),
        "pure_negative_emit_n": sum(pure_negative), "pure_negative_count": len(pure_negative),
        "mixed_invalid_any_emit": sum(mixed) / denom(len(mixed)),
        "mixed_invalid_emit_n": sum(mixed), "mixed_episode_count": len(mixed),
        "release_overlap": release_overlap / denom(release_steps),
        "release_overlap_n": release_overlap, "release_step_count": release_steps,
        "suite_macro": suite_macro, "task_macro": task_macro,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    checkpoint_seal = verify_checksum_manifest(args.checkpoint_root)
    s1_seal = verify_checksum_manifest(args.s1_root)
    teacher_seal = verify_checksum_manifest(args.teacher_root)
    decision_config_path = Path(args.decision_config)
    decision_config = json.loads(decision_config_path.read_text(encoding="utf-8"))
    if decision_config.get("schema") != "DETECTOR_V4_DECISION_CONFIG_V2":
        raise ValueError("wrong V4 decision config schema")
    if decision_config.get("formal_training_authorized") is not False or decision_config.get("formal_attack_authorized") is not False:
        raise ValueError("decision config cannot authorize training or attack")
    validation_ids = _load_fold_validation(args.fold_root, args.fold_id)
    ckpt_path = args.checkpoint_root / "checkpoint.pt"
    manifest = json.loads((args.checkpoint_root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("fold_id", -1)) != args.fold_id or int(manifest.get("seed", -1)) != args.seed:
        raise ValueError("checkpoint coordinate mismatch")
    if sha256_file(ckpt_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA mismatch")
    view = manifest["view"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalization = V4Normalization.from_dict(checkpoint["normalization"])
    model = V4StatefulQualityGRU(normalization.feature_count, aux_release=bool(manifest.get("aux_release", False)))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    episodes = []
    for identity in validation_ids:
        suite, task_name, state_name = identity.split("/")
        ep = load_v4_episode(args.s1_root, args.teacher_root, suite, int(task_name.split("_", 1)[1]), int(state_name.split("_", 1)[1]), view)
        if ep is None:
            raise ValueError(f"missing validation episode: {identity}")
        episodes.append(ep)
    if len(episodes) != 200 or any(ep.state_id not in FIT_STATES for ep in episodes):
        raise ValueError("validation set is not the exact FIT fold")
    probabilities = {}
    with torch.no_grad():
        for ep in episodes:
            x = normalization.normalize(ep.features.unsqueeze(0))
            valid = ep.student_valid_mask.unsqueeze(0)
            boundary = torch.zeros_like(valid)
            boundary[:, 0] = True
            probs = torch.sigmoid(model(x, valid, boundary)["quality"][0])
            probabilities[ep.canonical_parent_key] = probs
    thresholds = [float(value) for value in decision_config.get("threshold_grid", [])]
    if thresholds != sorted(set(thresholds)) or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("decision config threshold_grid must be sorted, unique, and in [0, 1]")
    if not thresholds:
        raise ValueError("decision config threshold_grid is empty")
    metrics = [_metrics(episodes, probabilities, threshold) for threshold in thresholds]
    working_point = select_working_point(metrics, decision_config)
    selected_threshold = working_point["threshold"]
    records = []
    for ep in episodes:
        prob = probabilities[ep.canonical_parent_key]
        records.extend(_episode_records(ep, prob, args.fold_id, args.seed, view, selected_threshold))
    payload = {
        "schema": "DETECTOR_V4_FIT_VALIDATION_BUNDLE_V2",
        "status": "PASS" if working_point["status"] == "PASS" else "HOLD",
        "fold_id": args.fold_id, "seed": args.seed, "view": view,
        "validation_identity_count": len(validation_ids),
        "validation_identity_sha256": identity_sha(validation_ids),
        "checkpoint_root_sha256s_sha256": checkpoint_seal["sha256sums_sha256"],
        "s1_root_sha256s_sha256": s1_seal["sha256sums_sha256"],
        "teacher_root_sha256s_sha256": teacher_seal["sha256sums_sha256"],
        "decision_config_sha256": sha256_file(decision_config_path),
        "threshold_metrics": metrics,
        "working_point": working_point,
        "emission_rule": "student_valid AND candidate_close AND quality_probability >= threshold",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    staging = args.output_root.parent / f".{args.output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "prediction_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        (staging / "evaluation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _seal(staging)
        os.replace(staging, args.output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--s1-root", type=Path, required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, required=True)
    p.add_argument("--fold-id", type=int, choices=range(4), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--decision-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(evaluate(p.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
