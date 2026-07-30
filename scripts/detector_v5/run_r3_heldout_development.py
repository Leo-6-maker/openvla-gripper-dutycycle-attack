"""Episode/task-held-out V5 Student development evaluation.

This is FIT-only development work.  It consumes sealed T4/G0/G1/G2 inputs,
initializes a new Student, selects thresholds on validation only, and reads
the test split once.  It never consumes the all-670 engineering checkpoint,
Teacher privileged features, protected roots, or runtime/attack inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_teacher import HEADS
from run_r3_full670_student_development import (
    ACTIVE_HEADS,
    INACTIVE_HEADS,
    _load_model,
    _load_records,
    _loss,
    _snapshot_matches,
)


EXPECTED_SPLITS = (
    "episode_train", "episode_validation", "episode_test",
    "task_train", "task_validation", "task_test",
)
CONFIG_HEADS = {
    "shared_four_head": ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"),
    "three_head": ("physical_criticality", "instability", "gripper_closing_state"),
    "physical_only": ("physical_criticality",),
    "gripper_only": ("gripper_closing_state",),
}
THRESHOLDS = tuple(round(0.05 * index, 2) for index in range(1, 20))
MODEL_HEAD = {"k10_feasibility": "k10_feasible"}
RISK_DIRECTION = {
    "physical_criticality": "probability_is_risk",
    "k10_feasibility": "invert_1_minus_probability_for_risk",
    "instability": "probability_is_risk",
    "gripper_closing_state": "probability_is_risk",
}
FORBIDDEN_OUTPUT_PARTS = {"protected", "cal", "check", "g10", "t2r-d", "attack", "rollout"}


def _load_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_file(relative: str, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise ValueError(f"unsafe {label} path")
    current = ROOT
    for part in raw.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked {label} component: {current}")
    path = current.resolve(strict=True)
    if ROOT.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a repository file")
    return path


def _validate_g2_permissions(value: Any) -> None:
    expected = {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": True, "development_inference": True, "privileged_oracle_diagnostic": True, "shadow_offline": False, "shadow_live": False, "formal_training": False, "full_fit": False, "rollout": False, "attack": False, "protected_reads": 0}
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("G2 permission matrix keys are not exact")
    for key, expected_value in expected.items():
        actual = value[key]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(f"G2 permission type/value mismatch: {key}")


def _load_g2(g2_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not g2_root.is_absolute() or g2_root.is_symlink() or any(part.casefold() in FORBIDDEN_OUTPUT_PARTS for part in g2_root.parts):
        raise ValueError("unsafe G2 root")
    g2_root = g2_root.resolve(strict=True)
    g2_seal = verify_seal(g2_root)
    transition = _load_json(g2_root / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V2.json")
    if transition.get("schema") != "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V2" or transition.get("status") != "PASS_G2_DEVELOPMENT_TRANSITION":
        raise ValueError("G2 transition is not development-consumable")
    _validate_g2_permissions(transition.get("permissions"))
    if type(transition.get("protected_reads")) is not int or transition.get("protected_reads") != 0 or transition.get("formal_training_authorized") is not False or transition.get("formal_inference_authorized") is not False or transition.get("attack_authorized") is not False:
        raise ValueError("G2 permission boundary is not FIT-only")
    if transition.get("teacher_privileged_fields_in_student") is not False or transition.get("consumable_for_scientific_promotion") is not False or transition.get("shadow_offline_authorized") is not False or transition.get("shadow_live_authorized") is not False:
        raise ValueError("G2 scientific boundary is not closed")
    if transition.get("model_boundary") != {"random_initialization_required": True, "all_670_engineering_checkpoint_allowed": False, "checkpoint_consumed": False, "privileged_oracle_nondeployable": True}:
        raise ValueError("G2 model boundary is not exact")
    if tuple(transition.get("expected_split_keys", ())) != EXPECTED_SPLITS:
        raise ValueError("G2 split key closure is not exact")
    snapshot = transition.get("code_snapshot")
    if not isinstance(snapshot, Mapping) or not _snapshot_matches(snapshot, allow_descendant_snapshot=False):
        raise ValueError("G2 source snapshot is not the exact consuming checkout")
    protocol_ref = transition.get("protocol")
    feature_ref = transition.get("feature_binding")
    if not isinstance(protocol_ref, Mapping) or protocol_ref.get("sha256") != sha256_file(_repo_file(str(protocol_ref.get("path")), "protocol")):
        raise ValueError("G2 protocol binding failed")
    if not isinstance(feature_ref, Mapping) or feature_ref.get("sha256") != sha256_file(_repo_file(str(feature_ref.get("path")), "feature binding")):
        raise ValueError("G2 feature binding failed")
    for name, ref in transition.get("source_files", {}).items():
        if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str) or Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts:
            raise ValueError(f"unsafe G2 source binding: {name}")
        source_path = _repo_file(ref["path"], f"source {name}")
        if sha256_file(source_path) != ref.get("sha256"):
            raise ValueError(f"G2 source binding mismatch: {name}")
    source = transition.get("source_files", {}).get("student_trainer_reference")
    expected_path = "scripts/detector_v5/run_r3_heldout_development.py"
    if not isinstance(source, Mapping) or source.get("path") != expected_path or source.get("sha256") != sha256_file(ROOT / expected_path):
        raise ValueError("G2 does not bind the held-out trainer source")
    g1 = transition.get("g1")
    if not isinstance(g1, Mapping) or not isinstance(g1.get("root"), str):
        raise ValueError("G2 G1 seal binding failed")
    g1_raw = Path(g1["root"])
    if not g1_raw.is_absolute() or g1_raw.is_symlink() or any(part.casefold() in FORBIDDEN_OUTPUT_PARTS for part in g1_raw.parts):
        raise ValueError("unsafe G1 root")
    g1_candidate = g1_raw.resolve(strict=True)
    if g1.get("seal_sha256sums_sha256") != verify_seal(g1_candidate)["sha256sums_sha256"]:
        raise ValueError("G2 G1 seal binding failed")
    g1_root = g1_candidate
    split_manifests = g1.get("split_manifests")
    if not isinstance(split_manifests, Mapping) or set(split_manifests) != set(EXPECTED_SPLITS):
        raise ValueError("G2 does not contain all six split bindings")
    binding = {"g2_root": str(g2_root), "g2_seal_sha256sums_sha256": g2_seal["sha256sums_sha256"], "g1_root": str(g1_root), "g1_seal_sha256sums_sha256": g1["seal_sha256sums_sha256"], "split_manifests": split_manifests}
    return transition, binding


def _load_split_ids(g1_root: Path, split_name: str, expected: Mapping[str, Any]) -> list[str]:
    files = {
        "episode_train": "EPISODE_TRAIN_MANIFEST.json",
        "episode_validation": "EPISODE_VAL_MANIFEST.json",
        "episode_test": "EPISODE_TEST_MANIFEST.json",
        "task_train": "TASK_TRAIN_MANIFEST.json",
        "task_validation": "TASK_VAL_MANIFEST.json",
        "task_test": "TASK_TEST_MANIFEST.json",
    }
    path = g1_root / files[split_name]
    if expected.get("sha256") != sha256_file(path) and expected.get("file_sha256") != sha256_file(path):
        raise ValueError(f"G1 split file binding mismatch: {split_name}")
    rows = _load_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or malformed split manifest: {split_name}")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) & {"labels", "teacher", "physical_state", "object_state"}:
            raise ValueError(f"Teacher/privileged field in split manifest: {split_name}")
        identity = row.get("episode_id")
        if not isinstance(identity, str) or identity in ids:
            raise ValueError(f"duplicate identity in split manifest: {split_name}")
        ids.append(identity)
    expected_ids = expected.get("identity_ids")
    if not isinstance(expected_ids, list) or sorted(expected_ids) != sorted(ids) or expected.get("identity_count") != len(ids):
        raise ValueError(f"G2/G1 identity binding mismatch: {split_name}")
    return ids


def _load_splits(g1_root: Path, family: str, expected_bindings: Mapping[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    names = {"episode": ("episode_train", "episode_validation", "episode_test"), "task": ("task_train", "task_validation", "task_test")}[family]
    split_ids = {name: _load_split_ids(g1_root, name, expected_bindings[name]) for name in names}
    sets = {name: set(ids) for name, ids in split_ids.items()}
    if any(sets[a] & sets[b] for a in names for b in names if a < b):
        raise ValueError(f"{family} split identities overlap")
    normalization = _load_json(g1_root / "NORMALIZATION.json")
    normal = normalization.get(f"{family}_heldout", {}).get("train")
    if not isinstance(normal, Mapping) or normal.get("source_split") != "train" or len(normal.get("mean", [])) != 25 or len(normal.get("std", [])) != 25:
        raise ValueError(f"{family} normalization is not train-only 25D")
    return split_ids, {"normalization": normal, "normalization_sha256": sha256_file(g1_root / "NORMALIZATION.json")}


def _check_split_closure(split_ids: Mapping[str, Sequence[str]], records: Sequence[Mapping[str, Any]], family: str, loaded_ids: set[str] | None = None) -> None:
    record_ids = {str(row["identity"]) for row in records}
    expected_ids = set().union(*(set(values) for values in split_ids.values())) if loaded_ids is None else loaded_ids
    if expected_ids != record_ids or len(record_ids) != len(expected_ids):
        raise ValueError(f"{family} split does not close over loaded identities")


def _batch(records_by_id: Mapping[str, Mapping[str, Any]], ids: Sequence[str], mean: np.ndarray, std: np.ndarray, device: torch.device):
    selected = [records_by_id[identity] for identity in ids]
    max_steps = max(len(item["features"]) for item in selected)
    x = torch.zeros((len(selected), max_steps, 25), dtype=torch.float32, device=device)
    valid = torch.zeros((len(selected), max_steps), dtype=torch.bool, device=device)
    targets = {head: torch.zeros((len(selected), max_steps), dtype=torch.float32, device=device) for head in HEADS}
    masks = {head: torch.zeros((len(selected), max_steps), dtype=torch.bool, device=device) for head in HEADS}
    weights = {head: torch.zeros((len(selected), max_steps), dtype=torch.float32, device=device) for head in HEADS}
    norm_mean = torch.tensor(mean, dtype=torch.float32, device=device)
    norm_std = torch.tensor(std, dtype=torch.float32, device=device)
    for index, item in enumerate(selected):
        length = len(item["features"])
        x[index, :length] = (torch.tensor(item["features"], dtype=torch.float32, device=device) - norm_mean) / norm_std
        valid[index, :length] = True
        for head in HEADS:
            targets[head][index, :length] = torch.tensor(item["targets"][head], dtype=torch.float32, device=device)
            masks[head][index, :length] = torch.tensor(item["masks"][head], dtype=torch.bool, device=device)
            weights[head][index, :length] = torch.tensor(item["weights"][head], dtype=torch.float32, device=device)
    return x, valid, targets, masks, weights


def _safe_output_root(path: Path, allowed_parent: Path) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError(f"output root must be a new absolute regular path: {path}")
    if any(part.casefold() in FORBIDDEN_OUTPUT_PARTS for part in path.parts):
        raise ValueError("output root is under a forbidden path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked output component: {current}")
    if path.parent.resolve(strict=False) != allowed_parent.resolve(strict=True):
        raise ValueError("output root must be a new sibling of the sealed G2 phase")
    return path


def _active_masks(masks: Mapping[str, torch.Tensor], active: Sequence[str]) -> dict[str, torch.Tensor]:
    return {head: (masks[head] if head in active else torch.zeros_like(masks[head])) for head in HEADS}


def _train(model: torch.nn.Module, batch: tuple[Any, ...], active: Sequence[str], epochs: int, lr: float, weight_decay: float, clip: float, targets_override: Mapping[str, torch.Tensor] | None = None) -> list[dict[str, Any]]:
    x, valid, targets, masks, weights = batch
    if targets_override is not None:
        targets = targets_override
    active_masks = _active_masks(masks, active)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, timestep_mask=valid)
        loss, components = _loss(logits, targets, active_masks, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite held-out training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite held-out gradient")
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), "components": components})
    return history


def _shuffle_targets(batch: tuple[Any, ...], active: Sequence[str], seed: int) -> dict[str, torch.Tensor]:
    _, _, targets, masks, _ = batch
    shuffled = {head: value.clone() for head, value in targets.items()}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for head in active:
        flat_mask = masks[head].reshape(-1)
        indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            continue
        flat = shuffled[head].reshape(-1)
        values = flat[indices].clone()
        permutation = torch.randperm(indices.numel(), generator=generator)
        if torch.equal(permutation, torch.arange(indices.numel())):
            permutation = torch.roll(permutation, 1)
        flat[indices] = values[permutation]
    return shuffled


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    # Rank from low to high; the Mann-Whitney formulation below then yields
    # P(score_positive > score_negative), including average ranks for ties.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _safe_auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    n = len(y)
    cum_pos = 0.0
    auprc = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and score[order[j]] == score[order[i]]:
            j += 1
        group_size = j - i
        group_pos = float(y_sorted[i:j].sum())
        if group_pos > 0:
            for k in range(group_size):
                expected_cum = cum_pos + (k + 1) * group_pos / group_size
                auprc += (expected_cum / (i + k + 1)) * (group_pos / group_size)
        cum_pos += group_pos
        i = j
    return float(auprc / positives)


def _binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    bal = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else None
    return {"count": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "auroc": _safe_auc(y, score), "auprc": _safe_auprc(y, score), "balanced_accuracy": bal, "mcc": mcc, "precision": tp / (tp + fp) if tp + fp else None, "recall": tpr, "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def _candidate_spans(item: Mapping[str, Any]) -> list[tuple[int, int]]:
    candidates = np.asarray(item["candidate_close"], dtype=bool)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(candidates):
        if value and start is None:
            start = index
        if not value and start is not None:
            spans.append((start, index - 1)); start = None
    if start is not None:
        spans.append((start, len(candidates) - 1))
    return spans


def _teacher_critical_spans(item: Mapping[str, Any], head: str) -> list[tuple[int, int]]:
    """Contiguous TRUE label spans independent of candidate_close."""
    masks = np.asarray(item["masks"][head], dtype=bool)
    targets = np.asarray(item["targets"][head], dtype=np.float32)
    known_true = masks & (targets == 1)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(known_true):
        if value and start is None:
            start = index
        if not value and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(known_true) - 1))
    return spans


def _event_label(item: Mapping[str, Any], head: str, start: int, end: int) -> str:
    masks = np.asarray(item["masks"][head], dtype=bool)[start:end + 1]
    targets = np.asarray(item["targets"][head], dtype=np.float32)[start:end + 1]
    known = targets[masks]
    if np.any(known == 1):
        return "TRUE"
    if len(known) == end - start + 1:
        return "FALSE"
    return "UNKNOWN"


def _event_metrics(items: Sequence[Mapping[str, Any]], ids: Sequence[str], head: str, probabilities: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    labels: list[int] = []; predictions: list[int] = []; episodes = 0; candidate_episodes = 0; false_emits = 0
    true_events_candidate = 0; unknown_events = 0; right_censored_events = 0; candidate_events = 0
    teacher_critical_total = 0; teacher_critical_reached_by_candidate = 0
    teacher_detected_events = 0; latencies: list[int] = []

    for item in items:
        if item["identity"] not in ids:
            continue
        episodes += 1
        tc_spans = _teacher_critical_spans(item, head)
        teacher_critical_total += len(tc_spans)
        spans = _candidate_spans(item)
        candidate_events += len(spans)
        candidate_episodes += int(bool(spans))

        has_candidate = np.zeros(len(item["features"]), dtype=bool)
        for s, e in spans:
            has_candidate[s:e + 1] = True
        for ts, te in tc_spans:
            if has_candidate[ts:te + 1].any():
                teacher_critical_reached_by_candidate += 1
                detector_hit = False
                for s, e in spans:
                    overlap_start = max(ts, s)
                    overlap_end = min(te, e)
                    if overlap_start > overlap_end:
                        continue
                    event_scores = probabilities[item["identity"]][overlap_start:overlap_end + 1]
                    if np.any(event_scores >= threshold):
                        detector_hit = True
                        first = overlap_start + int(np.flatnonzero(event_scores >= threshold)[0])
                        latencies.append(first - ts)
                        break
                if detector_hit:
                    teacher_detected_events += 1

        false_emits_in_episode = 0
        for s, e in spans:
            event_scores = probabilities[item["identity"]][s:e + 1]
            crossings = np.flatnonzero(event_scores >= threshold)
            label = _event_label(item, head, s, e)
            if label == "UNKNOWN":
                unknown_events += 1
                right_censored_events += int(bool(np.asarray(item.get("right_censored", {}).get(head, np.zeros(len(item["features"]), dtype=bool)))[s:e + 1].any()))
                continue
            is_true = label == "TRUE"
            predicted = bool(len(crossings))
            labels.append(int(is_true))
            predictions.append(int(predicted))
            if is_true:
                true_events_candidate += 1
            elif predicted:
                false_emits_in_episode += 1
        false_emits += false_emits_in_episode

    y = np.asarray(labels, dtype=np.int64); pred = np.asarray(predictions, dtype=np.int64)
    if len(y):
        tp = int(np.sum((y == 1) & (pred == 1))); tn = int(np.sum((y == 0) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1))); fn = int(np.sum((y == 1) & (pred == 0)))
        recall = tp / (tp + fn) if tp + fn else None
        negative_recall = tn / (tn + fp) if tn + fp else None
        binary = {"count": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
                  "balanced_accuracy": (recall + negative_recall) / 2 if recall is not None and negative_recall is not None else None,
                  "mcc": ((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                         if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) else None,
                  "precision": tp / (tp + fp) if tp + fp else None, "recall": recall,
                  "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}
    else:
        binary = {"count": 0, "positive": 0, "negative": 0, "balanced_accuracy": None, "mcc": None,
                  "precision": None, "recall": None, "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}
    negative_recall = (binary["confusion_matrix"]["tn"] / binary["negative"]) if binary["negative"] else None
    candidate_ceiling = (teacher_critical_reached_by_candidate / teacher_critical_total) if teacher_critical_total else None
    end_to_end_critical_recall = (teacher_detected_events / teacher_critical_total) if teacher_critical_total else None
    candidate_conditioned_recall = (teacher_detected_events / teacher_critical_reached_by_candidate) if teacher_critical_reached_by_candidate else None

    return {"metric_kind": "causal_streaming_first_threshold_crossing",
            "teacher_critical_events": teacher_critical_total,
            "candidate_events": candidate_events,
            "teacher_critical_events_reached_by_candidate": teacher_critical_reached_by_candidate,
            "candidate_ceiling": candidate_ceiling,
            "end_to_end_critical_recall": end_to_end_critical_recall,
            "candidate_conditioned_recall": candidate_conditioned_recall,
            "candidate_episodes": candidate_episodes,
            "no_candidate_episodes": episodes - candidate_episodes,
            "known_events": int(len(y)), "positive_events": int(y.sum()) if len(y) else 0,
            "negative_events": int(len(y) - y.sum()) if len(y) else 0,
            "unknown_events": unknown_events, "right_censored_events": right_censored_events,
            "unknown_event_coverage": (1.0 - unknown_events / candidate_events) if candidate_events else None,
            "event_recall": binary["recall"],
            "negative_event_recall": negative_recall,
            "minority_event_recall": min(binary["recall"], negative_recall)
                                     if binary["recall"] is not None and negative_recall is not None else None,
            "event_precision": binary["precision"],
            "known_negative_fpr": (binary["confusion_matrix"]["fp"] / binary["negative"]
                                   if binary["negative"] else None),
            "false_emits_per_episode": false_emits / episodes if episodes else None,
            "latency_mean": float(np.mean(latencies)) if latencies else None,
            "latency_count": len(latencies),
            "balanced_accuracy": binary["balanced_accuracy"], "mcc": binary["mcc"],
            "confusion_matrix": binary["confusion_matrix"], "threshold": threshold,
            "episodes": episodes, "true_events_candidate": true_events_candidate,
            "event_score_source": "none_first_crossing_only"}


def _select_threshold(items: Sequence[Mapping[str, Any]], ids: Sequence[str], head: str, probabilities: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidates = []
    for threshold in THRESHOLDS:
        metric = _event_metrics(items, ids, head, probabilities, threshold)
        candidates.append(metric)
    valid = [item for item in candidates if item["balanced_accuracy"] is not None]
    if not valid:
        return {"status": "HOLD_SPLIT_COVERAGE", "threshold": None, "grid": candidates}
    selected = max(valid, key=lambda item: (item["balanced_accuracy"], item["minority_event_recall"] if item["minority_event_recall"] is not None else -1.0, -(item["false_emits_per_episode"] if item["false_emits_per_episode"] is not None else float("inf")), -item["threshold"]))
    return {"status": "SELECTED_VALIDATION_ONLY", "threshold": selected["threshold"], "selection_metric": "event_balanced_accuracy", "grid": candidates}


def _ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float | None:
    if not len(y):
        return None
    total = 0.0
    for left, right in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (score >= left) & (score <= right if right == 1.0 else score < right)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return total


def _step_metrics(items: Sequence[Mapping[str, Any]], ids: Sequence[str], head: str, probabilities: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    ys: list[np.ndarray] = []; ss: list[np.ndarray] = []
    for item in items:
        if item["identity"] not in ids:
            continue
        mask = np.asarray(item["masks"][head], dtype=bool)
        ys.append(np.asarray(item["targets"][head], dtype=np.int64)[mask])
        ss.append(probabilities[item["identity"]][mask])
    y = np.concatenate(ys) if ys else np.asarray([], dtype=np.int64); s = np.concatenate(ss) if ss else np.asarray([], dtype=np.float64)
    result = _binary_metrics(y, s, threshold) if len(y) else {"count": 0, "positive": 0, "negative": 0, "auroc": None, "auprc": None, "balanced_accuracy": None, "mcc": None, "precision": None, "recall": None, "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}
    result["ece"] = _ece(y, s)
    return result


def _majority_probability(items: Sequence[Mapping[str, Any]], ids: Sequence[str], head: str) -> float:
    values: list[np.ndarray] = []
    for item in items:
        if item["identity"] in ids:
            mask = np.asarray(item["masks"][head], dtype=bool)
            values.append(np.asarray(item["targets"][head], dtype=np.float64)[mask])
    flat = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    return float(flat.mean() if len(flat) else 0.0)


def _predict(model: torch.nn.Module, batch: tuple[Any, ...], ids: Sequence[str], records_by_id: Mapping[str, Mapping[str, Any]], device: torch.device) -> dict[str, np.ndarray]:
    x, valid, *_ = batch
    model.eval()
    with torch.no_grad():
        logits = model(x, timestep_mask=valid)
    for name, value in logits.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite logit in {name} during prediction")
    def sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    result: dict[str, np.ndarray] = {}
    for index, identity in enumerate(ids):
        length = len(records_by_id[identity]["features"])
        result[identity] = np.stack([sigmoid(np.asarray(logits[MODEL_HEAD.get(head, head)][index, :length].detach().cpu(), dtype=np.float64)) for head in ACTIVE_HEADS], axis=1)
    return result


def _head_probability(probabilities: Mapping[str, np.ndarray], ids: Sequence[str], head: str) -> dict[str, np.ndarray]:
    index = ACTIVE_HEADS.index(head)
    return {identity: probabilities[identity][:, index] for identity in ids}


def _linear_probe(train_batch: tuple[Any, ...], target_batches: Mapping[str, tuple[Any, ...]], active: Sequence[str], target_ids: Mapping[str, Sequence[str]], records_by_id: Mapping[str, Mapping[str, Any]], epochs: int, device: torch.device) -> dict[str, Any]:
    x, valid, targets, masks, _ = train_batch
    trained: dict[str, torch.nn.Module] = {}
    for head in active:
        selected = valid & masks[head]
        if not bool(selected.any()):
            continue
        probe = torch.nn.Linear(25, 1).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-5)
        y = targets[head][selected]
        positive = float(y.sum().detach().cpu()); negative = float(len(y) - positive)
        pos_weight = torch.tensor(min(max(negative / max(positive, 1.0), 1.0), 20.0), dtype=torch.float32, device=device)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(x[selected]).squeeze(-1), y, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite linear probe loss")
            loss.backward(); optimizer.step()
        trained[head] = probe.eval()
    probabilities: dict[str, np.ndarray] = {}
    for split, batch in target_batches.items():
        ids = target_ids[split]
        split_x, split_valid, *_ = batch
        for index, identity in enumerate(ids):
            length = len(records_by_id[identity]["features"])
            values = []
            for head in ACTIVE_HEADS:
                if head not in trained:
                    values.append(np.full(length, 0.5, dtype=np.float64))
                    continue
                with torch.no_grad():
                    logits = trained[head](split_x[index, :length]).squeeze(-1)
                values.append((1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits.detach().cpu(), dtype=np.float64), -60.0, 60.0)))))
            probabilities[identity] = np.stack(values, axis=1)
    return {"probabilities": probabilities, "trained_heads": sorted(trained), "epochs": epochs}


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _write_predictions(root: Path, items: Sequence[Mapping[str, Any]], split_ids: Mapping[str, Sequence[str]], probabilities: Mapping[str, np.ndarray], threshold_by_head: Mapping[str, float | None], active: Sequence[str]) -> None:
    owner = {identity: split for split, ids in split_ids.items() for identity in ids}
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            identity = item["identity"]
            if identity not in owner:
                continue
            for step, row in enumerate(probabilities[identity]):
                payload = {"episode_id": identity, "step": step, "split": owner[identity], "candidate_close": bool(item["candidate_close"][step])}
                for index, head in ((ACTIVE_HEADS.index(head), head) for head in active):
                    known = bool(item["masks"][head][step])
                    payload[head] = {"probability": float(row[index]), "known": known, "target": int(item["targets"][head][step]) if known else None, "selected": bool(threshold_by_head.get(head) is not None and row[index] >= float(threshold_by_head[head]))}
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.config not in CONFIG_HEADS:
        raise ValueError(f"unknown config: {args.config}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    transition, g2_binding = _load_g2(args.g2_root)
    test_policy = transition.get("test_read_policy")
    if args.read_test:
        if test_policy != "G7_ONE_TIME_AFTER_VALIDATION_FREEZE":
            raise ValueError("--read-test requires G7 authorization; G2 transition test_read_policy is not G7_ONE_TIME_AFTER_VALIDATION_FREEZE")
        g7_transition_path = args.g2_root.parent / "G7_TEST_TRANSITION" / "G7_TEST_READ_TRANSITION.json"
        if not g7_transition_path.is_file():
            raise ValueError("G7 test-read transition not found; test payload read is not authorized at G4/G5/G6")
        g7 = _load_json(g7_transition_path)
        if g7.get("status") != "PASS_G7_TEST_READ_AUTHORIZED" or g7.get("schema") != "V5_R3_G7_TEST_READ_TRANSITION_V1":
            raise ValueError("G7 transition is not authorized for test read")
        if g7.get("consumption_count", 0) != 0:
            raise ValueError("G7 transition has already been consumed; test can only be read once")
        g7_binding = g7.get("binding", {})
        if (g7_binding.get("trainer_sha256") != sha256_file(Path(__file__)) or
            g7_binding.get("g1_test_manifest_sha256") != g2_binding["split_manifests"].get(f"{args.split_family}_test", {}).get("file_sha256")):
            raise ValueError("G7 transition binding does not match current execution context")
    allowed_parent = Path(g2_binding["g2_root"]).parent
    _safe_output_root(args.output_root, allowed_parent)
    family = args.split_family
    split_ids, split_meta = _load_splits(Path(g2_binding["g1_root"]), family, g2_binding["split_manifests"])
    evaluated_splits = ("train", "validation", "test") if args.read_test else ("train", "validation")
    non_train_splits = tuple(split for split in evaluated_splits if split != "train")
    required_ids = set().union(*(set(split_ids[f"{family}_{split}"]) for split in evaluated_splits))
    t4_root = Path(transition["t4"]["root"]).resolve()
    records, record_binding = _load_records(t4_root, allow_descendant_snapshot=False, identity_allowlist=required_ids)
    records_by_id = {item["identity"]: item for item in records}
    _check_split_closure({f"{family}_{split}": split_ids[f"{family}_{split}"] for split in evaluated_splits}, records, family, loaded_ids=required_ids)
    normal = split_meta["normalization"]
    train_ids = split_ids[f"{family}_train"]; mean = np.asarray(normal["mean"], dtype=np.float64); std = np.asarray(normal["std"], dtype=np.float64)
    recomputed = np.concatenate([records_by_id[identity]["features"] for identity in train_ids], axis=0)
    if not np.allclose(recomputed.mean(axis=0), mean, atol=1e-10, rtol=0.0) or not np.allclose(np.maximum(recomputed.std(axis=0), 1e-8), std, atol=1e-10, rtol=0.0):
        raise ValueError("G1 normalization is not exactly train-only")
    split_batches = {f"{family}_{split}": _batch(records_by_id, split_ids[f"{family}_{split}"], mean, std, device) for split in evaluated_splits}
    model_cls = _load_model()
    model = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
    init_rng_state = torch.get_rng_state()
    init_np_state = np.random.get_state()
    import hashlib as _hashlib
    init_state_digest = _hashlib.sha256(b"".join(param.detach().cpu().numpy().tobytes() for param in model.parameters())).hexdigest()
    active = CONFIG_HEADS[args.config]
    history = _train(model, split_batches[f"{family}_train"], active, args.epochs, args.learning_rate, args.weight_decay, args.gradient_clip)
    # Verify inactive heads (safe_release) receive zero gradient through shared encoder.
    model.zero_grad(set_to_none=True)
    train_x, train_valid, train_targets, train_masks, train_weights = split_batches[f"{family}_train"]
    loss, _ = _loss(model(train_x, timestep_mask=train_valid), train_targets, _active_masks(train_masks, active), train_weights)
    loss.backward()
    for head in INACTIVE_HEADS:
        idx = model_cls.HEAD_NAMES.index(head)
        grad_sum = float(sum(p.grad.abs().sum() for p in model.heads[idx].parameters() if p.grad is not None))
        if grad_sum != 0.0:
            raise AssertionError(f"inactive head {head} received nonzero gradient: {grad_sum}")
    probabilities = {}
    for split in evaluated_splits:
        probabilities.update(_predict(model, split_batches[f"{family}_{split}"], split_ids[f"{family}_{split}"], records_by_id, device))
    thresholds: dict[str, Any] = {}; metrics: dict[str, Any] = {split: {} for split in evaluated_splits}
    for head in active:
        val_prob = _head_probability(probabilities, split_ids[f"{family}_validation"], head)
        thresholds[head] = _select_threshold(records, split_ids[f"{family}_validation"], head, val_prob)
        threshold = thresholds[head]["threshold"]
        for split in evaluated_splits:
            ids = split_ids[f"{family}_{split}"]; head_prob = _head_probability(probabilities, ids, head)
            metrics[split][head] = {"threshold": threshold} if threshold is None else {"step": _step_metrics(records, ids, head, head_prob, threshold), "event": _event_metrics(records, ids, head, head_prob, threshold), "threshold": threshold}
    for head in INACTIVE_HEADS:
        for split in evaluated_splits:
            metrics[split][head] = {"status": "NOT_EVALUABLE_COVERAGE"}
        thresholds[head] = {"status": "HOLD_COVERAGE", "threshold": None}
    shuffle_results: dict[str, Any] = {}
    for shuffle_seed in (args.seed + 101, args.seed + 102, args.seed + 103):
        torch.set_rng_state(init_rng_state)
        np.random.set_state(init_np_state)
        shuffle_model = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
        shuffle_init_digest = _hashlib.sha256(b"".join(param.detach().cpu().numpy().tobytes() for param in shuffle_model.parameters())).hexdigest()
        if shuffle_init_digest != init_state_digest:
            raise AssertionError(f"shuffle model seed={shuffle_seed} does not share real model initialization")
        torch.manual_seed(shuffle_seed)
        np.random.seed(shuffle_seed)
        shuffled_targets = _shuffle_targets(split_batches[f"{family}_train"], active, shuffle_seed)
        shuffle_history = _train(shuffle_model, split_batches[f"{family}_train"], active, args.epochs, args.learning_rate, args.weight_decay, args.gradient_clip, targets_override=shuffled_targets)
        shuffle_probabilities: dict[str, np.ndarray] = {}
        for split in non_train_splits:
            shuffle_probabilities.update(_predict(shuffle_model, split_batches[f"{family}_{split}"], split_ids[f"{family}_{split}"], records_by_id, device))
        shuffle_metrics: dict[str, Any] = {}
        shuffle_thresholds: dict[str, Any] = {}
        for head in active:
            val_prob = _head_probability(shuffle_probabilities, split_ids[f"{family}_validation"], head)
            shuffle_thresholds[head] = _select_threshold(records, split_ids[f"{family}_validation"], head, val_prob)
            threshold = shuffle_thresholds[head]["threshold"]
            shuffle_metrics[head] = {"threshold": threshold} if threshold is None else {"validation": _event_metrics(records, split_ids[f"{family}_validation"], head, val_prob, threshold), **({"test": _event_metrics(records, split_ids[f"{family}_test"], head, _head_probability(shuffle_probabilities, split_ids[f"{family}_test"], head), threshold)} if args.read_test else {}), "threshold": threshold}
        shuffle_results[str(shuffle_seed)] = {"seed": shuffle_seed, "history": shuffle_history, "thresholds_validation_only": shuffle_thresholds, "metrics": shuffle_metrics}
    # G4 baselines are computed on the same held-out split; they never inspect
    # test labels when selecting a threshold.
    baselines: dict[str, Any] = {}
    for baseline in ("constant_true", "constant_false", "majority"):
        baselines[baseline] = {}
        for head in active:
            value = 1.0 if baseline == "constant_true" else 0.0 if baseline == "constant_false" else _majority_probability(records, train_ids, head)
            base_prob = {identity: np.full(len(records_by_id[identity]["features"]), value, dtype=np.float64) for identity in records_by_id}
            base_threshold = _select_threshold(records, split_ids[f"{family}_validation"], head, {identity: base_prob[identity] for identity in split_ids[f"{family}_validation"]})["threshold"]
            baselines[baseline][head] = {"train_value": value, "threshold": base_threshold} if base_threshold is None else {"train_value": value, "threshold": base_threshold, "validation": _event_metrics(records, split_ids[f"{family}_validation"], head, base_prob, base_threshold), **({"test": _event_metrics(records, split_ids[f"{family}_test"], head, base_prob, base_threshold)} if args.read_test else {})}
    linear_batches = {split: split_batches[f"{family}_{split}"] for split in non_train_splits}
    linear_ids = {split: split_ids[f"{family}_{split}"] for split in non_train_splits}
    linear = _linear_probe(split_batches[f"{family}_train"], linear_batches, active, linear_ids, records_by_id, min(args.epochs, 100), device)
    baselines["linear_probe"] = {"trained_heads": linear["trained_heads"], "epochs": linear["epochs"]}
    for head in active:
        val_prob = _head_probability(linear["probabilities"], split_ids[f"{family}_validation"], head)
        linear_threshold = _select_threshold(records, split_ids[f"{family}_validation"], head, val_prob)["threshold"]
        baselines["linear_probe"][head] = {"threshold": linear_threshold} if linear_threshold is None else {"validation": _event_metrics(records, split_ids[f"{family}_validation"], head, val_prob, linear_threshold), **({"test": _event_metrics(records, split_ids[f"{family}_test"], head, _head_probability(linear["probabilities"], split_ids[f"{family}_test"], head), linear_threshold)} if args.read_test else {}), "threshold": linear_threshold}
    if not args.read_test:
        metrics["test"] = {"status": "NOT_READ_BY_PROTOCOL"}
    # Compute safe_release gradient check result for the report.
    safe_release_grad_zero = True
    for head in INACTIVE_HEADS:
        idx = model_cls.HEAD_NAMES.index(head)
        grad_sum = float(sum(p.grad.abs().sum() for p in model.heads[idx].parameters() if p.grad is not None))
        if grad_sum != 0.0:
            safe_release_grad_zero = False

    report = {"schema": "V5_R3_HELDOUT_DEVELOPMENT_V3", "status": "ENGINEERING_DEVELOPMENT_NONCONSUMABLE", "split_family": family, "config": args.config, "seed": args.seed, "epochs": args.epochs, "device": str(device), "random_initialization": True, "all_670_checkpoint_loaded": False, "initial_state_sha256": init_state_digest, "label_shuffle_same_initialization": True, "risk_direction": dict(RISK_DIRECTION), "safe_release_gradient_zero": safe_release_grad_zero, "active_heads": list(active), "inactive_heads": list(INACTIVE_HEADS), "train_identity_count": len(train_ids), "validation_identity_count": len(split_ids[f"{family}_validation"]), "test_identity_count": len(split_ids[f"{family}_test"]), "test_payload_read": bool(args.read_test), "test_evaluation_performed": bool(args.read_test), "history": history, "thresholds_validation_only": thresholds, "metrics": metrics, "baselines": baselines, "label_shuffle": shuffle_results, "privileged_oracle": {"status": "NOT_AVAILABLE_SEPARATE_PRIVILEGED_INPUT", "deployable": False}, "binding": {"g2_root": g2_binding["g2_root"], "g2_seal_sha256sums_sha256": g2_binding["g2_seal_sha256sums_sha256"], "g1_root": g2_binding["g1_root"], "g1_seal_sha256sums_sha256": g2_binding["g1_seal_sha256sums_sha256"], "split_manifests": g2_binding["split_manifests"], "normalization_sha256": split_meta["normalization_sha256"], "t4_root": record_binding["t4_root"], "t4_seal_sha256sums_sha256": record_binding["t4_seal_sha256sums_sha256"], "teacher_root_sha256sums_sha256": record_binding["teacher_root_sha256sums_sha256"], "feature_order_sha256": record_binding["feature_order_sha256"], "trainer_sha256": sha256_file(Path(__file__))}, "permissions": {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": True, "development_inference": True, "privileged_oracle_diagnostic": False, "shadow_offline": False, "shadow_live": False, "formal_training": False, "full_fit": False, "rollout": False, "attack": False, "protected_reads": 0}, "threshold_selection_split": "validation_only", "test_read_once": True, "teacher_privileged_fields_in_student": False, "safe_release_status": "NOT_EVALUABLE_COVERAGE"}
    staging = args.output_root.with_name(f".{args.output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"staging root already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        torch.save({"model": copy.deepcopy(model.state_dict()), "epoch": args.epochs, "seed": args.seed, "config": args.config, "active_heads": list(active), "optimizer": "not_saved_for_evaluation"}, staging / "checkpoint.pt")
        (staging / "heldout_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evaluated_split_ids = {f"{family}_{split}": split_ids[f"{family}_{split}"] for split in evaluated_splits}
        _write_predictions(staging, records, evaluated_split_ids, probabilities, {head: item.get("threshold") for head, item in thresholds.items()}, active)
        seal = _write_seal(staging)
        rename_noreplace(staging, args.output_root)
    except Exception as exc:
        failure_root = args.output_root.with_name(f"{args.output_root.name}_FAILED_{os.getpid()}")
        try:
            _safe_output_root(failure_root, allowed_parent)
            (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_HELDOUT_DEVELOPMENT_FAILURE_V1", "error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_seal(staging)
            if not failure_root.exists() and not failure_root.is_symlink():
                rename_noreplace(staging, failure_root)
        except Exception:
            # Leave the staging directory in place if failure sealing itself fails.
            pass
        raise
    report["sha256sums_sha256"] = seal
    return report


def _write_failure_receipt(args: argparse.Namespace, exc: BaseException) -> None:
    """Best-effort sealed failure evidence for errors before normal staging."""
    try:
        g2_root = Path(args.g2_root).resolve(strict=True)
        failure_root = Path(args.output_root).with_name(f"{Path(args.output_root).name}_FAILED_{os.getpid()}")
        _safe_output_root(failure_root, g2_root.parent)
        staging = failure_root.with_name(f".{failure_root.name}.staging.{os.getpid()}")
        if staging.exists() or staging.is_symlink():
            return
        staging.mkdir(parents=True)
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_HELDOUT_DEVELOPMENT_FAILURE_V2", "error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        rename_noreplace(staging, failure_root)
    except Exception:
        # Never hide the original failure or overwrite an existing evidence root.
        return


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _run_impl(args)
    except Exception as exc:
        _write_failure_receipt(args, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-family", choices=("episode", "task"), default="episode")
    parser.add_argument("--config", choices=tuple(CONFIG_HEADS), default="shared_four_head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--read-test", action="store_true", help="read and evaluate the frozen test split exactly once; default is train/validation only")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
