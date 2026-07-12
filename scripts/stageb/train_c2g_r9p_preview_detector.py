"""Train R9P preview detector (Model A: 25D only, Model B: 25D+9D) on full episodes.

Loads per-episode NPZ files via the dataset index, batches variable-length episodes
with padding, and trains a causal GRU detector. Uses a standalone 6-head R9P loss
function with per-head weights and episode-level penalties aligned to the preview
model contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.gripper_attack.c2g_causal_vulnerability_detector import (
    _persistent_support_count,
    _persistent_score,
    masked_bce,
    positive_interval_triggerability,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    LOSS_WEIGHTS,
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

CHECKPOINT_SCHEMA_VERSION = "c2g.r9p.preview_checkpoint.2026-07-12.v1"
LANGUAGE_DIM = 128
VISUAL_DIM = 1152
R9P_PRIMARY_HEADS = ("window_start", "burst_feasible", "critical_window")
R9P_SAFETY_HEADS = ("release_safe", "contact_grasp")
R9P_AUX_HEADS = ("grounding_confidence",)


def _hash_language_embedding(text: str, dim: int = LANGUAGE_DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    projection = rng.randn(32, dim).astype(np.float32)
    values = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    if len(values) < 32:
        values = np.pad(values, (0, 32 - len(values)))
    values = values[:32]
    embedding = values @ projection
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def _legacy_r9p_preview_loss(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    masks: Dict[str, Tensor],
    *,
    sample_weight: Tensor | None = None,
    weight_start: float = 1.0,
    weight_burst: float = 0.5,
    weight_critical: float = 0.5,
    weight_release: float = 0.2,
    weight_contact: float = 0.2,
    weight_grounding: float = 0.2,
    weight_early_emit: float = 0.25,
    weight_episode_miss: float = 0.50,
    weight_negative_any_emit: float = 0.50,
    weight_release_safe_emit: float = 0.50,
    persistence_window: int = 3,
    persistence_required: int = 2,
) -> Dict[str, Tensor]:
    """Standalone 6-head R9P preview loss.

    Primary heads: window_start, burst_feasible, critical_window
    Safety heads: release_safe, contact_grasp
    Auxiliary: grounding_confidence (continuous, MSE)

    Episode-level losses use window_start as the trigger signal, matching
    the deployment FixedBurstTriggerScheduler semantics.
    """
    required = set(R9P_HEAD_NAMES)
    missing_outputs = sorted(required - set(outputs))
    missing_targets = sorted(required - set(targets))
    missing_masks = sorted(required - set(masks))
    if missing_outputs or missing_targets or missing_masks:
        raise ValueError(
            f"R9P loss requires exactly 6 heads. "
            f"missing outputs={missing_outputs} "
            f"targets={missing_targets} masks={missing_masks}"
        )
    extra_outputs = sorted(set(outputs) - required)
    if extra_outputs:
        raise ValueError(f"R9P loss accepts only 6 heads, got extra: {extra_outputs}")

    # Per-head BCE
    start_loss = masked_bce(
        outputs["window_start"], targets["window_start"], masks["window_start"], sample_weight,
    )
    burst_loss = masked_bce(
        outputs["burst_feasible"], targets["burst_feasible"], masks["burst_feasible"], sample_weight,
    )
    critical_loss = masked_bce(
        outputs["critical_window"], targets["critical_window"], masks["critical_window"], sample_weight,
    )
    release_loss = masked_bce(
        outputs["release_safe"], targets["release_safe"], masks["release_safe"], sample_weight,
    )
    contact_loss = masked_bce(
        outputs["contact_grasp"], targets["contact_grasp"], masks["contact_grasp"], sample_weight,
    )
    # Grounding uses MSE on continuous [0,1] target
    gc_out = outputs["grounding_confidence"]
    gc_tgt = targets["grounding_confidence"]
    gc_active = masks["grounding_confidence"].bool()
    if gc_active.any():
        grounding_loss = F.mse_loss(gc_out[gc_active], gc_tgt[gc_active])
    else:
        grounding_loss = gc_out.sum() * 0.0

    # Episode-level losses: use window_start as primary signal
    start_logits = outputs["window_start"]
    if start_logits.ndim != 2:
        zero = start_logits.sum() * 0.0
        episode = {
            "early_emit": zero, "episode_miss": zero,
            "negative_episode_any_emit": zero, "release_safe_emit": zero,
            "positive_episode_count": zero,
            "triggerable_positive_episode_count": zero,
            "untriggerable_positive_episode_count": zero,
            "persistent_positive_window_count": zero,
        }
    else:
        start_probs = torch.sigmoid(start_logits)
        zero = start_logits.sum() * 0.0
        early: list[Tensor] = []
        miss: list[Tensor] = []
        negative: list[Tensor] = []
        release_emit: list[Tensor] = []

        ep_fkn = masks.get("episode_fully_known_negative")
        for idx, (p, y_start, m_start) in enumerate(zip(
            start_probs, targets["window_start"].bool(), masks["window_start"].bool(),
        )):
            known = masks["critical_window"][idx].bool()
            positive_start = y_start & m_start
            explicit_negative = bool(ep_fkn[idx].item()) if ep_fkn is not None else False

            # Release-safe emit penalty: trigger during release_safe
            release_safe_tgt = targets["release_safe"][idx].bool()
            release_safe_known = masks["release_safe"][idx].bool()
            safe = release_safe_tgt & release_safe_known
            if safe.any():
                release_emit.append(_persistent_score(
                    p, safe, window=persistence_window, required=persistence_required,
                ))

            if positive_start.any():
                first_idx = int(torch.nonzero(positive_start, as_tuple=False)[0, 0])
                # Early emit: trigger before window_start
                early_mask = known.clone()
                early_mask[first_idx:] = False
                if early_mask.any():
                    early.append(_persistent_score(
                        p, early_mask, window=persistence_window, required=persistence_required,
                    ))
                # Miss: no trigger at or after window_start
                late_mask = known.clone()
                late_mask[:first_idx] = False
                if late_mask.any():
                    pos_score = _persistent_score(
                        p, late_mask, window=persistence_window, required=persistence_required,
                    )
                    miss.append(-torch.log(pos_score.clamp(min=1e-6)))
                continue

            fully_known_negative = bool(known.all()) if ep_fkn is None else explicit_negative
            if fully_known_negative and known.any():
                negative.append(_persistent_score(
                    p, known, window=persistence_window, required=persistence_required,
                ))

        diagnostics = positive_interval_triggerability(
            targets["window_start"], masks["window_start"],
            persistence_window=persistence_window,
            persistence_required=persistence_required,
        )
        episode = {
            "early_emit": torch.stack(early).mean() if early else zero,
            "episode_miss": torch.stack(miss).mean() if miss else zero,
            "negative_episode_any_emit": torch.stack(negative).mean() if negative else zero,
            "release_safe_emit": torch.stack(release_emit).mean() if release_emit else zero,
            **diagnostics,
        }

    total = (
        weight_start * start_loss
        + weight_burst * burst_loss
        + weight_critical * critical_loss
        + weight_release * release_loss
        + weight_contact * contact_loss
        + weight_grounding * grounding_loss
        + weight_early_emit * episode["early_emit"]
        + weight_episode_miss * episode["episode_miss"]
        + weight_negative_any_emit * episode["negative_episode_any_emit"]
        + weight_release_safe_emit * episode["release_safe_emit"]
    )
    return {
        "total": total,
        "window_start": start_loss,
        "burst_feasible": burst_loss,
        "critical_window": critical_loss,
        "release_safe": release_loss,
        "contact_grasp": contact_loss,
        "grounding_confidence": grounding_loss,
        **episode,
    }


def _r9p_runtime_gate_episode_losses(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    masks: Dict[str, Tensor],
    *,
    persistence_window: int,
    persistence_required: int,
) -> Dict[str, Tensor]:
    """Differentiable surrogate for the frozen tri-head runtime gate."""
    critical = torch.sigmoid(outputs["critical_window"])
    release = torch.sigmoid(outputs["release_safe"])
    grounding = torch.sigmoid(outputs["grounding_confidence"])
    gate = critical * (1.0 - release) * grounding
    zero = gate.sum() * 0.0
    if gate.ndim != 2:
        return {
            "early_emit": zero, "episode_miss": zero,
            "negative_episode_any_emit": zero, "release_safe_emit": zero,
            "positive_episode_count": zero,
            "triggerable_positive_episode_count": zero,
            "untriggerable_positive_episode_count": zero,
            "persistent_positive_window_count": zero,
        }

    early: list[Tensor] = []
    miss: list[Tensor] = []
    negative: list[Tensor] = []
    safe_emit: list[Tensor] = []
    positive_intervals: list[Tensor] = []
    positive_episode_count = 0
    triggerable_count = 0
    persistent_count = 0
    explicit_negative = masks.get("episode_fully_known_negative")

    for i in range(gate.shape[0]):
        known = (
            masks["critical_window"][i].bool()
            & masks["release_safe"][i].bool()
            & masks["grounding_confidence"][i].bool()
        )
        burst_known = masks["burst_feasible"][i].bool()
        burst_positive = (targets["burst_feasible"][i] > 0.5) & burst_known
        start_positive = (targets["window_start"][i] > 0.5) & masks["window_start"][i].bool()
        positive = burst_positive
        if positive.any():
            positive_episode_count += 1
            positive_intervals.append(positive)
            support = int(_persistent_support_count(positive, window=persistence_window, required=persistence_required))
            persistent_count += support
            triggerable_count += int(support > 0)
            first = int(torch.nonzero(positive, as_tuple=False)[0, 0])
            before = known.clone()
            before[first:] = False
            if before.any():
                early.append(_persistent_score(gate[i], before, window=persistence_window, required=persistence_required))
            interval = positive.clone()
            if int(interval.sum()) < persistence_required:
                interval = ((targets["critical_window"][i] > 0.5)
                            & masks["critical_window"][i].bool() & known)
            if interval.any():
                miss.append(-torch.log(_persistent_score(
                    gate[i], interval, window=persistence_window, required=persistence_required,
                ).clamp(min=1e-6)))
        else:
            fully_known = bool(known.all()) and bool(burst_known.all())
            explicit = bool(explicit_negative[i].item()) if explicit_negative is not None else False
            if explicit and (not fully_known or start_positive.any()):
                raise ValueError("episode_fully_known_negative contradicts known/positive labels")
            if fully_known and (explicit or not start_positive.any()):
                negative.append(_persistent_score(gate[i], known, window=persistence_window, required=persistence_required))

        safe = (targets["release_safe"][i] > 0.5) & masks["release_safe"][i].bool() & known
        if safe.any():
            safe_emit.append(_persistent_score(gate[i], safe, window=persistence_window, required=persistence_required))

    device = gate.device
    return {
        "early_emit": torch.stack(early).mean() if early else zero,
        "episode_miss": torch.stack(miss).mean() if miss else zero,
        "negative_episode_any_emit": torch.stack(negative).mean() if negative else zero,
        "release_safe_emit": torch.stack(safe_emit).mean() if safe_emit else zero,
        "positive_episode_count": torch.tensor(float(positive_episode_count), device=device),
        "triggerable_positive_episode_count": torch.tensor(float(triggerable_count), device=device),
        "untriggerable_positive_episode_count": torch.tensor(float(positive_episode_count - triggerable_count), device=device),
        "persistent_positive_window_count": torch.tensor(float(persistent_count), device=device),
    }


def r9p_preview_loss(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    masks: Dict[str, Tensor],
    *,
    sample_weight: Tensor | None = None,
    weight_start: float = 1.0,
    weight_burst: float = 0.5,
    weight_critical: float = 0.5,
    weight_release: float = 0.2,
    weight_contact: float = 0.2,
    weight_grounding: float = 0.2,
    weight_early_emit: float = 0.25,
    weight_episode_miss: float = 0.50,
    weight_negative_any_emit: float = 0.50,
    weight_release_safe_emit: float = 0.50,
    persistence_window: int = 3,
    persistence_required: int = 2,
    head_pos_weight: Dict[str, float] | None = None,
) -> Dict[str, Tensor]:
    """R9P six-head loss with runtime-gate-aligned episode penalties."""
    required = set(R9P_HEAD_NAMES)
    for label, values in (("outputs", outputs), ("targets", targets)):
        missing = sorted(required - set(values))
        extra = sorted(set(values) - required)
        if missing or extra:
            raise ValueError(f"{label} must contain exactly R9P heads; missing={missing}, extra={extra}")
    mask_extra = set(masks) - required - {"episode_fully_known_negative"}
    missing_masks = sorted(required - set(masks))
    if missing_masks or mask_extra:
        raise ValueError(f"masks invalid; missing={missing_masks}, extra={sorted(mask_extra)}")
    for h in R9P_HEAD_NAMES:
        if outputs[h].shape != targets[h].shape or outputs[h].shape != masks[h].shape:
            raise ValueError(f"shape mismatch for head {h}")

    losses = {
        h: masked_bce(
            outputs[h], targets[h], masks[h], sample_weight,
            None if head_pos_weight is None else head_pos_weight.get(h),
        )
        for h in R9P_HEAD_NAMES
    }
    episode = _r9p_runtime_gate_episode_losses(
        outputs, targets, masks,
        persistence_window=persistence_window,
        persistence_required=persistence_required,
    ) if outputs["critical_window"].ndim == 2 else {
        "early_emit": outputs["critical_window"].sum() * 0.0,
        "episode_miss": outputs["critical_window"].sum() * 0.0,
        "negative_episode_any_emit": outputs["critical_window"].sum() * 0.0,
        "release_safe_emit": outputs["critical_window"].sum() * 0.0,
        "positive_episode_count": outputs["critical_window"].sum() * 0.0,
        "triggerable_positive_episode_count": outputs["critical_window"].sum() * 0.0,
        "untriggerable_positive_episode_count": outputs["critical_window"].sum() * 0.0,
        "persistent_positive_window_count": outputs["critical_window"].sum() * 0.0,
    }
    total = (
        weight_start * losses["window_start"]
        + weight_burst * losses["burst_feasible"]
        + weight_critical * losses["critical_window"]
        + weight_release * losses["release_safe"]
        + weight_contact * losses["contact_grasp"]
        + weight_grounding * losses["grounding_confidence"]
        + weight_early_emit * episode["early_emit"]
        + weight_episode_miss * episode["episode_miss"]
        + weight_negative_any_emit * episode["negative_episode_any_emit"]
        + weight_release_safe_emit * episode["release_safe_emit"]
    )
    return {"total": total, **losses, **episode}


class R9PEpisodeDataset(Dataset):
    def __init__(self, index_rows: list[dict], materialization_root: Path,
                 split_filter: str | None = None):
        self.rows = [r for r in index_rows
                     if split_filter is None or r["preview_split"] == split_filter]
        self.root = materialization_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npz_path = self.root / row["npz_path"]
        data = np.load(npz_path, allow_pickle=False)
        return {
            "features_25d": torch.from_numpy(data["features_25d"].copy()),
            "features_9d": torch.from_numpy(data["features_9d"].copy()),
            "targets": {h: torch.from_numpy(data[f"y_{h}"].copy()) for h in R9P_HEAD_NAMES},
            "masks": {h: torch.from_numpy(data[f"m_{h}"].copy()) for h in R9P_HEAD_NAMES},
            "known_mask": torch.from_numpy(data["known_mask"].copy()),
            "valid_mask": torch.from_numpy(data["valid_mask"].copy()),
            "task_language": row.get("task_language", ""),
            "preview_split": row["preview_split"],
            "parent_key": row["parent_key"],
            "suite": row["suite"],
        }


def collate_episodes(batch: list[dict]) -> dict[str, Any]:
    lengths = torch.tensor([item["features_25d"].shape[0] for item in batch])
    max_len = lengths.max().item()
    B = len(batch)

    proprio = torch.zeros(B, max_len, 25)
    policy = torch.zeros(B, max_len, 9)
    padding_mask = torch.zeros(B, max_len, dtype=torch.bool)

    targets: dict[str, Tensor] = {h: torch.zeros(B, max_len) for h in R9P_HEAD_NAMES}
    masks: dict[str, Tensor] = {h: torch.zeros(B, max_len, dtype=torch.bool) for h in R9P_HEAD_NAMES}
    known_mask = torch.zeros(B, max_len, dtype=torch.bool)
    ep_fkn = torch.zeros(B, dtype=torch.bool)

    language_embeddings = []

    for i, item in enumerate(batch):
        T = item["features_25d"].shape[0]
        proprio[i, :T] = item["features_25d"]
        policy[i, :T] = item["features_9d"]
        padding_mask[i, :T] = True
        for h in R9P_HEAD_NAMES:
            targets[h][i, :T] = item["targets"][h]
            masks[h][i, :T] = item["masks"][h]
        known_mask[i, :T] = item["known_mask"]

        # Trigger-negative: every supervised head is fully known AND no masked
        # burst_feasible AND no masked window_start.  known_mask alone is not
        # sufficient because head-specific masks can still contain unknowns.
        all_known = all(bool(item["masks"][h].all()) for h in R9P_HEAD_NAMES)
        any_start = bool(((item["targets"]["window_start"] > 0.5) & item["masks"]["window_start"]).any()) if all_known else False
        any_burst = bool(((item["targets"]["burst_feasible"] > 0.5) & item["masks"]["burst_feasible"]).any()) if all_known else False
        ep_fkn[i] = bool(all_known and not any_start and not any_burst)

        lang = _hash_language_embedding(item.get("task_language", ""))
        language_embeddings.append(torch.from_numpy(lang))

    language = torch.stack(language_embeddings)
    masks["episode_fully_known_negative"] = ep_fkn

    return {
        "proprio_25d": proprio,
        "policy_intent": policy,
        "language": language,
        "targets": targets,
        "masks": masks,
        "padding_mask": padding_mask,
        "known_mask": known_mask,
        "lengths": lengths,
    }


def _evaluate_model(
    model: C2gGripperCriticalWindowDetector,
    dataloader: DataLoader,
    device: torch.device,
    use_policy_intent: bool,
    norm: dict | None = None,
) -> dict[str, float]:
    model.eval()
    all_probs: dict[str, list] = defaultdict(list)
    all_targets: dict[str, list] = defaultdict(list)
    all_masks: dict[str, list] = defaultdict(list)

    p_mean = torch.from_numpy(norm["proprio_mean"]).to(device) if norm else None
    p_std = torch.from_numpy(norm["proprio_std"]).to(device).clamp_min(1e-8) if norm else None
    pi_mean = torch.from_numpy(norm["policy_intent_mean"]).to(device) if norm else None
    pi_std = torch.from_numpy(norm["policy_intent_std"]).to(device).clamp_min(1e-8) if norm else None

    with torch.no_grad():
        for batch in dataloader:
            proprio_raw = batch["proprio_25d"].to(device)
            policy_raw = batch["policy_intent"].to(device) if use_policy_intent else None
            language = batch["language"].to(device)
            proprio = (proprio_raw - p_mean) / p_std if p_mean is not None else proprio_raw
            policy = (policy_raw - pi_mean) / pi_std if pi_mean is not None and policy_raw is not None else policy_raw
            outputs = model(
                proprio, language,
                policy_intent=policy,
                return_sequence=True,
            )
            for h in R9P_HEAD_NAMES:
                prob = torch.sigmoid(outputs[h])
                all_probs[h].append(prob.cpu())
                all_targets[h].append(batch["targets"][h])
                all_masks[h].append(batch["masks"][h])

    metrics = {}
    for h in R9P_HEAD_NAMES:
        probs = torch.cat([p.flatten() for p in all_probs[h]])
        tgt = torch.cat([t.flatten() for t in all_targets[h]])
        msk = torch.cat([m.flatten() for m in all_masks[h]])
        if msk.sum() > 0:
            pred = (probs[msk] >= 0.5).float()
            tp = (pred * tgt[msk]).sum().item()
            fp = (pred * (1 - tgt[msk])).sum().item()
            fn = ((1 - pred) * tgt[msk]).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[f"{h}_precision"] = precision
            metrics[f"{h}_recall"] = recall
        else:
            metrics[f"{h}_precision"] = 0.0
            metrics[f"{h}_recall"] = 0.0

    return metrics


def load_normalization(materialization_root: Path) -> dict | None:
    norm_path = materialization_root / "normalization.json"
    if not norm_path.exists():
        return None
    norm = read_json(norm_path)
    for key, dim in (("proprio_mean", 25), ("proprio_std", 25), ("policy_intent_mean", 9), ("policy_intent_std", 9)):
        if len(norm.get(key, [])) != dim:
            raise ValueError(f"normalization field {key} must have length {dim}")
        if not np.isfinite(np.asarray(norm[key], dtype=np.float32)).all():
            raise ValueError(f"normalization field {key} contains non-finite values")
    return {
        "proprio_mean": np.array(norm["proprio_mean"], dtype=np.float32),
        "proprio_std": np.array(norm["proprio_std"], dtype=np.float32),
        "policy_intent_mean": np.array(norm["policy_intent_mean"], dtype=np.float32),
        "policy_intent_std": np.array(norm["policy_intent_std"], dtype=np.float32),
        "sha256": sha256_file(norm_path),
    }


def train_model(
    *,
    materialization_root: Path,
    output_root: Path,
    model_label: str,
    seed: int,
    epochs: int = 30,
    early_stop_patience: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
    device_str: str = "cuda",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    index_path = materialization_root / "dataset_index.jsonl"
    index_rows = read_jsonl(index_path)

    norm = load_normalization(materialization_root)
    if norm is None:
        raise FileNotFoundError(
            "normalization.json not found in materialization root — "
            "run full materialization before training"
        )
    norm_sha = norm["sha256"]

    use_policy_intent = model_label == "b"

    train_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="FIT")
    cal_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CAL")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_episodes, drop_last=False)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_episodes, drop_last=False)

    config = C2gDetectorConfig(
        visual_dim=VISUAL_DIM,
        language_dim=LANGUAGE_DIM,
        policy_intent_dim=9,
        hidden=128,
        dropout=0.1,
        use_policy_intent=use_policy_intent,
        use_visual=False,
        use_language_conditioning=True,
        head_names=R9P_HEAD_NAMES,
    )
    model = C2gGripperCriticalWindowDetector(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -float("inf")
    best_state = None
    patience_counter = 0
    history = []

    # Pre-convert normalization to tensors
    p_mean = torch.from_numpy(norm["proprio_mean"]).to(device)
    p_std = torch.from_numpy(norm["proprio_std"]).to(device)
    pi_mean = torch.from_numpy(norm["policy_intent_mean"]).to(device)
    pi_std = torch.from_numpy(norm["policy_intent_std"]).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            proprio_raw = batch["proprio_25d"].to(device)
            policy_raw = batch["policy_intent"].to(device)
            language = batch["language"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}

            # Apply normalization
            proprio = (proprio_raw - p_mean) / p_std.clamp_min(1e-8)
            policy = (policy_raw - pi_mean) / pi_std.clamp_min(1e-8) if use_policy_intent else None

            outputs = model(
                proprio, language,
                policy_intent=policy,
                return_sequence=True,
            )

            pad_mask = batch["padding_mask"].to(device)
            for h in R9P_HEAD_NAMES:
                outputs[h] = outputs[h] * pad_mask.float()

            loss_dict = r9p_preview_loss(
                outputs, targets, masks,
                sample_weight=pad_mask.float(),
                weight_start=LOSS_WEIGHTS["start"],
                weight_burst=LOSS_WEIGHTS["burst"],
                weight_critical=LOSS_WEIGHTS["critical"],
                weight_release=LOSS_WEIGHTS["release"],
                weight_contact=LOSS_WEIGHTS["contact"],
                weight_grounding=LOSS_WEIGHTS["grounding"],
                weight_early_emit=LOSS_WEIGHTS["early_emit"],
                weight_episode_miss=LOSS_WEIGHTS["episode_miss"],
                weight_negative_any_emit=LOSS_WEIGHTS["negative_any_emit"],
                weight_release_safe_emit=LOSS_WEIGHTS["release_safe_emit"],
            )

            loss = loss_dict["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch}, batch={n_batches}")
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name} at epoch={epoch}, batch={n_batches}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"non-finite parameter in {name} at epoch={epoch}, batch={n_batches}")

            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.ndim == 0:
                    epoch_losses[k] += v.item()
            n_batches += 1

        cal_metrics = _evaluate_model(model, cal_loader, device, use_policy_intent, norm)
        score = (
            cal_metrics.get("window_start_recall", 0)
            + cal_metrics.get("critical_window_recall", 0)
            - (1 - cal_metrics.get("release_safe_precision", 1))
        )
        history.append({
            "epoch": epoch,
            "losses": {k: v / max(n_batches, 1) for k, v in epoch_losses.items()},
            "cal_metrics": cal_metrics,
            "composite_score": score,
        })

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = output_root / f"model_{model_label}_seed{seed}"
    if output_dir.exists():
        raise FileExistsError(f"training output root already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    materialization_report_path = materialization_root / "materialization_report.json"
    index_path = materialization_root / "dataset_index.jsonl"
    sums_path = materialization_root / "SHA256SUMS"
    if not materialization_report_path.is_file() or not index_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError("materialization report, dataset index, and SHA256SUMS are required for checkpoint provenance")
    materialization_report_sha = sha256_file(materialization_report_path)
    dataset_index_sha = sha256_file(index_path)
    materialization_sums_sha = sha256_file(sums_path)
    materialization_report = read_json(materialization_report_path)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "model_config": {
            "visual_dim": VISUAL_DIM,
            "language_dim": LANGUAGE_DIM,
            "policy_intent_dim": 9,
            "hidden": 128,
            "dropout": 0.1,
            "use_policy_intent": use_policy_intent,
            "use_visual": False,
            "use_language_conditioning": True,
            "head_names": list(R9P_HEAD_NAMES),
        },
        "history": history,
        "best_score": best_score,
        "seed": seed,
        "model_label": model_label,
        "normalization_sha256": norm_sha,
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            ).stdout.strip(),
            "plan_root": materialization_report.get("plan_root"),
            "plan_sha256": materialization_report.get("plan_sha256"),
            "plan_sha256s_sha256": materialization_report.get("plan_sha256s_sha256"),
            "materialization_report_sha256": materialization_report_sha,
            "materialization_sha256s_sha256": materialization_sums_sha,
            "dataset_index_sha256": dataset_index_sha,
            "normalization_sha256": norm_sha,
            "feature_schema_sha256": materialization_report.get("feature_schema_sha256"),
            "label_schema_sha256": materialization_report.get("label_schema_sha256"),
            "runtime_gate_contract": {
                "runtime_gate_heads": ["critical_window", "release_safe", "grounding_confidence"],
                "training_only_auxiliary_heads": ["window_start", "burst_feasible", "contact_grasp"],
                "burst_length": 10,
                "persistence_window": 3,
                "persistence_required": 2,
            },
            "language_embedding_contract": "deterministic_hash_identity_proxy_v1",
        },
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")

    report = {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "model_label": model_label,
        "seed": seed,
        "epochs_completed": len(history),
        "best_score": best_score,
        "final_cal_metrics": history[-1]["cal_metrics"] if history else {},
        "checkpoint_sha256": sha256_file(output_dir / "checkpoint.pt"),
        "normalization_sha256": norm_sha,
    }
    write_json(output_dir / "training_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R9P preview detector")
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["a", "b"],
                        help="Model A (25D only) or Model B (25D+9D)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_label = args.model
    print(f"Training Model {model_label.upper()} seed={args.seed}")
    report = train_model(
        materialization_root=args.materialization_root,
        output_root=args.output_root,
        model_label=model_label,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )
    status = "PASS" if report["epochs_completed"] > 0 else "FAIL"
    print(f"Training: {status}  best_score={report['best_score']:.4f}  "
          f"epochs={report['epochs_completed']}")
    return 0 if report["epochs_completed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
