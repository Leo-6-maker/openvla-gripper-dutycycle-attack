"""Train R9P preview detector (Model A: 25D only, Model B: 25D+9D) on full episodes.

Loads per-episode NPZ files via the dataset index, batches variable-length episodes
with padding, and trains a causal GRU detector using the existing detector architecture
and clean_window_loss. Uses deterministic hash-based language embeddings (no OpenVLA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    clean_window_loss,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
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
VISUAL_DIM = 1152  # SigLIP embedding dim (not used in preview)


def _hash_language_embedding(text: str, dim: int = LANGUAGE_DIM) -> np.ndarray:
    """Deterministic hash-based language embedding (no OpenVLA dependency)."""
    h = hashlib.sha256(text.encode()).digest()
    # Use hash bytes to seed a deterministic random projection
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    projection = rng.randn(32, dim).astype(np.float32)
    # Convert hash to 32 floats
    values = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    if len(values) < 32:
        values = np.pad(values, (0, 32 - len(values)))
    values = values[:32]
    embedding = values @ projection
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding.astype(np.float32)


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

        # Episode fully known negative
        all_known = item["known_mask"].all()
        any_positive = item["targets"]["critical_window"].any() if all_known else False
        ep_fkn[i] = bool(all_known and not any_positive)

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
) -> dict[str, float]:
    model.eval()
    all_probs: dict[str, list] = defaultdict(list)
    all_targets: dict[str, list] = defaultdict(list)
    all_masks: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            proprio = batch["proprio_25d"].to(device)
            policy = batch["policy_intent"].to(device) if use_policy_intent else None
            language = batch["language"].to(device)
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
            tn = ((1 - pred) * (1 - tgt[msk])).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics[f"{h}_precision"] = precision
            metrics[f"{h}_recall"] = recall
        else:
            metrics[f"{h}_precision"] = 0.0
            metrics[f"{h}_recall"] = 0.0

    return metrics


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

    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            proprio = batch["proprio_25d"].to(device)
            policy = batch["policy_intent"].to(device) if use_policy_intent else None
            language = batch["language"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}

            outputs = model(
                proprio, language,
                policy_intent=policy,
                return_sequence=True,
            )

            # Apply padding mask: zero out outputs beyond valid steps
            pad_mask = batch["padding_mask"].to(device)
            for h in R9P_HEAD_NAMES:
                outputs[h] = outputs[h] * pad_mask.float()

            loss_dict = clean_window_loss(
                outputs, targets, masks,
                sample_weight=pad_mask.float(),
                auxiliary_weight=0.2,
                start_weight=1.0,
                active_weight=0.5,
                early_weight=0.25,
                miss_weight=0.50,
                negative_episode_weight=0.50,
                release_safe_episode_weight=0.50,
                include_episode_losses=True,
            )

            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.ndim == 0:
                    epoch_losses[k] += v.item()
            n_batches += 1

        # Validation
        cal_metrics = _evaluate_model(model, cal_loader, device, use_policy_intent)
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

    # Save checkpoint
    output_dir = output_root / f"model_{model_label}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    model_label = args.model  # "a" or "b"
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
