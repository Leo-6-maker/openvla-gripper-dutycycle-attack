#!/usr/bin/env python3
"""train_early_grasp_detector.py — Causal TCN for early-grasp phase detection.

Model: 3-layer causal TCN with residual connections.
Input: 13-D runtime proprioceptive features.
Output: 3-class phase logits (pre_grasp, grasp_formation, post_grasp).

Uses only clean successful rollouts. No attack outcome. No privileged features.
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Causal TCN Block ──
class CausalConv1d(nn.Module):
    """1D convolution with causal padding (left-only)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))  # causal: pad left only
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = self.c1(x).transpose(1, 2)  # [B, T, C]
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # [B, C, T]
        out = self.c2(out).transpose(1, 2)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + residual


class EarlyGraspTCN(nn.Module):
    """Causal TCN for per-step phase classification."""
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=3,
                 kernel_size=3, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        B, T, _ = x.shape
        out = self.input_proj(x)  # [B, T, H]
        out = out.transpose(1, 2)  # [B, H, T]
        for block in self.blocks:
            out = block(out)
        out = out.transpose(1, 2)  # [B, T, H]
        logits = self.output_proj(out)  # [B, T, C]
        return logits


# ── Dataset ──
class SequenceDataset(Dataset):
    def __init__(self, npz_path, split_csv=None, split="train", split_col="split_task_holdout"):
        data = np.load(npz_path, allow_pickle=True)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.episode_ids = list(data.get("episode_ids", [f"ep_{i}" for i in range(len(self.X))]))

        if split_csv and os.path.exists(split_csv):
            with open(split_csv, newline="") as f:
                split_map = {r["episode_id"]: r[split_col] for r in csv.DictReader(f)}
            indices = [i for i, eid in enumerate(self.episode_ids)
                       if split_map.get(str(eid), "train") == split]
            self.X = self.X[indices]
            self.y = self.y[indices]
            self.mask = self.mask[indices]
            self.episode_ids = [self.episode_ids[i] for i in indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


# ── Metrics ──
def compute_metrics(logits, y, mask, phase_classes):
    """Per-step and trigger-level metrics."""
    pred = logits.argmax(dim=-1)  # [B, T]
    valid_mask = mask & (y != -100)

    # Per-step accuracy
    correct = (pred == y) & valid_mask
    acc = correct.sum().float() / valid_mask.sum().float()

    # Per-class F1
    f1s = {}
    for cls_name, cls_id in phase_classes.items():
        tp = ((pred == cls_id) & (y == cls_id) & valid_mask).sum().float()
        fp = ((pred == cls_id) & (y != cls_id) & valid_mask).sum().float()
        fn = ((pred != cls_id) & (y == cls_id) & valid_mask).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s[f"f1_{cls_name}"] = (2 * prec * rec / (prec + rec + 1e-8)).item()

    return {"acc": acc.item(), **f1s}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch, mask_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        logits = model(X_batch)  # [B, T, C]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_batch.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, phase_classes):
    model.eval()
    all_metrics = defaultdict(list)
    for X_batch, y_batch, mask_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        logits = model(X_batch)
        m = compute_metrics(logits, y_batch, mask_batch, phase_classes)
        for k, v in m.items():
            all_metrics[k].append(v)
    return {k: np.mean(v) for k, v in all_metrics.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v1.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_task_holdout")
    ap.add_argument("--output-dir", default="outputs/early_grasp_detector")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.npz_path):
        print(f"ERROR: NPZ not found: {args.npz_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    ds_kwargs = dict(npz_path=args.npz_path, split_csv=args.split_csv, split_col=args.split_col)
    train_ds = SequenceDataset(**ds_kwargs, split="train")
    val_ds = SequenceDataset(**ds_kwargs, split="val")
    test_ds = SequenceDataset(**ds_kwargs, split="test")

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    D = train_ds.X.shape[-1]
    model = EarlyGraspTCN(
        input_dim=D, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        kernel_size=args.kernel_size, num_classes=3, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, input_dim={D}")

    phase_classes = {"pre_grasp": 0, "grasp_formation": 1, "post_grasp": 2}

    if args.dry_run:
        print("DRY RUN complete.")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_metrics = eval_epoch(model, train_loader, device, phase_classes)
        val_metrics = eval_epoch(model, val_loader, device, phase_classes)
        scheduler.step()
        dt = time.time() - t0

        val_acc = val_metrics.get("acc", 0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_metrics.get("acc", 0), 4),
            "val_acc": round(val_acc, 4),
            "val_f1_grasp": round(val_metrics.get("f1_grasp_formation", 0), 4),
            "dt_s": round(dt, 2),
        })
        print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f} train_acc={train_metrics['acc']:.3f} "
              f"val_acc={val_acc:.3f} val_f1_grasp={val_metrics.get('f1_grasp_formation',0):.3f} "
              f"dt={dt:.1f}s")

    # Load best model (or last if no improvement)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = eval_epoch(model, test_loader, device, phase_classes)
    print(f"\nTest: acc={test_metrics['acc']:.4f} f1_grasp={test_metrics.get('f1_grasp_formation',0):.4f}")

    # Save outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"model_state": model.state_dict(), "config": vars(args)}, out_dir / "checkpoint.pt")

    metrics = {
        "best_val_acc": best_val_acc,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "n_params": n_params,
        **vars(args),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
