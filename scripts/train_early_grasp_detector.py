#!/usr/bin/env python3
"""train_early_grasp_detector.py — Causal TCN for early-grasp phase detection. v2.

Fixes (v2):
  - Class-weighted CE (--class-weight-mode inverse) for grasp imbalance.
  - Best checkpoint by val_f1_grasp_formation, NOT val_acc.
  - Per-class F1 printed each epoch.
  - Feature normalization from saved stats.
  - Class distribution printed before training.
  - checkpoint.pt includes full model_config for evaluation.
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


# ── Causal TCN (simplified: operate in [B,T,C] throughout) ──
class CausalConv1d(nn.Module):
    """1D causal conv operating in [B,T,C] format."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x):
        # x: [B,T,C] -> [B,C,T] -> conv -> [B,C,T] -> [B,T,C]
        xt = x.transpose(1, 2)
        return self.conv(F.pad(xt, (self.pad, 0))).transpose(1, 2)

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.n1 = nn.LayerNorm(ch)
        self.n2 = nn.LayerNorm(ch)
    def forward(self, x):
        # x: [B, T, C]
        residual = x
        out = self.n1(F.relu(self.c1(x)))
        out = self.dropout(out)
        out = self.n2(F.relu(self.c2(out)))
        out = self.dropout(out)
        return out + residual

class EarlyGraspTCN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=3,
                 kernel_size=3, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, 2**i, dropout) for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        # x: [B, T, D] -> [B, T, H]
        out = self.input_proj(x)
        for blk in self.blocks:
            out = blk(out)
        return self.output_proj(out)  # [B, T, C]


class SequenceDataset(Dataset):
    def __init__(self, npz_path, split_csv=None, split="train", split_col="split_state_holdout"):
        data = np.load(npz_path, allow_pickle=True)
        X_key = "X_norm" if "X_norm" in data else "X"
        self.X = torch.from_numpy(data[X_key]).float()
        self.y = torch.from_numpy(data["y"]).long()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.episode_ids = list(data.get("episode_ids", [f"ep_{i}" for i in range(len(self.X))]))
        if split_csv and os.path.exists(split_csv):
            with open(split_csv, newline="") as f:
                smap = {r["episode_id"]: r[split_col] for r in csv.DictReader(f)}
            idx = [i for i, eid in enumerate(self.episode_ids) if smap.get(str(eid), "train") == split]
            self.X = self.X[idx]; self.y = self.y[idx]; self.mask = self.mask[idx]
            self.episode_ids = [self.episode_ids[i] for i in idx]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.mask[idx]


# ── Metrics ──
PHASE_CLASSES = {"pre_grasp": 0, "grasp_formation": 1, "post_grasp": 2}
ID_TO_NAME = {v: k for k, v in PHASE_CLASSES.items()}

def compute_detailed_metrics(logits, y, mask):
    pred = logits.argmax(dim=-1)
    valid = mask & (y != -100)
    total_valid = valid.sum().float()
    acc = (pred == y)[valid].sum().float() / total_valid if total_valid > 0 else torch.tensor(0.0)
    metrics = {"acc": acc.item()}
    for name, cid in PHASE_CLASSES.items():
        tp = ((pred == cid) & (y == cid) & valid).sum().float()
        fp = ((pred == cid) & (y != cid) & valid).sum().float()
        fn = ((pred != cid) & (y == cid) & valid).sum().float()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        metrics[f"f1_{name}"] = (2 * prec * rec / (prec + rec + 1e-8)).item()
    # macro F1
    f1s = [metrics[f"f1_{n}"] for n in PHASE_CLASSES]
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    return metrics


def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0.0; n_batches = 0
    w = class_weights.to(device) if class_weights is not None else None
    for Xb, yb, mb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                               weight=w, ignore_index=-100)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_m = defaultdict(list)
    for Xb, yb, mb in loader:
        Xb = Xb.to(device); yb = yb.to(device); mb = mb.to(device)
        logits = model(Xb)
        m = compute_detailed_metrics(logits, yb, mb)
        for k, v in m.items(): all_m[k].append(v)
    return {k: np.mean(v) for k, v in all_m.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_state_holdout")
    ap.add_argument("--norm-stats", default="data/detector/object_clean_feature_norm_stats_v2.json")
    ap.add_argument("--output-dir", default="outputs/early_grasp_detector")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class-weight-mode", choices=["none", "inverse"], default="inverse")
    ap.add_argument("--max-class-weight", type=float, default=10.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.npz_path):
        print(f"ERROR: NPZ not found: {args.npz_path}"); sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    ds_kwargs = dict(npz_path=args.npz_path, split_csv=args.split_csv, split_col=args.split_col)
    train_ds = SequenceDataset(**ds_kwargs, split="train")
    val_ds = SequenceDataset(**ds_kwargs, split="val")
    test_ds = SequenceDataset(**ds_kwargs, split="test")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Class distribution
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = {}
        for cid in range(3):
            cnt = int(((ds.y != -100) & (ds.y == cid)).sum())
            counts[ID_TO_NAME[cid]] = cnt
        total = sum(counts.values())
        print(f"  {name}: " + ", ".join(f"{k}={v} ({100*v/max(total,1):.1f}%)" for k, v in counts.items()))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Class weights
    class_weights = None
    if args.class_weight_mode == "inverse":
        counts = [(int(((train_ds.y != -100) & (train_ds.y == cid)).sum())) for cid in range(3)]
        total = sum(counts)
        weights = [total / (3 * max(c, 1)) for c in counts]
        weights = [min(w, args.max_class_weight) for w in weights]
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights: {[round(w,2) for w in weights]}")

    # Model
    D = train_ds.X.shape[-1]
    model_config = dict(input_dim=D, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                        kernel_size=args.kernel_size, num_classes=3, dropout=args.dropout)
    model = EarlyGraspTCN(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, input_dim={D}")

    if args.dry_run:
        print("DRY RUN complete."); return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Track best by val_f1_grasp_formation
    best_metric = -1.0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        train_m = eval_epoch(model, train_loader, device)
        val_m = eval_epoch(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        val_f1_grasp = val_m.get("f1_grasp_formation", 0)
        if val_f1_grasp > best_metric:
            best_metric = val_f1_grasp
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        history.append(dict(epoch=epoch+1, train_loss=round(train_loss,4),
                           train_acc=round(train_m["acc"],4),
                           val_acc=round(val_m["acc"],4),
                           val_macro_f1=round(val_m["macro_f1"],4),
                           val_f1_pre=round(val_m["f1_pre_grasp"],4),
                           val_f1_grasp=round(val_f1_grasp,4),
                           val_f1_post=round(val_m["f1_post_grasp"],4),
                           dt_s=round(dt,2)))
        print(f"  E{epoch+1:3d}: loss={train_loss:.4f} acc={val_m['acc']:.3f} "
              f"macroF1={val_m['macro_f1']:.3f} graspF1={val_f1_grasp:.3f} "
              f"preF1={val_m['f1_pre_grasp']:.3f} postF1={val_m['f1_post_grasp']:.3f} "
              f"{'*' if val_f1_grasp == best_metric else ''}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest epoch: {best_epoch}, val_f1_grasp={best_metric:.4f}")
    else:
        print("\nNo improvement — using last epoch")

    test_m = eval_epoch(model, test_loader, device)
    print(f"Test: acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} "
          f"graspF1={test_m['f1_grasp_formation']:.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full config for evaluation
    full_config = {
        **model_config,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": [round(w.item(), 4) for w in class_weights] if class_weights is not None else None,
        "best_epoch": best_epoch,
        "best_val_f1_grasp": best_metric,
        "n_params": n_params,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
    }
    torch.save({"model_state": best_state if best_state is not None else model.state_dict(),
                "config": full_config}, out_dir / "checkpoint.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({**full_config, "test_metrics": {k: round(v,4) for k, v in test_m.items()}}, f, indent=2)

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
