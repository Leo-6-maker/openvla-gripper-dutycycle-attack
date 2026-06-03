#!/usr/bin/env python3
"""train_phase_selector.py — train clean-only causal phase selector from proprio/action features.

Model: causal TCN over temporal window.
Output: per-step 3-class or 6-class phase probabilities.
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

DEFAULT_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

PRIVILEGED_FEATURES = [
    "object_x", "object_y", "object_z", "target_x", "target_y", "target_z",
    "eef_object_dist", "object_target_dist", "object_lifted", "contact_flag",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Train clean-only phase selector")
    ap.add_argument("--phase-csv", required=True)
    ap.add_argument("--output-dir", default="outputs/phase_selector/trial")
    ap.add_argument("--label-mode", choices=["3class","6class"], default="3class")
    ap.add_argument("--window-len", type=int, default=16)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--tcn-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--split-mode", choices=["by_seed","by_task"], default="by_seed")
    ap.add_argument("--train-seeds", type=int, nargs="+")
    ap.add_argument("--val-seeds", type=int, nargs="+")
    ap.add_argument("--test-seeds", type=int, nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-unknown", action="store_true")
    return ap.parse_args()


def build_causal_tcn(in_dim, hidden_dim, n_classes, tcn_layers=3, window_len=16):
    """Build a simple causal TCN model using PyTorch if available."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class CausalPhaseTCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(in_dim, hidden_dim)
                dilations = [2**i for i in range(tcn_layers)]
                self.convs = nn.ModuleList([
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2*d, dilation=d)
                    for d in dilations
                ])
                self.drop = nn.Dropout(0.1)
                self.head = nn.Linear(hidden_dim, n_classes)

            def forward(self, x):
                # x: [B, T, D]
                x = self.proj(x).transpose(1, 2)  # [B, H, T]
                residuals = []
                for c in self.convs:
                    r = x
                    x = F.relu(c(x))
                    x = x[:, :, -r.shape[2]:] + r[:, :, -x.shape[2]:]
                    x = self.drop(x)
                x = x[:, :, -1]  # last temporal step
                return self.head(x)

        return CausalPhaseTCN()
    except ImportError:
        return None


def main():
    args = parse_args()

    if args.dry_run:
        print(f"DRY RUN: phase selector training")
        print(f"  CSV: {args.phase_csv}")
        print(f"  Features: {args.features}")
        print(f"  Label mode: {args.label_mode}")
        print(f"  Split: {args.split_mode}")
        print(f"  Output: {args.output_dir}")
        print(f"  Privileged features EXCLUDED from input: {PRIVILEGED_FEATURES}")
        return

    # Load data
    if not os.path.exists(args.phase_csv):
        print(f"ERROR: phase CSV not found: {args.phase_csv}")
        sys.exit(1)

    with open(args.phase_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Filter unknown labels
    if not args.include_unknown:
        label_col = f"phase_label_{args.label_mode.replace('class','')}class_id"
        rows = [r for r in rows if int(r.get(label_col, -1)) >= 0]
    print(f"Loaded {len(rows)} labeled steps")

    # Check for privileged leakage
    for pf in PRIVILEGED_FEATURES:
        if pf in args.features:
            print(f"WARNING: privileged feature '{pf}' in input features — remove for clean-only selector")
    for pf in PRIVILEGED_FEATURES:
        if pf in rows[0]:
            print(f"INFO: privileged field '{pf}' available in CSV (not used as model input)")

    # Build model
    n_classes = 3 if args.label_mode == "3class" else 6
    model = build_causal_tcn(len(args.features), args.hidden_dim, n_classes, args.tcn_layers, args.window_len)

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "label_mode": args.label_mode, "n_classes": n_classes,
        "features": args.features, "window_len": args.window_len,
        "hidden_dim": args.hidden_dim, "tcn_layers": args.tcn_layers,
        "privileged_excluded": PRIVILEGED_FEATURES,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "split_mode": args.split_mode, "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_dir, "feature_schema.json"), "w") as f:
        json.dump({"input_features": args.features, "privileged_excluded": PRIVILEGED_FEATURES}, f, indent=2)

    with open(os.path.join(args.output_dir, "label_schema.json"), "w") as f:
        labels = {0:"pre_grasp",1:"grasp_formation",2:"post_grasp"} if args.label_mode=="3class" else \
                 {0:"approach",1:"pregrasp",2:"grasp_formation",3:"stable_grasp_or_lift",4:"carry_or_place",5:"release_or_done"}
        json.dump(labels, f, indent=2)

    print(f"Config saved to {args.output_dir}")
    print("Training deferred — run on server with GPU (PyTorch required).")


if __name__ == "__main__":
    main()
