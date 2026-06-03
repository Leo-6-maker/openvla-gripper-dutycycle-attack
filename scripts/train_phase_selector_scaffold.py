#!/usr/bin/env python3
"""train_phase_selector_scaffold.py — clean-only causal phase selector training SCAFFOLD.

Training loop requires server GPU (PyTorch). This scaffold validates config,
prevents privileged leakage, and saves training-ready config for GPU execution.
Rename to train_phase_selector.py when training loop is implemented.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

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
    ap = argparse.ArgumentParser(description="Train clean-only phase selector (scaffold)")
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


def main():
    args = parse_args()

    # ── P8: HARD FAIL on privileged leakage ──
    for pf in PRIVILEGED_FEATURES:
        if pf in args.features:
            print(f"FATAL: privileged feature '{pf}' in input features. Remove it.")
            sys.exit(1)

    if args.dry_run:
        print("DRY RUN: train_phase_selector_scaffold")
        print(f"  Features: {args.features}")
        print(f"  Label mode: {args.label_mode}")
        print(f"  Split: {args.split_mode}")
        print(f"  Output: {args.output_dir}")
        print(f"  Privileged HARD EXCLUDED: {PRIVILEGED_FEATURES}")
        return

    if not os.path.exists(args.phase_csv):
        print(f"ERROR: phase CSV not found: {args.phase_csv}")
        sys.exit(1)

    with open(args.phase_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Validate features exist in CSV
    csv_cols = set(rows[0].keys()) if rows else set()
    feat_cols = set(f"feat_{f}" for f in args.features)
    missing = feat_cols - csv_cols
    if missing:
        print(f"WARNING: {len(missing)} features not found in CSV columns: {sorted(missing)[:5]}...")

    if not args.include_unknown:
        label_col = f"phase_label_{args.label_mode.replace('class','')}class_id"
        rows = [r for r in rows if int(r.get(label_col, -1)) >= 0]
    print(f"Loaded {len(rows)} labeled steps (label_mode={args.label_mode})")

    n_classes = 3 if args.label_mode == "3class" else 6

    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "scaffold": True, "training_loop": "deferred_to_gpu",
        "label_mode": args.label_mode, "n_classes": n_classes,
        "features": args.features, "window_len": args.window_len,
        "hidden_dim": args.hidden_dim, "tcn_layers": args.tcn_layers,
        "privileged_excluded": PRIVILEGED_FEATURES,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "split_mode": args.split_mode, "seed": args.seed,
        "n_labeled_samples": len(rows),
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.output_dir, "feature_schema.json"), "w") as f:
        json.dump({"runtime_input_features": args.features,
            "privileged_available_but_excluded": PRIVILEGED_FEATURES,
            "missing_feature_policy": "impute_zero_or_empty",
            "label_source": "heuristic_phase_labels"}, f, indent=2)
    with open(os.path.join(args.output_dir, "label_schema.json"), "w") as f:
        labels = {0:"pre_grasp",1:"grasp_formation",2:"post_grasp"} if args.label_mode=="3class" else \
                 {0:"approach",1:"pregrasp",2:"grasp_formation",3:"stable_grasp_or_lift",4:"carry_or_place",5:"release_or_done"}
        json.dump(labels, f, indent=2)

    print(f"Scaffold config saved to {args.output_dir}")
    print("Full training loop requires server GPU. Script validates config and prevents privilege leakage.")


if __name__ == "__main__":
    main()
