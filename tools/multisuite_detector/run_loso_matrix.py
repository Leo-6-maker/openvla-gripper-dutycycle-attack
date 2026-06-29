#!/usr/bin/env python3
"""Run the full 4-fold LOSO matrix: train on 3 suites, evaluate on held-out suite.

For each fold, the held-out suite is excluded from training, normalization,
threshold selection, and early stopping. Validation is drawn from 3 training suites.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

LOSO_FOLDS = [
    {"name": "loso_libero10", "train_suites": ["libero_object", "libero_spatial", "libero_goal"], "test_suite": "libero_10"},
    {"name": "loso_goal", "train_suites": ["libero_object", "libero_spatial", "libero_10"], "test_suite": "libero_goal"},
    {"name": "loso_spatial", "train_suites": ["libero_object", "libero_goal", "libero_10"], "test_suite": "libero_spatial"},
    {"name": "loso_object", "train_suites": ["libero_spatial", "libero_goal", "libero_10"], "test_suite": "libero_object"},
]


def run_fold(fold: dict, args) -> dict:
    """Run one LOSO fold: build split, train, evaluate."""
    fold_name = fold["name"]
    test_suite = fold["test_suite"]
    out_dir = Path(args.output_dir) / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fold: {fold_name} (test={test_suite})")
    print(f"{'='*60}")

    # Step 1: Build LOSO split
    split_file = out_dir / "split.json"
    cmd = [
        sys.executable, str(Path(__file__).parent / "build_detector_splits.py"),
        "--episode_index", args.episode_index,
        "--split_type", "loso",
        "--loso_fold", fold_name,
        "--output_dir", str(out_dir),
        "--seed", str(args.seed),
    ]
    print(f"Building split: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Step 2: Train
    ckpt_file = out_dir / f"best_model.pt"
    train_cmd = [
        sys.executable, str(Path(__file__).parent / "train_detector.py"),
        "--feature_csv", args.feature_csv,
        "--label_csv", args.label_csv,
        "--split_file", str(split_file),
        "--output_dir", str(out_dir),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--patience", str(args.patience),
    ]
    if args.dry_run:
        train_cmd.append("--dry_run")
    print(f"Training: python {' '.join(train_cmd[1:])}")
    subprocess.run(train_cmd, check=True)

    # Step 3: Evaluate
    eval_file = out_dir / "test_metrics.json"
    eval_cmd = [
        sys.executable, str(Path(__file__).parent / "evaluate_detector.py"),
        "--checkpoint", str(ckpt_file),
        "--feature_csv", args.feature_csv,
        "--label_csv", args.label_csv,
        "--split_file", str(split_file),
        "--split_key", "test",
        "--output", str(eval_file),
        "--tau_corridor", str(args.tau_corridor),
        "--tau_release", str(args.tau_release),
        "--guard", str(args.guard),
    ]
    print(f"Evaluating: python {' '.join(eval_cmd[1:])}")
    subprocess.run(eval_cmd, check=True)

    with open(eval_file) as f:
        metrics = json.load(f)

    return {
        "fold": fold_name,
        "test_suite": test_suite,
        "train_suites": fold["train_suites"],
        "metrics": metrics,
        "checkpoint": str(ckpt_file),
    }


def main():
    ap = argparse.ArgumentParser(description="Run full LOSO matrix")
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    results = []
    for fold in LOSO_FOLDS:
        t0 = time.time()
        result = run_fold(fold, args)
        result["wall_time_s"] = time.time() - t0
        results.append(result)

    summary = {
        "gate": "LOSO_MATRIX_COMPLETE",
        "n_folds": len(results),
        "folds": results,
        "aggregate": {},
    }
    summary_path = Path(args.output_dir) / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLOSO matrix written to {summary_path}")


if __name__ == "__main__":
    main()
