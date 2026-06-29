#!/usr/bin/env python3
"""Run 4-fold LOSO matrix with per-suite validation stratification.

Each fold uses a fold-specific config that must declare matching test_suite.
Validation split is stratified within each training suite (not pooled).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

LOSO_FOLDS = [
    {"name": "loso_libero10", "train": ["libero_object","libero_spatial","libero_goal"], "test": "libero_10"},
    {"name": "loso_goal", "train": ["libero_object","libero_spatial","libero_10"], "test": "libero_goal"},
    {"name": "loso_spatial", "train": ["libero_object","libero_goal","libero_10"], "test": "libero_spatial"},
    {"name": "loso_object", "train": ["libero_spatial","libero_goal","libero_10"], "test": "libero_object"},
]

TOOLS = Path(__file__).resolve().parent


def parse_config(path: str) -> dict:
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
    with open(path) as f:
        return json.load(f)


def validate_fold_config(config: dict, fold: dict) -> list[str]:
    """Verify config test_suite and train_suites match fold declaration."""
    errors = []
    cfg_test = config.get("test_suite")
    if cfg_test and cfg_test != fold["test"]:
        errors.append("Config test_suite={} != fold test_suite={}".format(cfg_test, fold["test"]))
    cfg_train = config.get("train_suites") or config.get("training_suites")
    if cfg_train and sorted(cfg_train) != sorted(fold["train"]):
        errors.append("Config train_suites mismatch")
    return errors


def run_fold(fold, args, config):
    fold_name = fold["name"]
    test_suite = fold["test"]
    out_dir = Path(args.output_dir) / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Fold: {} (test={})".format(fold_name, test_suite))
    print("=" * 60)

    # Validate config matches fold
    cfg_errors = validate_fold_config(config, fold)
    if cfg_errors:
        for e in cfg_errors:
            print("CONFIG ERROR: {}".format(e))
        raise ValueError("Fold config mismatch")

    # Step 1: Build LOSO split
    split_file = out_dir / "split_loso.json"
    cmd = [sys.executable, str(TOOLS / "build_detector_splits.py"),
           "--episode_index", args.episode_index,
           "--split_type", "loso", "--loso_fold", fold_name,
           "--output_dir", str(out_dir), "--seed", str(args.seed)]
    print("Build split: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    if args.dry_run:
        print("DRY_RUN: split built, skipping train/eval.")
        return {"fold": fold_name, "test_suite": test_suite, "dry_run": True}

    # Step 2: Train
    ckpt_file = out_dir / "best_model.pt"
    train_cmd = [sys.executable, str(TOOLS / "train_detector.py"),
                 "--config", args.config,
                 "--feature_csv", args.feature_csv,
                 "--label_csv", args.label_csv,
                 "--episode_index", args.episode_index,
                 "--split_file", str(split_file),
                 "--output_dir", str(out_dir),
                 "--seed", str(args.seed),
                 "--epochs", str(args.epochs),
                 "--batch_size", str(args.batch_size),
                 "--patience", str(args.patience)]
    print("Train: python " + " ".join(train_cmd[1:]))
    subprocess.run(train_cmd, check=True)
    if not ckpt_file.exists():
        raise RuntimeError("Training did not produce: {}".format(ckpt_file))

    # Step 3: Evaluate
    eval_file = out_dir / "test_metrics.json"
    eval_cmd = [sys.executable, str(TOOLS / "evaluate_detector.py"),
                "--checkpoint", str(ckpt_file),
                "--feature_csv", args.feature_csv,
                "--label_csv", args.label_csv,
                "--episode_index", args.episode_index,
                "--split_file", str(split_file),
                "--split_key", "test",
                "--output", str(eval_file),
                "--tau_corridor", str(args.tau_corridor),
                "--tau_release", str(args.tau_release),
                "--guard", str(args.guard)]
    print("Evaluate: python " + " ".join(eval_cmd[1:]))
    subprocess.run(eval_cmd, check=True)

    with open(eval_file) as f:
        metrics = json.load(f)
    return {"fold": fold_name, "test_suite": test_suite,
            "train_suites": fold["train"], "metrics": metrics,
            "checkpoint": str(ckpt_file)}


def main():
    ap = argparse.ArgumentParser(description="Run full LOSO matrix")
    ap.add_argument("--config", required=True, help="Base config or config directory")
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

    # Load base config
    base_config = parse_config(args.config) if Path(args.config).is_file() else {}

    results = []
    for fold in LOSO_FOLDS:
        # Try fold-specific config: config_dir/loso_object.yaml etc.
        config = base_config
        config_dir = Path(args.config).parent if Path(args.config).is_file() else Path(args.config)
        fold_config_path = config_dir / "{}.yaml".format(fold["name"])
        if fold_config_path.exists():
            config = parse_config(str(fold_config_path))
        elif config_dir.is_dir():
            alt = config_dir / "loso_{}.yaml".format(fold["test"].replace("libero_", ""))
            if alt.exists():
                config = parse_config(str(alt))

        t0 = time.time()
        result = run_fold(fold, args, config)
        result["wall_time_s"] = time.time() - t0
        results.append(result)

    summary = {"gate": "LOSO_MATRIX_COMPLETE", "n_folds": len(results), "folds": results}
    summary_path = Path(args.output_dir) / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nLOSO matrix: {}".format(summary_path))


if __name__ == "__main__":
    main()
