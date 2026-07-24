#!/usr/bin/env python3
"""Run leave-one-task-out or leave-one-suite-out C2g training folds.

The base NPZ is copied with only its episode-level split field changed.  No label,
feature, or attacked outcome is regenerated during fold construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "tools" / "multisuite_detector" / "train_c2g_clean_window_detector.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(seed: int, episode_key: str) -> float:
    value = hashlib.sha256(f"{seed}|{episode_key}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") / float(1 << 64)


def fold_split(
    suite: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    *,
    mode: str,
    held_out: str,
    seed: int,
    val_fraction: float,
) -> np.ndarray:
    result = np.empty(len(episode), dtype="<U5")
    for index, (suite_name, task_index, episode_key) in enumerate(zip(suite.astype(str), task, episode.astype(str))):
        identity = suite_name if mode == "loso" else f"{suite_name}:{int(task_index)}"
        if identity == held_out:
            result[index] = "test"
        else:
            result[index] = "val" if stable_fraction(seed, episode_key) < val_fraction else "train"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("loto", "loso"), default="loto")
    parser.add_argument("--held-out", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--no-use-visual", action="store_true")
    parser.add_argument("--no-use-policy-intent", action="store_true")
    parser.add_argument("--no-use-language-conditioning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    with np.load(args.dataset.resolve(), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    suite = data["suite"].astype(str)
    task = data["task_index"].astype(np.int64)
    episode = data["episode_key"].astype(str)
    if args.held_out:
        folds = list(dict.fromkeys(args.held_out))
    elif args.mode == "loso":
        folds = sorted(set(suite.tolist()))
    else:
        folds = sorted({f"{suite_name}:{int(task_index)}" for suite_name, task_index in zip(suite, task)})
    if not folds:
        raise RuntimeError("no folds found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(folds):
        safe_name = held_out.replace(":", "_").replace("/", "_")
        fold_root = args.output_root / f"fold_{fold_index:03d}_{safe_name}"
        fold_root.mkdir(parents=True, exist_ok=True)
        split = fold_split(
            suite,
            task,
            episode,
            mode=args.mode,
            held_out=held_out,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
        counts = {name: int(np.sum(split == name)) for name in ("train", "val", "test")}
        if min(counts.values()) <= 0:
            summaries.append({"held_out": held_out, "status": "HOLD_EMPTY_SPLIT", "split_counts": counts})
            continue
        fold_dataset = fold_root / "dataset.npz"
        np.savez_compressed(fold_dataset, **{**data, "split": split})
        train_root = fold_root / "training"
        command = [
            sys.executable,
            str(TRAINER),
            "--dataset", str(fold_dataset),
            "--output-dir", str(train_root),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--hidden", str(args.hidden),
            "--seed", str(args.seed + fold_index),
            "--git-commit", args.git_commit,
        ]
        if args.no_use_visual:
            command.append("--no-use-visual")
        if args.no_use_policy_intent:
            command.append("--no-use-policy-intent")
        if args.no_use_language_conditioning:
            command.append("--no-use-language-conditioning")
        if args.dry_run:
            summaries.append({"held_out": held_out, "status": "DRY_RUN", "split_counts": counts, "command": command})
            continue
        completed = subprocess.run(command, cwd=REPO)
        report_path = train_root / "c2g_clean_window_training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        summaries.append(
            {
                "held_out": held_out,
                "status": "PASS" if completed.returncode == 0 else "HOLD",
                "returncode": completed.returncode,
                "split_counts": counts,
                "dataset_sha256": sha256_file(fold_dataset),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "test_metrics": report.get("test_metrics"),
            }
        )
        if completed.returncode != 0:
            break

    status = "PASS_C2G_FOLD_TRAINING" if all(row["status"] in {"PASS", "DRY_RUN"} for row in summaries) else "HOLD_C2G_FOLD_TRAINING"
    output = {
        "gate": "C2G_CLEAN_WINDOW_FOLD_TRAINING",
        "status": status,
        "mode": args.mode,
        "base_dataset": str(args.dataset.resolve()),
        "base_dataset_sha256": sha256_file(args.dataset.resolve()),
        "fold_count": len(folds),
        "folds": summaries,
    }
    report_path = args.output_root / "c2g_fold_training_report.json"
    report_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
