#!/usr/bin/env python3
"""Materialize and merge four-suite C2g datasets with suite-specific OpenVLA models.

Each suite's RGB/language embeddings are extracted with the exact policy checkpoint
used for that suite at deployment.  Per-suite artifacts remain hash-bound and the
combined NPZ contains no task/suite identity in the model feature tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.multisuite_detector.materialize_c2g_clean_window_dataset import (
    DATASET_SCHEMA_VERSION,
    SUITES,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
BASE_MATERIALIZER = REPO / "tools" / "multisuite_detector" / "materialize_c2g_clean_window_dataset.py"


def load_model_map(path: Path | None, backend: str) -> dict[str, str]:
    if backend != "openvla_siglip":
        return {suite: "" for suite in SUITES}
    if path is None:
        raise ValueError("--suite-model-map is required for openvla_siglip")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("suite model map must be a JSON object")
    output = {suite: str(value.get(suite, "")).strip() for suite in SUITES}
    missing = [suite for suite, model_path in output.items() if not model_path]
    if missing:
        raise ValueError("suite model map missing: " + ", ".join(missing))
    for suite, model_path in output.items():
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"{suite} model directory not found: {model_path}")
    return output


def find_suite_root(input_root: Path, suite: str) -> Path:
    candidates = [
        input_root / "episodes" / suite,
        input_root / suite,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("step_records.jsonl")):
            return candidate
    raise FileNotFoundError(f"no clean episodes found for {suite} under {input_root}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data.get("schema_version", "")) != DATASET_SCHEMA_VERSION:
        raise ValueError(f"schema mismatch in {path}")
    return data


def merge_datasets(paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    datasets = [load_npz(path) for path in paths]
    keys = set(datasets[0])
    if any(set(dataset) != keys for dataset in datasets[1:]):
        raise ValueError("per-suite NPZ field sets differ")
    constants = {"schema_version", "feature_names_policy"}
    payload: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        values = [dataset[key] for dataset in datasets]
        if key in constants:
            if any(not np.array_equal(values[0], value) for value in values[1:]):
                raise ValueError(f"constant field differs across suites: {key}")
            payload[key] = values[0]
            continue
        if any(value.ndim < 1 for value in values):
            raise ValueError(f"sample field must have a leading dimension: {key}")
        trailing = [value.shape[1:] for value in values]
        if len(set(trailing)) != 1:
            raise ValueError(f"field shape differs across suites: {key}: {trailing}")
        payload[key] = np.concatenate(values, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "combined_samples": int(payload["X_proprio"].shape[0]),
        "visual_dim": int(payload["X_visual"].shape[-1]),
        "language_dim": int(payload["X_language"].shape[-1]),
        "split_counts": {
            name: int(np.sum(payload["split"].astype(str) == name))
            for name in ("train", "val", "test")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path)
    parser.add_argument("--backend", choices=("stats", "clip", "openvla_siglip"), default="openvla_siglip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--split-mode", choices=("within_task", "leave_one_task_out", "leave_one_suite_out"), default="within_task")
    parser.add_argument("--held-out-task", default="")
    parser.add_argument("--held-out-suite", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--max-episodes-per-suite", type=int, default=0)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_map = load_model_map(args.suite_model_map.resolve() if args.suite_model_map else None, args.backend)
    per_suite_paths: list[Path] = []
    per_suite_reports: dict[str, Any] = {}
    for suite in SUITES:
        suite_root = find_suite_root(input_root, suite)
        suite_output = output_dir / "per_suite" / suite
        command = [
            sys.executable,
            str(BASE_MATERIALIZER),
            "--input-root", str(suite_root),
            "--output-dir", str(suite_output),
            "--window", str(args.window),
            "--burst-length", str(args.burst_length),
            "--backend", args.backend,
            "--device", args.device,
            "--model-name", args.model_name,
            "--embedding-dim", str(args.embedding_dim),
            "--split-mode", args.split_mode,
            "--held-out-task", args.held_out_task,
            "--held-out-suite", args.held_out_suite,
            "--val-fraction", str(args.val_fraction),
            "--test-fraction", str(args.test_fraction),
            "--positive-weight", str(args.positive_weight),
            "--max-episodes", str(args.max_episodes_per_suite),
            "--seed", str(args.seed),
            "--git-commit", args.git_commit,
            "--require-zero-errors",
        ]
        if args.backend == "openvla_siglip":
            command.extend(["--openvla-model-path", model_map[suite]])
        completed = subprocess.run(command, cwd=REPO)
        if completed.returncode != 0:
            raise RuntimeError(f"per-suite materialization failed for {suite}")
        dataset_path = suite_output / f"c2g_clean_window_w{args.window:02d}_{args.backend}_{args.split_mode}.npz"
        report_path = suite_output / "c2g_clean_window_materialization_report.json"
        if not dataset_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"per-suite outputs missing for {suite}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not str(report.get("status", "")).startswith("PASS_"):
            raise RuntimeError(f"per-suite report is not PASS for {suite}")
        per_suite_paths.append(dataset_path)
        per_suite_reports[suite] = {
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "model_path": model_map[suite],
            "n_windows": report.get("n_windows"),
            "n_episodes_processed": report.get("n_episodes_processed"),
        }

    combined_path = output_dir / f"c2g_clean_window_w{args.window:02d}_{args.backend}_{args.split_mode}.npz"
    merged = merge_datasets(per_suite_paths, combined_path)
    report = {
        "gate": "C2G_MULTISUITE_DATASET_MATERIALIZATION",
        "status": "PASS_C2G_MULTISUITE_DATASET_MATERIALIZED",
        "schema_version": DATASET_SCHEMA_VERSION,
        "combined_dataset": str(combined_path),
        "combined_dataset_sha256": sha256_file(combined_path),
        "backend": args.backend,
        "split_mode": args.split_mode,
        "per_suite": per_suite_reports,
        **merged,
        "boundaries": {
            "clean_only": True,
            "attack_outcomes_read": False,
            "suite_task_identity_used_as_model_feature": False,
        },
    }
    report_path = output_dir / "c2g_multisuite_materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
