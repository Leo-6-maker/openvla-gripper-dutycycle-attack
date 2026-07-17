#!/usr/bin/env python3
"""Run one held-out 200-identity FIT validation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib.util

from gripper_attack.b3_formal import load_b3_checkpoint_bundle
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file
from gripper_attack.b3_v3_dataset import load_episode, load_formal_registry_csv, select_fit_fold_episodes


def _bundle_module():
    path = Path(__file__).with_name("build_b3_v3_prediction_bundle.py")
    spec = importlib.util.spec_from_file_location("b3_v3_prediction_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prediction bundle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_validation(*, checkpoint_root: Path, registry_csv: Path, s1_root: Path, fold_root: Path, output_root: Path, fold_id: int, seed: int, variant: str, policy_intent_root: Path | None = None) -> dict:
    model, config, normalization, payload, _ = load_b3_checkpoint_bundle(checkpoint_root, require_formal=True)
    if config.variant != variant or payload.get("checkpoint_status") != "FIT_FOLD_TRAINED_CANDIDATE":
        raise ValueError("checkpoint is not the requested fold candidate")
    coordinates = payload.get("extra", {})
    if coordinates.get("variant") != variant or int(coordinates.get("fold_id", -1)) != fold_id or int(coordinates.get("seed", -1)) != seed:
        raise ValueError("checkpoint coordinates do not match held-out validation request")
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    episodes = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None if policy_intent_root is None else policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    folds = load_fit_fold_bundle(fold_root)
    validation = select_fit_fold_episodes(episodes, folds, fold_id=fold_id, partition="validation")
    checkpoint_sha = sha256_file(checkpoint_root / "SHA256SUMS")
    module = _bundle_module()
    records = module.build_prediction_records(
        model, validation, normalization, checkpoint_sha256=checkpoint_sha,
        fold_id=fold_id, seed=seed, variant=variant,
    )
    validation_sha = folds["folds"][fold_id]["validation_identity_sha256"]
    return module.write_prediction_bundle(
        output_root, records, fold_id=fold_id, seed=seed, variant=variant,
        checkpoint_sha256=checkpoint_sha, validation_identity_sha256=validation_sha,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_root}")
    manifest = run_validation(
        checkpoint_root=args.checkpoint_root, registry_csv=args.registry_csv, s1_root=args.s1_root,
        fold_root=args.fold_root, output_root=args.output_root, fold_id=args.fold_id, seed=args.seed,
        variant=args.variant, policy_intent_root=args.policy_intent_root,
    )
    print(json.dumps({"status": "FIT_VALIDATION_BUNDLE_SEALED", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
