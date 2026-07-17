#!/usr/bin/env python3
"""Build one fold/variant-specific sealed normalization bundle.

The command is intentionally execution-gated.  Without the explicit
preparation flag it refuses to read any S1 root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, write_normalization_bundle
from gripper_attack.b3_v3_dataset import compute_fit_normalization, load_episode, load_formal_registry_csv, select_fit_fold_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--runner-binding-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-preparation", action="store_true")
    args = parser.parse_args()
    if not args.execute_preparation:
        raise SystemExit("FORMAL_TRAINING_HOLD: normalization preparation is not authorized without --execute-preparation")
    fold_manifest = load_fit_fold_bundle(args.fold_root)
    rows = load_formal_registry_csv(args.registry_csv, require_a_only=True)
    episodes: list = []
    if args.variant == "B3_25D9D" and args.policy_intent_root is None:
        raise SystemExit("B3_25D9D requires --policy-intent-root")
    for row in rows:
        root = args.s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None
        if args.policy_intent_root is not None:
            nine_d = args.policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    train = select_fit_fold_episodes(episodes, fold_manifest, fold_id=args.fold_id, partition="train")
    normalization = compute_fit_normalization(train, include_9d=args.variant == "B3_25D9D")
    binding = json.loads(args.runner_binding_json.read_text(encoding="utf-8"))
    write_normalization_bundle(
        args.output_root, normalization, fold_id=args.fold_id, variant=args.variant,
        train_identity_sha256=fold_manifest["folds"][args.fold_id]["train_identity_sha256"],
        registry_sha256=sha256_file(args.registry_csv),
        s1_corpus_sha256=sha256_file(args.s1_root / "SHA256SUMS"), runner_binding=binding,
    )
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "fold_id": args.fold_id, "variant": args.variant, "train_count": len(train)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
