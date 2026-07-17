#!/usr/bin/env python3
"""Build one fold/variant-specific sealed normalization bundle.

The command is intentionally execution-gated.  Without the explicit
preparation flag it refuses to read any S1 root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_formal import B3ModelConfig, json_sha
from gripper_attack.b3_official_v3_s1 import audit_materialized_root, build_s1_runner_binding, load_formal_fit_registry, verify_checksum_manifest
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, write_normalization_bundle
from gripper_attack.b3_v3_dataset import compute_fit_normalization, load_episode, select_fit_fold_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s1-root-audit", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--s1-protocol", type=Path, required=True)
    parser.add_argument("--feature-rebuilder", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--fit-scope", choices=("FIT_FOLD", "FULL_FIT"), default="FIT_FOLD")
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-preparation", action="store_true")
    args = parser.parse_args()
    if not args.execute_preparation:
        raise SystemExit("FORMAL_TRAINING_HOLD: normalization preparation is not authorized without --execute-preparation")
    fold_manifest = load_fit_fold_bundle(args.fold_root)
    verify_checksum_manifest(args.registry_csv.parent)
    verify_checksum_manifest(args.s1_root)
    rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    binding = build_s1_runner_binding(
        runner_repo=args.runner_repo, expected_runner_head=args.expected_runner_head,
        config_path=args.runner_config, runner_script_path=args.runner_script,
    )
    root_report = audit_materialized_root(
        args.s1_root, rows, require_runner_binding=True,
        feature_order_sha256=B3ModelConfig().feature_order_sha256,
        expected_runner_binding=binding,
        expected_input_sha256={
            "registry_csv_sha256": sha256_file(args.registry_csv),
            "registry_summary_sha256": sha256_file(args.registry_summary),
            "source_contract_sha256": sha256_file(args.source_contract),
            "protocol_sha256": sha256_file(args.s1_protocol),
            "feature_rebuilder_sha256": sha256_file(args.feature_rebuilder),
        },
    )
    if root_report.get("status") != "PASS":
        raise SystemExit("S1 root audit is not PASS")
    audit_payload = json.loads(args.s1_root_audit.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "PASS":
        raise SystemExit("independent S1 root audit input is not PASS")
    episodes: list = []
    if args.variant == "B3_25D9D" and args.policy_intent_root is None:
        raise SystemExit("B3_25D9D requires --policy-intent-root")
    for row in rows:
        root = args.s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None
        if args.policy_intent_root is not None:
            nine_d = args.policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    fold_id = "FULL_FIT" if args.fit_scope == "FULL_FIT" else int(args.fold_id)
    train = episodes if args.fit_scope == "FULL_FIT" else select_fit_fold_episodes(episodes, fold_manifest, fold_id=fold_id, partition="train")
    normalization = compute_fit_normalization(train, include_9d=args.variant == "B3_25D9D")
    write_normalization_bundle(
        args.output_root, normalization, fold_id=fold_id, variant=args.variant,
        train_identity_sha256=(json_sha(sorted(row["canonical_parent_key"] for row in rows)) if args.fit_scope == "FULL_FIT" else fold_manifest["folds"][fold_id]["train_identity_sha256"]),
        registry_sha256=sha256_file(args.registry_csv),
        s1_corpus_sha256=sha256_file(args.s1_root / "SHA256SUMS"), runner_binding=binding,
    )
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "fold_id": args.fold_id, "variant": args.variant, "train_count": len(train)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
