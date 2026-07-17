#!/usr/bin/env python3
"""Run one held-out 200-identity FIT validation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib.util

from gripper_attack.b3_formal import load_b3_checkpoint_bundle
from gripper_attack.b3_official_v3_s1 import build_s1_runner_binding
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, load_training_authorization_bundle, sha256_file
from gripper_attack.b3_v3_dataset import load_episode, load_formal_registry_csv, select_fit_fold_episodes


def _bundle_module():
    path = Path(__file__).with_name("build_b3_v3_prediction_bundle.py")
    spec = importlib.util.spec_from_file_location("b3_v3_prediction_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prediction bundle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer_module():
    path = Path(__file__).with_name("train_b3_v3_detector.py")
    spec = importlib.util.spec_from_file_location("b3_v3_formal_trainer_for_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_validation(
    *, checkpoint_root: Path,
    authorization_root: Path,
    registry_csv: Path,
    registry_summary: Path,
    s1_root: Path,
    s1_root_audit: Path,
    source_contract: Path,
    s1_protocol: Path,
    training_protocol: Path,
    feature_rebuilder: Path,
    normalization_root: Path,
    fold_root: Path,
    output_root: Path,
    fold_id: int,
    seed: int,
    variant: str,
    runner_repo: Path,
    runner_config: Path,
    runner_script: Path,
    policy_intent_root: Path | None = None,
) -> dict:
    if variant == "B3_25D" and policy_intent_root is not None:
        raise ValueError("B3_25D validation must not receive a 9D root")
    if variant == "B3_25D9D" and policy_intent_root is None:
        raise ValueError("B3_25D9D validation requires a sealed 9D root")
    authorization = load_training_authorization_bundle(authorization_root)
    model, config, normalization, payload, _ = load_b3_checkpoint_bundle(checkpoint_root, require_formal=True)
    if config.variant != variant or payload.get("checkpoint_status") != "FIT_FOLD_TRAINED_CANDIDATE":
        raise ValueError("checkpoint is not the requested fold candidate")
    coordinates = payload.get("extra", {})
    if coordinates.get("variant") != variant or int(coordinates.get("fold_id", -1)) != fold_id or int(coordinates.get("seed", -1)) != seed:
        raise ValueError("checkpoint coordinates do not match held-out validation request")
    embedded_authorization = payload.get("authorization")
    if not isinstance(embedded_authorization, dict) or embedded_authorization.get("authorization_payload_sha256") != authorization.get("authorization_payload_sha256"):
        raise ValueError("checkpoint and sealed authorization bundle do not match")
    if authorization.get("fit_scope") != "FIT_FOLD" or authorization.get("variant") != variant or authorization.get("fold_id") != fold_id or authorization.get("seed") != seed:
        raise ValueError("validation coordinates do not match authorization")
    trainer = _trainer_module()
    before = trainer.verify_authorized_inputs(
        authorization, registry_csv, registry_summary, s1_root, s1_root_audit, source_contract, s1_protocol,
        training_protocol, feature_rebuilder, normalization_root=normalization_root, fold_root=fold_root,
        fold_id=fold_id, variant=variant, policy_intent_root=policy_intent_root,
    )
    measured_binding = build_s1_runner_binding(
        runner_repo=runner_repo, expected_runner_head=authorization["runner_head"],
        config_path=runner_config, runner_script_path=runner_script,
    )
    if measured_binding != authorization["runner_binding"]:
        raise ValueError("validation runner binding does not match authorization")
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
    fold = folds["folds"][fold_id]
    actual_validation_ids = sorted(item.canonical_parent_key for item in validation)
    if actual_validation_ids != fold["validation_identities"]:
        raise ValueError("validation identities do not match sealed fold manifest")
    validation_sha = fold["validation_identity_sha256"]
    checkpoint_sha = sha256_file(checkpoint_root / "SHA256SUMS")
    source_bindings = {
        "registry_root_sha256": authorization["formal_registry_root_sha256"],
        "s1_root_sha256": authorization["s1_corpus_sha256"],
        "fold_bundle_sha256": authorization["fold_manifest_sha256"],
        "checkpoint_bundle_sha256": checkpoint_sha,
        "normalization_bundle_sha256": authorization["normalization_bundle_sha256"],
        "normalization_sha256": authorization["normalization_sha256"],
        "normalization_file_sha256": authorization["normalization_file_sha256"],
        "authorization_payload_sha256": authorization["authorization_payload_sha256"],
        "runner_binding_sha256": authorization["runner_binding"]["runner_binding_sha256"],
        "policy_intent_root_sha256": authorization.get("policy_intent_root_sha256"),
    }
    manifest = module.write_prediction_bundle(
        output_root, records, fold_id=fold_id, seed=seed, variant=variant,
        checkpoint_sha256=checkpoint_sha, validation_identity_sha256=validation_sha, source_bindings=source_bindings,
    )
    after = trainer.verify_authorized_inputs(
        authorization, registry_csv, registry_summary, s1_root, s1_root_audit, source_contract, s1_protocol,
        training_protocol, feature_rebuilder, normalization_root=normalization_root, fold_root=fold_root,
        fold_id=fold_id, variant=variant, policy_intent_root=policy_intent_root,
    )
    if after != before:
        raise ValueError("validation inputs changed during prediction")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s1-root-audit", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--s1-protocol", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--feature-rebuilder", type=Path, required=True)
    parser.add_argument("--normalization-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_root}")
    manifest = run_validation(
        checkpoint_root=args.checkpoint_root, authorization_root=args.authorization, registry_csv=args.registry_csv, registry_summary=args.registry_summary,
        s1_root=args.s1_root, s1_root_audit=args.s1_root_audit, source_contract=args.source_contract,
        s1_protocol=args.s1_protocol, training_protocol=args.training_protocol, feature_rebuilder=args.feature_rebuilder,
        normalization_root=args.normalization_root, fold_root=args.fold_root, output_root=args.output_root,
        fold_id=args.fold_id, seed=args.seed, variant=args.variant, runner_repo=args.runner_repo,
        runner_config=args.runner_config, runner_script=args.runner_script, policy_intent_root=args.policy_intent_root,
    )
    print(json.dumps({"status": "FIT_VALIDATION_BUNDLE_SEALED", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
