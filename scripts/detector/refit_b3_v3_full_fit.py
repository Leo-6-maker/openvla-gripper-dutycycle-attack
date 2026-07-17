#!/usr/bin/env python3
"""Full-FIT refit entrypoint, gated by a sealed viability PASS.

The fold trainer never silently changes from 600 to 800 identities.  This
entrypoint is the only path that may create a FULL_FIT_REFIT_CANDIDATE, and it
requires a distinct machine-built ``fit_scope=FULL_FIT`` authorization.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path

from gripper_attack.b3_formal import load_b3_checkpoint_bundle, save_b3_checkpoint_bundle
from gripper_attack.b3_training_protocol import load_normalization_bundle, load_training_authorization_bundle, sha256_file, verify_sealed_directory
from gripper_attack.b3_v3_dataset import compute_fit_normalization, load_episode, load_formal_registry_csv
from gripper_attack.b3_v3_viability_decision import load_viability_decision


def load_full_fit_episodes(registry_csv: Path, s1_root: Path, *, policy_intent_root: Path | None = None):
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    episodes = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None if policy_intent_root is None else policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    if len(episodes) != 800:
        raise ValueError("full-FIT refit requires exactly 800 episodes")
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viability-decision-root", type=Path, required=True)
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
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--output-checkpoint-bundle", type=Path, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute-formal", action="store_true")
    args = parser.parse_args()
    viability = load_viability_decision(args.viability_decision_root)
    if viability.get("status") != "PASS" or args.variant not in viability.get("selected_variants", []):
        raise SystemExit("FULL_FIT_REFIT_HOLD: sealed viability decision did not select this variant")
    if not args.execute_formal:
        print(json.dumps({"status": "FULL_FIT_REFIT_HOLD", "reason": "explicit formal execution flag is required", "viability_runs": viability["run_count"], "formal_attack_authorized": False}, sort_keys=True))
        return 0
    authorization = load_training_authorization_bundle(args.authorization)
    if authorization.get("fit_scope") != "FULL_FIT" or authorization.get("variant") != args.variant or authorization.get("seed") != args.seed:
        raise SystemExit("FULL_FIT_REFIT_HOLD: authorization scope/variant/seed mismatch")
    if args.variant == "B3_25D" and args.policy_intent_root is not None:
        raise SystemExit("B3_25D must not receive --policy-intent-root")
    if args.variant == "B3_25D9D" and args.policy_intent_root is None:
        raise SystemExit("B3_25D9D requires --policy-intent-root")
    episodes = load_full_fit_episodes(args.registry_csv, args.s1_root, policy_intent_root=args.policy_intent_root)
    trainer_path = Path(__file__).with_name("train_b3_v3_detector.py")
    spec = importlib.util.spec_from_file_location("b3_v3_formal_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load formal trainer")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    before = trainer.verify_authorized_inputs(
        authorization, args.registry_csv, args.registry_summary, args.s1_root, args.s1_root_audit,
        args.source_contract, args.s1_protocol, args.training_protocol, args.feature_rebuilder,
        normalization_root=args.normalization_root, fold_root=args.fold_root, fold_id="FULL_FIT",
        variant=args.variant, policy_intent_root=args.policy_intent_root,
    )
    from gripper_attack.b3_official_v3_s1 import build_s1_runner_binding
    measured_binding = build_s1_runner_binding(
        runner_repo=args.runner_repo, expected_runner_head=authorization["runner_head"],
        config_path=args.runner_config, runner_script_path=args.runner_script,
    )
    if measured_binding != authorization["runner_binding"]:
        raise SystemExit("FULL_FIT_REFIT_HOLD: runner binding does not match authorization")
    policy_root_sha256 = None if args.policy_intent_root is None else sha256_file(args.policy_intent_root / "SHA256SUMS")
    normalization, normalization_source = load_normalization_bundle(
        args.normalization_root, fold_id="FULL_FIT", variant=args.variant,
        policy_intent_root_sha256=policy_root_sha256,
    )
    recomputed = compute_fit_normalization(episodes, include_9d=args.variant == "B3_25D9D")
    if authorization.get("normalization_sha256") != recomputed.sha256 or normalization.sha256 != recomputed.sha256:
        raise SystemExit("FULL_FIT_REFIT_HOLD: normalization is not recomputed from all 800 FIT episodes")
    model, losses = trainer.train_model(episodes, variant=args.variant, normalization=recomputed, seed=args.seed, device=args.device)
    after = trainer.verify_authorized_inputs(
        authorization, args.registry_csv, args.registry_summary, args.s1_root, args.s1_root_audit,
        args.source_contract, args.s1_protocol, args.training_protocol, args.feature_rebuilder,
        normalization_root=args.normalization_root, fold_root=args.fold_root, fold_id="FULL_FIT",
        variant=args.variant, policy_intent_root=args.policy_intent_root,
    )
    if after != before:
        raise SystemExit("FULL_FIT_REFIT_HOLD: authorized inputs changed during training")
    save_b3_checkpoint_bundle(
        args.output_checkpoint_bundle, model, recomputed, authorization=authorization,
        checkpoint_status="FULL_FIT_REFIT_CANDIDATE",
        extra={"fit_scope": "FULL_FIT", "fit_episode_count": 800, "seed": args.seed, "loss_history": losses, "viability_decision_sha256": sha256_file(args.viability_decision_root / "SHA256SUMS")},
    )
    print(json.dumps({"status": "FULL_FIT_REFIT_CANDIDATE", "fit_episode_count": 800, "formal_attack_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
