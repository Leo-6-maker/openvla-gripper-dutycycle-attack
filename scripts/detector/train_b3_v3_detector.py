#!/usr/bin/env python3
"""Formal Trainer entrypoint with an explicit sealed authorization gate."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from gripper_attack.b3_formal import (
    B3ModelConfig, B3Normalization, build_b3_model, compute_b3_loss,
    save_b3_checkpoint_bundle, validate_training_authorization,
)
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, load_normalization_bundle, load_training_authorization_bundle, sha256_file
from gripper_attack.b3_v3_dataset import B3Episode, B3EpisodeSampler, compute_fit_normalization, load_episode, load_formal_registry_csv, pad_episode_batch, select_fit_fold_episodes
from gripper_attack.b3_official_v3_s1 import audit_materialized_root, build_s1_runner_binding, load_formal_fit_registry, verify_checksum_manifest


def load_authorization(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError("formal training requires a sealed authorization bundle directory")
    return load_training_authorization_bundle(path)


def verify_authorized_inputs(
    authorization: dict[str, Any], registry_csv: Path, registry_summary: Path, s1_root: Path,
    s1_root_audit: Path, source_contract: Path, s1_protocol: Path, training_protocol: Path, feature_rebuilder: Path,
    *, normalization_root: Path | None = None, fold_root: Path | None = None,
    fold_id: int | None = None, variant: str | None = None, policy_intent_root: Path | None = None,
) -> dict[str, str]:
    """Re-audit every sealed input before and after episode loading."""

    rows = load_formal_fit_registry(registry_csv, registry_summary)
    verify_checksum_manifest(registry_csv.parent)
    verify_checksum_manifest(s1_root)
    if variant == "B3_25D" and policy_intent_root is not None:
        raise ValueError("B3_25D must not consume a 9D root")
    policy_root_sha256 = None
    if variant == "B3_25D9D":
        if policy_intent_root is None:
            raise ValueError("B3_25D9D requires a sealed 9D root")
        verify_checksum_manifest(policy_intent_root)
        policy_root_sha256 = sha256_file(policy_intent_root / "SHA256SUMS")
    if fold_root is not None:
        verify_checksum_manifest(fold_root)
    normalization = None
    normalization_source = None
    if normalization_root is not None:
        from gripper_attack.b3_training_protocol import load_normalization_bundle
        normalization, normalization_source = load_normalization_bundle(
            normalization_root, fold_id=fold_id, variant=variant,
            policy_intent_root_sha256=policy_root_sha256,
        )
    aggregate = s1_root / "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1.json"
    if not aggregate.is_file() or sha256_file(aggregate) != authorization["teacher_aggregate_sha256"]:
        raise ValueError("Teacher aggregate does not match authorization")
    expected_inputs = authorization["input_snapshots"]
    actual_inputs = {
        "formal_fit_registry_sha256": sha256_file(registry_csv),
        "formal_registry_summary_sha256": sha256_file(registry_summary),
        "formal_registry_root_sha256": sha256_file(registry_csv.parent / "SHA256SUMS"),
        "s1_corpus_sha256": sha256_file(s1_root / "SHA256SUMS"),
        "s1_root_audit_sha256": sha256_file(s1_root_audit),
        "teacher_aggregate_sha256": sha256_file(aggregate),
        "training_protocol_sha256": sha256_file(training_protocol),
        "source_contract_sha256": sha256_file(source_contract),
        "protocol_sha256": sha256_file(s1_protocol),
        "feature_rebuilder_sha256": sha256_file(feature_rebuilder),
        "normalization_bundle_sha256": sha256_file(normalization_root / "SHA256SUMS") if normalization_root is not None else expected_inputs["normalization_bundle_sha256"],
        "normalization_sha256": normalization.sha256 if normalization is not None else expected_inputs["normalization_sha256"],
        "fold_manifest_sha256": sha256_file(fold_root / "SHA256SUMS") if fold_root is not None else expected_inputs["fold_manifest_sha256"],
        "normalization_file_sha256": sha256_file(normalization_root / "normalization.json") if normalization_root is not None else authorization["normalization_file_sha256"],
        "policy_intent_root_sha256": policy_root_sha256,
    }
    for name in ("formal_fit_registry_sha256", "formal_registry_summary_sha256", "formal_registry_root_sha256", "s1_corpus_sha256", "s1_root_audit_sha256", "teacher_aggregate_sha256", "source_contract_sha256", "protocol_sha256", "feature_rebuilder_sha256", "normalization_file_sha256", "policy_intent_root_sha256"):
        if name == "policy_intent_root_sha256" and variant == "B3_25D":
            continue
        expected_value = expected_inputs.get(name, authorization.get(name))
        if actual_inputs[name] != expected_value:
            raise ValueError(f"authorized input SHA mismatch: {name}")
    if actual_inputs["training_protocol_sha256"] != expected_inputs["training_protocol_sha256"]:
        raise ValueError("training protocol SHA does not match authorization")
    if fold_root is not None and actual_inputs["fold_manifest_sha256"] != expected_inputs["fold_manifest_sha256"]:
        raise ValueError("fold manifest SHA does not match authorization")
    if normalization is not None:
        if actual_inputs["normalization_bundle_sha256"] != expected_inputs["normalization_bundle_sha256"] or actual_inputs["normalization_sha256"] != expected_inputs["normalization_sha256"]:
            raise ValueError("normalization bundle/content SHA does not match authorization")
        if normalization_source is None or normalization_source.get("normalization_file_sha256") != expected_inputs.get("normalization_file_sha256"):
            raise ValueError("normalization file provenance does not match authorization")
    if actual_inputs["policy_intent_root_sha256"] != authorization.get("policy_intent_root_sha256"):
        raise ValueError("9D policy-intent root does not match authorization")
    root_report = audit_materialized_root(
        s1_root, rows, require_runner_binding=True, feature_order_sha256=B3ModelConfig().feature_order_sha256,
        expected_runner_binding=authorization["runner_binding"],
        expected_input_sha256={
            "registry_csv_sha256": actual_inputs["formal_fit_registry_sha256"],
            "registry_summary_sha256": actual_inputs["formal_registry_summary_sha256"],
            "source_contract_sha256": sha256_file(source_contract),
            "protocol_sha256": sha256_file(s1_protocol),
            "feature_rebuilder_sha256": sha256_file(feature_rebuilder),
        },
    )
    audit_payload = json.loads(s1_root_audit.read_text(encoding="utf-8"))
    if root_report.get("status") != "PASS" or audit_payload.get("status") != "PASS":
        raise ValueError("independent S1 root audit is not PASS")
    before = {
        "registry": sha256_file(registry_csv), "registry_root": sha256_file(registry_csv.parent / "SHA256SUMS"),
        "s1": sha256_file(s1_root / "SHA256SUMS"), "aggregate": sha256_file(aggregate),
        "normalization_bundle": actual_inputs["normalization_bundle_sha256"],
        "normalization_file": actual_inputs["normalization_file_sha256"],
        "fold": actual_inputs["fold_manifest_sha256"], "policy_intent": actual_inputs["policy_intent_root_sha256"] or "",
    }
    # Caller loads episodes after this return and must call this function again
    # or compare this snapshot with a post-load snapshot.  The CLI does both.
    return before


def _normalize_batch(batch, normalization: B3Normalization):
    mean25 = torch.tensor(normalization.mean_25d, dtype=batch.x25.dtype, device=batch.x25.device)
    std25 = torch.tensor(normalization.std_25d, dtype=batch.x25.dtype, device=batch.x25.device)
    x25 = (batch.x25 - mean25) / std25
    x9 = None
    if batch.x9 is not None:
        mean9 = torch.tensor(normalization.mean_9d, dtype=batch.x9.dtype, device=batch.x9.device)
        std9 = torch.tensor(normalization.std_9d, dtype=batch.x9.dtype, device=batch.x9.device)
        x9 = (batch.x9 - mean9) / std9
    return x25, x9


def train_model(
    episodes: Sequence[B3Episode],
    *,
    variant: str,
    normalization: B3Normalization,
    epochs: int = 30,
    batch_size: int = 8,
    seed: int = 20260717,
    device: str = "cpu",
) -> tuple[torch.nn.Module, list[float]]:
    """Train one fixed candidate; model selection is deliberately external."""

    if not episodes or any(item.split != "FIT_TRAIN" for item in episodes):
        raise ValueError("trainer accepts FIT_TRAIN episodes only")
    random.seed(seed)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    model = build_b3_model(B3ModelConfig(variant=variant)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses: list[float] = []
    sampler = B3EpisodeSampler(episodes, seed=seed)
    for _ in range(epochs):
        indices = sampler.ordered_indices(shuffle=True)
        epoch_terms: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch = pad_episode_batch([episodes[index] for index in indices[start:start + batch_size]])
            batch = batch.__class__(
                batch.x25.to(target_device),
                None if batch.x9 is None else batch.x9.to(target_device),
                {name: value.to(target_device) for name, value in batch.targets.items()},
                {name: value.to(target_device) for name, value in batch.known_masks.items()},
                batch.episode_valid_mask.to(target_device), batch.padding_mask.to(target_device), batch.episodes,
            )
            x25, x9 = _normalize_batch(batch, normalization)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x25, x9, mask=batch.padding_mask)
            loss = compute_b3_loss(logits, batch.targets, batch.known_masks, episode_valid_mask=batch.episode_valid_mask, padding_mask=batch.padding_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_terms.append(float(loss.detach()))
        losses.append(sum(epoch_terms) / len(epoch_terms))
    return model, losses


def load_normalization(path: Path) -> B3Normalization:
    if path.is_dir():
        normalization, _ = load_normalization_bundle(path)
        return normalization
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_NORMALIZATION_V1":
        raise ValueError("unexpected normalization schema")
    return B3Normalization.from_dict(value["normalization"])


def load_formal_fit_episodes(registry_csv: Path, s1_root: Path, *, include_9d_root: Path | None = None) -> list[B3Episode]:
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    episodes: list[B3Episode] = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None
        if include_9d_root is not None:
            nine_d = include_9d_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    if len(episodes) != 800:
        raise ValueError("formal FIT input is not exactly 800 episodes")
    return episodes


def load_fit_fold_episodes(
    registry_csv: Path, s1_root: Path, fold_root: Path, *, fold_id: int, partition: str,
    include_9d_root: Path | None = None,
) -> list[B3Episode]:
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    all_episodes: list[B3Episode] = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None
        if include_9d_root is not None:
            nine_d = include_9d_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        all_episodes.append(load_episode(root, row, include_9d_root=nine_d))
    return select_fit_fold_episodes(all_episodes, load_fit_fold_bundle(fold_root), fold_id=fold_id, partition=partition)


def main() -> int:
    parser = argparse.ArgumentParser(description="Official V3 formal trainer; real execution is authorization-gated")
    parser.add_argument("--authorization", type=Path, required=True, help="machine-built authorization bundle root or authorization.json")
    parser.add_argument("--registry-csv", type=Path)
    parser.add_argument("--registry-summary", type=Path)
    parser.add_argument("--s1-root", type=Path)
    parser.add_argument("--s1-root-audit", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--s1-protocol", type=Path)
    parser.add_argument("--training-protocol", type=Path)
    parser.add_argument("--feature-rebuilder", type=Path)
    parser.add_argument("--normalization", type=Path, help="sealed normalization bundle root")
    parser.add_argument("--fold-root", type=Path)
    parser.add_argument("--fold-id", type=int)
    parser.add_argument("--seed", type=int, choices=(20260717, 20260718, 20260719))
    parser.add_argument("--output-checkpoint", type=Path, help="legacy smoke file; formal runs require --output-checkpoint-bundle")
    parser.add_argument("--output-checkpoint-bundle", type=Path)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--runner-repo", type=Path)
    parser.add_argument("--runner-config", type=Path)
    parser.add_argument("--runner-script", type=Path)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--execute-formal", action="store_true", help="explicitly opt into a separately authorized real run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    args = parser.parse_args()
    authorization = load_authorization(args.authorization)
    if args.variant == "B3_25D" and args.policy_intent_root is not None:
        raise SystemExit("B3_25D must not receive --policy-intent-root")
    if args.variant == "B3_25D9D" and args.policy_intent_root is None:
        raise SystemExit("B3_25D9D requires --policy-intent-root")
    if authorization.get("fit_scope") != "FIT_FOLD":
        raise SystemExit("fold trainer requires fit_scope=FIT_FOLD authorization")
    if authorization.get("variant") != args.variant:
        raise SystemExit("authorization variant does not match trainer variant")
    if args.fold_id is not None and authorization.get("fold_id") != args.fold_id:
        raise SystemExit("authorization fold_id does not match trainer fold_id")
    if args.seed is not None and authorization.get("seed") != args.seed:
        raise SystemExit("authorization seed does not match trainer seed")
    if not args.execute_formal:
        raise SystemExit("FORMAL_TRAINING_HOLD: pass --execute-formal only after S1, audit, and authorization gates")
    for value, name in ((args.registry_csv, "--registry-csv"), (args.s1_root, "--s1-root"), (args.normalization, "--normalization")):
        if value is None:
            raise SystemExit(f"{name} is required for an explicitly authorized run")
    if args.runner_repo is None:
        raise SystemExit("--runner-repo is required for an explicitly authorized run")
    if args.runner_config is None or args.runner_script is None:
        raise SystemExit("--runner-config and --runner-script are required for an explicitly authorized run")
    required_paths = ((args.registry_summary, "--registry-summary"), (args.s1_root_audit, "--s1-root-audit"), (args.source_contract, "--source-contract"), (args.s1_protocol, "--s1-protocol"), (args.training_protocol, "--training-protocol"), (args.feature_rebuilder, "--feature-rebuilder"), (args.fold_root, "--fold-root"), (args.fold_id, "--fold-id"), (args.seed, "--seed"), (args.output_checkpoint_bundle, "--output-checkpoint-bundle"))
    for value, name in required_paths:
        if value is None:
            raise SystemExit(f"{name} is required for formal fold training")
    if args.output_checkpoint is not None:
        raise SystemExit("formal fold training does not write an unsealed checkpoint file")
    if args.output_checkpoint_bundle.exists():
        raise SystemExit(f"refusing to overwrite checkpoint bundle: {args.output_checkpoint_bundle}")
    before = verify_authorized_inputs(
        authorization, args.registry_csv, args.registry_summary, args.s1_root, args.s1_root_audit,
        args.source_contract, args.s1_protocol, args.training_protocol, args.feature_rebuilder,
        normalization_root=args.normalization, fold_root=args.fold_root, fold_id=args.fold_id,
        variant=args.variant, policy_intent_root=args.policy_intent_root,
    )
    measured_binding = build_s1_runner_binding(
        runner_repo=args.runner_repo, expected_runner_head=authorization["runner_head"],
        config_path=args.runner_config, runner_script_path=args.runner_script,
    )
    if measured_binding != authorization["runner_binding"]:
        raise SystemExit("measured runner binding does not match training authorization")
    episodes = load_fit_fold_episodes(args.registry_csv, args.s1_root, args.fold_root, fold_id=args.fold_id, partition="train", include_9d_root=args.policy_intent_root)
    if args.variant == "B3_25D9D" and any(item.features_9d is None for item in episodes):
        raise SystemExit("B3_25D9D requires a complete independent 9D ablation root")
    policy_root_sha256 = None if args.policy_intent_root is None else sha256_file(args.policy_intent_root / "SHA256SUMS")
    normalization, norm_source = load_normalization_bundle(
        args.normalization, fold_id=args.fold_id, variant=args.variant,
        policy_intent_root_sha256=policy_root_sha256,
    )
    if sha256_file(args.normalization / "SHA256SUMS") != authorization["normalization_bundle_sha256"]:
        raise SystemExit("normalization bundle SHA does not match authorization")
    if normalization.sha256 != authorization["normalization_sha256"]:
        raise SystemExit("normalization content SHA does not match authorization")
    if sha256_file(args.fold_root / "SHA256SUMS") != authorization["fold_manifest_sha256"]:
        raise SystemExit("fold manifest SHA does not match authorization")
    if norm_source.get("train_identity_sha256") != load_fit_fold_bundle(args.fold_root)["folds"][args.fold_id]["train_identity_sha256"]:
        raise SystemExit("normalization bundle is not bound to this fold's train identities")
    recomputed_normalization = compute_fit_normalization(episodes, include_9d=args.variant == "B3_25D9D")
    if recomputed_normalization.sha256 != normalization.sha256:
        raise SystemExit("normalization bundle does not match the measured 600-episode training fold")
    model, losses = train_model(episodes, variant=args.variant, normalization=normalization, seed=args.seed, device=args.device)
    save_b3_checkpoint_bundle(
        args.output_checkpoint_bundle, model, normalization, authorization=authorization,
        checkpoint_status="FIT_FOLD_TRAINED_CANDIDATE",
        extra={"variant": args.variant, "fit_scope": "FIT_FOLD", "fold_id": args.fold_id, "seed": args.seed, "device": args.device, "dtype": args.dtype, "epochs": len(losses), "loss_history": losses, "final_loss": losses[-1], "fit_episode_count": len(episodes)},
    )
    after = verify_authorized_inputs(
        authorization, args.registry_csv, args.registry_summary, args.s1_root, args.s1_root_audit,
        args.source_contract, args.s1_protocol, args.training_protocol, args.feature_rebuilder,
        normalization_root=args.normalization, fold_root=args.fold_root, fold_id=args.fold_id,
        variant=args.variant, policy_intent_root=args.policy_intent_root,
    )
    if before != after:
        raise SystemExit("TOCTOU detected: sealed training inputs changed during fold training")
    print(json.dumps({"status": "FIT_FOLD_TRAINED_CANDIDATE", "variant": args.variant, "fold_id": args.fold_id, "seed": args.seed, "fit_episode_count": len(episodes), "final_loss": losses[-1], "eligible_for_model_selection": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
