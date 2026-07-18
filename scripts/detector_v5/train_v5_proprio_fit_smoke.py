#!/usr/bin/env python3
"""Explicit FIT-only V5-A/V5-B development smoke.

This is not the formal trainer.  It requires an explicit development flag and
always writes a non-formal checkpoint bundle with attack authorization false.
It never reads a non-FIT identity.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from gripper_attack.v5_dataset import (
    V5Episode,
    causal_window_anchor_scores,
    classify_v5_episode_windows,
    compute_v5_normalization,
    compute_v5_intent_normalization,
    load_fit_registry,
    load_policy_intent_root,
    load_v5_episodes,
)
from gripper_attack.v5_protocol import (
    V5ModelContract,
    V5_FEATURES_9D,
    V5_PHYSICS_CANDIDATE_ALIASES,
    canonical_variant,
    json_sha,
    variant_uses_intent,
)
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker, V5LossConfigV2, compute_v5_loss_v2
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, seal_directory, sha256_file, verify_sealed_directory


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"object expected: {path}")
    return value


def _teacher_kind(root: Path) -> str:
    physics_v21 = list(root.glob("labels/*/task_*/state_*/physics_teacher_v21.jsonl"))
    physics = list(root.glob("labels/*/task_*/state_*/physics_teacher_v2.jsonl"))
    legacy = list(root.glob("labels/*/task_*/state_*/v5_teacher_utility.jsonl"))
    if physics_v21 and (physics or legacy):
        raise ValueError("Teacher root mixes Physics V2.1 with another label stream")
    if physics and legacy:
        raise ValueError("Teacher root mixes Physics and legacy V5 files")
    if physics_v21:
        return "physics_v21"
    if physics:
        return "physics_v2"
    if legacy:
        return "utility_v3"
    raise ValueError("Teacher root contains no supported V5 label stream")


def _validate_teacher_audit(teacher_audit: dict[str, Any], teacher_root: Path, kind: str) -> dict[str, Any]:
    if teacher_audit.get("status") != "PASS":
        raise ValueError("V5 Teacher audit is not PASS")
    if teacher_audit.get("formal_training_authorized") is not False or teacher_audit.get("formal_attack_authorized") is not False:
        raise ValueError("V5 Teacher audit is not a clean-only audit")
    binding = {"teacher_kind": kind, "teacher_root_sha256s_sha256": sha256_file(teacher_root / "SHA256SUMS")}
    if kind in {"physics_v2", "physics_v21"}:
        expected = {
            "DETECTOR_V5_PHYSICS_TEACHER_V2_INDEPENDENT_AUDIT_V1",
            "DETECTOR_V5_PHYSICS_TEACHER_V21_INDEPENDENT_AUDIT_V1",
        }
        if teacher_audit.get("schema") not in expected:
            raise ValueError("Physics Teacher audit schema is not independent")
        if teacher_audit.get("teacher_root_sha256sums_sha256") != binding["teacher_root_sha256s_sha256"]:
            raise ValueError("Physics Teacher audit is not bound to the supplied root")
    return binding


def _episode_loss(
    model: CausalMultimodalVulnerabilityRanker,
    episode: V5Episode,
    mean: torch.Tensor,
    std: torch.Tensor,
    intent_mean: torch.Tensor | None = None,
    intent_std: torch.Tensor | None = None,
    use_causal_targets: bool = False,
    loss_config: V5LossConfigV2 | None = None,
) -> dict[str, torch.Tensor]:
    x = ((episode.features_25d.to(mean.device) - mean) / std).unsqueeze(0)
    valid = episode.valid_mask.to(mean.device).unsqueeze(0)
    intent = None
    if model.intent_cell is not None:
        if intent_mean is None or intent_std is None:
            raise ValueError("V5-B requires train-only policy-intent normalization")
        intent = ((episode.policy_intent_9d.to(mean.device) - intent_mean) / intent_std).unsqueeze(0)
    output = model.forward_sequence(x, intent=intent, valid_mask=valid)
    return compute_v5_loss_v2(
        output["utility_logit"][0],
        output["release_logit"][0],
        output["regrasp_logit"][0],
        episode,
        config=loss_config or V5LossConfigV2(use_causal_targets=use_causal_targets),
    )


@torch.no_grad()
def _diagnostic(
    model: CausalMultimodalVulnerabilityRanker,
    episodes: list[V5Episode],
    mean: torch.Tensor,
    std: torch.Tensor,
    intent_mean: torch.Tensor | None = None,
    intent_std: torch.Tensor | None = None,
) -> dict[str, Any]:
    true_mixed_total = 0
    top1_hit = 0
    pure_negative_total = 0
    pure_negative_abstain = 0
    for episode in episodes:
        x = ((episode.features_25d.to(mean.device) - mean) / std).unsqueeze(0)
        intent = None
        if model.intent_cell is not None:
            if intent_mean is None or intent_std is None:
                raise ValueError("V5-B requires train-only policy-intent normalization")
            intent = ((episode.policy_intent_9d.to(mean.device) - intent_mean) / intent_std).unsqueeze(0)
        output = model.forward_sequence(x, intent=intent, valid_mask=episode.valid_mask.to(mean.device).unsqueeze(0))
        scores, windows = causal_window_anchor_scores(output["utility_logit"][0], episode)
        if not len(windows):
            continue
        values = [int(row["causal_utility_tier"] if row.get("causal_utility_tier") is not None else row["utility_tier"]) if row["utility_tier"] is not None else -1 for row in windows]
        best_index = int(torch.argmax(scores).item())
        has_positive = any(value >= 2 for value in values)
        has_negative = any(value <= 1 for value in values)
        category = "TRUE_MIXED" if has_positive and has_negative else "POSITIVE_ONLY" if has_positive else "PURE_NEGATIVE" if has_negative else "NO_CANDIDATE"
        positive = [value for value in values if value >= 2]
        if category == "TRUE_MIXED":
            true_mixed_total += 1
            if values[best_index] == max(positive):
                top1_hit += 1
        elif category == "PURE_NEGATIVE":
            pure_negative_total += 1
            if float(torch.sigmoid(scores).max()) < 0.5:
                pure_negative_abstain += 1
    return {
        "true_mixed_episode_count": true_mixed_total,
        "top1_highest_utility_hit_count": top1_hit,
        "top1_highest_utility_hit_rate": (top1_hit / true_mixed_total) if true_mixed_total else None,
        "pure_negative_episode_count": pure_negative_total,
        "pure_negative_abstain_count_at_0.5": pure_negative_abstain,
        "pure_negative_abstain_rate_at_0.5": (pure_negative_abstain / pure_negative_total) if pure_negative_total else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute_v5_fit_development:
        raise ValueError("pass --execute-v5-fit-development to run the non-formal smoke")
    supported = {"V5_A_PROPRIO", "V5_B_PROPRIO_POLICY_INTENT", *V5_PHYSICS_CANDIDATE_ALIASES}
    if args.candidate not in supported:
        raise ValueError("V5 development smoke supports V5-A/V5-B and their Physics Teacher aliases")
    variant = canonical_variant(args.candidate)
    physics_candidate = args.candidate in V5_PHYSICS_CANDIDATE_ALIASES
    if variant == "V5_B_PROPRIO_POLICY_INTENT" and args.policy_intent_root is None:
        raise ValueError("V5-B requires --policy-intent-root")
    if variant == "V5_B_PROPRIO_POLICY_INTENT" and args.development_protocol is None:
        raise ValueError("V5-B requires --development-protocol")
    if variant == "V5_B_PROPRIO_POLICY_INTENT" and (args.loss_protocol is None or args.decision_config is None):
        raise ValueError("V5-B requires --loss-protocol and --decision-config")
    if variant == "V5_A_PROPRIO" and args.policy_intent_root is not None:
        raise ValueError("V5-A must not consume a policy-intent root")
    if physics_candidate and (args.development_protocol is None or args.loss_protocol is None or args.decision_config is None):
        raise ValueError("Physics development requires development, loss, and decision protocols")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    teacher_kind = _teacher_kind(args.teacher_root.resolve())
    if physics_candidate and teacher_kind not in {"physics_v2", "physics_v21"}:
        raise ValueError("Physics candidate requires a Physics Teacher root")
    if not physics_candidate and teacher_kind in {"physics_v2", "physics_v21"}:
        raise ValueError("Physics Teacher V2 root requires an explicit Physics candidate alias")
    teacher_audit = _read_json(args.teacher_audit.resolve())
    teacher_binding = _validate_teacher_audit(teacher_audit, args.teacher_root.resolve(), teacher_kind)
    verify_sealed_directory(args.s1_root.resolve())
    verify_sealed_directory(args.teacher_root.resolve())
    registry_rows = load_fit_registry(args.registry_csv.resolve())
    fold = load_fit_fold_bundle(args.fold_root.resolve())
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == args.fold_id)
    by_key = {row["canonical_parent_key"]: row for row in registry_rows}
    train_keys = list(fold_row["train_identities"])
    valid_keys = list(fold_row["validation_identities"])
    if args.train_identity_file is not None:
        selected = json.loads(args.train_identity_file.resolve().read_text(encoding="utf-8"))
        if isinstance(selected, dict):
            selected = selected.get("identities")
        if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
            raise ValueError("train identity file must contain a JSON string list")
        if not set(selected).issubset(set(train_keys)):
            raise ValueError("stratified train identities are not a subset of fold train identities")
        train_keys = list(selected)
    if args.max_train_episodes is not None:
        train_keys = train_keys[: args.max_train_episodes]
    policy_index = None
    policy_meta: dict[str, Any] | None = None
    if args.policy_intent_root is not None:
        policy_index, policy_meta = load_policy_intent_root(args.policy_intent_root)
    protocol_sha256 = None
    if args.development_protocol is not None:
        protocol_sha256 = sha256_file(args.development_protocol.resolve())
    loss_protocol_sha256 = None if args.loss_protocol is None else sha256_file(args.loss_protocol.resolve())
    decision_config_sha256 = None if args.decision_config is None else sha256_file(args.decision_config.resolve())
    loss_protocol = _read_json(args.loss_protocol.resolve()) if args.loss_protocol is not None else {}
    components = loss_protocol.get("components", {})
    loss_config = V5LossConfigV2(
        pairwise_margin=float(components.get("pairwise_margin", 0.2)),
        ordinal_weight=float(components.get("ordinal_weight", 1.0)),
        pairwise_weight=float(components.get("pairwise_weight", 0.5)),
        listwise_weight=float(components.get("listwise_weight", 0.5)),
        abstention_weight=float(components.get("pure_negative_max_weight", 1.0)),
        release_weight=float(components.get("release_weight", 0.3)),
        regrasp_weight=float(components.get("regrasp_weight", 0.3)),
        use_causal_targets=physics_candidate,
    )
    train = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in train_keys], policy_index=policy_index)
    valid = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in valid_keys], policy_index=policy_index)
    if not train or len(valid) != 200:
        raise ValueError("V5 smoke requires non-empty train and exact 200 validation identities")
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but no CUDA device is available")
    mean, std = compute_v5_normalization(train)
    mean, std = mean.to(device), std.to(device)
    if variant_uses_intent(variant):
        assert policy_index is not None
        intent_mean, intent_std = compute_v5_intent_normalization(train)
        intent_mean, intent_std = intent_mean.to(device), intent_std.to(device)
    else:
        intent_mean = torch.zeros(9, device=device)
        intent_std = torch.ones(9, device=device)
    contract = V5ModelContract(variant)
    model = CausalMultimodalVulnerabilityRanker(contract).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_categories = {episode.canonical_parent_key: classify_v5_episode_windows(episode.windows, causal=physics_candidate) for episode in train}
    category_counts = dict(Counter(train_categories.values()))
    positive_count = sum(value for key, value in category_counts.items() if key in {"TRUE_MIXED", "POSITIVE_ONLY"})
    pure_count = category_counts.get("PURE_NEGATIVE", 0)
    category_weights = {
        key: (0.5 / positive_count if value in {"TRUE_MIXED", "POSITIVE_ONLY"} and positive_count else 0.0 if value == "NO_CANDIDATE" else 0.5 / pure_count if pure_count else 0.0)
        for key, value in train_categories.items()
    }
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(args.epochs):
        losses: list[float] = []
        component_values: dict[str, list[float]] = {}
        for episode in train:
            optimizer.zero_grad(set_to_none=True)
            parts = _episode_loss(model, episode, mean, std, intent_mean, intent_std, physics_candidate, loss_config)
            loss = parts["total"] * category_weights[episode.canonical_parent_key]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V5 smoke produced NaN/Inf loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in parts.items():
                component_values.setdefault(name, []).append(float(value.detach().cpu()))
        history.append({
            "epoch": epoch + 1,
            "train_loss_weighted_mean": sum(losses) / len(losses),
            "component_means": {name: sum(values) / len(values) for name, values in component_values.items()},
            "category_counts": category_counts,
        })
    model.eval()
    diagnostic = _diagnostic(model, valid, mean, std, intent_mean, intent_std)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        torch.save({
            "schema": "DETECTOR_V5_DEVELOPMENT_CHECKPOINT_V1",
            "status": "V5_FIT_DEVELOPMENT_SMOKE",
            "candidate": args.candidate,
            "model_variant": variant,
            "fold_id": args.fold_id,
            "seed": args.seed,
            "model_contract": contract.to_dict(),
            "normalization_mean_25d": mean.detach().cpu(),
            "normalization_std_25d": std.detach().cpu(),
            "normalization_mean_9d": intent_mean.detach().cpu(),
            "normalization_std_9d": intent_std.detach().cpu(),
            "model_state": model.state_dict(),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
            "eligible_for_model_selection": False,
        }, staging / "checkpoint.pt")
        manifest = {
            "schema": "DETECTOR_V5_DEVELOPMENT_CHECKPOINT_BUNDLE_V1",
            "status": "V5_FIT_DEVELOPMENT_SMOKE",
            "candidate": args.candidate,
            "fold_id": args.fold_id,
            "seed": args.seed,
            "train_identity_count": len(train),
            "validation_identity_count": len(valid),
            "checkpoint_sha256": sha256_file(staging / "checkpoint.pt"),
            "teacher_audit_sha256": sha256_file(args.teacher_audit.resolve()),
            "teacher_kind": teacher_binding["teacher_kind"],
            "teacher_root_sha256s_sha256": teacher_binding["teacher_root_sha256s_sha256"],
            "teacher_manifest_sha256": sha256_file(args.teacher_root.resolve() / "physics_teacher_v2_manifest.json") if teacher_kind == "physics_v2" else None,
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "s1_root_sha256s_sha256": sha256_file(args.s1_root.resolve() / "SHA256SUMS"),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root.resolve() / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root.resolve() / "SHA256SUMS"),
            "train_identity_sha256": json_sha(train_keys),
            "validation_identity_sha256": json_sha(valid_keys),
            "policy_intent_consumed": variant_uses_intent(variant),
            "loss_target": "causal_utility_tier_at_decision_anchor" if physics_candidate else "final_utility_tier",
            "train_category_counts": category_counts,
            "policy_intent_root_sha256s_sha256": None if policy_meta is None else policy_meta["policy_root_sha256s_sha256"],
            "policy_intent_manifest_sha256": None if policy_meta is None else policy_meta["policy_manifest_sha256"],
            "policy_intent_feature_order_sha256": None if policy_meta is None else policy_meta["policy_feature_order_sha256"],
            "policy_intent_source_artifact_index_sha256": None if policy_meta is None else policy_meta["policy_source_artifact_index_sha256"],
            "policy_intent_features": list(V5_FEATURES_9D) if policy_meta is not None else None,
            "development_protocol_sha256": protocol_sha256,
            "loss_protocol_sha256": loss_protocol_sha256,
            "decision_config_sha256": decision_config_sha256,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
            "eligible_for_model_selection": False,
            "device": str(device),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "loss_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "diagnostic_metrics.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        verify_sealed_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"manifest": manifest, "diagnostic": diagnostic, "history": history, "output_root": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-v5-fit-development", action="store_true")
    parser.add_argument("--candidate", default="V5_A_PROPRIO")
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--development-protocol", type=Path)
    parser.add_argument("--loss-protocol", type=Path)
    parser.add_argument("--decision-config", type=Path)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-audit", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--train-identity-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
