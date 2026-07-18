#!/usr/bin/env python3
"""Explicit FIT-only V5-A development smoke.

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
from pathlib import Path
from typing import Any

import torch
from gripper_attack.v5_dataset import (
    V5Episode,
    causal_window_anchor_scores,
    classify_v5_episode_windows,
    compute_v5_normalization,
    load_fit_registry,
    load_v5_episodes,
)
from gripper_attack.v5_protocol import V5ModelContract
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker, V5LossConfigV2, compute_v5_loss_v2
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, seal_directory, sha256_file


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


def _episode_loss(model: CausalMultimodalVulnerabilityRanker, episode: V5Episode, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = ((episode.features_25d.to(mean.device) - mean) / std).unsqueeze(0)
    valid = episode.valid_mask.to(mean.device).unsqueeze(0)
    output = model.forward_sequence(x, valid_mask=valid)
    return compute_v5_loss_v2(
        output["utility_logit"][0],
        output["release_logit"][0],
        output["regrasp_logit"][0],
        episode,
        config=V5LossConfigV2(),
    )["total"]


@torch.no_grad()
def _diagnostic(model: CausalMultimodalVulnerabilityRanker, episodes: list[V5Episode], mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    true_mixed_total = 0
    top1_hit = 0
    pure_negative_total = 0
    pure_negative_abstain = 0
    for episode in episodes:
        x = ((episode.features_25d.to(mean.device) - mean) / std).unsqueeze(0)
        output = model.forward_sequence(x, valid_mask=episode.valid_mask.to(mean.device).unsqueeze(0))
        scores, windows = causal_window_anchor_scores(output["utility_logit"][0], episode)
        if not len(windows):
            continue
        values = [int(row["utility_tier"]) if row["utility_tier"] is not None else -1 for row in windows]
        best_index = int(torch.argmax(scores).item())
        category = classify_v5_episode_windows(episode.windows)
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
    if args.candidate != "V5_A_PROPRIO":
        raise ValueError("the first V5 smoke only supports V5_A_PROPRIO")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    teacher_audit = _read_json(args.teacher_audit.resolve())
    if teacher_audit.get("status") != "PASS" or teacher_audit.get("formal_training_authorized") is not False or teacher_audit.get("formal_attack_authorized") is not False:
        raise ValueError("V5 Teacher audit is not a safe clean-only PASS")
    registry_rows = load_fit_registry(args.registry_csv.resolve())
    fold = load_fit_fold_bundle(args.fold_root.resolve())
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == args.fold_id)
    by_key = {row["canonical_parent_key"]: row for row in registry_rows}
    train_keys = list(fold_row["train_identities"])
    valid_keys = list(fold_row["validation_identities"])
    if args.train_identity_file is not None:
        selected = json.loads(args.train_identity_file.resolve().read_text(encoding="utf-8"))
        if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
            raise ValueError("train identity file must contain a JSON string list")
        if not set(selected).issubset(set(train_keys)):
            raise ValueError("stratified train identities are not a subset of fold train identities")
        train_keys = list(selected)
    if args.max_train_episodes is not None:
        train_keys = train_keys[: args.max_train_episodes]
    train = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in train_keys])
    valid = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), [by_key[key] for key in valid_keys])
    if not train or len(valid) != 200:
        raise ValueError("V5 smoke requires non-empty train and exact 200 validation identities")
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but no CUDA device is available")
    mean, std = compute_v5_normalization(train)
    mean, std = mean.to(device), std.to(device)
    model = CausalMultimodalVulnerabilityRanker(V5ModelContract("V5_A_PROPRIO")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(args.epochs):
        losses: list[float] = []
        for episode in train:
            optimizer.zero_grad(set_to_none=True)
            loss = _episode_loss(model, episode, mean, std)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V5 smoke produced NaN/Inf loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss_mean": sum(losses) / len(losses)})
    model.eval()
    diagnostic = _diagnostic(model, valid, mean, std)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        torch.save({
            "schema": "DETECTOR_V5_DEVELOPMENT_CHECKPOINT_V1",
            "status": "V5_FIT_DEVELOPMENT_SMOKE",
            "candidate": args.candidate,
            "fold_id": args.fold_id,
            "seed": args.seed,
            "model_contract": V5ModelContract("V5_A_PROPRIO").to_dict(),
            "normalization_mean_25d": mean.detach().cpu(),
            "normalization_std_25d": std.detach().cpu(),
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
            "teacher_audit_sha256": sha256_file(args.teacher_audit.resolve()),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
            "eligible_for_model_selection": False,
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "loss_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "diagnostic_metrics.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"manifest": manifest, "diagnostic": diagnostic, "history": history, "output_root": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-v5-fit-development", action="store_true")
    parser.add_argument("--candidate", default="V5_A_PROPRIO")
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
