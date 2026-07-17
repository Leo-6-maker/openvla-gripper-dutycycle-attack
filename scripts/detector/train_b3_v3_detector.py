#!/usr/bin/env python3
"""Formal Trainer entrypoint with an explicit sealed authorization gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from gripper_attack.b3_formal import B3ModelConfig, B3Normalization, build_b3_model, compute_b3_loss, save_b3_checkpoint, validate_training_authorization
from gripper_attack.b3_v3_dataset import B3Episode, B3EpisodeSampler, load_episode, load_formal_registry_csv, pad_episode_batch


def load_authorization(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training authorization must be an object")
    validate_training_authorization(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_authorized_inputs(authorization: dict[str, Any], registry_csv: Path, s1_root: Path) -> None:
    if sha256_file(registry_csv) != authorization["formal_fit_registry_sha256"]:
        raise ValueError("formal registry SHA does not match authorization")
    checksum_manifest = s1_root / "SHA256SUMS"
    aggregate = s1_root / "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1.json"
    if not checksum_manifest.is_file() or sha256_file(checksum_manifest) != authorization["s1_corpus_sha256"]:
        raise ValueError("S1 checksum manifest does not match authorization")
    if not aggregate.is_file() or sha256_file(aggregate) != authorization["teacher_aggregate_sha256"]:
        raise ValueError("Teacher aggregate does not match authorization")


def _normalize_batch(batch, normalization: B3Normalization):
    mean25 = torch.tensor(normalization.mean_25d, dtype=batch.x25.dtype)
    std25 = torch.tensor(normalization.std_25d, dtype=batch.x25.dtype)
    x25 = (batch.x25 - mean25) / std25
    x9 = None
    if batch.x9 is not None:
        mean9 = torch.tensor(normalization.mean_9d, dtype=batch.x9.dtype)
        std9 = torch.tensor(normalization.std_9d, dtype=batch.x9.dtype)
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
) -> tuple[torch.nn.Module, list[float]]:
    """Train one fixed candidate; model selection is deliberately external."""

    if not episodes or any(item.split != "FIT_TRAIN" for item in episodes):
        raise ValueError("trainer accepts FIT_TRAIN episodes only")
    random.seed(seed)
    torch.manual_seed(seed)
    model = build_b3_model(B3ModelConfig(variant=variant))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses: list[float] = []
    sampler = B3EpisodeSampler(episodes, seed=seed)
    for _ in range(epochs):
        indices = sampler.ordered_indices(shuffle=True)
        epoch_terms: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch = pad_episode_batch([episodes[index] for index in indices[start:start + batch_size]])
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Official V3 formal trainer; real execution is authorization-gated")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path)
    parser.add_argument("--s1-root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--runner-repo", type=Path)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--execute-formal", action="store_true", help="explicitly opt into a separately authorized real run")
    args = parser.parse_args()
    authorization = load_authorization(args.authorization)
    if not args.execute_formal:
        raise SystemExit("FORMAL_TRAINING_HOLD: pass --execute-formal only after S1, audit, and authorization gates")
    for value, name in ((args.registry_csv, "--registry-csv"), (args.s1_root, "--s1-root"), (args.normalization, "--normalization"), (args.output_checkpoint, "--output-checkpoint")):
        if value is None:
            raise SystemExit(f"{name} is required for an explicitly authorized run")
    if args.runner_repo is None:
        raise SystemExit("--runner-repo is required for an explicitly authorized run")
    if args.output_checkpoint.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {args.output_checkpoint}")
    verify_authorized_inputs(authorization, args.registry_csv, args.s1_root)
    actual_head = subprocess.check_output(["git", "-C", str(args.runner_repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_head != authorization["runner_head"]:
        raise SystemExit("runner Git HEAD does not match training authorization")
    dirty = subprocess.check_output(["git", "-C", str(args.runner_repo), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty.strip():
        raise SystemExit("runner worktree is dirty; refusing formal training")
    episodes = load_formal_fit_episodes(args.registry_csv, args.s1_root, include_9d_root=args.policy_intent_root)
    if args.variant == "B3_25D9D" and any(item.features_9d is None for item in episodes):
        raise SystemExit("B3_25D9D requires a complete independent 9D ablation root")
    normalization = load_normalization(args.normalization)
    model, losses = train_model(episodes, variant=args.variant, normalization=normalization, seed=20260717)
    save_b3_checkpoint(
        args.output_checkpoint, model, normalization, authorization=authorization, training_complete=True,
        extra={"variant": args.variant, "epochs": len(losses), "final_loss": losses[-1], "fit_episode_count": len(episodes)},
    )
    print(json.dumps({"status": "FORMAL_TRAINING_COMPLETED", "variant": args.variant, "fit_episode_count": len(episodes), "final_loss": losses[-1]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
