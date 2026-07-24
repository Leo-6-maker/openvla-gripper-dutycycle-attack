#!/usr/bin/env python3
"""Formal V4 Quality Detector trainer — B3 shell adapted for single quality head.

Reuses B3 patterns: authorization, fold selection, masked state transitions,
AdamW/FP32/clip, candidate-only checkpoint.

Requires: --s1-root, --v21-root, --fold-id, --view, --seed, --output
"""

from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path

import torch

from gripper_attack.v4_formal import (
    V4StatefulQualityGRU, V4Normalization, V4_CHECKPOINT_SCHEMA,
    V4_CHECKPOINT_STATUS, compute_v4_loss, save_v4_checkpoint_bundle,
    VIEW_FEATURE_COUNTS, sha256_file, json_sha,
)
from gripper_attack.v4_dataset import (
    V4Episode, V4Batch, V4EpisodeSampler,
    load_v4_episode, pad_v4_episode_batch,
    compute_v4_fold_normalization, select_fold_episodes,
    SUITES, FIT_STATES,
)


def load_all_fit_episodes(s1_root: Path, v21_root: Path, view: str
                          ) -> list[V4Episode]:
    episodes = []
    for suite in SUITES:
        for task in range(10):
            for state in sorted(FIT_STATES):
                ep = load_v4_episode(s1_root, v21_root, suite, task, state, view)
                if ep is not None:
                    episodes.append(ep)
    if len(episodes) != 800:
        raise ValueError(f"expected 800 FIT episodes, got {len(episodes)}")
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--v21-root", type=Path, required=True)
    ap.add_argument("--fold-id", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--view", choices=["A", "B", "C"], default="B")
    ap.add_argument("--aux-release", action="store_true")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--ranking-weight", type=float, default=0.5)
    ap.add_argument("--release-weight", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"=== V4 Formal Trainer: fold={args.fold_id} view={args.view} "
          f"aux_release={args.aux_release} seed={args.seed} ===")

    # Load all FIT episodes
    all_eps = load_all_fit_episodes(args.s1_root, args.v21_root, args.view)
    train_eps = select_fold_episodes(all_eps, args.fold_id, "train")
    print(f"Train episodes: {len(train_eps)} (expected 600)")

    if len(train_eps) != 600:
        raise ValueError(f"expected 600 train episodes, got {len(train_eps)}")

    # Compute normalization from training fold only
    norm = compute_v4_fold_normalization(train_eps, args.view)
    print(f"Normalization: {norm.feature_count}D  mean_range=[{min(norm.mean):.3f},{max(norm.mean):.3f}]")

    # Build model
    device = torch.device(args.device)
    model = V4StatefulQualityGRU(
        input_dim=VIEW_FEATURE_COUNTS[args.view],
        aux_release=args.aux_release,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    sampler = V4EpisodeSampler(train_eps, base_seed=args.seed)

    epoch_losses = []
    for epoch in range(args.epochs):
        indices = sampler.ordered_indices(epoch, shuffle=True)
        epoch_terms = []

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            batch_eps = [train_eps[i] for i in batch_idx]
            batch = pad_v4_episode_batch(batch_eps)

            x = norm.normalize(batch.features.to(device))
            svm = batch.student_valid_mask.to(device)
            boundaries = batch.episode_boundaries.to(device)
            q_target = batch.quality_target.to(device)
            q_mask = batch.quality_supervision_mask.to(device)
            r_target = batch.release_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, svm, boundaries)
            loss, components = compute_v4_loss(
                logits, q_target, q_mask,
                release_target=r_target if args.aux_release else None,
                release_weight=args.release_weight,
                ranking_weight=args.ranking_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_terms.append(float(loss.detach()))

        avg = sum(epoch_terms) / len(epoch_terms)
        epoch_losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{args.epochs}: loss={avg:.6f}")

    print(f"Final loss: {epoch_losses[-1]:.6f}")

    # Save checkpoint
    ckpt_sha = save_v4_checkpoint_bundle(
        model, args.output,
        view=args.view, aux_release=args.aux_release,
        seed=args.seed, fold_id=args.fold_id,
        normalization=norm, losses=epoch_losses,
        protocol_sha256=json_sha({"schema": "V4_FORMAL_TRAINER_V1"}),
        s1_root_sha256="15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf",
        v21_root_sha256="015040ac8b964ec5e148e254028028c8d918ffff60e52e9d5110fa9f1392a165",
        fold_bundle_sha256="efeb24ce17c2de0eaf83aaf54099d8043b456dd6acb5c871cfd8e0daae1ef946",
        normalization_bundle_sha256=norm.sha256,
        runner_binding_sha256=json_sha({"runner": "train_v4_quality_detector.py"}),
        train_episode_count=len(train_eps),
    )
    print(f"Checkpoint SHA: {ckpt_sha}")


if __name__ == "__main__":
    main()
