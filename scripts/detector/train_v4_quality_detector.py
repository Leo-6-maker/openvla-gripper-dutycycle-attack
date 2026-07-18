#!/usr/bin/env python3
"""Authorization-gated corrected V4 FIT-fold trainer.

This entry point intentionally refuses the old ``--output``/``--v21-root``
interface.  A sealed machine-built authorization bundle is mandatory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from gripper_attack.v4_contract import (
    FIT_STATES,
    SUITES,
    identity_sha,
    json_sha,
    measured_git_binding,
    sha256_file,
    verify_checksum_manifest,
)
from gripper_attack.v4_dataset import (
    V4EpisodeSampler,
    compute_v4_fold_normalization,
    load_v4_episode,
    pad_v4_episode_batch,
    select_fold_episodes,
)
from gripper_attack.v4_formal import (
    V4Normalization,
    V4StatefulQualityGRU,
    compute_v4_loss,
    save_v4_checkpoint_bundle,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_authorization(root: Path) -> dict:
    if not root.is_dir() or not (root / "authorization.json").is_file():
        raise ValueError("--authorization must be a sealed authorization directory")
    verify_checksum_manifest(root)
    payload = _load_json(root / "authorization.json")
    if payload.get("schema") != "DETECTOR_V4_TRAINING_AUTHORIZATION_V2":
        raise ValueError("wrong V4 authorization schema")
    if payload.get("formal_training_authorized") is not True or payload.get("formal_attack_authorized") is not False:
        raise ValueError("authorization does not authorize FIT training only")
    body = dict(payload)
    expected = body.pop("authorization_payload_sha256", None)
    if expected != json_sha(body):
        raise ValueError("authorization payload SHA mismatch")
    return payload


def _load_all(s1_root: Path, teacher_root: Path, view: str) -> list:
    result = []
    for suite in SUITES:
        for task in range(10):
            for state in sorted(FIT_STATES):
                episode = load_v4_episode(s1_root, teacher_root, suite, task, state, view)
                if episode is None:
                    raise ValueError(f"missing FIT episode: {suite}/task_{task:02d}/state_{state:02d}")
                result.append(episode)
    if len(result) != 800:
        raise ValueError(f"expected 800 FIT episodes, got {len(result)}")
    return result


def _candidate_config(candidate: str) -> tuple[str, bool, float, float]:
    return {
        "C0": ("A", False, 0.0, 0.0),
        "C1": ("B", False, 0.0, 0.0),
        "C2": ("B", False, 0.5, 0.0),
        "C3": ("C", True, 0.5, 0.3),
    }[candidate]


def train(args: argparse.Namespace) -> dict:
    if not args.execute_formal:
        raise ValueError("formal V4 training requires --execute-formal")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    auth = _load_authorization(args.authorization)
    if auth["candidate"] != args.candidate or int(auth["fold_id"]) != args.fold_id or int(auth["seed"]) != args.seed:
        raise ValueError("CLI candidate/fold/seed does not match authorization")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if args.dtype != "float32":
        raise ValueError("V4 formal protocol is FP32 only")

    s1_before = verify_checksum_manifest(args.s1_root)
    teacher_before = verify_checksum_manifest(args.teacher_root)
    fold_before = verify_checksum_manifest(args.fold_root)
    input_shas = auth["input_snapshots"]
    if input_shas["s1_root_sha256"] != s1_before["sha256sums_sha256"] or input_shas["fold_manifest_sha256"] != fold_before["sha256sums_sha256"]:
        raise ValueError("authorization input root SHA mismatch")
    runner = measured_git_binding(args.runner_repo, [args.runner_script, args.runner_config])
    if runner != auth["runner_binding"]:
        raise ValueError("measured runner binding differs from authorization")

    view, aux_release, ranking_weight, release_weight = _candidate_config(args.candidate)
    episodes = _load_all(args.s1_root, args.teacher_root, view)
    train_episodes = select_fold_episodes(episodes, args.fold_id, "train")
    if len(train_episodes) != 600:
        raise ValueError(f"expected 600 train episodes, got {len(train_episodes)}")
    train_sha = identity_sha([ep.canonical_parent_key for ep in train_episodes])
    if train_sha != auth["train_identity_sha256"]:
        raise ValueError("measured train identity SHA differs from authorization")

    normalization_payload = _load_json(args.normalization_root / "normalization.json")
    normalization = V4Normalization.from_dict(normalization_payload["normalization"])
    if normalization.sha256 != normalization_payload.get("normalization_semantic_sha256"):
        raise ValueError("normalization semantic SHA mismatch")
    recomputed = compute_v4_fold_normalization(train_episodes, view)
    if recomputed.sha256 != normalization.sha256:
        raise ValueError("normalization does not match measured 600-episode train fold")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = V4StatefulQualityGRU(normalization.feature_count, hidden_dim=128, aux_release=aux_release).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = V4EpisodeSampler(train_episodes, base_seed=args.seed)
    losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        terms = []
        ordered = sampler.ordered_indices(epoch, shuffle=True)
        for start in range(0, len(train_episodes), args.batch_size):
            indices = ordered[start:start + args.batch_size]
            batch = pad_v4_episode_batch([train_episodes[index] for index in indices])
            x = normalization.normalize(batch.features.to(device))
            valid = batch.student_valid_mask.to(device)
            boundary = batch.episode_boundaries.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, valid, boundary)
            loss, _ = compute_v4_loss(
                logits, batch.quality_target.to(device), batch.quality_supervision_mask.to(device),
                window_id=batch.window_id.to(device),
                release_target=batch.release_target.to(device) if aux_release else None,
                release_mask=batch.release_supervision_mask.to(device) if aux_release else None,
                ranking_weight=ranking_weight, release_weight=release_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            terms.append(float(loss.detach()))
        losses.append(sum(terms) / len(terms))

    s1_after = verify_checksum_manifest(args.s1_root)
    teacher_after = verify_checksum_manifest(args.teacher_root)
    fold_after = verify_checksum_manifest(args.fold_root)
    if (s1_before["sha256sums_sha256"], teacher_before["sha256sums_sha256"], fold_before["sha256sums_sha256"]) != (s1_after["sha256sums_sha256"], teacher_after["sha256sums_sha256"], fold_after["sha256sums_sha256"]):
        raise ValueError("TOCTOU: an input sealed root changed during training")

    checkpoint_sha = save_v4_checkpoint_bundle(
        model, args.output_root, view=view, aux_release=aux_release, seed=args.seed, fold_id=args.fold_id,
        normalization=normalization, losses=losses,
        protocol_sha256=input_shas["protocol_sha256"], feature_protocol_sha256=input_shas["feature_protocol_sha256"],
        teacher_protocol_sha256=input_shas["teacher_aggregate_sha256"], s1_root_sha256=input_shas["s1_root_sha256"],
        teacher_root_sha256=teacher_before["sha256sums_sha256"], fold_bundle_sha256=input_shas["fold_manifest_sha256"],
        normalization_bundle_sha256=input_shas["normalization_bundle_sha256"],
        authorization_sha256=auth["authorization_payload_sha256"], runner_binding_sha256=runner["runner_binding_sha256"],
        train_identity_sha256=train_sha, device=str(device), dtype=args.dtype,
        trainer_sha256=sha256_file(Path(__file__).resolve()), evaluator_sha256=sha256_file(args.evaluator_script),
        extra={"candidate": args.candidate, "ranking_weight": ranking_weight, "release_weight": release_weight,
               "learning_rate": args.lr, "weight_decay": args.weight_decay, "gradient_clip": args.gradient_clip,
               "protected_splits_read": False},
    )
    return {"status": "PASS", "candidate": args.candidate, "fold_id": args.fold_id, "seed": args.seed, "checkpoint_sha256": checkpoint_sha, "final_loss": losses[-1]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--execute-formal", action="store_true")
    p.add_argument("--candidate", choices=["C0", "C1", "C2", "C3"], required=True)
    p.add_argument("--fold-id", type=int, choices=range(4), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--s1-root", type=Path, required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, required=True)
    p.add_argument("--normalization-root", type=Path, required=True)
    p.add_argument("--runner-repo", type=Path, required=True)
    p.add_argument("--runner-script", required=True)
    p.add_argument("--runner-config", required=True)
    p.add_argument("--evaluator-script", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
