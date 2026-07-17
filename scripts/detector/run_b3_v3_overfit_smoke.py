#!/usr/bin/env python3
"""Synthetic-only overfit smoke for the Official V3 stateful trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gripper_attack.b3_formal import B3ModelConfig, B3Normalization, build_b3_model, compute_b3_loss, save_b3_checkpoint
from gripper_attack.b3_v3_dataset import B3Episode, pad_episode_batch


def synthetic_episodes(*, seed: int = 20260717, variant: str = "B3_25D") -> list[B3Episode]:
    generator = torch.Generator().manual_seed(seed)
    episodes: list[B3Episode] = []
    for episode_index in range(8):
        steps = 10 + episode_index % 3
        x25 = torch.randn(steps, 25, generator=generator)
        signal = x25[:, 0] > 0
        targets = {
            "grasp_support": signal.float(),
            "retention_active": (x25[:, 1] > 0).float(),
            "retention_continuation_t10": (x25[:, 0] + x25[:, 1] > 0).float(),
            "release_imminent": (x25[:, 2] > 0).float(),
        }
        masks = {head: torch.ones(steps, dtype=torch.bool) for head in targets}
        x9 = torch.randn(steps, 9, generator=generator) if variant == "B3_25D9D" else None
        episodes.append(B3Episode(
            canonical_parent_key=f"libero_object/task_{episode_index % 4:02d}/state_{episode_index:02d}",
            suite="libero_object", task_idx=episode_index % 4, state_id=episode_index,
            split="FIT_TRAIN", task_success=episode_index % 2 == 0, features_25d=x25,
            targets=targets, known_masks=masks, valid_mask=torch.ones(steps, dtype=torch.bool), features_9d=x9,
        ))
    return episodes


def _fit_overfit_smoke(*, variant: str, steps: int, seed: int):
    torch.manual_seed(seed)
    episodes = synthetic_episodes(seed=seed, variant=variant)
    batch = pad_episode_batch(episodes)
    model = build_b3_model(B3ModelConfig(variant=variant))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    initial = None
    final = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.x25, batch.x9, mask=batch.padding_mask)
        loss = compute_b3_loss(logits, batch.targets, batch.known_masks, episode_valid_mask=batch.episode_valid_mask, padding_mask=batch.padding_mask)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = float(loss.detach())
    assert initial is not None and final is not None
    if not final < initial:
        raise RuntimeError(f"synthetic overfit did not improve: {initial} -> {final}")
    return model, {"status": "PASS_SYNTHETIC_ONLY", "variant": variant, "initial_loss": initial, "final_loss": final}


def run_overfit_smoke(*, variant: str = "B3_25D", steps: int = 40, seed: int = 20260717) -> dict[str, float | str]:
    _, result = _fit_overfit_smoke(variant=variant, steps=steps, seed=seed)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), default="B3_25D")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-checkpoint", type=Path)
    args = parser.parse_args()
    model, result = _fit_overfit_smoke(variant=args.variant, steps=args.steps, seed=20260717)
    if args.output_checkpoint:
        save_b3_checkpoint(args.output_checkpoint, model, B3Normalization.identity(), extra={"smoke": result})
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
