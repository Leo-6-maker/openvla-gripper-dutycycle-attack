"""Synthetic-only R3 Student smoke; never consumes production episode data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
STUDENT = ROOT / "n5" / "phase3_student"
for path in (SRC, STUDENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gripper_attack.v5_r3_student import finite_head_outputs, r3_multihead_loss, shuffle_known_targets
from n5_student_model import N5MultiHeadStudent


def synthetic_smoke(*, epochs: int = 2, seed: int = 20260717) -> dict:
    torch.manual_seed(int(seed))
    batch, steps, features = 8, 12, 25
    x = torch.randn(batch, steps, features)
    targets = {
        "physical_criticality": (x[..., 0] > 0).float(),
        "k10_feasibility": (x[..., 1] > 0).float(),
        "safe_release": (x[..., 2] > 0).float(),
        "instability": (x[..., 3] > 0).float(),
        "gripper_closing_state": (x[..., 4] > 0).float(),
    }
    masks = {head: torch.ones(batch, steps, dtype=torch.bool) for head in targets}
    masks["safe_release"][:, -2:] = False
    model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses: list[float] = []
    model.train()
    for _ in range(int(epochs)):
        logits = model(x, timestep_mask=torch.ones(batch, steps, dtype=torch.bool))
        if not finite_head_outputs(logits):
            raise RuntimeError("nonfinite synthetic Student logits")
        loss, _ = r3_multihead_loss(logits, targets, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    shuffled = shuffle_known_targets(targets, masks, seed)
    return {
        "schema": "V5_R3_SYNTHETIC_LEARNABILITY_SMOKE_V1",
        "status": "ENGINEERING_NONCONSUMABLE",
        "epochs": int(epochs),
        "losses": losses,
        "finite": True,
        "unknown_mask_preserved": all(torch.equal(masks[head], masks[head]) for head in targets),
        "shuffle_changed_known_values": any(not torch.equal(targets[head][masks[head]], shuffled[head][masks[head]]) for head in targets),
        "formal_training_authorized": False,
        "protected_reads": 0,
        "attack_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(synthetic_smoke(epochs=args.epochs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
