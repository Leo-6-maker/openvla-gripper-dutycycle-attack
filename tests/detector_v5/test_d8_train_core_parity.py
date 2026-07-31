from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DETECTOR_SCRIPTS = ROOT / "scripts" / "detector_v5"
if str(DETECTOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SCRIPTS))

import torch
import torch.optim as optim

from d8_train_core import (
    checkpoint_roundtrip_parity,
    compute_loss,
    compute_normalization,
    continuation_parity,
    create_model,
)


def test_checkpoint_and_continuation_parity_cpu(tmp_path: Path):
    torch.manual_seed(7)
    x = torch.randn(64, 25)
    y = (torch.rand(64) > 0.5).float()
    w = torch.rand(64) + 0.1
    norm = compute_normalization(x, source_identity_digest="abc")
    model = create_model(seed=11)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model((x - torch.tensor(norm["mean"])) / torch.tensor(norm["std"]))
        loss = compute_loss(logits, y, w)
        loss.backward()
        optimizer.step()

    roundtrip = checkpoint_roundtrip_parity(
        model, optimizer, x[:16], y[:16], w[:16], norm,
        tmp_path / "roundtrip.pt", "cpu",
    )
    assert all(roundtrip[key] for key in (
        "pre_post_logits_match", "pre_post_loss_match", "params_match",
        "optimizer_match", "normalization_match",
    ))

    continuation = continuation_parity(
        model, optimizer, x[:16], y[:16], w[:16], norm,
        tmp_path / "continuation.pt", "cpu",
    )
    assert all(continuation[key] for key in (
        "pre_step_logits_match", "pre_step_loss_match",
        "post_step_params_match", "post_step_optimizer_match",
        "post_step_logits_match", "post_step_loss_match", "post_step_rng_match",
    ))
