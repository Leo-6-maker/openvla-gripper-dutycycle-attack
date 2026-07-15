import pytest


torch = pytest.importorskip("torch")

from gripper_attack.b3_stateful import (  # noqa: E402
    B3_25D,
    B3_25D9D,
    B3_HEADS,
    B3Normalization,
    B3StatefulRuntime,
    compute_b3_loss,
    load_b3_checkpoint,
    save_b3_checkpoint,
)


def _assert_hidden_equal(left, right):
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0.0, atol=1e-6)
    else:
        assert not isinstance(right, tuple)
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-6)


def _stepwise(model, x25, x9=None, mask=None, hidden=None):
    rows = {f"{name}_logit": [] for name in B3_HEADS}
    current = hidden
    for step in range(x25.shape[1]):
        step_logits, current = model.step(
            x25[:, step],
            None if x9 is None else x9[:, step],
            current,
            None if mask is None else mask[:, step],
        )
        for name, value in step_logits.items():
            rows[name].append(value)
    return {name: torch.stack(values, dim=1) for name, values in rows.items()}, current


@pytest.mark.parametrize("model_cls", [B3_25D, B3_25D9D])
@pytest.mark.parametrize("length", [1, 2, 3, 16, 220, 280, 300, 520])
def test_sequence_and_one_step_stateful_parity(model_cls, length):
    torch.manual_seed(11)
    model = model_cls().eval()
    x25 = torch.randn(2, length, 25)
    x9 = torch.randn(2, length, 9) if model_cls is B3_25D9D else None
    with torch.no_grad():
        sequence_logits, sequence_hidden = model.forward_sequence(x25, x9)
        step_logits, step_hidden = _stepwise(model, x25, x9)
    for name in sequence_logits:
        torch.testing.assert_close(sequence_logits[name], step_logits[name], rtol=0.0, atol=1e-6)
    _assert_hidden_equal(sequence_hidden, step_hidden)


def test_reset_is_episode_only_and_runtime_does_not_reset_on_event():
    torch.manual_seed(12)
    model = B3_25D9D().eval()
    x_first = torch.randn(1, 6, 25)
    z_first = torch.randn(1, 6, 9)
    x_second = torch.randn(1, 5, 25)
    z_second = torch.randn(1, 5, 9)

    with torch.no_grad():
        separate_second, _ = model.forward_sequence(x_second, z_second)
        first_logits, hidden = model.forward_sequence(x_first, z_first)
        del first_logits
        reset_second, _ = model.forward_sequence(x_second, z_second, hidden=model.initial_hidden(1))
        del hidden

    for name in separate_second:
        torch.testing.assert_close(separate_second[name], reset_second[name], rtol=0.0, atol=1e-6)

    runtime = B3StatefulRuntime(model)
    runtime.step(x_first[:, 0], z_first[:, 0])
    carried = runtime.step(x_first[:, 1], z_first[:, 1])
    runtime.step(x_first[:, 2], z_first[:, 2])
    continued = runtime.step(x_second[:, 0], z_second[:, 0])
    runtime.reset_episode()
    reset = runtime.step(x_second[:, 0], z_second[:, 0])
    assert any(not torch.equal(continued[name], reset[name]) for name in carried)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_tbptt_chunk_boundaries_carry_hidden_without_changing_forward(chunk_size):
    torch.manual_seed(13)
    model = B3_25D9D().eval()
    x25 = torch.randn(1, 101, 25)
    x9 = torch.randn(1, 101, 9)
    mask = torch.ones(1, 101, dtype=torch.bool)
    with torch.no_grad():
        full, full_hidden = model.forward_sequence(x25, x9, mask=mask)
        chunks = {f"{name}_logit": [] for name in B3_HEADS}
        hidden = None
        for start in range(0, x25.shape[1], chunk_size):
            end = min(start + chunk_size, x25.shape[1])
            part, hidden = model.forward_sequence(
                x25[:, start:end], x9[:, start:end], hidden=hidden, mask=mask[:, start:end]
            )
            for name, value in part.items():
                chunks[name].append(value)
            if isinstance(hidden, tuple):
                hidden = tuple(value.detach() for value in hidden)
            else:
                hidden = hidden.detach()
        chunked = {name: torch.cat(values, dim=1) for name, values in chunks.items()}
    for name in full:
        torch.testing.assert_close(full[name], chunked[name], rtol=0.0, atol=1e-6)
    _assert_hidden_equal(full_hidden, hidden)


def test_padding_does_not_update_hidden_and_unknown_loss_is_finite():
    torch.manual_seed(14)
    model = B3_25D9D().eval()
    x25 = torch.randn(2, 6, 25)
    x9 = torch.randn(2, 6, 9)
    mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, False, False, False]])
    with torch.no_grad():
        _, prefix_hidden = model.forward_sequence(x25[:, :3], x9[:, :3])
        outputs, padded_hidden = model.forward_sequence(x25, x9, mask=mask)
    _assert_hidden_equal(prefix_hidden, padded_hidden)

    targets = {name: torch.full_like(value, float("nan")) for name, value in outputs.items()}
    targets = {name.removesuffix("_logit"): value for name, value in targets.items()}
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    loss = compute_b3_loss(outputs, targets, masks)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_checkpoint_roundtrip_closes_engineering_contract(tmp_path):
    torch.manual_seed(15)
    model = B3_25D9D().eval()
    normalization = B3Normalization(
        mean_25d=tuple(float(i) / 100.0 for i in range(25)),
        std_25d=tuple(1.0 + float(i) / 100.0 for i in range(25)),
        mean_9d=tuple(float(i) / 100.0 for i in range(9)),
        std_9d=tuple(1.0 + float(i) / 100.0 for i in range(9)),
    )
    checkpoint = tmp_path / "b3_smoke.pt"
    save_b3_checkpoint(checkpoint, model, normalization)
    restored, config, restored_norm, payload = load_b3_checkpoint(checkpoint)
    assert payload["status"] == "ENGINEERING_SMOKE_ONLY"
    assert payload["formal_model"] is False
    assert config.sha256 == payload["config_hash"]
    assert restored_norm.sha256 == payload["normalization_hash"]

    x25 = torch.randn(1, 16, 25)
    x9 = torch.randn(1, 16, 9)
    with torch.no_grad():
        original, _ = model.forward_sequence(x25, x9)
        reloaded, _ = restored.forward_sequence(x25, x9)
    for name in original:
        torch.testing.assert_close(original[name], reloaded[name], rtol=0.0, atol=0.0)


def test_tiny_synthetic_overfit_smoke_is_finite_and_decreases_loss():
    torch.manual_seed(16)
    model = B3_25D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    x25 = torch.randn(8, 8, 25)
    labels = (x25[..., 0] > 0).float()
    targets = {name: labels for name in B3_HEADS}
    masks = {name: torch.ones_like(labels, dtype=torch.bool) for name in B3_HEADS}
    model.train()
    with torch.no_grad():
        initial = compute_b3_loss(model.forward_sequence(x25)[0], targets, masks).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model.forward_sequence(x25)
        loss = compute_b3_loss(logits, targets, masks)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    with torch.no_grad():
        final = compute_b3_loss(model.forward_sequence(x25)[0], targets, masks).item()
    assert final < initial
