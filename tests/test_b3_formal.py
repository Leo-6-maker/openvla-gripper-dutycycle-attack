import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_formal import (  # noqa: E402
    B3_HEADS,
    B3ModelConfig,
    B3Normalization,
    build_b3_model,
    compute_b3_loss,
    json_sha,
    load_b3_checkpoint,
    save_b3_checkpoint,
    validate_training_authorization,
)
from gripper_attack.b3_training_protocol import build_training_authorization


def _hidden_equal(left, right):
    if isinstance(left, tuple):
        return all(torch.equal(a, b) for a, b in zip(left, right))
    return torch.equal(left, right)


@pytest.mark.parametrize("variant", ["B3_25D", "B3_25D9D"])
def test_official_stateful_sequence_step_parity_and_padding(variant):
    torch.manual_seed(7)
    model = build_b3_model(B3ModelConfig(variant=variant)).eval()
    x25 = torch.randn(2, 6, 25)
    x9 = torch.randn(2, 6, 9) if variant == "B3_25D9D" else None
    mask = torch.tensor([[True, True, False, True, True, True], [True, True, True, False, True, True]])
    sequence_logits, sequence_hidden = model(x25, x9, mask=mask)
    hidden = None
    rows = {f"{head}_logit": [] for head in B3_HEADS}
    for step in range(6):
        current, hidden = model.step(x25[:, step], None if x9 is None else x9[:, step], hidden, mask[:, step])
        for name, value in current.items():
            rows[name].append(value)
    for name in rows:
        assert torch.allclose(sequence_logits[name], torch.stack(rows[name], dim=1), atol=0.0, rtol=0.0)
    assert _hidden_equal(sequence_hidden, hidden)


def test_all_unknown_loss_is_zero_and_strict_head_contract():
    logits = {f"{head}_logit": torch.randn(2, 3, requires_grad=True) for head in B3_HEADS}
    targets = {head: torch.zeros(2, 3) for head in B3_HEADS}
    masks = {head: torch.zeros(2, 3, dtype=torch.bool) for head in B3_HEADS}
    loss = compute_b3_loss(logits, targets, masks)
    assert float(loss) == 0.0
    loss.backward()
    assert all(value.grad is not None for value in logits.values())
    with pytest.raises(ValueError, match="all four"):
        compute_b3_loss({"grasp_support_logit": logits["grasp_support_logit"]}, targets, masks)


def test_checkpoint_has_separate_official_schema_but_defaults_to_smoke(tmp_path):
    model = build_b3_model(B3ModelConfig())
    path = tmp_path / "smoke.pt"
    save_b3_checkpoint(path, model, B3Normalization.identity())
    _, config, normalization, payload = load_b3_checkpoint(path)
    assert payload["schema"] == "c2g.b3.official_v3.detector_checkpoint.v1"
    assert payload["formal_model"] is False
    assert config.variant == "B3_25D"
    assert normalization.sha256 == payload["normalization_sha256"]


def test_formal_authorization_requires_machine_built_sealed_inputs(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        validate_training_authorization({"formal_training_authorized": True})
    runner = {
        "status": "PASS", "runner_head": "d" * 40, "runner_worktree_clean": True,
    }
    runner["runner_binding_sha256"] = json_sha({key: value for key, value in runner.items()})
    snapshots = {name: f"{index + 1:064x}" for index, name in enumerate((
        "formal_fit_registry_sha256", "formal_registry_summary_sha256", "formal_registry_root_sha256",
        "s1_corpus_sha256", "s1_root_audit_sha256", "teacher_aggregate_sha256",
        "training_protocol_sha256", "source_contract_sha256", "protocol_sha256", "feature_rebuilder_sha256",
        "normalization_bundle_sha256", "normalization_sha256", "fold_manifest_sha256",
    ))}
    auth = build_training_authorization(
        tmp_path / "authorization", variant="B3_25D", fold_id=0, seed=20260717,
        input_snapshots=snapshots, runner_binding=runner, generator_script_sha256="f" * 64,
    )
    validate_training_authorization(auth)
