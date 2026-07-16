import json
from pathlib import Path

import numpy as np
import pytest
import torch

import gripper_attack.official_openvla_adapter as adapter_module
from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
from gripper_attack.official_generation_contract import validate_generation_contract


class _Generation:
    def __init__(self):
        self.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.scores = [torch.zeros((1, 8)) for _ in range(7)]


class _Model:
    def __init__(self, passes: int):
        self.passes = passes
        self.generate_calls = 0

    def generate(self, **_kwargs):
        self.generate_calls += 1
        return _Generation()

    def predict_action(self, **_kwargs):
        for _ in range(self.passes):
            self.generate()
        return np.zeros(7, dtype=np.float32)


def _adapter(model: _Model) -> OfficialOpenVLAActionAdapter:
    value = object.__new__(OfficialOpenVLAActionAdapter)
    value.model = model
    value.processor = object()
    value.device = "cpu"
    value.unnorm_key = "libero_10"
    value.center_crop = True
    value.base_vla_name = ""
    return value


def _patch_adapter(monkeypatch, *, decoded_action=None):
    monkeypatch.setattr(
        adapter_module,
        "prepare_official_inputs",
        lambda *_args, **_kwargs: ({"input_ids": torch.zeros((1, 1), dtype=torch.long)}, "prompt", np.zeros((2, 2, 3))),
    )
    monkeypatch.setattr(adapter_module, "generated_action_tokens", lambda *_args, **_kwargs: list(range(7)))
    monkeypatch.setattr(
        adapter_module,
        "decode_official_generated_action",
        lambda *_args, **_kwargs: np.zeros(7, dtype=np.float32) if decoded_action is None else decoded_action,
    )


def test_one_real_generation_pass_is_recorded(monkeypatch):
    _patch_adapter(monkeypatch)
    model = _Model(1)

    _action, _generation, meta = _adapter(model).predict_action_with_scores(np.zeros((2, 2, 3)), "task")

    assert model.generate_calls == 1
    assert meta["generation_passes_per_step"] == 1


@pytest.mark.parametrize("passes", [0, 2])
def test_zero_or_multiple_generation_passes_fail_closed(monkeypatch, passes):
    _patch_adapter(monkeypatch)

    with pytest.raises(RuntimeError, match=rf"OFFICIAL_GENERATION_PASS_COUNT_FAIL:{passes}"):
        _adapter(_Model(passes)).predict_action_with_scores(np.zeros((2, 2, 3)), "task")


def test_score_action_from_a_different_generation_fails_closed(monkeypatch):
    _patch_adapter(monkeypatch, decoded_action=np.ones(7, dtype=np.float32))

    with pytest.raises(RuntimeError, match="SINGLE_GENERATION_ACTION_PARITY_FAIL"):
        _adapter(_Model(1)).predict_action_with_scores(np.zeros((2, 2, 3)), "task")


def _write_contract_root(root: Path, *, omit_step_count: bool = False, policy_rows: int = 2):
    root.mkdir()
    metadata = {"runtime_valid": True, "generation_passes_per_step": 1}
    runtime = {"runtime_valid": True, "generation_passes_per_step": 1}
    summary = {"steps": 2, "clean": True, "success": False}
    steps = []
    for index in range(2):
        row = {
            "step": index,
            "generation_passes_per_step": 1,
            "single_generation_parity_pass": True,
            "score_adapter_parity_pass": True,
            "action_token_ids": list(range(7)),
            "score_head_summary": [{} for _ in range(7)],
        }
        if omit_step_count:
            row.pop("generation_passes_per_step")
        steps.append(row)
    policies = [
        {
            "step": index,
            "generation_passes_per_step": 1,
            "single_generation_parity_pass": True,
            "score_adapter_parity_pass": True,
            "action_token_ids": list(range(7)),
        }
        for index in range(policy_rows)
    ]
    for name, value in [("episode_metadata.json", metadata), ("runtime_audit.json", runtime), ("episode_summary.json", summary)]:
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "step_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8")
    (root / "policy_intent_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in policies), encoding="utf-8")


def test_missing_generation_field_fails_at_sealing_contract(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_contract_root(root, omit_step_count=True)

    with pytest.raises(ValueError, match="GENERATION_PASS_FIELD_INVALID"):
        validate_generation_contract(root)


def test_generation_record_length_mismatch_fails_at_sealing_contract(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_contract_root(root, policy_rows=1)

    with pytest.raises(ValueError, match="GENERATION_RECORD_LENGTH_MISMATCH"):
        validate_generation_contract(root)
