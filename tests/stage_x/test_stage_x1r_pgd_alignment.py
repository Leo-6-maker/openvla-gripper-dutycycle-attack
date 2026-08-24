from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker  # noqa: E402
from gripper_attack.m3_controls import project_and_cast_processor_values  # noqa: E402
from gripper_attack.route_contract import (  # noqa: E402
    RouteContractError,
    route_config_from_attack_config,
    validate_attack_request,
)
from scripts.stage_x.audit_stage_x1r_pgd_alignment import (  # noqa: E402
    _Config,
    _TokenizerStubModel,
    action_token_logit_row_index,
    canonical_token_ids,
    run_numerical_audit,
    run_causal_row_toy,
)


def _stub() -> _TokenizerStubModel:
    config = json.loads((REPO / "configs" / "STAGE_X_X1R_T0_PGD_STANDARDIZATION_AND_TOKEN_AUTHORITY_PROTOCOL_V1.json").read_text())
    model_config = {"text_config": {"vocab_size": 32064}, "pad_to_multiple_of": 64, "n_action_bins": 256, "norm_stats": {"libero_goal": {"action": {"q01": [0.0] * 6 + [0.0], "q99": [1.0] * 6 + [1.0], "mask": [False] * 7}}}}
    return _TokenizerStubModel(model_config, "libero_goal", 32000, np.linspace(-1.0, 1.0, 255, dtype=np.float32))


def test_native_endpoint_and_project_helper_are_not_equivalent():
    model = _stub()
    helper = TokenPrefixPGDAttacker(model, object(), {}, device="cpu")
    raw = np.ones(7, dtype=np.float32)
    helper_ids = helper.action_to_token_ids(raw, "libero_goal").numpy()
    native_info = {
        "native": type("Native", (), {"min_action": -1.0, "max_action": 1.0, "bins": np.linspace(-1.0, 1.0, 256), "n_bins": 256})(),
        "tokenizer_vocab_size": 32000,
    }
    canonical = canonical_token_ids(native_info, np.ones(7, dtype=np.float32))
    assert int(canonical[-1]) == 31744
    assert int(helper_ids[-1]) == 31745


def test_action_token_logit_rows_cover_all_action_dimensions():
    assert [action_token_logit_row_index(dim, 7) for dim in range(7)] == [-8, -7, -6, -5, -4, -3, -2]
    assert run_causal_row_toy()["pass"] is True


def test_processor_projection_respects_budget_for_fp16_and_bfloat16():
    for dtype in (torch.float16, torch.bfloat16):
        original = torch.tensor([0.12345, -0.23456], dtype=dtype)
        candidate = original.float() + torch.tensor([0.1, -0.1])
        projected, _ = project_and_cast_processor_values(original, candidate, epsilon=0.1, candidate_is_delta=False)
        assert float((projected.float() - original.float()).abs().max()) <= 0.1000001


def test_numerical_audit_exercises_nonzero_cw_descent():
    report = run_numerical_audit()
    assert report["pass"] is True
    assert report["cw"]["initial_loss"] > 0.0
    assert report["cw"]["loss_after_sign_descent"] < report["cw"]["initial_loss"]


def test_strict_route_rejects_fallback():
    route = route_config_from_attack_config({"attack_optimizer": {"method": "token_prefix_pgd", "objective": "gripper_logit_margin_cw", "strict_route": True, "allow_fallback": True}})
    with pytest.raises(RouteContractError):
        validate_attack_request(route, target_action_present=True)


def test_t0_script_has_no_execution_path():
    text = (REPO / "scripts" / "stage_x" / "audit_stage_x1r_pgd_alignment.py").read_text(encoding="utf-8")
    assert ".step(" not in text
    assert "env.step" not in text
    assert "attack(" not in text
