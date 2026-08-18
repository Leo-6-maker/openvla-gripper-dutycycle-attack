import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.stage_x_x1r_v2_schedule_contract import (  # noqa: E402
    ATTACK_WINDOW_LENGTH,
    PHYSICAL_FOLLOWUP_LENGTH,
    PREV_DELTA_BOUNDARIES,
    NO_EMIT,
    attack_steps,
    first_emit_or_no_emit,
    followup_steps,
    legal_horizon,
)


def test_timing_contract_boundaries():
    assert attack_steps(7) == (7, 8, 9, 10, 11)
    assert followup_steps(7) == (12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    assert len(attack_steps(0)) == ATTACK_WINDOW_LENGTH == 5
    assert len(followup_steps(0)) == PHYSICAL_FOLLOWUP_LENGTH == 10
    assert legal_horizon(7, 22) is True
    assert legal_horizon(7, 21) is False
    assert first_emit_or_no_emit(None) == NO_EMIT
    assert first_emit_or_no_emit(7) == 7
    assert PREV_DELTA_BOUNDARIES["entry"] == "reset_to_zero_at_attack_window_entry"


def test_timing_contract_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        attack_steps(-1)
    with pytest.raises(ValueError):
        legal_horizon(0, -1)


def test_protocol_is_static_and_execution_locked():
    protocol = json.loads(
        (ROOT / "configs/STAGE_X_X1R_T1D0R1_PRECLEAN_INTEGRITY_AUTHORITY_V1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["selection"]["selection_salt"] == "STAGE_X_X1R_T1D0_PARENT_AUTHORITY_V1_20260818"
    assert protocol["historical_records"]["new_outcome_information_used"] is False
    plan = protocol["clean_materialization_plan"]
    assert plan["enabled"] is False
    assert plan["execution_authorized"] is False
    assert all(value is False for value in protocol["authorization"].values() if isinstance(value, bool))
    assert all(value == 0 for value in protocol["protected_boundary"]["counters"].values())
    script = (ROOT / "scripts/stage_x/audit_stage_x1r_t1d0r1_preclean_integrity.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in ("import torch", "transformers", "gym", "model.generate", "env.step("):
        assert forbidden not in script
