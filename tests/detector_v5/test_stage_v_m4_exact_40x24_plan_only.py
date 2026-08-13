from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.run_stage_v_m4_exact_40x24_plan_only import _validate_protocol


def _protocol() -> dict:
    return json.loads(Path("configs/STAGE_V_M4_EXACT_40X24_PLAN_ONLY_PROTOCOL_V1.json").read_text(encoding="utf-8"))


def test_plan_protocol_freezes_zero_outcome_boundary() -> None:
    _validate_protocol(_protocol())


def test_plan_protocol_rejects_intervention() -> None:
    protocol = _protocol()
    protocol["operation"]["intervention_executed"] = True
    with pytest.raises(RuntimeError, match="PLAN_OPERATION_BOUNDARY_INVALID:intervention_executed"):
        _validate_protocol(protocol)
