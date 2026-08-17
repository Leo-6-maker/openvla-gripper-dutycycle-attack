from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage_x" / "audit_stage_x1r_t1_detector_authority.py"
SPEC = importlib.util.spec_from_file_location("t1_detector_authority", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_scheduler_is_one_shot_and_horizon_bound():
    predictions = [
        {"physical_criticality": 0.9, "gripper_closing_state": 0.9},
        {"physical_criticality": 0.9, "gripper_closing_state": 0.9},
    ]
    result = MODULE.schedule(predictions, [True, True], t5=0, h_phys=0, physical_threshold=0.55, closing_threshold=0.8)
    assert result["first_emit_step"] == 0
    assert result["emitted_count"] == 1
    assert sum(row["emitted_this_step"] for row in result["traces"]) == 1


def test_scheduler_keeps_no_emit_as_valid_observation():
    predictions = [{"physical_criticality": 0.1, "gripper_closing_state": 0.1}]
    result = MODULE.schedule(predictions, [True], t5=5, h_phys=10, physical_threshold=0.55, closing_threshold=0.8)
    assert result["first_emit_step"] is None
    assert result["emitted_count"] == 0
