import json
from pathlib import Path


def test_stage_x0_protocol_is_frozen_and_protected_boundary_is_closed():
    path = Path(__file__).parents[1] / "configs" / "STAGE_X_X0_DUTY_CYCLE_MECHANISM_PROTOCOL_V1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_MEDIATOR_AVAILABILITY_AUDIT"
    assert protocol["protected_boundary"]["eval160"] == "UNREAD"
    assert protocol["protected_boundary"]["protected_evaluation"] == "UNREAD"
    assert all(value == 0 for value in protocol["protected_boundary"]["protected_counters"].values())
    assert protocol["scientific_scope"]["outcome_data_used_for_selection"] is False
    assert "exact intersection of relative_step" in protocol["population"]["observation_window"]
