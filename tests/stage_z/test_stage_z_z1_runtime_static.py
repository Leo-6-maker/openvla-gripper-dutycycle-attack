from __future__ import annotations

import json
from pathlib import Path

from stage_z_preparation.contract import validate_final_action


ROOT = Path(__file__).parents[2]


def test_z1_protocol_is_engineering_only_and_three_by_four() -> None:
    protocol = json.loads((ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN"
    assert protocol["authorized_scope"]["cells"] == 12
    assert protocol["execution_contract"]["scientific_parent_exposure"] == 0
    assert protocol["execution_contract"]["interventions"] == 0
    assert protocol["execution_contract"]["pgd_calls"] == 0
    assert protocol["authorized_scope"]["terminal_action"] == "STOP_FOR_PI"


def test_z1_runner_has_no_scientific_outcome_fields_in_action_contract() -> None:
    source = (ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py").read_text(encoding="utf-8")
    assert "physical_outcome_read" in source
    assert "attacked_env_steps" in source
    assert validate_final_action((0.0,) * 7) == (0.0,) * 7
