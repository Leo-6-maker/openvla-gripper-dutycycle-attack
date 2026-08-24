from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"


def test_contract_is_suite_matched_and_fail_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == "STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1"
    assert contract["scientific_authority"] == "X1R_NOT_AUTHORIZED"
    assert set(contract["suites"]) == {"libero_10", "libero_goal", "libero_object", "libero_spatial"}
    assert contract["historical_boundary"]["stage_v_launch_time_weight_identity"] == "NOT_IDENTIFIABLE"
    assert contract["historical_boundary"]["stage_vi_b2_launch_time_weight_identity"] == "NOT_IDENTIFIABLE"
    assert contract["action_semantics"]["token_id_binding"].startswith("per_suite_checkpoint")
    assert contract["protected_boundary"]["eval160"] == "UNREAD"
    assert all(value == 0 for value in contract["protected_boundary"]["protected_counters"].values())


def test_prospective_roles_share_one_suite_identity() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for suite, cfg in contract["suites"].items():
        assert cfg["model_path"]
        assert cfg["unnorm_key"] == suite
        assert cfg["prospective_roles"]["vphys_intervention_policy"].endswith("NOT_EXECUTED")
        assert cfg["prospective_roles"]["proposed_pgd_victim"].endswith("NOT_EXECUTED")
