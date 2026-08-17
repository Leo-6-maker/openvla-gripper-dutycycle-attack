from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_t1_score_audit_is_clean_only_and_does_not_call_attack_or_environment():
    text = (REPO / "scripts" / "stage_x" / "audit_stage_x1r_t1_score_path.py").read_text(encoding="utf-8")
    assert ".step(" not in text
    assert "env.step" not in text
    assert ".attack(" not in text
    assert "pgd_calls" in text
    assert "env_step_calls" in text


def test_iterate_selection_is_frozen_before_any_matrix_protocol():
    import json

    protocol = json.loads((REPO / "configs" / "STAGE_X_X1R_T1_PROSPECTIVE_DETECTOR_PGD_PROTOCOL_V1.json").read_text())
    assert protocol["iterate_selection"] == {
        "rule": "final_iterate_only",
        "frozen_before_prospective_outcome": True,
        "outcome_informed_selection": False,
    }
