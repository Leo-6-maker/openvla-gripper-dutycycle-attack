from __future__ import annotations

from scripts.detector_v5.audit_stage_v_m4_matched_parent import _truth_label


def test_m4_truth_table_never_promotes_invalid_control_or_treatment() -> None:
    assert _truth_label(True, True, 0, 0) == "NO_PHYSICAL_VULNERABILITY"
    assert _truth_label(True, True, 0, 1) == "V_PHYS"
    assert _truth_label(False, True, 1, 1) == "CONTROL_CONTAMINATION_ABSTAIN"
    assert _truth_label(False, True, 1, 0) == "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    assert _truth_label(True, False, 0, None) == "TREATMENT_INVALID_ABSTAIN"
