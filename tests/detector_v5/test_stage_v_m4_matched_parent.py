from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.audit_stage_v_m4_matched_parent import _truth_label
from scripts.detector_v5.audit_stage_v_m4_static import audit
from scripts.detector_v5.stage_v_m4_governance import M4GovernanceError, validate_formal_m4_corridor_gate


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_m4_truth_table_never_promotes_invalid_control_or_treatment() -> None:
    assert _truth_label(True, True, 0, 0) == "NO_PHYSICAL_VULNERABILITY"
    assert _truth_label(True, True, 0, 1) == "V_PHYS"
    assert _truth_label(False, True, 1, 1) == "CONTROL_CONTAMINATION_ABSTAIN"
    assert _truth_label(False, True, 1, 0) == "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    assert _truth_label(True, False, 0, None) == "TREATMENT_INVALID_ABSTAIN"


def test_stale_protocol_and_pass_authorization_fail_under_corridor_governance() -> None:
    protocol_path = REPO_ROOT / "configs/STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with pytest.raises(M4GovernanceError, match="SUPERSEDED"):
        validate_formal_m4_corridor_gate(
            protocol,
            protocol_path=protocol_path,
            split_path=Path("/nonexistent/formal_split.json"),
            source_commit=protocol["source_binding"]["runtime_commit"],
            source_tree=protocol["source_binding"]["runtime_tree"],
            authorization={"status": "PASS"},
        )

    static = audit(
        protocol_path,
        source_commit=protocol["source_binding"]["runtime_commit"],
        source_tree=protocol["source_binding"]["runtime_tree"],
    )
    assert static["status"] == "FAIL_STATIC_CONTRACT"
    assert static["checks"]["formal_corridor_gate_bound"] is False
