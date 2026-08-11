from __future__ import annotations

from gripper_attack.stage_v_causal_observation_snapshot import matched_action
from scripts.detector_v5.audit_stage_v_m3_5_v1_4_gate_b import _compliant
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_b import _pair_label


def test_matched_open_keeps_arm_exact() -> None:
    reference = {
        "raw_policy_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2],
        "env_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
    }
    action = matched_action(reference, forced_open=True)
    assert action["raw_policy_action"][:6] == reference["raw_policy_action"][:6]
    assert action["env_action"][:6] == reference["env_action"][:6]
    assert action["env_action"][-1] == -1.0
    assert action["arm_delta_linf"] == 0.0


def test_gate_b_treatment_compliance_requires_registered_open_steps() -> None:
    branch = {
        "treatment_compliant": True,
        "treatment_compliance": {"delivered_open_steps": 2},
        "treatment_receipts": [
            {"raw_policy_action": [0.0] * 6 + [1.0], "normalized_action": [0.0] * 6 + [1.0], "env_action": [0.0] * 6 + [-1.0], "arm_delta_linf": 0.0},
            {"raw_policy_action": [0.0] * 6 + [1.0], "normalized_action": [0.0] * 6 + [1.0], "env_action": [0.0] * 6 + [-1.0], "arm_delta_linf": 0.0},
        ],
    }
    assert _compliant(branch, 2) is True
    assert _compliant(branch, 3) is False


def test_gate_b_pair_label_never_coerces_invalid_control_to_negative() -> None:
    control = {"status": "FAIL", "state_restore_exact": False, "causal_input_binding_pass": False, "available_horizon_steps": 0, "rows": []}
    treatment = {"status": "FAIL", "state_restore_exact": False, "causal_input_binding_pass": False, "available_horizon_steps": 0, "rows": [], "treatment_compliant": False, "treatment_compliance": {}}
    result = _pair_label(control, treatment, 5)
    assert result["label_class"].endswith("_ABSTAIN")
