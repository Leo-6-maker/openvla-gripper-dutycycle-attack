from __future__ import annotations

from gripper_attack.stage_v_causal_observation_snapshot import matched_action
from scripts.detector_v5.audit_stage_v_m3_5_v1_4_gate_b import _compliant
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_b import _pair_label, _physical_outcome, _validate_protocol


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


def test_gate_b_matched_contact_loss_uses_control_reference() -> None:
    control = {"status": "PASS", "state_restore_exact": True, "causal_input_binding_pass": True, "available_horizon_steps": 2, "rows": [{"post_contact_telemetry_valid": True, "post_object_gripper_contact": True, "post_object_position": [0.0, 0.0, 0.0]}, {"post_contact_telemetry_valid": True, "post_object_gripper_contact": True, "post_object_position": [0.0, 0.0, 0.0]}]}
    treatment = {"status": "PASS", "state_restore_exact": True, "causal_input_binding_pass": True, "available_horizon_steps": 2, "rows": [{"post_contact_telemetry_valid": True, "post_object_gripper_contact": False, "post_object_position": [0.0, 0.0, 0.0]}, {"post_contact_telemetry_valid": True, "post_object_gripper_contact": False, "post_object_position": [0.0, 0.0, 0.0]}]}
    assert _physical_outcome(treatment, required_steps=2, reference=control)["class"] == "GRIPPER_CONTACT_LOSS"


def test_gate_b_protocol_requires_per_parent_gate_a_binding() -> None:
    protocol = {
        "schema": "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_B",
        "version": "V1.4-GATE-B",
        "status": "FROZEN_RUNTIME_AUTHORIZED",
        "runtime_authorized": True,
        "source_binding": {"runtime_commit": "commit", "runtime_tree": "tree"},
        "operation": {"fresh_render_primary_consumption": "HARD_STOP", "fresh_render_equality_gate_used": False, "native_closed_loop_policy_in_primary_window": False},
        "matrix": {"repetitions": 3, "conditions": ["CONTROL", "T3", "T5", "T10"]},
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "requires": {"gate_a_status": "PASS", "gate_a_binding_mode": "PER_PARENT_EXACT_SHA256", "gate_a_bindings": {"parent": {}}},
    }
    _validate_protocol(protocol, type("Args", (), {"source_commit": "commit", "source_tree": "tree"})())
