from __future__ import annotations

from pathlib import Path

from gripper_attack.stage_v_causal_observation_snapshot import matched_action
from scripts.detector_v5.audit_stage_v_m3_5_v1_4_gate_b import _compliant
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_b import HORIZON_CONTRACT, _pair_label, _physical_outcome, _validate_protocol


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _censor_branch(steps: int, *, compliant: bool, delivered: int) -> dict:
    rows = [{"post_contact_telemetry_valid": True, "post_object_gripper_contact": True, "post_object_position": [0.0, 0.0, 0.0]} for _ in range(steps)]
    return {
        "status": "PASS",
        "state_restore_exact": True,
        "causal_input_binding_pass": True,
        "available_horizon_steps": steps,
        "rows": rows,
        "treatment_compliant": compliant,
        "treatment_compliance": {"treatment_compliant": compliant, "delivered_open_steps": delivered},
    }


def test_gate_b_full_dose_with_short_follow_up_is_horizon_censored() -> None:
    pair = _pair_label(_censor_branch(15, compliant=True, delivered=0), _censor_branch(14, compliant=True, delivered=5), 5)
    assert pair["treatment_valid"] is False
    assert pair["censoring_class"] == "HORIZON_CENSORED_ABSTAIN"
    assert pair["label_class"] == "TREATMENT_INVALID_ABSTAIN"


def test_gate_b_incomplete_open_dose_is_treatment_invalid_censored() -> None:
    pair = _pair_label(_censor_branch(15, compliant=True, delivered=0), _censor_branch(4, compliant=False, delivered=4), 5)
    assert pair["censoring_class"] == "TREATMENT_INVALID_CENSORED_ABSTAIN"
    assert pair["label_class"] == "TREATMENT_INVALID_ABSTAIN"


def test_m4_runner_binds_censor_aware_contract_and_opt_in() -> None:
    runner = (REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_matched_parent.py").read_text(encoding="utf-8")
    assert HORIZON_CONTRACT in runner
    assert "allow_horizon_censoring=True" in runner


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
