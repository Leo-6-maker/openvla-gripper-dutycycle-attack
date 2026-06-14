"""CPU tests for Layer1/2 window contract."""

import copy

from gripper_attack.window_contract import (
    WindowProposal,
    validate_proposals,
)


def _valid_proposal(**overrides):
    kw = {
        "proposal_id": "test_p1",
        "selector_version": "l12_critical_close_selector_v1",
        "source_commit": "abc123",
        "source_trace_path": "/tmp/trace.csv",
        "task_key": "butter", "state_id": 2,
        "window_start": 70, "window_end": 80, "anchor_step": 78,
        "predicted_first_close_step": 78,
        "selector_score": 3.5,
        "eligible": True,
        "uses_clean_only": True,
        "uses_attack_outcome": False,
        "uses_random_outcome": False,
        "features_are_causal": True,
        "selection_is_causal": False,
        "selector_role": "student",
    }
    kw.update(overrides)
    return WindowProposal(**kw)


def test_valid_proposal():
    p = _valid_proposal()
    assert p.is_valid()
    assert p.validate() == []


def test_missing_proposal_id():
    p = _valid_proposal(proposal_id="")
    assert not p.is_valid()
    assert any("proposal_id" in i for i in p.validate())


def test_attack_outcome_leakage():
    p = _valid_proposal(uses_attack_outcome=True)
    assert not p.is_valid()
    assert any("attack_outcome" in i for i in p.validate())


def test_random_outcome_leakage():
    p = _valid_proposal(uses_random_outcome=True)
    assert not p.is_valid()
    assert any("random_outcome" in i for i in p.validate())


def test_student_with_privileged_state():
    p = _valid_proposal(selector_role="student", uses_privileged_state=True)
    assert not p.is_valid()
    assert any("student_uses_privileged" in i for i in p.validate())


def test_teacher_with_privileged_state_allowed():
    p = _valid_proposal(selector_role="teacher_only", uses_privileged_state=True)
    assert p.is_valid()


def test_window_order():
    p = _valid_proposal(window_start=80, window_end=70)
    assert not p.is_valid()
    assert any("WINDOW_END_NOT_GT_START" in i for i in p.validate())


def test_negative_window():
    p = _valid_proposal(window_start=-5)
    assert not p.is_valid()
    assert any("eligible_proposal:NEGATIVE_WINDOW_START" in i for i in p.validate())


def test_clean_only_false():
    p = _valid_proposal(uses_clean_only=False)
    assert not p.is_valid()


def test_validate_list_duplicate():
    p1 = _valid_proposal(proposal_id="dup")
    p2 = _valid_proposal(proposal_id="dup")
    issues, ok = validate_proposals([p1, p2])
    assert not ok
    assert any("DUPLICATE" in i for i in issues)


def test_validate_list_all_valid():
    p1 = _valid_proposal(proposal_id="a")
    p2 = _valid_proposal(proposal_id="b")
    issues, ok = validate_proposals([p1, p2])
    assert ok


def test_to_dict_roundtrip():
    p = _valid_proposal()
    d = p.to_dict()
    p2 = WindowProposal.from_dict(d)
    assert p2.proposal_id == p.proposal_id
    assert p2.uses_clean_only == p.uses_clean_only
    assert p2.is_valid()


def test_offline_mode_selection_not_causal_allowed():
    """Offline clean-repeat: features_are_causal=True, selection_is_causal=False is OK."""
    p = _valid_proposal(features_are_causal=True, selection_is_causal=False,
                        is_online=False, selection_mode="offline_clean_repeat")
    assert p.is_valid()


def test_online_mode_requires_selection_is_causal():
    """Online streaming must have selection_is_causal=True."""
    p = _valid_proposal(features_are_causal=True, selection_is_causal=False,
                        is_online=True, selection_mode="online_streaming")
    assert not p.is_valid()
    assert any("online_mode_requires_selection_is_causal" in i for i in p.validate())


def test_online_mode_valid_with_selection_causal():
    """Online streaming with selection_is_causal=True is valid."""
    p = _valid_proposal(features_are_causal=True, selection_is_causal=True,
                        is_online=True, selection_mode="online_streaming")
    assert p.is_valid()


def test_prediction_mode_roundtrip():
    """prediction_mode is preserved in to_dict/from_dict roundtrip."""
    p = _valid_proposal(prediction_mode="observed_close_interception")
    d = p.to_dict()
    p2 = WindowProposal.from_dict(d)
    assert p2.prediction_mode == "observed_close_interception"


def test_offline_mode_rejects_selection_is_causal_true():
    """offline_clean_repeat must NOT have selection_is_causal=True."""
    p = _valid_proposal(selection_mode="offline_clean_repeat", is_online=False,
                        selection_is_causal=True, features_are_causal=True)
    assert not p.is_valid()
    assert any("offline_mode_selection_is_causal" in i for i in p.validate())


def test_online_mode_rejects_is_online_false():
    """online_streaming must have is_online=True."""
    p = _valid_proposal(selection_mode="online_streaming", is_online=False,
                        features_are_causal=True, selection_is_causal=True)
    assert not p.is_valid()
    assert any("online_mode_is_online" in i for i in p.validate())


def test_online_mode_rejects_nonzero_horizon_for_interception():
    """observed_close_interception with non-zero horizon is invalid."""
    p = _valid_proposal(selection_mode="online_streaming", is_online=True,
                        features_are_causal=True, selection_is_causal=True,
                        prediction_mode="observed_close_interception",
                        first_close_horizon=4)
    assert not p.is_valid()
    assert any("interception_mode_horizon" in i for i in p.validate())


def test_future_forecast_rejected_until_implemented():
    """Any proposal claiming future_close_forecast must be rejected."""
    p = _valid_proposal(prediction_mode="future_close_forecast")
    assert not p.is_valid()
    assert any("future_close_forecast:NOT_IMPLEMENTED" in i for i in p.validate())

    p2 = _valid_proposal(selection_mode="online_streaming", is_online=True,
                         features_are_causal=True, selection_is_causal=True,
                         prediction_mode="future_close_forecast")
    assert not p2.is_valid()
    assert any("future_close_forecast:NOT_IMPLEMENTED" in i for i in p2.validate())


def test_selection_mode_and_is_online_must_agree():
    """offline with is_online=True is invalid."""
    p = _valid_proposal(selection_mode="offline_clean_repeat", is_online=True)
    assert not p.is_valid()
    assert any("offline_mode_is_online" in i for i in p.validate())


def test_unknown_prediction_mode_rejected():
    """Non-empty unknown prediction_mode must be caught."""
    p = _valid_proposal(prediction_mode="some_future_v3")
    assert not p.is_valid()
    assert any("prediction_mode:UNKNOWN" in i for i in p.validate())


def test_abstain_window_proposal_is_contract_valid():
    """Legitimate abstention (eligible=False, abstain_reason, negative window) is valid."""
    p = _valid_proposal(
        eligible=False,
        abstain_reason="no_online_trigger",
        window_start=-1,
        window_end=-1,
        anchor_step=-1,
        predicted_first_close_step=-1,
        is_online=True,
        selection_mode="online_streaming",
        features_are_causal=True,
        selection_is_causal=True,
        prediction_mode="",
    )
    assert p.is_valid(), f"Abstain proposal should be valid: {p.validate()}"


def test_negative_window_without_reason_is_invalid():
    """Negative window without abstain_reason is invalid."""
    p = _valid_proposal(
        eligible=False,
        abstain_reason="",
        window_start=-1,
    )
    assert not p.is_valid()
    assert any("abstain_proposal:MISSING_REASON" in i for i in p.validate())


def test_eligible_with_negative_window_is_invalid():
    """Eligible=True with negative window is a contract violation."""
    p = _valid_proposal(
        eligible=True,
        window_start=-1,
    )
    assert not p.is_valid()
    assert any("eligible_proposal:NEGATIVE_WINDOW_START" in i for i in p.validate())


def test_eligible_negative_window_with_reason_is_invalid():
    """Eligible=True with negative window AND abstain_reason still invalid."""
    p = _valid_proposal(
        eligible=True,
        window_start=-1,
        abstain_reason="no_online_trigger",
    )
    assert not p.is_valid()
    assert any("eligible_proposal:NEGATIVE_WINDOW_START" in i for i in p.validate())


def test_abstain_sentinel_mismatched_anchor_is_invalid():
    """Abstain sentinel with anchor_step != -1 is a violation."""
    p = _valid_proposal(
        eligible=False,
        abstain_reason="all_abstain",
        window_start=-1,
        window_end=-1,
        anchor_step=5,  # should be -1
        predicted_first_close_step=-1,
    )
    assert not p.is_valid()
    assert any("abstain_proposal:ANCHOR_NOT_NEG1" in i for i in p.validate())


def test_partial_negative_abstain_sentinel_is_invalid():
    """Abstain with window_start=-1 but window_end=5 → hybrid, invalid."""
    p = _valid_proposal(
        eligible=False,
        abstain_reason="no_online_trigger",
        window_start=-1,
        window_end=5,  # not -1
        anchor_step=12,
        predicted_first_close_step=-1,
    )
    assert not p.is_valid()


def test_abstain_with_positive_window_is_invalid():
    """Abstain with positive window → hybrid, invalid."""
    p = _valid_proposal(
        eligible=False,
        abstain_reason="all_abstain",
        window_start=10,
        window_end=20,
        anchor_step=15,
    )
    assert not p.is_valid()


def test_eligible_with_abstain_reason_is_invalid():
    """Eligible=True with non-empty abstain_reason → invalid hybrid."""
    p = _valid_proposal(
        eligible=True,
        abstain_reason="some_reason",
    )
    assert not p.is_valid()
    assert any("eligible_proposal:HAS_ABSTAIN_REASON" in i for i in p.validate())


def test_eligible_with_negative_anchor_is_invalid():
    """Eligible=True with anchor_step=-1 → invalid."""
    p = _valid_proposal(
        eligible=True,
        anchor_step=-1,
    )
    assert not p.is_valid()
    assert any("eligible_proposal:NEGATIVE_ANCHOR" in i for i in p.validate())


def test_feature_matrix_distinguishes_missing_from_valid_zero():
    """extract_deployment_features returns validity mask for missing fields."""
    from gripper_attack.critical_close_selector import extract_deployment_features
    # Build minimal trace locally
    records = []
    for t in range(50):
        rec = {
            "step": str(t),
            "clean_gripper_env": "1.0",
            "clean_gripper_raw": "0.7",
            "gripper_qpos_before": "0.0",
            "qpos_abs_before": "0.0",
            "eef_x": "0.0", "eef_y": "0.0", "eef_z": "0.2",
            "close_streak": "0",
            "decoded_open_bool": "0",
        }
        records.append(rec)
    records[10]["gripper_qpos_before"] = ""
    records[20]["eef_x"] = ""
    feats, validity = extract_deployment_features(records)
    assert not validity[10, 2], "Missing qpos should be flagged invalid"
    assert not validity[20, 4], "Missing eef_x should be flagged invalid"
    assert validity[30, 2], "Present qpos should be flagged valid"
    assert validity[30, 4], "Present eef_x should be flagged valid"


def test_missing_qpos_abs_with_valid_nonzero_qpos_not_fabricated():
    """When qpos_abs is missing but qpos is valid nonzero, don't fabricate."""
    from gripper_attack.critical_close_selector import extract_deployment_features
    records = []
    for t in range(10):
        rec = {
            "step": str(t),
            "clean_gripper_env": "1.0",
            "clean_gripper_raw": "0.7",
            "gripper_qpos_before": "0.03",  # valid nonzero
            "qpos_abs_before": "",  # missing
            "eef_x": "0.0", "eef_y": "0.0", "eef_z": "0.2",
            "close_streak": "0",
            "decoded_open_bool": "0",
        }
        records.append(rec)
    feats, validity = extract_deployment_features(records)
    # qpos_abs validity derives from qpos_before validity (same source)
    assert validity[0, 3], "qpos_abs should be valid (derived from valid qpos)"
    assert abs(feats[0, 3] - 0.03) < 1e-5, f"qpos_abs should be ~0.03, got {feats[0, 3]}"
