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
        "is_causal": True,
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
    assert any("window_end" in i for i in p.validate())


def test_negative_window():
    p = _valid_proposal(window_start=-5)
    assert not p.is_valid()
    assert any("NEGATIVE" in i for i in p.validate())


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
