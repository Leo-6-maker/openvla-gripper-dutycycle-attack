from detector.audit_b3_teacher_invariants import audit_episode


def _row(step: int, *, t10=None, t10_unknown=False, release=False):
    return {
        "step": step,
        "valid": True,
        "event_evidence_valid": True,
        "event_id": 0,
        "event_ordinal": 0,
        "event_support": True,
        "grasp_support": True,
        "grasp_support_mask": True,
        "retention_active": True,
        "retention_active_mask": True,
        "retention_continuation_t10": t10,
        "retention_unknown_mask": t10_unknown,
        "release_imminent": release if not t10_unknown else None,
        "release_imminent_mask": not t10_unknown,
        "event_release_onset": False,
        "released_event_id": -1,
    }


def test_t10_positive_checks_future_evidence_not_future_t10_labels():
    rows = [_row(step, t10=None if step >= 3 else step == 0, t10_unknown=step >= 3) for step in range(12)]
    report = audit_episode(rows, [{"event_id": 0, "start_step": 0, "end_step": 11}])
    assert report["status"] == "PASS"
    assert report["t10_positive_count"] == 1
    assert report["unknown_t10_count"] == 9


def test_missing_teacher_head_mask_is_a_hard_invariant_failure():
    rows = [_row(step, t10_unknown=True) for step in range(10)]
    rows[0].pop("retention_active_mask")
    report = audit_episode(rows)
    assert report["status"] == "HOLD"
    assert "STEP_0_retention_active_mask_NOT_BOOLEAN" in report["violations"]
