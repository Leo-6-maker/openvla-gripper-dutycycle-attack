from gripper_attack.b3_retention import (
    OneShotAttackScheduler,
    RetentionConfig,
    RetentionEventTracker,
    canonical_opening_abs_sum,
    canonical_qpos_sum,
    rebuild_retention_features,
)


def _record(step, close, x, *, valid=True):
    return {
        "step": step,
        "valid": valid,
        "raw_close": close,
        "gripper_qpos": 0.10 if close else 0.40,
        "gripper_opening_proxy": 0.10 if close else 0.40,
        "eef_x": x,
        "eef_y": 0.0,
        "eef_z": 0.3,
    }


def test_tracker_reopens_a_second_event_after_hysteresis_release():
    records = []
    step = 0
    for close, count in ((False, 3), (True, 6), (False, 3), (True, 6), (False, 3)):
        for _ in range(count):
            records.append(_record(step, close, step * 0.01))
            step += 1
    tracker = RetentionEventTracker(RetentionConfig(n_close=3, n_open=3))
    outputs = [tracker.update(record) for record in records]
    events = tracker.finish()
    assert [(event.event_id, event.start_step, event.release_step) for event in events] == [
        (0, 3, 11),
        (1, 12, 20),
    ]
    assert outputs[11]["event_id"] == -1
    assert outputs[11]["event_active"] is False
    assert outputs[11]["released_event_id"] == 0


def test_rebuilder_masks_incomplete_t10_instead_of_calling_it_negative():
    records = [_record(step, True, step * 0.01) for step in range(8)]
    result = rebuild_retention_features(records, RetentionConfig(n_close=3, n_open=3))
    rows = result["rows"]
    assert rows[-1]["retention_continuation_t10"] is None
    assert rows[-1]["retention_unknown_mask"] is True


def test_missing_robot_evidence_is_unknown_not_negative():
    records = [_record(step, True, step * 0.01) for step in range(12)]
    records[7].pop("gripper_qpos")
    result = rebuild_retention_features(records, RetentionConfig(n_close=3, n_open=3))
    assert result["rows"][0]["retention_continuation_t10"] is None
    assert result["rows"][0]["retention_unknown_mask"] is True


def test_rebuilder_keeps_event_ids_separate_and_rebuilds_onset():
    records = []
    step = 0
    for close, count in ((False, 3), (True, 6), (False, 3), (True, 6)):
        for _ in range(count):
            records.append(_record(step, close, step * 0.01))
            step += 1
    result = rebuild_retention_features(records, RetentionConfig(n_close=3, n_open=3))
    onset_steps = [row["step"] for row in result["rows"] if row["event_close_onset"]]
    assert onset_steps == [3, 12]
    assert {event["event_id"] for event in result["events"]} == {0, 1}


def test_scheduler_is_one_shot_but_waits_across_events():
    scheduler = OneShotAttackScheduler(persistence=2)
    scheduler.update(step=0, event_id=-1, event_active=False, p_retention=0.9, p_t10=0.9, p_release=0.1)
    scheduler.update(step=1, event_id=1, p_retention=0.9, p_t10=0.9, p_release=0.1)
    decision = scheduler.update(step=2, event_id=1, p_retention=0.9, p_t10=0.9, p_release=0.1)
    assert decision["state"] == "ATTACKING_T10"
    assert decision["emit_event_id"] == 1
    done = None
    for step in range(3, 13):
        done = scheduler.update(step=step, event_id=1, p_retention=0.0, p_t10=0.0, p_release=1.0)
    assert done is not None
    assert done["state"] == "DONE"
    assert done["attacked_frames_emitted"] == 10


def test_scheduler_uses_true_two_of_three_not_only_consecutive_true():
    scheduler = OneShotAttackScheduler()
    high = {"p_retention": 0.9, "p_t10": 0.9, "p_release": 0.1}
    low = {"p_retention": 0.1, "p_t10": 0.1, "p_release": 0.9}
    scheduler.update(step=0, event_id=0, **high)
    scheduler.update(step=1, event_id=0, **low)
    decision = scheduler.update(step=2, event_id=0, **high)
    assert decision["trigger_started"] is True
    assert decision["attack_index"] == 0
    assert decision["gate_history"] == [True, False, True]


def test_scheduler_has_exactly_ten_active_frames():
    scheduler = OneShotAttackScheduler()
    active = []
    for step in range(12):
        decision = scheduler.update(
            step=step,
            event_id=0,
            p_retention=0.9,
            p_t10=0.9,
            p_release=0.1,
        )
        if decision["attack_active"]:
            active.append((decision["step"], decision["attack_index"], decision["attacked_frames_emitted"]))
    assert len(active) == 10
    assert [row[1] for row in active] == list(range(10))
    assert active[-1][2] == 10


def test_raw_and_env_gripper_semantics_are_not_interchangeable():
    tracker = RetentionEventTracker()
    close = {"step": 0, "action_raw_7d": [0.2], "applied_action_7d": [1.0]}
    assert tracker.update(close)["valid"] is True
    mismatch = {"step": 1, "action_raw_7d": [0.2], "applied_action_7d": [-1.0]}
    try:
        tracker.update(mismatch)
    except ValueError as exc:
        assert "raw/env gripper semantics mismatch" in str(exc)
    else:
        raise AssertionError("expected raw/env semantic mismatch")


def test_qpos_and_opening_use_official_parity_functions():
    record = {
        "robot0_gripper_qpos": [-0.1, 0.2],
        "gripper_qpos": 0.1,
        "gripper_opening_proxy": 0.3,
    }
    assert canonical_qpos_sum(record) == 0.1
    assert canonical_opening_abs_sum(record) == 0.3
    bad = dict(record, gripper_qpos=0.2)
    try:
        canonical_qpos_sum(bad)
    except ValueError as exc:
        assert "qpos parity mismatch" in str(exc)
    else:
        raise AssertionError("expected qpos parity failure")


def test_four_head_labels_and_masks_are_present():
    records = [_record(step, True, step * 0.01) for step in range(16)]
    rows = rebuild_retention_features(records, RetentionConfig(n_close=3, n_open=3))["rows"]
    for name in (
        "grasp_support",
        "grasp_support_mask",
        "retention_active",
        "retention_active_mask",
        "retention_continuation_t10",
        "retention_unknown_mask",
        "release_imminent",
        "release_imminent_mask",
    ):
        assert name in rows[0]
    assert rows[0]["teacher_label_version"] == "RETENTION_WEAK_TEACHER_V1"


def test_invalid_step_is_not_silently_reordered():
    tracker = RetentionEventTracker()
    tracker.update(_record(0, False, 0.0))
    try:
        tracker.update(_record(2, True, 0.02))
    except ValueError as exc:
        assert "non-contiguous step" in str(exc)
    else:
        raise AssertionError("expected non-contiguous step failure")
