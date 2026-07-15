from gripper_attack.b3_retention import (
    OneShotAttackScheduler,
    RetentionConfig,
    RetentionEventTracker,
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
    for record in records:
        tracker.update(record)
    events = tracker.finish()
    assert [(event.event_id, event.start_step, event.release_step) for event in events] == [
        (0, 3, 11),
        (1, 12, 20),
    ]


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
    scheduler.update(step=4, event_id=0, p_retention=0.9, p_t10=0.2, p_release=0.1)
    scheduler.update(step=12, event_id=1, p_retention=0.9, p_t10=0.9, p_release=0.1)
    decision = scheduler.update(step=13, event_id=1, p_retention=0.9, p_t10=0.9, p_release=0.1)
    assert decision["state"] == "ATTACKING_T10"
    assert decision["emit_event_id"] == 1
    done = scheduler.update(step=22, event_id=1, p_retention=0.0, p_t10=0.0, p_release=1.0)
    assert done["state"] == "DONE"


def test_invalid_step_is_not_silently_reordered():
    tracker = RetentionEventTracker()
    tracker.update(_record(0, False, 0.0))
    try:
        tracker.update(_record(2, True, 0.02))
    except ValueError as exc:
        assert "non-contiguous step" in str(exc)
    else:
        raise AssertionError("expected non-contiguous step failure")
