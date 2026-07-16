"""Contract tests for B3_CAUSAL_25D_MULTIEVENT_V1."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gripper_attack.b3_causal_25d import (  # noqa: E402
    B3Causal25DMultieventV1,
    Causal25DConfig,
    FEATURE_NAMES,
    LEGACY_SOURCE_FEATURE_NAMES_25D,
    serialize_student_25d,
)


def _record(step: int, close: bool, *, valid: bool = True) -> dict:
    raw = 0.2 if close else 0.8
    env = 1.0 if close else -1.0
    return {
        "step": step,
        "valid": valid,
        "raw_close": close,
        "raw_gripper": raw,
        "env_gripper": env,
        "action_raw": [0.01, 0.0, 0.001, 0.0, 0.0, 0.0, raw],
        "action_env": [0.01, 0.0, 0.001, 0.0, 0.0, 0.0, env],
        "gripper_command": raw,
        "gripper_qpos": 0.10 if close else 0.40,
        "gripper_opening_proxy": 0.10 if close else 0.40,
        "eef_x": step * 0.01,
        "eef_y": 0.0,
        "eef_z": 0.25 + step * 0.001,
        "eef_vx": 0.01,
        "eef_vy": 0.0,
        "eef_vz": 0.001,
        "action_dx": 0.01,
        "action_dy": 0.0,
        "action_dz": 0.001,
        "action_gripper": raw,
    }


def test_two_close_release_events_reset_event_local_state():
    records = []
    step = 0
    for close, count in ((False, 3), (True, 5), (False, 3), (True, 5)):
        for _ in range(count):
            records.append(_record(step, close))
            step += 1
    result = B3Causal25DMultieventV1().rebuild(records)
    rows = result["rows"]
    assert [row["step"] for row in rows if row["close_onset"]] == [3, 11]
    assert rows[3]["features_25d"][FEATURE_NAMES.index("time_since_close")] == 0.0
    assert rows[4]["features_25d"][FEATURE_NAMES.index("time_since_close")] == 1.0
    assert rows[10]["event_local_state_reset"] is True
    assert rows[10]["event_active"] is False
    assert rows[10]["event_id"] == -1
    assert rows[10]["features_25d"][FEATURE_NAMES.index("time_since_close")] == -1.0
    assert rows[11]["features_25d"][FEATURE_NAMES.index("eef_z_delta_since_close")] == 0.0
    assert len(result["events"]) == 2


def test_flip_count_is_rolling_not_episode_cumulative():
    records = [_record(step, step % 2 == 0) for step in range(24)]
    records.extend(_record(step, False) for step in range(24, 44))
    rows = B3Causal25DMultieventV1().rebuild(records)["rows"]
    flip_index = FEATURE_NAMES.index("recent_gripper_flip_count")
    assert max(row["features_25d"][flip_index] for row in rows) <= 16.0
    assert rows[-1]["features_25d"][flip_index] == 0.0


def test_invalid_step_does_not_change_event_boundary():
    adapter = B3Causal25DMultieventV1()
    first = adapter.update(_record(0, True))
    invalid = adapter.update({"step": 1, "valid": False})
    third = adapter.update(_record(2, True))
    assert first["close_onset"] is True
    assert invalid["valid"] is False
    assert invalid["event_active"] is True
    assert third["event_active"] is True
    assert third["event_id"] == first["event_id"]


def test_l10_length_and_later_events_are_finite():
    records = []
    for step in range(520):
        phase = step % 10
        records.append(_record(step, close=3 <= phase < 7))
    result = B3Causal25DMultieventV1(Causal25DConfig(n_open=3)).rebuild(records)
    assert len(result["rows"]) == 520
    assert len(result["events"]) == 52
    for row in result["rows"]:
        assert row["valid"] is True
        assert len(row["features_25d"]) == 25
        assert all(math.isfinite(value) for value in row["features_25d"])
    assert any(row["event_id"] >= 1 for row in result["rows"])


def test_student_vector_excludes_event_identity_and_teacher_fields():
    result = B3Causal25DMultieventV1().rebuild([_record(0, False), _record(1, True)])
    assert len(result["rows"][0]["features_25d"]) == 25
    assert "event_id" not in FEATURE_NAMES
    assert "event_ordinal" not in FEATURE_NAMES
    assert "teacher_label" not in FEATURE_NAMES


def test_three_events_have_contiguous_ids():
    records = []
    step = 0
    for _ in range(3):
        for close, count in ((False, 3), (True, 3), (False, 3)):
            for _ in range(count):
                records.append(_record(step, close))
                step += 1
    result = B3Causal25DMultieventV1().rebuild(records)
    assert [event["event_id"] for event in result["events"]] == [0, 1, 2]


def test_raw_close_cannot_synthesize_actions():
    row = B3Causal25DMultieventV1().update({"step": 0, "raw_close": True})
    assert row["valid"] is False
    assert row["features_25d"] is None


def test_feature_vector_order_length_and_named_parity_are_bound():
    record = _record(0, False)
    vector = [0.0] * 25
    for index, name in enumerate(LEGACY_SOURCE_FEATURE_NAMES_25D[:13]):
        vector[index] = record[name]
    record["features_25d"] = vector
    record["feature_names_25d"] = list(LEGACY_SOURCE_FEATURE_NAMES_25D)
    record["feature_order_sha256"] = "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366"
    assert B3Causal25DMultieventV1().update(record)["valid"] is True

    bad_length = dict(record, features_25d=vector[:-1])
    assert B3Causal25DMultieventV1().update(bad_length)["valid"] is False
    bad_order = dict(record, feature_names_25d=list(reversed(LEGACY_SOURCE_FEATURE_NAMES_25D)))
    assert B3Causal25DMultieventV1().update(bad_order)["valid"] is False
    bad_hash = dict(record, feature_order_sha256="0" * 64)
    assert B3Causal25DMultieventV1().update(bad_hash)["valid"] is False
    bad_named = dict(record, features_25d=[1.0] + vector[1:])
    assert B3Causal25DMultieventV1().update(bad_named)["valid"] is False


def test_robot_sidecar_parity_is_fail_closed():
    record = _record(0, False)
    record["robot0_gripper_qpos"] = [0.9, 0.9]
    row = B3Causal25DMultieventV1().update(record)
    assert row["valid"] is False


def test_raw_and_env_arm_vectors_must_match():
    record = _record(0, False)
    record["action_env"] = [0.02, 0.0, 0.001, 0.0, 0.0, 0.0, -1.0]
    row = B3Causal25DMultieventV1().update(record)
    assert row["valid"] is False


def test_student_serializer_rejects_side_channels():
    adapter = B3Causal25DMultieventV1()
    row = adapter.update(_record(0, False))
    student = {key: row[key] for key in ("schema", "source_schema", "valid", "features_25d")}
    assert len(serialize_student_25d(student)) == 25
    try:
        serialize_student_25d(row)
    except ValueError:
        pass
    else:
        raise AssertionError("student serializer accepted event side channels")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
    print("b3-causal-25d-multievent: PASS")
