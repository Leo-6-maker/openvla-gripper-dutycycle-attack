"""Offline-only robot-centric retention event reconstruction.

This module intentionally does not alter the Official CLEAN collector.  It
rebuilds event-local features from immutable raw step records and keeps the
event tracker separate from the one-shot attack scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Iterable


T10 = 10


@dataclass(frozen=True)
class RetentionConfig:
    n_close: int = 3
    n_open: int = 3
    stability_window: int = 3
    qpos_range_max: float = 0.05
    opening_range_max: float = 0.05
    min_transport_steps: int = 3
    min_transport_displacement: float = 0.002
    t10: int = T10


@dataclass
class RetentionEvent:
    event_id: int
    start_step: int
    end_step: int | None = None
    release_step: int | None = None
    closed_by: str = ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _scalar(record: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(record.get(name))
        if value is not None:
            return value
    return None


def _last_action_value(record: dict[str, Any]) -> float | None:
    for name in ("action_raw_7d", "action_raw", "applied_action_7d", "action_env"):
        value = record.get(name)
        if isinstance(value, (list, tuple)) and value:
            number = _number(value[-1])
            if number is not None:
                return number
    return _scalar(record, "action_gripper", "gripper_command", "raw_gripper")


def _raw_close(record: dict[str, Any]) -> bool:
    explicit = record.get("raw_close")
    if isinstance(explicit, bool):
        return explicit
    value = _last_action_value(record)
    if value is None:
        raise ValueError("missing raw gripper/action value")
    # Frozen official OpenVLA raw-gripper convention.
    return value <= 0.5


def _eef(record: dict[str, Any]) -> tuple[float, float, float] | None:
    value = record.get("robot0_eef_pos")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        xyz = tuple(_number(x) for x in value[:3])
        if all(x is not None for x in xyz):
            return tuple(float(x) for x in xyz)  # type: ignore[arg-type]
    xyz = tuple(_scalar(record, name) for name in ("eef_x", "eef_y", "eef_z"))
    if all(x is not None for x in xyz):
        return tuple(float(x) for x in xyz)  # type: ignore[arg-type]
    return None


def _qpos(record: dict[str, Any]) -> float | None:
    value = record.get("robot0_gripper_qpos")
    if isinstance(value, (list, tuple)) and value:
        numbers = [_number(x) for x in value]
        if all(x is not None for x in numbers):
            return float(sum(abs(float(x)) for x in numbers))
    return _scalar(record, "gripper_qpos")


def _opening(record: dict[str, Any]) -> float | None:
    return _scalar(record, "gripper_opening_proxy", "gripper_width", "opening_proxy")


def _step(record: dict[str, Any], fallback: int) -> int:
    value = record.get("step", record.get("step_idx", fallback))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid step: {value!r}") from exc


class RetentionEventTracker:
    """Causal hysteresis tracker for multiple robot-centric events."""

    def __init__(self, config: RetentionConfig = RetentionConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.events: list[RetentionEvent] = []
        self.current: RetentionEvent | None = None
        self._close_streak = 0
        self._open_streak = 0
        self._seen_release = False
        self._last_step = -1

    def update(self, record: dict[str, Any]) -> dict[str, Any]:
        step = _step(record, self._last_step + 1)
        if step != self._last_step + 1:
            raise ValueError(f"non-contiguous step: expected {self._last_step + 1}, got {step}")
        self._last_step = step
        valid = record.get("valid", True) is not False
        if not valid:
            self._close_streak = self._open_streak = 0
            return self._row(step, valid=False, close_onset=False, release_onset=False)

        close = _raw_close(record)
        if close:
            self._close_streak += 1
            self._open_streak = 0
        else:
            self._open_streak += 1
            self._close_streak = 0

        close_onset = False
        release_onset = False
        if self.current is None and close and self._close_streak == self.config.n_close:
            start = step - self.config.n_close + 1
            self.current = RetentionEvent(len(self.events), start_step=start)
            close_onset = True

        if self.current is not None and not close and self._open_streak == self.config.n_open:
            self.current.end_step = step
            self.current.release_step = step
            self.current.closed_by = "HYSTERESIS_RELEASE"
            self.events.append(self.current)
            self.current = None
            self._seen_release = True
            release_onset = True

        event_id = self.current.event_id if self.current is not None else (
            self.events[-1].event_id if self.events else -1
        )
        return self._row(step, valid=True, close_onset=close_onset, release_onset=release_onset, event_id=event_id)

    def finish(self) -> list[RetentionEvent]:
        if self.current is not None:
            self.current.end_step = self._last_step
            self.current.closed_by = "EPISODE_END"
            self.events.append(self.current)
            self.current = None
        return [RetentionEvent(**event.__dict__) for event in self.events]

    def _row(self, step: int, *, valid: bool, close_onset: bool, release_onset: bool, event_id: int = -1) -> dict[str, Any]:
        return {
            "step": step,
            "valid": valid,
            "event_id": event_id,
            "event_ordinal": event_id,
            "event_close_onset": close_onset,
            "event_release_onset": release_onset,
            "close_streak_local": self._close_streak,
            "open_streak_local": self._open_streak,
        }


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return hypot(hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2])


def rebuild_retention_features(records: Iterable[dict[str, Any]], config: RetentionConfig = RetentionConfig()) -> dict[str, Any]:
    """Rebuild event-local causal features and masked future labels offline."""
    raw = list(records)
    if not raw:
        return {"schema": "B3_RETENTION_DERIVED_FEATURES_V1", "rows": [], "events": []}
    if [_step(row, i) for i, row in enumerate(raw)] != list(range(len(raw))):
        raise ValueError("records must contain contiguous episode steps starting at zero")

    tracker = RetentionEventTracker(config)
    rows = [tracker.update(row) for row in raw]
    events = tracker.finish()
    by_id = {event.event_id: event for event in events}

    # The causal tracker can only confirm an event after hysteresis.  Backfill
    # the event identity over the confirmed close streak so the derived stream
    # represents the full event, not only the confirmation suffix.
    for row in rows:
        row["event_id"] = -1
        row["event_ordinal"] = -1
        for event in events:
            end = event.end_step if event.end_step is not None else row["step"]
            if event.start_step <= row["step"] <= end:
                row["event_id"] = event.event_id
                row["event_ordinal"] = event.event_id
                break

    for row, record in zip(rows, raw):
        source_valid = (
            row["valid"]
            and _eef(record) is not None
            and _qpos(record) is not None
            and _opening(record) is not None
        )
        row["event_evidence_valid"] = bool(source_valid)
        event = by_id.get(row["event_id"])
        if event is None:
            row.update({
                "event_start_step": -1,
                "event_end_step": -1,
                "event_path_length_since_close": 0.0,
                "event_eef_displacement_since_close": 0.0,
                "event_support": False,
            })
            continue
        start = event.start_step
        current = row["step"]
        path = 0.0
        first_eef = _eef(raw[start])
        prev_eef = first_eef
        for index in range(start + 1, current + 1):
            eef = _eef(raw[index])
            if eef is None or prev_eef is None:
                path = float("nan")
                break
            path += _distance(prev_eef, eef)
            prev_eef = eef
        current_eef = _eef(record)
        displacement = float("nan") if first_eef is None or current_eef is None else _distance(first_eef, current_eef)
        q_values = [_qpos(raw[index]) for index in range(max(start, current - config.stability_window + 1), current + 1)]
        o_values = [_opening(raw[index]) for index in range(max(start, current - config.stability_window + 1), current + 1)]
        q_known = all(value is not None for value in q_values)
        o_known = all(value is not None for value in o_values)
        q_stable = q_known and max(q_values) - min(q_values) <= config.qpos_range_max
        o_stable = o_known and max(o_values) - min(o_values) <= config.opening_range_max
        moving = (
            current - start + 1 >= config.min_transport_steps
            and isfinite(displacement)
            and displacement >= config.min_transport_displacement
        )
        row.update({
            "event_start_step": start,
            "event_end_step": event.end_step if event.end_step is not None else -1,
            "event_path_length_since_close": path,
            "event_eef_displacement_since_close": displacement,
            "event_qpos_stable": q_stable,
            "event_opening_stable": o_stable,
            "event_evidence_valid": bool(source_valid and q_known and o_known),
            "event_support": bool(_raw_close(record) and q_stable and o_stable and moving),
        })

    for index, row in enumerate(rows):
        future = rows[index:index + config.t10]
        if len(future) < config.t10 or any(
            not item["valid"] or not item.get("event_evidence_valid", False)
            for item in future
        ):
            row["retention_continuation_t10"] = None
            row["retention_unknown_mask"] = True
            continue
        same_event = row["event_id"] >= 0 and all(item["event_id"] == row["event_id"] for item in future)
        row["retention_continuation_t10"] = bool(same_event and all(item["event_support"] for item in future))
        row["retention_unknown_mask"] = False

    for event in events:
        event_rows = [row for row in rows if row["event_id"] == event.event_id]
        if event_rows:
            event.start_step = min(row["event_start_step"] for row in event_rows)
            for row in event_rows:
                row["event_close_onset"] = row["step"] == event.start_step
                row["event_release_onset"] = event.release_step == row["step"]

    return {
        "schema": "B3_RETENTION_DERIVED_FEATURES_V1",
        "source_schema": "OFFICIAL_25D_V1",
        "rows": rows,
        "events": [event.__dict__.copy() for event in events],
    }


class OneShotAttackScheduler:
    """Runtime-only gate; it does not segment events or train the detector."""

    def __init__(self, tau_retention: float = 0.7, tau_t10: float = 0.7, tau_release: float = 0.3, persistence: int = 2, t10: int = T10):
        self.tau_retention = tau_retention
        self.tau_t10 = tau_t10
        self.tau_release = tau_release
        self.persistence = persistence
        self.t10 = t10
        self.reset()

    def reset(self) -> None:
        self.state = "ARMED"
        self.emit_step = -1
        self.emit_event_id = -1
        self._candidate_event_id = -1
        self._candidate_streak = 0

    def update(self, *, step: int, event_id: int, p_retention: float, p_t10: float, p_release: float) -> dict[str, Any]:
        if self.state == "DONE":
            return self._decision(step, emitted=False)
        if self.state == "ATTACKING_T10":
            if step >= self.emit_step + self.t10 - 1:
                self.state = "DONE"
            return self._decision(step, emitted=False)
        gate = event_id >= 0 and p_retention >= self.tau_retention and p_t10 >= self.tau_t10 and p_release < self.tau_release
        if gate and event_id == self._candidate_event_id:
            self._candidate_streak += 1
        elif gate:
            self._candidate_event_id = event_id
            self._candidate_streak = 1
        else:
            self._candidate_event_id = -1
            self._candidate_streak = 0
        if self._candidate_streak >= self.persistence:
            self.state = "ATTACKING_T10"
            self.emit_step = step
            self.emit_event_id = event_id
        return self._decision(step, emitted=self.state == "ATTACKING_T10")

    def _decision(self, step: int, *, emitted: bool) -> dict[str, Any]:
        return {
            "state": self.state,
            "step": step,
            "emit_step": self.emit_step,
            "emit_event_id": self.emit_event_id,
            "emitted": emitted,
            "candidate_streak": self._candidate_streak,
        }


__all__ = [
    "OneShotAttackScheduler",
    "RetentionConfig",
    "RetentionEventTracker",
    "rebuild_retention_features",
]
