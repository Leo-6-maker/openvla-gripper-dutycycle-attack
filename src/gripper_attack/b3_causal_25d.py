"""Offline reconstruction of event-local causal 25D features.

This is deliberately separate from ``SC5StreamingFeatureAdapterV2``.  The
legacy adapter remains unchanged for provenance; this version resets event
local state after hysteresis release and uses a rolling flip window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import pvariance
from typing import Any, Iterable


SCHEMA = "B3_CAUSAL_25D_MULTIEVENT_V1"
SOURCE_SCHEMA = "OFFICIAL_25D_V1"
FEATURE_NAMES = (
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
)


@dataclass(frozen=True)
class Causal25DConfig:
    n_open: int = 3
    rolling_flip_window: int = 16

    def __post_init__(self) -> None:
        if self.n_open < 1 or self.rolling_flip_window < 1:
            raise ValueError("n_open and rolling_flip_window must be positive")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _last_vector(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, (list, tuple)) and value:
            number = _number(value[-1])
            if number is not None:
                return number
    return None


def _scalar(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        number = _number(record.get(name))
        if number is not None:
            return number
    return None


def _feature_value(record: dict[str, Any], index: int, names: tuple[str, ...]) -> float | None:
    vector = record.get("features_25d")
    if isinstance(vector, (list, tuple)) and len(vector) >= 13:
        number = _number(vector[index])
        if number is not None:
            return number
    return _scalar(record, names)


def _normalise_record(record: dict[str, Any], fallback_step: int) -> dict[str, Any]:
    step_value = record.get("step", record.get("step_idx", fallback_step))
    try:
        step = int(step_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid step: {step_value!r}") from exc

    raw = _last_vector(record, ("clean_action_raw_7d", "action_raw_7d", "action_raw"))
    env = _last_vector(record, ("applied_action_7d", "action_env"))
    explicit_close = record.get("raw_close")
    if raw is None:
        raw = _scalar(record, ("raw_gripper", "gripper_command"))
    if env is None:
        env = _scalar(record, ("env_gripper",))
    if raw is None and isinstance(explicit_close, bool):
        raw = 0.0 if explicit_close else 1.0
    if env is None and isinstance(explicit_close, bool):
        env = 1.0 if explicit_close else -1.0
    if raw is None or env is None:
        raise ValueError("missing raw/env gripper action")

    raw_close = raw <= 0.5
    env_close = env > 0.0
    if raw_close != env_close:
        raise ValueError("raw/env gripper semantics mismatch")
    if isinstance(explicit_close, bool) and explicit_close != raw_close:
        raise ValueError("raw_close contradicts raw action")

    direct_specs = (
        (0, ("gripper_command",)),
        (1, ("gripper_qpos",)),
        (2, ("gripper_opening_proxy", "gripper_width", "opening_proxy")),
        (3, ("eef_x",)), (4, ("eef_y",)), (5, ("eef_z",)),
        (6, ("eef_vx",)), (7, ("eef_vy",)), (8, ("eef_vz",)),
        (9, ("action_dx",)), (10, ("action_dy",)), (11, ("action_dz",)),
        (12, ("action_gripper",)),
    )
    direct = [_feature_value(record, index, names) for index, names in direct_specs]

    # Sidecar aliases are used only when the sealed 25D direct fields are not
    # present.  They are still current-step values, never future observations.
    eef = record.get("robot0_eef_pos")
    if isinstance(eef, (list, tuple)) and len(eef) >= 3:
        for index, value in zip((3, 4, 5), eef[:3]):
            if direct[index] is None:
                direct[index] = _number(value)
    qpos = record.get("robot0_gripper_qpos")
    if isinstance(qpos, (list, tuple)) and len(qpos) >= 2 and direct[1] is None:
        values = [_number(value) for value in qpos[:2]]
        if all(value is not None for value in values):
            direct[1] = float(values[0] + values[1])
    if isinstance(qpos, (list, tuple)) and len(qpos) >= 2 and direct[2] is None:
        values = [_number(value) for value in qpos[:2]]
        if all(value is not None for value in values):
            direct[2] = abs(float(values[0])) + abs(float(values[1]))

    if direct[0] is None:
        direct[0] = raw
    if direct[12] is None:
        direct[12] = raw
    if direct[9] is None or direct[10] is None or direct[11] is None:
        action = record.get("clean_action_raw_7d", record.get("action_raw"))
        if isinstance(action, (list, tuple)) and len(action) >= 7:
            for index, action_index in ((9, 0), (10, 1), (11, 2)):
                if direct[index] is None:
                    direct[index] = _number(action[action_index])

    if any(value is None for value in direct):
        missing = [FEATURE_NAMES[index] for index, value in enumerate(direct) if value is None]
        raise ValueError(f"missing current-step 13D fields: {missing}")
    return {
        "step": step,
        "raw_close": raw_close,
        "direct": tuple(float(value) for value in direct),
        "eef_z": float(direct[5]),
    }


class B3Causal25DMultieventV1:
    """Stateful causal feature builder with event-local reset semantics."""

    def __init__(self, config: Causal25DConfig = Causal25DConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._history: list[dict[str, Any] | None] = []
        self._prev_close: bool | None = None
        self._close_streak = 0
        self._open_streak = 0
        self._flip_window: deque[int] = deque(maxlen=self.config.rolling_flip_window)
        self._event_active = False
        self._event_id = -1
        self._next_event_id = 0
        self._last_close_step = -1
        self._close_eef_z: float | None = None
        self._events: list[dict[str, Any]] = []

    def update(self, record: dict[str, Any]) -> dict[str, Any]:
        step_value = record.get("step", record.get("step_idx", self._next_step))
        try:
            step = int(step_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid step: {step_value!r}") from exc
        if step != self._next_step:
            raise ValueError(f"non-contiguous step: expected {self._next_step}, got {step}")
        self._next_step += 1

        if record.get("valid", True) is False:
            # Invalid rows do not alter event boundaries or history state.
            self._history.append(None)
            return self._invalid_row(step, "source_step_invalid")

        try:
            normal = _normalise_record(record, step)
        except ValueError as exc:
            self._history.append(None)
            return self._invalid_row(step, str(exc))

        raw_close = normal["raw_close"]
        previous_close = self._prev_close
        if raw_close:
            self._close_streak += 1
            self._open_streak = 0
        else:
            self._open_streak += 1
            self._close_streak = 0
        flipped = int(previous_close is not None and previous_close != raw_close)
        self._flip_window.append(flipped)
        self._prev_close = raw_close

        close_onset = bool(not self._event_active and raw_close and previous_close is not True)
        if close_onset:
            self._event_active = True
            self._event_id = self._next_event_id
            self._next_event_id += 1
            self._last_close_step = step
            self._close_eef_z = normal["eef_z"]
            self._events.append({
                "event_id": self._event_id,
                "start_step": step,
                "end_step": None,
                "release_step": None,
                "closed_by": None,
            })

        release_onset = bool(self._event_active and not raw_close and self._open_streak == self.config.n_open)
        released_event_id = -1
        if release_onset:
            released_event_id = self._event_id
            self._events[-1].update({
                "end_step": step - 1,
                "release_step": step,
                "closed_by": "HYSTERESIS_RELEASE",
            })
            self._event_active = False
            self._event_id = -1
            self._last_close_step = -1
            self._close_eef_z = None

        direct = normal["direct"]
        eef_speed = sqrt(direct[6] ** 2 + direct[7] ** 2 + direct[8] ** 2)
        eef_z_delta = (
            direct[5] - self._close_eef_z
            if self._event_active and self._close_eef_z is not None
            else 0.0
        )
        qpos_delta_1 = self._lag_delta(1, 1, direct[1])
        qpos_delta_3 = self._lag_delta(1, 3, direct[1])
        opening_delta_3 = self._lag_delta(2, 3, direct[2])
        opening_values = self._recent_values(2, 4, direct[2])
        speed_values = self._recent_speed_values(eef_speed)
        features = list(direct) + [
            float(self._close_streak), float(self._open_streak), float(sum(self._flip_window)),
            float(close_onset), float(step - self._last_close_step if self._event_active else -1),
            eef_speed, eef_z_delta, qpos_delta_1, qpos_delta_3, opening_delta_3,
            pvariance(opening_values) if len(opening_values) >= 5 else 0.0,
            pvariance(speed_values) if len(speed_values) >= 5 else 0.0,
        ]
        if len(features) != len(FEATURE_NAMES) or not all(isfinite(value) for value in features):
            raise ValueError("non-finite or incorrectly sized B3 25D feature row")

        normal["direct"] = direct
        self._history.append(normal)
        row = {
            "schema": SCHEMA,
            "source_schema": SOURCE_SCHEMA,
            "step": step,
            "valid": True,
            "features_25d": features,
            "event_id": self._event_id,
            "event_ordinal": self._event_id,
            "event_active": self._event_active,
            "close_onset": close_onset,
            "release_onset": release_onset,
            "released_event_id": released_event_id,
            "event_local_state_reset": release_onset,
        }
        return row

    def rebuild(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = [self.update(record) for record in records]
        for event in self._events:
            if event["end_step"] is None:
                event["end_step"] = rows[-1]["step"] if rows else event["start_step"]
                event["closed_by"] = "EPISODE_END"
        return {
            "schema": SCHEMA,
            "source_schema": SOURCE_SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "rows": rows,
            "events": [dict(event) for event in self._events],
        }

    def _invalid_row(self, step: int, reason: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source_schema": SOURCE_SCHEMA,
            "step": step,
            "valid": False,
            "features_25d": None,
            "event_id": self._event_id if self._event_active else -1,
            "event_ordinal": self._event_id if self._event_active else -1,
            "event_active": self._event_active,
            "close_onset": False,
            "release_onset": False,
            "released_event_id": -1,
            "event_local_state_reset": False,
            "error": reason,
        }

    def _recent_values(self, direct_index: int, prior_count: int, current: float) -> list[float]:
        values = [row["direct"][direct_index] for row in self._history[-prior_count:] if row is not None]
        values.append(current)
        return values

    def _recent_speed_values(self, current: float) -> list[float]:
        values = []
        for row in self._history[-4:]:
            if row is not None:
                direct = row["direct"]
                values.append(sqrt(direct[6] ** 2 + direct[7] ** 2 + direct[8] ** 2))
        values.append(current)
        return values

    def _lag_delta(self, direct_index: int, lag: int, current_value: float) -> float:
        if len(self._history) < lag:
            return 0.0
        previous = self._history[-lag]
        if previous is None:
            return 0.0
        return current_value - previous["direct"][direct_index]


__all__ = ["B3Causal25DMultieventV1", "Causal25DConfig", "FEATURE_NAMES", "SCHEMA"]
