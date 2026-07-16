"""Offline reconstruction of event-local causal 25D features.

This module is deliberately separate from ``SC5StreamingFeatureAdapterV2``.
The legacy adapter remains unchanged for provenance; this version resets
event-local state after hysteresis release and uses a rolling flip window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
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

# This is the frozen order stored by the Official CLEAN artifacts.  It is an
# input binding only; the old event-local values are never copied into B3.
LEGACY_SOURCE_FEATURE_NAMES_25D = (
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
)
LEGACY_SOURCE_FEATURE_ORDER_SHA256 = hashlib.sha256(
    json.dumps(list(LEGACY_SOURCE_FEATURE_NAMES_25D), separators=(",", ":")).encode()
).hexdigest()
ACTION_PARITY_TOLERANCE = 1e-6
ROBOT_QPOS_PARITY_TOLERANCE = 1e-6
# EEF sidecar values and the stored 25D observation can come from separate
# float32 observation paths.  Keep this explicit and tighter than the
# existing materializer's 1e-3 EEF observation/site contract.
ROBOT_EEF_PARITY_TOLERANCE = 1e-3
STUDENT_ALLOWED_KEYS = frozenset({"schema", "source_schema", "valid", "features_25d"})
STUDENT_FORBIDDEN_FEATURE_NAMES = frozenset({
    "event_id", "event_ordinal", "teacher_label", "task_id", "state_id",
    "normalized_step", "suite", "task_language", "success", "env_success",
    "object_state", "mujoco_contact_pairs", "attack_outcome", "event_active",
    "release_onset", "released_event_id", "event_local_state_reset",
})


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


def _vector(record: dict[str, Any], name: str, minimum_length: int) -> list[float] | None:
    if name not in record:
        return None
    value = record[name]
    if not isinstance(value, (list, tuple)) or len(value) < minimum_length:
        raise ValueError(f"{name} must be a vector of length >= {minimum_length}")
    values = [_number(item) for item in value]
    if any(item is None for item in values):
        raise ValueError(f"{name} contains a non-finite value")
    return [float(item) for item in values]


def _measured_action(record: dict[str, Any], names: tuple[str, ...], label: str) -> list[float]:
    candidates = []
    for name in names:
        values = _vector(record, name, 7)
        if values is not None:
            if len(values) != 7:
                raise ValueError(f"{name} must contain exactly 7 action values")
            candidates.append((name, values))
    if not candidates:
        raise ValueError(f"missing measured {label} action vector")
    reference = candidates[0][1]
    for name, values in candidates[1:]:
        if max(abs(a - b) for a, b in zip(reference, values)) > ACTION_PARITY_TOLERANCE:
            raise ValueError(f"{label} action aliases disagree: {candidates[0][0]} vs {name}")
    return reference


def _scalar(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in record:
            number = _number(record[name])
            if number is None:
                raise ValueError(f"{name} is not finite")
            return number
    return None


def _feature_vector(record: dict[str, Any]) -> tuple[float, ...] | None:
    if "features_25d" not in record:
        return None
    vector = record["features_25d"]
    if not isinstance(vector, (list, tuple)) or len(vector) != 25:
        raise ValueError("features_25d must have exactly 25 values")
    names = record.get("feature_names_25d")
    if list(names or ()) != list(LEGACY_SOURCE_FEATURE_NAMES_25D):
        raise ValueError("features_25d feature order is not bound to the frozen legacy order")
    order_sha = record.get("feature_order_sha256")
    if order_sha != LEGACY_SOURCE_FEATURE_ORDER_SHA256:
        raise ValueError("features_25d feature order SHA256 mismatch")
    values = tuple(_number(item) for item in vector)
    if any(item is None for item in values):
        raise ValueError("features_25d contains a non-finite value")
    return tuple(float(item) for item in values)


def _named_value(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    values = []
    for name in names:
        if name in record:
            number = _number(record[name])
            if number is None:
                raise ValueError(f"{name} is not finite")
            values.append((name, number))
    if not values:
        return None
    reference = values[0][1]
    for name, value in values[1:]:
        if abs(reference - value) > ACTION_PARITY_TOLERANCE:
            raise ValueError(f"named feature aliases disagree: {values[0][0]} vs {name}")
    return reference


def _direct_value(
    record: dict[str, Any],
    vector: tuple[float, ...] | None,
    index: int,
    names: tuple[str, ...],
) -> float | None:
    vector_value = None if vector is None else vector[index]
    named_value = _named_value(record, names)
    if vector_value is not None and named_value is not None and abs(vector_value - named_value) > ACTION_PARITY_TOLERANCE:
        raise ValueError(f"features_25d vs named field mismatch for {names[0]}")
    return vector_value if vector_value is not None else named_value


def _sidecar_vector(record: dict[str, Any], name: str, length: int) -> list[float] | None:
    if name not in record:
        return None
    values = _vector(record, name, length)
    assert values is not None
    return values[:length]


def _parity_or_fallback(
    direct: float | None,
    sidecar: float | None,
    name: str,
    tolerance: float = ACTION_PARITY_TOLERANCE,
) -> float | None:
    if direct is not None and sidecar is not None and abs(direct - sidecar) > tolerance:
        raise ValueError(f"{name} step/sidecar parity mismatch")
    return direct if direct is not None else sidecar


def _normalise_record(record: dict[str, Any], fallback_step: int) -> dict[str, Any]:
    step_value = record.get("step", record.get("step_idx", fallback_step))
    try:
        step = int(step_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid step: {step_value!r}") from exc

    # These must be measured action vectors.  raw_close is a consistency check
    # only and is never a source for raw/env action values.
    raw_action = _measured_action(record, ("clean_action_raw_7d", "action_raw_7d", "action_raw"), "raw")
    env_action = _measured_action(record, ("applied_action_7d", "action_env"), "env")
    if max(abs(raw_action[index] - env_action[index]) for index in range(6)) > ACTION_PARITY_TOLERANCE:
        raise ValueError("raw/env arm action mismatch")
    raw = raw_action[-1]
    env = env_action[-1]
    raw_close = raw <= 0.5
    env_close = env > 0.0
    if raw_close != env_close:
        raise ValueError("raw/env gripper semantics mismatch")
    explicit_close = record.get("raw_close")
    if explicit_close is not None and (not isinstance(explicit_close, bool) or explicit_close != raw_close):
        raise ValueError("raw_close contradicts measured raw action")
    recorded_raw = _scalar(record, ("raw_gripper",))
    if recorded_raw is not None and abs(recorded_raw - raw) > ACTION_PARITY_TOLERANCE:
        raise ValueError("raw_gripper vs measured raw action mismatch")
    recorded_env = _scalar(record, ("env_gripper",))
    if recorded_env is not None and abs(recorded_env - env) > ACTION_PARITY_TOLERANCE:
        raise ValueError("env_gripper vs measured env action mismatch")

    vector = _feature_vector(record)
    direct_specs = (
        (0, ("gripper_command",)),
        (1, ("gripper_qpos",)),
        (2, ("gripper_opening_proxy", "gripper_width", "opening_proxy")),
        (3, ("eef_x",)), (4, ("eef_y",)), (5, ("eef_z",)),
        (6, ("eef_vx",)), (7, ("eef_vy",)), (8, ("eef_vz",)),
        (9, ("action_dx",)), (10, ("action_dy",)), (11, ("action_dz",)),
        (12, ("action_gripper",)),
    )
    direct = [_direct_value(record, vector, index, names) for index, names in direct_specs]

    eef_sidecar = _sidecar_vector(record, "robot0_eef_pos", 3)
    if eef_sidecar is not None:
        for index, value in zip((3, 4, 5), eef_sidecar):
            direct[index] = _parity_or_fallback(
                direct[index], value, FEATURE_NAMES[index], ROBOT_EEF_PARITY_TOLERANCE
            )

    qpos_sidecar = _sidecar_vector(record, "robot0_gripper_qpos", 2)
    if qpos_sidecar is not None:
        direct[1] = _parity_or_fallback(
            direct[1], sum(qpos_sidecar), "gripper_qpos", ROBOT_QPOS_PARITY_TOLERANCE
        )
        direct[2] = _parity_or_fallback(
            direct[2], sum(abs(value) for value in qpos_sidecar),
            "gripper_opening_proxy", ROBOT_QPOS_PARITY_TOLERANCE,
        )

    if direct[0] is None:
        direct[0] = raw
    elif abs(direct[0] - raw) > ACTION_PARITY_TOLERANCE:
        raise ValueError("gripper_command vs measured raw action mismatch")
    if direct[12] is None:
        direct[12] = raw
    elif abs(direct[12] - raw) > ACTION_PARITY_TOLERANCE:
        raise ValueError("action_gripper vs measured raw action mismatch")

    for index, action_index in ((9, 0), (10, 1), (11, 2)):
        if abs(direct[index] - raw_action[action_index]) > ACTION_PARITY_TOLERANCE:
            raise ValueError(f"{FEATURE_NAMES[index]} vs measured raw action mismatch")

    if any(value is None for value in direct):
        missing = [FEATURE_NAMES[index] for index, value in enumerate(direct) if value is None]
        raise ValueError(f"missing current-step 13D fields: {missing}")
    return {
        "step": step,
        "raw_close": raw_close,
        "feature_order_bound": vector is not None or all(_named_value(record, names) is not None for _, names in direct_specs),
        "direct": tuple(float(value) for value in direct),
        "eef_z": float(direct[5]),
    }


def serialize_student_25d(row: dict[str, Any]) -> tuple[float, ...]:
    """Project exactly one validated student row; reject all side channels."""
    extra = set(row) - STUDENT_ALLOWED_KEYS
    missing = STUDENT_ALLOWED_KEYS - set(row)
    if extra or missing:
        raise ValueError(f"student row key contract violation; extra={sorted(extra)}, missing={sorted(missing)}")
    if row["schema"] != SCHEMA or row["source_schema"] != SOURCE_SCHEMA or row["valid"] is not True:
        raise ValueError("student row schema/valid contract violation")
    vector = row["features_25d"]
    if not isinstance(vector, (list, tuple)) or len(vector) != 25:
        raise ValueError("student features_25d must have exactly 25 values")
    values = tuple(_number(item) for item in vector)
    if any(item is None for item in values):
        raise ValueError("student features_25d contains a non-finite value")
    if any(name in STUDENT_FORBIDDEN_FEATURE_NAMES for name in FEATURE_NAMES):
        raise ValueError("forbidden student feature name")
    return tuple(float(item) for item in values)


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
        return {
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
            "feature_order_bound": normal["feature_order_bound"],
        }

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
            "feature_order_bound": False,
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


__all__ = [
    "B3Causal25DMultieventV1",
    "ACTION_PARITY_TOLERANCE",
    "Causal25DConfig",
    "FEATURE_NAMES",
    "LEGACY_SOURCE_FEATURE_NAMES_25D",
    "LEGACY_SOURCE_FEATURE_ORDER_SHA256",
    "ROBOT_EEF_PARITY_TOLERANCE",
    "ROBOT_QPOS_PARITY_TOLERANCE",
    "SCHEMA",
    "SOURCE_SCHEMA",
    "serialize_student_25d",
]
