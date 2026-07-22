"""Fail-closed bridge for Factorized V2 scheduler-ready records.

This module only joins sealed runtime fields with sealed predictions.  It does
not invent utility or regrasp scores and has no model, simulator, or attack
dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_contract import CanonicalActionState

SCHEDULER_READY_SCHEMA = "FACTORIZED_V2_SCHEDULER_READY_PREDICTION_V1"
FIELD_STATUSES = frozenset({"DIRECT", "DERIVED", "MISSING", "FORBIDDEN"})
RUNTIME_ACTION_FIELDS = ("clean_action_raw_7d", "action_raw_7d", "action_raw")
SCALAR_RAW_FIELDS = ("raw_gripper", "gripper_command")
REQUIRED_READY_FIELDS = frozenset(
    {
        "schema", "episode", "step", "route", "route_supported", "student_valid",
        "candidate_close", "candidate_close_source", "raw_gripper", "raw_gripper_source",
        "action_intent", "action_intent_source", "utility_probability", "utility_source",
        "release_probability", "release_source", "regrasp_probability", "regrasp_source",
        "uncertainty_probability", "uncertainty_source", "features_25d", "feature_dtype",
        "feature_order_sha256", "checkpoint_sha256", "source_commit", "input_artifact_seal",
        "causal_field_declaration", "field_statuses",
    }
)
FORBIDDEN_READY_FIELDS = frozenset(
    {
        "event_id", "event_role", "event_ordinal", "event_duration", "release_target",
        "release_known_mask", "grasp_target", "grasp_known_mask", "manipulation_target",
        "manipulation_known_mask", "teacher_phase", "teacher_utility", "window_end",
        "future_score", "future_utility", "contact", "object_state", "attack_outcome",
        "action", "clean_action", "executed_action",
    }
)


class SchedulerBridgeError(ValueError):
    """Stable fail-closed error for bridge input violations."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise SchedulerBridgeError(code)
    return value.lower()


def _commit(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise SchedulerBridgeError("SOURCE_COMMIT_INVALID")
    return value.lower()


def _status(value: Any, code: str) -> str:
    if value not in FIELD_STATUSES:
        raise SchedulerBridgeError(code)
    return str(value)


def _probability(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchedulerBridgeError(code)
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SchedulerBridgeError(code)
    return value


def _step(row: Mapping[str, Any]) -> int:
    value = row.get("step", row.get("step_index"))
    if isinstance(value, bool):
        raise SchedulerBridgeError("STEP_INVALID")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise SchedulerBridgeError("STEP_INVALID") from exc
    if value < 0:
        raise SchedulerBridgeError("STEP_INVALID")
    return value


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("episode", row.get("canonical_parent_key"))
    if not isinstance(value, str) or value.count("/") != 2:
        raise SchedulerBridgeError("EPISODE_INVALID")
    return value


def _runtime_action_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the runtime gate from the canonical raw action field only."""

    for field in RUNTIME_ACTION_FIELDS:
        value = row.get(field)
        if isinstance(value, (list, tuple)) and len(value) >= 7:
            state = CanonicalActionState.from_step(dict(row), field=field)
            if state.raw_gripper is None:
                raise SchedulerBridgeError("RUNTIME_RAW_GRIPPER_INVALID")
            return {
                "raw_gripper": state.raw_gripper,
                "action_intent": state.action_intent,
                "candidate_close": state.candidate_close,
                "raw_gripper_source": "DIRECT" if field == "clean_action_raw_7d" else "DERIVED",
                "raw_gripper_source_field": f"{field}[6]",
                "action_intent_source": "DERIVED",
                "candidate_close_source": "DERIVED",
                "candidate_close_source_field": f"{field}[6]",
            }
    for field in SCALAR_RAW_FIELDS:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            state = CanonicalActionState.from_step({"clean_action_raw_7d": [0.0] * 6 + [float(value)]})
            if state.raw_gripper is None:
                raise SchedulerBridgeError("RUNTIME_RAW_GRIPPER_INVALID")
            return {
                "raw_gripper": state.raw_gripper,
                "action_intent": state.action_intent,
                "candidate_close": state.candidate_close,
                "raw_gripper_source": "DIRECT",
                "raw_gripper_source_field": field,
                "action_intent_source": "DERIVED",
                "candidate_close_source": "DERIVED",
                "candidate_close_source_field": field,
            }
    raise SchedulerBridgeError("RUNTIME_RAW_GRIPPER_MISSING")


def _index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        step = _step(row)
        if step in indexed:
            raise SchedulerBridgeError(f"DUPLICATE_{label.upper()}_STEP")
        indexed[step] = row
    return indexed


def exact_step_join(
    prediction_rows: Iterable[Mapping[str, Any]],
    student_rows: Iterable[Mapping[str, Any]],
    runtime_rows: Iterable[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    """Join three per-step streams and reject any gap, duplicate, or drift."""

    pred = _index(prediction_rows, "prediction")
    student = _index(student_rows, "student")
    runtime = _index(runtime_rows, "runtime")
    if set(pred) != set(student) or set(pred) != set(runtime):
        raise SchedulerBridgeError("STEP_SET_MISMATCH")
    if sorted(pred) != list(range(len(pred))):
        raise SchedulerBridgeError("STEP_SEQUENCE_INVALID")
    return [(pred[i], student[i], runtime[i]) for i in range(len(pred))]


def _feature_vector(student: Mapping[str, Any], expected_sha: str) -> list[float]:
    values = student.get("features_25d")
    if not isinstance(values, list) or len(values) != 25:
        raise SchedulerBridgeError("FEATURE_WIDTH_INVALID")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise SchedulerBridgeError("FEATURE_NONFINITE")
    return vector


def build_scheduler_ready_record(
    prediction: Mapping[str, Any],
    student: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source_commit: str,
    input_artifact_seal: str,
    feature_order_sha256: str,
) -> dict[str, Any]:
    """Build one authoritative-ready row; missing semantic heads are fatal."""

    identity = _identity(prediction)
    step = _step(prediction)
    if _identity(student) != identity or _identity(runtime) != identity:
        raise SchedulerBridgeError("IDENTITY_JOIN_MISMATCH")
    if _step(student) != step or _step(runtime) != step:
        raise SchedulerBridgeError("STEP_JOIN_MISMATCH")
    checkpoint_sha256 = _sha(checkpoint_sha256, "CHECKPOINT_SHA_INVALID")
    input_artifact_seal = _sha(input_artifact_seal, "INPUT_SEAL_INVALID")
    feature_order_sha256 = _sha(feature_order_sha256, "FEATURE_ORDER_SHA_INVALID")
    source_commit = _commit(source_commit)

    action = _runtime_action_state(runtime)
    utility = prediction.get("utility_probability")
    if utility is None:
        raise SchedulerBridgeError("UTILITY_MISSING_NO_DEFAULT_OR_PROXY")
    regrasp = prediction.get("regrasp_probability")
    if regrasp is None:
        raise SchedulerBridgeError("REGRASP_MISSING_NO_DEFAULT_OR_PROXY")
    release = prediction.get("release_probability")
    if release is None:
        raise SchedulerBridgeError("RELEASE_MISSING_NO_DEFAULT_OR_PROXY")
    for proxy in ("grasp_prob", "manipulation_prob", "release_prob"):
        if prediction.get("utility_source") == proxy or prediction.get("regrasp_source") == proxy:
            raise SchedulerBridgeError("PROXY_HEAD_REJECTED")
    utility_source = _status(prediction.get("utility_source"), "UTILITY_SOURCE_STATUS_INVALID")
    release_source = _status(prediction.get("release_source"), "RELEASE_SOURCE_STATUS_INVALID")
    regrasp_source = _status(prediction.get("regrasp_source"), "REGRASP_SOURCE_STATUS_INVALID")
    if utility_source in {"MISSING", "FORBIDDEN"}:
        raise SchedulerBridgeError("UTILITY_SOURCE_NOT_AUTHORITATIVE")
    if regrasp_source in {"MISSING", "FORBIDDEN"}:
        raise SchedulerBridgeError("REGRASP_SOURCE_NOT_AUTHORITATIVE")

    uncertainty = prediction.get("uncertainty_probability", 0.0)
    uncertainty_source = _status(prediction.get("uncertainty_source", "DERIVED"), "UNCERTAINTY_SOURCE_STATUS_INVALID")
    if uncertainty_source != "DERIVED" or float(uncertainty) != 0.0:
        _probability(uncertainty, "UNCERTAINTY_INVALID")

    student_valid = student.get("valid")
    if not isinstance(student_valid, bool):
        raise SchedulerBridgeError("STUDENT_VALID_MISSING")
    route = prediction.get("route", prediction.get("mechanism_route"))
    if not isinstance(route, str) or not route:
        raise SchedulerBridgeError("ROUTE_MISSING")
    route_supported = prediction.get("route_supported")
    if not isinstance(route_supported, bool):
        raise SchedulerBridgeError("ROUTE_SUPPORTED_MISSING")

    record = {
        "schema": SCHEDULER_READY_SCHEMA,
        "episode": identity,
        "step": step,
        "route": route,
        "route_supported": route_supported,
        "student_valid": student_valid,
        "candidate_close": bool(action["candidate_close"]),
        "candidate_close_source": action["candidate_close_source"],
        "raw_gripper": action["raw_gripper"],
        "raw_gripper_source": action["raw_gripper_source"],
        "action_intent": action["action_intent"],
        "action_intent_source": action["action_intent_source"],
        "utility_probability": _probability(utility, "UTILITY_INVALID"),
        "utility_source": utility_source,
        "release_probability": _probability(release, "RELEASE_INVALID"),
        "release_source": release_source,
        "regrasp_probability": _probability(regrasp, "REGRASP_INVALID"),
        "regrasp_source": regrasp_source,
        "uncertainty_probability": 0.0,
        "uncertainty_source": uncertainty_source,
        "features_25d": _feature_vector(student, feature_order_sha256),
        "feature_dtype": "float32",
        "feature_order_sha256": feature_order_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "input_artifact_seal": input_artifact_seal,
        "causal_field_declaration": {
            "future_steps_consumed": False,
            "teacher_fields_consumed": False,
            "attack_fields_consumed": False,
            "uncertainty_veto_enabled": False,
        },
        "field_statuses": {
            "candidate_close": "DERIVED", "raw_gripper": action["raw_gripper_source"],
            "action_intent": "DERIVED", "student_valid": "DIRECT",
            "utility_probability": utility_source, "release_probability": release_source,
            "regrasp_probability": regrasp_source, "uncertainty_probability": "DERIVED",
            "features_25d": "DIRECT", "checkpoint_sha256": "DIRECT",
            "source_commit": "DIRECT", "input_artifact_seal": "DIRECT",
        },
    }
    validate_scheduler_ready_record(record)
    return record


def validate_scheduler_ready_record(record: Mapping[str, Any]) -> None:
    forbidden = set(record) & FORBIDDEN_READY_FIELDS
    if forbidden:
        raise SchedulerBridgeError(f"FORBIDDEN_FIELDS:{','.join(sorted(forbidden))}")
    if set(record) != REQUIRED_READY_FIELDS:
        raise SchedulerBridgeError("READY_FIELD_SET_MISMATCH")
    if record["schema"] != SCHEDULER_READY_SCHEMA:
        raise SchedulerBridgeError("READY_SCHEMA")
    _identity(record)
    _step(record)
    for name in ("route_supported", "student_valid", "candidate_close"):
        if not isinstance(record[name], bool):
            raise SchedulerBridgeError(f"{name.upper()}_INVALID")
    for name in ("candidate_close_source", "raw_gripper_source", "action_intent_source", "utility_source", "release_source", "regrasp_source", "uncertainty_source"):
        _status(record[name], f"{name.upper()}_STATUS_INVALID")
    if record["utility_source"] in {"MISSING", "FORBIDDEN"} or record["regrasp_source"] in {"MISSING", "FORBIDDEN"}:
        raise SchedulerBridgeError("SEMANTIC_HEAD_MISSING")
    _probability(record["utility_probability"], "UTILITY_INVALID")
    _probability(record["release_probability"], "RELEASE_INVALID")
    _probability(record["regrasp_probability"], "REGRASP_INVALID")
    if record["uncertainty_source"] != "DERIVED" or float(record["uncertainty_probability"]) != 0.0:
        raise SchedulerBridgeError("UNCERTAINTY_MUST_BE_EXPLICITLY_DISABLED")
    _probability(record["uncertainty_probability"], "UNCERTAINTY_INVALID")
    _sha(record["checkpoint_sha256"], "CHECKPOINT_SHA_INVALID")
    _sha(record["input_artifact_seal"], "INPUT_SEAL_INVALID")
    _commit(record["source_commit"])
    features = record["features_25d"]
    if record["feature_dtype"] != "float32" or not isinstance(features, list) or len(features) != 25:
        raise SchedulerBridgeError("FEATURE_CONTRACT_INVALID")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in features):
        raise SchedulerBridgeError("FEATURE_NONFINITE")
    if not isinstance(record["causal_field_declaration"], dict) or any(record["causal_field_declaration"].get(key) is not False for key in ("future_steps_consumed", "teacher_fields_consumed", "attack_fields_consumed")):
        raise SchedulerBridgeError("CAUSAL_DECLARATION_INVALID")
    statuses = record["field_statuses"]
    if not isinstance(statuses, dict) or any(value not in FIELD_STATUSES for value in statuses.values()):
        raise SchedulerBridgeError("FIELD_STATUS_INVALID")


__all__ = [
    "FIELD_STATUSES", "FORBIDDEN_READY_FIELDS", "REQUIRED_READY_FIELDS",
    "SCHEDULER_READY_SCHEMA", "SchedulerBridgeError", "build_scheduler_ready_record",
    "exact_step_join", "sha256_file", "validate_scheduler_ready_record",
]
