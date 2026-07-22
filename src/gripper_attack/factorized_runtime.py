"""Strict Factorized V2 runtime input bridge.

This module is intentionally separate from the legacy V5 utility/regrasp
bridge. It accepts only Factorized grasp/manipulation/release heads and emits
no Teacher labels or action fields.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .action_contract import CanonicalActionState


RUNTIME_SCHEMA = "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_V1"
FIELD_STATUSES = frozenset({"DIRECT", "DERIVED", "MISSING", "FORBIDDEN"})
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "utility_probability", "utility_logit", "regrasp_probability", "regrasp_logit",
    "event_id", "event_role", "teacher_phase", "known_mask", "grasp_known_mask",
    "manipulation_known_mask", "release_known_mask", "strict_k10_feasible",
    "strict_k10_known_mask", "future_score", "future_utility", "attack_outcome",
    "object_state", "contact", "action", "clean_action", "executed_action",
    "mutated_action",
})
RUNTIME_FIELDS = frozenset({
    "schema", "episode", "step", "route", "route_supported", "student_valid",
    "student_valid_source", "candidate_close", "action_known", "action_intent",
    "raw_gripper", "candidate_close_source", "raw_gripper_source_field",
    "grasp_logit", "grasp_probability", "grasp_source",
    "manipulation_logit", "manipulation_probability", "manipulation_source",
    "release_logit", "release_probability", "release_source",
    "feature_window_valid", "feature_order_sha256", "checkpoint_sha256",
    "source_commit", "prediction_artifact_seal", "runtime_artifact_seal",
    "causal_field_declaration",
})
OPTIONAL_RUNTIME_FIELDS = frozenset({"split", "scheduler_source_sha256", "structural_config_sha256"})


class FactorizedRuntimeError(ValueError):
    pass


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise FactorizedRuntimeError(code)
    return value.lower()


def _commit(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise FactorizedRuntimeError("SOURCE_COMMIT_INVALID")
    return value.lower()


def _probability(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorizedRuntimeError(code)
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FactorizedRuntimeError(code)
    return value


def _logit(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FactorizedRuntimeError(code)
    return float(value)


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("episode", row.get("canonical_parent_key"))
    if not isinstance(value, str) or value.count("/") != 2:
        raise FactorizedRuntimeError("EPISODE_INVALID")
    return value


def _step(row: Mapping[str, Any]) -> int:
    value = row.get("step", row.get("step_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorizedRuntimeError("STEP_INVALID")
    return value


def _fallback_certified(manifest: Mapping[str, Any] | None) -> bool:
    expected = {
        "field_semantics": "OPENVLA_RAW_ACTION",
        "field_stage": "CLEAN_PRE_ATTACK_DECODE",
        "field_dimension": 7,
        "gripper_index": 6,
        "postprocessed": False,
        "attacked": False,
    }
    pending: list[Any] = [manifest]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            if all(item.get(key) == value for key, value in expected.items()):
                return True
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _raw_action_state(row: Mapping[str, Any], runtime_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if FORBIDDEN_RUNTIME_FIELDS & set(row):
        raise FactorizedRuntimeError("FORBIDDEN_RUNTIME_SOURCE_FIELD")
    if any(name in row for name in ("attacked_action", "attack_action", "mutated_action")):
        raise FactorizedRuntimeError("ATTACKED_ACTION_FORBIDDEN")

    def raw_from(value: Any) -> float | None:
        if not isinstance(value, (list, tuple)) or len(value) < 7:
            return None
        try:
            value = float(value[6])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or not -0.1 <= value <= 1.1:
            return None
        return value

    clean_present = "clean_action_raw_7d" in row
    fallback_present = "action_raw" in row
    clean = raw_from(row.get("clean_action_raw_7d"))
    fallback = raw_from(row.get("action_raw"))
    if clean_present and clean is None:
        raise FactorizedRuntimeError("RUNTIME_RAW_GRIPPER_INVALID")
    if fallback_present and fallback is None:
        raise FactorizedRuntimeError("RUNTIME_RAW_GRIPPER_INVALID")
    if clean is not None and fallback is not None and abs(clean - fallback) > 1e-6:
        raise FactorizedRuntimeError("RAW_ACTION_FIELDS_MISMATCH")
    if clean is not None:
        raw, field = clean, "clean_action_raw_7d[6]"
        source = "DIRECT"
    elif fallback is not None:
        if not _fallback_certified(runtime_manifest):
            raise FactorizedRuntimeError("FALLBACK_RAW_ACTION_UNCERTIFIED")
        raw, field, source = fallback, "action_raw[6]", "DIRECT"
    else:
        return {
            "raw_gripper": None, "raw_gripper_source_field": "MISSING",
            "candidate_close": False, "action_known": False,
            "action_intent": "UNKNOWN", "candidate_close_source": "MISSING",
        }
    state = CanonicalActionState.from_step({"clean_action_raw_7d": [0.0] * 6 + [raw]})
    if state.raw_gripper is None:
        raise FactorizedRuntimeError("RUNTIME_RAW_GRIPPER_INVALID")
    return {
        "raw_gripper": state.raw_gripper,
        "raw_gripper_source_field": field,
        "candidate_close": state.candidate_close,
        "action_known": state.action_known,
        "action_intent": state.action_intent,
        "candidate_close_source": "DERIVED",
        "raw_gripper_source": source,
    }


def _head(prediction: Mapping[str, Any], name: str) -> tuple[float, float]:
    aliases = (f"{name}_probability", f"{name}_prob")
    probability = next((prediction[key] for key in aliases if key in prediction), None)
    if probability is None:
        raise FactorizedRuntimeError(f"{name.upper()}_PROBABILITY_MISSING")
    logit = prediction.get(f"{name}_logit")
    if logit is None:
        raise FactorizedRuntimeError(f"{name.upper()}_LOGIT_MISSING")
    return _logit(logit, f"{name.upper()}_LOGIT_INVALID"), _probability(probability, f"{name.upper()}_PROBABILITY_INVALID")


def build_runtime_record(
    prediction: Mapping[str, Any],
    student: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source_commit: str,
    prediction_artifact_seal: str,
    runtime_artifact_seal: str,
    feature_order_sha256: str,
    runtime_manifest: Mapping[str, Any] | None = None,
    scheduler_source_sha256: str | None = None,
    structural_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one strict runtime row without consuming offline labels."""
    if {"utility_probability", "regrasp_probability"} & set(prediction):
        raise FactorizedRuntimeError("LEGACY_HEADS_FORBIDDEN_IN_FACTORIZED_RUNTIME")
    identity = _identity(prediction)
    step = _step(prediction)
    if _identity(student) != identity or _identity(runtime) != identity or _step(student) != step or _step(runtime) != step:
        raise FactorizedRuntimeError("IDENTITY_OR_STEP_JOIN_MISMATCH")
    checkpoint_sha256 = _sha(checkpoint_sha256, "CHECKPOINT_SHA_INVALID")
    prediction_artifact_seal = _sha(prediction_artifact_seal, "PREDICTION_SEAL_INVALID")
    runtime_artifact_seal = _sha(runtime_artifact_seal, "RUNTIME_SEAL_INVALID")
    feature_order_sha256 = _sha(feature_order_sha256, "FEATURE_ORDER_SHA_INVALID")
    source_commit = _commit(source_commit)
    if (scheduler_source_sha256 is None) != (structural_config_sha256 is None):
        raise FactorizedRuntimeError("RUNTIME_CONTRACT_CONTEXT_INCOMPLETE")
    if scheduler_source_sha256 is not None:
        scheduler_source_sha256 = _sha(scheduler_source_sha256, "SCHEDULER_SOURCE_SHA_INVALID")
        structural_config_sha256 = _sha(structural_config_sha256, "STRUCTURAL_CONFIG_SHA_INVALID")
    action = _raw_action_state(runtime, runtime_manifest)
    route = prediction.get("route", prediction.get("mechanism_route"))
    if not isinstance(route, str) or not route:
        raise FactorizedRuntimeError("ROUTE_MISSING")
    route_supported = prediction.get("route_supported")
    if not isinstance(route_supported, bool):
        raise FactorizedRuntimeError("ROUTE_SUPPORTED_MISSING")
    feature_valid = student.get("feature_window_valid", student.get("valid"))
    if not isinstance(feature_valid, bool):
        raise FactorizedRuntimeError("FEATURE_WINDOW_VALID_MISSING")
    student_valid_source = student.get("valid", feature_valid)
    if not isinstance(student_valid_source, bool):
        raise FactorizedRuntimeError("STUDENT_VALID_SOURCE_INVALID")
    grasp_logit, grasp_probability = _head(prediction, "grasp")
    manipulation_logit, manipulation_probability = _head(prediction, "manipulation")
    release_logit, release_probability = _head(prediction, "release")
    record = {
        "schema": RUNTIME_SCHEMA,
        "episode": identity,
        "step": step,
        "route": route,
        "route_supported": route_supported,
        "student_valid": bool(student_valid_source and feature_valid and route_supported and action["action_known"]),
        "student_valid_source": "DERIVED",
        "candidate_close": bool(action["candidate_close"]),
        "action_known": bool(action["action_known"]),
        "action_intent": action["action_intent"],
        "raw_gripper": action["raw_gripper"],
        "candidate_close_source": action["candidate_close_source"],
        "raw_gripper_source_field": action["raw_gripper_source_field"],
        "grasp_logit": grasp_logit,
        "grasp_probability": grasp_probability,
        "grasp_source": "DIRECT",
        "manipulation_logit": manipulation_logit,
        "manipulation_probability": manipulation_probability,
        "manipulation_source": "DIRECT",
        "release_logit": release_logit,
        "release_probability": release_probability,
        "release_source": "DIRECT",
        "feature_window_valid": feature_valid,
        "feature_order_sha256": feature_order_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "prediction_artifact_seal": prediction_artifact_seal,
        "runtime_artifact_seal": runtime_artifact_seal,
        "causal_field_declaration": {
            "future_steps_consumed": False,
            "teacher_fields_consumed": False,
            "attack_fields_consumed": False,
        },
    }
    if scheduler_source_sha256 is not None and structural_config_sha256 is not None:
        record["scheduler_source_sha256"] = scheduler_source_sha256
        record["structural_config_sha256"] = structural_config_sha256
    validate_runtime_record(record)
    return record


def validate_runtime_record(record: Mapping[str, Any]) -> None:
    forbidden = set(record) & FORBIDDEN_RUNTIME_FIELDS
    if forbidden:
        raise FactorizedRuntimeError(f"FORBIDDEN_RUNTIME_FIELDS:{','.join(sorted(forbidden))}")
    if not RUNTIME_FIELDS.issubset(record) or set(record) - RUNTIME_FIELDS - OPTIONAL_RUNTIME_FIELDS:
        raise FactorizedRuntimeError("RUNTIME_FIELD_SET_MISMATCH")
    if record.get("schema") != RUNTIME_SCHEMA:
        raise FactorizedRuntimeError("RUNTIME_SCHEMA")
    identity = _identity(record)
    if identity.count("/") != 2:
        raise FactorizedRuntimeError("EPISODE_INVALID")
    _step(record)
    for key in ("route_supported", "student_valid", "candidate_close", "action_known", "feature_window_valid"):
        if not isinstance(record[key], bool):
            raise FactorizedRuntimeError("RUNTIME_BOOL_INVALID")
    if record["candidate_close"] and (not record["action_known"] or record["action_intent"] != "CLOSE"):
        raise FactorizedRuntimeError("CANDIDATE_CLOSE_INCONSISTENT")
    if record["action_intent"] not in {"CLOSE", "OPEN", "BOUNDARY", "UNKNOWN"}:
        raise FactorizedRuntimeError("ACTION_INTENT_INVALID")
    if not record["action_known"] and record["candidate_close"]:
        raise FactorizedRuntimeError("ACTION_BOUNDARY_INCONSISTENT")
    if record["raw_gripper"] is not None and not math.isfinite(float(record["raw_gripper"])):
        raise FactorizedRuntimeError("RAW_GRIPPER_NONFINITE")
    for key in ("grasp_probability", "manipulation_probability", "release_probability"):
        _probability(record[key], f"{key.upper()}_INVALID")
    for key in ("grasp_logit", "manipulation_logit", "release_logit"):
        _logit(record[key], f"{key.upper()}_INVALID")
    for key in ("grasp_source", "manipulation_source", "release_source", "candidate_close_source"):
        if record[key] not in {"DIRECT", "DERIVED"}:
            raise FactorizedRuntimeError("RUNTIME_SOURCE_NOT_AVAILABLE")
    _sha(record["feature_order_sha256"], "FEATURE_ORDER_SHA_INVALID")
    _sha(record["checkpoint_sha256"], "CHECKPOINT_SHA_INVALID")
    _sha(record["prediction_artifact_seal"], "PREDICTION_SEAL_INVALID")
    _sha(record["runtime_artifact_seal"], "RUNTIME_SEAL_INVALID")
    _commit(record["source_commit"])
    if "split" in record and (not isinstance(record["split"], str) or len(record["split"]) != 5 or record["split"][0] != "o" or record["split"][2:4] != "_i"):
        raise FactorizedRuntimeError("RUNTIME_SPLIT_INVALID")
    if ("scheduler_source_sha256" in record) != ("structural_config_sha256" in record):
        raise FactorizedRuntimeError("RUNTIME_CONTRACT_CONTEXT_INCOMPLETE")
    if "scheduler_source_sha256" in record:
        _sha(record["scheduler_source_sha256"], "SCHEDULER_SOURCE_SHA_INVALID")
        _sha(record["structural_config_sha256"], "STRUCTURAL_CONFIG_SHA_INVALID")
    declaration = record["causal_field_declaration"]
    if not isinstance(declaration, Mapping) or any(declaration.get(key) is not False for key in ("future_steps_consumed", "teacher_fields_consumed", "attack_fields_consumed")):
        raise FactorizedRuntimeError("CAUSAL_DECLARATION_INVALID")


def exact_runtime_step_join(
    prediction_rows: Iterable[Mapping[str, Any]],
    student_rows: Iterable[Mapping[str, Any]],
    runtime_rows: Iterable[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    def index(rows: Iterable[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            step = _step(row)
            if step in result:
                raise FactorizedRuntimeError(f"DUPLICATE_{name.upper()}_STEP")
            result[step] = row
        return result

    prediction = index(prediction_rows, "prediction")
    student = index(student_rows, "student")
    runtime = index(runtime_rows, "runtime")
    if set(prediction) != set(student) or set(prediction) != set(runtime):
        raise FactorizedRuntimeError("STEP_SET_MISMATCH")
    if sorted(prediction) != list(range(len(prediction))):
        raise FactorizedRuntimeError("STEP_SEQUENCE_INVALID")
    return [(prediction[i], student[i], runtime[i]) for i in range(len(prediction))]


__all__ = [
    "FORBIDDEN_RUNTIME_FIELDS", "OPTIONAL_RUNTIME_FIELDS", "RUNTIME_FIELDS", "RUNTIME_SCHEMA", "FactorizedRuntimeError",
    "build_runtime_record", "exact_runtime_step_join", "validate_runtime_record",
]
