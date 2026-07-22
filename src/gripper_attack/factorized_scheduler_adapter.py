"""Single application point for the sealed Factorized V2 calibration contract."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .factorized_scheduler import FactorizedSchedulerConfig, FactorizedV2OneShotScheduler

CALIBRATION_V2 = "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2"
HEADS = ("grasp", "manipulation", "release")


class FactorizedSchedulerAdapterError(ValueError):
    pass


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FactorizedSchedulerAdapterError(code)
    return float(value)


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise FactorizedSchedulerAdapterError(code)
    return value.lower()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def validate_calibration_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "checkpoint_sha256", "split", "scheduler_source_sha256",
        "structural_config_sha256", "student_source_commit", "feature_order_sha256",
        "grasp", "manipulation", "release", "formal_selection_eligible",
        "training_authorized", "attack_authorized",
    }
    if set(value) != required or value.get("schema") != CALIBRATION_V2:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V2_SCHEMA")
    if not isinstance(value.get("split"), str) or not re.fullmatch(r"o[0-3]_i[0-2]", value["split"]):
        raise FactorizedSchedulerAdapterError("CALIBRATION_V2_SPLIT")
    _sha(value.get("checkpoint_sha256"), "CHECKPOINT_SHA_INVALID")
    _sha(value.get("scheduler_source_sha256"), "SCHEDULER_SOURCE_SHA_INVALID")
    _sha(value.get("structural_config_sha256"), "STRUCTURAL_CONFIG_SHA_INVALID")
    _sha(value.get("feature_order_sha256"), "FEATURE_ORDER_SHA_INVALID")
    if not isinstance(value.get("student_source_commit"), str) or len(value["student_source_commit"]) != 40:
        raise FactorizedSchedulerAdapterError("STUDENT_SOURCE_COMMIT_INVALID")
    if any(value.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "attack_authorized")):
        raise FactorizedSchedulerAdapterError("CALIBRATION_AUTHORIZATION_NOT_DISABLED")
    for head_name in HEADS:
        head = value.get(head_name)
        if not isinstance(head, Mapping):
            raise FactorizedSchedulerAdapterError("CALIBRATION_HEAD_MISSING")
        required_head = {
            "method", "a", "b", "threshold", "transform", "method_valid",
            "transform_valid", "fit_data_valid", "provenance", "fit_manifest_sha256",
            "policy_selection_manifest_sha256",
        }
        if set(head) != required_head:
            raise FactorizedSchedulerAdapterError("CALIBRATION_HEAD_SCHEMA")
        if head["method"] not in {"RAW", "INTERCEPT_ONLY", "PLATT"}:
            raise FactorizedSchedulerAdapterError("CALIBRATION_METHOD_INVALID")
        if head["transform"] != "probability=sigmoid(a*raw_logit+b)" or head["method_valid"] is not True or head["transform_valid"] is not True:
            raise FactorizedSchedulerAdapterError("CALIBRATION_TRANSFORM_INVALID")
        _finite(head["a"], "CALIBRATION_A_INVALID")
        _finite(head["b"], "CALIBRATION_B_INVALID")
        threshold = _finite(head["threshold"], "CALIBRATION_THRESHOLD_INVALID")
        if not 0.0 <= threshold <= 1.0:
            raise FactorizedSchedulerAdapterError("CALIBRATION_THRESHOLD_INVALID")
        if not isinstance(head["fit_data_valid"], bool) or not isinstance(head["provenance"], (str, Mapping)):
            raise FactorizedSchedulerAdapterError("CALIBRATION_PROVENANCE_INVALID")
        _sha(head["fit_manifest_sha256"], "FIT_MANIFEST_SHA_INVALID")
        _sha(head["policy_selection_manifest_sha256"], "POLICY_SELECTION_SHA_INVALID")
    return dict(value)


def apply_calibration(head: Mapping[str, Any], raw_logit: Any) -> float:
    """Apply exactly sigmoid(a * raw_logit + b); never fit or choose values."""
    validate_head = {**head}
    if validate_head.get("method") not in {"RAW", "INTERCEPT_ONLY", "PLATT"} or validate_head.get("transform") != "probability=sigmoid(a*raw_logit+b)" or validate_head.get("method_valid") is not True or validate_head.get("transform_valid") is not True:
        raise FactorizedSchedulerAdapterError("CALIBRATION_METHOD_INVALID")
    z = _finite(raw_logit, "RAW_LOGIT_INVALID")
    a = _finite(validate_head.get("a"), "CALIBRATION_A_INVALID")
    b = _finite(validate_head.get("b"), "CALIBRATION_B_INVALID")
    return _sigmoid(a * z + b)


class FactorizedV2SchedulerAdapter:
    """Adapter used by DeepSeek; threshold/calibration values are external."""

    def __init__(self, structure: Mapping[str, Any], calibration: Mapping[str, Any]):
        self.calibration = validate_calibration_v2(calibration)
        self.config = FactorizedSchedulerConfig.from_mapping(structure, self.calibration)
        self.scheduler = FactorizedV2OneShotScheduler(self.config)

    def reset(self) -> None:
        self.scheduler.reset()

    def step(self, runtime_record: Mapping[str, Any]) -> dict[str, Any]:
        forbidden = {"utility_probability", "regrasp_probability", "event_id", "teacher_phase", "known_mask", "future_fields", "attack_outcome"}
        if forbidden & set(runtime_record):
            raise FactorizedSchedulerAdapterError("FORBIDDEN_RUNTIME_FIELD")
        row = dict(runtime_record)
        probabilities = {}
        for head_name in HEADS:
            logit_name = f"{head_name}_logit"
            if logit_name not in row:
                raise FactorizedSchedulerAdapterError(f"{head_name.upper()}_LOGIT_MISSING")
            probability = apply_calibration(self.calibration[head_name], row.pop(logit_name))
            row[f"{head_name}_probability"] = probability
            probabilities[head_name] = probability
        trace = self.scheduler.step(row)
        trace["calibration_schema"] = CALIBRATION_V2
        trace["calibration_split"] = self.calibration["split"]
        trace["probabilities"] = probabilities
        return trace


__all__ = [
    "CALIBRATION_V2", "FactorizedSchedulerAdapterError", "FactorizedV2SchedulerAdapter",
    "apply_calibration", "validate_calibration_v2",
]
