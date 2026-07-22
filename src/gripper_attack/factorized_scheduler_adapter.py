"""Single, fail-closed application point for Factorized scheduler contracts.

This module applies externally supplied calibration values and runs the real
scheduler.  It never fits calibration, reads Teacher labels, or mutates an
action path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .factorized_scheduler import FactorizedSchedulerConfig, FactorizedV2OneShotScheduler

CALIBRATION_V2 = "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2"
CALIBRATION_V3 = "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3"
HEADS = ("grasp", "manipulation", "release")
SPLIT_RE = re.compile(r"o[0-3]_i[0-2]")
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")


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


def _commit(value: Any, code: str = "STUDENT_SOURCE_COMMIT_INVALID") -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise FactorizedSchedulerAdapterError(code)
    return value.lower()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _validate_common_top(value: Mapping[str, Any], *, schema: str) -> None:
    if not isinstance(value.get("split"), str) or not SPLIT_RE.fullmatch(value["split"]):
        raise FactorizedSchedulerAdapterError(f"{schema}_SPLIT")
    _sha(value.get("checkpoint_sha256"), "CHECKPOINT_SHA_INVALID")
    _sha(value.get("scheduler_source_sha256"), "SCHEDULER_SOURCE_SHA_INVALID")
    _sha(value.get("structural_config_sha256"), "STRUCTURAL_CONFIG_SHA_INVALID")
    _sha(value.get("feature_order_sha256"), "FEATURE_ORDER_SHA_INVALID")
    _commit(value.get("student_source_commit"))


def _validate_head_v2(head: Mapping[str, Any]) -> None:
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


def _validate_head_v3(head: Mapping[str, Any]) -> None:
    required_head = {
        "method", "a", "b", "threshold", "transform", "method_valid",
        "transform_valid", "fit_data_valid", "provenance_class", "fit_manifest_sha256",
        "policy_selection_manifest_sha256",
    }
    if set(head) != required_head:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V3_HEAD_SCHEMA")
    if head["provenance_class"] not in {"INDEPENDENT_CALIBRATION", "TRAIN_RESUBSTITUTION_CALIBRATION", "UNKNOWN"}:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V3_PROVENANCE_INVALID")
    _validate_head_v2({
        **{key: value for key, value in head.items() if key != "provenance_class"},
        "provenance": head["provenance_class"],
    })


def validate_calibration_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "checkpoint_sha256", "split", "scheduler_source_sha256",
        "structural_config_sha256", "student_source_commit", "feature_order_sha256",
        "grasp", "manipulation", "release", "formal_selection_eligible",
        "training_authorized", "attack_authorized",
    }
    if set(value) != required or value.get("schema") != CALIBRATION_V2:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V2_SCHEMA")
    _validate_common_top(value, schema="CALIBRATION_V2")
    if any(value.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "attack_authorized")):
        raise FactorizedSchedulerAdapterError("CALIBRATION_AUTHORIZATION_NOT_DISABLED")
    for head_name in HEADS:
        head = value.get(head_name)
        if not isinstance(head, Mapping):
            raise FactorizedSchedulerAdapterError("CALIBRATION_HEAD_MISSING")
        _validate_head_v2(head)
    return dict(value)


def validate_calibration_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "status", "split", "checkpoint_sha256", "scheduler_source_sha256",
        "structural_config_sha256", "student_source_commit", "feature_order_sha256",
        "calibration_fit_authoritative", "threshold_selection_authoritative",
        "l3_evaluation_eligible", "training_authorized", "full_fit_authorized",
        "attack_authorized", "grasp", "manipulation", "release",
    }
    if set(value) != required or value.get("schema") != CALIBRATION_V3:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V3_SCHEMA")
    _validate_common_top(value, schema="CALIBRATION_V3")
    if value.get("status") not in {"DIAGNOSTIC", "AUTHORITATIVE"}:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V3_STATUS")
    for flag in ("calibration_fit_authoritative", "threshold_selection_authoritative", "l3_evaluation_eligible", "training_authorized", "full_fit_authorized", "attack_authorized"):
        if not isinstance(value.get(flag), bool):
            raise FactorizedSchedulerAdapterError("CALIBRATION_V3_FLAG")
    if any(value.get(flag) is not False for flag in ("training_authorized", "full_fit_authorized", "attack_authorized")):
        raise FactorizedSchedulerAdapterError("CALIBRATION_AUTHORIZATION_NOT_DISABLED")
    for head_name in HEADS:
        head = value.get(head_name)
        if not isinstance(head, Mapping):
            raise FactorizedSchedulerAdapterError("CALIBRATION_V3_HEAD_MISSING")
        _validate_head_v3(head)
    authoritative = all(value.get(flag) is True for flag in ("calibration_fit_authoritative", "threshold_selection_authoritative", "l3_evaluation_eligible"))
    if value["status"] == "AUTHORITATIVE" and not authoritative:
        raise FactorizedSchedulerAdapterError("CALIBRATION_V3_AUTHORITATIVE_FLAGS")
    if value["l3_evaluation_eligible"]:
        if not authoritative or any(value[name]["provenance_class"] != "INDEPENDENT_CALIBRATION" or value[name]["fit_data_valid"] is not True for name in HEADS):
            raise FactorizedSchedulerAdapterError("CALIBRATION_V3_L3_PROVENANCE")
    return dict(value)


def validate_calibration_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == CALIBRATION_V2:
        return validate_calibration_v2(value)
    if schema == CALIBRATION_V3:
        return validate_calibration_v3(value)
    raise FactorizedSchedulerAdapterError("CALIBRATION_SCHEMA_UNSUPPORTED")


def apply_calibration(head: Mapping[str, Any], raw_logit: Any) -> float:
    """Apply exactly sigmoid(a * raw_logit + b); never fit or choose values."""
    method = head.get("method")
    if method not in {"RAW", "INTERCEPT_ONLY", "PLATT"} or head.get("transform") != "probability=sigmoid(a*raw_logit+b)" or head.get("method_valid") is not True or head.get("transform_valid") is not True:
        raise FactorizedSchedulerAdapterError("CALIBRATION_METHOD_INVALID")
    z = _finite(raw_logit, "RAW_LOGIT_INVALID")
    a = _finite(head.get("a"), "CALIBRATION_A_INVALID")
    b = _finite(head.get("b"), "CALIBRATION_B_INVALID")
    return _sigmoid(a * z + b)


class FactorizedV2SchedulerAdapter:
    """Apply a sealed contract and expose the canonical scheduler trace."""

    def __init__(self, structure: Mapping[str, Any], calibration_contract: Mapping[str, Any], require_l3_eligible: bool = False):
        self.calibration = validate_calibration_contract(calibration_contract)
        self.calibration_schema = self.calibration["schema"]
        self.require_l3_eligible = bool(require_l3_eligible)
        if self.require_l3_eligible:
            if self.calibration_schema != CALIBRATION_V3 or not self.calibration.get("calibration_fit_authoritative") or not self.calibration.get("threshold_selection_authoritative") or not self.calibration.get("l3_evaluation_eligible"):
                raise FactorizedSchedulerAdapterError("L3_ELIGIBILITY_REQUIRED")
            if any(self.calibration[name].get("provenance_class") != "INDEPENDENT_CALIBRATION" or self.calibration[name].get("fit_data_valid") is not True for name in HEADS):
                raise FactorizedSchedulerAdapterError("L3_INDEPENDENT_CALIBRATION_REQUIRED")
        self.config = FactorizedSchedulerConfig.from_mapping(structure, self.calibration)
        self.scheduler = FactorizedV2OneShotScheduler(self.config)

    @property
    def l3_evaluation_eligible(self) -> bool:
        return bool(self.calibration.get("l3_evaluation_eligible", False)) if self.calibration_schema == CALIBRATION_V3 else False

    def reset(self) -> None:
        self.scheduler.reset()

    def _validate_runtime_binding(self, row: Mapping[str, Any]) -> None:
        if self.calibration_schema != CALIBRATION_V3:
            return
        required = ("checkpoint_sha256", "source_commit", "feature_order_sha256", "split", "scheduler_source_sha256", "structural_config_sha256")
        if any(row.get(key) in (None, "") for key in required):
            raise FactorizedSchedulerAdapterError("RUNTIME_CONTRACT_BINDING_MISSING")
        comparisons = (
            ("checkpoint_sha256", "checkpoint_sha256", "CHECKPOINT_BINDING_MISMATCH"),
            ("source_commit", "student_source_commit", "SOURCE_COMMIT_BINDING_MISMATCH"),
            ("feature_order_sha256", "feature_order_sha256", "FEATURE_ORDER_BINDING_MISMATCH"),
            ("split", "split", "SPLIT_BINDING_MISMATCH"),
            ("scheduler_source_sha256", "scheduler_source_sha256", "SCHEDULER_SOURCE_BINDING_MISMATCH"),
            ("structural_config_sha256", "structural_config_sha256", "STRUCTURAL_CONFIG_BINDING_MISMATCH"),
        )
        for runtime_name, contract_name, code in comparisons:
            runtime_value = str(row[runtime_name]).lower()
            contract_value = str(self.calibration[contract_name]).lower()
            if runtime_value != contract_value:
                raise FactorizedSchedulerAdapterError(code)

    def step(self, runtime_record: Mapping[str, Any]) -> dict[str, Any]:
        forbidden = {
            "utility_probability", "regrasp_probability", "event_id", "teacher_phase", "known_mask",
            "strict_k10_feasible", "strict_k10_known_mask", "future_fields", "attack_outcome",
            "teacher_label", "object_state", "contact", "executed_action", "mutated_action",
        }
        if forbidden & set(runtime_record):
            raise FactorizedSchedulerAdapterError("FORBIDDEN_RUNTIME_FIELD")
        self._validate_runtime_binding(runtime_record)
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
        trace["calibration_schema"] = self.calibration_schema
        trace["calibration_split"] = self.calibration["split"]
        trace["probabilities"] = probabilities
        trace["l3_evaluation_eligible"] = self.l3_evaluation_eligible
        trace["diagnostic_only"] = not self.l3_evaluation_eligible
        return trace

    def run_episode(self, runtime_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(runtime_rows)
        if not rows:
            raise FactorizedSchedulerAdapterError("EPISODE_EMPTY")
        steps = [row.get("step") for row in rows]
        if steps != list(range(len(rows))):
            raise FactorizedSchedulerAdapterError("EPISODE_STEP_SEQUENCE_INVALID")
        self.reset()
        traces: list[dict[str, Any]] = []
        first_emit_step: int | None = None
        first_emit_trace: dict[str, Any] | None = None
        for row in rows:
            trace = self.step(row)
            traces.append(trace)
            if trace.get("emit") is True and first_emit_step is None:
                first_emit_step = int(trace["step"])
                first_emit_trace = dict(trace)
        return {
            "per_step_trace": traces,
            "ever_emitted": first_emit_step is not None,
            "first_emit_step": first_emit_step,
            "first_emit_trace": first_emit_trace,
            "final_state": traces[-1]["state_after"],
            "reason_histogram": dict(sorted(Counter(str(trace["reason"]) for trace in traces).items())),
            "l3_evaluation_eligible": self.l3_evaluation_eligible,
            "diagnostic_only": not self.l3_evaluation_eligible,
        }


__all__ = [
    "CALIBRATION_V2", "CALIBRATION_V3", "FactorizedSchedulerAdapterError", "FactorizedV2SchedulerAdapter",
    "apply_calibration", "validate_calibration_contract", "validate_calibration_v2", "validate_calibration_v3",
]
