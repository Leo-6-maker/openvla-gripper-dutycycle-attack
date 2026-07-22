"""Fail-closed, detector-only Factorized V2 one-shot scheduler."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


STRUCTURE_SCHEMA = "FACTORIZED_V2_SCHEDULER_STRUCTURE_V1"
CALIBRATION_SCHEMA_V1 = "FACTORIZED_V2_CALIBRATION_CONTRACT_V1"
CALIBRATION_SCHEMA_V2 = "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2"
CALIBRATION_SCHEMA = CALIBRATION_SCHEMA_V2


class FactorizedSchedulerError(ValueError):
    pass


class FactorizedState(str, Enum):
    IDLE = "IDLE"
    GRASP_EVIDENCE = "GRASP_EVIDENCE"
    MANIPULATION_READY = "MANIPULATION_READY"
    EMIT_ONCE = "EMIT_ONCE"
    DONE = "DONE"


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorizedSchedulerError(f"{name}_MISSING_OR_INVALID")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise FactorizedSchedulerError(f"{name}_MISSING_OR_INVALID")
    return value


def _threshold(value: Any, name: str) -> float:
    return _probability(value, name)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise FactorizedSchedulerError(f"{name}_SHA_INVALID")
    return value.lower()


@dataclass(frozen=True)
class FactorizedSchedulerConfig:
    grasp_threshold: float
    manipulation_threshold: float
    release_veto_threshold: float
    candidate_dwell: int
    candidate_dwell_counts_before_grasp: bool
    persistence_window: int
    persistence_required: int
    warmup_steps: int
    invalid_step_policy: str
    attack_enabled: bool = False
    calibration_status: str = "TEST_ONLY_NOT_SELECTION_ELIGIBLE"
    calibration_checkpoint_sha256: str = ""
    calibration_fit_manifest_sha256: str = ""

    @classmethod
    def from_mapping(cls, structure: Mapping[str, Any], calibration: Mapping[str, Any] | None = None) -> "FactorizedSchedulerConfig":
        if calibration is None:
            raise FactorizedSchedulerError("CALIBRATION_CONTRACT_REQUIRED")
        structure_keys = {
            "schema", "candidate_dwell", "candidate_dwell_counts_before_grasp",
            "persistence_window", "persistence_required", "warmup_steps",
            "invalid_step_policy", "attack_enabled", "formal_selection_eligible",
            "training_authorized", "attack_authorized",
        }
        if set(structure) != structure_keys or structure.get("schema") != STRUCTURE_SCHEMA:
            raise FactorizedSchedulerError("STRUCTURAL_CONFIG_SCHEMA")
        if any(structure.get(key) is not False for key in ("attack_enabled", "formal_selection_eligible", "training_authorized", "attack_authorized")):
            raise FactorizedSchedulerError("STRUCTURAL_AUTHORIZATION_NOT_DISABLED")
        if structure.get("invalid_step_policy") != "reset":
            raise FactorizedSchedulerError("INVALID_STEP_POLICY")
        ints = ("candidate_dwell", "persistence_window", "persistence_required", "warmup_steps")
        if any(isinstance(structure[key], bool) or not isinstance(structure[key], int) or structure[key] < 0 for key in ints):
            raise FactorizedSchedulerError("STRUCTURAL_INTEGER_INVALID")
        if structure["candidate_dwell"] < 1 or structure["persistence_window"] < 1 or not 1 <= structure["persistence_required"] <= structure["persistence_window"]:
            raise FactorizedSchedulerError("STRUCTURAL_PERSISTENCE_INVALID")
        if not isinstance(structure["candidate_dwell_counts_before_grasp"], bool):
            raise FactorizedSchedulerError("DWELL_SEMANTICS_INVALID")

        calibration_schema = calibration.get("schema")
        if calibration_schema == CALIBRATION_SCHEMA_V2:
            calibration_keys = {
                "schema", "checkpoint_sha256", "split", "scheduler_source_sha256",
                "structural_config_sha256", "student_source_commit", "feature_order_sha256",
                "grasp", "manipulation", "release", "formal_selection_eligible",
                "training_authorized", "attack_authorized",
            }
            if set(calibration) != calibration_keys:
                raise FactorizedSchedulerError("CALIBRATION_CONTRACT_SCHEMA")
            if not re.fullmatch(r"o[0-3]_i[0-2]", str(calibration.get("split", ""))):
                raise FactorizedSchedulerError("CALIBRATION_SPLIT_INVALID")
            checkpoint_sha = _sha(calibration.get("checkpoint_sha256"), "CALIBRATION_CHECKPOINT")
            fit_manifest_sha = _sha(calibration.get("feature_order_sha256"), "CALIBRATION_FEATURE_ORDER")
            _sha(calibration.get("scheduler_source_sha256"), "CALIBRATION_SCHEDULER_SOURCE")
            _sha(calibration.get("structural_config_sha256"), "CALIBRATION_STRUCTURAL_CONFIG")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", str(calibration.get("student_source_commit", ""))):
                raise FactorizedSchedulerError("CALIBRATION_STUDENT_COMMIT_INVALID")
            if any(calibration.get(key) is not False for key in ("formal_selection_eligible", "training_authorized", "attack_authorized")):
                raise FactorizedSchedulerError("CALIBRATION_AUTHORIZATION_NOT_DISABLED")
            heads: dict[str, Mapping[str, Any]] = {}
            for head in ("grasp", "manipulation", "release"):
                value = calibration.get(head)
                if not isinstance(value, Mapping):
                    raise FactorizedSchedulerError("CALIBRATION_HEAD_INVALID")
                if value.get("method") not in {"RAW", "INTERCEPT_ONLY", "PLATT"}:
                    raise FactorizedSchedulerError("CALIBRATION_METHOD_INVALID")
                if value.get("transform") != "probability=sigmoid(a*raw_logit+b)":
                    raise FactorizedSchedulerError("CALIBRATION_TRANSFORM_INVALID")
                if value.get("method_valid") is not True or value.get("transform_valid") is not True:
                    raise FactorizedSchedulerError("CALIBRATION_METHOD_NOT_VALID")
                for name in ("a", "b"):
                    if isinstance(value.get(name), bool) or not isinstance(value.get(name), (int, float)) or not isfinite(float(value[name])):
                        raise FactorizedSchedulerError("CALIBRATION_PARAMETER_INVALID")
                _sha(value.get("fit_manifest_sha256"), "CALIBRATION_FIT_MANIFEST")
                _sha(value.get("policy_selection_manifest_sha256"), "CALIBRATION_POLICY_SELECTION")
                heads[head] = value
            thresholds = {
                "grasp_threshold": _threshold(heads["grasp"].get("threshold"), "GRASP_THRESHOLD"),
                "manipulation_threshold": _threshold(heads["manipulation"].get("threshold"), "MANIPULATION_THRESHOLD"),
                "release_veto_threshold": _threshold(heads["release"].get("threshold"), "RELEASE_THRESHOLD"),
            }
            calibration_status = "SEALED_EXTERNAL_CALIBRATION"
        else:
            # Historical V1 is retained only so old synthetic fixtures remain
            # replayable; V3 production paths use the V2 contract above.
            calibration_keys = {
                "schema", "status", "checkpoint_sha256", "fit_manifest_sha256", "grasp",
                "manipulation", "release", "formal_selection_eligible",
                "training_authorized", "attack_authorized",
            }
            if set(calibration) != calibration_keys or calibration_schema != CALIBRATION_SCHEMA_V1:
                raise FactorizedSchedulerError("CALIBRATION_CONTRACT_SCHEMA")
            if calibration.get("status") not in {"TEST_ONLY_NOT_SELECTION_ELIGIBLE", "SEALED_EXTERNAL_CALIBRATION"}:
                raise FactorizedSchedulerError("CALIBRATION_STATUS_INVALID")
            if any(calibration.get(key) is not False for key in ("formal_selection_eligible", "training_authorized", "attack_authorized")):
                raise FactorizedSchedulerError("CALIBRATION_AUTHORIZATION_NOT_DISABLED")
            checkpoint_sha = _sha(calibration.get("checkpoint_sha256"), "CALIBRATION_CHECKPOINT")
            fit_manifest_sha = _sha(calibration.get("fit_manifest_sha256"), "CALIBRATION_FIT_MANIFEST")
            heads = {}
            for head in ("grasp", "manipulation", "release"):
                value = calibration.get(head)
                if not isinstance(value, Mapping):
                    raise FactorizedSchedulerError("CALIBRATION_HEAD_INVALID")
                heads[head] = value
            thresholds = {
                "grasp_threshold": _threshold(heads["grasp"].get("threshold"), "GRASP_THRESHOLD"),
                "manipulation_threshold": _threshold(heads["manipulation"].get("threshold"), "MANIPULATION_THRESHOLD"),
                "release_veto_threshold": _threshold(heads["release"].get("threshold"), "RELEASE_THRESHOLD"),
            }
            for head in ("grasp", "manipulation", "release"):
                temperature = heads[head].get("temperature")
                if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not isfinite(float(temperature)) or float(temperature) <= 0:
                    raise FactorizedSchedulerError("CALIBRATION_TEMPERATURE_INVALID")
            calibration_status = calibration["status"]
        return cls(
            **thresholds,
            candidate_dwell=structure["candidate_dwell"],
            candidate_dwell_counts_before_grasp=structure["candidate_dwell_counts_before_grasp"],
            persistence_window=structure["persistence_window"],
            persistence_required=structure["persistence_required"],
            warmup_steps=structure["warmup_steps"],
            invalid_step_policy="reset",
            attack_enabled=False,
            calibration_status=calibration_status,
            calibration_checkpoint_sha256=checkpoint_sha,
            calibration_fit_manifest_sha256=fit_manifest_sha,
        )

    @classmethod
    def from_files(cls, structure_path: Path, calibration_path: Path) -> "FactorizedSchedulerConfig":
        return cls.from_mapping(
            json.loads(Path(structure_path).read_text(encoding="utf-8")),
            json.loads(Path(calibration_path).read_text(encoding="utf-8")),
        )


@dataclass(frozen=True)
class FactorizedStep:
    step: int
    candidate_close: bool
    action_known: bool
    student_valid: bool
    route_supported: bool
    grasp_probability: float
    manipulation_probability: float
    release_probability: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FactorizedStep":
        forbidden = {
            "event_id", "event_role", "teacher_phase", "teacher_label", "known_mask",
            "grasp_known", "manipulation_known", "release_known", "strict_k10_feasible",
            "strict_k10_known_mask", "future_score", "attack_outcome", "object_state",
            "action", "clean_action", "executed_action", "utility_probability", "regrasp_probability",
        }
        if forbidden & set(value):
            raise FactorizedSchedulerError("FORBIDDEN_RUNTIME_FIELD")
        required = ("step", "candidate_close", "action_known", "student_valid", "route_supported")
        if any(name not in value for name in required):
            raise FactorizedSchedulerError("RUNTIME_FIELD_MISSING")
        for name in ("candidate_close", "action_known", "student_valid", "route_supported"):
            if not isinstance(value[name], bool):
                raise FactorizedSchedulerError("RUNTIME_BOOL_INVALID")
        if isinstance(value["step"], bool) or not isinstance(value["step"], int) or value["step"] < 0:
            raise FactorizedSchedulerError("RUNTIME_STEP_INVALID")
        return cls(
            value["step"], value["candidate_close"], value["action_known"], value["student_valid"],
            value["route_supported"], _probability(value.get("grasp_probability"), "GRASP_PROBABILITY"),
            _probability(value.get("manipulation_probability"), "MANIPULATION_PROBABILITY"),
            _probability(value.get("release_probability"), "RELEASE_PROBABILITY"),
        )


@dataclass
class FactorizedV2OneShotScheduler:
    config: FactorizedSchedulerConfig
    state: FactorizedState = FactorizedState.IDLE
    emitted: bool = False
    step_count: int = 0
    dwell: int = 0
    manipulation_history: list[bool] = field(default_factory=list)

    def reset(self) -> None:
        self.state = FactorizedState.IDLE
        self.emitted = False
        self.step_count = 0
        self.dwell = 0
        self.manipulation_history.clear()

    def _reset(self, state: FactorizedState = FactorizedState.IDLE) -> None:
        self.state = state
        self.dwell = 0
        self.manipulation_history.clear()

    def _trace(self, item: FactorizedStep, before: FactorizedState, *, emit: bool, reason: str) -> dict[str, Any]:
        return {
            "state_before": before.value,
            "state_after": self.state.value,
            "step": item.step,
            "candidate_close": item.candidate_close,
            "action_known": item.action_known,
            "student_valid": item.student_valid,
            "grasp_probability": item.grasp_probability,
            "manipulation_probability": item.manipulation_probability,
            "release_probability": item.release_probability,
            "dwell": self.dwell,
            "candidate_dwell_counts_before_grasp": self.config.candidate_dwell_counts_before_grasp,
            "persistence_history": list(self.manipulation_history),
            "release_veto": item.release_probability >= self.config.release_veto_threshold,
            "emit": emit,
            "reason": reason,
            "attack_enabled": self.config.attack_enabled,
        }

    def step(self, record: Mapping[str, Any]) -> dict[str, Any]:
        item = FactorizedStep.from_mapping(record)
        before = self.state
        self.step_count += 1
        if self.emitted:
            self.state = FactorizedState.DONE
            return self._trace(item, before, emit=False, reason="ONE_SHOT_LATCHED")
        if item.step < self.config.warmup_steps:
            self._reset()
            return self._trace(item, before, emit=False, reason="WARMUP")
        if not item.route_supported:
            self._reset()
            return self._trace(item, before, emit=False, reason="UNSUPPORTED_ROUTE")
        if not item.student_valid:
            self._reset()
            return self._trace(item, before, emit=False, reason="STUDENT_INVALID")
        if not item.action_known:
            self._reset()
            return self._trace(item, before, emit=False, reason="ACTION_UNKNOWN")
        if not item.candidate_close:
            self._reset()
            return self._trace(item, before, emit=False, reason="NO_CLOSE_INTENT")

        self.dwell += 1
        if item.grasp_probability < self.config.grasp_threshold:
            self.state = FactorizedState.GRASP_EVIDENCE
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="GRASP_BELOW_THRESHOLD")
        if item.release_probability >= self.config.release_veto_threshold:
            self._reset()
            return self._trace(item, before, emit=False, reason="RELEASE_VETO")
        active = item.manipulation_probability >= self.config.manipulation_threshold
        self.manipulation_history.append(active)
        self.manipulation_history = self.manipulation_history[-self.config.persistence_window:]
        self.state = FactorizedState.MANIPULATION_READY
        persistent = len(self.manipulation_history) >= self.config.persistence_required and sum(self.manipulation_history) >= self.config.persistence_required
        if self.dwell >= self.config.candidate_dwell and persistent:
            self.state = FactorizedState.EMIT_ONCE
            self.emitted = True
            self.state = FactorizedState.DONE
            return self._trace(item, before, emit=True, reason="FACTOR_CRITICAL_WINDOW")
        return self._trace(item, before, emit=False, reason="PERSISTENCE_OR_DWELL_INCOMPLETE")


__all__ = [
    "CALIBRATION_SCHEMA", "CALIBRATION_SCHEMA_V1", "CALIBRATION_SCHEMA_V2",
    "FactorizedSchedulerConfig", "FactorizedSchedulerError",
    "FactorizedState", "FactorizedStep", "FactorizedV2OneShotScheduler", "STRUCTURE_SCHEMA",
]
