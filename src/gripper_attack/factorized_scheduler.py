"""Minimal, fail-closed scheduler for the Factorized V2 Student heads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


class FactorizedSchedulerError(ValueError):
    pass


class FactorizedState(str, Enum):
    IDLE = "IDLE"
    GRASP_EVIDENCE = "GRASP_EVIDENCE"
    MANIPULATION_READY = "MANIPULATION_READY"
    EMIT_ONCE = "EMIT_ONCE"
    DONE = "DONE"


@dataclass(frozen=True)
class FactorizedSchedulerConfig:
    grasp_threshold: float
    manipulation_threshold: float
    release_veto_threshold: float
    candidate_dwell: int
    persistence_window: int
    persistence_required: int
    warmup_steps: int
    unknown_policy: str
    attack_enabled: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FactorizedSchedulerConfig":
        required = {
            "schema",
            "grasp_threshold", "manipulation_threshold", "release_veto_threshold",
            "candidate_dwell", "persistence_window", "persistence_required",
            "warmup_steps", "unknown_policy", "attack_enabled",
        }
        if set(value) != required or value.get("schema") != "FACTORIZED_V2_SCHEDULER_PROTOCOL_V1":
            raise FactorizedSchedulerError("SCHEDULER_CONFIG_SCHEMA")
        if value["unknown_policy"] not in {"pause", "reset"}:
            raise FactorizedSchedulerError("UNKNOWN_POLICY_INVALID")
        ints = ("candidate_dwell", "persistence_window", "persistence_required", "warmup_steps")
        if any(isinstance(value[k], bool) or not isinstance(value[k], int) or value[k] < 0 for k in ints):
            raise FactorizedSchedulerError("SCHEDULER_INTEGER_CONFIG_INVALID")
        if value["candidate_dwell"] < 1 or value["persistence_window"] < 1 or not 1 <= value["persistence_required"] <= value["persistence_window"]:
            raise FactorizedSchedulerError("SCHEDULER_PERSISTENCE_CONFIG_INVALID")
        thresholds = ("grasp_threshold", "manipulation_threshold", "release_veto_threshold")
        if any(isinstance(value[k], bool) or not isinstance(value[k], (int, float)) or not isfinite(float(value[k])) or not 0 <= float(value[k]) <= 1 for k in thresholds):
            raise FactorizedSchedulerError("SCHEDULER_THRESHOLD_INVALID")
        if value["attack_enabled"] is not False:
            raise FactorizedSchedulerError("ATTACK_MUST_REMAIN_DISABLED")
        return cls(
            grasp_threshold=float(value["grasp_threshold"]),
            manipulation_threshold=float(value["manipulation_threshold"]),
            release_veto_threshold=float(value["release_veto_threshold"]),
            candidate_dwell=value["candidate_dwell"],
            persistence_window=value["persistence_window"],
            persistence_required=value["persistence_required"],
            warmup_steps=value["warmup_steps"],
            unknown_policy=value["unknown_policy"],
            attack_enabled=False,
        )

    @classmethod
    def from_json(cls, path: Path) -> "FactorizedSchedulerConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FactorizedStep:
    step: int
    candidate_close: bool
    action_known: bool
    student_valid: bool
    route_supported: bool
    grasp_probability: float
    grasp_known: bool
    manipulation_probability: float
    manipulation_known: bool
    release_probability: float
    release_known: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FactorizedStep":
        forbidden = {"event_id", "teacher_phase", "teacher_label", "future_score", "attack_outcome", "object_state", "action", "clean_action", "executed_action"}
        if forbidden & set(value):
            raise FactorizedSchedulerError("FORBIDDEN_SCHEDULER_FIELD")
        names = ("step", "candidate_close", "action_known", "student_valid", "route_supported", "grasp_known", "manipulation_known", "release_known")
        if any(name not in value for name in names):
            raise FactorizedSchedulerError("SCHEDULER_STEP_FIELD_MISSING")
        for name in ("candidate_close", "action_known", "student_valid", "route_supported", "grasp_known", "manipulation_known", "release_known"):
            if not isinstance(value[name], bool):
                raise FactorizedSchedulerError("SCHEDULER_STEP_BOOL_INVALID")
        probs = ("grasp_probability", "manipulation_probability", "release_probability")
        for name in probs:
            if name not in value or isinstance(value[name], bool) or not isinstance(value[name], (int, float)) or not isfinite(float(value[name])) or not 0 <= float(value[name]) <= 1:
                raise FactorizedSchedulerError("SCHEDULER_STEP_PROBABILITY_INVALID")
        if isinstance(value["step"], bool) or not isinstance(value["step"], int) or value["step"] < 0:
            raise FactorizedSchedulerError("SCHEDULER_STEP_INDEX_INVALID")
        return cls(value["step"], value["candidate_close"], value["action_known"], value["student_valid"], value["route_supported"], float(value["grasp_probability"]), value["grasp_known"], float(value["manipulation_probability"]), value["manipulation_known"], float(value["release_probability"]), value["release_known"])


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

    def _trace(self, item: FactorizedStep, before: FactorizedState, *, emit: bool, reason: str) -> dict[str, Any]:
        return {
            "state_before": before.value,
            "state_after": self.state.value,
            "step": item.step,
            "candidate_close": item.candidate_close,
            "action_known": item.action_known,
            "student_valid": item.student_valid,
            "grasp_probability": item.grasp_probability,
            "grasp_known": item.grasp_known,
            "manipulation_probability": item.manipulation_probability,
            "manipulation_known": item.manipulation_known,
            "release_probability": item.release_probability,
            "release_known": item.release_known,
            "dwell": self.dwell,
            "persistence_history": list(self.manipulation_history),
            "release_veto": bool(item.release_known and item.release_probability >= self.config.release_veto_threshold),
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
            self.state = FactorizedState.IDLE
            self.dwell = 0
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="WARMUP")
        if not item.route_supported:
            self.state = FactorizedState.IDLE
            self.dwell = 0
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="UNSUPPORTED_ROUTE")
        if not item.student_valid:
            self.state = FactorizedState.IDLE
            self.dwell = 0
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="STUDENT_INVALID")
        if not item.action_known or not item.candidate_close:
            self.state = FactorizedState.IDLE
            self.dwell = 0
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="NO_CLOSE_INTENT")
        self.dwell += 1
        if not item.grasp_known:
            self.state = FactorizedState.GRASP_EVIDENCE
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="GRASP_UNKNOWN")
        if item.grasp_probability < self.config.grasp_threshold:
            self.state = FactorizedState.GRASP_EVIDENCE
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="GRASP_BELOW_THRESHOLD")
        if not item.manipulation_known:
            self.state = FactorizedState.MANIPULATION_READY
            self.manipulation_history.clear()
            return self._trace(item, before, emit=False, reason="MANIPULATION_UNKNOWN")
        if not item.release_known:
            if self.config.unknown_policy == "reset":
                self.state = FactorizedState.IDLE
                self.dwell = 0
                self.manipulation_history.clear()
            else:
                self.state = FactorizedState.MANIPULATION_READY
                self.manipulation_history.append(False)
                self.manipulation_history = self.manipulation_history[-self.config.persistence_window:]
            return self._trace(item, before, emit=False, reason="RELEASE_UNKNOWN_PAUSE")
        if item.release_probability >= self.config.release_veto_threshold:
            self.state = FactorizedState.IDLE
            self.dwell = 0
            self.manipulation_history.clear()
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


__all__ = ["FactorizedSchedulerConfig", "FactorizedSchedulerError", "FactorizedState", "FactorizedStep", "FactorizedV2OneShotScheduler"]
