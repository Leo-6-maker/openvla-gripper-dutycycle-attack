"""Causal, candidate-gated, one-shot V5 scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V5SchedulerConfig:
    utility_threshold: float = 0.5
    release_veto_threshold: float = 0.5
    regrasp_veto_threshold: float = 0.5
    uncertainty_veto_threshold: float = 0.5
    release_veto_enabled: bool = True
    regrasp_veto_enabled: bool = True
    uncertainty_veto_enabled: bool = False
    minimum_candidate_dwell: int = 10
    persistence_window: int = 5
    persistence_required: int = 3

    def __post_init__(self) -> None:
        for value in (
            self.utility_threshold,
            self.release_veto_threshold,
            self.regrasp_veto_threshold,
            self.uncertainty_veto_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("V5 scheduler thresholds must be in [0,1]")
        if self.persistence_window != 5 or self.persistence_required != 3 or self.minimum_candidate_dwell != 10:
            raise ValueError("V5 development freezes a 3-of-5 persistence rule")


class V5OneShotScheduler:
    def __init__(self, config: V5SchedulerConfig = V5SchedulerConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = "IDLE"
        self.emitted = False
        self.history: deque[tuple[bool, float]] = deque(maxlen=self.config.persistence_window)
        self.emit_step = -1
        self.candidate_dwell = 0

    def update(
        self,
        *,
        step: int,
        candidate_close: bool,
        valid: bool,
        utility_probability: float,
        release_probability: float,
        regrasp_probability: float,
        uncertainty_probability: float,
    ) -> dict[str, Any]:
        if self.emitted:
            return self._result(step, False)
        if not valid or not candidate_close:
            self.history.clear()
            self.state = "IDLE"
            self.candidate_dwell = 0
            return self._result(step, False)
        self.candidate_dwell += 1
        veto = (
            (self.config.release_veto_enabled and release_probability >= self.config.release_veto_threshold)
            or (self.config.regrasp_veto_enabled and regrasp_probability >= self.config.regrasp_veto_threshold)
            or (self.config.uncertainty_veto_enabled and uncertainty_probability >= self.config.uncertainty_veto_threshold)
        )
        eligible = (not veto) and utility_probability >= self.config.utility_threshold
        self.history.append((eligible, float(utility_probability)))
        self.state = "PEAK_WAIT" if eligible else "ARMED"
        recent_eligible = sum(flag for flag, _ in self.history)
        previous_scores = [score for flag, score in list(self.history)[:-1] if flag]
        is_local_peak = not previous_scores or utility_probability >= max(previous_scores)
        emit = (
            self.candidate_dwell >= self.config.minimum_candidate_dwell
            and recent_eligible >= self.config.persistence_required
            and is_local_peak
        )
        if emit:
            self.emitted = True
            self.emit_step = int(step)
            self.state = "EMIT_ONCE"
        return self._result(step, emit)

    def _result(self, step: int, emit: bool) -> dict[str, Any]:
        return {
            "step": int(step),
            "state": "DONE" if self.emitted else self.state,
            "emit": bool(emit),
            "one_shot_emitted": bool(self.emitted),
            "emit_step": self.emit_step,
            "candidate_dwell": self.candidate_dwell,
            "uncertainty_veto_enabled": self.config.uncertainty_veto_enabled,
            "release_veto_enabled": self.config.release_veto_enabled,
            "regrasp_veto_enabled": self.config.regrasp_veto_enabled,
            "history": [{"eligible": flag, "utility_probability": score} for flag, score in self.history],
            "teacher_inputs_consumed": False,
            "attack_enabled": False,
        }


__all__ = ["V5SchedulerConfig", "V5OneShotScheduler"]
