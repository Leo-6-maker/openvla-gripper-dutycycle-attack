"""Official V3 streaming Detector runtime and one-shot scheduler.

The runtime consumes only Student features.  Teacher labels, contact and
object sidecars are intentionally absent from this interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .b3_formal import B3Normalization, B3OfficialStatefulGRU


@dataclass(frozen=True)
class B3RuntimeThresholds:
    retention_active: float
    retention_continuation_t10: float
    release_imminent_veto: float
    persistence_steps: int = 3
    persistence_required: int = 2

    def __post_init__(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in (self.retention_active, self.retention_continuation_t10, self.release_imminent_veto)):
            raise ValueError("runtime thresholds must be probabilities")
        if self.persistence_steps != 3 or self.persistence_required != 2:
            raise ValueError("Official V3 fixes scheduler persistence to 2-of-3")


class B3V3OneShotScheduler:
    def __init__(self, thresholds: B3RuntimeThresholds) -> None:
        self.thresholds = thresholds
        self.history: deque[bool] = deque(maxlen=thresholds.persistence_steps)
        self.emitted = False

    def reset(self) -> None:
        self.history.clear()
        self.emitted = False

    def update(self, probabilities: dict[str, float], *, valid: bool = True) -> dict[str, Any]:
        if not valid:
            return {"emit": False, "one_shot_emitted": self.emitted, "scheduler_history": list(self.history)}
        eligible = (
            probabilities["retention_active"] >= self.thresholds.retention_active
            and probabilities["retention_continuation_t10"] >= self.thresholds.retention_continuation_t10
            and probabilities["release_imminent"] < self.thresholds.release_imminent_veto
        )
        self.history.append(bool(eligible))
        emit = not self.emitted and sum(self.history) >= self.thresholds.persistence_required
        if emit:
            self.emitted = True
        return {"emit": emit, "one_shot_emitted": self.emitted, "scheduler_history": list(self.history)}


class B3V3StreamingRuntime:
    """Stateful online adapter for a frozen B3 checkpoint."""

    def __init__(self, model: B3OfficialStatefulGRU, normalization: B3Normalization, thresholds: B3RuntimeThresholds) -> None:
        self.model = model.eval()
        self.normalization = normalization
        self.scheduler = B3V3OneShotScheduler(thresholds)
        self.hidden = None

    def reset_episode(self) -> None:
        self.hidden = None
        self.scheduler.reset()

    def step(self, features_25d: torch.Tensor, features_9d: torch.Tensor | None = None, *, valid: bool = True) -> dict[str, Any]:
        if features_25d.ndim == 1:
            features_25d = features_25d.unsqueeze(0)
        if features_25d.shape != (1, 25):
            raise ValueError("runtime expects one [1,25] Student feature row")
        mean25 = torch.tensor(self.normalization.mean_25d, dtype=features_25d.dtype, device=features_25d.device)
        std25 = torch.tensor(self.normalization.std_25d, dtype=features_25d.dtype, device=features_25d.device)
        x25 = (features_25d - mean25) / std25
        x9 = None
        if self.model.config.variant == "B3_25D9D":
            if features_9d is None:
                raise ValueError("B3_25D9D runtime requires the independent 9D stream")
            if features_9d.ndim == 1:
                features_9d = features_9d.unsqueeze(0)
            if features_9d.shape != (1, 9):
                raise ValueError("runtime expects one [1,9] policy-intent row")
            mean9 = torch.tensor(self.normalization.mean_9d, dtype=features_9d.dtype, device=features_9d.device)
            std9 = torch.tensor(self.normalization.std_9d, dtype=features_9d.dtype, device=features_9d.device)
            x9 = (features_9d - mean9) / std9
        valid_mask = torch.tensor([valid], dtype=torch.bool, device=x25.device)
        with torch.no_grad():
            logits, self.hidden = self.model.step(x25, x9, self.hidden, valid_mask)
        probabilities = {name.removesuffix("_logit"): float(torch.sigmoid(value)[0]) for name, value in logits.items()}
        scheduler = self.scheduler.update(probabilities, valid=valid)
        return {"probabilities": probabilities, **scheduler, "teacher_inputs_consumed": False, "attack_enabled": False}


__all__ = ["B3RuntimeThresholds", "B3V3OneShotScheduler", "B3V3StreamingRuntime"]
