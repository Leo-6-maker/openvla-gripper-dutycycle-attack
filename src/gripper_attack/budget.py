from __future__ import annotations
import math
from .types import BudgetDecision


class OnlineBudgetController:
    def __init__(self, rho: float, max_steps: int, rounding: str = "floor", min_budget_steps: int = 1):
        self.rho = float(rho)
        self.max_steps = int(max_steps)
        self.rounding = str(rounding)
        self.min_budget_steps = int(min_budget_steps)
        self._budget = self._compute_budget()
        self._used = 0

    def _compute_budget(self) -> int:
        if self.rho <= 0.0 or self.max_steps <= 0:
            return 0
        raw = self.rho * self.max_steps
        if self.rounding == "ceil":
            b = int(math.ceil(raw))
        elif self.rounding == "round":
            b = int(round(raw))
        else:
            b = int(math.floor(raw))
        return max(self.min_budget_steps, b)

    def reset(self) -> None:
        self._used = 0

    @property
    def budget_max_steps(self) -> int:
        return int(self._budget)

    @property
    def used(self) -> int:
        return int(self._used)

    def decide(self, raw_active: bool) -> BudgetDecision:
        before = self._used
        active = bool(raw_active) and before < self._budget
        if active:
            self._used += 1
        return BudgetDecision(
            raw_active=bool(raw_active),
            attack_active=bool(active),
            budget_used_before=int(before),
            budget_remaining_after=int(max(0, self._budget - self._used)),
            budget_blocked=bool(raw_active) and not active,
        )
