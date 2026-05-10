from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TriggerContext:
    task_id: str
    episode_id: int
    step_idx: int
    rho: float
    prefix_logits: Optional[Any] = None
    decoded_prefix: Optional[Any] = None
    clean_action: Optional[Any] = None
    prev_decoded_prefix: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerDecision:
    trigger_name: str
    raw_active: bool
    score: Optional[float] = None
    signal_available: bool = True
    fallback: bool = False
    reason: str = ""
    oracle: bool = False
    privileged: bool = False
    threshold_active: Optional[bool] = None


@dataclass
class BudgetDecision:
    raw_active: bool
    attack_active: bool
    budget_used_before: int
    budget_remaining_after: int
    budget_blocked: bool


@dataclass
class AttackResult:
    x_adv: Any = None
    action_adv: Any = None
    attack_method: str = "none"
    directional_loss_available: bool = False
    num_attack_steps: int = 0
    epsilon: float = 0.0
    step_size: float = 0.0
    observation_perturb_linf: float = 0.0
    observation_perturb_l2: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)
