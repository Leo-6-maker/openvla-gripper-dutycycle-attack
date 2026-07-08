#!/usr/bin/env python3
"""Resettable multi-event FSM for multi-suite detector (D8C prototype).

Extends the one-shot IDLE→ARMED→EMITTED FSM to support L10 multi-event
long-horizon tasks:

    IDLE → ARMED → EMITTED → COOLDOWN → IDLE (→ next event)

Each episode can produce multiple candidate events with typed roles:
primary_attackable, auxiliary_manipulation, distractor_or_setup,
unsupported_or_abstain.

CPU-only. No env.step, no OpenVLA, no MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── FSM states ──
FSM_IDLE = "IDLE"
FSM_ARMED = "ARMED"
FSM_EMITTED = "EMITTED"
FSM_COOLDOWN = "COOLDOWN"

# ── event roles ──
ROLE_PRIMARY = "primary_attackable"
ROLE_AUXILIARY = "auxiliary_manipulation"
ROLE_DISTRACTOR = "distractor_or_setup"
ROLE_UNSUPPORTED = "unsupported_or_abstain"


@dataclass
class DetectorFrame:
    """Single-step detector output."""
    step: int
    emit_p: float
    suppress_p: float
    corridor_p: float = 0.0
    release_p: float = 0.0
    event_role: str = ROLE_UNSUPPORTED
    primary_ok: bool = False
    valid: bool = True


@dataclass
class FsmEvent:
    """A detected event emitted by the FSM."""
    event_id: int
    event_role: str
    arm_step: int
    emit_step: int
    cooldown_end: int
    primary_attackable: bool
    abstain_reason: str = ""
    emit_p_at_emit: float = 0.0
    suppress_p_at_emit: float = 0.0


@dataclass
class FsmConfig:
    """FSM hyperparameters (frozen for a given detector bundle)."""
    tau_emit: float = 0.33
    tau_suppress: float = 0.67
    tau_corridor: float = 0.5
    tau_release: float = 0.5
    arm_guard_steps: int = 5         # steps after ARM before EMIT allowed
    cooldown_steps: int = 20         # steps after EMIT before re-ARM
    require_primary: bool = True     # only emit on primary_attackable events
    max_events_per_episode: int = 5


class ResettableMultiEventFSM:
    """Per-episode FSM that allows multiple emit/cooldown cycles.

    Usage::

        fsm = ResettableMultiEventFSM(C2e3GRUDetectorRuntime(...).tau_emit, ...)
        for step, frame in enumerate(detector_frames):
            event = fsm.update(step, frame)
            if event is not None:
                print(f"Event {event.event_id}: {event.event_role} at step {event.emit_step}")
    """

    def __init__(self, config: FsmConfig):
        self._cfg = config
        self.reset()

    def reset(self) -> None:
        """Reset per-episode state."""
        self._state = FSM_IDLE
        self._arm_step = -1
        self._armed_since = 0
        self._emit_count = 0
        self._events: List[FsmEvent] = []
        self._last_emit_step = -1

    @property
    def events(self) -> List[FsmEvent]:
        return list(self._events)

    @property
    def state(self) -> str:
        return self._state

    def update(self, step: int, frame: DetectorFrame) -> Optional[FsmEvent]:
        """Process one detector frame. Returns a new event if emitted, else None."""
        cfg = self._cfg

        if not frame.valid:
            return None

        if self._state == FSM_IDLE:
            # Transition to ARMED when corridor is active and (if required) primary is OK
            corridor_ok = frame.corridor_p >= cfg.tau_corridor
            primary_ok = (not cfg.require_primary) or (frame.primary_ok and frame.event_role == ROLE_PRIMARY)
            if corridor_ok and primary_ok:
                self._state = FSM_ARMED
                self._arm_step = step
                self._armed_since = 0
                return None

        elif self._state == FSM_ARMED:
            self._armed_since += 1
            guard_passed = self._armed_since >= cfg.arm_guard_steps
            emit_ok = (frame.emit_p >= cfg.tau_emit and frame.suppress_p <= cfg.tau_suppress)
            release_safe = frame.release_p < cfg.tau_release

            if guard_passed and emit_ok and release_safe:
                if self._emit_count >= cfg.max_events_per_episode:
                    return None  # exceeded max events
                self._state = FSM_EMITTED
                self._emit_count += 1
                self._last_emit_step = step
                event = FsmEvent(
                    event_id=self._emit_count,
                    event_role=frame.event_role,
                    arm_step=self._arm_step,
                    emit_step=step,
                    cooldown_end=step + cfg.cooldown_steps,
                    primary_attackable=(frame.event_role == ROLE_PRIMARY and frame.primary_ok),
                    emit_p_at_emit=frame.emit_p,
                    suppress_p_at_emit=frame.suppress_p,
                )
                self._events.append(event)
                # Auto-transition to COOLDOWN
                self._state = FSM_COOLDOWN
                return event

            # Fall back to IDLE if corridor lost
            if frame.corridor_p < cfg.tau_corridor:
                self._state = FSM_IDLE
                self._arm_step = -1
                self._armed_since = 0

        elif self._state == FSM_COOLDOWN:
            if step >= self._last_emit_step + cfg.cooldown_steps:
                self._state = FSM_IDLE
                self._arm_step = -1
                self._armed_since = 0

        return None

    def summary(self) -> Dict[str, Any]:
        """Return per-episode summary for audit."""
        return {
            "fsm_type": "resettable_multi_event",
            "total_events": len(self._events),
            "primary_events": sum(1 for e in self._events if e.primary_attackable),
            "auxiliary_events": sum(1 for e in self._events if e.event_role == ROLE_AUXILIARY),
            "events": [
                {
                    "event_id": e.event_id,
                    "role": e.event_role,
                    "arm_step": e.arm_step,
                    "emit_step": e.emit_step,
                    "cooldown_end": e.cooldown_end,
                    "primary_attackable": e.primary_attackable,
                }
                for e in self._events
            ],
        }
