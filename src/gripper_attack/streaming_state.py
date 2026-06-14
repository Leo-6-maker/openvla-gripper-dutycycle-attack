"""Phase F: True streaming replay state machine.

Maintains internal state that only sees records[:t+1] at each step t.
Never peeks at future records. Matches batch replay output exactly.
"""

from __future__ import annotations

from typing import Optional

from .critical_close_selector import (
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON, TIE_TOLERANCE,
    _safe_float,
)
from .phase_detector import _safe_float as _sf


class CloseEventStreamingState:
    """Streaming state machine for online close-event interception.

    Maintains minimal internal state (last raw gripper, close streak,
    recent EEF positions, cooldown, triggered status). At each step,
    consumes exactly one record and updates internal state.

    Produces per-step prediction dicts identical to batch
    rule_based_close_predictor (verified by test).
    """

    def __init__(self,
                 score_threshold: float = 1.5,
                 confirmation_steps: int = 1,
                 cooldown_steps: int = 20,
                 window_len: int = WINDOW_LEN,
                 pre_offset: int = PRE_OFFSET):
        self._score_threshold = score_threshold
        self._confirmation_steps = confirmation_steps
        self._cooldown_steps = cooldown_steps
        self._window_len = window_len
        self._pre_offset = pre_offset

        # Internal causal state
        self._step = 0
        self._last_raw = 0.5      # initial OPEN assumption
        self._close_streak = 0
        self._triggered = False
        self._confirm_count = 0
        self._last_trigger = -cooldown_steps
        self._trigger_step = -1

        # EEF history (last 6 positions for 3-step velocity)
        self._eef_history = []   # list of (x, y, z)

        # All predictions for batch comparison
        self._predictions = []

    def update(self, record: dict) -> dict:
        """Process one record causally. Returns per-step prediction dict."""
        t = self._step
        r = record

        # Extract features — try clean_gripper_raw first, fall back to proxy
        raw_now = _safe_float(r.get("clean_gripper_raw",
                                     r.get("clean_gripper_raw_proxy", 0.5)))
        clean_close = int(_safe_float(r.get("clean_close", 0)))
        close_onset = int(_safe_float(r.get("close_onset", 0)))
        close_streak = int(_safe_float(r.get("close_streak", 0)))
        decoded_open = int(_safe_float(r.get("decoded_open_bool", 0)))
        qpos = _safe_float(r.get("gripper_qpos_before", 0))
        eef_x = _safe_float(r.get("eef_x", 0))
        eef_y = _safe_float(r.get("eef_y", 0))
        eef_z = _safe_float(r.get("eef_z", 0))

        # Update EEF history
        self._eef_history.append((eef_x, eef_y, eef_z))
        if len(self._eef_history) > 6:
            self._eef_history.pop(0)

        # ── Scoring (same logic as rule_based_close_predictor) ──
        score = 0.0

        # Raw crossing
        raw_open_to_close_crossing = False
        if self._last_raw > 0.5 and raw_now <= 0.5:
            raw_open_to_close_crossing = True
            score += 1.5

        # Close streak == 1
        if close_streak == 1:
            score += 1.0

        # Close onset with low qpos
        if close_onset and qpos < 0.005:
            score += 0.5

        # EEF deceleration: 3-step velocity parity with batch _eef_speed
        # history stores up to 6 positions: [t-5, t-4, t-3, t-2, t-1, t]
        # speed_now  = pos[t]   - pos[t-3] = h[-1] - h[-4]
        # speed_prev = pos[t-1] - pos[t-4] = h[-2] - h[-5]
        if len(self._eef_history) >= 4:
            h = self._eef_history
            dx_now = h[-1][0] - h[-4][0]
            dy_now = h[-1][1] - h[-4][1]
            dz_now = h[-1][2] - h[-4][2]
            speed_now = (dx_now**2 + dy_now**2 + dz_now**2)**0.5

            if len(self._eef_history) >= 5:
                dx_prev = h[-2][0] - h[-5][0]
                dy_prev = h[-2][1] - h[-5][1]
                dz_prev = h[-2][2] - h[-5][2]
                speed_prev = (dx_prev**2 + dy_prev**2 + dz_prev**2)**0.5
                if speed_prev > 0 and speed_now < speed_prev and speed_now < 0.01:
                    score += 0.5

        # Qpos ready
        if qpos < 0.01 and not decoded_open:
            score += 0.3

        # Penalty
        if decoded_open:
            score -= 2.0

        score = max(0.0, score)

        # ── Close event candidate flag ──
        is_close_event_candidate = (
            raw_open_to_close_crossing
            or bool(close_onset)
            or close_streak == 1
        )

        # ── Abstain ──
        abstain = ""
        if decoded_open:
            abstain = "gripper_already_open"
        elif t < 3:
            abstain = "too_early"
        elif score < 0.5:
            abstain = "low_confidence"

        # ── Online trigger logic ──
        triggered_this_step = False
        if not self._triggered and not abstain:
            if score >= self._score_threshold and t - self._last_trigger >= self._cooldown_steps:
                self._confirm_count += 1
                if self._confirm_count >= self._confirmation_steps:
                    self._triggered = True
                    self._trigger_step = t
                    self._last_trigger = t
                    triggered_this_step = True
            else:
                self._confirm_count = 0

        # Build prediction
        pred = {
            "step": t,
            "score": score,
            "abstain": abstain,
            "clean_close": clean_close,
            "close_onset": close_onset,
            "qpos": qpos,
            "raw_open_to_close_crossing": raw_open_to_close_crossing,
            "is_close_event_candidate": is_close_event_candidate,
            "triggered": triggered_this_step,
            "trigger_step": self._trigger_step if self._triggered else -1,
        }

        # Update internal state
        self._last_raw = raw_now
        if clean_close:
            self._close_streak = close_streak
        else:
            self._close_streak = 0
        self._step += 1
        self._predictions.append(pred)

        return pred

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_step(self) -> int:
        return self._trigger_step

    @property
    def predictions(self) -> list[dict]:
        return self._predictions

    def online_window(self) -> dict:
        """Return the online trigger window if triggered, else abstain sentinel."""
        if self._triggered:
            ws = max(self._trigger_step, self._trigger_step - self._pre_offset)
            we = ws + self._window_len
            return {
                "window_start": ws,
                "window_end": we,
                "anchor_step": self._trigger_step,
                "trigger_step": self._trigger_step,
                "score": 0.0,
                "abstain_reason": "",
                "prediction_mode": "observed_close_interception",
            }
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "trigger_step": -1,
            "score": 0.0,
            "abstain_reason": "no_online_trigger",
            "prediction_mode": "",
        }
