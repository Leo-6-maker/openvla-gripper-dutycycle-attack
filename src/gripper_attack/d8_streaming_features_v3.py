"""D8 streaming feature adapter V3 — multi-event causal, no single-close assumption.

Key differences from SC5StreamingFeatureAdapterV2:
  - close_onset: fires on EVERY OPEN->CLOSE transition (not just first)
  - time_since_close: relative to most recent close onset
  - eef_z_delta_since_close: relative to most recent close onset
  - flip_count: windowed via deque(maxlen=32), max=31, not episode-cumulative
  - action_gripper: LIBERO postprocessed env gripper (semantically distinct from gripper_command)
  - gripper_qpos: SIGNED sum of qpos[0]+qpos[1] (distinct from opening_proxy=abs sum)
  - history_len frozen at 32

V2 is preserved as-is for baseline comparison.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

import numpy as np

HISTORY_LEN = 32  # frozen — must match MIN_HISTORY
MIN_HISTORY = HISTORY_LEN
MAX_FLIPS = HISTORY_LEN - 1  # maximum possible flips in a 32-state window

FEATURE_NAMES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

# Frozen feature contract — verified at import time
_EXPECTED = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]
if FEATURE_NAMES != _EXPECTED:
    raise SystemError("V3 feature names diverged from frozen schema order")


def _is_valid_float(v) -> bool:
    if v is None: return False
    if isinstance(v, bool): return False
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return False
        return True
    return False


class D8StreamingFeatureAdapterV3:
    """Continuous per-step feature adapter — multi-event causal.

    Every OPEN->CLOSE transition produces close_onset=1.
    All derived features reference the most recent close onset.
    Flip count uses a fixed-length deque(maxlen=32) — max value is 31.
    """

    def __init__(self, history_len: int = MIN_HISTORY):
        if history_len != MIN_HISTORY:
            raise ValueError(f"history_len is frozen at {MIN_HISTORY}")
        self.history_len = MIN_HISTORY
        self._reset()

    def _reset(self):
        self.history: List[dict] = []
        self._flip_window: Deque[bool] = deque(maxlen=self.history_len)
        self._next_expected_step = 0
        self._close_streak = 0
        self._open_streak = 0
        self._last_close_step = -1
        self._prev_gripper_close: Optional[bool] = None

    def reset(self):
        self._reset()

    @property
    def next_expected_step(self) -> int:
        return self._next_expected_step

    def _windowed_flip_count(self) -> int:
        """Count gripper state flips within the deque window.

        With maxlen=32, the max possible flip count is 31
        (31 transitions between 32 states).
        """
        count = 0
        items = list(self._flip_window)
        for i in range(1, len(items)):
            if items[i - 1] != items[i]:
                count += 1
        return count

    def update(self, step_id: int,
               raw_gripper: float, env_gripper: float,
               gripper_qpos: float, gripper_opening_proxy: float,
               eef_x: float, eef_y: float, eef_z: float,
               eef_vx: float, eef_vy: float, eef_vz: float,
               action_dx: float, action_dy: float, action_dz: float,
               action_gripper: float) -> dict:
        """Process one step.

        Args:
            gripper_qpos: SIGNED sum qpos[0] + qpos[1]
            gripper_opening_proxy: ABSOLUTE sum |qpos[0]| + |qpos[1]|
            action_gripper: LIBERO postprocessed env gripper
        """

        if step_id != self._next_expected_step:
            raise ValueError(
                f"Step sequence violation: expected {self._next_expected_step}, got {step_id}")
        self._next_expected_step = step_id + 1

        required = {
            'raw_gripper': raw_gripper, 'env_gripper': env_gripper,
            'gripper_qpos': gripper_qpos, 'gripper_opening_proxy': gripper_opening_proxy,
            'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
            'eef_vx': eef_vx, 'eef_vy': eef_vy, 'eef_vz': eef_vz,
            'action_dx': action_dx, 'action_dy': action_dy, 'action_dz': action_dz,
            'action_gripper': action_gripper,
        }
        missing = [k for k, v in required.items() if not _is_valid_float(v)]
        if missing:
            self.history.append({'step': step_id, 'valid': False, 'features': None})
            return {'step': step_id, 'valid': False, 'features': None,
                    'error': f'missing_fields: {missing}'}

        # Gripper semantics
        raw_close = raw_gripper <= 0.5
        env_close = env_gripper > 0
        if raw_close != env_close:
            self.history.append({'step': step_id, 'valid': False, 'features': None})
            return {'step': step_id, 'valid': False, 'features': None,
                    'error': 'gripper_semantics_invalid'}

        # Update flip window BEFORE computing flip count
        # (window includes the current state after append)
        if self._flip_window:
            prev_close = self._flip_window[-1]
        else:
            prev_close = None
        self._flip_window.append(raw_close)

        # Streaks
        if raw_close:
            self._close_streak += 1
            self._open_streak = 0
        else:
            self._open_streak += 1
            self._close_streak = 0

        # V3: close_onset fires on EVERY open->close transition
        close_onset = 1 if (raw_close and self._close_streak == 1) else 0
        if close_onset:
            self._last_close_step = step_id

        # V3: time_since_close relative to most recent close onset
        time_since_close = step_id - self._last_close_step if self._last_close_step >= 0 else -1

        # EEF speed
        eef_speed = np.sqrt(eef_vx**2 + eef_vy**2 + eef_vz**2)

        # V3: eef_z_delta_since_close relative to most recent close onset
        if self._last_close_step >= 0 and self._last_close_step < len(self.history):
            prev_eef_z = self.history[self._last_close_step].get('eef_z', eef_z)
            eef_z_delta = eef_z - prev_eef_z
        else:
            eef_z_delta = 0.0

        # V3: windowed flip count — count transitions within the deque window
        flip_count = self._windowed_flip_count()

        self._prev_gripper_close = raw_close

        # Qpos deltas (using gripper_qpos = signed sum)
        qpos_delta_1 = 0.0
        qpos_delta_3 = 0.0
        if len(self.history) >= 1 and self.history[-1].get('valid'):
            qpos_delta_1 = gripper_qpos - self.history[-1]['gripper_qpos']
        if len(self.history) >= 3 and self.history[-3].get('valid'):
            qpos_delta_3 = gripper_qpos - self.history[-3]['gripper_qpos']

        # Opening proxy deltas (using gripper_opening_proxy = abs sum)
        op_delta_3 = 0.0
        op_var_5 = 0.0
        recent_proxies = []
        for h in self.history[-4:]:
            if h is not None and h.get('valid'):
                recent_proxies.append(h.get('gripper_opening_proxy', 0))
        recent_proxies.append(gripper_opening_proxy)
        if len(recent_proxies) >= 4:
            op_delta_3 = gripper_opening_proxy - recent_proxies[-4]
        if len(recent_proxies) >= 5:
            op_var_5 = float(np.var(recent_proxies[-5:]))

        # EEF speed variance
        eef_speeds = []
        for h in self.history[-4:]:
            if h.get('valid'):
                eef_speeds.append(np.sqrt(h['eef_vx']**2 + h['eef_vy']**2 + h['eef_vz']**2))
        eef_speeds.append(eef_speed)
        eef_speed_var_5 = float(np.var(eef_speeds)) if len(eef_speeds) >= 5 else 0.0

        features = {
            'gripper_command': raw_gripper,
            # H1-R2: gripper_qpos = SIGNED sum (feature 1), distinct from opening_proxy
            'gripper_qpos': gripper_qpos,
            'gripper_opening_proxy': gripper_opening_proxy,
            'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
            'eef_vx': eef_vx, 'eef_vy': eef_vy, 'eef_vz': eef_vz,
            'action_dx': action_dx, 'action_dy': action_dy, 'action_dz': action_dz,
            'action_gripper': action_gripper,
            'recent_close_streak': self._close_streak,
            'recent_open_streak': self._open_streak,
            # H1-R3: windowed flip count, max=31, via deque
            'recent_gripper_flip_count': flip_count,
            'close_onset': close_onset,
            'time_since_close': time_since_close,
            'eef_speed': eef_speed,
            'eef_z_delta_since_close': eef_z_delta,
            'qpos_delta_1': qpos_delta_1,
            'qpos_delta_3': qpos_delta_3,
            'opening_proxy_delta_3': op_delta_3,
            'opening_proxy_variance_5': op_var_5,
            'eef_speed_variance_5': eef_speed_var_5,
        }

        record = {'step': step_id, 'valid': True, 'features': features,
                  'raw_close': raw_close, 'env_close': env_close,
                  'gripper_qpos': gripper_qpos, 'gripper_opening_proxy': gripper_opening_proxy,
                  'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
                  'eef_vx': eef_vx, 'eef_vy': eef_vy, 'eef_vz': eef_vz}
        self.history.append(record)
        return record
