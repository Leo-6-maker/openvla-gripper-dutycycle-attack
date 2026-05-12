from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import os
import numpy as np


@dataclass
class GraspPhaseTracker:
    initial_bowl_z: float = 0.0
    gripper_history: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    first_gate_step: int | None = None
    first_close_step: int | None = None
    first_lift_step: int | None = None

    def reset(self, initial_bowl_z: float) -> None:
        self.initial_bowl_z = float(initial_bowl_z)
        self.gripper_history.clear()
        self.first_gate_step = None
        self.first_close_step = None
        self.first_lift_step = None

    def update(self, step_idx: int, env_gripper_action: float, gate_active: bool, bowl_z_delta: float) -> None:
        g = float(env_gripper_action)
        if lift_env_gripper_closed(g) and self.first_close_step is None:
            self.first_close_step = int(step_idx)
        if gate_active and self.first_gate_step is None:
            self.first_gate_step = int(step_idx)
        if bowl_z_delta > 0.05 and self.first_lift_step is None:
            self.first_lift_step = int(step_idx)
        self.gripper_history.append(g)


def lift_env_gripper_closed(value: float) -> bool:
    # Current OpenVLA + LIBERO postprocess shows env +1 while the bowl is held.
    sign = os.environ.get("V4_LIFT_CLOSED_GRIPPER_SIGN", "positive").strip().lower()
    threshold = abs(float(os.environ.get("V4_LIFT_CLOSED_GRIPPER_THRESHOLD", "0.5")))
    v = float(value)
    if sign in {"positive", "+", "+1", "1", "pos"}:
        return v > threshold
    return v < -threshold


def _body_pos(env: Any, body_name: str) -> np.ndarray | None:
    try:
        bid = env.sim.model.body_name2id(body_name)
        return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)
    except Exception:
        return None


def _site_pos(env: Any, site_name: str) -> np.ndarray | None:
    try:
        sid = env.sim.model.site_name2id(site_name)
        return np.asarray(env.sim.data.site_xpos[sid], dtype=np.float32)
    except Exception:
        return None


def object_pos(env: Any, object_name: str) -> np.ndarray | None:
    for name in (f"{object_name}_main", object_name):
        pos = _body_pos(env, name)
        if pos is not None:
            return pos
    return _site_pos(env, f"{object_name}_default_site")


def eef_pos(env: Any) -> np.ndarray | None:
    for body in ("gripper0_eef", "robot0_eef", "eef"):
        pos = _body_pos(env, body)
        if pos is not None:
            return pos
    for site in ("gripper0_grip_site", "robot0_grip_site"):
        pos = _site_pos(env, site)
        if pos is not None:
            return pos
    return None


def close_transition_active(history: list[float], current: float) -> bool:
    vals = [float(x) for x in history[-5:]] + [float(current)]
    if len(vals) < 2:
        return False
    if lift_env_gripper_closed(vals[-1]) and any(not lift_env_gripper_closed(v) for v in vals[:-1]):
        return True
    return any((not lift_env_gripper_closed(vals[i - 1])) and lift_env_gripper_closed(vals[i]) for i in range(1, len(vals)))


def compute_grasp_metadata(
    env: Any,
    step_idx: int,
    clean_action: Any,
    env_action: Any,
    tracker: GraspPhaseTracker,
    *,
    bowl_name: str | None = None,
    plate_name: str | None = None,
    gate_dist_threshold: float = 0.10,
) -> dict[str, Any]:
    bowl_name = str(bowl_name or os.environ.get("V4_TARGET_OBJECT_NAME", "akita_black_bowl_1"))
    plate_name = str(plate_name or os.environ.get("V4_TARGET_RECEPTACLE_NAME", "plate_1"))
    eef = eef_pos(env)
    bowl = object_pos(env, bowl_name)
    plate = object_pos(env, plate_name)
    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1) if clean_action is not None else np.zeros(7, dtype=np.float32)
    executed = np.asarray(env_action, dtype=np.float32).reshape(-1) if env_action is not None else clean
    gripper_env = float(executed[-1]) if executed.size else 0.0
    gripper_clean = float(clean[-1]) if clean.size else 0.0
    eef_bowl_dist = float(np.linalg.norm(eef - bowl)) if eef is not None and bowl is not None else 1e9
    bowl_z_delta = float(bowl[2] - tracker.initial_bowl_z) if bowl is not None else 0.0
    bowl_plate_dxy = float(np.linalg.norm(bowl[:2] - plate[:2])) if bowl is not None and plate is not None else 1e9
    bowl_plate_dz = float(bowl[2] - plate[2]) if bowl is not None and plate is not None else 0.0
    recent = list(tracker.gripper_history)
    close_transition = close_transition_active(recent, gripper_env)
    close_intent = bool(lift_env_gripper_closed(gripper_env) or close_transition)
    gate_active = bool(
        eef_bowl_dist < float(gate_dist_threshold)
        and int(step_idx) < 160
        and close_intent
        and bowl_z_delta < 0.05
    )
    rel_first_gate = None if tracker.first_gate_step is None else int(step_idx) - int(tracker.first_gate_step)
    meta = {
        "grasp_target_object_name": bowl_name,
        "grasp_target_receptacle_name": plate_name,
        "grasp_eef_bowl_dist": eef_bowl_dist,
        "grasp_bowl_z_delta": bowl_z_delta,
        "grasp_bowl_plate_dxy": bowl_plate_dxy,
        "grasp_bowl_plate_dz": bowl_plate_dz,
        "grasp_env_gripper_action": gripper_env,
        "grasp_clean_gripper_action": gripper_clean,
        "grasp_gate_dist_threshold": float(gate_dist_threshold),
        "grasp_close_intent": close_intent,
        "grasp_close_transition": bool(close_transition),
        "grasp_gate_active": gate_active,
        "priv_lift_bowl_z_delta_min": float(os.environ.get("V4_PRIV_LIFT_BOWL_Z_DELTA_MIN", "0.04")),
        "priv_lift_carry_gate_active": bool(lift_env_gripper_closed(gripper_env) and bowl_z_delta >= float(os.environ.get("V4_PRIV_LIFT_BOWL_Z_DELTA_MIN", "0.04"))),
        "grasp_first_gate_step": tracker.first_gate_step,
        "grasp_first_close_step": tracker.first_close_step,
        "grasp_first_lift_step": tracker.first_lift_step,
        "first_attack_step_relative_to_grasp": rel_first_gate,
    }
    tracker.update(int(step_idx), gripper_env, gate_active, bowl_z_delta)
    return meta


def proxy_grasp_metadata(step_idx: int, clean_action: Any, env_action: Any, history: list[float]) -> dict[str, Any]:
    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1) if clean_action is not None else np.zeros(7, dtype=np.float32)
    executed = np.asarray(env_action, dtype=np.float32).reshape(-1) if env_action is not None else clean
    gripper_env = float(executed[-1]) if executed.size else 0.0
    close_transition = close_transition_active(history, gripper_env)
    motion_norm = float(np.linalg.norm(clean[:3])) if clean.size >= 3 else 0.0
    z_up = bool(clean.size >= 3 and clean[2] > 0.02)
    close_intent = bool(lift_env_gripper_closed(gripper_env) or close_transition)
    step_gate = bool(int(step_idx) < 160)
    close_only = bool(step_gate and close_intent)
    transition_only = bool(step_gate and close_transition)
    close_motion_loose = bool(step_gate and close_intent and motion_norm > 0.01)
    close_motion = bool(step_gate and close_intent and motion_norm > 0.02)
    close_motion_strict = bool(step_gate and close_intent and motion_norm > 0.05)
    close_motion_zup = bool(close_motion and z_up)
    return {
        "proxy_grasp_gate_active": close_motion,
        "proxy_grasp_gate_close_intent_only": close_only,
        "proxy_grasp_gate_close_transition_only": transition_only,
        "proxy_grasp_gate_close_motion_loose": close_motion_loose,
        "proxy_grasp_gate_close_motion": close_motion,
        "proxy_grasp_gate_close_motion_strict": close_motion_strict,
        "proxy_grasp_gate_close_motion_zup": close_motion_zup,
        "proxy_grasp_close_intent": close_intent,
        "proxy_grasp_close_transition": bool(close_transition),
        "proxy_grasp_motion_norm": motion_norm,
        "proxy_grasp_z_up": z_up,
    }


def infer_failure_phase(steps: list[dict], success: bool) -> str:
    if success:
        return "success_libero"
    if not steps:
        return "unknown"
    max_lift = max(float(r.get("grasp_bowl_z_delta", 0.0) or 0.0) for r in steps)
    min_eef = min(float(r.get("grasp_eef_bowl_dist", 1e9) or 1e9) for r in steps)
    ever_close = any(bool(r.get("grasp_close_intent")) for r in steps)
    end_dxy = float(steps[-1].get("grasp_bowl_plate_dxy", 1e9) or 1e9)
    if min_eef > 0.12 or not ever_close:
        return "no_grasp"
    if max_lift < 0.03:
        return "no_grasp"
    if max_lift < 0.08:
        return "lift_fail"
    if end_dxy > 0.05:
        return "transport_fail"
    return "placement_unstable"
