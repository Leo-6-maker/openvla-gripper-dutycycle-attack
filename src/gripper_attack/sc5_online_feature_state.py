"""Canonical online physical state extraction for SC5 feature streaming.

This module intentionally stops before the 25D causal feature adapter.  It
defines the raw per-step physical quantities that both the clean collector and
the exact-restore runner must feed into ``SC5StreamingFeatureAdapterV2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SC5PhysicalState:
    gripper_qpos: float
    gripper_opening_proxy: float
    eef_x: float
    eef_y: float
    eef_z: float
    eef_vx: float
    eef_vy: float
    eef_vz: float
    next_prev_eef: tuple[float, float, float] | None
    gripper_qpos_values: tuple[float, ...]
    gripper_state_source: str
    eef_site_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gripper_qpos": self.gripper_qpos,
            "gripper_opening_proxy": self.gripper_opening_proxy,
            "eef_x": self.eef_x,
            "eef_y": self.eef_y,
            "eef_z": self.eef_z,
            "eef_vx": self.eef_vx,
            "eef_vy": self.eef_vy,
            "eef_vz": self.eef_vz,
            "gripper_qpos_values": list(self.gripper_qpos_values),
            "gripper_state_source": self.gripper_state_source,
            "eef_site_name": self.eef_site_name,
        }


def _default_physical_gripper_state(env: Any, obs: Any) -> Mapping[str, Any] | None:
    from v4_run_eval_openvla import physical_gripper_state

    return physical_gripper_state(env, obs)


def _gripper_qpos_from_physical_state(
    env: Any,
    obs: Any,
    physical_gripper_state_fn: Callable[[Any, Any], Mapping[str, Any] | None] | None,
) -> tuple[tuple[float, ...], str]:
    fn = physical_gripper_state_fn or _default_physical_gripper_state
    state = fn(env, obs)
    qpos = state.get("qpos", []) if state else []
    values = tuple(float(x) for x in qpos)
    return values, "physical_gripper_state"


def _eef_position(env: Any, *, site_name: str) -> tuple[float, float, float]:
    site_id = env.sim.model.site_name2id(site_name)
    eef_pos = env.sim.data.site_xpos[site_id]
    return float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])


def initialize_sc5_prev_eef(env: Any, *, eef_site_name: str = "gripper0_grip_site") -> tuple[float, float, float]:
    return _eef_position(env, site_name=eef_site_name)


def extract_sc5_physical_state(
    *,
    env: Any,
    obs: Any,
    prev_eef: Sequence[float] | None,
    physical_gripper_state_fn: Callable[[Any, Any], Mapping[str, Any] | None] | None = None,
    eef_site_name: str = "gripper0_grip_site",
) -> SC5PhysicalState:
    qpos_values, qpos_source = _gripper_qpos_from_physical_state(env, obs, physical_gripper_state_fn)
    q0 = qpos_values[0] if len(qpos_values) > 0 else float("nan")
    q1 = qpos_values[1] if len(qpos_values) > 1 else float("nan")
    if np.isfinite(q0) and np.isfinite(q1):
        gripper_qpos = float(q0 + q1)
        opening_proxy = float(abs(q0) + abs(q1))
    else:
        gripper_qpos = float("nan")
        opening_proxy = float("nan")

    eef = _eef_position(env, site_name=eef_site_name)
    if prev_eef is not None and np.all(np.isfinite([*eef, *prev_eef])):
        eef_v = (
            float(eef[0] - float(prev_eef[0])),
            float(eef[1] - float(prev_eef[1])),
            float(eef[2] - float(prev_eef[2])),
        )
    else:
        eef_v = (float("nan"), float("nan"), float("nan"))
    next_prev_eef = eef if np.all(np.isfinite(eef)) else None
    return SC5PhysicalState(
        gripper_qpos=gripper_qpos,
        gripper_opening_proxy=opening_proxy,
        eef_x=eef[0],
        eef_y=eef[1],
        eef_z=eef[2],
        eef_vx=eef_v[0],
        eef_vy=eef_v[1],
        eef_vz=eef_v[2],
        next_prev_eef=next_prev_eef,
        gripper_qpos_values=qpos_values,
        gripper_state_source=qpos_source,
        eef_site_name=eef_site_name,
    )
