"""H1-R1: Unified strict FIT670 episode telemetry loader.

SINGLE canonical path for all telemetry loading — cache builder, smoke tests,
and GPU training must all go through these functions. No duplicate field parsing.

Fail-closed invariants enforced on every call.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.d8_streaming_features_v3 import (
    D8StreamingFeatureAdapterV3,
    FEATURE_NAMES,
    HISTORY_LEN,
)
from gripper_attack.action_contract import action_semantics_parity


# ── Exceptions ─────────────────────────────────────────────────────────

class FormalContractError(ValueError):
    """Explicit contract violation — never caught silently."""


# ── Low-level field accessors ──────────────────────────────────────────

def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise FormalContractError(f"{name}: expected shape ({size},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise FormalContractError(f"{name}: non-finite values")
    return arr


def _validate_raw_gripper_range(raw_g: float, eid: str, step: int) -> None:
    if not (0.0 <= raw_g <= 1.0):
        raise FormalContractError(f"{eid} step {step}: raw gripper {raw_g} outside [0,1]")


def _validate_env_gripper_range(env_g: float, eid: str, step: int) -> None:
    if not (-1.0 <= env_g <= 1.0):
        raise FormalContractError(f"{eid} step {step}: env gripper {env_g} outside [-1,1]")


# ── Episode loading ────────────────────────────────────────────────────

def load_episode_telemetry(ep_path: Path) -> dict[str, Any]:
    """Load one episode.json and validate top-level structure."""
    if not ep_path.is_file():
        raise FormalContractError(f"episode.json not found: {ep_path}")
    data = json.loads(ep_path.read_text(encoding="utf-8"))
    eid = data.get("episode_id")
    if not isinstance(eid, str) or not eid:
        raise FormalContractError(f"missing or empty episode_id in {ep_path}")
    steps = data.get("steps")
    telemetry = data.get("telemetry")
    if not isinstance(steps, list):
        raise FormalContractError(f"{eid}: steps is not a list")
    if not isinstance(telemetry, list):
        raise FormalContractError(f"{eid}: telemetry is not a list")
    if len(steps) != len(telemetry):
        raise FormalContractError(f"{eid}: steps ({len(steps)}) != telemetry ({len(telemetry)})")
    if len(steps) == 0:
        raise FormalContractError(f"{eid}: zero-length episode")
    return {"episode_id": eid, "ep_path": str(ep_path), "steps": steps, "telemetry": telemetry,
            "n_steps": len(steps)}


def validate_episode_step_integrity(ep: dict) -> None:
    """Verify step sequence: 0-indexed, contiguous, no duplicates, identity match."""
    eid = ep["episode_id"]
    steps = ep["steps"]
    teles = ep["telemetry"]
    n = len(steps)
    seen = set()
    for i, (s, t) in enumerate(zip(steps, teles)):
        if not isinstance(s, Mapping) or not isinstance(t, Mapping):
            raise FormalContractError(f"{eid} step {i}: not a dict")
        st = s.get("step")
        tt = t.get("step")
        if st != i:
            raise FormalContractError(f"{eid}: step[{i}] step field={st}, expected {i}")
        if tt != i:
            raise FormalContractError(f"{eid}: telemetry[{i}] step field={tt}, expected {i}")
        if i in seen:
            raise FormalContractError(f"{eid}: duplicate step {i}")
        seen.add(i)
    if len(seen) != n:
        raise FormalContractError(f"{eid}: step count mismatch {len(seen)} != {n}")


def validate_field_invariants(ep: dict) -> None:
    """Verify all required fields exist with correct shapes and finite values."""
    eid = ep["episode_id"]
    steps = ep["steps"]
    teles = ep["telemetry"]
    for i, (s, t) in enumerate(zip(steps, teles)):
        raw = _finite_vector(s.get("raw_action_7d"), 7, f"{eid} step {i} raw_action_7d")
        env = _finite_vector(s.get("action_env_7d"), 7, f"{eid} step {i} action_env_7d")
        _validate_raw_gripper_range(float(raw[6]), eid, i)
        _validate_env_gripper_range(float(env[6]), eid, i)
        if not action_semantics_parity(float(raw[6]), float(env[6])):
            raise FormalContractError(f"{eid} step {i}: raw/env semantics mismatch")
        alt_raw = s.get("action_raw_7d")
        if alt_raw is not None:
            alt = _finite_vector(alt_raw, 7, f"{eid} step {i} action_raw_7d")
            if not np.array_equal(raw, alt):
                raise FormalContractError(f"{eid} step {i}: raw_action_7d != action_raw_7d")
        _finite_vector(t.get("robot0_eef_pos"), 3, f"{eid} step {i} robot0_eef_pos")
        _finite_vector(t.get("robot0_gripper_qpos"), 2, f"{eid} step {i} robot0_gripper_qpos")


# ── Feature materialization — single canonical path ────────────────────

def materialize_episode_features(ep: dict) -> dict[str, Any]:
    """Run D8StreamingFeatureAdapterV3 over one pre-validated episode.

    H1-R2: gripper_qpos = SIGNED sum qpos[0]+qpos[1]
           gripper_opening_proxy = ABSOLUTE sum |qpos[0]|+|qpos[1]|

    Returns: {episode_id, n_steps, features_25d: np.ndarray[n,25]}
    """
    eid = ep["episode_id"]
    steps = ep["steps"]
    teles = ep["telemetry"]
    adapter = D8StreamingFeatureAdapterV3()
    previous_eef = None
    features_list = []

    for i, (s, t) in enumerate(zip(steps, teles)):
        raw_action = np.array(s["raw_action_7d"], dtype=np.float64)
        env_action = np.array(s["action_env_7d"], dtype=np.float64)
        eef = np.array(t["robot0_eef_pos"], dtype=np.float64)
        qpos = np.array(t["robot0_gripper_qpos"], dtype=np.float64)

        velocity = np.zeros(3) if previous_eef is None else (eef - previous_eef)
        previous_eef = eef

        # H1-R2: signed sum for gripper_qpos, absolute sum for opening_proxy
        signed_qpos = float(qpos[0] + qpos[1])
        abs_proxy = float(abs(qpos[0]) + abs(qpos[1]))

        result = adapter.update(
            step_id=i,
            raw_gripper=float(raw_action[6]),
            env_gripper=float(env_action[6]),
            gripper_qpos=signed_qpos,
            gripper_opening_proxy=abs_proxy,
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
            action_dx=float(raw_action[0]), action_dy=float(raw_action[1]), action_dz=float(raw_action[2]),
            action_gripper=float(env_action[6]),
        )
        if not result.get("valid"):
            raise FormalContractError(f"{eid} step {i}: V3 adapter rejected: {result.get('error')}")

        values = result["features"]
        vec = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
        if vec.shape != (25,):
            raise FormalContractError(f"{eid} step {i}: feature vector shape {vec.shape}")
        if not np.isfinite(vec).all():
            raise FormalContractError(f"{eid} step {i}: non-finite feature")
        features_list.append(vec)

    return {
        "episode_id": eid,
        "n_steps": len(features_list),
        "features_25d": np.array(features_list, dtype=np.float32),
    }


# ── Bulk loader for cache builder ──────────────────────────────────────

def build_telemetry_index(telemetry_root: Path) -> dict[str, Path]:
    """Index all episode.json files by episode_id. Read-only."""
    ep_root = telemetry_root / "episodes"
    if not ep_root.is_dir():
        raise FormalContractError(f"episodes directory not found: {ep_root}")
    index = {}
    for suite in sorted(os.listdir(ep_root)):
        sd = ep_root / suite
        if not sd.is_dir(): continue
        for task in sorted(os.listdir(sd)):
            td = sd / task
            if not td.is_dir(): continue
            for state in sorted(os.listdir(td)):
                stated = td / state
                if not stated.is_dir(): continue
                epf = stated / "episode.json"
                if not epf.is_file(): continue
                meta = json.loads(epf.read_text(encoding="utf-8"))
                eid = meta.get("episode_id", "")
                if not eid:
                    raise FormalContractError(f"empty episode_id in {epf}")
                if eid in index:
                    raise FormalContractError(f"duplicate episode_id: {eid}")
                index[eid] = epf
    return index


def load_and_validate_all(
    telemetry_root: Path,
    ep_file_map: dict[str, Path] | None = None,
) -> tuple[list[dict], dict]:
    """Load and validate all 670 FIT670 episodes.

    Returns: (validated_episodes_list, census_report)
    """
    if ep_file_map is None:
        ep_file_map = build_telemetry_index(telemetry_root)

    if len(ep_file_map) != 670:
        raise FormalContractError(f"telemetry episodes: {len(ep_file_map)} != 670")

    episodes = []
    identity_set = set()
    total_steps = 0
    issues = {}

    for eid in sorted(ep_file_map.keys()):
        ep_data = load_episode_telemetry(ep_file_map[eid])
        if ep_data["episode_id"] != eid:
            raise FormalContractError(f"path identity {ep_data['episode_id']} != index key {eid}")
        if eid in identity_set:
            raise FormalContractError(f"duplicate episode_id: {eid}")
        identity_set.add(eid)
        validate_episode_step_integrity(ep_data)
        validate_field_invariants(ep_data)
        total_steps += ep_data["n_steps"]
        episodes.append(ep_data)

    if len(episodes) != 670:
        raise FormalContractError(f"loaded {len(episodes)} != 670 episodes")
    if total_steps != 196483:
        raise FormalContractError(f"total steps {total_steps} != 196483")

    census = {
        "episode_count": len(episodes),
        "total_steps": total_steps,
        "identity_closure": True,
        "field_issues": issues,
    }
    return episodes, census
