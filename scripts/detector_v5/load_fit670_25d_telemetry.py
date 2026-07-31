"""P4-1: Strict FIT670 episode telemetry loader.

Loads episode.json files from fresh670_v5_v2_formal, validates all invariants,
feeds D8StreamingFeatureAdapterV3 with verified field mapping.

Fail-closed: missing field → raise, non-finite → raise, shape mismatch → raise.
No zero-fill, no forward-fill, no object state substitution.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.d8_streaming_features_v3 import (
    D8StreamingFeatureAdapterV3,
    FEATURE_NAMES,
)
from gripper_attack.action_contract import action_semantics_parity


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.isfinite(arr).all():
        raise ValueError(f"{name}: expected finite shape ({size},), got {arr.shape}, finite={np.isfinite(arr).all()}")
    return arr


def load_episode_telemetry(ep_path: Path) -> dict[str, Any]:
    """Load one episode.json and validate structure."""
    data = json.loads(ep_path.read_text(encoding="utf-8"))
    steps = data.get("steps")
    telemetry = data.get("telemetry")
    eid = data.get("episode_id")
    if not isinstance(eid, str) or not eid:
        raise ValueError(f"missing or empty episode_id in {ep_path}")
    if not isinstance(steps, list) or not isinstance(telemetry, list):
        raise ValueError(f"steps/telemetry not lists in {eid}")
    if len(steps) != len(telemetry):
        raise ValueError(f"steps ({len(steps)}) != telemetry ({len(telemetry)}) in {eid}")
    if len(steps) == 0:
        raise ValueError(f"zero-length episode {eid}")
    return {"episode_id": eid, "steps": steps, "telemetry": telemetry, "n_steps": len(steps)}


def validate_episode_step_integrity(ep: dict) -> None:
    """Verify step sequence: 0-indexed, contiguous, no duplicates."""
    eid = ep["episode_id"]
    steps = ep["steps"]
    teles = ep["telemetry"]
    seen = set()
    for i, (s, t) in enumerate(zip(steps, teles)):
        st = s.get("step")
        tt = t.get("step")
        if st != i:
            raise ValueError(f"{eid}: step[{i}] step field={st}, expected {i}")
        if tt != i:
            raise ValueError(f"{eid}: telemetry[{i}] step field={tt}, expected {i}")
        if i in seen:
            raise ValueError(f"{eid}: duplicate step {i}")
        seen.add(i)


def materialize_episode_features(ep: dict) -> dict[str, Any]:
    """Run D8StreamingFeatureAdapterV3 over one episode. Returns per-step features."""
    eid = ep["episode_id"]
    steps = ep["steps"]
    teles = ep["telemetry"]
    adapter = D8StreamingFeatureAdapterV3()
    previous_eef: np.ndarray | None = None
    features_list = []

    for i, (s, t) in enumerate(zip(steps, teles)):
        # Extract and validate raw action
        raw_action = _finite_vector(s.get("raw_action_7d"), 7, f"{eid} step {i} raw_action_7d")
        if not 0.0 <= float(raw_action[6]) <= 1.0:
            raise ValueError(f"{eid} step {i}: raw_action_7d[6]={raw_action[6]} outside [0,1]")

        # Extract and validate env action
        env_action = _finite_vector(s.get("action_env_7d"), 7, f"{eid} step {i} action_env_7d")
        if not -1.0 <= float(env_action[6]) <= 1.0:
            raise ValueError(f"{eid} step {i}: action_env_7d[6]={env_action[6]} outside [-1,1]")

        # Verify alias consistency
        alt_raw = s.get("action_raw_7d")
        if alt_raw is not None:
            alt_arr = _finite_vector(alt_raw, 7, f"{eid} step {i} action_raw_7d")
            if not np.array_equal(raw_action, alt_arr):
                raise ValueError(f"{eid} step {i}: raw_action_7d != action_raw_7d")

        # Verify semantics parity
        if not action_semantics_parity(float(raw_action[6]), float(env_action[6])):
            raise ValueError(f"{eid} step {i}: raw/env gripper semantics boundary or inconsistent")

        # EEF position
        eef = _finite_vector(t.get("robot0_eef_pos"), 3, f"{eid} step {i} robot0_eef_pos")

        # EEF velocity
        if previous_eef is None:
            velocity = np.zeros(3, dtype=np.float64)
        else:
            velocity = eef - previous_eef
        previous_eef = eef

        # Gripper qpos
        qpos = _finite_vector(t.get("robot0_gripper_qpos"), 2, f"{eid} step {i} robot0_gripper_qpos")
        qpos_sum = float(abs(qpos[0]) + abs(qpos[1]))

        result = adapter.update(
            step_id=i,
            raw_gripper=float(raw_action[6]),
            env_gripper=float(env_action[6]),
            gripper_qpos=qpos_sum,
            gripper_opening_proxy=qpos_sum,
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
            action_dx=float(raw_action[0]), action_dy=float(raw_action[1]), action_dz=float(raw_action[2]),
            action_gripper=float(env_action[6]),
        )
        if not result.get("valid"):
            raise ValueError(f"{eid} step {i}: D8 V3 adapter rejected: {result.get('error')}")

        values = result.get("features")
        vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
        if vector.shape != (25,) or not np.isfinite(vector).all():
            raise ValueError(f"{eid} step {i}: invalid 25D vector shape={vector.shape} finite={np.isfinite(vector).all()}")

        features_list.append(vector)

    return {
        "episode_id": eid,
        "n_steps": len(features_list),
        "features_25d": np.array(features_list, dtype=np.float32),  # [n_steps, 25]
    }


def load_all_fit670(
    formal_root: Path,
    require_all: bool = True,
) -> tuple[list[dict], dict]:
    """Load all 670 FIT670 episodes with full validation.

    Returns: (episode_data_list, census_report)
    """
    ep_root = formal_root / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"episodes directory not found: {ep_root}")

    episodes = []
    identity_set = set()
    total_steps = 0
    field_issues = Counter()
    suite_counts = Counter()

    for suite in sorted(os.listdir(ep_root)):
        suite_dir = ep_root / suite
        if not suite_dir.is_dir():
            continue
        for task in sorted(os.listdir(suite_dir)):
            task_dir = suite_dir / task
            if not task_dir.is_dir():
                continue
            for state in sorted(os.listdir(task_dir)):
                state_dir = task_dir / state
                if not state_dir.is_dir():
                    continue
                ep_file = state_dir / "episode.json"
                if not ep_file.is_file():
                    field_issues["missing_episode_json"] += 1
                    continue

                ep_data = load_episode_telemetry(ep_file)
                eid = ep_data["episode_id"]

                if eid in identity_set:
                    raise ValueError(f"duplicate episode_id: {eid}")
                identity_set.add(eid)

                validate_episode_step_integrity(ep_data)

                suite_key = "/".join(eid.split("/")[:2])
                suite_counts[suite_key] += 1
                total_steps += ep_data["n_steps"]

                # Store minimal data for feature materialization
                episodes.append({
                    "episode_id": eid,
                    "ep_path": str(ep_file),
                    "n_steps": ep_data["n_steps"],
                })

    census = {
        "episode_count": len(episodes),
        "total_steps": total_steps,
        "identity_set_size": len(identity_set),
        "expected_episodes": 670,
        "expected_steps": 196483,
        "suite_counts": dict(suite_counts),
        "field_issues": dict(field_issues),
        "identity_closure": len(episodes) == 670,
        "step_closure": total_steps == 196483,
    }

    if require_all and len(episodes) != 670:
        raise ValueError(f"episode count {len(episodes)} != 670")

    return episodes, census


def materialize_single_episode(ep_info: dict) -> dict[str, Any]:
    """Materialize 25D features for one episode from its episode.json path."""
    ep_data = load_episode_telemetry(Path(ep_info["ep_path"]))
    validate_episode_step_integrity(ep_data)
    return materialize_episode_features(ep_data)


if __name__ == "__main__":
    # Quick smoke: load first episode and materialize features
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Test first 3 episodes only")
    args = parser.parse_args()

    if args.smoke:
        episodes, census = load_all_fit670(args.formal_root, require_all=False)
        print(f"Total episodes found: {census['episode_count']}")
        print(f"Total steps: {census['total_steps']}")
        print(f"Identity closure: {census['identity_closure']}")
        print(f"Step closure: {census['step_closure']}")
        print(f"Suites: {census['suite_counts']}")
        print(f"Issues: {census['field_issues']}")

        for ep in episodes[:3]:
            result = materialize_single_episode(ep)
            feats = result["features_25d"]
            print(f"\n{result['episode_id']}: {result['n_steps']} steps, "
                  f"features shape={feats.shape}, "
                  f"finite={np.isfinite(feats).all()}, "
                  f"range=[{feats.min():.4f}, {feats.max():.4f}]")
    else:
        print("Full materialization not yet requested. Use --smoke for quick test.")
