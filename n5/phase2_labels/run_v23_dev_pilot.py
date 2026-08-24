"""V23 Teacher replay runner for the frozen DEV pilot.

This is intentionally separate from every V22 adapter.  It consumes only
physical sidecar rows plus an independently sealed geometry case stream; raw
policy/action/outcome fields are never used to form a head input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from n5.phase2_labels.c3_t0_semantic_contract import (
    ContractError,
    FALSE,
    TRUE,
    UNKNOWN,
    apply_persistence,
    aggregate_tri_conjunction,
    aggregate_tri_disjunction,
    evaluate_heads,
    k10_feasible,
    protocol_horizon_for_suite,
    protocol_steps_remaining,
)
from n5.phase3_student.c3_g_predicate_evaluator import evaluate_case, load_contract


RUNNER_SCHEMA = "C3_T1_V23_TEACHER_RUNNER_V1"
SEMANTIC_SCHEMA = "C3_T0_TEACHER_SEMANTIC_CONTRACT_V1_1"
DEFAULT_SEMANTIC_CONTRACT = Path(__file__).resolve().parents[2] / "configs" / "C3_T0_TEACHER_SEMANTIC_CONTRACT_V1_1.json"
PERSISTENCE_MIN_STEPS = {
    "physical_criticality": 2,
    "safe_release": 2,
    "instability": 1,
}
FORBIDDEN_FIELDS = frozenset({
    "task_success", "task_terminal", "terminal", "terminal_state", "reward",
    "outcome", "attack", "future", "action", "action_raw",
    "clean_action_raw_7d", "applied_action_7d", "policy_action", "command",
    "close_intent",
})


class RunnerHold(RuntimeError):
    pass


def load_semantic_contract(path: Path = DEFAULT_SEMANTIC_CONTRACT) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SEMANTIC_SCHEMA or data.get("status") != "FROZEN_FIT_DEV_ONLY":
        raise RunnerHold("semantic contract is not the frozen V1.1 FIT-only contract")
    thresholds = data.get("quality_thresholds")
    if not isinstance(thresholds, Mapping):
        raise RunnerHold("quality thresholds are missing")
    required = {
        "comotion": ("object_min_displacement_m", "eef_min_displacement_m", "cosine_threshold"),
        "placement_stability": ("relative_translation_tolerance_m", "relative_rotation_tolerance_rad", "min_consecutive_steps"),
        "slip": ("relative_motion_tolerance_m",),
    }
    for section, keys in required.items():
        values = thresholds.get(section)
        if not isinstance(values, Mapping) or any(key not in values for key in keys):
            raise RunnerHold(f"quality threshold section incomplete: {section}")
        for key in keys:
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RunnerHold(f"non-finite quality threshold: {section}.{key}")
    if thresholds["comotion"]["object_min_displacement_m"] <= 0 or thresholds["comotion"]["eef_min_displacement_m"] <= 0:
        raise RunnerHold("co-motion minimum displacement must be positive")
    if not 0.0 < thresholds["comotion"]["cosine_threshold"] <= 1.0:
        raise RunnerHold("co-motion cosine threshold is invalid")
    if thresholds["placement_stability"]["relative_translation_tolerance_m"] <= 0 or thresholds["placement_stability"]["relative_rotation_tolerance_rad"] <= 0:
        raise RunnerHold("placement stability thresholds must be positive")
    if int(thresholds["placement_stability"]["min_consecutive_steps"]) < 2:
        raise RunnerHold("placement stability requires at least two consecutive steps")
    if thresholds["slip"]["relative_motion_tolerance_m"] <= 0:
        raise RunnerHold("slip threshold must be positive")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, width: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != width:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _reject_forbidden(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ContractError(f"forbidden physical input: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _tri(value: Any) -> str:
    if value is True or value == TRUE:
        return TRUE
    if value is False or value == FALSE:
        return FALSE
    return UNKNOWN


def _contact_state(row: Mapping[str, Any], object_ids: set[str]) -> str:
    pairs = row.get("mujoco_contact_pairs")
    if not object_ids or not isinstance(pairs, list):
        return UNKNOWN
    found = False
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            return UNKNOWN
        endpoints = {str(item) for item in pair}
        object_hit = any(any(endpoint == name or endpoint.startswith(name + "_") for name in object_ids) for endpoint in endpoints)
        gripper_hit = any("gripper0" in endpoint or "finger1" in endpoint or "finger2" in endpoint for endpoint in endpoints)
        if object_hit and gripper_hit:
            found = True
    return TRUE if found else FALSE


def _qpos_scalar(row: Mapping[str, Any]) -> float | None:
    qpos = _finite_vector(row.get("robot0_gripper_qpos"), 2)
    if qpos is None:
        return None
    return sum(abs(value) for value in qpos) / 2.0


def _qpos_open(row: Mapping[str, Any], threshold: float) -> bool | None:
    value = _qpos_scalar(row)
    return None if value is None else value > threshold


def _geometry_relations(geometry: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if geometry is None:
        return []
    relations = geometry.get("relations")
    if isinstance(relations, list):
        return [item for item in relations if isinstance(item, Mapping)]
    return [geometry]


def _object_positions(geometry: Mapping[str, Any] | None) -> dict[Any, list[float]]:
    result: dict[Any, list[float]] = {}
    for index, relation in enumerate(_geometry_relations(geometry)):
        pose = relation.get("object", {}).get("pose", {}) if isinstance(relation.get("object"), Mapping) else {}
        position = _finite_vector(pose.get("pos"), 3) if isinstance(pose, Mapping) else None
        if position is None:
            position = _finite_vector(relation.get("object_position"), 3)
        if position is not None:
            result[relation.get("relation_index", index)] = position
    return result


def _relation_map(geometry: Mapping[str, Any] | None) -> dict[Any, Mapping[str, Any]]:
    if geometry is None:
        return {}
    return {
        relation.get("relation_index", index): relation
        for index, relation in enumerate(_geometry_relations(geometry))
        if isinstance(relation, Mapping)
    }


def _pose_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        prev_object = previous["object"]["pose"]
        prev_target = previous["target"]["pose"]
        curr_object = current["object"]["pose"]
        curr_target = current["target"]["pose"]
        prev_rel = _relative_pose(prev_target, prev_object)
        curr_rel = _relative_pose(curr_target, curr_object)
    except (KeyError, TypeError):
        return None
    if prev_rel is None or curr_rel is None:
        return None
    translation = math.sqrt(sum((curr_rel[0][i] - prev_rel[0][i]) ** 2 for i in range(3)))
    dot = abs(sum(curr_rel[1][i] * prev_rel[1][i] for i in range(4)))
    dot = max(-1.0, min(1.0, dot))
    rotation = 2.0 * math.acos(dot)
    return translation, rotation


def _relative_pose(target: Mapping[str, Any], object_pose: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    target_pos = _finite_vector(target.get("pos"), 3)
    object_pos = _finite_vector(object_pose.get("pos"), 3)
    target_quat = _finite_vector(target.get("quat"), 4)
    object_quat = _finite_vector(object_pose.get("quat"), 4)
    if None in (target_pos, object_pos, target_quat, object_quat):
        return None
    target_norm = math.sqrt(sum(value * value for value in target_quat))
    object_norm = math.sqrt(sum(value * value for value in object_quat))
    if target_norm <= 0 or object_norm <= 0:
        return None
    tq = [value / target_norm for value in target_quat]
    oq = [value / object_norm for value in object_quat]
    inverse = [tq[0], -tq[1], -tq[2], -tq[3]]
    delta = [object_pos[i] - target_pos[i] for i in range(3)]
    rotated = _rotate_vector(inverse, delta)
    relative_quat = _quat_multiply(inverse, oq)
    return rotated, relative_quat


def _rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx]


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    raw = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
           w1*x2 + x1*w2 + y1*z2 - z1*y2,
           w1*y2 - x1*z2 + y1*w2 + z1*x2,
           w1*z2 + x1*y2 - y1*x2 + z1*w2]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw] if norm > 0 and math.isfinite(norm) else []


def _comotion(previous: Mapping[str, Any] | None, current: Mapping[str, Any],
              geometry: Mapping[str, Any] | None,
              previous_geometry: Mapping[str, Any] | None,
              thresholds: Mapping[str, Any]) -> str:
    if previous is None or geometry is None or previous_geometry is None:
        return UNKNOWN
    eef_now = _finite_vector(current.get("robot0_eef_pos"), 3)
    eef_prev = _finite_vector(previous.get("robot0_eef_pos"), 3)
    if eef_now is None or eef_prev is None:
        return UNKNOWN
    eef_delta = [eef_now[i] - eef_prev[i] for i in range(3)]
    eef_norm = math.sqrt(sum(item * item for item in eef_delta))
    cfg = thresholds["comotion"]
    eef_min = float(cfg["eef_min_displacement_m"])
    object_min = float(cfg["object_min_displacement_m"])
    if eef_norm < eef_min:
        return UNKNOWN
    current_positions = _object_positions(geometry)
    previous_positions = _object_positions(previous_geometry)
    common = sorted(set(current_positions) & set(previous_positions))
    if not common:
        return UNKNOWN
    values = []
    for key in common:
        object_delta = [current_positions[key][i] - previous_positions[key][i] for i in range(3)]
        object_norm = math.sqrt(sum(item * item for item in object_delta))
        if object_norm < object_min:
            values.append(FALSE)
        else:
            cosine = sum(object_delta[i] * eef_delta[i] for i in range(3)) / (object_norm * eef_norm)
            values.append(TRUE if cosine >= float(cfg["cosine_threshold"]) else FALSE)
    return aggregate_tri_disjunction(values)


def _placement(geometry: Mapping[str, Any] | None, predicate_contract: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if geometry is None:
        return UNKNOWN, {"reason": "MISSING_GEOMETRY_CASE"}
    results = [evaluate_case(item, predicate_contract) for item in _geometry_relations(geometry)]
    values = [_tri(item.get("value")) for item in results]
    value = aggregate_tri_conjunction(values)
    return value, {"aggregate": value, "relations": results}


def _placement_stability(previous_geometry: Mapping[str, Any] | None,
                         geometry: Mapping[str, Any] | None,
                         predicate_contract: Mapping[str, Any],
                         thresholds: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if previous_geometry is None or geometry is None:
        return UNKNOWN, {"reason": "PLACEMENT_HISTORY_UNKNOWN"}
    previous_relations = _relation_map(previous_geometry)
    current_relations = _relation_map(geometry)
    if not previous_relations or set(previous_relations) != set(current_relations):
        return UNKNOWN, {"reason": "PLACEMENT_RELATION_SET_UNKNOWN"}
    translation_limit = float(thresholds["placement_stability"]["relative_translation_tolerance_m"])
    rotation_limit = float(thresholds["placement_stability"]["relative_rotation_tolerance_rad"])
    values = []
    evidence = []
    for index in sorted(current_relations):
        current_result = evaluate_case(current_relations[index], predicate_contract)
        previous_result = evaluate_case(previous_relations[index], predicate_contract)
        current_value = _tri(current_result.get("value"))
        previous_value = _tri(previous_result.get("value"))
        delta = _pose_delta(previous_relations[index], current_relations[index])
        if delta is None or current_value is None or previous_value is None:
            value = UNKNOWN
        elif current_value == FALSE or previous_value == FALSE:
            value = FALSE
        elif delta[0] <= translation_limit and delta[1] <= rotation_limit:
            value = TRUE
        else:
            value = FALSE
        values.append(value)
        evidence.append({"relation_index": index, "value": value,
                         "relative_translation_delta_m": None if delta is None else delta[0],
                         "relative_rotation_delta_rad": None if delta is None else delta[1],
                         "translation_tolerance_m": translation_limit,
                         "rotation_tolerance_rad": rotation_limit,
                         "support_state": current_value})
    return aggregate_tri_conjunction(values), {"aggregate": aggregate_tri_conjunction(values), "relations": evidence}


def evaluate_step(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    geometry: Mapping[str, Any] | None,
    geometry_contract: Mapping[str, Any],
    step: int,
    observed_step_count: int,
    protocol_horizon: int,
    object_ids: set[str],
    qpos_close_threshold: float | None,
    semantic_config: Mapping[str, Any],
    previous_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one prefix using only current/past physical observations."""
    # Reject forbidden fields before projecting the physical allowlist.  A
    # silent projection would make injected outcome/action data look harmless
    # instead of proving that it never entered the runner boundary.
    _reject_forbidden(current)
    physical_keys = {
        key: current[key]
        for key in ("mujoco_contact_pairs", "robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
        if key in current
    }
    _reject_forbidden(physical_keys)
    contact_now = _contact_state(current, object_ids)
    contact_prev = _contact_state(previous, object_ids) if previous is not None else UNKNOWN
    if qpos_close_threshold is None or not math.isfinite(float(qpos_close_threshold)):
        raise RunnerHold("qpos threshold must be supplied by a MuJoCo calibration binding")
    qpos = _qpos_scalar(current)
    close = None if qpos is None else qpos <= qpos_close_threshold
    open_now = _qpos_open(current, qpos_close_threshold)
    open_prev = _qpos_open(previous, qpos_close_threshold) if previous is not None else None
    released_state = (
        UNKNOWN if open_now is None or contact_now == UNKNOWN else
        TRUE if open_now and contact_now == FALSE else FALSE
    )
    release_event = UNKNOWN if open_now is None or open_prev is None else (
        TRUE if not open_prev and open_now else FALSE
    )
    stable = UNKNOWN if contact_now == UNKNOWN or close is None else (TRUE if contact_now == TRUE and close else FALSE)
    transport = _comotion(previous, current, geometry, previous_geometry,
                          semantic_config["quality_thresholds"])
    placement, geometry_record = _placement(geometry, geometry_contract)
    placement_stability, stability_record = _placement_stability(
        previous_geometry, geometry, geometry_contract,
        semantic_config["quality_thresholds"],
    )
    stability = UNKNOWN if previous is None else (TRUE if stable == TRUE and contact_now == contact_prev else FALSE)
    contact_loss = UNKNOWN if previous is None or contact_prev == UNKNOWN else (TRUE if contact_prev == TRUE and contact_now == FALSE else FALSE)
    regrasp = UNKNOWN if previous is None or contact_prev == UNKNOWN else (TRUE if contact_prev == FALSE and contact_now == TRUE and close is True else FALSE)
    slip = UNKNOWN
    slip_record: dict[str, Any] = {"reason": "SLIP_HISTORY_UNKNOWN"}
    if previous is not None and contact_prev != UNKNOWN and contact_now != UNKNOWN:
        if contact_prev == TRUE and contact_now == TRUE:
            previous_positions = _object_positions(previous_geometry)
            current_positions = _object_positions(geometry)
            eef_prev = _finite_vector(previous.get("robot0_eef_pos"), 3)
            eef_now = _finite_vector(current.get("robot0_eef_pos"), 3)
            common = sorted(set(previous_positions) & set(current_positions))
            if eef_prev is None or eef_now is None or not common:
                slip = UNKNOWN
            else:
                eef_delta = [eef_now[i] - eef_prev[i] for i in range(3)]
                relative_deltas = []
                for relation_index in common:
                    object_delta = [current_positions[relation_index][i] - previous_positions[relation_index][i] for i in range(3)]
                    relative_deltas.append(math.sqrt(sum((object_delta[i] - eef_delta[i]) ** 2 for i in range(3))))
                max_relative_delta = max(relative_deltas)
                slip_limit = float(semantic_config["quality_thresholds"]["slip"]["relative_motion_tolerance_m"])
                slip = TRUE if max_relative_delta > slip_limit else FALSE
                slip_record = {"max_relative_motion_m": max_relative_delta,
                               "relative_motion_tolerance_m": slip_limit,
                               "contact_maintained": True}
        else:
            slip = FALSE
            slip_record = {"contact_maintained": False, "reason": "CONTACT_NOT_MAINTAINED"}
    physical_known = contact_now != UNKNOWN and qpos is not None
    record = {
        "physical_known": physical_known,
        "stable_grasp": stable,
        "transport_or_manipulation": transport,
        "placement": placement,
        "released_state": released_state,
        "release_event": release_event,
        "placement_stability": placement_stability,
        "protocol_steps_remaining": protocol_horizon - step - 1,
        "observed_future_steps_available": observed_step_count - step - 1,
        "slip": slip,
        "regrasp": regrasp,
        "contact_loss": contact_loss,
        "gripper_qpos": qpos,
        "qpos_close_threshold": qpos_close_threshold,
    }
    heads = evaluate_heads(record)
    return {
        "step": step,
        "protocol_steps_remaining": record["protocol_steps_remaining"],
        "observed_future_steps_available": record["observed_future_steps_available"],
        "geometry": geometry_record,
        "placement_stability_evidence": stability_record,
        "slip_evidence": slip_record,
        "geometry_case_present": geometry is not None,
        "physical_components": {
            "contact": contact_now,
            "stable_grasp": stable,
            "transport_or_manipulation": transport,
            "released_state": released_state,
            "placement": placement,
            "placement_stability": placement_stability,
            "slip": slip,
            "regrasp": regrasp,
            "contact_loss": contact_loss,
        },
        "heads": heads,
    }


def run_episode(sidecar_rows: Sequence[Mapping[str, Any]], geometry_cases: Mapping[int, Mapping[str, Any]],
                episode_id: str, geometry_contract: Mapping[str, Any], object_ids: set[str],
                qpos_close_threshold: float | None = None,
                protocol_horizon: int | None = None,
                semantic_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not sidecar_rows:
        raise RunnerHold("empty physical sidecar")
    for index, row in enumerate(sidecar_rows):
        if row.get("step") != index:
            raise RunnerHold("sidecar step stream is not contiguous")
    suite = episode_id.split("/", 1)[0]
    frozen_horizon = protocol_horizon if protocol_horizon is not None else protocol_horizon_for_suite(suite)
    if frozen_horizon is None:
        raise RunnerHold(f"unknown suite horizon: {suite}")
    if len(sidecar_rows) > frozen_horizon:
        raise RunnerHold("observed episode exceeds frozen protocol horizon")
    semantic_config = semantic_config or load_semantic_contract()
    outputs = []
    for index, row in enumerate(sidecar_rows):
        current = dict(row)
        outputs.append(evaluate_step(
            sidecar_rows[index - 1] if index else None,
            current,
            geometry_cases.get(index),
            geometry_contract,
            index,
            len(sidecar_rows),
            frozen_horizon,
            object_ids,
            qpos_close_threshold,
            semantic_config,
            geometry_cases.get(index - 1) if index else None,
        ))
    configured_persistence = semantic_config.get("persistence", {}).get("per_head_min_steps", PERSISTENCE_MIN_STEPS)
    for head, min_steps in configured_persistence.items():
        if head not in PERSISTENCE_MIN_STEPS:
            continue
        persisted = apply_persistence([item["heads"][head] for item in outputs], min_steps)
        for item, value in zip(outputs, persisted):
            item["heads"][head] = value
    for item in outputs:
        safe = item["heads"]["safe_release"]
        item["heads"]["k10_feasible"] = k10_feasible({
            "protocol_steps_remaining": item["protocol_steps_remaining"],
            "safe_release_computed": safe,
        })
        observed = item["observed_future_steps_available"]
        item["audit_observation_mask"] = bool(isinstance(observed, int) and observed >= 10)
        item["audit_censor_reason"] = None if item["audit_observation_mask"] else "OBSERVED_SUFFIX_SHORTER_THAN_K10"
    for item in outputs:
        if item["heads"]["safe_release"]["value"] == TRUE and item["heads"]["k10_feasible"]["value"] == TRUE:
            raise ContractError("safe_release TRUE with k10 TRUE")
    return {
        "schema": RUNNER_SCHEMA,
        "episode_id": episode_id,
        "step_count": len(outputs),
        "steps": outputs,
        "unknown_to_false": sum(
            1 for item in outputs for head in item["heads"].values()
            if head["value"] == FALSE and not head["mask"]
        ),
        "forbidden_reads": 0,
    }


def _strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_file(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunnerHold(f"source file is not regular: {path}")
    if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
        raise RunnerHold(f"source file SHA/size mismatch: {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise RunnerHold(f"non-object JSONL row: {path}")
                rows.append(row)
    return rows


def _verify_root_seal(root: Path) -> None:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or sums.is_symlink() or sidecar.is_symlink() or not sums.is_file() or not sidecar.is_file():
        raise RunnerHold(f"sealed root is incomplete: {root}")
    sidecar_text = sidecar.read_text(encoding="utf-8").strip().split()
    if len(sidecar_text) != 2 or sidecar_text[1] != "SHA256SUMS" or sidecar_text[0] != sha256_file(sums):
        raise RunnerHold(f"SHA256SUMS sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RunnerHold(f"malformed SHA256SUMS entry: {root}")
        name = parts[1].strip()
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise RunnerHold(f"unsafe checksum path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != parts[0]:
            raise RunnerHold(f"sealed file mismatch: {path}")
        listed.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if listed != actual:
        raise RunnerHold(f"sealed file closure mismatch: {root}")


def _pilot_episode_ids(manifest: Mapping[str, Any]) -> set[str]:
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 40:
        raise RunnerHold("pilot records are not a closed 40-episode set")
    ids = {record.get("episode_id") for record in records if isinstance(record, Mapping)}
    if len(ids) != 40 or any(not isinstance(item, str) for item in ids):
        raise RunnerHold("pilot episode identities are not unique")
    return ids


def _verify_registry_binding(registry_root: Path) -> None:
    _verify_root_seal(registry_root)
    summary = registry_root / "run_A" / "ENTITY_REGISTRY_V2_SUMMARY.json"
    if not summary.is_file():
        raise RunnerHold("C1-V2 registry summary is missing")
    data = _strict_json(summary)
    if data.get("version") != "C1-V2" or data.get("n_tasks") != 40 or data.get("n_ok") != 40:
        raise RunnerHold("C1-V2 registry is not a closed 40-task PASS")
    task_rows = data.get("per_task")
    if not isinstance(task_rows, list) or len(task_rows) != 40 or any(row.get("status") != "OK" for row in task_rows):
        raise RunnerHold("C1-V2 per-task registry closure failed")


def _verify_geometry_binding(geometry_root: Path, geometry_allowlist: Path,
                             pilot_ids: set[str]) -> None:
    allowlist = _strict_json(geometry_allowlist)
    allowed = allowlist.get("allowed_episode_geometry_roots")
    expected_path = str(geometry_root.resolve())
    binding = next((item for item in (allowed or [])
                    if isinstance(item, Mapping)
                    and str(item.get("path", "")).rstrip("/") == expected_path.rstrip("/")), None)
    if binding is None:
        raise RunnerHold("geometry root is not explicitly allowlisted")
    _verify_root_seal(geometry_root)
    manifest_path = geometry_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RunnerHold("geometry dataset manifest is missing")
    if binding.get("manifest_sha256") is not None and binding["manifest_sha256"] != sha256_file(manifest_path):
        raise RunnerHold("geometry manifest is not bound by allowlist")
    if binding.get("sha256sums_sha256") is not None:
        actual_sums_sha = sha256_file(geometry_root / "SHA256SUMS")
        if binding["sha256sums_sha256"] != actual_sums_sha:
            raise RunnerHold("geometry root seal is not bound by allowlist")
    data = _strict_json(manifest_path)
    geometry_ids = {
        row.get("episode_id") for row in data.get("episodes", [])
        if isinstance(row, Mapping)
    }
    if geometry_ids != pilot_ids:
        raise RunnerHold("geometry root does not bind exactly the frozen pilot identities")
    if data.get("clean2000_payload_read") is True or data.get("protected_payload_read") is True:
        raise RunnerHold("geometry root declares protected/Clean2000 payload access")


def _load_geometry_episode(geometry_root: Path, episode_id: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    directory = geometry_root / "episodes" / episode_id.replace("/", "__")
    manifest_path = directory / "episode_manifest.json"
    cases_path = directory / "geometry_cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise RunnerHold(f"geometry episode files are missing: {episode_id}")
    manifest = _strict_json(manifest_path)
    if manifest.get("episode_id") != episode_id:
        raise RunnerHold(f"geometry episode identity mismatch: {episode_id}")
    rows = _load_jsonl(cases_path)
    expected_count = manifest.get("step_count")
    if not isinstance(expected_count, int) or len(rows) != expected_count:
        raise RunnerHold(f"geometry step count mismatch: {episode_id}")
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int) or step in by_step:
            raise RunnerHold(f"geometry step stream is not unique: {episode_id}")
        by_step[step] = row
    if sorted(by_step) != list(range(len(rows))):
        raise RunnerHold(f"geometry step stream is not contiguous: {episode_id}")
    return by_step, manifest


def _registry_object_ids(registry_root: Path, episode_id: str) -> set[str]:
    suite, task, _state = episode_id.split("/")
    task_key = f"{suite}/{task}"
    row_path = registry_root / "run_A" / "per_task" / (task_key.replace("/", "_") + ".json")
    data = _strict_json(row_path)
    legacy = data.get("legacy")
    if not isinstance(legacy, Mapping):
        raise RunnerHold(f"registry legacy relation data missing: {task_key}")
    if not isinstance(legacy.get("relations"), list):
        raise RunnerHold(f"registry relation list missing: {task_key}")
    ids = set()
    for relation in legacy.get("relations", []):
        if not isinstance(relation, Mapping):
            raise RunnerHold(f"malformed registry relation: {task_key}")
        resolution = relation.get("object_resolution")
        if not isinstance(resolution, Mapping):
            raise RunnerHold(f"object resolution missing: {task_key}")
        name = resolution.get("alias_to") or resolution.get("name")
        if not isinstance(name, str) or not name:
            raise RunnerHold(f"object identity missing: {task_key}")
        ids.add(name)
    return ids


def _sealed_output(output_parent: Path, output_name: str, episodes: Sequence[Mapping[str, Any]],
                   bindings: Mapping[str, Any], geometry_root: Path, smoke: bool = False) -> dict[str, Any]:
    output_parent.mkdir(parents=True, exist_ok=True)
    final = output_parent / output_name
    if final.exists() or final.is_symlink():
        raise RunnerHold(f"pilot output already exists: {final}")
    staging = output_parent / f".staging_{output_name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for episode in episodes:
            safe = str(episode["episode_id"]).replace("/", "__")
            directory = staging / "episodes" / safe
            directory.mkdir(parents=True)
            steps = episode["steps"]
            with (directory / "labels.jsonl").open("w", encoding="utf-8") as handle:
                for row in steps:
                    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            (directory / "episode_manifest.json").write_text(
                json.dumps({"episode_id": episode["episode_id"], "step_count": episode["step_count"],
                            "schema": RUNNER_SCHEMA}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": "C3_T1_V23_TEACHER_PILOT_OUTPUT_V1",
            "status": "PASS_FIT_DEV_SMOKE" if smoke else "PASS_FIT_DEV_ONLY",
            "episode_count": len(episodes),
            "episodes": [{"episode_id": item["episode_id"], "step_count": item["step_count"]}
                         for item in episodes],
            "geometry_root": str(geometry_root),
            "bindings": dict(bindings),
            "protected_payload_read": False,
            "model_inference": False,
            "student_training": False,
            "rollout": False,
            "attack": False,
        }
        (staging / "PILOT_OUTPUT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "runtime_audit.json").write_text(json.dumps({
            "status": "PASS", "episodes": len(episodes),
            "unknown_to_false": sum(int(item["unknown_to_false"]) for item in episodes),
            "forbidden_reads": sum(int(item["forbidden_reads"]) for item in episodes),
            "protected_payload_read": False, "model_inference": False,
            "student_training": False, "rollout": False, "attack": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(staging / name)}  {name}\n" for name in files), encoding="utf-8")
        sums_sha = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, final)
        return {"root": str(final), "status": "PASS", "sha256sums_sha256": sums_sha,
                "episode_count": len(episodes)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def execute_pilot(manifest_path: Path, registry_root: Path, geometry_root: Path,
                  geometry_allowlist: Path, predicate_contract: Path,
                  semantic_contract: Path, output_parent: Path, output_name: str,
                  selected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    preflight_result = preflight(manifest_path, registry_root, geometry_root,
                                 geometry_allowlist, predicate_contract, semantic_contract)
    manifest = _strict_json(manifest_path)
    semantic_config = load_semantic_contract(semantic_contract)
    records = sorted(manifest["records"], key=lambda item: item["episode_id"])
    if selected_ids is not None:
        requested = list(selected_ids)
        if len(set(requested)) != len(requested) or not requested:
            raise RunnerHold("selected episode identities are not unique")
        available = {str(item["episode_id"]) for item in records}
        if not set(requested) <= available:
            raise RunnerHold("selected episode identity is outside frozen pilot")
        records = [item for item in records if str(item["episode_id"]) in set(requested)]
        records.sort(key=lambda item: requested.index(str(item["episode_id"])))
    episodes = []
    for record in records:
        episode_id = str(record["episode_id"])
        files = {str(item["name"]): item for item in record.get("source_files", [])}
        sidecar_path = Path(str(record["source_episode_root"])) / "privileged_teacher_sidecar.jsonl"
        _verify_source_file(sidecar_path, files["privileged_teacher_sidecar.jsonl"])
        sidecar_rows = _load_jsonl(sidecar_path)
        geometry_cases, geometry_manifest = _load_geometry_episode(geometry_root, episode_id)
        if len(sidecar_rows) != geometry_manifest["step_count"]:
            raise RunnerHold(f"sidecar/geometry count mismatch: {episode_id}")
        threshold = geometry_manifest.get("qpos_close_threshold")
        object_ids = _registry_object_ids(registry_root, episode_id)
        episode = run_episode(sidecar_rows, geometry_cases, episode_id,
                              load_contract(predicate_contract), object_ids,
                              threshold, protocol_horizon_for_suite(episode_id.split("/", 1)[0]),
                              semantic_config)
        episodes.append(episode)
    return _sealed_output(output_parent, output_name, episodes,
                          {"preflight": preflight_result, "manifest_sha256": sha256_file(manifest_path),
                           "selected_episode_ids": [item["episode_id"] for item in records]},
                          geometry_root, smoke=selected_ids is not None)


def verify_required_contract_files(registry_root: Path, geometry_root: Path, geometry_allowlist: Path,
                                   predicate_contract: Path, semantic_contract: Path) -> dict[str, str]:
    paths = (registry_root, geometry_root, geometry_allowlist, predicate_contract, semantic_contract)
    if not all(path.exists() for path in paths):
        raise RunnerHold("required C1/C3 geometry contract root is not mounted")
    if not geometry_root.is_dir() or not geometry_allowlist.is_file():
        raise RunnerHold("geometry root/allowlist is not consumable")
    return {str(path): sha256_file(path) if path.is_file() else "DIRECTORY_SEAL_REQUIRED" for path in paths}


def preflight(manifest_path: Path, registry_root: Path, geometry_root: Path,
              geometry_allowlist: Path, predicate_contract: Path, semantic_contract: Path) -> dict[str, Any]:
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != "V23_DEV_PILOT_V1" or manifest.get("status") != "FROZEN_INPUT_BYTES_ONLY":
        raise RunnerHold("pilot input manifest is not frozen")
    if manifest.get("episode_count") != 40 or len(manifest.get("records", [])) != 40:
        raise RunnerHold("pilot manifest does not contain 40 episodes")
    pilot_ids = _pilot_episode_ids(manifest)
    bindings = verify_required_contract_files(registry_root, geometry_root, geometry_allowlist, predicate_contract, semantic_contract)
    _verify_registry_binding(registry_root)
    _verify_geometry_binding(geometry_root, geometry_allowlist, pilot_ids)
    contract = load_contract(predicate_contract)
    semantic = load_semantic_contract(semantic_contract)
    return {
        "schema": RUNNER_SCHEMA,
        "status": "PREFLIGHT_PASS_NO_EPISODE_EXECUTION",
        "manifest_sha256": sha256_file(manifest_path),
        "contract_bindings": bindings,
        "predicate_contract_schema": contract.get("schema"),
        "semantic_contract_schema": semantic.get("schema"),
        "protected_payload_read": False,
        "model_inference": False,
        "rollout": False,
        "attack": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--geometry-allowlist", required=True, type=Path)
    parser.add_argument("--predicate-contract", required=True, type=Path)
    parser.add_argument("--semantic-contract", required=True, type=Path)
    parser.add_argument("--execute-output-parent", type=Path)
    parser.add_argument("--execute-output-name")
    parser.add_argument("--episode-id", action="append", dest="episode_ids")
    args = parser.parse_args()
    try:
        if (args.execute_output_parent is None) != (args.execute_output_name is None):
            raise RunnerHold("execution output parent/name must be supplied together")
        if args.execute_output_parent is None:
            result = preflight(
                args.manifest, args.registry_root, args.geometry_root,
                args.geometry_allowlist, args.predicate_contract, args.semantic_contract,
            )
        else:
            result = execute_pilot(
                args.manifest, args.registry_root, args.geometry_root,
                args.geometry_allowlist, args.predicate_contract, args.semantic_contract,
                args.execute_output_parent, args.execute_output_name, args.episode_ids,
            )
    except (RunnerHold, ContractError) as exc:
        print(json.dumps({
            "schema": RUNNER_SCHEMA,
            "status": "PREFLIGHT_HOLD",
            "reason": str(exc),
            "protected_payload_read": False,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        }, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
