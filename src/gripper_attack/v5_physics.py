"""Clean-only physics criticality proxy for Official V5.

The functions here consume privileged clean evidence only.  They produce a
transparent proxy for ranking development windows; they do not estimate
counterfactual attack failure probability.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PHYSICS_PHASES = (
    "PRE_SUPPORT",
    "VALID_RETENTION",
    "RELEASE_IMMINENT_TAIL",
    "POST_RELEASE",
    "UNSTABLE_TRANSITION",
    "UNKNOWN",
)
PHYSICS_TEACHER_FIELDS = frozenset(
    {
        "step", "candidate_close", "student_valid", "gripper_contact_score", "object_contact", "support_contact",
        "relative_pose_stability", "object_eef_comotion_score", "lift_score", "target_progress", "target_progress_known",
        "task_grasp_necessity", "stable_grasp_score", "stable_grasp_dwell", "release_risk",
        "regrasp_or_instability_risk", "support_removed", "utility_score", "known_mask", "utility_tier", "phase_name",
        "teacher_confidence", "window_id", "window_start", "window_end", "suite", "task_idx", "manipulated_objects",
        "target_names", "support_names", "task_role_status", "task_role_reason", "physics_teacher_proxy",
        "counterfactual_attack_label", "canonical_parent_key", "state_id", "source_artifact_recursive_sha256",
        "physics_protocol_schema",
    }
)
PHYSICS_TEACHER_V21_FIELDS = PHYSICS_TEACHER_FIELDS | frozenset(
    {"causal_trigger_eligible", "component_valid_mask", "tier_onset_step"}
)
PHYSICS_TEACHER_V21C_FIELDS = PHYSICS_TEACHER_V21_FIELDS | frozenset(
    {"raw_gripper", "action_intent", "action_known"}
)
_ROLE_PREDICATES = {"In", "On", "Inside", "Contains", "Stack"}
_SUPPORT_SUFFIXES = (
    "_contain_region",
    "_heating_region",
    "_cook_region",
    "_top_region",
    "_bottom_region",
    "_middle_region",
    "_top_side",
    "_front_region",
    "_back_contain_region",
    "_region",
)


@dataclass(frozen=True)
class PhysicsTaskRole:
    suite: str
    task_idx: int
    manipulated_objects: tuple[str, ...]
    target_names: tuple[str, ...]
    support_names: tuple[str, ...]
    goal_predicates: tuple[tuple[str, str, str | None], ...]
    status: str
    reason: str

    @property
    def applicable(self) -> bool:
        return self.status == "PASS"


def parse_bddl_task_role(
    text: str, *, suite: str, task_idx: int, object_names: Sequence[str]
) -> PhysicsTaskRole:
    """Decode task roles from BDDL syntax, never from labels or results."""

    object_set = set(object_names)
    goal_match = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    init_match = re.search(r"\(:init\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not goal_match:
        return PhysicsTaskRole(suite, task_idx, (), (), (), (), "ABSTAIN_DECODER_HOLD", "goal section missing")
    predicates: list[tuple[str, str, str | None]] = []
    for match in re.finditer(r"\(([A-Za-z_]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)", goal_match.group(1)):
        predicate, first, second = match.groups()
        if predicate in _ROLE_PREDICATES:
            predicates.append((predicate, first, second))
    manipulated: list[str] = []
    targets: list[str] = []
    for _, first, second in predicates:
        if first in object_set and first not in manipulated:
            manipulated.append(first)
            if second:
                targets.append(second)
    if not manipulated:
        return PhysicsTaskRole(
            suite, task_idx, (), tuple(targets), (), tuple(predicates),
            "NO_MANIPULATION_TARGET", "goal is explicitly a non-grasp action with no BDDL object target",
        )
    supports: list[str] = []
    if init_match:
        for match in re.finditer(r"\(On\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", init_match.group(1)):
            if match.group(1) in manipulated and match.group(2) not in supports:
                supports.append(match.group(2))
    return PhysicsTaskRole(
        suite, task_idx, tuple(manipulated), tuple(targets), tuple(supports),
        tuple(predicates), "PASS", "goal object and target decoded from BDDL",
    )


def _finite_vector(value: Any, width: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    result = [float(item) for item in value]
    return result if all(math.isfinite(item) for item in result) else None


def _dist(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(item) ** 2 for item in left))
    right_norm = math.sqrt(sum(float(item) ** 2 for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _object_slice(object_slices: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return object_slices.get(name)


def _slice_vector(state: Sequence[float], spec: Mapping[str, Any], key: str) -> list[float] | None:
    bounds = spec.get(key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    start, end = int(bounds[0]), int(bounds[1])
    return _finite_vector(state[start:end], end - start)


def _base_name(value: str) -> str:
    result = value
    changed = True
    while changed:
        changed = False
        for suffix in _SUPPORT_SUFFIXES:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                break
    return result


def _endpoint_matches(endpoint: str, name: str) -> bool:
    return endpoint == name or endpoint.startswith(name + "_")


def _is_gripper(endpoint: str) -> bool:
    return "gripper0" in endpoint or "finger1" in endpoint or "finger2" in endpoint


def _contact_flags(
    pairs: Sequence[Sequence[Any]], role: PhysicsTaskRole
) -> tuple[bool, bool, bool]:
    object_contact = False
    gripper_contact = False
    support_contact = False
    support_bases = {_base_name(name) for name in role.support_names}
    for pair in pairs:
        endpoints = [str(item) for item in pair]
        if not any(_endpoint_matches(endpoint, name) for endpoint in endpoints for name in role.manipulated_objects):
            continue
        object_contact = True
        if any(_is_gripper(endpoint) for endpoint in endpoints):
            gripper_contact = True
        for endpoint in endpoints:
            if _is_gripper(endpoint):
                continue
            if any(_endpoint_matches(endpoint, base) for base in support_bases):
                support_contact = True
            if endpoint in {"floor", "table_collision"} or any(token in endpoint for token in ("stove", "cabinet", "shelf", "rack", "desk", "counter")):
                support_contact = True
    return object_contact, gripper_contact, support_contact


def _relative_stability(history: Sequence[Mapping[str, Any]], role: PhysicsTaskRole, slices: Mapping[str, Mapping[str, Any]], scale_pos: float, scale_quat: float, *, reducer: str = "mean") -> float:
    values: list[float] = []
    for name in role.manipulated_objects:
        spec = _object_slice(slices, name)
        if spec is None:
            continue
        positions = [_slice_vector(item.get("object_state", []), spec, "to_eef_pos") for item in history]
        quats = [_slice_vector(item.get("object_state", []), spec, "to_eef_quat") for item in history]
        positions = [item for item in positions if item is not None]
        quats = [item for item in quats if item is not None]
        if len(positions) >= 2 and len(quats) >= 2:
            reduce_values = _median if reducer == "median" else _mean
            pos_delta = reduce_values([_dist(a, b) for a, b in zip(positions[1:], positions[:-1])])
            quat_delta = reduce_values([_dist(a, b) for a, b in zip(quats[1:], quats[:-1])])
            values.append(_clip(math.exp(-(pos_delta / scale_pos) - (quat_delta / scale_quat))))
    return _mean(values, 0.0)


def _comotion(history: Sequence[Mapping[str, Any]], role: PhysicsTaskRole, slices: Mapping[str, Mapping[str, Any]], *, inactive_value: float | None = None) -> float:
    eef = [_finite_vector(item.get("robot0_eef_pos"), 3) for item in history]
    if len(eef) < 2 or any(item is None for item in eef):
        return 0.0 if inactive_value is not None else 0.5
    values: list[float] = []
    for name in role.manipulated_objects:
        spec = _object_slice(slices, name)
        if spec is None:
            continue
        positions = [_slice_vector(item.get("object_state", []), spec, "pos") for item in history]
        if len(positions) < 2 or any(item is None for item in positions):
            continue
        similarities = []
        for object_left, object_right, eef_left, eef_right in zip(positions[1:], positions[:-1], eef[1:], eef[:-1]):
            object_delta = [a - b for a, b in zip(object_left, object_right)]
            eef_delta = [a - b for a, b in zip(eef_left, eef_right)]
            if inactive_value is not None and (
                math.isclose(math.sqrt(sum(value * value for value in object_delta)), 0.0)
                or math.isclose(math.sqrt(sum(value * value for value in eef_delta)), 0.0)
            ):
                continue
            similarities.append((_cosine(object_delta, eef_delta) + 1.0) / 2.0)
        values.append(_mean(similarities, inactive_value if inactive_value is not None else 0.5))
    return _mean(values, inactive_value if inactive_value is not None else 0.5)


def _lift(history: Sequence[Mapping[str, Any]], current: Mapping[str, Any], role: PhysicsTaskRole, slices: Mapping[str, Mapping[str, Any]], scale: float) -> float:
    values: list[float] = []
    for name in role.manipulated_objects:
        spec = _object_slice(slices, name)
        if spec is None:
            continue
        initial = _slice_vector(history[0].get("object_state", []), spec, "pos")
        position = _slice_vector(current.get("object_state", []), spec, "pos")
        if initial is not None and position is not None:
            values.append(_clip((position[2] - initial[2]) / scale))
    return max(values, default=0.0)


def _target_progress(
    first: Mapping[str, Any], current: Mapping[str, Any], role: PhysicsTaskRole,
    slices: Mapping[str, Mapping[str, Any]], scale: float,
    *, unknown_default: float = 0.5,
) -> tuple[float, bool]:
    values: list[float] = []
    known = False
    for object_name, target_name in zip(role.manipulated_objects, role.target_names):
        object_spec = _object_slice(slices, object_name)
        target_spec = _object_slice(slices, target_name)
        if object_spec is None or target_spec is None:
            continue
        first_object = _slice_vector(first.get("object_state", []), object_spec, "pos")
        first_target = _slice_vector(first.get("object_state", []), target_spec, "pos")
        current_object = _slice_vector(current.get("object_state", []), object_spec, "pos")
        current_target = _slice_vector(current.get("object_state", []), target_spec, "pos")
        if None not in (first_object, first_target, current_object, current_target):
            known = True
            initial_distance = _dist(first_object, first_target)
            current_distance = _dist(current_object, current_target)
            values.append(_clip((initial_distance - current_distance) / scale))
    return _mean(values, unknown_default), known


def _candidate_close(step: Mapping[str, Any], threshold: float) -> bool:
    """FIXED (Gate D2.1.3): Returns structured CanonicalActionState via shared helper.

    BOUNDARY, missing, and non-finite → action_known=False, candidate_close=False.
    Only known CLOSE steps return candidate_close=True.

    Previous bug: used raw[6] >= threshold, which selected the OPEN region.
    """
    from .action_contract import CanonicalActionState
    state = CanonicalActionState.from_step(step, field="clean_action_raw_7d")
    step["_action_state"] = state
    return state.candidate_close


def _assign_segments(rows: list[dict[str, Any]]) -> None:
    segment = -1
    start = 0
    end = -1
    for index, row in enumerate(rows):
        if row["candidate_close"]:
            if index == 0 or not rows[index - 1]["candidate_close"]:
                if segment >= 0:
                    for prior in rows[start : end + 1]:
                        prior["window_end"] = end
                segment += 1
                start = index
            end = index
            row["window_id"] = f"candidate:{segment}"
            row["window_start"] = start
            row["window_end"] = end
        else:
            if segment >= 0 and index > start and rows[index - 1]["candidate_close"]:
                for prior in rows[start : end + 1]:
                    prior["window_end"] = end
            row["window_id"] = f"none:{index}"
            row["window_start"] = index
            row["window_end"] = index
    if segment >= 0:
        for prior in rows[start : end + 1]:
            prior["window_end"] = end


def derive_episode_rows(
    step_rows: Sequence[Mapping[str, Any]], sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole, object_slices: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(step_rows) != len(sidecar_rows) or not step_rows:
        raise ValueError("step and physics sidecar lengths must match and be nonzero")
    constants = protocol["fixed_constants"]
    schema = protocol.get("schema", "")
    v21 = schema in ("DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21",
                     "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL")
    history_size = int(protocol["history"]["score_window_steps"])
    threshold = float(protocol["candidate_close"]["close_threshold"])
    base: list[dict[str, Any]] = []
    previous_contact: list[tuple[bool, bool, bool]] = []
    for index, (step, sidecar) in enumerate(zip(step_rows, sidecar_rows)):
        history = sidecar_rows[max(0, index - history_size + 1) : index + 1]
        contact = _contact_flags(sidecar.get("mujoco_contact_pairs", []), role)
        previous_contact.append(contact)
        stable = _relative_stability(history, role, object_slices, float(constants["relative_position_scale_m"]), float(constants["relative_quaternion_scale"]), reducer="median" if v21 else "mean")
        comotion = _comotion(history, role, object_slices, inactive_value=0.0 if v21 else None)
        lift = _lift(sidecar_rows[: index + 1], sidecar, role, object_slices, float(constants["lift_scale_m"]))
        target_progress, target_known = _target_progress(sidecar_rows[0], sidecar, role, object_slices, float(constants["target_progress_scale_m"]), unknown_default=0.0 if v21 else 0.5)
        cc = _candidate_close(step, threshold)
        action_state = step.get("_action_state")
        base.append({
            "step": index,
            "candidate_close": cc,
            "student_valid": bool(step.get("valid", True)),
            "gripper_contact_score": 1.0 if contact[1] else 0.0,
            "object_contact": contact[0],
            "support_contact": contact[2],
            "relative_pose_stability": stable,
            "object_eef_comotion_score": comotion,
            "lift_score": lift,
            "target_progress": target_progress,
            "target_progress_known": target_known,
            "task_grasp_necessity": 1.0 if role.applicable else 0.0,
            "stable_grasp_score": _clip(
                0.35 * stable
                + 0.25 * comotion
                + 0.20 * (1.0 if contact[1] else 0.0)
                + 0.20 * lift
            ),
            "raw_gripper": action_state.raw_gripper if action_state is not None else None,
            "action_intent": action_state.action_intent if action_state is not None else "UNKNOWN",
            "action_known": action_state.action_known if action_state is not None else False,
        })
    _assign_segments(base)
    for index, row in enumerate(base):
        future = base[index : min(len(base), index + history_size)]
        # D2.1.3: future_open uses base row fields (not _action_state from step_rows)
        future_open = any(
            item.get("action_known") is True
            and item.get("action_intent") == "OPEN"
            for item in future[1:]
        )
        future_contact_loss = any(item["gripper_contact_score"] < row["gripper_contact_score"] for item in future[1:])
        contact_history = [item["gripper_contact_score"] for item in base[max(0, index - history_size + 1) : index + 1]]
        toggles = sum(left != right for left, right in zip(contact_history[1:], contact_history[:-1]))
        toggle_rate = toggles / max(1, len(contact_history) - 1)
        if v21 and row["candidate_close"]:
            start = int(row["window_start"])
            prior_support = any(item["support_contact"] for item in base[:start])
            current_window_support_free = not any(item["support_contact"] for item in base[start:index + 1])
            row["support_removed"] = 1.0 if prior_support and current_window_support_free else 0.0
        else:
            row["support_removed"] = 1.0 if row["support_contact"] is False and any(item["support_contact"] for item in base[: index + 1]) else 0.0
        row["release_risk"] = 0.5 * float(future_open) + 0.5 * float(future_contact_loss)
        row["regrasp_or_instability_risk"] = _clip(0.5 * (1.0 - row["relative_pose_stability"]) + 0.5 * toggle_rate)
        row["stable_grasp_score"] = _clip(
            0.35 * row["relative_pose_stability"]
            + 0.25 * row["object_eef_comotion_score"]
            + 0.20 * row["gripper_contact_score"]
            + 0.20 * row["lift_score"]
        )
        row["stable_grasp_dwell"] = sum(
            item["stable_grasp_score"] >= float(constants["tier2_min_stable_grasp"])
            for item in base[max(0, index - int(protocol["history"]["minimum_stable_grasp_dwell"]) + 1) : index + 1]
        )
        positive = (
            0.30 * row["stable_grasp_score"]
            + 0.20 * row["lift_score"]
            + 0.20 * row["object_eef_comotion_score"]
            + 0.15 * row["support_removed"]
        )
        if row["target_progress_known"]:
            positive += 0.15 * row["target_progress"]
        elif v21:
            positive /= 0.85
        row["utility_score"] = _clip(positive - 0.20 * row["release_risk"] - 0.20 * row["regrasp_or_instability_risk"])
        if not role.applicable or not row["student_valid"] or not row["action_known"]:
            row["known_mask"] = False
            row["utility_tier"] = None
            row["phase_name"] = "UNKNOWN"
        else:
            row["known_mask"] = True
            if row["release_risk"] >= float(constants["tier2_max_release_risk"]):
                row["phase_name"] = "RELEASE_IMMINENT_TAIL"
            elif row["regrasp_or_instability_risk"] >= float(constants["tier2_max_regrasp_risk"]):
                row["phase_name"] = "UNSTABLE_TRANSITION"
            elif row["candidate_close"] and row["stable_grasp_score"] >= float(constants["tier2_min_stable_grasp"]):
                row["phase_name"] = "VALID_RETENTION"
            elif row["candidate_close"]:
                row["phase_name"] = "PRE_SUPPORT"
            else:
                row["phase_name"] = "UNKNOWN"
            if row["utility_score"] >= float(constants["tier3_min_utility"]) and row["stable_grasp_score"] >= float(constants["tier3_min_stable_grasp"]) and row["lift_score"] >= float(constants["tier3_min_lift"]) and row["stable_grasp_dwell"] >= int(protocol["history"]["minimum_stable_grasp_dwell"]) and row["release_risk"] <= float(constants["tier3_max_release_risk"]) and row["regrasp_or_instability_risk"] <= float(constants["tier3_max_regrasp_risk"]):
                row["utility_tier"] = 3
            elif row["utility_score"] >= float(constants["tier2_min_utility"]) and row["stable_grasp_score"] >= float(constants["tier2_min_stable_grasp"]) and row["stable_grasp_dwell"] >= int(protocol["history"]["minimum_stable_grasp_dwell"]) and row["release_risk"] <= float(constants["tier2_max_release_risk"]) and row["regrasp_or_instability_risk"] <= float(constants["tier2_max_regrasp_risk"]):
                row["utility_tier"] = 2
            elif row["candidate_close"] and row["utility_score"] >= float(constants["tier1_min_utility"]):
                row["utility_tier"] = 1
            else:
                row["utility_tier"] = 0
        row["teacher_confidence"] = (
            0.0 if not row["known_mask"] else
            1.0 if role.applicable and row["target_progress_known"] else
            0.8 if role.applicable else 0.0
        )
    if v21:
        for segment_id in {row["window_id"] for row in base if row["candidate_close"]}:
            members = [row for row in base if row["window_id"] == segment_id]
            onset = next((int(row["step"]) for row in members if row["utility_tier"] is not None and int(row["utility_tier"]) >= 2), None)
            for row in members:
                row["tier_onset_step"] = onset
                row["causal_trigger_eligible"] = bool(
                    row["candidate_close"] and row["student_valid"] and row["known_mask"]
                    and row["stable_grasp_dwell"] >= int(protocol["history"]["minimum_stable_grasp_dwell"])
                    and row["release_risk"] < float(constants["tier2_max_release_risk"])
                    and row["regrasp_or_instability_risk"] < float(constants["tier2_max_regrasp_risk"])
                )
                row["component_valid_mask"] = {
                    "relative_pose_stability": bool(role.applicable),
                    "object_eef_comotion_score": bool(role.applicable),
                    "lift_score": bool(role.applicable),
                    "target_progress": bool(row["target_progress_known"]),
                    "support_removed": bool(role.applicable),
                    "release_risk": bool(role.applicable),
                    "regrasp_or_instability_risk": bool(role.applicable),
                }
        for row in base:
            row.setdefault("tier_onset_step", None)
            row.setdefault("causal_trigger_eligible", False)
            row.setdefault("component_valid_mask", {
                "relative_pose_stability": False,
                "object_eef_comotion_score": False,
                "lift_score": False,
                "target_progress": False,
                "support_removed": False,
                "release_risk": False,
                "regrasp_or_instability_risk": False,
            })
    output: list[dict[str, Any]] = []
    for row in base:
        output.append({
            **row,
            "suite": role.suite,
            "task_idx": role.task_idx,
            "manipulated_objects": list(role.manipulated_objects),
            "target_names": list(role.target_names),
            "support_names": list(role.support_names),
            "task_role_status": role.status,
            "task_role_reason": role.reason,
            "physics_teacher_proxy": True,
            "counterfactual_attack_label": False,
        })
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in output:
        window_id = row["window_id"]
        if window_id in seen or window_id.startswith("none:"):
            continue
        seen.add(window_id)
        members = [item for item in output if item["window_id"] == window_id]
        tiers = [item["utility_tier"] for item in members if item["utility_tier"] is not None]
        windows.append({
            "window_id": window_id,
            "start_step": members[0]["step"],
            "end_step": members[-1]["step"],
            "step_count": len(members),
            "phase_name": max((item["phase_name"] for item in members), key=lambda value: (value == "VALID_RETENTION", value)),
            "utility_tier": max(tiers) if tiers else None,
            "known": bool(tiers),
            "candidate_close": True,
            "rankable": bool(tiers) and role.applicable,
        })
    return output, windows


__all__ = ["PHYSICS_PHASES", "PHYSICS_TEACHER_FIELDS", "PHYSICS_TEACHER_V21_FIELDS", "PHYSICS_TEACHER_V21C_FIELDS", "PhysicsTaskRole", "parse_bddl_task_role", "derive_episode_rows"]
