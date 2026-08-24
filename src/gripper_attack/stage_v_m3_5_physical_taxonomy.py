"""Outcome-blind M3.5 object and contact telemetry.

The binding comes from the task BDDL goal and the live MuJoCo model.  No task
success, branch outcome, or future label is consumed here.  Missing or
ambiguous geometry is an explicit invalid telemetry record, never a default.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "STAGE_V_M3_5_PHYSICAL_TAXONOMY_V2"
_GOAL_OBJECT_RE = re.compile(r"\((?:In|On)\s+([A-Za-z0-9_]+)\s+", re.IGNORECASE)
_OBJECT_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_ ]+)\s+-\s+([A-Za-z0-9_]+)\s*$")
_GRIPPER_WORDS = ("gripper", "finger", "eef", "hand")
ALREADY_OPEN_APERTURE_MIN = 0.03
APERTURE_RESPONSE_DELTA_MIN = 0.005
PHYSICAL_THRESHOLDS = {
    "contact_absence_consecutive_frames": 2,
    "object_release_l2_threshold": 0.01,
    "object_drop_height_threshold": 0.02,
    "already_open_aperture_min": ALREADY_OPEN_APERTURE_MIN,
    "aperture_response_delta_min": APERTURE_RESPONSE_DELTA_MIN,
}


class PhysicalTaxonomyError(ValueError):
    """Raised only for malformed static binding input."""


def _section(text: str, name: str) -> str:
    match = re.search(r"\(:" + re.escape(name) + r"\b(.*?)(?=\n\s*\(:|\n\s*\)\s*$)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def parse_declared_object_ids(bddl_text: str) -> tuple[str, ...]:
    """Return object IDs declared in the BDDL ``:objects`` section."""
    result: list[str] = []
    for line in _section(bddl_text, "objects").splitlines():
        match = _OBJECT_LINE_RE.match(line)
        if not match:
            continue
        result.extend(token for token in match.group(1).split() if token not in result)
    return tuple(result)


def parse_goal_object_ids(bddl_text: str) -> tuple[str, ...]:
    """Return source objects in the registered ``In``/``On`` goal predicates."""
    declared = set(parse_declared_object_ids(bddl_text))
    goal = _section(bddl_text, "goal")
    result: list[str] = []
    for object_id in _GOAL_OBJECT_RE.findall(goal):
        if object_id in declared and object_id not in result:
            result.append(object_id)
    return tuple(result)


def taxonomy_eligibility_from_bddl(path: Path) -> dict[str, Any]:
    """Classify fixture-only goals prospectively without inventing an object."""
    text = path.read_text(encoding="utf-8")
    declared = parse_declared_object_ids(text)
    targets = parse_goal_object_ids(text)
    return {
        "schema": SCHEMA,
        "status": "PASS" if targets else "INELIGIBLE",
        "eligible": bool(targets),
        "reason": "" if targets else "UNSUPPORTED_FIXTURE_ONLY_GOAL_NO_IN_OR_ON_SOURCE_OBJECT",
        "rule": "at least one declared source object in an In/On goal predicate",
        "declared_object_ids": list(declared),
        "target_object_ids": list(targets),
        "fixture_binding_inference_allowed": False,
    }


def target_object_ids_from_bddl(path: Path) -> tuple[str, ...]:
    targets = tuple(taxonomy_eligibility_from_bddl(path)["target_object_ids"])
    if not targets:
        raise PhysicalTaxonomyError("GOAL_OBJECT_BINDING_EMPTY")
    return targets


def _name(value: Any) -> str:
    return str(value or "")


def _finite_vector(value: Any) -> list[float] | None:
    try:
        values = list(value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    result = [float(item) for item in values]
    return result if all(math.isfinite(item) for item in result) else None


def aperture_metric(qpos: Any) -> float | None:
    """Use the established absolute finger-qpos sum as the aperture metric."""
    try:
        values = [float(item) for item in qpos]
    except (TypeError, ValueError):
        return None
    if not values or not all(math.isfinite(item) for item in values):
        return None
    return float(sum(abs(item) for item in values))


def build_forced_open_action(control_raw_action: Any, control_env_action: Any) -> dict[str, Any]:
    """Replace only the gripper component of a matched control action."""
    try:
        raw = [float(item) for item in control_raw_action]
        env = [float(item) for item in control_env_action]
    except (TypeError, ValueError) as exc:
        raise PhysicalTaxonomyError("ACTION_VECTOR_INVALID") from exc
    if len(raw) != 7 or len(env) != 7 or not all(math.isfinite(item) for item in raw + env):
        raise PhysicalTaxonomyError("ACTION_VECTOR_DIMENSION_OR_FINITE_INVALID")
    forced_raw = [*raw[:6], 1.0]
    forced_env = [*env[:6], -1.0]
    arm_delta = [forced_env[index] - env[index] for index in range(6)]
    if any(abs(item) > 1e-7 for item in arm_delta):
        raise PhysicalTaxonomyError("SURGICAL_ARM_ISOLATION_FAILED")
    return {
        "raw_policy_action": forced_raw,
        "normalized_action": forced_raw,
        "env_action": forced_env,
        "arm_delta": arm_delta,
        "arm_delta_linf": max((abs(item) for item in arm_delta), default=0.0),
        "gripper_delta_env": forced_env[-1] - env[-1],
        "forced_gripper_raw": 1.0,
        "forced_gripper_env": -1.0,
    }


def evaluate_treatment_compliance(
    receipts: list[Mapping[str, Any]],
    *,
    expected_steps: int | None = None,
    aperture_delta_min: float = APERTURE_RESPONSE_DELTA_MIN,
    already_open_min: float = ALREADY_OPEN_APERTURE_MIN,
) -> dict[str, Any]:
    """Evaluate command delivery and physical aperture response separately."""
    if not receipts:
        return {
            "treatment_compliant": False,
            "command_delivery_valid": False,
            "compliance_reason": "NO_TREATMENT_STEPS",
            "delivered_open_steps": 0,
            "expected_open_steps": expected_steps,
        }
    command_failures: list[str] = []
    pre_metrics: list[float] = []
    post_metrics: list[float] = []
    if expected_steps is not None and len(receipts) != int(expected_steps):
        command_failures.append(f"DELIVERED_STEP_COUNT:{len(receipts)}/{int(expected_steps)}")
    for index, row in enumerate(receipts):
        raw = row.get("raw_policy_action")
        normalized = row.get("normalized_action")
        env = row.get("env_action")
        try:
            raw_valid = isinstance(raw, list) and len(raw) == 7 and all(math.isfinite(float(item)) for item in raw) and abs(float(raw[-1]) - 1.0) <= 1e-7
            normalized_valid = isinstance(normalized, list) and len(normalized) == 7 and all(math.isfinite(float(item)) for item in normalized) and abs(float(normalized[-1]) - 1.0) <= 1e-7
            env_valid = isinstance(env, list) and len(env) == 7 and all(math.isfinite(float(item)) for item in env) and abs(float(env[-1]) + 1.0) <= 1e-7
        except (TypeError, ValueError):
            raw_valid = normalized_valid = env_valid = False
        if not raw_valid:
            command_failures.append(f"STEP_{index}_RAW_OPEN_INVALID")
        if not normalized_valid:
            command_failures.append(f"STEP_{index}_NORMALIZED_OPEN_INVALID")
        if not env_valid:
            command_failures.append(f"STEP_{index}_ENV_OPEN_INVALID")
        try:
            arm_delta_linf = float(row.get("arm_delta_linf", math.inf))
        except (TypeError, ValueError):
            arm_delta_linf = math.inf
        if not math.isfinite(arm_delta_linf) or arm_delta_linf > 1e-7:
            command_failures.append(f"STEP_{index}_ARM_DELTA_NONZERO")
        pre = row.get("pre_aperture")
        post = row.get("post_aperture")
        try:
            if pre is not None and math.isfinite(float(pre)):
                pre_metrics.append(float(pre))
            if post is not None and math.isfinite(float(post)):
                post_metrics.append(float(post))
        except (TypeError, ValueError):
            command_failures.append(f"STEP_{index}_APERTURE_INVALID")
    if command_failures:
        return {
            "treatment_compliant": False,
            "command_delivery_valid": False,
            "already_open_state": False,
            "aperture_response": False,
            "compliance_reason": "COMMAND_DELIVERY_INVALID",
            "command_failures": command_failures,
            "delivered_open_steps": len(receipts),
            "expected_open_steps": expected_steps,
        }
    # Command delivery is the treatment gate.  Aperture movement is a
    # descriptive mediator and may be blocked by contact or dynamics; it is
    # not permission to relabel a delivered OPEN command as noncompliance.
    already_open = bool(pre_metrics and pre_metrics[0] >= float(already_open_min))
    response_delta = (max(post_metrics) - pre_metrics[0]) if pre_metrics and post_metrics else None
    response = response_delta is not None and response_delta >= float(aperture_delta_min)
    return {
        "treatment_compliant": True,
        "command_delivery_valid": True,
        "already_open_state": already_open,
        "aperture_response": bool(response),
        "pre_aperture": pre_metrics[0] if pre_metrics else None,
        "max_post_aperture": max(post_metrics) if post_metrics else None,
        "aperture_delta": response_delta,
        "compliance_reason": "COMMAND_DELIVERY",
        "mediator_reason": "ALREADY_OPEN" if already_open else ("APERTURE_RESPONSE" if response else "APERTURE_RESPONSE_NOT_SATISFIED"),
        "command_failures": [],
        "delivered_open_steps": len(receipts),
        "expected_open_steps": expected_steps,
    }


def repeatability_receipt(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen 3-of-3 outcome gate without promoting abstains."""
    if len(rows) != 3:
        return {"status": "HOLD_STOCHASTIC_INTERVENTION_OUTCOME", "reason": "REPETITION_COUNT_INVALID"}
    classes = [str(row.get("outcome_class", "UNKNOWN")) for row in rows]
    if len(set(classes)) != 1:
        return {"status": "HOLD_STOCHASTIC_INTERVENTION_OUTCOME", "reason": "VALID_OUTCOME_DISCORDANCE", "classes": classes}
    if classes[0].endswith("_ABSTAIN") or classes[0] in {"UNKNOWN", "HORIZON_CENSORED"}:
        return {"status": "STABLE_ABSTAIN", "reason": "IDENTICAL_ABSTAIN_CLASS", "outcome_class": classes[0]}
    if not all(row.get("treatment_compliant") is True for row in rows):
        return {"status": "TREATMENT_NONCOMPLIANCE_ABSTAIN", "reason": "ONE_OR_MORE_REPETITIONS_NONCOMPLIANT"}
    return {"status": "PASS_REPEATABILITY_3_OF_3", "outcome_class": classes[0], "repetitions": 3}


def v_phys_label(*, control_valid: bool, treatment_valid: bool, f_control: int | None, f_open: int | None) -> str:
    """Return the registered truth-table class; never coerce unknown to zero."""
    if f_control == 1:
        return "CONTROL_CONTAMINATION_ABSTAIN" if f_open == 1 else "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    if not control_valid:
        return "CONTROL_INVALID_ABSTAIN"
    if not treatment_valid:
        return "TREATMENT_INVALID_ABSTAIN"
    if f_control is None or f_open is None:
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    return "V_PHYS" if f_open == 1 else "NO_PHYSICAL_VULNERABILITY"


def _model_names(model: Any, count_attr: str, name_fn: str) -> list[str]:
    count = int(getattr(model, count_attr, 0) or 0)
    fn = getattr(model, name_fn, None)
    names: list[str] = []
    for index in range(count):
        try:
            names.append(_name(fn(index)) if callable(fn) else "")
        except Exception:
            names.append("")
    return names


def _body_id(model: Any, target: str, body_names: list[str]) -> int | None:
    exact = [index for index, name in enumerate(body_names) if name == target]
    if exact:
        return exact[0]
    prefix = [
        index for index, name in enumerate(body_names)
        if name.startswith(target + "_") or name.startswith(target + "/")
    ]
    return min(prefix, key=lambda index: (len(body_names[index]), index)) if prefix else None


def bind_object_taxonomy(env: Any, bddl_path: Path) -> dict[str, Any]:
    """Bind BDDL goal objects to one live MuJoCo body each, fail-closed."""
    eligibility = taxonomy_eligibility_from_bddl(bddl_path)
    targets = tuple(eligibility["target_object_ids"])
    if not targets:
        return {**eligibility, "bddl_path": str(bddl_path), "target_body_ids": {}, "eef_site_id": None}
    model = getattr(getattr(env, "sim", None), "model", None)
    if model is None:
        return {"schema": SCHEMA, "status": "ABSTAIN", "reason": "SIM_MODEL_MISSING", "target_object_ids": list(targets)}
    body_names = _model_names(model, "nbody", "body_id2name")
    body_ids = {target: _body_id(model, target, body_names) for target in targets}
    site_id = None
    try:
        site_id = int(model.site_name2id("gripper0_grip_site"))
    except Exception:
        pass
    missing = [target for target, body_id in body_ids.items() if body_id is None]
    status = "PASS" if not missing and site_id is not None else "ABSTAIN"
    return {
        "schema": SCHEMA,
        "status": status,
        "reason": "" if status == "PASS" else "OBJECT_OR_EEF_BINDING_MISSING",
        "bddl_path": str(bddl_path),
        "target_object_ids": list(targets),
        "target_body_ids": {key: value for key, value in body_ids.items()},
        "target_body_names": {key: body_names[value] for key, value in body_ids.items() if value is not None},
        "eef_site_name": "gripper0_grip_site",
        "eef_site_id": site_id,
        "body_names": body_names,
        "taxonomy_eligibility": eligibility,
    }


def _body_is_gripper(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _GRIPPER_WORDS)


def _contact_body_ids(model: Any, contact: Any) -> tuple[int, int]:
    return int(model.geom_bodyid[int(contact.geom1)]), int(model.geom_bodyid[int(contact.geom2)])


def telemetry_from_env(env: Any, binding: Mapping[str, Any], *, target_object_id: str | None = None) -> dict[str, Any]:
    """Collect current geometry/contact facts without reading task outcome."""
    base = {
        "schema": SCHEMA,
        "contact_telemetry_valid": False,
        "object_identity": str(target_object_id or ""),
        "registered_target_object_ids": [str(item) for item in binding.get("target_object_ids", [])],
        "object_position": None,
        "eef_position": None,
        "object_eef_distance_m": None,
        "object_gripper_contact": False,
        "object_support_contact": False,
        "telemetry_reason": "",
    }
    if binding.get("status") != "PASS":
        base["telemetry_reason"] = str(binding.get("reason") or "OBJECT_BINDING_INVALID")
        return base
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    try:
        registered = [str(item) for item in binding.get("target_object_ids", [])]
        if target_object_id is not None and str(target_object_id) not in registered:
            base["telemetry_reason"] = "TARGET_OBJECT_NOT_REGISTERED"
            return base
        eef = _finite_vector(data.site_xpos[int(binding["eef_site_id"])])
        body_positions = data.body_xpos
        candidates = []
        for object_id in registered:
            if target_object_id is not None and object_id != str(target_object_id):
                continue
            body_id = int(binding["target_body_ids"][object_id])
            position = _finite_vector(body_positions[body_id])
            if position is not None:
                distance = math.sqrt(sum((position[index] - eef[index]) ** 2 for index in range(3))) if eef else math.inf
                candidates.append((distance, str(object_id), body_id, position))
        if eef is None or not candidates:
            base["telemetry_reason"] = "OBJECT_OR_EEF_POSITION_INVALID"
            return base
        _distance, selected_id, selected_body, position = min(candidates)
        target_body_ids = {int(selected_body)}
        body_names = list(binding.get("body_names", []))
        object_gripper_contact = False
        object_support_contact = False
        for index in range(int(getattr(data, "ncon", 0) or 0)):
            left, right = _contact_body_ids(model, data.contact[index])
            if left not in target_body_ids and right not in target_body_ids:
                continue
            other = right if left in target_body_ids else left
            other_name = body_names[other] if 0 <= other < len(body_names) else str(other)
            if _body_is_gripper(other_name):
                object_gripper_contact = True
            else:
                object_support_contact = True
        distance = math.sqrt(sum((position[index] - eef[index]) ** 2 for index in range(3)))
        base.update({
            "contact_telemetry_valid": True,
            "object_identity": selected_id,
            "selected_target_body_id": selected_body,
            "object_position": position,
            "eef_position": eef,
            "object_eef_distance_m": distance,
            "object_gripper_contact": object_gripper_contact,
            "object_support_contact": object_support_contact,
        })
        return base
    except Exception as exc:
        base["telemetry_reason"] = f"CONTACT_TELEMETRY_ERROR:{type(exc).__name__}"
        return base


__all__ = [
    "SCHEMA",
    "ALREADY_OPEN_APERTURE_MIN",
    "APERTURE_RESPONSE_DELTA_MIN",
    "PHYSICAL_THRESHOLDS",
    "PhysicalTaxonomyError",
    "aperture_metric",
    "build_forced_open_action",
    "bind_object_taxonomy",
    "evaluate_treatment_compliance",
    "parse_declared_object_ids",
    "parse_goal_object_ids",
    "target_object_ids_from_bddl",
    "telemetry_from_env",
    "taxonomy_eligibility_from_bddl",
    "repeatability_receipt",
    "v_phys_label",
]
