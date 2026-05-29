"""Extract teacher-privileged simulation state from LIBERO MuJoCo env.

Privileged state (object/target pose, distance) is TEACHER-ONLY.
It must NOT enter deployed student features.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


def _normalize_name(name: str) -> str:
    """Normalize object/receptacle name for matching."""
    return name.lower().strip().replace("_", " ").replace("-", " ").replace("the ", "")


def classify_mechanism(task_instruction: str) -> dict[str, Any]:
    """Classify task mechanism type and extract parsed objects.

    Returns dict with:
        mechanism_type, gripper_duty_eligible, parsed_object, parsed_target,
        target_type, segments, parser_confidence, unsupported_reason
    """
    text = task_instruction.lower().strip()

    # Default
    result: dict[str, Any] = {
        "mechanism_type": "unsupported_or_low_signal",
        "gripper_duty_eligible": False,
        "parsed_object": None,
        "parsed_target": None,
        "target_type": "unknown",
        "segments": [],
        "parser_confidence": "low",
        "unsupported_reason": "",
    }

    # ── Pattern: "pick up the X and place it in/on the Y" (Object, Spatial) ──
    m = re.search(r"pick up the (.+?) and place it (?:in|on) the (.+)", text)
    if m:
        result["mechanism_type"] = "pick_place_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(2).strip()
        result["target_type"] = "body_or_site"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "put both the X and the Y in/on the Z" (L10 multi-object) ──
    m = re.search(r"put both the (.+?) and the (.+?) (?:in|on) the (.+)", text)
    if m:
        result["mechanism_type"] = "multi_object_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(3).strip()
        result["target_type"] = "body_or_site"
        result["segments"] = [
            {"object": m.group(1).strip(), "target": m.group(3).strip()},
            {"object": m.group(2).strip(), "target": m.group(3).strip()},
        ]
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "put both Xs on the Y" (L10) ──
    m = re.search(r"put both (.+?) on the (.+)", text)
    if m:
        result["mechanism_type"] = "multi_object_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(2).strip()
        result["target_type"] = "body_or_site"
        result["parser_confidence"] = "medium"
        return result

    # ── Pattern: "put the X on/in the Y and put the Z on/to the W" (L10 multi-step, BEFORE simple put) ──
    m = re.search(r"put the (.+?) (?:in|on) the (.+?) and put the (.+?) (?:in|on|to|on top of) the (.+)", text)
    if m:
        result["mechanism_type"] = "multi_object_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(2).strip()
        result["target_type"] = "body_or_site"
        result["segments"] = [
            {"object": m.group(1).strip(), "target": m.group(2).strip()},
            {"object": m.group(3).strip(), "target": m.group(4).strip()},
        ]
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "put the X in/on/to/on top of the Y" (simple, Goal/L10, AFTER multi-step) ──
    m = re.search(r"put the (.+?) (?:in|on|to|on top of) the (.+?)(?: and (?:put|close)|$)", text)
    if m:
        obj = m.group(1).strip()
        rec = m.group(2).strip()
        rec = re.split(r"\s+and\s+put\b", rec)[0].strip()
        rec = re.split(r"\s+and\s+close\b", rec)[0].strip()
        result["mechanism_type"] = "pick_place_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = obj
        result["parsed_target"] = rec
        result["target_type"] = "body_or_site"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "open the X" (articulated, AFTER compound open+put) ──
    if " and put " not in text and " and close" not in text:
        m = re.search(r"open the (.+?)(?: of the .+)?$", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = None
        result["target_type"] = "articulated_joint"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "open the X and put the Y inside/in" ──
    m = re.search(r"open the (.+?) and put the (.+?) (?:inside|in)", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_object"] = m.group(2).strip()  # the object being moved
        result["parsed_target"] = m.group(1).strip()   # the drawer/door
        result["target_type"] = "articulated_joint"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "turn on the X and put the Y on it" ──
    m = re.search(r"turn (?:on|off) the (.+?) and put the (.+?) on", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_object"] = m.group(2).strip()
        result["parsed_target"] = m.group(1).strip()
        result["target_type"] = "articulated_joint"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "close the X" or "put X in Y and close it" ──
    m = re.search(r"(?:put the .+? (?:in|on) the .+? and )?close (?:it|the )?(.+)", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_target"] = m.group(1).strip() if m.group(1) else "door_or_drawer"
        result["unsupported_reason"] = "articulated_close_action"
        result["parser_confidence"] = "medium"
        return result

    # ── Pattern: "turn on/off the X" ──
    m = re.search(r"turn (?:on|off) the (.+)", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_target"] = m.group(1).strip()
        result["target_type"] = "articulated_joint"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "push the X to the Y" ──
    m = re.search(r"push the (.+?) (?:to|into) the (.+)", text)
    if m:
        result["mechanism_type"] = "planar_rearrangement_or_spatial_relation"
        result["gripper_duty_eligible"] = False
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(2).strip()
        result["target_type"] = "spatial_region"
        result["parser_confidence"] = "medium"
        return result

    # ── Pattern: "turn on the X and put the Y on it" ──
    m = re.search(r"turn on the (.+?) and put the (.+?) on", text)
    if m:
        result["mechanism_type"] = "articulated_object"
        result["gripper_duty_eligible"] = False
        result["parsed_object"] = m.group(2).strip()
        result["parsed_target"] = m.group(1).strip()
        result["target_type"] = "articulated_joint"
        result["parser_confidence"] = "high"
        return result

    # ── Pattern: "pick up the X and place it in the Y" (L10 variant) ──
    m = re.search(r"pick up the (.+?) and place it (?:in|on) the (.+)", text)
    if m:
        result["mechanism_type"] = "pick_place_transfer"
        result["gripper_duty_eligible"] = True
        result["parsed_object"] = m.group(1).strip()
        result["parsed_target"] = m.group(2).strip()
        result["target_type"] = "body_or_site"
        result["parser_confidence"] = "high"
        return result

    # ── Fallback ──
    result["unsupported_reason"] = f"no_matching_pattern: '{text[:80]}'"
    return result


def _parse_task_objects(task_instruction: str) -> tuple[str | None, str | None]:
    """Extract target object and receptacle from task instruction.
    Kept for backward compatibility. Prefer classify_mechanism() for new code.
    """
    result = classify_mechanism(task_instruction)
    return result["parsed_object"], result["parsed_target"]


def _find_obs_key(obs: dict[str, Any], object_name: str | None, suffix: str = "_pos") -> str | None:
    """Find obs key matching an object name with given suffix.

    Matches patterns like 'milk_1_pos', 'basket_1_pos'.
    """
    if not object_name:
        return None
    norm_target = _normalize_name(object_name)
    # Collect candidates
    candidates: list[tuple[str, int]] = []
    for key in obs:
        if not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]  # e.g. "milk_1"
        # Remove numeric suffix like _1, _2
        stem_normalized = _normalize_name(re.sub(r"_\d+$", "", stem))
        if stem_normalized == norm_target:
            # Exact match score
            candidates.append((key, 2))
        elif norm_target in stem_normalized or stem_normalized in norm_target:
            candidates.append((key, 1))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def _get_pose_from_obs(obs: dict[str, Any], object_name: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Get object position and quaternion from obs dict."""
    pos_key = _find_obs_key(obs, object_name, "_pos")
    quat_key = _find_obs_key(obs, object_name, "_quat")
    pos = np.asarray(obs[pos_key], dtype=np.float32) if pos_key else None
    quat = np.asarray(obs[quat_key], dtype=np.float32) if quat_key else None
    return pos, quat


def get_sim_body_names(env) -> list[str]:
    """Get all body names from MuJoCo sim."""
    try:
        return list(env.env.sim.model.body_names)
    except Exception:
        return []


def get_sim_site_names(env) -> list[str]:
    """Get all site names from MuJoCo sim."""
    try:
        return list(env.env.sim.model.site_names)
    except Exception:
        return []


def get_sim_geom_names(env) -> list[str]:
    """Get all geom names from MuJoCo sim."""
    try:
        return list(env.env.sim.model.geom_names)
    except Exception:
        return []


def dump_sim_names(env) -> dict[str, Any]:
    """Dump all MuJoCo body/site/geom names for debug."""
    return {
        "body_names": get_sim_body_names(env),
        "site_names": get_sim_site_names(env),
        "geom_names": get_sim_geom_names(env),
        "env_type": str(type(env)),
        "inner_env_type": str(type(env.env)) if hasattr(env, "env") else "N/A",
    }


def extract_teacher_privileged_state(
    env,
    obs: dict[str, Any],
    task_instruction: str,
) -> dict[str, Any]:
    """Extract teacher-privileged state from LIBERO env for a single step.

    Uses obs dict keys ({object}_1_pos, {object}_1_quat) as primary source.
    Falls back to MuJoCo sim body/site positions if obs keys not found.

    Returns dict with:
        teacher_privileged_state_available: bool
        object_pose_json: JSON array [x, y, z, qx, qy, qz, qw] or ""
        target_pose_json: JSON array [x, y, z] or ""
        object_to_target_distance: float or ""
        object_height_delta_from_start: float or ""
        object_eef_distance: float or ""
        privileged_state_error: str or ""
    """
    result: dict[str, Any] = {
        "teacher_privileged_state_available": False,
        "object_pose_json": "",
        "target_pose_json": "",
        "object_to_target_distance": "",
        "object_height_delta_from_start": "",
        "object_eef_distance": "",
        "privileged_state_error": "",
    }

    try:
        mech_info = classify_mechanism(task_instruction)
    except Exception as e:
        result["privileged_state_error"] = f"parse_task: {e}"
        return result

    # Handle articulated / unsupported / planar — no pick-place privileged state
    if mech_info["mechanism_type"] in ("articulated_object", "unsupported_or_low_signal",
                                        "planar_rearrangement_or_spatial_relation"):
        result["privileged_state_error"] = f"mechanism={mech_info['mechanism_type']} gripper_duty_eligible=false"
        return result

    # Get primary object/target from mechanism info
    target_obj = mech_info["parsed_object"]
    target_rec = mech_info["parsed_target"]
    segments = mech_info.get("segments", [])

    # For multi-object transfer, try each segment until one resolves
    if not target_obj and segments:
        for seg in segments:
            target_obj = seg.get("object")
            target_rec = seg.get("target")
            if target_obj and target_rec:
                break

    obj_pos = obj_quat = rec_pos = None
    errors: list[str] = []

    # 1. Try obs dict first (primary source)
    try:
        obj_pos, obj_quat = _get_pose_from_obs(obs, target_obj)
        rec_pos, _ = _get_pose_from_obs(obs, target_rec)
    except Exception as e:
        errors.append(f"obs_lookup: {e}")

    # 2. Fallback: try MuJoCo sim body/site names
    if obj_pos is None or rec_pos is None:
        try:
            sim = env.env.sim
            # Search body names for object match
            if obj_pos is None and target_obj:
                norm_obj = _normalize_name(target_obj)
                for i, name in enumerate(sim.model.body_names):
                    if norm_obj in _normalize_name(name):
                        obj_pos = sim.data.body_xpos[i].copy()
                        obj_quat = sim.data.body_xquat[i].copy()
                        break

            # Search body/site names for receptacle match
            if rec_pos is None and target_rec:
                norm_rec = _normalize_name(target_rec)
                for i, name in enumerate(sim.model.body_names):
                    if norm_rec in _normalize_name(name):
                        rec_pos = sim.data.body_xpos[i].copy()
                        break
                if rec_pos is None:
                    for i, name in enumerate(sim.model.site_names):
                        if norm_rec in _normalize_name(name) and "contain" not in name.lower():
                            rec_pos = sim.data.site_xpos[i].copy()
                            break
        except Exception as e:
            errors.append(f"sim_lookup: {e}")

    # 3. Build result
    if obj_pos is not None:
        result["teacher_privileged_state_available"] = True
        pose = obj_pos.tolist() if len(obj_pos) == 3 else obj_pos[:3].tolist()
        if obj_quat is not None:
            quat = obj_quat.tolist() if len(obj_quat) == 4 else obj_quat[:4].tolist()
            result["object_pose_json"] = json.dumps(pose + quat)
        else:
            result["object_pose_json"] = json.dumps(pose)

    if rec_pos is not None:
        result["target_pose_json"] = json.dumps(rec_pos[:3].tolist())
        result["teacher_privileged_state_available"] = True

    if obj_pos is not None and rec_pos is not None:
        dist = float(np.linalg.norm(obj_pos[:3] - rec_pos[:3]))
        result["object_to_target_distance"] = dist

    # 4. EEF distance
    try:
        eef_pos = obs.get("robot0_eef_pos")
        if eef_pos is not None and obj_pos is not None:
            result["object_eef_distance"] = float(np.linalg.norm(np.asarray(eef_pos)[:3] - obj_pos[:3]))
    except Exception:
        pass

    if errors:
        result["privileged_state_error"] = "; ".join(errors)
    elif obj_pos is None and rec_pos is None:
        result["privileged_state_error"] = (
            f"no_obs_or_sim_match: object='{target_obj}' receptacle='{target_rec}'"
        )

    return result


def build_sim_debug_metadata(env, task_instruction: str) -> dict[str, Any]:
    """Build debug metadata for sim_names.json."""
    target_obj, target_rec = _parse_task_objects(task_instruction)
    sim_names = dump_sim_names(env)

    selected_object_body = ""
    selected_target_body_or_site = ""
    selection_reason = ""
    unmatched_reason = ""

    if target_obj:
        norm_obj = _normalize_name(target_obj)
        for name in sim_names["body_names"]:
            if norm_obj in _normalize_name(name):
                selected_object_body = name
                selection_reason = f"body match: {name} for '{target_obj}'"
                break
        if not selected_object_body:
            unmatched_reason = f"no body match for object '{target_obj}'"

    if target_rec:
        norm_rec = _normalize_name(target_rec)
        for name in sim_names["body_names"]:
            if norm_rec in _normalize_name(name):
                selected_target_body_or_site = name
                selection_reason += f"; body match: {name} for '{target_rec}'"
                break
        if not selected_target_body_or_site:
            for name in sim_names["site_names"]:
                if norm_rec in _normalize_name(name):
                    selected_target_body_or_site = name
                    selection_reason += f"; site match: {name} for '{target_rec}'"
                    break
        if not selected_target_body_or_site:
            unmatched_reason += f"; no match for receptacle '{target_rec}'"

    return {
        "body_names": sim_names["body_names"],
        "site_names": sim_names["site_names"],
        "geom_names": sim_names["geom_names"][:50],
        "env_unwrap_chain": [sim_names["env_type"], sim_names["inner_env_type"]],
        "task_instruction": task_instruction,
        "parsed_target_object": target_obj,
        "parsed_target_receptacle": target_rec,
        "selected_object_body": selected_object_body,
        "selected_target_body_or_site": selected_target_body_or_site,
        "selection_reason": selection_reason,
        "unmatched_reason": unmatched_reason,
    }
