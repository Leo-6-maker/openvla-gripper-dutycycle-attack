"""Per-episode contract checks for the R8R Clean2000 reuse audit."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from src.gripper_attack.c2g_clean_window_schema import CLEAN_TEACHER_SCHEMA_VERSION
from src.gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds, build_clean_teacher_episode,
)
from tools.multisuite_detector.c2g_r8r_common import read_json, read_jsonl, sha256_file

A_DIRECT = "A_DIRECT_REUSE"
B_AUGMENT = "B_OFFLINE_AUGMENTABLE"
C_LEGACY = "C_LEGACY_ONLY"
D_RECOLLECT = "D_RECOLLECT_REQUIRED"
CLASSIFICATIONS = (A_DIRECT, B_AUGMENT, C_LEGACY, D_RECOLLECT)
FORBIDDEN = (
    "attack_outcome", "attacked_rgb", "post_intervention", "counterfactual",
    "vis_success", "random_success", "task_failure_after", "qpos_delta_after",
    "open_count_after",
)
ERROR_TOKENS = ("error", "exception", "traceback", "cuda_oom", "egl_error")
LEGACY_FIELDS = (
    "teacher_phase", "teacher_label", "phase_label", "phase_idx", "phase_id",
    "corridor_label", "release_label", "event_role", "event_role_label",
    "primary_event", "primary_attackable", "supplementary_event",
    "label_known_mask", "teacher_known", "y_corridor", "y_release",
)
CONTACT_FIELDS = (
    "mujoco_contact_pairs", "contact_pairs", "contacts",
    "active_target_bilateral_contact", "bilateral_goal_targets",
    "per_target_contact_reason",
)
PROGRESS_FIELDS = (
    "object_relative_lift", "target_object_relative_lift", "relative_lift",
    "target_distance_decrease", "object_target_progress", "target_relative_progress",
    "lift_transport_or_constraint", "manipulation_progress_active",
    "constrained_manipulation_active", "fixture_joint_motion",
)
RELEASE_FIELDS = (
    "release_safe", "release_safe_evidence", "near_target", "target_near",
    "pre_release_near_target", "supported_at_target", "target_support_contact",
    "stable_target_support",
)
TARGET_FIELDS = (
    "resolved_target_objects", "structured_goal_metadata", "object_declarations",
    "objects", "target_objects", "goal_event_bindings", "active_target_entity",
)
LOGIT_FIELDS = (
    "clean_gripper_logits", "clean_action_token_logits", "gripper_token_logits",
    "action_token_logits", "clean_action_token_probabilities",
)


def _ordered(path: Path):
    rows = sorted(read_jsonl(path), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if len(steps) != len(set(steps)) or any(right != left + 1 for left, right in zip(steps, steps[1:])):
        raise ValueError("non-contiguous or duplicate steps")
    return rows


def _bad_keys(mapping):
    return sorted(str(key) for key in mapping if any(token in str(key).lower() for token in FORBIDDEN))


def _has_error(metadata, rows):
    for mapping in (metadata, *rows):
        if any(
            any(token in str(key).lower() for token in ERROR_TOKENS)
            and value not in (None, "", False, 0, [], {})
            for key, value in mapping.items()
        ):
            return True
        try:
            if int(mapping.get("step", 0)) < 0:
                return True
        except Exception:
            return True
    return False


def _finite_vector(value, length):
    try:
        return (
            isinstance(value, (list, tuple)) and len(value) == length
            and all(math.isfinite(float(item)) for item in value)
        )
    except Exception:
        return False


def _feature_names(metadata, rows):
    values = []
    for mapping in (metadata, rows[0] if rows else {}):
        for key in ("feature_names_25d", "features_25d_names", "sc5_v2_feature_names"):
            if isinstance(mapping.get(key), (list, tuple)):
                values.append(tuple(map(str, mapping[key])))
    return values[0] if values and len(set(values)) == 1 else None


def _policy_9d(rows):
    for row in rows:
        value = next((
            row[key] for key in ("clean_policy_intent_9d", "clean_policy_features", "policy_intent")
            if row.get(key) is not None
        ), None)
        if value is None and all(row.get(key) is not None for key in CLEAN_POLICY_FEATURE_NAMES):
            value = [row[key] for key in CLEAN_POLICY_FEATURE_NAMES]
        if not _finite_vector(value, len(CLEAN_POLICY_FEATURE_NAMES)):
            return False
    return bool(rows)


def _any_field(mappings, fields):
    return any(any(mapping.get(key) is not None for key in fields) for mapping in mappings)


def _legacy_fields(metadata, rows):
    keys = set(metadata)
    for row in rows:
        keys.update(row)
    return sorted(key for key in LEGACY_FIELDS if key in keys)


def teacher_v2_support(metadata, rows):
    mappings = [metadata, *rows]
    support = {
        "teacher_v2_schema_marker_present": any(
            mapping.get("teacher_schema_version") == CLEAN_TEACHER_SCHEMA_VERSION
            for mapping in mappings
        ),
        "teacher_v2_target_raw_present": _any_field([metadata], TARGET_FIELDS),
        "teacher_v2_contact_raw_present": _any_field(rows, CONTACT_FIELDS),
        "teacher_v2_progress_raw_present": _any_field(rows, PROGRESS_FIELDS),
        "teacher_v2_release_raw_present": _any_field(rows, RELEASE_FIELDS),
        "teacher_v2_command_semantics_present": bool(
            str(metadata.get("gripper_command_semantics", "")).strip()
        ) or _any_field(rows, (
            "clean_close_intent", "clean_gripper_close_intent",
            "clean_gripper_is_closed_command",
        )),
    }
    support["teacher_v2_raw_evidence_complete"] = all(
        support[key] for key in support if key != "teacher_v2_schema_marker_present"
    )
    return support


def _rgb_path(step_path: Path, value):
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (step_path.parent / path).resolve()


def audit_episode(view_row: Mapping[str, Any], view, registry: Mapping[str, Any]):
    metadata_path = Path(view_row["metadata_path"])
    step_path = Path(view_row["step_records_path"])
    metadata = read_json(metadata_path)
    result = {
        "suite": registry["suite"], "task_index": registry["task_index"],
        "state_id": registry["state_id"], "parent_key": registry["parent_key"],
        "cohort": registry["cohort"], "split": registry["split"],
        "source_view_name": view.name, "source_root": str(view.root),
        "source_class": view.source_class, "metadata_path": str(metadata_path),
        "step_records_path": str(step_path), "metadata_sha256": sha256_file(metadata_path),
        "step_records_sha256": sha256_file(step_path), "classification": D_RECOLLECT,
        "classification_reason": "UNASSESSED", "teacher_v2_rebuild_attempted": False,
        "teacher_v2_rebuild_success": False, "teacher_v2_rebuild_error": None,
    }
    try:
        rows = _ordered(step_path)
    except Exception as exc:
        result.update(
            n_steps=0, w16_window_count=0, rgb_count=0,
            classification_reason=f"STEP_TRACE_INVALID: {type(exc).__name__}: {exc}",
        )
        return result
    result.update(n_steps=len(rows), w16_window_count=max(0, len(rows) - 15))
    forbidden = sorted(set(_bad_keys(metadata) + sum((_bad_keys(row) for row in rows), [])))
    condition = str(metadata.get("condition", "")).strip().upper()
    result["forbidden_clean_boundary_keys"] = forbidden
    result["error_record_present"] = _has_error(metadata, rows)
    result["condition"] = condition
    result["clean_boundary_valid"] = bool(
        (condition == "CLEAN" or (not condition and view.clean_only)) and not forbidden
    )
    explicit_runtime = metadata.get("runtime_valid")
    result["runtime_valid"] = bool(
        (explicit_runtime is True or (explicit_runtime is None and view.runtime_valid_by_manifest))
        and not result["error_record_present"] and rows
    )
    language = str(metadata.get("task_language") or rows[0].get("task_language", "")).strip()
    result["task_language_present"] = bool(language)
    rgb_paths = [_rgb_path(step_path, row.get("rgb_path", "")) for row in rows]
    existing = [path for path in rgb_paths if str(path) and path.is_file()]
    result.update(
        rgb_count=len(existing), rgb_missing_count=len(rows) - len(existing),
        rgb_complete=bool(rows) and len(existing) == len(rows),
        rgb_bytes=sum(path.stat().st_size for path in existing),
    )
    names = _feature_names(metadata, rows)
    result["features_25d_shape_complete"] = bool(rows) and all(
        _finite_vector(row.get("features_25d"), len(SC5_V2_FEATURES)) for row in rows
    )
    result["feature_names_25d"] = list(names) if names else None
    result["features_25d_names_exact"] = names == tuple(SC5_V2_FEATURES)
    result["feature_25d_order_bound_by_manifest"] = view.feature_25d_order_bound
    result["canonical_25d_complete"] = bool(
        result["features_25d_shape_complete"]
        and (result["features_25d_names_exact"] or view.feature_25d_order_bound)
    )
    result["policy_intent_9d_complete"] = _policy_9d(rows)
    result["raw_policy_logits_complete"] = bool(rows) and all(
        any(row.get(key) is not None for key in LOGIT_FIELDS) for row in rows
    )
    result["model_provenance_bound"] = view.model_provenance_bound
    result["processor_provenance_bound"] = view.processor_provenance_bound
    result["derived_feature_reconstruction_possible"] = bool(
        result["rgb_complete"] and result["task_language_present"]
        and view.model_provenance_bound and view.processor_provenance_bound
    )
    legacy = _legacy_fields(metadata, rows)
    result["legacy_label_fields_present"] = legacy
    result["teacher_v1_label_present"] = bool(legacy)
    result.update(teacher_v2_support(metadata, rows))
    if result["teacher_v2_raw_evidence_complete"]:
        result["teacher_v2_rebuild_attempted"] = True
        try:
            labels = build_clean_teacher_episode(rows, metadata, thresholds=CleanTeacherThresholds())
            if len(labels) != len(rows):
                raise RuntimeError("Teacher-v2 row count mismatch")
            known = [bool(item["label_known_mask"]) for item in labels]
            positive = [
                bool(item["y_gripper_critical_window"]) if known[index] else False
                for index, item in enumerate(labels)
            ]
            result.update(
                teacher_v2_rebuild_success=True,
                known_positive_steps=sum(k and p for k, p in zip(known, positive)),
                known_negative_steps=sum(k and not p for k, p in zip(known, positive)),
                unknown_steps=sum(not k for k in known),
                positive_episode=any(k and p for k, p in zip(known, positive)),
                fully_known_negative_episode=all(known) and not any(positive),
                triggerable_positive_episode=any(
                    all(positive[start:start + 10]) for start in range(max(0, len(positive) - 9))
                ),
            )
        except Exception as exc:
            result["teacher_v2_rebuild_error"] = f"{type(exc).__name__}: {exc}"
    defaults = {
        "known_positive_steps": 0, "known_negative_steps": 0,
        "unknown_steps": len(rows), "positive_episode": False,
        "fully_known_negative_episode": False, "triggerable_positive_episode": False,
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    result["clean_success_observed"] = bool(metadata.get(
        "clean_success_observed", metadata.get("clean_success", metadata.get("success", False))
    ))
    checks = {
        "clean_boundary": result["clean_boundary_valid"], "runtime": result["runtime_valid"],
        "trajectory_length": len(rows) >= 16, "rgb": result["rgb_complete"],
        "language": result["task_language_present"],
        "canonical_25d": result["canonical_25d_complete"],
        "model_provenance": view.model_provenance_bound,
        "processor_provenance": view.processor_provenance_bound,
    }
    failures = [key for key, passed in checks.items() if not passed]
    result["base_contract_checks"] = checks
    result["base_contract_valid"] = not failures
    if failures:
        result.update(
            classification=D_RECOLLECT,
            classification_reason="BASE_CONTRACT_FAILURE: " + ",".join(failures),
        )
    elif result["teacher_v2_raw_evidence_complete"] and result["teacher_v2_rebuild_success"]:
        if result["policy_intent_9d_complete"]:
            result.update(classification=A_DIRECT, classification_reason="CURRENT_SOURCE_CONTRACT_COMPLETE")
        elif result["derived_feature_reconstruction_possible"]:
            result.update(
                classification=B_AUGMENT,
                classification_reason="MISSING_ONLY_DETERMINISTIC_DERIVED_FEATURES",
            )
        else:
            result.update(
                classification=D_RECOLLECT,
                classification_reason="DERIVED_FEATURE_PROVENANCE_MISSING",
            )
    elif result["teacher_v1_label_present"]:
        result.update(
            classification=C_LEGACY,
            classification_reason="CURRENT_TEACHER_V2_RAW_EVIDENCE_INCOMPLETE",
        )
    else:
        result.update(classification=D_RECOLLECT, classification_reason="NO_CURRENT_OR_LEGACY_SUPERVISION")
    result["legacy_semantic_salvage_candidate"] = result["classification"] == C_LEGACY
    return result
