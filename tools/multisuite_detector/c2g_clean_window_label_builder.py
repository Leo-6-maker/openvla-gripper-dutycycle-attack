"""Pure-CPU clean-only Teacher-v2 label builder for C2g Detector v2.

The builder reuses the repository's structured task-target resolver and MuJoCo
contact canonicalizer. It deliberately fails closed when command polarity,
target identity, contact evidence, progress, or release semantics are unresolved.
No attacked rollout or post-intervention field is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.gripper_attack.c2g_clean_window_schema import (
    CLEAN_TEACHER_SCHEMA_VERSION,
    validate_clean_teacher_row,
)
from src.gripper_attack.c2g_teacher_v2_contact_identity import analyze_contact_pairs
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets


@dataclass(frozen=True)
class CleanTeacherThresholds:
    burst_length: int = 10
    contact_persistence_steps: int = 2
    relative_lift_threshold: float = 0.015
    target_progress_threshold: float = 0.01
    grounding_confidence_threshold: float = 0.5
    command_threshold: float = 0.0

    def validate(self) -> None:
        if self.burst_length <= 0:
            raise ValueError("burst_length must be positive")
        if self.contact_persistence_steps <= 0:
            raise ValueError("contact_persistence_steps must be positive")
        for name in (
            "relative_lift_threshold",
            "target_progress_threshold",
            "grounding_confidence_threshold",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.grounding_confidence_threshold <= 1.0:
            raise ValueError("grounding_confidence_threshold must be in [0,1]")


_FORBIDDEN_INPUT_TOKENS = (
    "attack_outcome",
    "post_intervention",
    "counterfactual",
    "vis_success",
    "random_success",
    "qpos_delta_after",
    "open_count_after",
)


def _reject_attacked_fields(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    bad = sorted(
        {
            key
            for source in (metadata, *rows)
            for key in source
            if any(token in key.lower() for token in _FORBIDDEN_INPUT_TOKENS)
        }
    )
    if bad:
        raise ValueError("clean teacher input contains attacked/post-intervention fields: " + ", ".join(bad))


def _first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> tuple[Any, bool]:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name], True
    return None, False


def _explicit_bool(mapping: Mapping[str, Any], names: Iterable[str]) -> tuple[bool, bool]:
    names = tuple(names)
    value, present = _first_present(mapping, names)
    if not present:
        return False, False
    if type(value) is bool:
        return value, True
    if value in (0, 1):
        return bool(value), True
    raise ValueError(f"expected explicit boolean for one of {names}")


def _finite_float(mapping: Mapping[str, Any], names: Iterable[str]) -> tuple[float, bool]:
    names = tuple(names)
    value, present = _first_present(mapping, names)
    if not present:
        return 0.0, False
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"expected finite float for one of {names}")
    return value, True


def _clean_close_intent(
    row: Mapping[str, Any], metadata: Mapping[str, Any], threshold: float
) -> tuple[bool, bool]:
    explicit, known = _explicit_bool(
        row,
        (
            "clean_close_intent",
            "clean_gripper_close_intent",
            "clean_gripper_is_closed_command",
        ),
    )
    if known:
        return explicit, True
    command, present = _finite_float(
        row,
        ("clean_gripper_command", "gripper_command", "raw_gripper_command"),
    )
    if not present:
        return False, False
    semantics = str(metadata.get("gripper_command_semantics", "")).strip().lower()
    if semantics == "positive_is_close":
        return command > threshold, True
    if semantics == "negative_is_close":
        return command < -threshold, True
    return False, False


def _release_safe(row: Mapping[str, Any]) -> tuple[bool, bool]:
    explicit, known = _explicit_bool(row, ("release_safe", "release_safe_evidence"))
    if known:
        return explicit, True
    near_target, near_known = _explicit_bool(
        row, ("near_target", "target_near", "pre_release_near_target")
    )
    supported, support_known = _explicit_bool(
        row,
        ("supported_at_target", "target_support_contact", "stable_target_support"),
    )
    if near_known and not near_target:
        return False, True
    if near_known and support_known:
        return bool(near_target and supported), True
    return False, False


def _progress_signal(
    row: Mapping[str, Any],
    *,
    thresholds: CleanTeacherThresholds,
) -> tuple[bool, bool, float | None, str]:
    explicit, known = _explicit_bool(
        row,
        (
            "lift_transport_or_constraint",
            "manipulation_progress_active",
            "constrained_manipulation_active",
        ),
    )
    if known:
        constrained, _ = _explicit_bool(row, ("constrained_manipulation_active",))
        return explicit, True, None, "constraint" if constrained else "explicit"

    relative_lift, lift_known = _finite_float(
        row,
        ("object_relative_lift", "target_object_relative_lift", "relative_lift"),
    )
    if lift_known:
        return (
            relative_lift >= thresholds.relative_lift_threshold,
            True,
            relative_lift,
            "relative_lift",
        )

    progress_delta, progress_known = _finite_float(
        row,
        ("target_distance_decrease", "object_target_progress", "target_relative_progress"),
    )
    if progress_known:
        return (
            progress_delta >= thresholds.target_progress_threshold,
            True,
            None,
            "target_progress",
        )

    # Absolute EEF-z is intentionally not accepted as evidence.
    return False, False, None, "unresolved"


def _mechanism_eligible(mechanism_type: str) -> bool:
    return mechanism_type in {
        "pick_place_transfer",
        "multi_object_transfer",
        "articulated_object",
        "constrained_manipulation",
    }


def _as_contact_pairs(row: Mapping[str, Any]) -> tuple[list[Any], bool]:
    value, present = _first_present(row, ("mujoco_contact_pairs", "contact_pairs", "contacts"))
    if not present:
        return [], False
    if not isinstance(value, (list, tuple)):
        raise ValueError("contact pairs must be a list or tuple")
    return list(value), True


def _contact_evidence(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[set[str], float, bool, bool, str]:
    explicit_contact, explicit_known = _explicit_bool(
        row,
        ("contact_or_grasp_stable", "contact_stable", "grasp_stable"),
    )
    pairs, pairs_present = _as_contact_pairs(row)
    if not pairs_present:
        return (
            set(),
            1.0 if explicit_known else 0.0,
            explicit_contact,
            explicit_known,
            "EXPLICIT" if explicit_known else "UNRESOLVED",
        )

    identity = analyze_contact_pairs(
        pairs,
        object_names=metadata.get("object_declarations", metadata.get("objects", ())),
        receptacle_names=metadata.get(
            "receptacle_declarations", metadata.get("receptacles", ())
        ),
        manipulable_receptacle_names=metadata.get("manipulable_receptacles", ()),
        fixture_names=metadata.get("fixture_declarations", metadata.get("fixtures", ())),
        static_names=metadata.get("static_names", ()),
        finger_aliases=metadata.get("finger_aliases"),
    )
    contacted = set(identity.contacted_objects) | set(identity.contacted_manipulable_entities)
    stable = explicit_contact if explicit_known else bool(identity.bilateral_grasp_candidate)
    confidence = (
        1.0
        if explicit_known or identity.ambiguity_reason == "NO_OBJECT_CONTACT"
        else float(identity.contacted_object_confidence)
    )
    known = explicit_known or identity.ambiguity_reason in {
        "",
        "NO_OBJECT_CONTACT",
        "UNILATERAL_OBJECT_CONTACT",
    }
    return contacted, confidence, stable, known, identity.ambiguity_reason or "RESOLVED"


def _phase_and_reason(
    *,
    mechanism_eligible: bool,
    targets_resolved: bool,
    contact_known: bool,
    close_known: bool,
    progress_known: bool,
    release_known: bool,
    target_relevant: bool,
    contacted_entities: set[str],
    contact_stable: bool,
    close_intent: bool,
    progress: bool,
    progress_source: str,
    release_safe: bool,
    critical: bool,
) -> tuple[str, str]:
    if not mechanism_eligible:
        return "UNSUPPORTED_MECHANISM", "UNSUPPORTED_MECHANISM"
    if not targets_resolved:
        return "TARGET_UNRESOLVED", "TARGET_UNRESOLVED"
    if not contact_known:
        return "CONTACT_UNRESOLVED", "CONTACT_UNRESOLVED"
    if not close_known:
        return "OTHER", "CLOSE_SEMANTICS_UNRESOLVED"
    if not progress_known:
        return "OTHER", "PROGRESS_SEMANTICS_UNRESOLVED"
    if not release_known:
        return "OTHER", "RELEASE_SEMANTICS_UNRESOLVED"
    if not target_relevant and contacted_entities:
        return "DISTRACTOR_CONTACT", "DISTRACTOR_CONTACT"
    if not contacted_entities:
        return "APPROACH", "APPROACH_NO_CONTACT"
    if target_relevant and not contact_stable:
        return "TARGET_CONTACT", "TARGET_CONTACT_NO_STABLE_GRASP"
    if release_safe:
        return "RELEASE_SAFE", "TARGET_RELEASE_SAFE"
    if critical:
        if progress_source == "constraint":
            return "CONSTRAINED_MANIPULATION", "TARGET_CONSTRAINED_MANIPULATION"
        if progress_source == "relative_lift":
            return "LIFT_ONSET", "TARGET_LIFT_ACTIVE"
        return "TRANSPORT", "TARGET_CRITICAL_WINDOW"
    if target_relevant and contact_stable and close_intent and progress:
        return "PRE_RELEASE", "TARGET_PRE_RELEASE"
    if target_relevant and contact_stable:
        return "STABLE_GRASP", "TARGET_STABLE_GRASP"
    if target_relevant:
        return "TARGET_CONTACT", "TARGET_CONTACT_NO_STABLE_GRASP"
    return "OTHER", "APPROACH_NO_CONTACT"


def _mark_contact_persistence(rows: list[dict[str, Any]], minimum: int) -> None:
    run_entity: tuple[str, ...] = ()
    run_indices: list[int] = []
    previous_step: int | None = None
    for index, row in enumerate(rows):
        entities = tuple(row["contacted_entities"])
        contiguous = previous_step is not None and int(row["step"]) == previous_step + 1
        if entities and entities == run_entity and contiguous:
            run_indices.append(index)
        else:
            if len(run_indices) >= minimum:
                for item in run_indices:
                    rows[item]["_contact_persistent"] = True
            run_entity = entities
            run_indices = [index] if entities else []
        previous_step = int(row["step"])
    if len(run_indices) >= minimum:
        for item in run_indices:
            rows[item]["_contact_persistent"] = True


def _mark_burst_targets(rows: list[dict[str, Any]], burst_length: int) -> None:
    for row in rows:
        if not row["label_known_mask"]:
            continue
        row["y_burst_feasible"] = False
        row["y_attack_start_b"] = False

    index = 0
    while index < len(rows):
        if not rows[index]["label_known_mask"] or not rows[index]["y_gripper_critical_window"]:
            index += 1
            continue
        interval = [index]
        while (
            interval[-1] + 1 < len(rows)
            and rows[interval[-1] + 1]["step"] == rows[interval[-1]]["step"] + 1
            and rows[interval[-1] + 1]["label_known_mask"]
            and rows[interval[-1] + 1]["y_gripper_critical_window"]
        ):
            interval.append(interval[-1] + 1)
        if len(interval) >= burst_length:
            feasible = interval[: len(interval) - burst_length + 1]
            for item in feasible:
                rows[item]["y_burst_feasible"] = True
            start = feasible[0]
            rows[start]["y_attack_start_b"] = True
            rows[start]["teacher_reason_code"] = "TARGET_CRITICAL_WINDOW_START"
        index = interval[-1] + 1


def build_clean_teacher_episode(
    step_rows: Sequence[Mapping[str, Any]],
    episode_metadata: Mapping[str, Any],
    *,
    thresholds: CleanTeacherThresholds = CleanTeacherThresholds(),
) -> list[dict[str, Any]]:
    """Build validated clean-only Teacher-v2 rows for one episode."""

    thresholds.validate()
    if not step_rows:
        raise ValueError("step_rows cannot be empty")
    _reject_attacked_fields(step_rows, episode_metadata)
    ordered = sorted((dict(row) for row in step_rows), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if any(step < 0 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("steps must be unique non-negative integers")

    resolution = resolve_task_targets(episode_metadata)
    targets = set(resolution.resolved_target_objects)
    target_manipulable = set(resolution.resolved_manipulable_entities)
    target_entities = targets | target_manipulable
    targets_resolved = bool(target_entities) and (
        resolution.resolution_confidence >= thresholds.grounding_confidence_threshold
    )

    mechanism_type = str(episode_metadata.get("mechanism_type", "unsupported_or_unknown"))
    eligible = _mechanism_eligible(mechanism_type)
    suite = str(episode_metadata.get("suite", ""))
    task_index = int(episode_metadata.get("task_index", -1))
    episode_key = str(episode_metadata.get("episode_key", ""))
    if not suite or task_index < 0 or not episode_key:
        raise ValueError("episode_metadata requires episode_key, suite, and non-negative task_index")

    provisional: list[dict[str, Any]] = []
    signal_cache: list[dict[str, Any]] = []
    for row in ordered:
        contacted, contact_confidence, stable, contact_known, contact_reason = (
            _contact_evidence(row, episode_metadata)
        )
        close_intent, close_known = _clean_close_intent(
            row, episode_metadata, thresholds.command_threshold
        )
        progress, progress_known, relative_lift, progress_source = _progress_signal(
            row, thresholds=thresholds
        )
        release_safe, release_known = _release_safe(row)
        target_relevant = bool(target_entities & contacted) if targets_resolved and contact_known else False
        target_relevant_known = targets_resolved and contact_known
        signal_cache.append(
            {
                "contacted": contacted,
                "contact_confidence": contact_confidence,
                "stable": stable,
                "contact_known": contact_known,
                "contact_reason": contact_reason,
                "close_intent": close_intent,
                "close_known": close_known,
                "progress": progress,
                "progress_known": progress_known,
                "relative_lift": relative_lift,
                "progress_source": progress_source,
                "release_safe": release_safe,
                "release_known": release_known,
                "target_relevant": target_relevant,
                "target_relevant_known": target_relevant_known,
            }
        )
        provisional.append(
            {
                "step": int(row["step"]),
                "contacted_entities": sorted(contacted),
                "_contact_persistent": False,
            }
        )

    _mark_contact_persistence(provisional, thresholds.contact_persistence_steps)

    output: list[dict[str, Any]] = []
    for row_index, (provisional_row, signals) in enumerate(zip(provisional, signal_cache)):
        contact_stable = bool(signals["stable"] or provisional_row["_contact_persistent"])
        gripper_dependency, dependency_known = _explicit_bool(
            ordered[row_index],
            ("gripper_dependency", "gripper_dependent_manipulation"),
        )
        if not dependency_known:
            gripper_dependency = bool(signals["target_relevant"] and contact_stable)
            dependency_known = bool(
                signals["target_relevant_known"] and signals["contact_known"]
            )

        known = bool(
            eligible
            and targets_resolved
            and signals["contact_known"]
            and signals["target_relevant_known"]
            and signals["close_known"]
            and signals["progress_known"]
            and signals["release_known"]
            and dependency_known
        )
        if known:
            target_relevant = bool(signals["target_relevant"])
            close_intent = bool(signals["close_intent"])
            progress = bool(signals["progress"])
            release_safe = bool(signals["release_safe"])
            critical = bool(
                target_relevant
                and gripper_dependency
                and close_intent
                and progress
                and not release_safe
            )
            labels: Dict[str, Any] = {
                "y_target_relevant": target_relevant,
                "y_contact_or_grasp_stable": contact_stable,
                "y_gripper_dependency": bool(gripper_dependency),
                "y_clean_close_intent": close_intent,
                "y_lift_transport_or_constraint": progress,
                "y_release_safe": release_safe,
                "y_gripper_critical_window": critical,
                "y_burst_feasible": False,
                "y_attack_start_b": False,
            }
        else:
            critical = False
            labels = {
                "y_target_relevant": None,
                "y_contact_or_grasp_stable": None,
                "y_gripper_dependency": None,
                "y_clean_close_intent": None,
                "y_lift_transport_or_constraint": None,
                "y_release_safe": None,
                "y_gripper_critical_window": None,
                "y_burst_feasible": None,
                "y_attack_start_b": None,
            }

        phase, reason = _phase_and_reason(
            mechanism_eligible=eligible,
            targets_resolved=targets_resolved,
            contact_known=signals["contact_known"],
            close_known=signals["close_known"],
            progress_known=signals["progress_known"],
            release_known=signals["release_known"],
            target_relevant=bool(signals["target_relevant"]),
            contacted_entities=set(signals["contacted"]),
            contact_stable=contact_stable,
            close_intent=bool(signals["close_intent"]),
            progress=bool(signals["progress"]),
            progress_source=str(signals["progress_source"]),
            release_safe=bool(signals["release_safe"]),
            critical=critical,
        )
        confidence_parts = [
            float(resolution.resolution_confidence),
            float(signals["contact_confidence"]),
        ]
        if not known:
            confidence_parts.append(0.0)
        built = {
            "teacher_schema_version": CLEAN_TEACHER_SCHEMA_VERSION,
            "episode_key": episode_key,
            "step": provisional_row["step"],
            "suite": suite,
            "task_index": task_index,
            "mechanism_type": mechanism_type,
            "mechanism_eligible": eligible,
            "teacher_phase": phase,
            "teacher_reason_code": reason,
            "teacher_confidence": max(0.0, min(confidence_parts)),
            "grounding_confidence": float(resolution.resolution_confidence),
            "teacher_known": known,
            "label_known_mask": known,
            "resolved_target_objects": sorted(targets),
            "resolved_target_manipulable_entities": sorted(target_manipulable),
            "contacted_entities": sorted(signals["contacted"]),
            "uses_privileged_sim_state": True,
            "uses_attack_outcome": False,
            "uses_future_student_input": False,
            "object_relative_lift": signals["relative_lift"],
            "contact_identity_reason": signals["contact_reason"],
            "progress_source": signals["progress_source"],
            **labels,
        }
        output.append(built)

    _mark_burst_targets(output, thresholds.burst_length)
    for row in output:
        validate_clean_teacher_row(row)
    return output
