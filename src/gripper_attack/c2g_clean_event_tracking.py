"""Clean-only per-step goal-event binding for Detector-v2 Teacher evidence.

This module converts structured goal bindings into a causal *active target* identity
using only the current clean MuJoCo contact buffer. It never reads attack outcomes,
future student inputs, task indices, or language heuristics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .c2g_teacher_v2_contact_identity import analyze_contact_pairs
from .c2g_teacher_v2_target_resolution import TargetResolution


@dataclass(frozen=True)
class GoalEventBinding:
    subgoal_index: int
    operator: str
    target_entity: str
    destination_entity: str = ""
    interaction_site: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def goal_event_bindings(resolution: TargetResolution) -> tuple[GoalEventBinding, ...]:
    """Return deterministic goal bindings emitted by the structured resolver."""

    output: list[GoalEventBinding] = []
    for index, raw in enumerate(resolution.goal_bindings):
        values = tuple(str(value).strip() for value in raw)
        if len(values) != 4:
            raise ValueError(f"goal binding {index} must have four fields")
        operator, target, destination, interaction_site = values
        if not operator or not target:
            raise ValueError(f"goal binding {index} lacks operator/target")
        output.append(
            GoalEventBinding(
                subgoal_index=index,
                operator=operator,
                target_entity=target,
                destination_entity=destination,
                interaction_site=interaction_site,
            )
        )
    if not output:
        # Fallback for older direct structured metadata. Only accept the simple,
        # unambiguous one-target case; do not invent multi-target ordering.
        targets = tuple(resolution.resolved_target_objects) + tuple(
            resolution.resolved_manipulable_entities
        )
        destinations = tuple(resolution.resolved_destination_entities) or (
            tuple(resolution.resolved_receptacles) + tuple(resolution.resolved_sites)
        )
        if len(targets) == 1:
            operator = (
                str(resolution.ordered_subgoals[0][0])
                if resolution.ordered_subgoals else "unspecified"
            )
            output.append(
                GoalEventBinding(
                    subgoal_index=0,
                    operator=operator,
                    target_entity=targets[0],
                    destination_entity=destinations[0] if len(destinations) == 1 else "",
                    interaction_site=(
                        resolution.resolved_sites[0]
                        if resolution.resolved_manipulable_entities
                        and len(resolution.resolved_sites) == 1
                        else ""
                    ),
                )
            )
    return tuple(output)


def _target_contact_state(
    contact_pairs: Iterable[Any],
    *,
    target: str,
    is_manipulable: bool,
    finger_aliases: Mapping[str, str] | None = None,
) -> tuple[bool, bool, str]:
    identity = analyze_contact_pairs(
        contact_pairs,
        object_names=() if is_manipulable else (target,),
        fixture_names=(target,) if is_manipulable else (),
        finger_aliases=finger_aliases,
    )
    contacted = target in set(identity.contacted_objects) | set(
        identity.contacted_manipulable_entities
    )
    return contacted, bool(identity.bilateral_grasp_candidate), identity.ambiguity_reason


def select_active_goal_event(
    contact_pairs: Iterable[Any],
    bindings: Sequence[GoalEventBinding],
    *,
    manipulable_targets: Sequence[str] = (),
    finger_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select one active goal event from current finger-target contacts.

    Exactly one contacted goal target is sufficient to identify the event. Bilateral
    contact is reported separately and remains the stable-grasp criterion. Multiple
    simultaneously contacted targets or one target mapped to multiple incompatible
    subgoals remain unknown rather than being converted to a negative label.
    """

    manipulable = {str(value) for value in manipulable_targets}
    by_target: dict[str, list[GoalEventBinding]] = {}
    for binding in bindings:
        by_target.setdefault(binding.target_entity, []).append(binding)

    contacted: list[str] = []
    bilateral: list[str] = []
    reasons: dict[str, str] = {}
    pairs = list(contact_pairs)
    for target in sorted(by_target):
        is_contacted, is_bilateral, reason = _target_contact_state(
            pairs,
            target=target,
            is_manipulable=target in manipulable,
            finger_aliases=finger_aliases,
        )
        reasons[target] = reason
        if is_contacted:
            contacted.append(target)
        if is_bilateral:
            bilateral.append(target)

    if len(contacted) > 1:
        return {
            "active_target_known": False,
            "active_target_entity": None,
            "active_subgoal_index": None,
            "active_operator": None,
            "active_destination_entity": None,
            "active_interaction_site": None,
            "active_target_bilateral_contact": False,
            "active_target_reason": "MULTIPLE_CONTACTED_GOAL_TARGETS",
            "contacted_goal_targets": contacted,
            "bilateral_goal_targets": bilateral,
            "per_target_contact_reason": reasons,
        }
    if not contacted:
        return {
            "active_target_known": False,
            "active_target_entity": None,
            "active_subgoal_index": None,
            "active_operator": None,
            "active_destination_entity": None,
            "active_interaction_site": None,
            "active_target_bilateral_contact": False,
            "active_target_reason": "NO_GOAL_TARGET_CONTACT",
            "contacted_goal_targets": [],
            "bilateral_goal_targets": [],
            "per_target_contact_reason": reasons,
        }

    target = contacted[0]
    candidates = by_target[target]
    signatures = {
        (item.operator, item.destination_entity, item.interaction_site)
        for item in candidates
    }
    if len(signatures) != 1:
        return {
            "active_target_known": False,
            "active_target_entity": target,
            "active_subgoal_index": None,
            "active_operator": None,
            "active_destination_entity": None,
            "active_interaction_site": None,
            "active_target_bilateral_contact": target in bilateral,
            "active_target_reason": "TARGET_HAS_MULTIPLE_INCOMPATIBLE_SUBGOALS",
            "contacted_goal_targets": contacted,
            "bilateral_goal_targets": bilateral,
            "per_target_contact_reason": reasons,
        }

    binding = candidates[0]
    return {
        "active_target_known": True,
        "active_target_entity": target,
        "active_subgoal_index": int(binding.subgoal_index),
        "active_operator": binding.operator,
        "active_destination_entity": binding.destination_entity or None,
        "active_interaction_site": binding.interaction_site or None,
        "active_target_bilateral_contact": target in bilateral,
        "active_target_reason": "RESOLVED_CURRENT_TARGET_CONTACT",
        "contacted_goal_targets": contacted,
        "bilateral_goal_targets": bilateral,
        "per_target_contact_reason": reasons,
    }


def joint_hint_from_interaction_site(target_entity: str, interaction_site: str) -> str:
    """Extract a conservative joint selector such as ``middle`` from a region."""

    target = str(target_entity).strip()
    site = str(interaction_site).strip()
    if not target or not site:
        return ""
    prefix = target + "_"
    local = site[len(prefix):] if site.startswith(prefix) else site
    for suffix in ("_region", "_site", "_handle"):
        if local.endswith(suffix):
            local = local[: -len(suffix)]
            break
    return local.strip("_")
