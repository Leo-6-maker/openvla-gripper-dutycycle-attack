"""Structured, fail-closed mechanism routing for clean Detector-v2 labels."""
from __future__ import annotations

from typing import Any, Mapping

from .c2g_teacher_v2_target_resolution import TargetResolution, resolve_task_targets


MECHANISM_TYPES = (
    "pick_place_transfer",
    "multi_object_transfer",
    "articulated_object",
    "constrained_manipulation",
    "planar_or_rearrangement",
    "unsupported_or_unknown",
)
_ARTICULATED_OPERATORS = {
    "open",
    "close",
    "turn_on",
    "turn_off",
    "toggle",
    "press",
    "push",
    "pull",
    "slide",
    "rotate",
}
_TRANSFER_OPERATORS = {"in", "inside", "on", "at", "place", "put", "stack", "move", "pour", "contains"}
_CONSTRAINT_OPERATORS = {"grasp", "hold", "lift"}


def _fatal_resolution_ambiguity(resolution: TargetResolution) -> bool:
    """Return true only for ambiguities that invalidate structured event roles.

    Official LIBERO language often mentions contextual objects used to disambiguate an
    initial location (for example a bowl *between the plate and the ramekin*) even
    though only the bowl and destination plate appear in the goal predicate. A
    LANGUAGE_STRUCTURED_CONFLICT is therefore a grounding diagnostic, not a reason to
    discard an otherwise exact BDDL goal binding.
    """

    for value in resolution.ambiguities:
        token = str(value)
        if token == "LANGUAGE_STRUCTURED_CONFLICT":
            continue
        if token.startswith("UNSUPPORTED_OPERATORS:"):
            return True
        if token.startswith("AMBIGUOUS_"):
            return True
    return False


def infer_clean_mechanism_type(
    metadata: Mapping[str, Any],
    *,
    resolution: TargetResolution | None = None,
) -> str:
    """Infer a coarse mechanism only from structured clean task metadata.

    Explicit valid metadata wins. Otherwise the structured resolver output is used.
    Language-only guessing and task-index lookup tables are intentionally forbidden.
    """

    explicit = str(metadata.get("mechanism_type", "")).strip()
    if explicit:
        if explicit not in MECHANISM_TYPES:
            raise ValueError(f"unknown explicit mechanism_type: {explicit}")
        return explicit

    resolved = resolution or resolve_task_targets(metadata)
    if resolved.unresolved_tokens or _fatal_resolution_ambiguity(resolved):
        return "unsupported_or_unknown"

    subgoals = tuple(resolved.ordered_subgoals)
    operators = {str(item[0]).strip().lower() for item in subgoals if item}
    target_objects = tuple(resolved.resolved_target_objects)
    target_manipulable = tuple(resolved.resolved_manipulable_entities)
    target_destinations = tuple(resolved.resolved_destination_entities) or (
        tuple(resolved.resolved_receptacles) + tuple(resolved.resolved_sites)
    )

    if operators & _ARTICULATED_OPERATORS and target_manipulable:
        return "articulated_object"
    if target_manipulable:
        return "constrained_manipulation" if operators & _CONSTRAINT_OPERATORS else "articulated_object"
    if len(target_objects) > 1:
        if target_destinations and (not operators or operators & _TRANSFER_OPERATORS):
            return "multi_object_transfer"
        return "unsupported_or_unknown"
    if len(target_objects) == 1 and target_destinations and (not operators or operators & _TRANSFER_OPERATORS):
        return "pick_place_transfer"
    if len(target_objects) == 1 and operators & _CONSTRAINT_OPERATORS:
        return "constrained_manipulation"
    if len(target_objects) == 1 and operators & {"move", "push", "slide"}:
        return "planar_or_rearrangement"
    return "unsupported_or_unknown"
