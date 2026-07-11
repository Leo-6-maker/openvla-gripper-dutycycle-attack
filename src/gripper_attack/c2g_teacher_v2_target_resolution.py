"""Pure structured target resolution for C2g Teacher-v2.

The resolver is deliberately simulator-free. It consumes already-parsed task/BDDL
metadata, applies operator-specific semantic roles, validates every resolved entity
against declarations, and fails closed on ambiguity.

Official LIBERO goals frequently refer to fully-qualified region/site names rather
than directly to the owning object or fixture. The resolver therefore keeps three
concepts separate:

* manipulated target entities;
* destination entities/sites;
* an interaction site whose owner is the manipulated fixture.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .c2g_semantic_aliases import normalize_goal_operator


@dataclass(frozen=True)
class TargetResolution:
    resolved_target_objects: tuple[str, ...]
    resolved_receptacles: tuple[str, ...]
    resolved_sites: tuple[str, ...]
    resolved_manipulable_entities: tuple[str, ...]
    ordered_subgoals: tuple[tuple[str, ...], ...]
    resolution_source: str
    resolution_confidence: float
    reason_code: str
    ambiguities: tuple[str, ...]
    unresolved_tokens: tuple[str, ...]
    # Backward-compatible extension fields. Existing positional construction remains
    # valid because these fields have defaults.
    resolved_destination_entities: tuple[str, ...] = ()
    goal_bindings: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("name", "id", "object", "site", "receptacle", "fixture", "entity"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


def _names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = [value]
    return tuple(sorted(dict.fromkeys(name for item in value for name in [_name(item)] if name)))


def _first_names(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    for key in keys:
        names = _names(metadata.get(key))
        if names:
            return names
    return ()


def _normalize_operator(value: Any) -> str:
    return normalize_goal_operator(value)


def _predicate_parts(predicate: Any) -> tuple[str, tuple[str, ...]]:
    if isinstance(predicate, (list, tuple)) and predicate:
        return _normalize_operator(predicate[0]), tuple(str(item) for item in predicate[1:])
    if not isinstance(predicate, Mapping):
        return "", ()
    operator = _normalize_operator(predicate.get("predicate", predicate.get("operator", predicate.get("name", ""))))
    args = predicate.get("args", predicate.get("arguments"))
    if args is None:
        args = [predicate.get("subject"), predicate.get("object", predicate.get("target"))]
    return operator, tuple(str(item) for item in args if item not in (None, ""))


def _flatten_predicates(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        nested = value.get("predicates", value.get("subgoals", value.get("goals")))
        if nested is not None:
            return _flatten_predicates(nested)
        return [value]
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str) and _normalize_operator(value[0]) not in {"and", "or", "ordered"}:
            return [value]
        out: list[Any] = []
        iterable = value[1:] if value and isinstance(value[0], str) else value
        for item in iterable:
            out.extend(_flatten_predicates(item))
        return out
    return []


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(value for value in values if value)))


def _resolve_declared(token: str, declarations: set[str]) -> tuple[str, ...]:
    return (token,) if token in declarations else ()


# Role names are internal and operator-specific. A destination may be a movable
# object (plate/basket), a declared receptacle, a site/region, or a fixture.
_OPERATOR_ROLES: dict[str, tuple[str, ...]] = {
    "in": ("object", "destination"),
    "inside": ("object", "destination"),
    "on": ("object", "destination"),
    "at": ("object", "destination"),
    "place": ("object", "destination"),
    "put": ("object", "destination"),
    "stack": ("object", "destination"),
    "contains": ("destination", "object"),
    "open": ("manipulable_or_site_owner",),
    "close": ("manipulable_or_site_owner",),
    "turn_on": ("manipulable_or_site_owner",),
    "turn_off": ("manipulable_or_site_owner",),
    "toggle": ("manipulable_or_site_owner",),
    "press": ("manipulable_or_site_owner",),
    "push": ("object_or_manipulable_or_site_owner",),
    "pull": ("object_or_manipulable_or_site_owner",),
    "slide": ("object_or_manipulable_or_site_owner",),
    "rotate": ("object_or_manipulable_or_site_owner",),
    "grasp": ("object_or_manipulable_or_site_owner",),
    "hold": ("object_or_manipulable_or_site_owner",),
    "lift": ("object_or_manipulable_or_site_owner",),
    "move": ("object_or_manipulable_or_site_owner", "destination"),
    "pour": ("object_or_manipulable_or_site_owner", "destination"),
}

_ARTICULATED_OPERATORS = {
    "open", "close", "turn_on", "turn_off", "toggle", "press", "pull", "rotate"
}


def _region_owner_map(metadata: Mapping[str, Any]) -> dict[str, str]:
    value = metadata.get("region_owner_by_site", {})
    if not isinstance(value, Mapping):
        return {}
    return {
        str(site).strip(): str(owner).strip()
        for site, owner in value.items()
        if str(site).strip() and str(owner).strip()
    }


def _structured_targets(
    predicates: Iterable[Any],
    *,
    objects: set[str],
    receptacles: set[str],
    sites: set[str],
    manipulable: set[str],
    region_owner_by_site: Mapping[str, str],
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...],
    tuple[tuple[str, ...], ...], tuple[str, ...], tuple[str, ...],
    tuple[str, ...], tuple[tuple[str, ...], ...],
]:
    target_objects: list[str] = []
    destination_entities: list[str] = []
    target_receptacles: list[str] = []
    target_sites: list[str] = []
    target_manipulable: list[str] = []
    subgoals: list[tuple[str, ...]] = []
    bindings: list[tuple[str, ...]] = []
    unresolved: list[str] = []
    unsupported_operators: list[str] = []

    all_destinations = objects | receptacles | sites | manipulable

    for predicate in predicates:
        operator, args = _predicate_parts(predicate)
        if not operator or not args:
            continue
        subgoals.append((operator, *args))
        roles = _OPERATOR_ROLES.get(operator)
        if roles is None:
            unsupported_operators.append(operator)
            continue

        resolved_by_index: dict[int, tuple[str, str]] = {}
        for index, role in enumerate(roles):
            if index >= len(args):
                unresolved.append(f"{operator}:missing_arg_{index}")
                continue
            token = args[index]
            resolved: tuple[str, ...] = ()
            resolved_entity = ""
            resolved_kind = ""

            if role == "object":
                resolved = _resolve_declared(token, objects)
                target_objects.extend(resolved)
                if resolved:
                    resolved_entity, resolved_kind = resolved[0], "object"
            elif role == "destination":
                resolved = _resolve_declared(token, all_destinations)
                if resolved:
                    destination_entities.extend(resolved)
                    if token in sites:
                        target_sites.append(token)
                        resolved_kind = "site"
                    else:
                        # Legacy field name: this collection is the destination side,
                        # including movable destination objects such as plate_1.
                        target_receptacles.append(token)
                        resolved_kind = "destination"
                    resolved_entity = token
            elif role == "manipulable_or_site_owner":
                if token in manipulable:
                    resolved = (token,)
                    target_manipulable.append(token)
                    resolved_entity, resolved_kind = token, "manipulable"
                elif token in sites:
                    owner = str(region_owner_by_site.get(token, ""))
                    if owner in manipulable:
                        resolved = (owner,)
                        target_sites.append(token)
                        target_manipulable.append(owner)
                        resolved_entity, resolved_kind = owner, "manipulable_site"
            elif role == "object_or_manipulable_or_site_owner":
                if token in objects:
                    resolved = (token,)
                    target_objects.append(token)
                    resolved_entity, resolved_kind = token, "object"
                elif token in manipulable:
                    resolved = (token,)
                    target_manipulable.append(token)
                    resolved_entity, resolved_kind = token, "manipulable"
                elif token in sites:
                    owner = str(region_owner_by_site.get(token, ""))
                    if owner in manipulable:
                        resolved = (owner,)
                        target_sites.append(token)
                        target_manipulable.append(owner)
                        resolved_entity, resolved_kind = owner, "manipulable_site"

            if resolved:
                resolved_by_index[index] = (resolved_entity, resolved_kind)
            else:
                unresolved.append(token)

        # Canonical binding tuple:
        # (operator, manipulated_target, destination, interaction_site)
        target_index = 1 if operator == "contains" else 0
        destination_index = 0 if operator == "contains" else (1 if len(roles) > 1 else -1)
        target_entity = resolved_by_index.get(target_index, ("", ""))[0]
        destination_entity = (
            resolved_by_index.get(destination_index, ("", ""))[0]
            if destination_index >= 0 else ""
        )
        interaction_site = ""
        if target_index < len(args) and args[target_index] in sites:
            interaction_site = args[target_index]
        if target_entity:
            bindings.append((operator, target_entity, destination_entity, interaction_site))

    return (
        _dedupe(target_objects), _dedupe(target_receptacles), _dedupe(target_sites),
        _dedupe(target_manipulable), tuple(subgoals), _dedupe(unresolved),
        _dedupe(unsupported_operators), _dedupe(destination_entities), tuple(bindings),
    )


def _mentioned(names: Iterable[str], language: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", " ", language.lower()).strip()
    found: list[str] = []
    for name in sorted(names):
        token = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        aliases = {token, re.sub(r"\s+\d+$", "", token)}
        if any(alias and re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in sorted(aliases)):
            found.append(name)
    return tuple(found)


def _validate_direct(names: tuple[str, ...], declarations: set[str], unresolved: list[str]) -> tuple[str, ...]:
    valid = tuple(name for name in names if name in declarations)
    unresolved.extend(name for name in names if name not in declarations)
    return _dedupe(valid)


def _ordered_from_metadata(metadata: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    ordered = metadata.get("ordered_subgoals")
    if not ordered:
        return ()
    return tuple(
        (operator, *args)
        for operator, args in (_predicate_parts(item) for item in _flatten_predicates(ordered))
        if operator and args
    )


def resolve_task_targets(metadata: Mapping[str, Any]) -> TargetResolution:
    """Resolve manipulated targets and destinations without arbitrary tie-breaking."""

    objects = set(_first_names(metadata, ("object_declarations", "objects")))
    receptacles = set(_first_names(metadata, ("receptacle_declarations", "receptacles")))
    sites = set(_first_names(metadata, ("site_declarations", "sites")))
    fixtures = set(_first_names(metadata, ("fixture_declarations", "fixtures", "manipulable_entities")))
    manipulable = fixtures | set(_first_names(metadata, ("manipulable_receptacles",)))
    region_owner_by_site = _region_owner_map(metadata)
    language = str(metadata.get("task_language", metadata.get("language", "")))
    ambiguities: list[str] = []
    unresolved = list(_names(metadata.get("unresolved_tokens")))

    # Preserve full predicate semantics before considering the collector's lossy
    # direct summary. This is essential for multi-target event binding.
    predicate_block = metadata.get("goal_predicates", metadata.get("bddl_goal_predicates"))
    predicates = _flatten_predicates(predicate_block)
    if predicates:
        (
            target_objects, target_receptacles, target_sites, target_manipulable,
            subgoals, missed, unsupported, destinations, bindings,
        ) = _structured_targets(
            predicates,
            objects=objects,
            receptacles=receptacles,
            sites=sites,
            manipulable=manipulable,
            region_owner_by_site=region_owner_by_site,
        )
        unresolved.extend(missed)
        if unsupported:
            ambiguities.append("UNSUPPORTED_OPERATORS:" + ",".join(unsupported))
        if target_objects or target_receptacles or target_sites or target_manipulable:
            language_entities = set(_mentioned(objects | receptacles | sites | manipulable, language))
            structured_entities = (
                set(target_objects) | set(target_receptacles) | set(target_sites)
                | set(target_manipulable) | set(destinations)
            )
            if language_entities and not language_entities.issubset(structured_entities):
                ambiguities.append("LANGUAGE_STRUCTURED_CONFLICT")
            ordered_subgoals = _ordered_from_metadata(metadata) or subgoals
            confidence = 1.0
            if ambiguities:
                confidence = min(confidence, 0.85)
            if unresolved:
                confidence = min(confidence, 0.75)
            reason = "RESOLVED_STRUCTURED"
            if unsupported:
                reason = "RESOLVED_STRUCTURED_WITH_UNSUPPORTED_OPERATORS"
            elif unresolved:
                reason = "RESOLVED_STRUCTURED_WITH_UNRESOLVED"
            elif ambiguities:
                reason = "RESOLVED_STRUCTURED_WITH_LANGUAGE_CONFLICT"
            return TargetResolution(
                target_objects,
                target_receptacles,
                target_sites,
                target_manipulable,
                ordered_subgoals,
                "structured_bddl_predicates",
                confidence,
                reason,
                _dedupe(ambiguities),
                _dedupe(unresolved),
                destinations,
                bindings,
            )

    block = metadata.get("structured_goal_metadata")
    if isinstance(block, Mapping):
        direct_objects = _validate_direct(_first_names(block, ("target_objects", "target_object")), objects, unresolved)
        direct_receptacles = _validate_direct(
            _first_names(block, ("target_receptacles", "target_receptacle")),
            objects | receptacles | manipulable,
            unresolved,
        )
        direct_sites = _validate_direct(_first_names(block, ("target_sites", "target_site")), sites, unresolved)
        direct_manipulable = _validate_direct(
            _first_names(block, ("target_fixtures", "target_fixture", "target_manipulable_entities")),
            manipulable,
            unresolved,
        )
        direct_destinations = _validate_direct(
            _first_names(block, ("target_destinations", "destination_entities")),
            objects | receptacles | sites | manipulable,
            unresolved,
        )
        if direct_objects or direct_receptacles or direct_sites or direct_manipulable:
            has_unresolved = bool(unresolved)
            return TargetResolution(
                direct_objects,
                direct_receptacles,
                direct_sites,
                direct_manipulable,
                _ordered_from_metadata(metadata),
                "structured_goal_metadata",
                0.75 if has_unresolved else 1.0,
                "RESOLVED_STRUCTURED_WITH_UNRESOLVED" if has_unresolved else "RESOLVED_STRUCTURED",
                (),
                _dedupe(unresolved),
                direct_destinations or _dedupe((*direct_receptacles, *direct_sites)),
                (),
            )

    explicit_objects = _validate_direct(
        _first_names(metadata, ("target_objects", "target_object", "valid_target_objects")), objects, unresolved
    )
    explicit_receptacles = _validate_direct(
        _first_names(metadata, ("target_receptacles", "target_receptacle")),
        objects | receptacles | manipulable,
        unresolved,
    )
    explicit_sites = _validate_direct(_first_names(metadata, ("target_sites", "target_site")), sites, unresolved)
    explicit_manipulable = _validate_direct(
        _first_names(metadata, ("target_fixtures", "target_fixture", "target_manipulable_entities")),
        manipulable,
        unresolved,
    )
    explicit_count = len(explicit_objects) + len(explicit_manipulable)
    if explicit_objects or explicit_receptacles or explicit_sites or explicit_manipulable:
        if explicit_count > 1 and not metadata.get("ordered_subgoals"):
            ambiguities.append("MULTIPLE_VALID_TARGET_ENTITIES")
        return TargetResolution(
            explicit_objects,
            explicit_receptacles,
            explicit_sites,
            explicit_manipulable,
            _ordered_from_metadata(metadata),
            "explicit_task_metadata",
            0.5 if ambiguities or unresolved else 0.8,
            "AMBIGUOUS_MULTIPLE_TARGETS" if ambiguities else (
                "RESOLVED_EXPLICIT_WITH_UNRESOLVED" if unresolved else "RESOLVED_EXPLICIT_METADATA"
            ),
            _dedupe(ambiguities),
            _dedupe(unresolved),
            _dedupe((*explicit_receptacles, *explicit_sites)),
            (),
        )

    all_entities = objects | receptacles | sites | manipulable
    mentioned = _mentioned(all_entities, language)
    if len(mentioned) == 1:
        entity = mentioned[0]
        return TargetResolution(
            (entity,) if entity in objects else (),
            (entity,) if entity in receptacles else (),
            (entity,) if entity in sites else (),
            (entity,) if entity in manipulable else (),
            (),
            "language_fallback",
            0.5,
            "RESOLVED_LANGUAGE_FALLBACK",
            (),
            _dedupe(unresolved),
            (),
            (),
        )
    if len(mentioned) > 1:
        ambiguities.append("LANGUAGE_MULTIPLE_TARGET_ENTITIES")
    return TargetResolution(
        (), (), (), (), (), "unresolved", 0.0,
        "AMBIGUOUS_LANGUAGE_TARGET" if ambiguities else "TARGET_METADATA_MISSING",
        _dedupe(ambiguities), _dedupe(unresolved), (), (),
    )
