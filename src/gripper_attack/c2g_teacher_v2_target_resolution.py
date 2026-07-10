"""Pure structured target resolution for C2g Teacher-v2.

The resolver is deliberately simulator-free. It consumes already-parsed task/BDDL
metadata, applies operator-specific semantic roles, validates every resolved entity
against declarations, and fails closed on ambiguity.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


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
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


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


_OPERATOR_ROLES: dict[str, tuple[str, ...]] = {
    "in": ("object", "receptacle"),
    "inside": ("object", "receptacle"),
    "on": ("object", "receptacle_or_site"),
    "at": ("object", "site_or_receptacle"),
    "place": ("object", "receptacle_or_site"),
    "put": ("object", "receptacle_or_site"),
    "stack": ("object", "object_or_receptacle"),
    "contains": ("receptacle", "object"),
    "open": ("manipulable",),
    "close": ("manipulable",),
    "turn_on": ("manipulable",),
    "turn_off": ("manipulable",),
    "toggle": ("manipulable",),
    "press": ("manipulable",),
    "push": ("manipulable",),
    "pull": ("manipulable",),
    "slide": ("manipulable",),
    "rotate": ("manipulable",),
    "grasp": ("object_or_manipulable",),
    "hold": ("object_or_manipulable",),
    "lift": ("object_or_manipulable",),
    "move": ("object_or_manipulable", "receptacle_or_site"),
    "pour": ("object_or_manipulable", "receptacle_or_site"),
}


def _structured_targets(
    predicates: Iterable[Any],
    *,
    objects: set[str],
    receptacles: set[str],
    sites: set[str],
    manipulable: set[str],
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...],
    tuple[tuple[str, ...], ...], tuple[str, ...], tuple[str, ...],
]:
    target_objects: list[str] = []
    target_receptacles: list[str] = []
    target_sites: list[str] = []
    target_manipulable: list[str] = []
    subgoals: list[tuple[str, ...]] = []
    unresolved: list[str] = []
    unsupported_operators: list[str] = []

    for predicate in predicates:
        operator, args = _predicate_parts(predicate)
        if not operator or not args:
            continue
        subgoals.append((operator, *args))
        roles = _OPERATOR_ROLES.get(operator)
        if roles is None:
            unsupported_operators.append(operator)
            continue
        for index, role in enumerate(roles):
            if index >= len(args):
                unresolved.append(f"{operator}:missing_arg_{index}")
                continue
            token = args[index]
            if role == "object":
                resolved = _resolve_declared(token, objects)
                target_objects.extend(resolved)
            elif role == "receptacle":
                resolved = _resolve_declared(token, receptacles)
                target_receptacles.extend(resolved)
            elif role in {"receptacle_or_site", "site_or_receptacle"}:
                resolved_r = _resolve_declared(token, receptacles)
                resolved_s = _resolve_declared(token, sites)
                target_receptacles.extend(resolved_r)
                target_sites.extend(resolved_s)
                resolved = resolved_r or resolved_s
            elif role == "object_or_receptacle":
                resolved_o = _resolve_declared(token, objects)
                resolved_r = _resolve_declared(token, receptacles)
                target_objects.extend(resolved_o)
                target_receptacles.extend(resolved_r)
                resolved = resolved_o or resolved_r
            elif role == "manipulable":
                resolved = _resolve_declared(token, manipulable)
                target_manipulable.extend(resolved)
            elif role == "object_or_manipulable":
                resolved_o = _resolve_declared(token, objects)
                resolved_m = _resolve_declared(token, manipulable)
                target_objects.extend(resolved_o)
                target_manipulable.extend(resolved_m)
                resolved = resolved_o or resolved_m
            else:
                resolved = ()
            if not resolved:
                unresolved.append(token)

    return (
        _dedupe(target_objects), _dedupe(target_receptacles), _dedupe(target_sites),
        _dedupe(target_manipulable), tuple(subgoals), _dedupe(unresolved),
        _dedupe(unsupported_operators),
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


def resolve_task_targets(metadata: Mapping[str, Any]) -> TargetResolution:
    """Resolve task targets without simulator access or arbitrary tie-breaking."""
    objects = set(_first_names(metadata, ("object_declarations", "objects")))
    receptacles = set(_first_names(metadata, ("receptacle_declarations", "receptacles")))
    sites = set(_first_names(metadata, ("site_declarations", "sites")))
    fixtures = set(_first_names(metadata, ("fixture_declarations", "fixtures", "manipulable_entities")))
    manipulable = fixtures | set(_first_names(metadata, ("manipulable_receptacles",)))
    language = str(metadata.get("task_language", metadata.get("language", "")))
    ambiguities: list[str] = []
    unresolved = list(_names(metadata.get("unresolved_tokens")))

    structured_blocks = [
        ("structured_goal_metadata", metadata.get("structured_goal_metadata")),
        ("structured_bddl_predicates", metadata.get("goal_predicates", metadata.get("bddl_goal_predicates"))),
    ]
    for source, block in structured_blocks:
        if isinstance(block, Mapping):
            direct_objects = _validate_direct(_first_names(block, ("target_objects", "target_object")), objects, unresolved)
            direct_receptacles = _validate_direct(_first_names(block, ("target_receptacles", "target_receptacle")), receptacles, unresolved)
            direct_sites = _validate_direct(_first_names(block, ("target_sites", "target_site")), sites, unresolved)
            direct_manipulable = _validate_direct(
                _first_names(block, ("target_fixtures", "target_fixture", "target_manipulable_entities")),
                manipulable,
                unresolved,
            )
            if direct_objects or direct_receptacles or direct_sites or direct_manipulable:
                has_unresolved = bool(unresolved)
                return TargetResolution(
                    direct_objects, direct_receptacles, direct_sites, direct_manipulable, (), source,
                    0.75 if has_unresolved else 1.0,
                    "RESOLVED_STRUCTURED_WITH_UNRESOLVED" if has_unresolved else "RESOLVED_STRUCTURED",
                    (), _dedupe(unresolved),
                )

        predicates = _flatten_predicates(block)
        if not predicates:
            continue
        (
            target_objects, target_receptacles, target_sites, target_manipulable,
            subgoals, missed, unsupported,
        ) = _structured_targets(
            predicates, objects=objects, receptacles=receptacles, sites=sites, manipulable=manipulable,
        )
        unresolved.extend(missed)
        if unsupported:
            ambiguities.append("UNSUPPORTED_OPERATORS:" + ",".join(unsupported))
        if target_objects or target_receptacles or target_sites or target_manipulable:
            language_entities = set(_mentioned(objects | receptacles | sites | manipulable, language))
            structured_entities = set(target_objects) | set(target_receptacles) | set(target_sites) | set(target_manipulable)
            if language_entities and not language_entities.issubset(structured_entities):
                ambiguities.append("LANGUAGE_STRUCTURED_CONFLICT")
            ordered = metadata.get("ordered_subgoals")
            ordered_subgoals = tuple(
                (operator, *args) for operator, args in (_predicate_parts(item) for item in _flatten_predicates(ordered))
            ) if ordered else subgoals
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
                target_objects, target_receptacles, target_sites, target_manipulable,
                ordered_subgoals, source, confidence, reason,
                _dedupe(ambiguities), _dedupe(unresolved),
            )

    explicit_objects = _validate_direct(
        _first_names(metadata, ("target_objects", "target_object", "valid_target_objects")), objects, unresolved
    )
    explicit_receptacles = _validate_direct(
        _first_names(metadata, ("target_receptacles", "target_receptacle")), receptacles, unresolved
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
            explicit_objects, explicit_receptacles, explicit_sites, explicit_manipulable, (),
            "explicit_task_metadata", 0.5 if ambiguities or unresolved else 0.8,
            "AMBIGUOUS_MULTIPLE_TARGETS" if ambiguities else (
                "RESOLVED_EXPLICIT_WITH_UNRESOLVED" if unresolved else "RESOLVED_EXPLICIT_METADATA"
            ),
            _dedupe(ambiguities), _dedupe(unresolved),
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
            (), "language_fallback", 0.5, "RESOLVED_LANGUAGE_FALLBACK", (), _dedupe(unresolved),
        )
    if len(mentioned) > 1:
        ambiguities.append("LANGUAGE_MULTIPLE_TARGET_ENTITIES")
    return TargetResolution(
        (), (), (), (), (), "unresolved", 0.0,
        "AMBIGUOUS_LANGUAGE_TARGET" if ambiguities else "TARGET_METADATA_MISSING",
        _dedupe(ambiguities), _dedupe(unresolved),
    )
