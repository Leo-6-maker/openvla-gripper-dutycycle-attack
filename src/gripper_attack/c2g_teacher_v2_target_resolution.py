"""Pure structured target resolution for C2g Teacher-v2."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TargetResolution:
    resolved_target_objects: tuple[str, ...]
    resolved_receptacles: tuple[str, ...]
    resolved_sites: tuple[str, ...]
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
        for key in ("name", "id", "object", "site", "receptacle"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


def _names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = [value]
    return tuple(dict.fromkeys(name for item in value for name in [_name(item)] if name))


def _first_names(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    for key in keys:
        names = _names(metadata.get(key))
        if names:
            return names
    return ()


def _predicate_parts(predicate: Any) -> tuple[str, tuple[str, ...]]:
    if isinstance(predicate, (list, tuple)) and predicate:
        return str(predicate[0]).lower(), tuple(str(item) for item in predicate[1:])
    if not isinstance(predicate, Mapping):
        return "", ()
    operator = str(predicate.get("predicate", predicate.get("operator", predicate.get("name", "")))).lower()
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
        if value and isinstance(value[0], str) and value[0].lower() not in {"and", "or", "ordered"}:
            return [value]
        out: list[Any] = []
        for item in value[1:] if value and isinstance(value[0], str) else value:
            out.extend(_flatten_predicates(item))
        return out
    return []


def _structured_targets(
    predicates: Iterable[Any],
    objects: set[str],
    receptacles: set[str],
    sites: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...], tuple[str, ...]]:
    target_objects: list[str] = []
    target_receptacles: list[str] = []
    target_sites: list[str] = []
    subgoals: list[tuple[str, ...]] = []
    unresolved: list[str] = []
    relation_predicates = {"in", "inside", "on", "at", "place", "put", "contains", "stack"}
    for predicate in predicates:
        operator, args = _predicate_parts(predicate)
        if not args:
            continue
        subgoals.append((operator, *args))
        if operator not in relation_predicates:
            continue
        source = args[0]
        destination = args[1] if len(args) > 1 else ""
        if source in objects:
            target_objects.append(source)
        else:
            unresolved.append(source)
        if destination in receptacles:
            target_receptacles.append(destination)
        elif destination in sites:
            target_sites.append(destination)
        elif destination:
            unresolved.append(destination)
    unique = lambda values: tuple(dict.fromkeys(values))
    return unique(target_objects), unique(target_receptacles), unique(target_sites), tuple(subgoals), unique(unresolved)


def _mentioned(names: Iterable[str], language: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", " ", language.lower()).strip()
    found = []
    for name in names:
        token = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        aliases = {token, re.sub(r"\s+\d+$", "", token)}
        if any(alias and re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            found.append(name)
    return tuple(found)


def resolve_task_targets(metadata: Mapping[str, Any]) -> TargetResolution:
    """Resolve task targets without simulator access or arbitrary tie-breaking."""
    objects = set(_first_names(metadata, ("object_declarations", "objects")))
    receptacles = set(_first_names(metadata, ("receptacle_declarations", "receptacles")))
    sites = set(_first_names(metadata, ("site_declarations", "sites")))
    language = str(metadata.get("task_language", metadata.get("language", "")))
    ambiguities: list[str] = []
    unresolved = list(_names(metadata.get("unresolved_tokens")))

    structured_blocks = [
        ("structured_goal_metadata", metadata.get("structured_goal_metadata")),
        ("structured_bddl_predicates", metadata.get("goal_predicates", metadata.get("bddl_goal_predicates"))),
    ]
    for source, block in structured_blocks:
        if isinstance(block, Mapping):
            direct_objects = _first_names(block, ("target_objects", "target_object"))
            direct_receptacles = _first_names(block, ("target_receptacles", "target_receptacle"))
            direct_sites = _first_names(block, ("target_sites", "target_site"))
            if direct_objects or direct_receptacles or direct_sites:
                return TargetResolution(
                    direct_objects, direct_receptacles, direct_sites, (), source, 1.0,
                    "RESOLVED_STRUCTURED", (), tuple(dict.fromkeys(unresolved)),
                )
        predicates = _flatten_predicates(block)
        if not predicates:
            continue
        target_objects, target_receptacles, target_sites, subgoals, missed = _structured_targets(
            predicates, objects, receptacles, sites,
        )
        unresolved.extend(missed)
        if target_objects or target_receptacles or target_sites:
            language_objects = _mentioned(objects, language)
            if language_objects and set(language_objects) != set(target_objects):
                ambiguities.append("LANGUAGE_STRUCTURED_CONFLICT")
            ordered = metadata.get("ordered_subgoals")
            ordered_subgoals = tuple(_predicate_parts(item)[1] for item in _flatten_predicates(ordered)) if ordered else subgoals
            return TargetResolution(
                target_objects,
                target_receptacles,
                target_sites,
                ordered_subgoals,
                source,
                0.85 if ambiguities else 1.0,
                "RESOLVED_STRUCTURED_WITH_LANGUAGE_CONFLICT" if ambiguities else "RESOLVED_STRUCTURED",
                tuple(ambiguities),
                tuple(dict.fromkeys(unresolved)),
            )

    explicit_objects = _first_names(metadata, ("target_objects", "target_object", "valid_target_objects"))
    explicit_receptacles = _first_names(metadata, ("target_receptacles", "target_receptacle"))
    explicit_sites = _first_names(metadata, ("target_sites", "target_site"))
    if explicit_objects or explicit_receptacles or explicit_sites:
        if len(explicit_objects) > 1 and not metadata.get("ordered_subgoals"):
            ambiguities.append("MULTIPLE_VALID_TARGET_OBJECTS")
        return TargetResolution(
            explicit_objects,
            explicit_receptacles,
            explicit_sites,
            (),
            "explicit_task_metadata",
            0.5 if ambiguities else 0.8,
            "AMBIGUOUS_MULTIPLE_TARGETS" if ambiguities else "RESOLVED_EXPLICIT_METADATA",
            tuple(ambiguities),
            tuple(dict.fromkeys(unresolved)),
        )

    language_objects = _mentioned(objects, language)
    language_receptacles = _mentioned(receptacles, language)
    language_sites = _mentioned(sites, language)
    if len(language_objects) == 1 and len(language_receptacles) + len(language_sites) <= 1:
        return TargetResolution(
            language_objects,
            language_receptacles,
            language_sites,
            (),
            "language_fallback",
            0.5,
            "RESOLVED_LANGUAGE_FALLBACK",
            (),
            tuple(dict.fromkeys(unresolved)),
        )
    if len(language_objects) > 1:
        ambiguities.append("LANGUAGE_MULTIPLE_TARGET_OBJECTS")
    return TargetResolution(
        (), (), (), (), "unresolved", 0.0,
        "AMBIGUOUS_LANGUAGE_TARGET" if ambiguities else "TARGET_METADATA_MISSING",
        tuple(ambiguities), tuple(dict.fromkeys(unresolved)),
    )
