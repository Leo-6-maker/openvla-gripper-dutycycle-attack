"""Pure MuJoCo name canonicalization and role-aware contact identity helpers."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


_SUFFIX_PATTERNS = (
    r"_link\d+$",
    r"_(?:visual|collision|geom)(?:_?\d+)?$",
    r"_(?:mesh|shape)(?:_?\d+)?$",
)
_STATIC_DEFAULTS = {"world", "table", "floor", "wall", "ground", "workspace"}
_ROLE_PRIORITY = {
    "static_receptacle": 0,
    "object": 1,
    "manipulable_receptacle": 2,
    "manipulable_fixture": 3,
}


@dataclass(frozen=True)
class ContactIdentity:
    contacted_objects: tuple[str, ...]
    contacted_manipulable_entities: tuple[str, ...]
    entity_roles: tuple[tuple[str, str], ...]
    contacted_object_confidence: float
    left_finger_contact: bool
    right_finger_contact: bool
    bilateral_grasp_candidate: bool
    raw_contact_pairs: tuple[tuple[str, str], ...]
    canonical_contact_pairs: tuple[tuple[str, str], ...]
    ambiguity_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_mujoco_name(name: str) -> str:
    value = str(name).strip().lower().replace("/", "_").replace("::", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    previous = None
    while value and value != previous:
        previous = value
        for pattern in _SUFFIX_PATTERNS:
            value = re.sub(pattern, "", value)
    return value


def finger_side(name: str, aliases: Mapping[str, str] | None = None) -> str:
    canonical = canonicalize_mujoco_name(name)
    if aliases:
        normalized_aliases = {canonicalize_mujoco_name(key): str(value).lower() for key, value in aliases.items()}
        side = normalized_aliases.get(canonical, "")
        if side in {"left", "right"}:
            return side
    if not any(token in canonical for token in ("finger", "gripper", "jaw")):
        return ""

    # In the official Panda assets the two jaws are also exposed as numbered
    # ``finger_joint1``/``finger_joint2`` components, often with a ``_tip`` suffix.
    # The side names here are deterministic jaw identities; downstream bilateral
    # grasp logic is symmetric and does not depend on a geometric handedness claim.
    left_patterns = (
        r"(?:^|_)left(?:_|$)",
        r"(?:^|_)l_finger(?:_|$)",
        r"(?:^|_)finger_l(?:_|$)",
        r"leftfinger",
        r"finger1(?:_|$)",
        r"jaw1(?:_|$)",
        r"(?:^|_)finger_joint_?1(?:_|$)",
    )
    right_patterns = (
        r"(?:^|_)right(?:_|$)",
        r"(?:^|_)r_finger(?:_|$)",
        r"(?:^|_)finger_r(?:_|$)",
        r"rightfinger",
        r"finger2(?:_|$)",
        r"jaw2(?:_|$)",
        r"(?:^|_)finger_joint_?2(?:_|$)",
    )
    if any(re.search(pattern, canonical) for pattern in left_patterns):
        return "left"
    if any(re.search(pattern, canonical) for pattern in right_patterns):
        return "right"
    return ""


def _declared_name(value: Any) -> str:
    if isinstance(value, str):
        return canonicalize_mujoco_name(value)
    if isinstance(value, Mapping):
        return canonicalize_mujoco_name(str(value.get("name", value.get("id", ""))))
    return ""


def _pair(value: Any) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return str(value.get("geom1", value.get("a", ""))), str(value.get("geom2", value.get("b", "")))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return str(value[0]), str(value[1])
    raise ValueError("contact pair must contain exactly two geom names")


def _entity_declarations(
    *,
    object_names: Sequence[Any],
    receptacle_names: Sequence[Any],
    manipulable_receptacle_names: Sequence[Any],
    fixture_names: Sequence[Any],
) -> list[tuple[str, str]]:
    """Return one deterministic role per canonical entity identity.

    Official LIBERO movable destinations such as baskets are declared in ``:objects``.
    Some legacy adapters also repeat them in a receptacle list. Duplicating one name
    with two roles makes every contact look ambiguous, so roles are merged by an
    explicit priority: manipulable fixture/receptacle > object > static receptacle.
    """

    manipulable_receptacles = {_declared_name(value) for value in manipulable_receptacle_names}
    candidates: list[tuple[str, str]] = []
    candidates.extend((_declared_name(value), "object") for value in object_names)
    for value in receptacle_names:
        name = _declared_name(value)
        candidates.append(
            (name, "manipulable_receptacle" if name in manipulable_receptacles else "static_receptacle")
        )
    candidates.extend((_declared_name(value), "manipulable_fixture") for value in fixture_names)

    by_name: dict[str, str] = {}
    for name, role in candidates:
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None or _ROLE_PRIORITY[role] > _ROLE_PRIORITY[existing]:
            by_name[name] = role
    return sorted(by_name.items())


def _map_component(name: str, declarations: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    candidates = [(item, role) for item, role in declarations if name == item or name.startswith(item + "_")]
    if not candidates:
        return ()
    longest = max(len(item) for item, _ in candidates)
    return tuple(sorted((item, role) for item, role in candidates if len(item) == longest))


def analyze_contact_pairs(
    contact_pairs: Iterable[Any],
    *,
    object_names: Sequence[Any],
    receptacle_names: Sequence[Any] = (),
    manipulable_receptacle_names: Sequence[Any] = (),
    fixture_names: Sequence[Any] = (),
    static_names: Sequence[str] = (),
    finger_aliases: Mapping[str, str] | None = None,
) -> ContactIdentity:
    declarations = _entity_declarations(
        object_names=object_names,
        receptacle_names=receptacle_names,
        manipulable_receptacle_names=manipulable_receptacle_names,
        fixture_names=fixture_names,
    )
    static = {canonicalize_mujoco_name(value) for value in static_names} | _STATIC_DEFAULTS
    static_receptacles = {name for name, role in declarations if role == "static_receptacle"}
    raw_pairs: list[tuple[str, str]] = []
    canonical_pairs: list[tuple[str, str]] = []
    side_entities: dict[str, set[tuple[str, str]]] = {"left": set(), "right": set()}
    mapping_ambiguity = False

    for value in contact_pairs:
        raw_a, raw_b = _pair(value)
        a, b = canonicalize_mujoco_name(raw_a), canonicalize_mujoco_name(raw_b)
        raw_pairs.append((raw_a, raw_b))
        canonical_pairs.append((a, b))
        side_a, side_b = finger_side(raw_a, finger_aliases), finger_side(raw_b, finger_aliases)
        if side_a and side_b:
            continue
        side, other = (side_a, b) if side_a else (side_b, a) if side_b else ("", "")
        if not side or not other or other in static or other in static_receptacles or other.startswith("robot0_"):
            continue
        mapped = _map_component(other, declarations)
        if len(mapped) > 1:
            mapping_ambiguity = True
        for entity in mapped:
            if entity[1] != "static_receptacle":
                side_entities[side].add(entity)

    all_entities = side_entities["left"] | side_entities["right"]
    contacted_objects = tuple(sorted(name for name, role in all_entities if role == "object"))
    contacted_manipulable = tuple(sorted(name for name, role in all_entities if role != "object"))
    entity_roles = tuple(sorted(all_entities))
    bilateral_entities = side_entities["left"] & side_entities["right"]
    bilateral = len(bilateral_entities) == 1 and len(all_entities) == 1

    if mapping_ambiguity:
        ambiguity = "AMBIGUOUS_CANONICAL_OBJECT_MAPPING"
    elif len(all_entities) > 1:
        ambiguity = (
            "MULTIPLE_SIMULTANEOUS_CONTACTED_OBJECTS"
            if all(role == "object" for _, role in all_entities)
            else "MULTIPLE_SIMULTANEOUS_CONTACTED_ENTITIES"
        )
    elif not all_entities:
        ambiguity = "NO_OBJECT_CONTACT"
    elif not bilateral:
        ambiguity = "UNILATERAL_OBJECT_CONTACT"
    else:
        ambiguity = ""
    confidence = 1.0 if bilateral else 0.6 if len(all_entities) == 1 else 0.25 if all_entities else 0.0
    return ContactIdentity(
        contacted_objects,
        contacted_manipulable,
        entity_roles,
        confidence,
        bool(side_entities["left"]),
        bool(side_entities["right"]),
        bilateral,
        tuple(raw_pairs),
        tuple(canonical_pairs),
        ambiguity,
    )
