"""Pure MuJoCo name canonicalization and contact identity helpers."""
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


@dataclass(frozen=True)
class ContactIdentity:
    contacted_objects: tuple[str, ...]
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
    """Normalize render/collision/link suffixes while retaining instance IDs."""
    value = str(name).strip().lower().replace("/", "_").replace("::", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    previous = None
    while value and value != previous:
        previous = value
        for pattern in _SUFFIX_PATTERNS:
            value = re.sub(pattern, "", value)
    return value


def finger_side(name: str) -> str:
    canonical = canonicalize_mujoco_name(name)
    if "finger" not in canonical and "gripper" not in canonical:
        return ""
    if re.search(r"(?:^|_)left(?:_|$)", canonical):
        return "left"
    if re.search(r"(?:^|_)right(?:_|$)", canonical):
        return "right"
    return ""


def _declared_name(value: Any) -> str:
    if isinstance(value, str):
        return canonicalize_mujoco_name(value)
    if isinstance(value, Mapping):
        return canonicalize_mujoco_name(str(value.get("name", value.get("id", ""))))
    return ""


def _map_component(name: str, declared: Sequence[str]) -> tuple[str, ...]:
    candidates = [item for item in declared if name == item or name.startswith(item + "_")]
    if not candidates:
        return ()
    longest = max(len(item) for item in candidates)
    return tuple(sorted(item for item in candidates if len(item) == longest))


def _pair(value: Any) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return str(value.get("geom1", value.get("a", ""))), str(value.get("geom2", value.get("b", "")))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return str(value[0]), str(value[1])
    raise ValueError("contact pair must contain exactly two geom names")


def analyze_contact_pairs(
    contact_pairs: Iterable[Any],
    *,
    object_names: Sequence[Any],
    receptacle_names: Sequence[Any] = (),
    static_names: Sequence[str] = (),
) -> ContactIdentity:
    """Map finger contacts to canonical object instances with ambiguity reporting."""
    objects = tuple(filter(None, (_declared_name(value) for value in object_names)))
    receptacles = set(filter(None, (_declared_name(value) for value in receptacle_names)))
    static = {canonicalize_mujoco_name(value) for value in static_names} | _STATIC_DEFAULTS
    raw_pairs: list[tuple[str, str]] = []
    canonical_pairs: list[tuple[str, str]] = []
    side_objects: dict[str, set[str]] = {"left": set(), "right": set()}
    mapping_ambiguity = False

    for value in contact_pairs:
        raw_a, raw_b = _pair(value)
        a, b = canonicalize_mujoco_name(raw_a), canonicalize_mujoco_name(raw_b)
        raw_pairs.append((raw_a, raw_b))
        canonical_pairs.append((a, b))
        side_a, side_b = finger_side(raw_a), finger_side(raw_b)
        if side_a and side_b:
            continue
        side, other = (side_a, b) if side_a else (side_b, a) if side_b else ("", "")
        if not side or not other or other in static or other in receptacles or other.startswith("robot0_"):
            continue
        mapped = _map_component(other, objects)
        if len(mapped) > 1:
            mapping_ambiguity = True
        side_objects[side].update(mapped)

    contacted = tuple(sorted(side_objects["left"] | side_objects["right"]))
    bilateral_objects = side_objects["left"] & side_objects["right"]
    bilateral = len(bilateral_objects) == 1 and len(contacted) == 1
    if mapping_ambiguity:
        ambiguity = "AMBIGUOUS_CANONICAL_OBJECT_MAPPING"
    elif len(contacted) > 1:
        ambiguity = "MULTIPLE_SIMULTANEOUS_CONTACTED_OBJECTS"
    elif not contacted:
        ambiguity = "NO_OBJECT_CONTACT"
    elif not bilateral:
        ambiguity = "UNILATERAL_OBJECT_CONTACT"
    else:
        ambiguity = ""
    confidence = 1.0 if bilateral else 0.6 if len(contacted) == 1 else 0.25 if contacted else 0.0
    return ContactIdentity(
        contacted,
        confidence,
        bool(side_objects["left"]),
        bool(side_objects["right"]),
        bilateral,
        tuple(raw_pairs),
        tuple(canonical_pairs),
        ambiguity,
    )
