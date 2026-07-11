"""Structured BDDL/PDDL metadata extraction for clean Teacher-v2 collection.

The parser preserves the distinctions used by official LIBERO task files:

* ``:objects`` are movable/object declarations even when their type name is
  destination-like (for example ``basket`` or ``plate``);
* ``:fixtures`` are parsed independently;
* ``:regions`` are converted to the fully-qualified MuJoCo/BDDL site names used
  in goals, together with an explicit site -> owner mapping;
* goal predicates retain their source order for later per-step event binding.

This module remains simulator-free and performs no language-only target guessing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.gripper_attack.c2g_semantic_aliases import normalize_goal_operator
from tools.multisuite_detector.audit_c2g_static_assets import parse_sexpr


# Backward-compatible local name used throughout the parser. The implementation is
# shared with the strict live-asset audit so collection and audit cannot disagree
# about reviewed BDDL syntax aliases such as ``turnon`` -> ``turn_on``.
normalize_operator = normalize_goal_operator


_DESTINATION_TYPE_TOKENS = (
    "receptacle",
    "container",
    "basket",
    "bowl",
    "plate",
    "tray",
    "rack",
    "caddy",
    "microwave",
)
_FIXTURE_TYPE_TOKENS = (
    "fixture",
    "drawer",
    "cabinet",
    "stove",
    "button",
    "handle",
    "door",
    "microwave",
    "table",
    "floor",
)


def _walk(node: Any) -> Iterable[list[Any]]:
    if isinstance(node, list):
        yield node
        for child in node:
            yield from _walk(child)


def _section(parsed: list[Any], name: str) -> list[Any]:
    wanted = normalize_operator(name)
    for node in _walk(parsed):
        if node and isinstance(node[0], str) and normalize_operator(node[0]) == wanted:
            return node[1:]
    return []


def _typed_symbols(tokens: list[Any]) -> list[tuple[str, str]]:
    flat = [str(token) for token in tokens if isinstance(token, str)]
    pending: list[str] = []
    result: list[tuple[str, str]] = []
    index = 0
    while index < len(flat):
        token = flat[index]
        if token == "-" and index + 1 < len(flat):
            type_name = flat[index + 1]
            result.extend((symbol, type_name) for symbol in pending)
            pending.clear()
            index += 2
            continue
        if not token.startswith(":"):
            pending.append(token)
        index += 1
    result.extend((symbol, "") for symbol in pending)
    return result


def _goal_predicates(node: Any) -> list[list[str]]:
    if not isinstance(node, list) or not node:
        return []
    head = normalize_operator(node[0]) if isinstance(node[0], str) else ""
    if head in {"and", "or", "ordered"}:
        output: list[list[str]] = []
        for child in node[1:]:
            output.extend(_goal_predicates(child))
        return output
    if head in {"not", "exists", "forall", "when", "imply", "implies", "preference"}:
        output: list[list[str]] = []
        for child in node[1:]:
            output.extend(_goal_predicates(child))
        return output
    if head and not head.startswith("_") and not str(node[0]).startswith(("?", ":")):
        args = [str(value) for value in node[1:] if isinstance(value, str) and not str(value).startswith("?")]
        return [[head, *args]]
    return []


def _region_target(region_node: Any) -> tuple[str, str] | None:
    """Return ``(local_region_name, owner_entity)`` for one :regions entry."""

    if not isinstance(region_node, list) or not region_node or not isinstance(region_node[0], str):
        return None
    local_name = str(region_node[0]).strip()
    owner = ""
    for child in region_node[1:]:
        if (
            isinstance(child, list)
            and len(child) >= 2
            and isinstance(child[0], str)
            and normalize_operator(child[0]) == "target"
        ):
            owner = str(child[1]).strip()
            break
    if not local_name or not owner:
        return None
    return local_name, owner


def _qualified_region_name(owner: str, local_name: str) -> str:
    owner = str(owner).strip()
    local_name = str(local_name).strip()
    prefix = owner + "_"
    return local_name if local_name.startswith(prefix) else prefix + local_name


def parse_bddl_task_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    parsed = parse_sexpr(path.read_text(encoding="utf-8"))

    object_typed = _typed_symbols(_section(parsed, "objects"))
    fixture_typed = _typed_symbols(_section(parsed, "fixtures"))

    # Official LIBERO puts movable destinations such as baskets and plates in
    # ``:objects``. Preserve them as objects; expose destination capability in a
    # separate field rather than deleting them from object_declarations.
    object_names = sorted({name for name, _ in object_typed})
    destination_objects = sorted(
        {
            name
            for name, type_name in object_typed
            if any(token in normalize_operator(type_name) for token in _DESTINATION_TYPE_TOKENS)
        }
    )

    fixture_names = {
        name for name, _ in fixture_typed
    } | {
        name
        for name, type_name in object_typed
        if any(token in normalize_operator(type_name) for token in _FIXTURE_TYPE_TOKENS)
    }

    region_rows: list[dict[str, str]] = []
    region_owner_by_site: dict[str, str] = {}
    region_local_name_by_site: dict[str, str] = {}
    for node in _section(parsed, "regions"):
        parsed_region = _region_target(node)
        if parsed_region is None:
            continue
        local_name, owner = parsed_region
        full_name = _qualified_region_name(owner, local_name)
        if full_name in region_owner_by_site and region_owner_by_site[full_name] != owner:
            raise ValueError(f"conflicting region owner for {full_name}")
        region_owner_by_site[full_name] = owner
        region_local_name_by_site[full_name] = local_name
        region_rows.append({"name": full_name, "local_name": local_name, "owner": owner})

    goal_nodes = _section(parsed, "goal")
    predicates: list[list[str]] = []
    for node in goal_nodes:
        predicates.extend(_goal_predicates(node))
    if not predicates:
        raise ValueError(f"no goal predicates parsed from {path}")

    object_interest = sorted(
        {
            str(value).strip()
            for value in _section(parsed, "obj_of_interest")
            if isinstance(value, str) and str(value).strip()
        }
    )

    typed_rows = [
        {"name": name, "type": type_name, "section": "objects"}
        for name, type_name in object_typed
    ] + [
        {"name": name, "type": type_name, "section": "fixtures"}
        for name, type_name in fixture_typed
    ]

    return {
        "bddl_path": str(path),
        "object_declarations": object_names,
        # Backward-compatible destination summary. These names remain present in
        # object_declarations and must not be treated as a disjoint object class.
        "receptacle_declarations": destination_objects,
        "destination_object_declarations": destination_objects,
        "site_declarations": sorted(region_owner_by_site),
        "fixture_declarations": sorted(fixture_names),
        "region_declarations": sorted(region_rows, key=lambda row: row["name"]),
        "region_owner_by_site": dict(sorted(region_owner_by_site.items())),
        "region_local_name_by_site": dict(sorted(region_local_name_by_site.items())),
        "object_of_interest": object_interest,
        "typed_declarations": typed_rows,
        "goal_predicates": predicates,
        "ordered_subgoals": predicates,
    }
