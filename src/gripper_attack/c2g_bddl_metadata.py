"""Structured BDDL/PDDL metadata extraction for clean Teacher-v2 collection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from tools.multisuite_detector.audit_c2g_static_assets import normalize_operator, parse_sexpr


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


def parse_bddl_task_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    parsed = parse_sexpr(path.read_text(encoding="utf-8"))
    typed = _typed_symbols(_section(parsed, "objects"))
    object_names: list[str] = []
    receptacles: list[str] = []
    sites: list[str] = []
    fixtures: list[str] = []
    typed_rows: list[dict[str, str]] = []
    for name, type_name in typed:
        normalized_type = normalize_operator(type_name)
        typed_rows.append({"name": name, "type": type_name})
        if any(token in normalized_type for token in ("site", "region", "zone", "marker")):
            sites.append(name)
        elif any(token in normalized_type for token in ("receptacle", "container", "bowl", "plate", "tray", "basket")):
            receptacles.append(name)
        elif any(token in normalized_type for token in ("fixture", "drawer", "cabinet", "stove", "button", "handle", "door")):
            fixtures.append(name)
        else:
            object_names.append(name)

    goal_nodes = _section(parsed, "goal")
    predicates: list[list[str]] = []
    for node in goal_nodes:
        predicates.extend(_goal_predicates(node))
    if not predicates:
        raise ValueError(f"no goal predicates parsed from {path}")
    return {
        "bddl_path": str(path),
        "object_declarations": sorted(set(object_names)),
        "receptacle_declarations": sorted(set(receptacles)),
        "site_declarations": sorted(set(sites)),
        "fixture_declarations": sorted(set(fixtures)),
        "typed_declarations": typed_rows,
        "goal_predicates": predicates,
        "ordered_subgoals": predicates,
    }
