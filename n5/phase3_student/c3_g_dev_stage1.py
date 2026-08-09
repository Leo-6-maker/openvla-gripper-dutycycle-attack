"""C3-G-DEV stage 1: role-safe In/On/Stack eligibility only.

This is a pure contract adapter. It does not discover roots, read episodes, or
turn unknown geometry into a negative label.
"""
from __future__ import annotations

from typing import Any, Mapping

from t2rc1_v2_registry import resolve_relation


SUPPORTED_PREDICATES = frozenset({"In", "On", "Stack"})


def classify_relation(
    predicate: str,
    object_name: str,
    target_name: str,
    bddl_object_names: set[str],
    sites: Mapping[str, Mapping[str, Any]],
    bodies: Mapping[str, Mapping[str, Any]],
    geoms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if predicate not in SUPPORTED_PREDICATES:
        raise ValueError(f"unsupported placement predicate: {predicate}")
    relation = resolve_relation(predicate, object_name, target_name, bddl_object_names, sites, bodies, geoms)
    if not relation["relation_ok"]:
        return {
            "status": "HOLD_UNKNOWN",
            "eligible": False,
            "unknown_is_negative": False,
            "predicate": predicate,
            "relation": relation,
        }
    return {
        "status": "PASS",
        "eligible": True,
        "unknown_is_negative": False,
        "predicate": predicate,
        "relation": relation,
    }
