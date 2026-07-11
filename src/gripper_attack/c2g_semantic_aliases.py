"""Canonical syntax aliases observed in official LIBERO assets.

The mappings in this module are syntax normalization only. They do not change the
Teacher-v2 scientific target or infer a task from language. Raw BDDL spellings are
mapped to the already supported canonical operator vocabulary before target-role
resolution.
"""
from __future__ import annotations

import re
from typing import Any


# Official LIBERO task files use compact spellings for a small number of predicates.
# Keep this map explicit and deliberately small; unknown predicates must still fail
# closed in the static inventory and Teacher target resolver.
GOAL_OPERATOR_ALIASES: dict[str, str] = {
    "turnon": "turn_on",
    "turnoff": "turn_off",
}


def normalize_goal_operator(value: Any) -> str:
    """Normalize one BDDL/PDDL operator and apply reviewed syntax aliases."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return GOAL_OPERATOR_ALIASES.get(normalized, normalized)
