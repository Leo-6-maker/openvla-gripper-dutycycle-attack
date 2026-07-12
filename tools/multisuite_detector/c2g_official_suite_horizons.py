#!/usr/bin/env python3
"""Frozen OpenVLA official per-suite policy-step horizons.

These values are extracted from the upstream OpenVLA LIBERO evaluator
and must not be changed without a new protocol generation.

Canonical mode enforces exact match.  Experimental horizons must use a
different schema and status.
"""
from __future__ import annotations

from typing import Set

OFFICIAL_MAX_POLICY_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

OFFICIAL_DUMMY_WAIT_STEPS: int = 10

CANONICAL_SUITES: Set[str] = set(OFFICIAL_MAX_POLICY_STEPS.keys())


def official_max_policy_steps(suite: str) -> int:
    """Return the official OpenVLA max policy steps for *suite*."""
    if suite not in OFFICIAL_MAX_POLICY_STEPS:
        raise KeyError(f"unknown suite: {suite!r}")
    return OFFICIAL_MAX_POLICY_STEPS[suite]


def validate_official_suite_horizon(suite: str, max_policy_steps: int) -> None:
    """Fail-closed if *max_policy_steps* does not match the official value."""
    if suite not in OFFICIAL_MAX_POLICY_STEPS:
        raise ValueError(f"unknown suite: {suite!r}")
    expected = OFFICIAL_MAX_POLICY_STEPS[suite]
    if int(max_policy_steps) != expected:
        raise ValueError(
            f"horizon mismatch for {suite}: "
            f"got {max_policy_steps}, official is {expected}"
        )


def validate_all_canonical_horizons(
    suite_horizons: dict[str, int],
) -> None:
    """Validate that every canonical suite has the correct horizon."""
    for suite in CANONICAL_SUITES:
        if suite not in suite_horizons:
            raise ValueError(f"missing horizon for canonical suite: {suite!r}")
    for suite, steps in suite_horizons.items():
        validate_official_suite_horizon(suite, steps)
