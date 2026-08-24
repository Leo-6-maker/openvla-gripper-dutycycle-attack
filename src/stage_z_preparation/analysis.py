"""Synthetic-only analysis helpers for the future Stage-Z report pipeline."""

from __future__ import annotations

import random
from collections import Counter
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .contract import FROZEN_PARENT_COUNT, StageZHold
from .panel import SUITE_COUNTS
from .telemetry import validate_synthetic_row


ANALYSIS_SCHEMA = "STAGE_Z_SYNTHETIC_ANALYSIS_PREP_V1"


def summarize_synthetic_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a test-only schema summary, never a scientific result."""

    for row in rows:
        validate_synthetic_row(row)
    suites = Counter(str(row.get("suite")) for row in rows)
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "TEST_ONLY_NON_SCIENTIFIC",
        "unit": "MODEL_PARENT_PAIR",
        "frozen_parent_denominator": FROZEN_PARENT_COUNT,
        "suite_denominators": dict(SUITE_COUNTS),
        "rows_seen": len(rows),
        "suite_rows_seen": dict(suites),
        "complete_case_substitution": False,
        "structural_missing_cells_are_failures": False,
        "official_sr_is_primary": False,
    }


def parent_bootstrap_mean(values: Sequence[float], *, seed: int, replicates: int = 2_000) -> tuple[float, float, float]:
    """Deterministic parent-unit bootstrap interval for future offline analysis."""

    if not values:
        raise StageZHold("BOOTSTRAP_REQUIRES_PARENT_VALUES")
    if int(replicates) <= 0:
        raise StageZHold("BOOTSTRAP_REPLICATES_MUST_BE_POSITIVE")
    parent_values = tuple(float(value) for value in values)
    rng = random.Random(int(seed))
    estimates = [mean(rng.choice(parent_values) for _ in parent_values) for _ in range(int(replicates))]
    estimates.sort()
    lower = estimates[max(0, int(0.025 * len(estimates)) - 1)]
    upper = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return mean(parent_values), lower, upper


__all__ = ["ANALYSIS_SCHEMA", "parent_bootstrap_mean", "summarize_synthetic_rows"]
