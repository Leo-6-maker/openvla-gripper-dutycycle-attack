"""Frozen Stage-Z population membership checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contract import FROZEN_PARENT_COUNT, StageZHold


STRUCTURAL_MISSING = frozenset(
    {
        "libero_goal/task_01",
        "libero_goal/task_04",
        "libero_goal/task_06",
        "libero_goal/task_09",
    }
)
SUITE_COUNTS = {"libero_10": 10, "libero_goal": 6, "libero_object": 10, "libero_spatial": 10}


@dataclass(frozen=True)
class FrozenPanel:
    """The 36 consumable shared parents and four structural missing cells."""

    selected_parent_keys: frozenset[str]
    structural_missing_cells: frozenset[str] = STRUCTURAL_MISSING
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.selected_parent_keys) != FROZEN_PARENT_COUNT:
            raise StageZHold("FROZEN_PANEL_COUNT_INVALID")
        if self.structural_missing_cells != STRUCTURAL_MISSING:
            raise StageZHold("STRUCTURAL_MISSINGNESS_SET_CHANGED")
        self._validate_suite_counts()

    def _validate_suite_counts(self) -> None:
        counts = {suite: 0 for suite in SUITE_COUNTS}
        for key in self.selected_parent_keys:
            suite, task, state = key.split("/")
            if suite not in counts or not task.startswith("task_") or not state.startswith("state_"):
                raise StageZHold(f"INVALID_PARENT_KEY:{key}")
            counts[suite] += 1
        if counts != SUITE_COUNTS:
            raise StageZHold(f"FROZEN_PANEL_SUITE_COUNTS_INVALID:{counts}")

    def contains(self, parent_key: str) -> bool:
        return parent_key in self.selected_parent_keys

    def require_scientific_parent(self, parent_key: str) -> None:
        if parent_key in self.structural_missing_cells:
            raise StageZHold("STRUCTURAL_MISSING_CELL_NOT_A_PARENT")
        if parent_key not in self.selected_parent_keys:
            raise StageZHold("PARENT_NOT_IN_FROZEN_STAGE_Z_PANEL")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrozenPanel":
        if record.get("status") != "STAGE_Z_Z0R1_36_PANEL_SEALED":
            raise StageZHold("FROZEN_PANEL_STATUS_NOT_SEALED")
        selected = record.get("selected_parent_keys")
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
            raise StageZHold("FROZEN_PANEL_SELECTED_KEYS_MISSING")
        missing = record.get("structural_missing_cells")
        if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
            missing = record.get("missing_task_cells")
        if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
            raise StageZHold("STRUCTURAL_MISSINGNESS_RECORD_MISSING")
        return cls(
            selected_parent_keys=frozenset(str(value) for value in selected),
            structural_missing_cells=frozenset(str(value) for value in missing),
            source_sha256=str(record.get("source_sha256")) if record.get("source_sha256") else None,
        )


__all__ = ["FrozenPanel", "STRUCTURAL_MISSING", "SUITE_COUNTS"]
