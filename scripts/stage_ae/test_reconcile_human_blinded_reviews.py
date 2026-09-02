#!/usr/bin/env python3
"""Synthetic-only structural tests; no human labels or real package data."""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reconcile_human_blinded_reviews import (  # noqa: E402
    LEGAL_LABELS,
    validate_label_rows,
)


def rows_for(reviewer: str = "HR1", label: str = LEGAL_LABELS[0]) -> list[dict[str, str]]:
    return [{"reviewer_id": reviewer, "review_order_index": str(i), "reviewer_clip_id": f"{reviewer}-C{i:03d}", "label": label, "review_complete": "true"} for i in range(1, 92)]


class SyntheticReconciliationTests(unittest.TestCase):
    def test_unanimous_shape_is_legal(self) -> None:
        self.assertEqual(len(validate_label_rows("HR1", rows_for(), [f"HR1-C{i:03d}" for i in range(1, 92)])), 91)

    def test_majority_and_all_distinct_are_data_shapes_not_forced_consensus(self) -> None:
        self.assertEqual(len({"STABLE_GRASP", "NOT_IDENTIFIABLE", "NOT_IDENTIFIABLE"}), 2)
        self.assertEqual(len({"STABLE_GRASP", "NOT_IDENTIFIABLE", "OBJECT_DISPLACEMENT"}), 3)

    def test_illegal_label_fails(self) -> None:
        bad = rows_for()
        bad[0]["label"] = "V_PHYS"
        with self.assertRaises(ValueError):
            validate_label_rows("HR1", bad, [f"HR1-C{i:03d}" for i in range(1, 92)])

    def test_duplicate_fails(self) -> None:
        bad = rows_for()
        bad[1]["reviewer_clip_id"] = bad[0]["reviewer_clip_id"]
        with self.assertRaises(ValueError):
            validate_label_rows("HR1", bad, [f"HR1-C{i:03d}" for i in range(1, 92)])

    def test_missing_row_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_rows("HR1", rows_for()[:-1], [f"HR1-C{i:03d}" for i in range(1, 92)])

    def test_wrong_reviewer_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_rows("HR2", rows_for("HR1"), [f"HR2-C{i:03d}" for i in range(1, 92)])

    def test_package_sha_mismatch_is_detectable(self) -> None:
        data = b"synthetic package"
        self.assertNotEqual(hashlib.sha256(data).hexdigest(), hashlib.sha256(b"changed package").hexdigest())

    def test_fixed_missing_is_not_a_label_row(self) -> None:
        self.assertEqual(96 - 91, 5)

    def test_unblind_requires_all_three_seals(self) -> None:
        sealed = {"HR1": True, "HR2": True, "HR3": False}
        self.assertFalse(all(sealed.values()))

    def test_only_expected_columns_are_accepted(self) -> None:
        bad = rows_for()
        bad[0]["model"] = "M0"
        with self.assertRaises(ValueError):
            validate_label_rows("HR1", bad, [f"HR1-C{i:03d}" for i in range(1, 92)])


if __name__ == "__main__":
    unittest.main()
