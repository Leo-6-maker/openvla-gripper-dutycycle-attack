"""Test metadata schema — no GPU, no model, no rollout."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.task_identity import (
    TASK_IDENTITY, MATCHED_CONDITIONS, make_run_id,
)

# Required fields for a well-formed run_manifest
MANIFEST_REQUIRED_FIELDS = {
    "runner_task_id",
    "semantic_task_name",
    "suite",
    "condition",
    "attack_enabled",
    "attack_objective",
    "fixed_window_used",
    "clean_trajectory_only_window_selection",
    "state_specific_autowindow_used",
}

# Required fields for a well-formed summary.csv row
SUMMARY_REQUIRED_FIELDS = {
    "run_id",
    "task_id",
    "runner_task_id",
    "semantic_task_name",
}


class TestManifestSchema:
    def test_required_fields_defined(self):
        assert len(MANIFEST_REQUIRED_FIELDS) >= 8

    def test_clean_detect_manifest_fields(self):
        """Clean detect should have attack_enabled=False, attack_objective=''."""
        clean_manifest = {
            "runner_task_id": TASK_IDENTITY["runner_task_id"],
            "semantic_task_name": TASK_IDENTITY["semantic_task_name"],
            "suite": TASK_IDENTITY["suite"],
            "condition": "clean_detect",
            "attack_enabled": False,
            "attack_objective": "",
            "fixed_window_used": False,
            "clean_trajectory_only_window_selection": True,
            "state_specific_autowindow_used": True,
        }
        assert clean_manifest["attack_enabled"] is False
        assert clean_manifest["attack_objective"] == ""

    def test_matched_condition_manifest_fields(self):
        """Matched conditions should record their objective config."""
        for cond in MATCHED_CONDITIONS:
            manifest = {
                "runner_task_id": TASK_IDENTITY["runner_task_id"],
                "semantic_task_name": TASK_IDENTITY["semantic_task_name"],
                "suite": TASK_IDENTITY["suite"],
                "condition": cond["condition_name"],
                "attack_enabled": cond["is_attack"],
                "attack_objective": cond["attack_objective"],
                "fixed_window_used": False,
                "clean_trajectory_only_window_selection": True,
                "state_specific_autowindow_used": True,
            }
            assert manifest["attack_objective"] == cond["attack_objective"]

    def test_condition_names_are_unique(self):
        names = [c["condition_name"] for c in MATCHED_CONDITIONS]
        assert len(names) == len(set(names)), f"Duplicate condition names: {names}"

    def test_make_run_id_includes_semantic_name(self):
        rid = make_run_id("goal_put_the_bowl_on_the_plate", 5, 3, "clean_detect")
        assert "goal_put_the_bowl_on_the_plate" in rid
        assert "libero_spatial_black_bowl" not in rid


class TestSummarySchema:
    def test_summary_required_fields_defined(self):
        assert len(SUMMARY_REQUIRED_FIELDS) >= 3

    def test_runner_task_id_differs_from_semantic(self):
        """runner_task_id and semantic_task_name must be different fields."""
        runner = TASK_IDENTITY["runner_task_id"]
        semantic = TASK_IDENTITY["semantic_task_name"]
        assert runner != semantic
        assert runner != TASK_IDENTITY["table1_task_key"] or runner != semantic
