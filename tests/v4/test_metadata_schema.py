"""Test metadata schema — no GPU, no model."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.task_identity import TASK_IDENTITY
from src.utils.condition_protocols import (
    COMMAND_OPEN_ORACLE_PROTOCOL,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
)

MANIFEST_REQUIRED_FIELDS = {
    "runner_task_id", "semantic_task_name", "suite",
    "condition", "condition_name",
    "attack_enabled", "attack_objective",
    "rho", "epsilon", "step_size", "attack_steps",
    "fixed_window_used", "clean_trajectory_only_window_selection",
    "state_specific_autowindow_used",
    "is_black_bowl_related",
}

SUMMARY_REQUIRED_FIELDS = {
    "run_id", "task_id",
    "runner_task_id", "semantic_task_name",
    "condition", "attack_enabled", "attack_objective",
}


class TestManifestSchema:
    def test_required_fields_defined(self):
        assert len(MANIFEST_REQUIRED_FIELDS) >= 14

    def test_clean_detect_manifest(self):
        clean = {
            "runner_task_id": TASK_IDENTITY["runner_task_id"],
            "semantic_task_name": TASK_IDENTITY["semantic_task_name"],
            "suite": TASK_IDENTITY["suite"],
            "condition": "clean_detect",
            "condition_name": "clean_detect",
            "attack_enabled": False,
            "attack_objective": "",
            "rho": 0.0, "epsilon": 0.0, "step_size": 0.0, "attack_steps": 0,
            "fixed_window_used": False,
            "clean_trajectory_only_window_selection": True,
            "state_specific_autowindow_used": True,
            "is_black_bowl_related": True,
        }
        assert clean["attack_enabled"] is False

    def test_condition_names_unique(self):
        names = [c["condition_name"] for c in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS]
        assert len(names) == len(set(names))


class TestSummarySchema:
    def test_summary_required_fields_defined(self):
        assert len(SUMMARY_REQUIRED_FIELDS) >= 5

    def test_runner_task_id_differs_from_semantic(self):
        assert TASK_IDENTITY["runner_task_id"] != TASK_IDENTITY["semantic_task_name"]


class TestCommandOpenValidation:
    def test_rho_must_be_positive(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["rho"] > 0, \
            "command_open rho must be >0 for oracle override"

    def test_objective_must_be_oracle_env(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["attack_objective"] == "oracle_env_gripper_open"


class TestCodexTargetedProtocol:
    def test_is_force_gripper_open_token_ce(self):
        c = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        assert c["attack_objective"] == "force_gripper_open_token_ce"

    def test_not_gripper_logit_margin_cw(self):
        c = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        assert c["attack_objective"] != "gripper_logit_margin_cw"
