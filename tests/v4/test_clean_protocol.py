"""Test clean/autowindow protocol invariants — no GPU, no model."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.condition_protocols import (
    CLEAN_DETECT_PROTOCOL,
    COMMAND_OPEN_ORACLE_PROTOCOL,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
    DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
)


class TestCleanProtocol:
    def test_attack_enabled_false(self):
        assert CLEAN_DETECT_PROTOCOL["attack_enabled"] is False

    def test_attack_objective_empty_string(self):
        assert CLEAN_DETECT_PROTOCOL["attack_objective"] == ""

    def test_rho_zero(self):
        assert CLEAN_DETECT_PROTOCOL["rho"] == 0.0

    def test_not_attack_not_control(self):
        assert CLEAN_DETECT_PROTOCOL.get("is_attack") is False
        assert CLEAN_DETECT_PROTOCOL.get("is_control") is False


class TestCommandOpen:
    def test_rho_positive(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["rho"] > 0

    def test_oracle_objective(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["attack_objective"] == "oracle_env_gripper_open"

    def test_has_env_extra(self):
        assert "env_extra" in COMMAND_OPEN_ORACLE_PROTOCOL
        assert "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE" in COMMAND_OPEN_ORACLE_PROTOCOL["env_extra"]


class TestMechanismDistinction:
    def test_force_gripper_open_ce_vs_margin_cw_are_different(self):
        """These are fundamentally different mechanisms — not interchangeable."""
        ce = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        margin = DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL
        assert ce["attack_objective"] != margin["attack_objective"]

    def test_codex_targeted_uses_ce_not_margin(self):
        c = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        assert c["attack_objective"] == "force_gripper_open_token_ce"


class TestLegacyCodexConditions:
    def test_all_have_positive_rho(self):
        for c in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS:
            assert c["rho"] > 0, f"{c['condition_name']}: rho must be >0"

    def test_matched_count(self):
        assert len(LEGACY_CODEX_STATE5_MATCHED_CONDITIONS) == 4
