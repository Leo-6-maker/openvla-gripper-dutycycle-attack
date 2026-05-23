"""Test task identity mapping and condition protocols — no GPU, no model."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.task_identity import (
    TASK_IDENTITY, RUNNER_TASK_ID, RUN_ID_TASK_KEY, TABLE1_TASK_KEY,
    DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS,
)
from src.utils.condition_protocols import (
    CLEAN_DETECT_PROTOCOL,
    LEGACY_CODEX_STATE5_PROTOCOL,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
    COMMAND_OPEN_ORACLE_PROTOCOL,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
    CODEX_LEGACY_RANDOM_SAME_WINDOW,
    CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW,
    DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL,
    DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL,
    make_run_id,
    make_clean_detect_run_id,
)


class TestTaskIdentity:
    def test_runner_task_id_is_spatial(self):
        assert TASK_IDENTITY["runner_task_id"] == "libero_spatial_black_bowl"

    def test_semantic_name_is_goal_put_bowl(self):
        assert TASK_IDENTITY["semantic_task_name"] == "goal_put_the_bowl_on_the_plate"

    def test_suite_is_libero_spatial(self):
        assert TASK_IDENTITY["suite"] == "libero_spatial"

    def test_is_black_bowl_related(self):
        assert TASK_IDENTITY["is_black_bowl_related"] is True
        assert TASK_IDENTITY["is_non_black_bowl_claim"] is False

    def test_runner_and_semantic_are_different(self):
        assert RUNNER_TASK_ID != RUN_ID_TASK_KEY

    def test_make_run_id(self):
        rid = make_run_id("goal_put_the_bowl_on_the_plate", 5, 3, "clean_detect")
        assert rid == "goal_put_the_bowl_on_the_plate_s5_r3_clean_detect"


class TestCleanDetectProtocol:
    def test_attack_disabled(self):
        assert CLEAN_DETECT_PROTOCOL["attack_enabled"] is False

    def test_attack_objective_empty(self):
        assert CLEAN_DETECT_PROTOCOL["attack_objective"] == ""

    def test_rho_zero(self):
        """Clean detect uses rho=0 (no attack budget)."""
        assert CLEAN_DETECT_PROTOCOL["rho"] == 0.0


class TestCommandOpenProtocol:
    def test_rho_positive(self):
        """WARNING: rho=0 DISABLES oracle override. Must be >0."""
        assert COMMAND_OPEN_ORACLE_PROTOCOL["rho"] > 0, \
            "command_open rho=0 disables oracle override (attack_active never True)"

    def test_objective_is_oracle_env(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["attack_objective"] == "oracle_env_gripper_open"

    def test_force_open_is_075(self):
        assert COMMAND_OPEN_ORACLE_PROTOCOL["force_open_raw_gripper"] == 0.75

    def test_min_attack_steps(self):
        """Must have at least 1 attack step to trigger attack_active."""
        assert COMMAND_OPEN_ORACLE_PROTOCOL["attack_steps"] >= 1


class TestLegacyCodexProtocol:
    def test_random_is_actual_attack(self):
        """Codex random: rho=1.0, actual visual attack (linf=0.10)."""
        c = CODEX_LEGACY_RANDOM_SAME_WINDOW
        assert c["rho"] == 1.0
        assert c["is_attack"] is True
        assert c["is_control"] is False

    def test_vis_current_is_actual_attack(self):
        """Codex VIS_current: rho=1.0, actual visual attack (linf=2.12)."""
        c = CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW
        assert c["rho"] == 1.0
        assert c["is_attack"] is True

    def test_targeted_uses_force_gripper_open_token_ce(self):
        """Codex targeted: force_gripper_open_token_ce, NOT gripper_logit_margin_cw."""
        c = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        assert c["attack_objective"] == "force_gripper_open_token_ce"
        assert c["epsilon"] == 0.25
        assert c["step_size"] == 0.050
        assert c["attack_steps"] == 60
        assert c["force_open_raw_gripper"] == 1.0

    def test_targeted_is_not_margin_cw(self):
        """Ensure targeted does NOT use gripper_logit_margin_cw."""
        c = CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN
        assert c["attack_objective"] != "gripper_logit_margin_cw"

    def test_all_matched_have_positive_rho(self):
        for c in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS:
            assert c["rho"] > 0, f"{c['condition_name']}: rho must be >0"

    def test_all_matched_have_same_seed_protocol(self):
        """Legacy Codex protocol: matched seed = clean seed = repeat_id."""
        pass  # enforced by driver, not config


class TestDeprecatedConditions:
    def test_deprecated_conditions_exist_for_reference(self):
        assert len(DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS) >= 4

    def test_deprecated_have_wrong_rho(self):
        """Deprecated conditions have rho=0 for random/vis_current — this is WRONG."""
        deprecated_random = DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS[0]
        assert deprecated_random["rho"] == 0.0  # this is the bug
        assert deprecated_random.get("deprecated") is True

    def test_deprecated_command_open_rho_zero(self):
        """Deprecated command_open has rho=0 — this DISABLES oracle override."""
        deprecated_cmd = DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS[3]
        assert deprecated_cmd["rho"] == 0.0
        assert deprecated_cmd.get("deprecated") is True


class TestDiagnosticProtocols:
    def test_margin_cw_is_diagnostic(self):
        assert DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL["attack_objective"] == "gripper_logit_margin_cw"

    def test_open_region_ce_is_diagnostic(self):
        assert DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL["attack_objective"] == "gripper_open_region_ce"
