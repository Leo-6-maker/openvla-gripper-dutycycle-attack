"""Test protocol_validation fail-fast guards — no GPU, no model."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.task_identity import (
    MATCHED_CONDITIONS,
    TRIAGE_MATCHED_CONDITIONS,
)
from src.utils.condition_protocols import (
    COMMAND_OPEN_ORACLE_PROTOCOL,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
    CODEX_LEGACY_RANDOM_SAME_WINDOW,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
)
from src.utils.protocol_validation import (
    ProtocolValidationError,
    validate_command_open_protocol,
    validate_same_seed_protocol,
    validate_window_source,
    validate_codex_targeted_protocol,
    validate_condition_config_schema,
)


# ── Fail-fast deprecated aliases ──────────────────────────────────────
class TestDeprecatedMatchedConditionsFailFast:
    def test_iter_raises(self):
        with pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5"):
            list(MATCHED_CONDITIONS)

    def test_getitem_raises(self):
        with pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5"):
            _ = MATCHED_CONDITIONS[0]

    def test_len_raises(self):
        with pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5"):
            len(MATCHED_CONDITIONS)

    def test_bool_raises(self):
        with pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5"):
            _ = bool(MATCHED_CONDITIONS)

    def test_triage_also_raises(self):
        with pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5"):
            list(TRIAGE_MATCHED_CONDITIONS)


# ── command_open validation ───────────────────────────────────────────
class TestCommandOpenRhoZeroRejected:
    def test_valid_protocol_passes(self):
        validate_command_open_protocol(COMMAND_OPEN_ORACLE_PROTOCOL)

    def test_rho_zero_rejected(self):
        broken = dict(COMMAND_OPEN_ORACLE_PROTOCOL, rho=0.0)
        with pytest.raises(ProtocolValidationError, match="disables oracle override"):
            validate_command_open_protocol(broken)

    def test_wrong_objective_rejected(self):
        broken = dict(COMMAND_OPEN_ORACLE_PROTOCOL, attack_objective="wrong")
        with pytest.raises(ProtocolValidationError, match="oracle_env_gripper_open"):
            validate_command_open_protocol(broken)

    def test_missing_env_extra_rejected(self):
        broken = {k: v for k, v in COMMAND_OPEN_ORACLE_PROTOCOL.items()
                  if k != "env_extra"}
        with pytest.raises(ProtocolValidationError, match="env_extra"):
            validate_command_open_protocol(broken)

    def test_missing_oracle_env_value_rejected(self):
        broken = dict(COMMAND_OPEN_ORACLE_PROTOCOL,
                      env_extra={"V4_ORACLE_GRIPPER_PATTERN": "continuous_open"})
        with pytest.raises(ProtocolValidationError, match="V4_ORACLE_FORCE_GRIPPER_ENV_VALUE"):
            validate_command_open_protocol(broken)

    def test_attack_steps_zero_rejected(self):
        broken = dict(COMMAND_OPEN_ORACLE_PROTOCOL, attack_steps=0)
        with pytest.raises(ProtocolValidationError, match="attack_steps"):
            validate_command_open_protocol(broken)


# ── same-seed protocol ────────────────────────────────────────────────
class TestSameSeedProtocolRequired:
    def test_matching_seeds_pass(self):
        validate_same_seed_protocol(5, 5)

    def test_mismatched_seeds_raise(self):
        with pytest.raises(ProtocolValidationError, match="Seed drift"):
            validate_same_seed_protocol(105, 5)


# ── window source validation ──────────────────────────────────────────
class TestTable1PriorWindowRejected:
    def test_table1_prior_rejected(self):
        with pytest.raises(ProtocolValidationError, match="table1_prior_window"):
            validate_window_source("table1_prior_window_139_148")

    def test_clean_detect_source_accepted(self):
        validate_window_source("fresh_clean_detect_autowindow_144_153")


# ── Codex targeted protocol exact check ───────────────────────────────
class TestCodexTargetedProtocolExact:
    def test_correct_protocol_passes(self):
        validate_codex_targeted_protocol(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN)

    def test_wrong_epsilon_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN, epsilon=0.10)
        with pytest.raises(ProtocolValidationError, match="epsilon"):
            validate_codex_targeted_protocol(broken)

    def test_wrong_step_size_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN, step_size=0.02)
        with pytest.raises(ProtocolValidationError, match="step_size"):
            validate_codex_targeted_protocol(broken)

    def test_wrong_attack_steps_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN, attack_steps=20)
        with pytest.raises(ProtocolValidationError, match="attack_steps"):
            validate_codex_targeted_protocol(broken)

    def test_wrong_objective_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
                      attack_objective="gripper_logit_margin_cw")
        with pytest.raises(ProtocolValidationError, match="force_gripper_open_token_ce"):
            validate_codex_targeted_protocol(broken)

    def test_wrong_fo_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN, force_open_raw_gripper=0.75)
        with pytest.raises(ProtocolValidationError, match="force_open_raw_gripper"):
            validate_codex_targeted_protocol(broken)


# ── Schema validation ─────────────────────────────────────────────────
class TestConditionConfigSchema:
    def test_all_legacy_codex_pass_schema(self):
        for c in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS:
            validate_condition_config_schema(c)

    def test_missing_field_rejected(self):
        broken = {"condition_name": "test"}
        with pytest.raises(ProtocolValidationError, match="Missing required fields"):
            validate_condition_config_schema(broken)

    def test_missing_recommended_field_rejected(self):
        broken = dict(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN)
        del broken["force_open_raw_gripper"]
        with pytest.raises(ProtocolValidationError, match="Missing recommended fields"):
            validate_condition_config_schema(broken)


# ── attack_objective=None CLI-arg semantics ───────────────────────────
class TestNoneObjectiveOmitsCliFlag:
    def test_random_has_omit_attack_objective_cli_arg(self):
        assert CODEX_LEGACY_RANDOM_SAME_WINDOW["attack_objective"] is None
        assert CODEX_LEGACY_RANDOM_SAME_WINDOW["attack_objective_raw_arg"] is None
        assert CODEX_LEGACY_RANDOM_SAME_WINDOW["omit_attack_objective_cli_arg"] is True

    def test_none_means_omit_not_string_none(self):
        """attack_objective=None means the --attack_objective CLI flag is OMITTED,
        not passed as the string 'None' or empty string."""
        c = CODEX_LEGACY_RANDOM_SAME_WINDOW
        assert c["attack_objective"] is not None or c["omit_attack_objective_cli_arg"] is True
        # The driver must check omit_attack_objective_cli_arg and skip the flag
        assert c["effective_attack_objective_expected"] != "None"
        assert c["effective_attack_objective_expected"] != ""
