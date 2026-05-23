"""Test autowindow protocol definitions and validators — no GPU, no model."""
import sys
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.autowindow_protocols import (
    TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
    STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED,
    VALID_DETECTOR_MODES,
    REQUIRED_DETECTOR_OUTPUT_FIELDS,
    REQUIRED_PHASE_CUE_FIELDS,
)
from src.utils.autowindow_validation import (
    validate_autowindow_protocol,
    validate_no_done_based_detector_for_matched_rollout,
    validate_window_source_is_fresh_generic_autowindow,
    validate_detector_hash_present,
    validate_phase_cues_present,
    validate_generic_autowindow_outputs,
    ProtocolValidationError,
)
from src.utils.autowindow_runner import (
    build_generic_autowindow_command,
    build_generic_autowindow_command_string,
    filter_eligible_rows,
)


class TestTable1GenericAutowindowProtocol:
    def test_detector_name(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["detector_name"] == "table1_generic_autowindow_phase_cue"

    def test_window_source(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["window_source"] == "fresh_clean_generic_autowindow"

    def test_uses_attack_outcome_false(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["uses_attack_outcome"] is False

    def test_forbids_done_based_fallback(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["forbidden_fallback"] == "standard_done_minus_11_2"

    def test_requires_config_hash(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["requires_detector_config_hash"] is True

    def test_requires_detector_mode(self):
        assert TABLE1_GENERIC_AUTOWINDOW_PROTOCOL["requires_detector_mode"] is True

    def test_valid_modes_include_all_expected(self):
        for mode in ("release_intent", "preplace_cue", "near_target_late",
                      "late_carry_fallback", "eef_descent", "failed_no_signal"):
            assert mode in VALID_DETECTOR_MODES, f"missing mode: {mode}"


class TestStandardDoneBasedDeprecated:
    def test_marked_deprecated(self):
        assert STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED["deprecated"] is True

    def test_do_not_use_for_matched_rollout(self):
        assert STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED["do_not_use_for_matched_rollout"] is True

    def test_has_deprecation_reason(self):
        assert len(STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED["deprecation_reason"]) > 50


class TestValidateAutowindowProtocol:
    def test_accepts_table1_protocol(self):
        validate_autowindow_protocol(TABLE1_GENERIC_AUTOWINDOW_PROTOCOL)

    def test_rejects_deprecated_protocol(self):
        with pytest.raises(ProtocolValidationError):
            validate_autowindow_protocol(STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED)


class TestValidateNoDoneBasedDetector:
    def test_rejects_standard_done_minus_11_2(self):
        with pytest.raises(ProtocolValidationError):
            validate_no_done_based_detector_for_matched_rollout("standard_done_minus_11_2")

    def test_rejects_empty(self):
        with pytest.raises(ProtocolValidationError):
            validate_no_done_based_detector_for_matched_rollout("")

    def test_rejects_none(self):
        with pytest.raises(ProtocolValidationError):
            validate_no_done_based_detector_for_matched_rollout(None)

    def test_accepts_table1_detector_name(self):
        validate_no_done_based_detector_for_matched_rollout("table1_generic_autowindow_phase_cue")


class TestValidateWindowSource:
    def test_accepts_fresh_generic_autowindow(self):
        validate_window_source_is_fresh_generic_autowindow("fresh_clean_generic_autowindow")

    def test_rejects_table1_prior(self):
        with pytest.raises(ProtocolValidationError):
            validate_window_source_is_fresh_generic_autowindow("table1_prior_window_139_148")

    def test_rejects_empty(self):
        with pytest.raises(ProtocolValidationError):
            validate_window_source_is_fresh_generic_autowindow("")


class TestValidateDetectorHash:
    def test_accepts_valid_hash(self):
        validate_detector_hash_present("abc123def456")

    def test_rejects_empty(self):
        with pytest.raises(ProtocolValidationError):
            validate_detector_hash_present("")

    def test_rejects_none(self):
        with pytest.raises(ProtocolValidationError):
            validate_detector_hash_present(None)


class TestValidatePhaseCues:
    def test_accepts_complete_row(self):
        validate_phase_cues_present({
            "grasp_step": 10, "lift_step": 20, "carry_start_step": 25,
            "release_intent_step": 140, "done_step": 161,
        })

    def test_rejects_missing_cues(self):
        with pytest.raises(ProtocolValidationError):
            validate_phase_cues_present({"grasp_step": 10, "lift_step": 20})


class TestValidateGenericAutowindowOutputs:
    def test_accepts_valid_row(self):
        row = {
            "run_id": "test_run",
            "detector_mode": "release_intent",
            "detector_config_hash": "abc123",
            "window_detected": True,
            "auto_window_start": "100",
            "auto_window_end": "109",
            "confidence": "high",
            "clean_success": True,
            "grasp_step": 10, "lift_step": 20, "carry_start_step": 25,
            "release_intent_step": 140, "done_step": 161,
        }
        validate_generic_autowindow_outputs(row)

    def test_rejects_invalid_detector_mode(self):
        with pytest.raises(ProtocolValidationError):
            validate_generic_autowindow_outputs({
                "run_id": "test", "detector_mode": "invalid_mode",
                "detector_config_hash": "abc",
                "grasp_step": 10, "lift_step": 20, "carry_start_step": 25,
                "release_intent_step": 140, "done_step": 161,
            })

    def test_rejects_clean_failure_as_eligible(self):
        with pytest.raises(ProtocolValidationError):
            validate_generic_autowindow_outputs({
                "run_id": "test", "detector_mode": "release_intent",
                "detector_config_hash": "abc",
                "clean_success": False,
                "grasp_step": 10, "lift_step": 20, "carry_start_step": 25,
                "release_intent_step": 140, "done_step": 161,
            })

    def test_rejects_low_confidence_non_failure(self):
        with pytest.raises(ProtocolValidationError):
            validate_generic_autowindow_outputs({
                "run_id": "test", "detector_mode": "release_intent",
                "detector_config_hash": "abc",
                "confidence": "low", "clean_success": True,
                "grasp_step": 10, "lift_step": 20, "carry_start_step": 25,
                "release_intent_step": 140, "done_step": 161,
            })


class TestAutowindowRunner:
    def test_command_includes_config(self):
        cmd = build_generic_autowindow_command(
            input_root="/tmp/runs", output_csv="/tmp/out.csv",
        )
        cmd_str = " ".join(cmd)
        assert "detect_contact_window_from_clean.py" in cmd_str
        assert "--input_root /tmp/runs" in cmd_str
        assert "--output_csv /tmp/out.csv" in cmd_str
        assert "generic_autowindow_detector.yaml" in cmd_str

    def test_command_includes_phase_cues_and_summary(self):
        cmd = build_generic_autowindow_command(
            input_root="/tmp/runs", output_csv="/tmp/out.csv",
            phase_cues_csv="/tmp/cues.csv", summary_md="/tmp/summary.md",
        )
        cmd_str = " ".join(cmd)
        assert "--phase_cues_csv /tmp/cues.csv" in cmd_str
        assert "--summary_md /tmp/summary.md" in cmd_str

    def test_command_string_is_shell_safe(self):
        cmd_str = build_generic_autowindow_command_string(
            input_root="/tmp/path with spaces/runs", output_csv="/tmp/out.csv",
        )
        assert "'" in cmd_str  # spaces should be quoted

    def test_filter_eligible_rows(self):
        rows = [
            {"run_id": "r1", "window_detected": True, "clean_success": True,
             "detector_mode": "release_intent", "confidence": "high"},
            {"run_id": "r2", "window_detected": True, "clean_success": True,
             "detector_mode": "preplace_cue", "confidence": "medium"},
            {"run_id": "r3", "window_detected": False, "clean_success": True,
             "detector_mode": "failed_no_signal", "confidence": "low"},
            {"run_id": "r4", "window_detected": True, "clean_success": False,
             "detector_mode": "late_carry_fallback", "confidence": "medium"},
            {"run_id": "r5", "window_detected": True, "clean_success": True,
             "detector_mode": "release_intent", "confidence": "low"},
        ]
        eligible = filter_eligible_rows(rows, min_confidence="medium")
        assert len(eligible) == 2
        ids = {r["run_id"] for r in eligible}
        assert ids == {"r1", "r2"}


class TestPhaseCuesZeroStep:
    """Zero is a valid step value — should not be treated as missing."""
    def test_accepts_grasp_step_zero(self):
        validate_phase_cues_present({
            "grasp_step": 0, "lift_step": 5, "carry_start_step": 8,
            "release_intent_step": 20, "done_step": 30,
        })

    def test_accepts_synthetic_row_with_grasp_zero(self):
        row = {
            "run_id": "syn", "detector_config_hash": "abc",
            "detector_mode": "release_intent", "confidence": "high",
            "auto_window_start": "10", "auto_window_end": "19",
            "window_detected": True, "mechanism_type": "trajectory_transfer_candidate",
            "clean_success": True,
            "grasp_step": 0, "lift_step": 5, "carry_start_step": 8,
            "release_intent_step": 20, "done_step": 30,
        }
        validate_generic_autowindow_outputs(row)


class TestValidateAutowindowProtocolRejectsDeviations:
    def test_rejects_wrong_detector_name(self):
        with pytest.raises(ProtocolValidationError):
            validate_autowindow_protocol(dict(
                TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
                detector_name="wrong_name",
            ))

    def test_rejects_uses_attack_outcome_true(self):
        with pytest.raises(ProtocolValidationError):
            validate_autowindow_protocol(dict(
                TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
                uses_attack_outcome=True,
            ))

    def test_rejects_wrong_window_source(self):
        with pytest.raises(ProtocolValidationError):
            validate_autowindow_protocol(dict(
                TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
                window_source="table1_prior",
            ))

    def test_rejects_missing_phase_cues_requirement(self):
        with pytest.raises(ProtocolValidationError):
            validate_autowindow_protocol(dict(
                TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
                requires_phase_cues_csv=False,
            ))


class TestProtocolValidationError:
    def test_is_value_error(self):
        assert issubclass(ProtocolValidationError, ValueError)

    def test_error_uses_protocol_validation_error_not_assert(self):
        """All validators must use ProtocolValidationError, not AssertionError."""
        import inspect
        from src.utils import autowindow_validation as av
        for name, obj in inspect.getmembers(av, inspect.isfunction):
            if name.startswith("validate_"):
                src = inspect.getsource(obj)
                assert "assert " not in src, f"{name} uses assert instead of ProtocolValidationError"
