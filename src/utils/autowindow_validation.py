"""Autowindow protocol validation — fail-fast guards.

All validators use explicit ``if / raise ProtocolValidationError``,
NOT ``assert``, so they survive ``python -O``.
"""

from src.utils.protocol_validation import ProtocolValidationError

from src.utils.autowindow_protocols import (
    TABLE1_GENERIC_AUTOWINDOW_PROTOCOL,
    VALID_DETECTOR_MODES,
    REQUIRED_DETECTOR_OUTPUT_FIELDS,
    REQUIRED_PHASE_CUE_FIELDS,
)


def validate_autowindow_protocol(protocol):
    """Validate an autowindow protocol dict against the Table1 baseline.

    Checks all canonical field values — not just deprecation status.
    Any deviation from the Table1 generic autowindow protocol raises
    ProtocolValidationError.
    """
    name = protocol.get("detector_name", "unknown")

    # Reject deprecated protocols first
    if protocol.get("deprecated"):
        raise ProtocolValidationError(
            f"autowindow protocol {name} is deprecated: {protocol.get('deprecation_reason')}"
        )

    # Canonical field checks
    if protocol.get("detector_name") != "table1_generic_autowindow_phase_cue":
        raise ProtocolValidationError(
            f"expected detector_name='table1_generic_autowindow_phase_cue', "
            f"got {protocol.get('detector_name')!r}"
        )
    if protocol.get("detector_script") != "scripts/detect_contact_window_from_clean.py":
        raise ProtocolValidationError(
            f"expected detector_script='scripts/detect_contact_window_from_clean.py', "
            f"got {protocol.get('detector_script')!r}"
        )
    if protocol.get("detector_config") != "configs/generic_autowindow_detector.yaml":
        raise ProtocolValidationError(
            f"expected detector_config='configs/generic_autowindow_detector.yaml', "
            f"got {protocol.get('detector_config')!r}"
        )
    if protocol.get("window_source") != "fresh_clean_generic_autowindow":
        raise ProtocolValidationError(
            f"expected window_source='fresh_clean_generic_autowindow', "
            f"got {protocol.get('window_source')!r}"
        )
    if protocol.get("input") != "clean_step_records_only":
        raise ProtocolValidationError(
            f"expected input='clean_step_records_only', "
            f"got {protocol.get('input')!r}"
        )
    if protocol.get("uses_attack_outcome") is not False:
        raise ProtocolValidationError(
            f"uses_attack_outcome must be False, got {protocol.get('uses_attack_outcome')!r}"
        )
    if protocol.get("requires_detector_config_hash") is not True:
        raise ProtocolValidationError(
            f"requires_detector_config_hash must be True"
        )
    if protocol.get("requires_detector_mode") is not True:
        raise ProtocolValidationError(
            f"requires_detector_mode must be True"
        )
    if protocol.get("requires_phase_cues_csv") is not True:
        raise ProtocolValidationError(
            f"requires_phase_cues_csv must be True"
        )
    if "standard_done_minus_11_2" not in str(protocol.get("forbidden_fallback", "")):
        raise ProtocolValidationError(
            f"forbidden_fallback must include 'standard_done_minus_11_2', "
            f"got {protocol.get('forbidden_fallback')!r}"
        )


def validate_no_done_based_detector_for_matched_rollout(detector_name):
    """Reject the deprecated standard done-based detector for matched rollouts."""
    if detector_name in (None, ""):
        raise ProtocolValidationError("detector_name must not be empty for matched rollout")
    if detector_name == "standard_done_minus_11_2":
        raise ProtocolValidationError(
            "standard_done_minus_11_2 is DEPRECATED for matched rollouts. "
            "Use table1_generic_autowindow_phase_cue instead. "
            "The done-based formula produced post-release windows on all "
            "true non-BB relay candidates."
        )


def validate_window_source_is_fresh_generic_autowindow(window_source):
    """Reject table1_prior_window or non-fresh window sources for matched rollouts."""
    if window_source in (None, ""):
        raise ProtocolValidationError("window_source must not be empty")
    if "table1_prior" in str(window_source).lower():
        raise ProtocolValidationError(
            "table1_prior_window is PROVENANCE ONLY. "
            "Matched rollouts MUST use fresh_clean_generic_autowindow "
            "from the current clean_detect run."
        )
    if "table1" in str(window_source).lower() and "prior" in str(window_source).lower():
        raise ProtocolValidationError(
            "table1_prior_window is PROVENANCE ONLY, not a rollout input."
        )


def validate_detector_hash_present(detector_config_hash):
    """Reject empty or missing detector config hash."""
    if detector_config_hash in (None, ""):
        raise ProtocolValidationError(
            "detector_config_hash must not be empty. "
            "The generic autowindow detector must record its config hash."
        )


def validate_phase_cues_present(row):
    """Reject rows with missing required phase cue fields.

    Only None and empty string are treated as missing. Zero (0) is a
    valid step value — e.g. grasp_step=0 indicates grasp on the first
    observation frame.
    """
    missing = REQUIRED_PHASE_CUE_FIELDS - set(k for k, v in row.items() if v not in (None, ""))
    if missing:
        raise ProtocolValidationError(
            f"missing phase cue fields: {sorted(missing)}"
        )


def validate_generic_autowindow_outputs(row):
    """Validate a generic autowindow detector output row.

    Checks: detector_mode valid, clean_success gating, hash present,
    window_detected consistency, confidence level.
    """
    run_id = row.get("run_id", "unknown")

    # detector_mode must be a valid value
    mode = row.get("detector_mode", "")
    if mode not in VALID_DETECTOR_MODES:
        raise ProtocolValidationError(
            f"{run_id}: invalid detector_mode {mode!r}. "
            f"Must be one of {sorted(VALID_DETECTOR_MODES)}"
        )

    # config hash required
    validate_detector_hash_present(row.get("detector_config_hash", ""))

    # phase cues required
    validate_phase_cues_present(row)

    # window_detected must be boolean
    wd = row.get("window_detected")
    if isinstance(wd, str):
        wd = wd.strip().lower() in ("true", "1", "yes")
    if wd is True:
        if not row.get("auto_window_start") or not row.get("auto_window_end"):
            raise ProtocolValidationError(
                f"{run_id}: window_detected=True but auto_window_start/end missing"
            )

    # clean_success=false rows cannot be matched-rollout eligible
    clean_success = row.get("clean_success")
    if isinstance(clean_success, str):
        clean_success = clean_success.strip().lower() in ("true", "1", "yes")
    if clean_success is False and mode != "failed_no_signal":
        raise ProtocolValidationError(
            f"{run_id}: clean_success=False but detector_mode={mode}. "
            "Clean-failed runs cannot be eligible for matched rollout."
        )

    # low confidence cannot be command-open eligible
    confidence = row.get("confidence", "")
    if str(confidence).lower() == "low" and mode != "failed_no_signal":
        raise ProtocolValidationError(
            f"{run_id}: low confidence with mode={mode}. "
            "Low-confidence detections cannot be command-open eligible "
            "unless explicitly diagnostic."
        )
