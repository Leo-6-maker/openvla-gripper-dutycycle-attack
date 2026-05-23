"""Autowindow protocol definitions — canonical detector protocols.

Separates the Table1 generic phase-cue autowindow detector from the
deprecated standard done-based formula.  All future matched rollouts
MUST use the generic autowindow detector.

IMPORTANT: The generic autowindow detector consumes clean step_records
only.  It MUST NOT be tuned using attack outcome.
"""

# ============================================================
# Table1 Generic Autowindow — canonical protocol
# ============================================================
TABLE1_GENERIC_AUTOWINDOW_PROTOCOL = {
    "detector_name": "table1_generic_autowindow_phase_cue",
    "detector_script": "scripts/detect_contact_window_from_clean.py",
    "detector_config": "configs/generic_autowindow_detector.yaml",
    "window_source": "fresh_clean_generic_autowindow",
    "input": "clean_step_records_only",
    "uses_attack_outcome": False,
    "window_len": 10,
    "primary_anchor_order": [
        "release_intent",
        "eef_descent",
        "near_target_supported",
        "near_target_late",
        "late_carry_fallback",
    ],
    "forbidden_fallback": "standard_done_minus_11_2",
    "requires_detector_config_hash": True,
    "requires_detector_mode": True,
    "requires_phase_cues_csv": True,
    "notes": (
        "Window formula: start = selected_preplace_step - window_len, "
        "end = selected_preplace_step - 1. "
        "selected_preplace_step is chosen from the primary anchor order "
        "(release_intent > eef_descent > near_target_supported > near_target_late). "
        "late_carry_fallback uses (lift_step + late_offset_ratio * (done_step - lift_step)) "
        "as the window center."
    ),
}

# ============================================================
# Deprecated: standard done-based formula
# ============================================================
STANDARD_DONE_BASED_AUTOWINDOW_DEPRECATED = {
    "detector_name": "standard_done_minus_11_2",
    "deprecated": True,
    "deprecation_reason": (
        "Does not match Table1 generic_autowindow detector. "
        "Produced post-release / gripper-open-throughout windows on all "
        "true non-BB relay candidates (2026-05-23). "
        "Table1 used the generic_autowindow phase-cue detector, not this formula."
    ),
    "do_not_use_for_matched_rollout": True,
}

# ============================================================
# Valid detector modes
# ============================================================
VALID_DETECTOR_MODES = frozenset({
    "release_intent",
    "preplace_cue",
    "near_target_late",
    "late_carry_fallback",
    "eef_descent",
    "failed_no_signal",
})

# ============================================================
# Required output fields per candidate row
# ============================================================
REQUIRED_DETECTOR_OUTPUT_FIELDS = frozenset({
    "run_id",
    "detector_config_hash",
    "detector_mode",
    "confidence",
    "auto_window_start",
    "auto_window_end",
    "window_detected",
    "mechanism_type",
})

REQUIRED_PHASE_CUE_FIELDS = frozenset({
    "grasp_step",
    "lift_step",
    "carry_start_step",
    "release_intent_step",
    "done_step",
})
