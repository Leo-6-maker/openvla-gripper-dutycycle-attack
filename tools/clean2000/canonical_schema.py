"""CLEAN2000 canonical episode schema — source-format-independent contract.

This module defines the canonical fields that every episode must populate,
regardless of whether it originates from CLEAN1500 or Object500 collectors.

Design principle: the canonical index extracts only scientifically common fields.
Attack/detector runtime fields from Object500 are preserved in source_provenance
but do NOT become required canonical columns.
"""

CANONICAL_FIELDS = {
    # ── Identity ──
    "episode_key": str,           # {parent_key}/clean/attempt_{n:02d}
    "parent_key": str,            # {suite}/task_{task_id:02d}/state_{state_id:03d}
    "suite": str,                 # libero_object | libero_spatial | libero_goal | libero_10
    "task_id": int,               # 0-9
    "task_name": str,
    "state_id": int,              # 0-49
    "eval_seed": int,
    "attempt_id": int,
    "source_format": str,         # "clean1500_v1" | "object500_v1"
    "source_root": str,           # absolute path to episode directory

    # ── Clean state ──
    "condition": str,             # MUST be "CLEAN"
    "task_success": bool,
    "gate_pass": bool,
    "complete": bool,             # COMPLETE.json exists
    "teacher_eligible": bool,
    "mechanism_eligible": bool,
    "abstain_reason": str,        # "" if eligible

    # ── Telemetry completeness ──
    "n_steps": int,
    "n_telemetry_rows": int,
    "first_valid_step": int,      # -1 if none
    "n_valid_steps": int,
    "invalid_feature_steps": int,
    "step_index_contiguous": bool,
    "valid_steps_contiguous": bool,
    "duplicate_step_count": int,
    "missing_step_count": int,

    # ── Feature schema ──
    "feature_schema_id": str,     # "SC5_PROPRIO_NO_STEP_V1" or similar
    "feature_schema_sha256": str,

    # ── Provenance ──
    "episode_summary_sha256": str,
    "step_telemetry_sha256": str,
    "privileged_records_sha256": str,
    "complete_marker_sha256": str,
    "artifact_inventory_sha256": str,
    "collector_code_sha256": str,
    "source_manifest_sha256": str,
    "observed_initial_state_sha256": str,
}

# Fields validated by validate_clean2000_corpus.py
IDENTITY_FIELDS = [
    "episode_key", "parent_key", "suite", "task_id", "task_name",
    "state_id", "eval_seed", "attempt_id", "source_format", "source_root",
]

CLEAN_STATE_FIELDS = [
    "condition", "task_success", "gate_pass", "complete",
    "teacher_eligible", "mechanism_eligible", "abstain_reason",
]

TELEMETRY_FIELDS = [
    "n_steps", "n_telemetry_rows", "first_valid_step", "n_valid_steps",
    "invalid_feature_steps", "step_index_contiguous", "valid_steps_contiguous",
    "duplicate_step_count", "missing_step_count",
]

FEATURE_FIELDS = [
    "feature_schema_id", "feature_schema_sha256",
]

PROVENANCE_FIELDS = [
    "episode_summary_sha256", "step_telemetry_sha256",
    "privileged_records_sha256", "complete_marker_sha256",
    "artifact_inventory_sha256", "collector_code_sha256",
    "source_manifest_sha256", "observed_initial_state_sha256",
]

VALID_SUITES = frozenset({
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10",
})

REQUIRED_CONDITION = "CLEAN"

# Suite-specific expected counts for CLEAN2000
EXPECTED_PER_SUITE = {
    "libero_object": 500,
    "libero_spatial": 500,
    "libero_goal": 500,
    "libero_10": 500,
}
EXPECTED_TOTAL = 2000

# Valid task_id range
TASK_ID_RANGE = (0, 9)
# Valid state_id range
STATE_ID_RANGE = (0, 49)

# De-normalized suite names for cross-referencing
SUITE_ALIASES = {
    "object": "libero_object",
    "libero_object": "libero_object",
}
