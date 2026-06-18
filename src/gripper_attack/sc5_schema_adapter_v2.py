#!/usr/bin/env python3
"""SC5 Canonical Schema Adapter v2 — validates and maps step records to 25D canonical features.

Reuses: sc5_streaming_features_v2.FEATURE_NAMES (canonical 25D ordering).
New: field-level provenance tracking, source_type classification,
     gripper semantics validation, velocity recovery from position history.

Prohibited (fail-closed):
  - missing-to-zero fill
  - future-row features
  - attacked action as clean action
  - normalized_step or absolute episode position
  - object pose, teacher anchor, task/state/run identity as student input
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple

# Canonical 25D feature names (frozen — matches sc5_streaming_features_v2.FEATURE_NAMES)
CANONICAL_25D = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

# 13D direct/vector features
PROPRIO_13D = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

# source_type enum
SOURCE_DIRECT = "direct"
SOURCE_VECTOR_EXTRACTED = "vector_extracted"
SOURCE_CAUSALLY_DERIVED = "causally_derived"
SOURCE_MISSING = "missing"
SOURCE_AMBIGUOUS = "ambiguous"

# Field aliases mapping source field names -> canonical names
# Ordered by priority: first match wins
FIELD_ALIASES: Dict[str, List[str]] = {
    "gripper_command": ["gripper_command"],
    "gripper_qpos": ["gripper_qpos"],
    "gripper_opening_proxy": ["gripper_width", "gripper_opening_proxy"],
    "eef_x": ["eef_x"],
    "eef_y": ["eef_y"],
    "eef_z": ["eef_z"],
    "eef_vx": ["eef_vx"],
    "eef_vy": ["eef_vy"],
    "eef_vz": ["eef_vz"],
    "action_dx": ["action_dx"],
    "action_dy": ["action_dy"],
    "action_dz": ["action_dz"],
    "action_gripper": ["action_gripper"],
}

# Fields that can be recovered from raw_action vector
RAW_ACTION_INDEX: Dict[str, int] = {
    "action_dx": 0,
    "action_dy": 1,
    "action_dz": 2,
    "action_gripper": 6,
}

# Gripper semantics (frozen)
# raw_gripper <= 0.5 -> CLOSE intent
# env_gripper > 0    -> CLOSE command


class FieldProvenance:
    """Provenance record for a single canonical field."""

    __slots__ = ("canonical_name", "source_field", "source_type",
                 "conversion", "unit", "valid", "invalid_reason", "value")

    def __init__(self, canonical_name: str, source_field: str = "",
                 source_type: str = SOURCE_MISSING, conversion: str = "",
                 unit: str = "", valid: bool = False,
                 invalid_reason: str = "", value=None):
        self.canonical_name = canonical_name
        self.source_field = source_field
        self.source_type = source_type
        self.conversion = conversion
        self.unit = unit
        self.valid = valid
        self.invalid_reason = invalid_reason
        self.value = value

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "source_field": self.source_field,
            "source_type": self.source_type,
            "conversion": self.conversion,
            "unit": self.unit,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


class SC5SchemaAdapterV2:
    """Validates step records against canonical 25D schema.

    Fail-closed: missing/ambiguous fields are flagged, not zero-filled.
    No future-row access. No task/state/run identity as features.
    """

    def __init__(self):
        self._eef_history: List[Dict[str, float]] = []
        self._expected_step: Optional[int] = None

    def reset(self):
        """Reset per-episode state."""
        self._eef_history = []
        self._expected_step = None

    def validate_record(self, step_record: dict) -> Dict[str, FieldProvenance]:
        """Validate one step record against canonical schema.

        Returns {canonical_name: FieldProvenance} for all 13D proprio fields.
        Does NOT compute causally-derived features (that's the streaming adapter's job).
        """
        provenances = {}

        for canonical_name in PROPRIO_13D:
            prov = self._resolve_field(canonical_name, step_record)
            provenances[canonical_name] = prov

        return provenances

    def _resolve_field(self, canonical_name: str, step_record: dict) -> FieldProvenance:
        """Resolve a single field from step record using alias chain."""
        aliases = FIELD_ALIASES.get(canonical_name, [canonical_name])

        # Try direct aliases first
        for alias in aliases:
            if alias in step_record:
                raw_val = step_record[alias]
                if self._is_valid_value(raw_val):
                    val = float(raw_val)
                    return FieldProvenance(
                        canonical_name=canonical_name,
                        source_field=alias,
                        source_type=SOURCE_DIRECT,
                        conversion="identity",
                        unit=self._unit_for(canonical_name),
                        valid=True,
                        value=val,
                    )

        # Try raw_action recovery for action fields
        if canonical_name in RAW_ACTION_INDEX and "raw_action" in step_record:
            raw_action = step_record["raw_action"]
            if isinstance(raw_action, (list, tuple)) and len(raw_action) > RAW_ACTION_INDEX[canonical_name]:
                val = float(raw_action[RAW_ACTION_INDEX[canonical_name]])
                return FieldProvenance(
                    canonical_name=canonical_name,
                    source_field="raw_action",
                    source_type=SOURCE_VECTOR_EXTRACTED,
                    conversion=f"raw_action[{RAW_ACTION_INDEX[canonical_name]}]",
                    unit=self._unit_for(canonical_name),
                    valid=True,
                    value=val,
                )

        # Velocity recovery from EEF position history
        if canonical_name in ("eef_vx", "eef_vy", "eef_vz") and len(self._eef_history) >= 2:
            axis = {"eef_vx": "x", "eef_vy": "y", "eef_vz": "z"}[canonical_name]
            p_curr = self._eef_history[-1].get(axis)
            p_prev = self._eef_history[-2].get(axis)
            if p_curr is not None and p_prev is not None:
                val = p_curr - p_prev  # assume dt=1
                return FieldProvenance(
                    canonical_name=canonical_name,
                    source_field=f"eef_{axis}",
                    source_type=SOURCE_CAUSALLY_DERIVED,
                    conversion=f"backward_difference(eef_{axis}, window=2)",
                    unit="meters/step",
                    valid=True,
                    value=val,
                )

        # Missing
        return FieldProvenance(
            canonical_name=canonical_name,
            source_type=SOURCE_MISSING,
            valid=False,
            invalid_reason=f"no_valid_source: tried {aliases}",
        )

    def all_valid(self, provenances: Dict[str, FieldProvenance]) -> bool:
        """Check if all 13D proprio fields are valid."""
        return all(p.valid for p in provenances.values())

    def missing_fields(self, provenances: Dict[str, FieldProvenance]) -> List[str]:
        """Return list of canonical names with invalid provenance."""
        return [name for name, p in provenances.items() if not p.valid]

    def extract_values(self, provenances: Dict[str, FieldProvenance]) -> Dict[str, float]:
        """Extract validated float values. Returns dict with NaN for invalid fields (never zero)."""
        return {name: (p.value if p.valid else float('nan'))
                for name, p in provenances.items()}

    def track_eef(self, eef_x: float, eef_y: float, eef_z: float):
        """Track EEF position for velocity recovery."""
        self._eef_history.append({"x": eef_x, "y": eef_y, "z": eef_z})

    def validate_gripper_semantics(self, step_record: dict) -> dict:
        """Validate gripper raw/env semantics consistency.

        Returns dict with:
          - semantics_ok: bool
          - raw_gripper: float
          - env_gripper: float
          - raw_close: bool (raw <= 0.5)
          - env_close: bool (env > 0)
          - conflict: bool (raw_close != env_close)
        """
        raw_gripper = step_record.get("gripper_command", None)
        if raw_gripper is None or raw_gripper == "" or raw_gripper == "nan":
            return {"semantics_ok": False, "error": "missing_gripper_command",
                    "raw_gripper": float('nan'), "env_gripper": float('nan')}

        raw_gripper = float(raw_gripper)
        raw_close = raw_gripper <= 0.5

        # Try to get env_gripper from env_action
        env_action = step_record.get("env_action", None)
        env_gripper = float('nan')
        env_close = None

        if isinstance(env_action, (list, tuple)) and len(env_action) >= 7:
            env_gripper = float(env_action[6])
            env_close = env_gripper > 0

        if env_close is None:
            return {"semantics_ok": False, "error": "missing_env_gripper",
                    "raw_gripper": raw_gripper, "raw_close": raw_close}

        conflict = raw_close != env_close

        return {
            "semantics_ok": not conflict,
            "raw_gripper": raw_gripper,
            "env_gripper": env_gripper,
            "raw_close": raw_close,
            "env_close": env_close,
            "conflict": conflict,
        }

    def validate_clean_provenance(self, step_record: dict, manifest: dict = None) -> dict:
        """Validate clean provenance: no attack, no intervention, clean action only.

        Returns dict with:
          - clean_provenance: bool
          - attack_flags: list of detected attack indicators
        """
        flags = []

        # Check for attack markers
        if step_record.get("attack_applied", False) in (True, "True", "true", 1, "1"):
            flags.append("attack_applied")

        if step_record.get("attack_condition", "") not in ("", "none", "clean", None):
            flags.append(f"attack_condition={step_record.get('attack_condition')}")

        if "attacked_env_action" in step_record:
            flags.append("attacked_env_action_present")

        if step_record.get("detector_trigger_now", False) in (True, "True", "true", 1, "1"):
            flags.append("detector_trigger_active")

        if step_record.get("oracle_attack", False) in (True, "True", "true", 1, "1"):
            flags.append("oracle_attack")

        # Check manifest
        if manifest:
            if manifest.get("attack_type", "") not in ("", "none", "clean", None):
                flags.append(f"manifest_attack_type={manifest.get('attack_type')}")
            if manifest.get("intervention", "") not in ("", "none", None):
                flags.append("manifest_intervention")

        return {
            "clean_provenance": len(flags) == 0,
            "attack_flags": flags,
        }

    @staticmethod
    def _is_valid_value(v) -> bool:
        """Check if value is a valid float (not None, empty string, nan, inf)."""
        if v is None:
            return False
        if isinstance(v, bool):
            return False
        if isinstance(v, str):
            if v.strip() in ("", "nan", "NaN", "NAN", "inf", "-inf", "Infinity"):
                return False
            try:
                float(v)
                return True
            except (ValueError, TypeError):
                return False
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False
            return True
        return False

    @staticmethod
    def _unit_for(canonical_name: str) -> str:
        """Return unit string for a canonical field."""
        units = {
            "gripper_command": "normalized [0,1]",
            "gripper_qpos": "radians",
            "gripper_opening_proxy": "meters",
            "eef_x": "meters",
            "eef_y": "meters",
            "eef_z": "meters",
            "eef_vx": "meters/second",
            "eef_vy": "meters/second",
            "eef_vz": "meters/second",
            "action_dx": "normalized",
            "action_dy": "normalized",
            "action_dz": "normalized",
            "action_gripper": "normalized [-1,1]",
        }
        return units.get(canonical_name, "unknown")


def build_field_source_audit(episodes: List[dict]) -> List[dict]:
    """Build per-field source audit across all episodes.

    Returns list of {canonical_name, n_episodes, n_valid, n_missing,
                       n_direct, n_vector_extracted, n_causally_derived}
    """
    from collections import Counter
    stats = {name: Counter() for name in PROPRIO_13D}

    for ep in episodes:
        for name in PROPRIO_13D:
            stats[name]["episodes"] += 1

    # This is a stub — actual audit requires processing all rows.
    # The full audit runs in build_sc5_canonical_corpus_v2.py.
    return []
