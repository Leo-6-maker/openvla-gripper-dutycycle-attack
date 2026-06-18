#!/usr/bin/env python3
"""Test SC5 schema adapter: field resolution, aliases, gripper semantics, provenance."""
import json, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_schema_adapter_v2 import (
    SC5SchemaAdapterV2, FieldProvenance, PROPRIO_13D,
    SOURCE_DIRECT, SOURCE_VECTOR_EXTRACTED, SOURCE_MISSING,
)


def test_direct_field_resolution():
    """All 13D fields resolve as direct from standard privileged artifact schema."""
    adapter = SC5SchemaAdapterV2()
    record = {
        "gripper_command": "0.3",
        "gripper_qpos": "0.05",
        "gripper_width": "0.02",
        "eef_x": "-0.15", "eef_y": "0.0", "eef_z": "0.25",
        "eef_vx": "0.01", "eef_vy": "0.0", "eef_vz": "-0.02",
        "action_dx": "0.0", "action_dy": "0.0", "action_dz": "0.0",
        "action_gripper": "0.0",
    }
    provenances = adapter.validate_record(record)
    assert adapter.all_valid(provenances), f"Missing: {adapter.missing_fields(provenances)}"
    for name in PROPRIO_13D:
        p = provenances[name]
        assert p.valid, f"{name} not valid: {p.invalid_reason}"
        assert p.source_type == SOURCE_DIRECT, f"{name} source_type={p.source_type}, expected direct"
        assert p.value is not None
    print("PASS: test_direct_field_resolution")


def test_gripper_opening_proxy_alias():
    """gripper_width and gripper_opening_proxy both resolve."""
    adapter = SC5SchemaAdapterV2()

    # gripper_width
    r1 = {"gripper_width": "0.03", "gripper_command": "0.3",
          "gripper_qpos": "0.05", "eef_x": "0", "eef_y": "0", "eef_z": "0",
          "eef_vx": "0", "eef_vy": "0", "eef_vz": "0",
          "action_dx": "0", "action_dy": "0", "action_dz": "0", "action_gripper": "0"}
    p1 = adapter.validate_record(r1)
    assert p1["gripper_opening_proxy"].valid
    assert p1["gripper_opening_proxy"].source_field == "gripper_width"
    assert abs(p1["gripper_opening_proxy"].value - 0.03) < 1e-9

    # gripper_opening_proxy
    adapter.reset()
    r2 = {"gripper_opening_proxy": "0.04", "gripper_command": "0.3",
          "gripper_qpos": "0.05", "eef_x": "0", "eef_y": "0", "eef_z": "0",
          "eef_vx": "0", "eef_vy": "0", "eef_vz": "0",
          "action_dx": "0", "action_dy": "0", "action_dz": "0", "action_gripper": "0"}
    p2 = adapter.validate_record(r2)
    assert p2["gripper_opening_proxy"].valid
    assert p2["gripper_opening_proxy"].source_field == "gripper_opening_proxy"
    print("PASS: test_gripper_opening_proxy_alias")


def test_raw_action_fallback():
    """Action fields fall back to raw_action when direct fields missing."""
    adapter = SC5SchemaAdapterV2()
    record = {
        "gripper_command": "0.3", "gripper_qpos": "0.05", "gripper_width": "0.02",
        "eef_x": "0", "eef_y": "0", "eef_z": "0",
        "eef_vx": "0", "eef_vy": "0", "eef_vz": "0",
        # action_dx/dy/dz/gripper MISSING
        "raw_action": [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.5],
    }
    provenances = adapter.validate_record(record)
    assert provenances["action_dx"].valid
    assert provenances["action_dx"].source_type == SOURCE_VECTOR_EXTRACTED
    assert abs(provenances["action_dx"].value - 0.1) < 1e-9
    assert abs(provenances["action_dy"].value - (-0.2)) < 1e-9
    assert abs(provenances["action_dz"].value - 0.3) < 1e-9
    assert abs(provenances["action_gripper"].value - 0.5) < 1e-9
    print("PASS: test_raw_action_fallback")


def test_missing_fields():
    """Missing fields are flagged, not zero-filled."""
    adapter = SC5SchemaAdapterV2()
    record = {
        "gripper_command": "0.3",
        # gripper_qpos MISSING
        "gripper_width": "0.02",
        # eef fields MISSING
        # action fields MISSING
    }
    provenances = adapter.validate_record(record)
    assert not adapter.all_valid(provenances)
    missing = adapter.missing_fields(provenances)
    assert "gripper_qpos" in missing
    assert "eef_x" in missing
    assert "action_dx" in missing

    # Verify missing fields have nan values, not zero
    values = adapter.extract_values(provenances)
    import math
    assert math.isnan(values["gripper_qpos"])
    assert math.isnan(values["eef_x"])
    assert not math.isnan(values["gripper_command"])
    print("PASS: test_missing_fields")


def test_invalid_values():
    """Empty strings, nan strings are treated as missing."""
    adapter = SC5SchemaAdapterV2()
    record = {
        "gripper_command": "",  # empty string
        "gripper_qpos": "nan",
        "gripper_width": "NaN",
        "eef_x": None,
        "eef_y": "inf",
        "eef_z": "0.25",
        "eef_vx": "0", "eef_vy": "0", "eef_vz": "0",
        "action_dx": "0", "action_dy": "0", "action_dz": "0", "action_gripper": "0",
    }
    provenances = adapter.validate_record(record)
    missing = adapter.missing_fields(provenances)
    assert "gripper_command" in missing
    assert "gripper_qpos" in missing
    assert "gripper_opening_proxy" in missing  # canonical name for gripper_width
    assert "eef_x" in missing
    assert "eef_y" in missing
    # eef_z should be valid
    assert provenances["eef_z"].valid
    print("PASS: test_invalid_values")


def test_gripper_semantics():
    """Raw/env gripper semantics validation."""
    adapter = SC5SchemaAdapterV2()

    # raw=0.3 (CLOSE), env=1.0 (CLOSE) -> OK
    r1 = {"gripper_command": "0.3", "env_action": [0, 0, 0, 0, 0, 0, 1.0]}
    s1 = adapter.validate_gripper_semantics(r1)
    assert s1["semantics_ok"]
    assert s1["raw_close"] == True
    assert s1["env_close"] == True

    # raw=0.7 (OPEN), env=-1.0 (OPEN) -> OK
    r2 = {"gripper_command": "0.7", "env_action": [0, 0, 0, 0, 0, 0, -1.0]}
    s2 = adapter.validate_gripper_semantics(r2)
    assert s2["semantics_ok"]
    assert s2["raw_close"] == False
    assert s2["env_close"] == False

    # raw=0.3 (CLOSE), env=-1.0 (OPEN) -> CONFLICT
    r3 = {"gripper_command": "0.3", "env_action": [0, 0, 0, 0, 0, 0, -1.0]}
    s3 = adapter.validate_gripper_semantics(r3)
    assert not s3["semantics_ok"]
    assert s3["conflict"]

    print("PASS: test_gripper_semantics")


def test_clean_provenance():
    """Attack markers are detected."""
    adapter = SC5SchemaAdapterV2()

    # Clean record
    r1 = {}
    p1 = adapter.validate_clean_provenance(r1)
    assert p1["clean_provenance"]

    # Attack record
    r2 = {"attack_applied": True}
    p2 = adapter.validate_clean_provenance(r2)
    assert not p2["clean_provenance"]
    assert "attack_applied" in p2["attack_flags"]

    # Detector trigger
    r3 = {"detector_trigger_now": 1}
    p3 = adapter.validate_clean_provenance(r3)
    assert not p3["clean_provenance"]

    # Manifest attack
    p4 = adapter.validate_clean_provenance({}, {"attack_type": "VIS"})
    assert not p4["clean_provenance"]

    print("PASS: test_clean_provenance")


def test_velocity_recovery():
    """EEF velocity recovered from position history via backward difference."""
    adapter = SC5SchemaAdapterV2()
    adapter.track_eef(0.0, 0.0, 0.20)
    adapter.track_eef(0.01, 0.0, 0.22)
    adapter.track_eef(0.02, 0.0, 0.25)  # current step's position

    record = {
        "gripper_command": "0.3", "gripper_qpos": "0.05", "gripper_width": "0.02",
        "eef_x": "0.02", "eef_y": "0.0", "eef_z": "0.25",
        # eef_vx, eef_vy, eef_vz MISSING
        "action_dx": "0", "action_dy": "0", "action_dz": "0", "action_gripper": "0",
    }
    provenances = adapter.validate_record(record)

    # Should recover velocities from position history
    assert provenances["eef_vx"].valid
    assert provenances["eef_vx"].source_type == "causally_derived"
    assert abs(provenances["eef_vx"].value - (0.02 - 0.01)) < 1e-9
    assert abs(provenances["eef_vz"].value - (0.25 - 0.22)) < 1e-9
    print("PASS: test_velocity_recovery")


def test_no_zero_fill():
    """Missing values return nan, never zero."""
    adapter = SC5SchemaAdapterV2()
    record = {"gripper_command": "0.3"}  # only one field
    provenances = adapter.validate_record(record)
    values = adapter.extract_values(provenances)

    import math
    # gripper_command is present
    assert not math.isnan(values["gripper_command"])
    # All others should be nan
    for name in PROPRIO_13D:
        if name != "gripper_command":
            assert math.isnan(values[name]), f"{name} should be nan, got {values[name]}"
    print("PASS: test_no_zero_fill")


def test_reset_clears_history():
    """Reset clears EEF history for new episode."""
    adapter = SC5SchemaAdapterV2()
    adapter.track_eef(0.0, 0.0, 0.20)
    adapter.track_eef(0.01, 0.0, 0.22)
    assert len(adapter._eef_history) == 2
    adapter.reset()
    assert len(adapter._eef_history) == 0
    print("PASS: test_reset_clears_history")


if __name__ == '__main__':
    test_direct_field_resolution()
    test_gripper_opening_proxy_alias()
    test_raw_action_fallback()
    test_missing_fields()
    test_invalid_values()
    test_gripper_semantics()
    test_clean_provenance()
    test_velocity_recovery()
    test_no_zero_fill()
    test_reset_clears_history()
    print("\nAll SC5 schema adapter tests passed.")
