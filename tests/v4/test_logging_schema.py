import pytest
from gripper_attack.logging_schema import RUN_FIELDS, validate_step_record, validate_run_manifest

def test_missing_step_raises():
    with pytest.raises(ValueError):
        validate_step_record({})

def test_manifest_required_ok():
    validate_run_manifest({k: "" for k in RUN_FIELDS})
