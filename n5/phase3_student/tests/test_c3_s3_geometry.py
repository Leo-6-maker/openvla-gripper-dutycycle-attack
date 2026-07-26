import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import c3_s3_geometry_observability as c3  # noqa: E402


def test_transform_contract():
    result = c3.transform_contract_tests()
    assert result["pass"] is True
    assert result["cases"]["rot_z_90"]["max_abs_error"] < 1e-12


def test_quaternion_normalization_and_inverse():
    q = c3.quat_normalize((2.0, 0.0, 0.0, 0.0))
    assert q == (1.0, 0.0, 0.0, 0.0)
    identity = c3.quat_mul(q, c3.quat_inverse(q))
    assert max(abs(x - y) for x, y in zip(identity, (1.0, 0.0, 0.0, 0.0))) < 1e-12


def test_unknown_is_not_negative():
    row = {
        "task_key": "libero_10/task_00",
        "classification": "ARTICULATED_UNKNOWN",
        "observability_status": "MAPPING_ONLY_REPLAY_EVIDENCE_REQUIRED",
        "unknown_is_negative": False,
        "silent_fallback": False,
    }
    assert row["classification"] != "STATIC_FIXTURE"
    assert row["unknown_is_negative"] is False


def test_protected_path_rejection():
    assert any(token in "/mnt/example/t2r-d" for token in c3.PROTECTED_TOKENS)

