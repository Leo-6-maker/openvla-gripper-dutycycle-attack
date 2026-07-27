import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import verify_v23_recorded_geometry_independent as verifier


def _quat_z(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def test_q_and_neg_q_are_identical():
    q = _quat_z(0.4)
    assert verifier.quat_distance(q, [-x for x in q]) == 0.0


@pytest.mark.parametrize("angle", [1e-10, 1e-8, 1e-7])
def test_stable_small_rotation_is_finite_and_ordered(angle):
    value = verifier.quat_distance(_quat_z(0.0), _quat_z(angle))
    assert math.isfinite(value)
    assert value == pytest.approx(angle, abs=1e-15)


def test_non_unit_quaternions_are_normalized():
    assert verifier.quat_distance([2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) == 0.0


@pytest.mark.parametrize("bad", [[math.nan, 0.0, 0.0, 0.0], [math.inf, 0.0, 0.0, 0.0]])
def test_nonfinite_quaternion_fails_closed(bad):
    with pytest.raises(verifier.ReviewHold):
        verifier.quat_distance(bad, [1.0, 0.0, 0.0, 0.0])


def test_near_identity_does_not_use_acos_conditioning():
    value = verifier.quat_distance([1.0, 1e-12, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert math.isfinite(value)
    assert value >= 0.0
