import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))

from c3_g_dev_stage1 import classify_relation  # noqa: E402


SITES = {"bin_region": {"id": 1, "body_id": 10}}
BODIES = {
    "cube": {"id": 20, "parent_id": 0},
    "plate": {"id": 30, "parent_id": 0},
    "bin": {"id": 10, "parent_id": 0},
}
GEOMS = {}


def _run(predicate, obj, target, names=None):
    return classify_relation(predicate, obj, target, set(names or {obj, target}), SITES, BODIES, GEOMS)


@pytest.mark.parametrize(
    ("predicate", "target"),
    [("In", "bin_region"), ("On", "plate"), ("Stack", "plate")],
)
def test_supported_predicates_are_eligible_when_roles_resolve(predicate, target):
    result = _run(predicate, "cube", target, names={"cube", "plate"} if predicate != "In" else {"cube"})
    assert result["status"] == "PASS"
    assert result["eligible"] is True
    assert result["unknown_is_negative"] is False


def test_unknown_is_fail_closed_not_negative():
    result = _run("On", "missing_object", "plate", names={"missing_object", "plate"})
    assert result["status"] == "HOLD_UNKNOWN"
    assert result["eligible"] is False
    assert result["unknown_is_negative"] is False


def test_region_body_fallback_is_blocked():
    result = _run("In", "cube", "bin", names={"cube"})
    assert result["status"] == "HOLD_UNKNOWN"
    assert result["relation"]["target_blocked"] is True


def test_object_site_resolution_is_blocked():
    result = _run("On", "bin_region", "plate", names={"bin_region", "plate"})
    assert result["status"] == "HOLD_UNKNOWN"
    assert result["relation"]["object_blocked"] is True


def test_unsupported_predicate_rejected():
    with pytest.raises(ValueError):
        _run("Near", "cube", "plate")
