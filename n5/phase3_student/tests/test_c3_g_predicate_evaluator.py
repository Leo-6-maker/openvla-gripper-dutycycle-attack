import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))

from c3_g_predicate_evaluator import evaluate_case, load_contract  # noqa: E402


CONTRACT = load_contract()


def _case(predicate="In", object_pos=(0.0, 0.0, 0.0), target_pos=(0.0, 0.0, 0.0), target_quat=(1.0, 0.0, 0.0, 0.0), target_role="REGION_TARGET"):
    return {
        "episode_id": "e0",
        "step": 0,
        "predicate": predicate,
        "object": {
            "id": "object_0",
            "role": "MANIPULATED_OBJECT",
            "pose": {"pos": list(object_pos), "quat": [1.0, 0.0, 0.0, 0.0]},
            "half_extents": [0.1, 0.1, 0.1],
        },
        "target": {
            "id": "target_0",
            "role": target_role,
            "pose": {"pos": list(target_pos), "quat": list(target_quat)},
            "half_extents": [1.0, 1.0, 1.0],
            "stackable": True,
        },
        "expected_identity": {"episode_id": "e0", "step": 0, "object_id": "object_0", "target_id": "target_0"},
    }


@pytest.mark.parametrize("predicate,target_role", [("In", "REGION_TARGET"), ("On", "REGION_TARGET"), ("On", "OBJECT_TARGET"), ("Stack", "OBJECT_TARGET")])
def test_true_geometry(predicate, target_role):
    position = (0.0, 0.0, 1.1) if predicate in {"On", "Stack"} else (0.0, 0.0, 0.0)
    assert evaluate_case(_case(predicate, object_pos=position, target_role=target_role), CONTRACT)["value"] == "TRUE"


def test_coordinate_transform_and_q_sign_equivalence():
    half = math.sqrt(0.5)
    target_quat = (half, 0.0, 0.0, half)
    result = evaluate_case(_case("In", object_pos=(1.0, 1.01, 0.0), target_pos=(1.0, 1.0, 0.0), target_quat=target_quat), CONTRACT)
    negated = evaluate_case(_case("In", object_pos=(1.0, 1.01, 0.0), target_pos=(1.0, 1.0, 0.0), target_quat=tuple(-x for x in target_quat)), CONTRACT)
    assert result["value"] == negated["value"] == "TRUE"


def test_boundary_is_inclusive_but_hard_negative_is_false():
    tol = CONTRACT["tolerance"]["position_m"]
    limit = 0.9
    boundary = evaluate_case(_case("In", object_pos=(limit + tol, 0.0, 0.0)), CONTRACT)
    outside = evaluate_case(_case("In", object_pos=(limit + 2.0 * tol, 0.0, 0.0)), CONTRACT)
    assert boundary["value"] == "TRUE"
    assert outside["value"] == "FALSE"


def test_pose_hard_negative_for_support_is_false():
    result = evaluate_case(_case("On", object_pos=(0.9 + 2e-6, 0.0, 1.1)), CONTRACT)
    assert result["value"] == "FALSE"


def test_identity_mismatch_is_unknown():
    case = _case()
    case["expected_identity"]["step"] = 1
    assert evaluate_case(case, CONTRACT)["value"] == "UNKNOWN"


@pytest.mark.parametrize("which", ["object", "target"])
def test_role_swap_is_unknown(which):
    case = _case()
    case[which]["role"] = "OBJECT_TARGET" if which == "object" else "MANIPULATED_OBJECT"
    assert evaluate_case(case, CONTRACT)["value"] == "UNKNOWN"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nan_inf_is_unknown(bad):
    case = _case()
    case["object"]["pose"]["pos"][0] = bad
    assert evaluate_case(case, CONTRACT)["value"] == "UNKNOWN"


def test_forbidden_outcome_field_is_not_consumed():
    case = _case()
    case["task_success"] = True
    assert evaluate_case(case, CONTRACT)["value"] == "UNKNOWN"
