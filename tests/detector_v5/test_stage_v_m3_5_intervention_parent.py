import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "detector_v5" / "run_stage_v_m3_5_intervention_parent.py"
_SPEC = importlib.util.spec_from_file_location("stage_v_m3_5_intervention_parent", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_pair_label = _MODULE._pair_label


def _branch(contact=True):
    rows = []
    for step in range(10):
        rows.append({
            "object_gripper_contact": contact,
            "post_object_gripper_contact": contact,
            "post_object_support_contact": True,
            "post_object_position": [0.0, 0.0, 0.1],
        })
    return {
        "status": "PASS",
        "state_restore_exact": True,
        "physical_horizon_complete": True,
        "treatment_compliant": True,
        "rows": rows,
    }


def test_pair_label_promotes_only_control_valid_open_failure():
    control = _branch(True)
    treatment = _branch(True)
    treatment["rows"][1]["post_object_gripper_contact"] = False
    treatment["rows"][2]["post_object_gripper_contact"] = False
    pair = _pair_label(control, treatment)
    assert pair["control_valid"] is True
    assert pair["treatment_valid"] is True
    assert pair["label_class"] == "V_PHYS"


def test_pair_label_abstains_without_control_contact_eligibility():
    pair = _pair_label(_branch(False), _branch(False))
    assert pair["label_class"] == "CONTROL_INVALID_ABSTAIN"
