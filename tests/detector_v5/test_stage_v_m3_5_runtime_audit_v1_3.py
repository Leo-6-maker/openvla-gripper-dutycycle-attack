import importlib.util
from pathlib import Path


_ROOT = Path(__file__).parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AUDIT = _load("stage_v_m3_5_runtime_audit_v1_3", "scripts/detector_v5/audit_stage_v_m3_5_runtime_v1_3.py")
_RUNNER = _load("stage_v_m3_5_runner_v1_3", "scripts/detector_v5/run_stage_v_m3_5_intervention_parent.py")
_STATIC = _load("stage_v_m3_5_static_audit_v1_3", "scripts/detector_v5/audit_stage_v_m3_5_static_contract_v1_3.py")


def _branch(contact=True):
    rows = [{
        "relative_step": step,
        "post_contact_telemetry_valid": True,
        "post_object_identity": "bowl_1",
        "post_object_position": [0.0, 0.0, 0.1],
        "post_eef_position": [0.0, 0.0, 0.1],
        "post_object_eef_distance_m": 0.0,
        "post_object_gripper_contact": contact,
        "post_object_support_contact": True,
        "task_success": False,
    } for step in range(20)]
    return {
        "status": "PASS",
        "state_restore_exact": True,
        "causal_input_binding_pass": True,
        "control_clean_action_equivalence": True,
        "available_horizon_steps": 20,
        "rows": rows,
    }


def test_independent_physical_classifier_matches_producer():
    control, treatment = _branch(), _branch()
    treatment["rows"][1]["post_object_gripper_contact"] = False
    treatment["rows"][2]["post_object_gripper_contact"] = False
    expected = _RUNNER._physical_outcome(treatment, required_steps=13, reference=control)
    actual = _AUDIT._physical(treatment, 13, control)
    assert actual == expected
    assert actual["class"] == "GRIPPER_CONTACT_LOSS"
    assert len(actual["telemetry_evidence"]) == 2


def test_registered_release_and_drop_predicates_are_executable():
    control = _branch()
    release = _branch()
    release["rows"][0]["post_object_position"] = [0.02, 0.0, 0.1]
    release["rows"][0]["post_object_gripper_contact"] = False
    assert _AUDIT._physical(release, 13, control) == _RUNNER._physical_outcome(release, required_steps=13, reference=control)
    assert _AUDIT._physical(release, 13, control)["class"] == "PREMATURE_OBJECT_RELEASE"

    drop = _branch()
    drop["rows"][0]["post_object_position"] = [0.0, 0.0, 0.07]
    drop["rows"][0]["post_object_gripper_contact"] = False
    drop["rows"][0]["post_object_support_contact"] = False
    drop["rows"][1].update(drop["rows"][0])
    drop["rows"][1]["relative_step"] = 1
    assert _AUDIT._physical(drop, 13, control) == _RUNNER._physical_outcome(drop, required_steps=13, reference=control)
    assert _AUDIT._physical(drop, 13, control)["class"] == "OBJECT_DROP"


def test_malformed_runtime_artifact_fails_closed(tmp_path):
    malformed = _branch()
    malformed["available_horizon_steps"] = "twenty"
    assert _AUDIT._physical(malformed, 13)["class"] == "HORIZON_CENSORED_ABSTAIN"
    assert _AUDIT._repeat([{"repetition": 0}, {"repetition": "1"}, {"repetition": 2}])[0] == "HOLD_STOCHASTIC_INTERVENTION_OUTCOME"
    assert _AUDIT._inside(tmp_path, "../escape.png") is None


def test_manual_pair_selection_is_deterministic_and_in_range():
    first = _STATIC._manual_pair("a" * 64, "libero_goal/task_00/state_00")
    assert first == _STATIC._manual_pair("a" * 64, "libero_goal/task_00/state_00")
    assert first["probe_id"] in {f"Q{index:02d}" for index in range(24)}
    assert first["repetition"] in range(3)
    assert first["dose"] in {"T3", "T5", "T10"}
