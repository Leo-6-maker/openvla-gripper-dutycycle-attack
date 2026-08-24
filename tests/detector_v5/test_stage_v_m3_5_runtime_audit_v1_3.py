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


def test_exact_a800_receipt_is_bound_to_tests_code_and_clean_source(tmp_path):
    test_file = tmp_path / "tests/detector_v5/test_stage_v_z.py"
    second_test_file = tmp_path / "tests/detector_v5/test_stage_v_a.py"
    code_file = tmp_path / "scripts/example.py"
    test_file.parent.mkdir(parents=True)
    code_file.parent.mkdir(parents=True)
    test_file.write_text("assert True\n", encoding="utf-8")
    second_test_file.write_text("assert True\n", encoding="utf-8")
    code_file.write_text("VALUE = 1\n", encoding="utf-8")
    bindings = {path.relative_to(tmp_path).as_posix(): _STATIC._sha256(path) for path in (code_file, test_file, second_test_file)}
    binding = {
        "runtime_python": "/exact/python",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "expected_collected": 2,
        "test_files": [test_file.relative_to(tmp_path).as_posix(), second_test_file.relative_to(tmp_path).as_posix()],
        "tested_bindings": bindings,
    }
    receipt = {
        "schema": "STAGE_V_M3_5_EXACT_A800_REGRESSION_RECEIPT_V1",
        "status": "PASS",
        "runtime_python": "/exact/python",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "source_status_porcelain": "",
        "cuda_visible_devices": "",
        "test_files": binding["test_files"],
        "tested_bindings": bindings,
        "collected": 2,
        "passed": 1,
        "skipped": 1,
        "failed": 0,
        "errors": 0,
        "deselected": 0,
        "py_compile_status": "PASS",
        "protected_counters": dict(_STATIC.COUNTERS),
    }
    assert _STATIC._exact_regression_valid(tmp_path, binding, receipt, "/exact/python")[0]
    receipt["source_status_porcelain"] = " M dirty.py"
    assert not _STATIC._exact_regression_valid(tmp_path, binding, receipt, "/exact/python")[0]
    receipt["source_status_porcelain"] = ""
    code_file.write_text("VALUE = 2\n", encoding="utf-8")
    assert not _STATIC._exact_regression_valid(tmp_path, binding, receipt, "/exact/python")[0]
