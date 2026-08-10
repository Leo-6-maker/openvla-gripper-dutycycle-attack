from pathlib import Path

from gripper_attack.stage_v_m3_5_physical_taxonomy import (
    build_forced_open_action,
    bind_object_taxonomy,
    evaluate_treatment_compliance,
    parse_goal_object_ids,
    repeatability_receipt,
    taxonomy_eligibility_from_bddl,
    telemetry_from_env,
    v_phys_label,
)


BDDL = """
(:objects
  bowl_1 - bowl
  plate_1 - plate
)
(:goal
  (And (On bowl_1 plate_1))
)
"""


class _Contact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


class _Model:
    nbody = 4
    nsite = 1
    ngeom = 3
    geom_bodyid = [1, 2, 3]

    def body_id2name(self, index):
        return ["world", "bowl_1", "plate_1", "gripper0_hand"][index]

    def site_name2id(self, name):
        assert name == "gripper0_grip_site"
        return 0


class _Data:
    ncon = 2
    contact = [_Contact(0, 2), _Contact(0, 1)]
    body_xpos = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [0.3, 0.0, 0.1], [0.0, 0.0, 0.0]]
    site_xpos = [[0.0, 0.0, 0.2]]


class _Sim:
    model = _Model()
    data = _Data()


class _Env:
    sim = _Sim()


def test_goal_parser_uses_goal_source_objects_only():
    assert parse_goal_object_ids(BDDL) == ("bowl_1",)


def test_taxonomy_binds_and_separates_gripper_from_support_contact(tmp_path: Path):
    path = tmp_path / "task.bddl"
    path.write_text(BDDL, encoding="utf-8")
    binding = bind_object_taxonomy(_Env(), path)
    assert binding["status"] == "PASS"
    telemetry = telemetry_from_env(_Env(), binding)
    assert telemetry["contact_telemetry_valid"] is True
    assert telemetry["object_gripper_contact"] is True
    assert telemetry["object_support_contact"] is True
    assert telemetry["object_identity"] == "bowl_1"


def test_missing_live_body_abstains(tmp_path: Path):
    path = tmp_path / "task.bddl"
    path.write_text(BDDL.replace("bowl_1", "missing_bowl"), encoding="utf-8")
    binding = bind_object_taxonomy(_Env(), path)
    assert binding["status"] == "ABSTAIN"
    telemetry = telemetry_from_env(_Env(), binding)
    assert telemetry["contact_telemetry_valid"] is False


def test_fixture_only_goal_is_explicitly_ineligible(tmp_path: Path):
    path = tmp_path / "fixture.bddl"
    path.write_text("(:objects cabinet_1 - cabinet) (:goal (And (Open cabinet_1)))", encoding="utf-8")
    eligibility = taxonomy_eligibility_from_bddl(path)
    assert eligibility["status"] == "INELIGIBLE"
    assert eligibility["fixture_binding_inference_allowed"] is False
    assert bind_object_taxonomy(_Env(), path)["status"] == "INELIGIBLE"


def test_telemetry_is_bound_to_one_registered_target():
    binding = {
        "status": "PASS",
        "target_object_ids": ["bowl_1", "plate_1"],
        "target_body_ids": {"bowl_1": 1, "plate_1": 2},
        "eef_site_id": 0,
        "body_names": ["world", "bowl_1", "plate_1", "gripper0_hand"],
    }
    telemetry = telemetry_from_env(_Env(), binding, target_object_id="plate_1")
    assert telemetry["object_identity"] == "plate_1"
    assert telemetry["object_gripper_contact"] is False
    assert telemetry["object_support_contact"] is True


def test_forced_open_changes_only_gripper():
    result = build_forced_open_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0], [1, 2, 3, 4, 5, 6, 1])
    assert result["raw_policy_action"][-1] == 1.0
    assert result["env_action"][-1] == -1.0
    assert result["arm_delta_linf"] == 0.0


def test_compliance_distinguishes_command_from_response():
    receipt = [{
        "raw_policy_action": [0, 0, 0, 0, 0, 0, 1],
        "normalized_action": [0, 0, 0, 0, 0, 0, 1],
        "env_action": [0, 0, 0, 0, 0, 0, -1],
        "arm_delta_linf": 0.0,
        "pre_aperture": 0.0,
        "post_aperture": 0.006,
    }]
    result = evaluate_treatment_compliance(receipt, expected_steps=1)
    assert result["treatment_compliant"] is True
    assert result["command_delivery_valid"] is True
    assert result["aperture_response"] is True

    incomplete = evaluate_treatment_compliance(receipt, expected_steps=3)
    assert incomplete["treatment_compliant"] is False
    assert "DELIVERED_STEP_COUNT:1/3" in incomplete["command_failures"]


def test_repeatability_and_truth_table_fail_closed():
    rows = [{"outcome_class": "NO_PHYSICAL_VULNERABILITY", "treatment_compliant": True}] * 3
    assert repeatability_receipt(rows)["status"] == "PASS_REPEATABILITY_3_OF_3"
    assert v_phys_label(control_valid=True, treatment_valid=True, f_control=0, f_open=1) == "V_PHYS"
    assert v_phys_label(control_valid=True, treatment_valid=True, f_control=None, f_open=1) == "PHYSICAL_AMBIGUITY_ABSTAIN"
