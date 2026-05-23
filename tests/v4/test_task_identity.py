"""Test task identity mapping — no GPU, no model, no rollout."""
import sys
from pathlib import Path

# Ensure src/ is importable
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.task_identity import (
    TASK_IDENTITY,
    BOWL_ON_PLATE_SPATIAL,
    RUNNER_TASK_ID,
    RUN_ID_TASK_KEY,
    TABLE1_TASK_KEY,
    MATCHED_CONDITIONS,
    OPTIONAL_CONDITION,
    TRIAGE_MATCHED_CONDITIONS,
    make_run_id,
    make_clean_detect_run_id,
)


class TestTaskIdentity:
    def test_runner_task_id_is_spatial(self):
        assert TASK_IDENTITY["runner_task_id"] == "libero_spatial_black_bowl"

    def test_semantic_name_is_goal_put_bowl(self):
        assert TASK_IDENTITY["semantic_task_name"] == "goal_put_the_bowl_on_the_plate"

    def test_suite_is_libero_spatial(self):
        assert TASK_IDENTITY["suite"] == "libero_spatial"

    def test_is_black_bowl_related(self):
        assert TASK_IDENTITY["is_black_bowl_related"] is True
        assert TASK_IDENTITY["is_non_black_bowl_claim"] is False

    def test_runner_and_semantic_are_different(self):
        assert RUNNER_TASK_ID != RUN_ID_TASK_KEY

    def test_table1_key_matches_semantic(self):
        assert TABLE1_TASK_KEY == RUN_ID_TASK_KEY

    def test_convenience_aliases_match(self):
        assert RUNNER_TASK_ID == BOWL_ON_PLATE_SPATIAL["runner_task_id"]
        assert RUN_ID_TASK_KEY == BOWL_ON_PLATE_SPATIAL["semantic_task_name"]

    def test_make_run_id(self):
        rid = make_run_id("goal_put_the_bowl_on_the_plate", 5, 3, "clean_detect")
        assert rid == "goal_put_the_bowl_on_the_plate_s5_r3_clean_detect"

    def test_make_clean_detect_run_id(self):
        rid = make_clean_detect_run_id("goal_put_the_bowl_on_the_plate", 0, 7)
        assert rid == "goal_put_the_bowl_on_the_plate_s0_r7_clean_detect"


class TestMatchedConditions:
    def test_all_conditions_have_required_fields(self):
        required = {"condition_name", "attack_objective", "temporal_init",
                    "force_open_raw_gripper", "rho", "cw_margin",
                    "epsilon", "step_size", "attack_steps"}
        for i, c in enumerate(MATCHED_CONDITIONS):
            missing = required - set(c.keys())
            assert not missing, f"Condition {i} ({c.get('condition_name','?')}) missing: {missing}"

    def test_all_tuples_same_length(self):
        """Dict-based conditions cannot have tuple-length bugs."""
        for i, c in enumerate(MATCHED_CONDITIONS):
            assert len(c) >= 10, f"Condition {i}: expected >=10 keys, got {len(c)}"

    def test_control_conditions_have_rho_zero(self):
        for c in MATCHED_CONDITIONS:
            if c.get("is_control"):
                assert c["rho"] == 0.0, f"{c['condition_name']}: control should have rho=0.0"

    def test_attack_conditions_have_attack_steps_positive(self):
        for c in MATCHED_CONDITIONS:
            if c.get("is_attack"):
                assert c["attack_steps"] > 0, f"{c['condition_name']}: attack should have attack_steps>0"

    def test_random_has_noise_objective(self):
        rand = MATCHED_CONDITIONS[0]
        assert rand["attack_objective"] == "random_noise"
        assert rand["is_control"] is True

    def test_targeted_vis_has_cw_objective(self):
        targeted = MATCHED_CONDITIONS[2]
        assert targeted["attack_objective"] == "gripper_logit_margin_cw"
        assert targeted["is_attack"] is True

    def test_command_open_has_oracle_objective(self):
        cmd = MATCHED_CONDITIONS[3]
        assert cmd["attack_objective"] == "oracle_env_gripper_open"
        assert cmd["force_open_raw_gripper"] == 0.75

    def test_optional_condition_has_expected_name(self):
        assert OPTIONAL_CONDITION["condition_name"] == "VIS_gripper_open_region_ce_same_autowindow"

    def test_triage_conditions_include_command_open_1_00(self):
        names = [c["condition_name"] for c in TRIAGE_MATCHED_CONDITIONS]
        assert "command_open_1.00_same_autowindow" in names
