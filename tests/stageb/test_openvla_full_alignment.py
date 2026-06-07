"""OpenVLA official-alignment regression tests for Stage-B v1.1."""

from pathlib import Path

from gripper_attack.openvla_libero_exec_spec import (
    OFFICIAL_PROMPT_STYLE,
    boundary_token_ids_from_decoded_action,
    close_token_ids_from_decoded_action,
    env_gripper_is_close,
    env_gripper_is_open,
    official_prompt,
    open_token_ids_from_decoded_action,
    raw_gripper_is_boundary,
    raw_gripper_is_close,
    raw_gripper_is_open,
    raw_gripper_to_env_gripper,
)


REPO = Path(__file__).resolve().parents[2]


def test_official_boundary_raw_0p5_is_neutral_or_excluded():
    assert raw_gripper_to_env_gripper(0.5) == 0.0
    assert raw_gripper_is_boundary(0.5)
    assert not raw_gripper_is_open(0.5)
    assert not raw_gripper_is_close(0.5)
    assert not env_gripper_is_open(0.0)
    assert not env_gripper_is_close(0.0)


def test_attack_open_token_region_excludes_boundary():
    token_action_map = {1: 0.0, 2: 0.499, 3: 0.5, 4: 0.501, 5: 1.0}
    assert open_token_ids_from_decoded_action(token_action_map) == [4, 5]
    assert close_token_ids_from_decoded_action(token_action_map) == [1, 2]
    assert boundary_token_ids_from_decoded_action(token_action_map) == [3]


def test_runner_prompt_branch_non_v01_uses_in_out():
    prompt = official_prompt("pick up the ketchup")
    assert OFFICIAL_PROMPT_STYLE == "official_in_out"
    assert prompt == "In: What action should the robot take to pick up the ketchup?\nOut:"


def test_image_preprocess_style_is_explicit_not_silent():
    runner = (REPO / "scripts" / "run_stageb_vis_labeling.py").read_text()
    assert "IMAGE_PREPROCESS_STYLE" in runner
    assert "official_rot180_only" in runner
    assert "legacy_direct_agentview_no_rotation" in runner


def test_no_direct_open_comparison_in_stageb_main_path():
    """Stage-B v1.1 open decisions should go through the spec helper."""
    checked = [
        REPO / "scripts" / "run_stageb_vis_labeling.py",
        REPO / "scripts" / "stageb" / "postprocess_traces_v1_1.py",
        REPO / "scripts" / "stageb" / "build_pair_labels_v1_1.py",
    ]
    forbidden = [
        "env_action_6 < -0.5",
        "env_action_6 > 0.5",
        "env_grip > 0",
        "env_grip < 0",
        "raw_gripper >= 0.5",
        "raw_gripper < 0.5",
    ]
    for path in checked:
        text = path.read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{path} contains direct open comparison {pattern!r}"
