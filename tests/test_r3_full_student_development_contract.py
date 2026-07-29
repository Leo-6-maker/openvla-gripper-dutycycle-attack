import importlib.util
from pathlib import Path

import pytest

from gripper_attack.v5_r3_features import materialize_fit670_features


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_r3_full670_student_development",
    ROOT / "scripts" / "detector_v5" / "run_r3_full670_student_development.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_development_scope_excludes_safe_release():
    assert MODULE.ACTIVE_HEADS == (
        "physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"
    )
    assert MODULE.INACTIVE_HEADS == ("safe_release",)


def test_t4_permission_matrix_is_exact_and_fail_closed():
    permissions = {
        "teacher_label_read": True,
        "student_dataset_generation": True,
        "student_training": True,
        "student_training_scope": "DEVELOPMENT_ONLY",
        "development_student_training_authorized": True,
        "development_inference": True,
        "development_inference_authorized": True,
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
        "shadow_authorized": False,
        "rollout_authorized": False,
        "protected_reads": 0,
        "CAL_READ": False,
        "CHECK_READ": False,
        "G10_READ": False,
        "T2R_D_READ": False,
        "attack_authorized": False,
    }
    MODULE._validate_t4_permissions(permissions)
    permissions["CAL_READ"] = True
    with pytest.raises(ValueError):
        MODULE._validate_t4_permissions(permissions)


def test_unknown_and_right_censored_labels_are_not_known():
    base = {"value": "TRUE", "valid_mask": True, "mask": True, "right_censored": False}
    assert MODULE._known_label(base, active=True)
    assert not MODULE._known_label({**base, "value": "UNKNOWN"}, active=True)
    assert not MODULE._known_label({**base, "right_censored": True}, active=True)
    assert not MODULE._known_label(base, active=False)


def test_malformed_label_fields_fail_closed():
    with pytest.raises(ValueError):
        MODULE._known_label({"value": "FALSE", "valid_mask": True, "mask": True}, active=True)


def test_episode_path_rejects_future_escape():
    with pytest.raises(ValueError):
        MODULE._safe_episode(Path("/tmp/formal"), "episodes/../future/episode.json")


def test_label_shuffle_preserves_known_masks_and_changes_population():
    import torch

    targets = {head: torch.tensor([[0.0, 1.0, 0.0]]) for head in MODULE.HEADS}
    masks = {head: torch.tensor([[True, True, True]]) for head in MODULE.HEADS}
    shuffled = MODULE._shuffle_known_targets(targets, masks, seed=7)
    assert torch.equal(shuffled["safe_release"], targets["safe_release"])
    assert torch.equal(shuffled["physical_criticality"].sort().values, targets["physical_criticality"].sort().values)
    assert any(not torch.equal(shuffled[head], targets[head]) for head in MODULE.ACTIVE_HEADS)


def test_output_root_must_be_safe_new_sibling(tmp_path):
    with pytest.raises(ValueError):
        MODULE._safe_output(tmp_path / "CAL", parent=tmp_path)
    with pytest.raises(ValueError):
        MODULE._safe_output(Path("relative-output"), parent=tmp_path)


def test_checkpoint_continuation_optimizer_states_are_independent():
    import copy
    import torch

    model = MODULE._load_model()(input_dim=25, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    checkpoint_state = copy.deepcopy(optimizer.state_dict())
    left = MODULE._restore_optimizer(MODULE._load_model()(input_dim=25, dropout=0.0), checkpoint_state, learning_rate=1e-3, weight_decay=1e-5)
    right = MODULE._restore_optimizer(MODULE._load_model()(input_dim=25, dropout=0.0), checkpoint_state, learning_rate=1e-3, weight_decay=1e-5)
    left_state = next(iter(left.state.values()))
    right_state = next(iter(right.state.values()))
    assert left_state["exp_avg"].data_ptr() != right_state["exp_avg"].data_ptr()


def test_feature_prefix_is_invariant_to_future_suffix():
    def episode(last_value):
        steps = []
        telemetry = []
        for step in range(3):
            raw = [0.1, 0.0, 0.01, 0.0, 0.0, 0.0, 0.2]
            if step == 2:
                raw[0] = last_value
            steps.append({"step": step, "raw_action_7d": raw, "action_raw_7d": list(raw), "action_env_7d": [*raw[:6], 1.0]})
            telemetry.append({"step": step, "robot0_gripper_qpos": [0.03, -0.03], "robot0_eef_pos": [0.0, 0.0, 0.5 + 0.01 * step]})
        return {"steps": steps, "telemetry": telemetry}

    left = materialize_fit670_features(episode(0.1))
    right = materialize_fit670_features(episode(9.0))
    assert [row["features_25d"] for row in left[:2]] == [row["features_25d"] for row in right[:2]]
