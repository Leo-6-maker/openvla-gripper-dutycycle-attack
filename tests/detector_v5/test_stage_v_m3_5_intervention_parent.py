import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "detector_v5" / "run_stage_v_m3_5_intervention_parent.py"
_SPEC = importlib.util.spec_from_file_location("stage_v_m3_5_intervention_parent", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_pair_label = _MODULE._pair_label
_branch_record = _MODULE._branch_record
_treatment_observation = _MODULE._treatment_observation
_collapsed_label = _MODULE._collapsed_label
_directory_tree_binding = _MODULE._directory_tree_binding
_new_env = _MODULE._new_env


def _branch(contact=True):
    rows = []
    for step in range(20):
        rows.append({
            "object_gripper_contact": contact,
            "post_object_gripper_contact": contact,
            "post_object_support_contact": True,
            "post_contact_telemetry_valid": True,
            "post_object_position": [0.0, 0.0, 0.1],
        })
    return {
        "status": "PASS",
        "state_restore_exact": True,
        "causal_input_binding_pass": True,
        "control_clean_action_equivalence": True,
        "available_horizon_steps": 20,
        "treatment_compliant": True,
        "treatment_compliance": {"delivered_open_steps": 3},
        "rows": rows,
    }


def test_pair_label_promotes_only_control_valid_open_failure():
    control = _branch(True)
    treatment = _branch(True)
    treatment["rows"][1]["post_object_gripper_contact"] = False
    treatment["rows"][2]["post_object_gripper_contact"] = False
    pair = _pair_label(control, treatment, dose_steps=3)
    assert pair["control_valid"] is True
    assert pair["treatment_valid"] is True
    assert pair["label_class"] == "V_PHYS"


def test_pair_label_abstains_when_matched_control_loses_contact():
    pair = _pair_label(_branch(False), _branch(False), dose_steps=3)
    assert pair["f_control"] == 1
    assert pair["label_class"] == "CONTROL_PHYSICAL_FAILURE_ABSTAIN"


def test_pair_label_enforces_per_dose_horizon():
    treatment = _branch(True)
    treatment["available_horizon_steps"] = 14
    treatment["rows"] = treatment["rows"][:14]
    treatment["treatment_compliance"]["delivered_open_steps"] = 5
    pair = _pair_label(_branch(True), treatment, dose_steps=5)
    assert pair["required_horizon_steps"] == 15
    assert pair["label_class"] == "TREATMENT_INVALID_ABSTAIN"


def test_collapsed_label_preserves_three_explicit_matched_controls():
    observations = []
    for repetition in range(3):
        control = _branch(True)
        treatment = _branch(True)
        treatment["rows"][1]["post_object_gripper_contact"] = False
        treatment["rows"][2]["post_object_gripper_contact"] = False
        pair = _pair_label(control, treatment, dose_steps=3)
        control_record = _branch_record(
            control, parent_key="libero_goal/task_00/state_00", probe_id="Q00",
            probe_step=7, repetition=repetition, arm="CONTROL",
        )
        treatment_record = _branch_record(
            treatment, parent_key="libero_goal/task_00/state_00", probe_id="Q00",
            probe_step=7, repetition=repetition, arm="T3",
            shared_control_branch_id=control_record["branch_id"],
            shared_control_result_sha256=control_record["branch_result_sha256"], pair=pair,
        )
        observations.append(_treatment_observation(control_record, treatment_record))
    collapsed = _collapsed_label(observations)
    assert collapsed["collapsed_label_class"] == "V_PHYS"
    assert collapsed["binary_label_consumable"] is True
    assert len({row["branch_id"] for row in collapsed["matched_control_lineage"]}) == 3


def test_model_tree_binding_covers_relative_paths_and_contents(tmp_path):
    (tmp_path / "a").write_text("weights", encoding="utf-8")
    first = _directory_tree_binding(tmp_path)
    (tmp_path / "a").rename(tmp_path / "b")
    second = _directory_tree_binding(tmp_path)
    assert first["file_count"] == second["file_count"] == 1
    assert first["tree_sha256"] != second["tree_sha256"]


def test_env_uses_physical_egl_index_after_cuda_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "before")
    class Env:
        def __init__(self, **kwargs):
            self.render_gpu_device_id = kwargs["render_gpu_device_id"]

        def seed(self, _seed):
            pass

        def reset(self):
            return {}

        def set_init_state(self, _state):
            return {}

        def step(self, _action):
            return {}, 0.0, False, {}

    args = type("Args", (), {})()
    env, _obs = _new_env(Env, "task.bddl", 10, 5, [0.0], args, tmp_path)
    assert env.render_gpu_device_id == 5
    assert _MODULE.os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert _MODULE.os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"
