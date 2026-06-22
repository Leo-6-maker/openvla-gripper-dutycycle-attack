import numpy as np
import pytest

from gripper_attack.sc5_online_feature_state import (
    extract_sc5_physical_state,
    initialize_sc5_prev_eef,
)


class _Model:
    def site_name2id(self, name):
        assert name == "gripper0_grip_site"
        return 0


class _Data:
    def __init__(self):
        self.qpos = np.array([100.0, 200.0], dtype=np.float64)
        self.site_xpos = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)


class _Sim:
    def __init__(self):
        self.model = _Model()
        self.data = _Data()


class _Env:
    def __init__(self):
        self.sim = _Sim()


def test_extract_sc5_physical_state_uses_canonical_physical_gripper_state_not_qpos_prefix():
    env = _Env()

    def physical_state(_env, _obs):
        return {"qpos": [0.01, -0.03]}

    out = extract_sc5_physical_state(
        env=env,
        obs={"agentview_image": "dummy"},
        prev_eef=(0.5, 1.0, 2.0),
        physical_gripper_state_fn=physical_state,
    )

    assert out.gripper_qpos == pytest.approx(-0.02)
    assert out.gripper_opening_proxy == pytest.approx(0.04)
    assert out.gripper_qpos_values == (0.01, -0.03)
    assert out.eef_x == 1.0
    assert out.eef_y == 2.0
    assert out.eef_z == 3.0
    assert out.eef_vx == 0.5
    assert out.eef_vy == 1.0
    assert out.eef_vz == 1.0


def test_initialize_sc5_prev_eef_reads_gripper_site():
    assert initialize_sc5_prev_eef(_Env()) == (1.0, 2.0, 3.0)
