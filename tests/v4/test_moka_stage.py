import numpy as np

from gripper_attack.grasp import MokaTwoPotStageTracker, compute_moka_two_pot_stage_metadata


class _Model:
    def __init__(self):
        self._map = {
            "moka_pot_1": 0,
            "moka_pot_2": 1,
            "flat_stove_1": 2,
            "moka_pot_1_main": 0,
            "moka_pot_2_main": 1,
            "flat_stove_1_main": 2,
        }

    def body_name2id(self, name):
        if name not in self._map:
            raise KeyError(name)
        return self._map[name]

    def site_name2id(self, name):
        raise KeyError(name)


class _Data:
    def __init__(self, body_xpos):
        self.body_xpos = body_xpos
        self.site_xpos = []


class _Sim:
    def __init__(self, body_xpos):
        self.model = _Model()
        self.data = _Data(body_xpos)


class _Env:
    def __init__(self, body_xpos):
        self.sim = _Sim(body_xpos)


def test_moka_stage_anchor_and_second_phase():
    # idx0: pot1, idx1: pot2, idx2: stove
    stove = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pot1 = np.array([0.01, 0.01, 0.01], dtype=np.float32)  # on stove
    pot2_far = np.array([0.3, 0.3, 0.0], dtype=np.float32)
    pot2_on = np.array([0.02, 0.02, 0.01], dtype=np.float32)

    tracker = MokaTwoPotStageTracker(stable_steps=2)
    tracker.reset()

    env_a = _Env(np.stack([pot1, pot2_far, stove], axis=0))
    m0 = compute_moka_two_pot_stage_metadata(env_a, 0, tracker, enabled=True, stage_anchor="first_pot_on_stove_stable")
    assert m0["moka_stage_id"] == "first_pot_phase"
    assert m0["moka_stage_anchor_step"] is None

    m1 = compute_moka_two_pot_stage_metadata(env_a, 1, tracker, enabled=True, stage_anchor="first_pot_on_stove_stable")
    assert m1["moka_stage_anchor_step"] == 1
    assert m1["moka_stage_id"] == "second_pot_phase"

    env_b = _Env(np.stack([pot1, pot2_on, stove], axis=0))
    m2 = compute_moka_two_pot_stage_metadata(env_b, 2, tracker, enabled=True, stage_anchor="first_pot_on_stove_stable")
    assert m2["moka_stage_id"] == "second_pot_done"
    assert m2["moka_second_pot_on_stove"] is True

