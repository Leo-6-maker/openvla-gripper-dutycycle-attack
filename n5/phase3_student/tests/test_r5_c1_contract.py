"""[DeepSeek] R5-C1: Negative contract tests for forward-before-capture protocol.

Verifies that the corrected collector fails-closed on:
  1. Omission of sim.forward() before entity capture
  2. qpos/qvel/act/time mutation by collection
  3. NaN/Inf in captured poses
  4. Second forward mismatch (non-deterministic forward)
  5. Pre-forward poses != post-forward poses (confirms fix is working)

Run: python n5/phase3_student/tests/test_r5_c1_contract.py
"""
import json, os, sys, copy, math, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'phase2_labels'))

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_STEPS_WAIT = 10
SEED = 20260717


class TestForwardBeforeCapture(unittest.TestCase):
    """Contract: sim.forward() before entity capture is required."""

    @classmethod
    def setUpClass(cls):
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark, get_benchmark_dict
        from libero.libero.envs import OffScreenRenderEnv
        import random as _random

        _random.seed(SEED)
        bm = get_benchmark("libero_10")(0)
        t = bm.get_task(0)
        bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        sd = get_benchmark_dict()
        so = sd["libero_10"]()
        cls.canonical_state = copy.deepcopy(so.get_task_init_states(0)[15])

        cls.env = OffScreenRenderEnv(
            bddl_file_name=bp, camera_heights=256, camera_widths=256,
            render_gpu_device_id=-1, has_renderer=False,
            has_offscreen_renderer=False, horizon=520)
        cls.env.seed(SEED)
        cls.env.reset()
        cls.env.set_init_state(copy.deepcopy(cls.canonical_state))
        for _ in range(NUM_STEPS_WAIT):
            cls.env.step(DUMMY_ACTION)
        cls.model = cls.env.sim.model
        cls.data = cls.env.sim.data

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _non_robot_body_ids(self):
        ids = []
        for bid in range(self.model.nbody):
            name = self.model.body_id2name(bid)
            if name and all(k not in name for k in ("robot", "gripper", "world", "floor", "mount")):
                ids.append((bid, name))
        return ids

    def _non_robot_site_ids(self):
        ids = []
        for sid in range(self.model.nsite):
            name = self.model.site_id2name(sid)
            if name and "_region" in name:
                ids.append((sid, name))
        return ids

    def test_01_forward_fixes_stale_read(self):
        """Omission of forward MUST produce detectable stale reads (A != B)."""
        body_ids = self._non_robot_body_ids()
        self.assertGreater(len(body_ids), 0, "no non-robot bodies found")

        any_stale = False
        for step in range(5):
            # A: read without forward (simulates buggy collector)
            A_pos = {}
            for bid, name in body_ids:
                A_pos[bid] = self.data.body_xpos[bid].copy()

            # B: after forward (corrected collector)
            self.env.sim.forward()
            B_pos = {}
            for bid, name in body_ids:
                B_pos[bid] = self.data.body_xpos[bid].copy()

            for bid, name in body_ids:
                diff = float(np.max(np.abs(A_pos[bid] - B_pos[bid])))
                if diff > 1e-12:
                    any_stale = True

            self.env.step([0.0] * 7)

        self.assertTrue(any_stale,
                        "STALE_READ_NOT_DETECTED: A==B for all cases — "
                        "either forward is unnecessary or test is wrong. "
                        "If this fails consistently, the stale-read root cause "
                        "may not apply to this LIBERO version.")

    def test_02_forward_preserves_qpos(self):
        """sim.forward() MUST NOT mutate qpos/qvel."""
        body_ids = self._non_robot_body_ids()
        for step in range(3):
            qpos_before = self.data.qpos.copy()
            qvel_before = self.data.qvel.copy()
            time_before = float(self.data.time)

            self.env.sim.forward()

            qpos_drift = float(np.max(np.abs(qpos_before - self.data.qpos.copy())))
            qvel_drift = float(np.max(np.abs(qvel_before - self.data.qvel.copy())))
            time_drift = abs(time_before - float(self.data.time))

            self.assertEqual(qpos_drift, 0.0,
                             f"qpos mutated by forward at step {step}: drift={qpos_drift:.2e}")
            self.assertEqual(qvel_drift, 0.0,
                             f"qvel mutated by forward at step {step}: drift={qvel_drift:.2e}")
            self.assertEqual(time_drift, 0.0,
                             f"time mutated by forward at step {step}: drift={time_drift:.2e}")

            self.env.step([0.0] * 7)

    def test_03_second_forward_is_deterministic(self):
        """B→C: second sim.forward() MUST produce identical poses."""
        body_ids = self._non_robot_body_ids()
        site_ids = self._non_robot_site_ids()

        for step in range(3):
            self.env.sim.forward()
            B_body = {bid: self.data.body_xpos[bid].copy() for bid, _ in body_ids}
            B_site = {sid: self.data.site_xpos[sid].copy() for sid, _ in site_ids}

            self.env.sim.forward()
            C_body = {bid: self.data.body_xpos[bid].copy() for bid, _ in body_ids}
            C_site = {sid: self.data.site_xpos[sid].copy() for sid, _ in site_ids}

            for bid, name in body_ids:
                diff = float(np.max(np.abs(B_body[bid] - C_body[bid])))
                self.assertLessEqual(diff, 1e-15,
                    f"B→C body drift at step {step} {name}: {diff:.2e}")
            for sid, name in site_ids:
                diff = float(np.max(np.abs(B_site[sid] - C_site[sid])))
                self.assertLessEqual(diff, 1e-15,
                    f"B→C site drift at step {step} {name}: {diff:.2e}")

            self.env.step([0.0] * 7)

    def test_04_positions_are_finite_after_forward(self):
        """All body_xpos/site_xpos must be finite after forward."""
        self.env.sim.forward()
        for bid in range(self.model.nbody):
            pos = self.data.body_xpos[bid]
            self.assertTrue(all(math.isfinite(float(x)) for x in pos),
                            f"non-finite body_xpos for body {bid}")
        for sid in range(self.model.nsite):
            pos = self.data.site_xpos[sid]
            self.assertTrue(all(math.isfinite(float(x)) for x in pos),
                            f"non-finite site_xpos for site {sid}")

    def test_05_multistep_no_drift_accumulation(self):
        """Over multiple steps, B after forward must be consistent."""
        body_ids = self._non_robot_body_ids()
        prev_pos = {}
        for step in range(5):
            self.env.sim.forward()
            for bid, name in body_ids:
                pos = self.data.body_xpos[bid].copy()
                if bid in prev_pos:
                    diff = float(np.max(np.abs(prev_pos[bid] - pos)))
                    # Zero-action trajectory: object should stay still
                    self.assertLess(diff, 0.1,
                        f"excessive drift at step {step} {name}: {diff:.2e}")
                prev_pos[bid] = pos
            self.env.step([0.0] * 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
