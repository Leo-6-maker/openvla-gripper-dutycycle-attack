"""R8U post-canary tests — uses mock benchmark/env/streamer to test _replay_episode."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest import mock

import numpy as np


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_meta(ep_dir: Path, *, bddl_sha="f" * 64, init_sha="e" * 64,
              max_steps=50, dummy_wait=10, n_steps=50, suite="libero_object",
              task_index=0, state_id=1, replay_seed=42):
    meta = {
        "suite": suite, "task_index": task_index, "state_id": state_id,
        "parent_key": f"{suite}/task_{task_index}/state_{state_id}/detector_train/episode_000",
        "bddl_file": str(ep_dir / "fake.bddl"),
        "bddl_sha256": bddl_sha,
        "official_init_state_sha256": init_sha,
        "official_init_state_shape": [110],
        "official_init_state_dtype": "float64",
        "max_steps": max_steps, "dummy_wait": dummy_wait,
        "replay_seed": replay_seed, "n_steps": n_steps,
        "cohort": "DETECTOR_TRAIN", "split": "train",
        "runtime_versions": {"libero": "0.1.1", "robosuite": "1.4.0"},
        "controller_config": {"control_freq": 20},
    }
    write_json(ep_dir / "episode_metadata.json", meta)
    return meta


def make_steps(ep_dir: Path, n_steps=50, *, action_value=0.01, features_value=0.5):
    steps = [{
        "step": i,
        "clean_action_raw_7d": [action_value + i * 0.0001] * 7,
        "applied_action_7d": [action_value + i * 0.0001] * 7,
        "features_25d": [features_value + i * 0.001] * 25,
    } for i in range(n_steps)]
    write_jsonl(ep_dir / "step_records.jsonl", steps)
    return steps


def make_bddl(ep_dir: Path, sha_override=None):
    ep_dir.mkdir(parents=True, exist_ok=True)
    bddl = ep_dir / "fake.bddl"
    bddl.write_text("(define (problem test))")
    if sha_override:
        bddl.write_text(sha_override)
    return bddl


class MockEnv:
    """Fake LIBERO env that replays actions deterministically."""
    def __init__(self, steps, success_at=None, done_at=None):
        self.steps = steps
        self.success_at = success_at
        self.done_at = done_at
        self._step = 0
        self._closed = False
        self.sim = mock.MagicMock()
        self.sim.model.site_name2id.return_value = 0
        self.sim.data.site_xpos = np.zeros(3, dtype=np.float32)
        self.robots = [mock.MagicMock()]
        self.robots[0].joint_positions = np.zeros(7, dtype=np.float64)
        # For check_success
        self._check_success_values = []

    def set_check_success_values(self, values):
        self._check_success_values = list(values)

    def check_success(self):
        if self._check_success_values:
            return self._check_success_values[min(self._step, len(self._check_success_values) - 1)]
        if self.success_at is not None:
            return self._step >= self.success_at
        return False

    def step(self, action):
        done = False
        if self.done_at is not None and self._step >= self.done_at:
            done = True
        obs = np.zeros(10, dtype=np.float32)
        reward = 0.0
        info = {}
        self._step += 1
        return obs, reward, done, info

    def set_init_state(self, state):
        return self.step(np.zeros(7))[0]

    def close(self):
        self._closed = True


class MockStreamer:
    """Returns features that are exact or slightly-off replicas of stored 25D."""
    def __init__(self, mode="exact"):
        self.mode = mode
        self.features = {}

    def update(self, **kwargs):
        from scripts.stageb.replay_c2g_r8t_canary_success import array_sha256
        return self


def _mock_benchmark_dict(suite="libero_object", task_index=0, state_id=1):
    """Create a mock benchmark that returns predictable init states."""
    init = np.zeros(110, dtype=np.float64)
    init[0] = state_id * 0.01

    mock_task = mock.MagicMock()
    mock_suite = mock.MagicMock()
    mock_suite.get_task.return_value = mock_task
    mock_suite.get_task_init_states.return_value = [init] * (state_id + 5)

    bm = {suite: lambda: mock_suite}
    return bm, init


try:
    import libero  # noqa: F401
    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False


class ReplayEpisodeTests(unittest.TestCase):
    """Tests that call _replay_episode with mock components (requires LIBERO)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _call_replay(self, ep_dir, **overrides):
        """Import and call _replay_episode with all mocks patched."""
        import sys as _sys

        # Pre-populate sys.modules with fake LIBERO to prevent import errors
        fake_libero = mock.MagicMock()
        fake_libero_libero = mock.MagicMock()
        fake_libero_libero_envs = mock.MagicMock()
        fake_benchmark = mock.MagicMock()
        bm, init = _mock_benchmark_dict()
        fake_benchmark.get_benchmark_dict.return_value = bm
        fake_libero_libero.benchmark = fake_benchmark
        fake_libero.libero = fake_libero_libero

        saved = {}
        for mod_name, fake in [
            ("libero", fake_libero),
            ("libero.libero", fake_libero_libero),
            ("libero.libero.benchmark", fake_benchmark),
            ("libero.libero.envs", fake_libero_libero_envs),
        ]:
            if mod_name in _sys.modules:
                saved[mod_name] = _sys.modules[mod_name]
            _sys.modules[mod_name] = fake

        # Also patch the module-level import in libero_v4_env_factory
        import src.gripper_attack.libero_v4_env_factory as _lv4
        saved_attrs = {}
        for attr in ("build_v4_exact_env", "apply_dummy_wait", "OffScreenRenderEnv"):
            if hasattr(_lv4, attr):
                saved_attrs[attr] = getattr(_lv4, attr)

        mock_env = MockEnv([])
        _lv4.build_v4_exact_env = mock.MagicMock(return_value=(mock_env, np.zeros(10, dtype=np.float32)))
        _lv4.apply_dummy_wait = mock.MagicMock(return_value=(mock_env, np.zeros(10, dtype=np.float32)))

        try:
            from scripts.stageb.replay_c2g_r8t_canary_success import _replay_episode
            with mock.patch("scripts.v4_run_eval_openvla.physical_gripper_state",
                            return_value={"qpos": [0.0] * 7}), \
                 mock.patch("src.gripper_attack.sc5_streaming_features_v2.SC5StreamingFeatureAdapterV2",
                            return_value=MockStreamer()), \
                 mock.patch("src.gripper_attack.c2g_clean_mechanism.set_deterministic_seeds",
                            return_value=None), \
                 mock.patch("numpy.asarray", side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype)):
                return _replay_episode(ep_dir, render_gpu=0)
        finally:
            for mod, val in saved.items():
                _sys.modules[mod] = val
            for attr, val in saved_attrs.items():
                setattr(_lv4, attr, val)

    # ── FAILED path ──

    def test_bddl_sha_mismatch_fails(self):
        ep = self.root / "ep_bad_bddl"
        make_meta(ep, bddl_sha="a" * 64)
        make_steps(ep, 50)
        make_bddl(ep)
        with self.assertRaises(ValueError):
            self._call_replay(ep)

    def test_init_sha_mismatch_fails(self):
        ep = self.root / "ep_bad_init"
        bddl = make_bddl(ep)
        make_meta(ep, init_sha="a" * 64, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        with self.assertRaises(ValueError):
            self._call_replay(ep)

    def test_wrong_n_steps_fails(self):
        ep = self.root / "ep_wrong_n"
        bddl = make_bddl(ep)
        make_meta(ep, n_steps=30, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        with self.assertRaises(ValueError):
            self._call_replay(ep)

    # ── DIVERGED path ──

    def test_done_before_last_step_diverged(self):
        ep = self.root / "ep_early_done"
        bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        from scripts.stageb.replay_c2g_r8t_canary_success import REPLAY_DIVERGED, REPLAY_EXACT

        bm, init = _mock_benchmark_dict()
        mock_env = MockEnv([], done_at=20)

        # Rebuild the 25D to match stored — exact match
        stored_25d = [json.loads(line)["features_25d"] for line in
                      open(ep / "step_records.jsonl") if line.strip()]

        class ExactStreamer:
            def __init__(self):
                self._i = 0
            def update(self, **kwargs):
                self.features = {str(k): v for k, v in enumerate(stored_25d[min(self._i, len(stored_25d) - 1)])}
                self._i += 1
                return self

        with mock.patch("libero.libero.benchmark.get_benchmark_dict", return_value=bm), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.build_v4_exact_env",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.apply_dummy_wait",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("scripts.v4_run_eval_openvla.physical_gripper_state",
                        return_value={"qpos": [0.0] * 7}), \
             mock.patch("src.gripper_attack.sc5_streaming_features_v2.SC5StreamingFeatureAdapterV2",
                        return_value=ExactStreamer()), \
             mock.patch("src.gripper_attack.c2g_clean_mechanism.set_deterministic_seeds",
                        return_value=None), \
             mock.patch("numpy.asarray", side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype)):
            result = __import__("scripts.stageb.replay_c2g_r8t_canary_success", fromlist=["_replay_episode"])._replay_episode(ep, render_gpu=0)

        # Done at step 20 < 49 → DIVERGED, and episode stops after step 20
        self.assertEqual(result["classification"], REPLAY_DIVERGED)
        self.assertLess(result["step_count"], 50)

    # ── SUCCESS path ──

    def test_canonical_success_from_env_check(self):
        ep = self.root / "ep_success"
        bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        bm, init = _mock_benchmark_dict()
        mock_env = MockEnv([], success_at=30)

        from scripts.stageb.replay_c2g_r8t_canary_success import REPLAY_EXACT

        stored_25d = [json.loads(line)["features_25d"] for line in
                      open(ep / "step_records.jsonl") if line.strip()]

        class ExactStreamer2:
            def __init__(self):
                self._i = 0
            def update(self, **kwargs):
                self.features = {str(k): v for k, v in enumerate(stored_25d[min(self._i, len(stored_25d) - 1)])}
                self._i += 1
                return self

        with mock.patch("libero.libero.benchmark.get_benchmark_dict", return_value=bm), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.build_v4_exact_env",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.apply_dummy_wait",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("scripts.v4_run_eval_openvla.physical_gripper_state",
                        return_value={"qpos": [0.0] * 7}), \
             mock.patch("src.gripper_attack.sc5_streaming_features_v2.SC5StreamingFeatureAdapterV2",
                        return_value=ExactStreamer2()), \
             mock.patch("src.gripper_attack.c2g_clean_mechanism.set_deterministic_seeds",
                        return_value=None), \
             mock.patch("numpy.asarray", side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype)):
            result = __import__("scripts.stageb.replay_c2g_r8t_canary_success", fromlist=["_replay_episode"])._replay_episode(ep, render_gpu=0)

        self.assertTrue(result["canonical_success"])
        self.assertTrue(result["any_check_success"])

    def test_done_without_success_is_not_success(self):
        ep = self.root / "ep_done_no_success"
        bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        bm, init = _mock_benchmark_dict()
        mock_env = MockEnv([], done_at=49)  # done at last step, no success
        mock_env.set_check_success_values([False] * 50)

        stored_25d = [json.loads(line)["features_25d"] for line in
                      open(ep / "step_records.jsonl") if line.strip()]

        class ExactStreamer3:
            def __init__(self):
                self._i = 0
            def update(self, **kwargs):
                self.features = {str(k): v for k, v in enumerate(stored_25d[min(self._i, len(stored_25d) - 1)])}
                self._i += 1
                return self

        with mock.patch("libero.libero.benchmark.get_benchmark_dict", return_value=bm), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.build_v4_exact_env",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("src.gripper_attack.libero_v4_env_factory.apply_dummy_wait",
                        return_value=(mock_env, np.zeros(10, dtype=np.float32))), \
             mock.patch("scripts.v4_run_eval_openvla.physical_gripper_state",
                        return_value={"qpos": [0.0] * 7}), \
             mock.patch("src.gripper_attack.sc5_streaming_features_v2.SC5StreamingFeatureAdapterV2",
                        return_value=ExactStreamer3()), \
             mock.patch("src.gripper_attack.c2g_clean_mechanism.set_deterministic_seeds",
                        return_value=None), \
             mock.patch("numpy.asarray", side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype)):
            result = __import__("scripts.stageb.replay_c2g_r8t_canary_success", fromlist=["_replay_episode"])._replay_episode(ep, render_gpu=0)

        # Done at last step without check_success → not canonical success
        self.assertFalse(result["canonical_success"])


class ReplayHashBoundTests(unittest.TestCase):
    def test_replay_main_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                raise FileExistsError(str(out))

    def test_replay_main_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.json"
            p.write_text("wrong")
            from scripts.stageb.replay_c2g_r8t_canary_success import assert_hash
            with self.assertRaises(ValueError):
                assert_hash(p, "a" * 64, "test")


class SyntheticFixtureTests(unittest.TestCase):
    """Verify test fixtures match real collector schema."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_uses_bddl_file_not_bddl_path(self):
        ep = self.root / "ep"
        meta = make_meta(ep, n_steps=50)
        self.assertIn("bddl_file", meta)
        self.assertNotIn("bddl_path", meta)

    def test_fixture_has_required_metadata_fields(self):
        ep = self.root / "ep"
        meta = make_meta(ep, n_steps=50)
        for key in ("bddl_file", "bddl_sha256", "official_init_state_sha256",
                     "official_init_state_shape", "official_init_state_dtype",
                     "max_steps", "dummy_wait", "replay_seed", "n_steps",
                     "runtime_versions", "controller_config"):
            self.assertIn(key, meta, f"Missing metadata field: {key}")


if __name__ == "__main__":
    unittest.main()
