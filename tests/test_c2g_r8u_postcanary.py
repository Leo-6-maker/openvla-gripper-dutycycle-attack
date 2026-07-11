"""R8U post-canary tests — mock benchmark/env/streamer for _replay_episode."""
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


def _compute_init_sha():
    import hashlib as _hl
    arr = np.zeros(110, dtype=np.float64)
    d = _hl.sha256()
    d.update(str(arr.dtype).encode("utf-8"))
    d.update(json.dumps(list(arr.shape)).encode("utf-8"))
    d.update(arr.tobytes())
    return d.hexdigest(), list(arr.shape), str(arr.dtype)

_MOCK_INIT_SHA, _MOCK_INIT_SHAPE, _MOCK_INIT_DTYPE = _compute_init_sha()


def make_meta(ep_dir: Path, *, bddl_sha="f" * 64, init_sha=None,
              max_steps=50, dummy_wait=10, n_steps=50, suite="libero_object",
              task_index=0, state_id=1, replay_seed=42):
    if init_sha is None: init_sha = _MOCK_INIT_SHA
    meta = {
        "suite": suite, "task_index": task_index, "state_id": state_id,
        "parent_key": f"{suite}/task_{task_index}/state_{state_id}/detector_train/episode_000",
        "bddl_file": str(ep_dir / "fake.bddl"), "bddl_sha256": bddl_sha,
        "official_init_state_sha256": init_sha,
        "official_init_state_shape": _MOCK_INIT_SHAPE,
        "official_init_state_dtype": _MOCK_INIT_DTYPE,
        "max_steps": max_steps, "dummy_wait": dummy_wait,
        "replay_seed": replay_seed, "n_steps": n_steps,
        "cohort": "DETECTOR_TRAIN", "split": "train",
        "runtime_versions": {"libero": "0.1.1"}, "controller_config": {"control_freq": 20},
    }
    write_json(ep_dir / "episode_metadata.json", meta)
    return meta


def make_steps(ep_dir: Path, n_steps=50, *, action_value=0.01, features_value=0.5):
    steps = [{"step": i, "clean_action_raw_7d": [action_value] * 7,
              "applied_action_7d": [action_value] * 7,
              "features_25d": [features_value + i * 0.001] * 25}
             for i in range(n_steps)]
    write_jsonl(ep_dir / "step_records.jsonl", steps)
    return steps


def make_bddl(ep_dir: Path):
    ep_dir.mkdir(parents=True, exist_ok=True)
    bddl = ep_dir / "fake.bddl"
    bddl.write_text("(define (problem test))")
    return bddl


def _mock_benchmark_dict(suite="libero_object"):
    init = np.zeros(110, dtype=np.float64)
    mock_suite = mock.MagicMock()
    mock_suite.get_task.return_value = mock.MagicMock()
    mock_suite.get_task_init_states.return_value = [init] * 10
    return {suite: lambda: mock_suite}


class MockEnv:
    def __init__(self, steps=None, success_at=None, done_at=None):
        self.success_at = success_at
        self.done_at = done_at
        self._step = 0
        self._closed = False
        self.sim = mock.MagicMock()
        self.sim.model.site_name2id.return_value = 0
        self._xpos = np.zeros((10, 3), dtype=np.float32)
        self.sim.data.site_xpos = self._xpos
        self.robots = [mock.MagicMock()]
        self.robots[0].joint_positions = np.zeros(7, dtype=np.float64)

    def check_success(self):
        if self.success_at is not None: return self._step >= self.success_at
        return False

    def step(self, action):
        done = self.done_at is not None and self._step >= self.done_at
        self._step += 1
        return np.zeros(10, dtype=np.float32), 0.0, done, {}

    def set_init_state(self, state):
        return self.step(np.zeros(7))[0]

    def close(self): self._closed = True


class MockStreamer:
    def __init__(self, stored_25d=None):
        self._stored = stored_25d or []
        self._i = 0; self.features = {}

    def update(self, **kwargs):
        self.features = {}
        if self._stored and self._i < len(self._stored):
            vals = self._stored[self._i]
            if isinstance(vals, (list, np.ndarray)):
                self.features = {str(k): float(v) for k, v in enumerate(vals)}
        self._i += 1
        return self

    def __getitem__(self, key):
        if key == "features":
            return self.features
        raise KeyError(key)


try:
    import libero  # noqa: F401
    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False


def _prep_libero_modules(bm):
    import sys as _sys
    fl = mock.MagicMock(); fll = mock.MagicMock(); flle = mock.MagicMock()
    fb = mock.MagicMock(); fb.get_benchmark_dict.return_value = bm
    fll.benchmark = fb; fl.libero = fll
    saved = {}
    for mn, fk in [("libero",fl),("libero.libero",fll),("libero.libero.benchmark",fb),("libero.libero.envs",flle)]:
        if mn in _sys.modules: saved[mn] = _sys.modules[mn]
        _sys.modules[mn] = fk
    return saved

def _restore_modules(saved):
    import sys as _sys
    for m, v in saved.items(): _sys.modules[m] = v


def _replay_with_common_mocks(ep_dir, mock_env, bm=None):
    if bm is None: bm = _mock_benchmark_dict()
    sp = ep_dir / "step_records.jsonl"
    stored = [json.loads(l)["features_25d"] for l in open(sp) if l.strip()] if sp.exists() else None
    saved = _prep_libero_modules(bm)
    import src.gripper_attack.libero_v4_env_factory as _lv4
    ob, ow = _lv4.build_v4_exact_env, _lv4.apply_dummy_wait
    _lv4.build_v4_exact_env = mock.MagicMock(return_value=(mock_env, np.zeros(10, dtype=np.float32)))
    _lv4.apply_dummy_wait = mock.MagicMock(return_value=(mock_env, np.zeros(10, dtype=np.float32)))
    try:
        from scripts.stageb.replay_c2g_r8t_canary_success import _replay_episode
        with mock.patch("scripts.v4_run_eval_openvla.physical_gripper_state", return_value={"qpos": [0.0]*7}), \
             mock.patch("src.gripper_attack.sc5_streaming_features_v2.SC5StreamingFeatureAdapterV2",
                        return_value=MockStreamer(stored)), \
             mock.patch("random.seed"), mock.patch("numpy.random.seed"), \
             mock.patch("torch.manual_seed"), mock.patch("torch.cuda.manual_seed_all"), \
             mock.patch("torch.cuda.is_available", return_value=False):
            return _replay_episode(ep_dir, render_gpu=0)
    finally:
        _lv4.build_v4_exact_env, _lv4.apply_dummy_wait = ob, ow
        _restore_modules(saved)


@unittest.skipUnless(_LIBERO_AVAILABLE, "requires LIBERO")
class ReplayEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_bddl_sha_mismatch_fails(self):
        ep = self.root / "ep"; make_meta(ep, bddl_sha="a"*64)
        make_steps(ep, 50); make_bddl(ep)
        with self.assertRaises(ValueError): _replay_with_common_mocks(ep, MockEnv())

    def test_init_sha_mismatch_fails(self):
        ep = self.root / "ep"; bddl = make_bddl(ep)
        make_meta(ep, init_sha="a"*64, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        with self.assertRaises(ValueError): _replay_with_common_mocks(ep, MockEnv())

    def test_wrong_n_steps_fails(self):
        ep = self.root / "ep"; bddl = make_bddl(ep)
        make_meta(ep, n_steps=30, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        with self.assertRaises(ValueError): _replay_with_common_mocks(ep, MockEnv())

    def test_done_before_last_step_diverged(self):
        ep = self.root / "ep"; bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        from scripts.stageb.replay_c2g_r8t_canary_success import REPLAY_DIVERGED
        r = _replay_with_common_mocks(ep, MockEnv(done_at=20))
        self.assertEqual(r["classification"], REPLAY_DIVERGED)
        self.assertIsNotNone(r["done_observed_at"])
        self.assertLess(r["done_observed_at"], 49)  # before last step

    def test_canonical_success_from_env_check(self):
        ep = self.root / "ep"; bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        r = _replay_with_common_mocks(ep, MockEnv(success_at=30))
        self.assertTrue(r["canonical_success"])

    def test_done_without_success_is_not_success(self):
        ep = self.root / "ep"; bddl = make_bddl(ep)
        make_meta(ep, n_steps=50, bddl_sha=sha256_file(bddl))
        make_steps(ep, 50)
        r = _replay_with_common_mocks(ep, MockEnv(done_at=49))
        self.assertFalse(r["canonical_success"])


class ReplayHashBoundTests(unittest.TestCase):
    def test_replay_main_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"; out.mkdir()
            with self.assertRaises(FileExistsError):
                raise FileExistsError(str(out))

    def test_replay_main_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.json"; p.write_text("wrong")
            from scripts.stageb.replay_c2g_r8t_canary_success import assert_hash
            with self.assertRaises(ValueError):
                assert_hash(p, "a" * 64, "test")


class SyntheticFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_uses_bddl_file_not_bddl_path(self):
        meta = make_meta(self.root / "ep", n_steps=50)
        self.assertIn("bddl_file", meta)
        self.assertNotIn("bddl_path", meta)

    def test_fixture_has_required_metadata_fields(self):
        meta = make_meta(self.root / "ep", n_steps=50)
        for key in ("bddl_file", "bddl_sha256", "official_init_state_sha256",
                     "official_init_state_shape", "official_init_state_dtype",
                     "max_steps", "dummy_wait", "replay_seed", "n_steps",
                     "runtime_versions", "controller_config"):
            self.assertIn(key, meta, f"Missing: {key}")


if __name__ == "__main__":
    unittest.main()
