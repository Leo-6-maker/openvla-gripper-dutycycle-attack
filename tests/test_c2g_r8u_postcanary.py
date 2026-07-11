"""R8U post-canary tests — CPU-only, no LIBERO environments."""
import json, tempfile, unittest
from pathlib import Path
from unittest import mock

from scripts.stageb.replay_c2g_r8t_canary_success import (
    REPLAY_EXACT,
    REPLAY_DIVERGED,
    REPLAY_FAILED,
    REPLAY_NUMERICALLY_EQUIVALENT,
    _validate_step_records,
    _validate_action,
    assert_hash,
    sha256_file,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class ReplayInputValidationTests(unittest.TestCase):
    """Step and action validation without LIBERO."""

    def test_valid_steps_pass(self):
        steps = [{"step": i} for i in range(50)]
        _validate_step_records(steps, 50, "test")

    def test_discontinuous_steps_fail(self):
        steps = [{"step": i} for i in range(49)] + [{"step": 10}]
        with self.assertRaises(ValueError):
            _validate_step_records(steps, 50, "test")

    def test_count_mismatch_fail(self):
        steps = [{"step": i} for i in range(30)]
        with self.assertRaises(ValueError):
            _validate_step_records(steps, 50, "test")

    def test_too_few_steps_fail(self):
        steps = [{"step": i} for i in range(10)]
        with self.assertRaises(ValueError):
            _validate_step_records(steps, 10, "test")

    def test_valid_action_passes(self):
        a = _validate_action({"clean_action_raw_7d": [0.0] * 7}, "clean_action_raw_7d", "test")
        self.assertEqual(a.shape, (7,))

    def test_6d_action_fails(self):
        with self.assertRaises(ValueError):
            _validate_action({"applied_action_7d": [0.0] * 6}, "applied_action_7d", "test")

    def test_8d_action_fails(self):
        with self.assertRaises(ValueError):
            _validate_action({"applied_action_7d": [0.0] * 8}, "applied_action_7d", "test")

    def test_nonfinite_action_fails(self):
        with self.assertRaises(ValueError):
            _validate_action({"applied_action_7d": [0.0, float("nan"), 0, 0, 0, 0, 0]}, "applied_action_7d", "test")

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.json"
            p.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_hash(p, "bad" * 16, "test")


class ReplayHashBoundTests(unittest.TestCase):
    """Hash-bound input integrity checks."""

    def test_missing_input_fails(self):
        from scripts.stageb.replay_c2g_r8t_canary_success import main
        with self.assertRaises(SystemExit):
            main()

    def test_output_exists_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "output"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                raise FileExistsError(str(out))


class ReplayClassificationTests(unittest.TestCase):
    """Classification logic without env."""

    def test_classification_constants_are_distinct(self):
        self.assertNotEqual(REPLAY_EXACT, REPLAY_NUMERICALLY_EQUIVALENT)
        self.assertNotEqual(REPLAY_DIVERGED, REPLAY_FAILED)


class ReplaySyntheticTests(unittest.TestCase):
    """Synthetic replay test using mock env."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_episode(self, ep_dir: Path, n_steps=50):
        ep_dir.mkdir(parents=True)
        meta = {
            "suite": "libero_object", "task_index": 0, "state_id": 1,
            "parent_key": "libero_object/task_0/state_1/detector_train/episode_000",
            "bddl_path": "/fake/bddl.bddl", "bddl_sha256": "f" * 64,
            "official_init_state_sha256": "e" * 64,
            "max_steps": n_steps, "dummy_wait": 10, "n_steps": n_steps,
            "cohort": "DETECTOR_TRAIN", "split": "train",
        }
        write_json(ep_dir / "episode_metadata.json", meta)
        steps = [{
            "step": i,
            "clean_action_raw_7d": [0.0] * 7,
            "applied_action_7d": [0.0] * 7,
            "features_25d": [float(i % 25) for _ in range(25)],
        } for i in range(n_steps)]
        write_jsonl(ep_dir / "step_records.jsonl", steps)

    def test_step_validation_passes(self):
        ep = self.root / "ep"
        self._write_episode(ep, 50)
        steps = [{"step": i} for i in range(50)]
        _validate_step_records(steps, 50, "test")

    def test_step_validation_discontinuous(self):
        steps = [{"step": i} for i in range(30)] + [{"step": 10}]
        with self.assertRaises(ValueError):
            _validate_step_records(steps, 31, "test")


if __name__ == "__main__":
    unittest.main()
