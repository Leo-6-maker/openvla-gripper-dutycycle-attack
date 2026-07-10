import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.stageb import run_c2f_canary_worker as worker


class DummyConfig:
    norm_stats = {"libero_goal": {}, "libero_10": {}}


class DummyModel:
    norm_stats = {"libero_goal": {}, "libero_10": {}}
    config = DummyConfig()


class C2fTrackAStaticTests(unittest.TestCase):
    def test_deterministic_rand_seed_and_noise(self):
        parent = "libero_object/task_00/state_005/clean/attempt_01"
        seed1 = worker._rand_seed(parent, worker.COND_RAND, 78)
        seed2 = worker._rand_seed(parent, worker.COND_RAND, 78)
        self.assertEqual(seed1, seed2)
        shape = (7,)
        n1 = np.random.default_rng(seed1).standard_normal(shape).astype(np.float32)
        n2 = np.random.default_rng(seed2).standard_normal(shape).astype(np.float32)
        n1 = n1 / (np.linalg.norm(n1) + 1e-8) * worker.EPSILON
        n2 = n2 / (np.linalg.norm(n2) + 1e-8) * worker.EPSILON
        np.testing.assert_allclose(n1, n2)

    def test_runtime_error_writes_invalid_metadata_and_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo = tmp / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "tracked.txt").write_text("ok")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            out = tmp / "out"
            ckpt = tmp / "dummy.pt"
            ckpt.write_bytes(b"not used")
            argv = [
                "worker", "--parent-key", "bad_suite/task_00/state_000/clean/attempt_01",
                "--condition", worker.COND_CLEAN, "--checkpoint", str(ckpt),
                "--output-dir", str(out), "--git-commit", "unit",
            ]
            with mock.patch.object(worker, "REPO", repo), mock.patch.object(sys, "argv", argv):
                rc = worker.main()
            self.assertNotEqual(rc, 0)
            meta = json.loads((out / "bad_suite/task_00/state_000/clean/attempt_01/CLEAN/episode_metadata.json").read_text())
            self.assertFalse(meta["runtime_valid"])
            self.assertIsNone(meta["success"])
            self.assertTrue(meta["error_type"])
            self.assertIn("git_provenance", meta)

    def test_strict_frozen_condition_names(self):
        self.assertEqual({worker.COND_CLEAN, worker.COND_TRUE, worker.COND_RAND}, {
            "CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"
        })
        with self.assertRaises(SystemExit):
            with mock.patch.object(sys, "argv", [
                "worker", "--parent-key", "libero_object/task_00/state_005/clean/attempt_01",
                "--condition", "TRUE_T10", "--checkpoint", "x", "--output-dir", "y",
            ]):
                worker.main()

    def test_goal_norm_stats_key_resolution(self):
        self.assertEqual(worker._norm_stats_keys(DummyModel()), ["libero_10", "libero_goal"])
        self.assertEqual(worker._resolve_unnorm_key("libero_goal", ["libero_10", "libero_goal"]), "libero_goal")
        with self.assertRaises(RuntimeError):
            worker._resolve_unnorm_key("libero_goal", ["libero_10"])

    def test_git_and_file_provenance_clean_tree_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            f = repo / "tracked.txt"
            f.write_text("abc")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            prov = worker._git_provenance(repo, enforce_clean=True)
            self.assertTrue(prov["repo_clean"])
            self.assertEqual(worker._sha256_file(f), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            f.write_text("dirty")
            with self.assertRaises(RuntimeError):
                worker._git_provenance(repo, enforce_clean=True)


if __name__ == "__main__":
    unittest.main()