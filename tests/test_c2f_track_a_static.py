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

from scripts.stageb import audit_c2f_track_a_run as run_audit
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
        seed3 = worker._rand_seed(parent, worker.COND_RAND, 79)
        seed4 = worker._rand_seed("libero_object/task_00/state_006/clean/attempt_01", worker.COND_RAND, 78)
        self.assertEqual(seed1, seed2)
        self.assertNotEqual(seed1, seed3)
        self.assertNotEqual(seed1, seed4)
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
                "--output-dir", str(out), "--expected-git-commit", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            ]
            with mock.patch.object(worker, "REPO", repo), mock.patch.object(sys, "argv", argv):
                rc = worker.main()
            self.assertNotEqual(rc, 0)
            meta = json.loads((out / "bad_suite/task_00/state_000/clean/attempt_01/CLEAN/episode_metadata.json").read_text())
            self.assertFalse(meta["runtime_valid"])
            self.assertIsNone(meta["success"])
            self.assertTrue(meta["error_type"])
            self.assertIn("git_provenance", meta)

    def test_invalid_metadata_is_archived_for_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ep = root / "out/libero_goal/task_00/state_000/clean/attempt_01/CLEAN"
            ep.mkdir(parents=True)
            (ep / "episode_metadata.json").write_text(json.dumps({"runtime_valid": False, "success": None}))
            self.assertFalse(run_audit.is_runtime_valid_metadata(ep / "episode_metadata.json"))
            moved = run_audit.archive_invalid_attempt(
                root / "out", "libero_goal/task_00/state_000/clean/attempt_01", "CLEAN", root / "invalid_attempts"
            )
            self.assertTrue(moved)
            self.assertFalse(ep.exists())
            self.assertTrue((root / "invalid_attempts/libero_goal/task_00/state_000/clean/attempt_01/CLEAN/attempt_001/episode_metadata.json").exists())

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

    def test_goal_manifest_sha_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "status": "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED",
                "model_path": str(model),
                "unnorm_key": "libero_goal",
                "missing_referenced_shards": [],
            }))
            out = worker._validate_goal_manifest(str(manifest), model.resolve(), "libero_goal")
            self.assertEqual(out["policy_model_manifest_path"], str(manifest.resolve()))
            self.assertEqual(out["policy_model_manifest_sha256"], worker._sha256_file(manifest))

    def test_action_trace_fields_and_noise_norm(self):
        clean = np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6, -0.7], dtype=np.float32)
        noise = np.array([0, 0, 0, 0, 0, 0, worker.EPSILON], dtype=np.float32)
        attacked = np.clip(clean + noise, -1.0, 1.0)
        ev = worker._action_evidence(clean, attacked, clean, attacked, noise, worker.COND_RAND)
        for key in [
            "clean_raw_action", "intervened_raw_action", "executed_env_action", "action_delta",
            "rand_noise_vector", "rand_noise_norm", "clean_gripper_raw", "intervened_gripper_raw",
            "clean_gripper_env", "executed_gripper_env",
        ]:
            self.assertIn(key, ev)
        self.assertAlmostEqual(ev["rand_noise_norm"], worker.EPSILON, places=6)

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
            with self.assertRaises(RuntimeError):
                worker._git_provenance(repo, enforce_clean=True, expected_commit="0" * 40)
            self.assertEqual(worker._sha256_file(f), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            f.write_text("dirty")
            with self.assertRaises(RuntimeError):
                worker._git_provenance(repo, enforce_clean=True)

    def test_launcher_uses_dynamic_worktree_not_legacy_repo(self):
        for rel in [
            "scripts/stageb/run_c2f_table1_candidate_gpu17.sh",
            "scripts/stageb/run_c2f_track_a_smoke5.sh",
        ]:
            text = Path(rel).read_text(encoding="utf-8-sig")
            self.assertIn("CODE_REPO=", text)
            self.assertNotIn("REPO=/mnt/sdc/dty_user/openvla_attack", text)


if __name__ == "__main__":
    unittest.main()
