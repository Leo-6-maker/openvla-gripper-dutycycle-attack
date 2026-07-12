"""Test R8Y full-500 audit contract (authorization enforcement)."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8y_l10_520_full500 import (
    PASS_STATUS,
    HOLD_STATUS,
    audit_full500,
)


class FullAuditAuthorizationTests(unittest.TestCase):
    def test_unauthorized_cannot_run(self):
        """The full500 audit gating is enforced in main() via --authorization flag."""
        import subprocess, sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "schema": "c2g.r8y.l10_520_plan.2026-07-12.v1",
                "status": "PASS_C2G_R8Y_L10_520_PLAN",
                "shard_index": str(root / "shard_index.json"),
            }))
            (root / "old_l10").mkdir()
            # Calling audit_full500 directly bypasses the CLI gate;
            # the CLI --authorization flag is the enforcement point.
            # This test verifies the gate works at the CLI layer.
            result = subprocess.run(
                [sys.executable, "-m", "tools.multisuite_detector.audit_c2g_r8y_l10_520_full500",
                 "--collection-root", str(root),
                 "--plan-report", str(plan),
                 "--old-l10-source-root", str(root / "old_l10"),
                 "--authorization", "WRONG_TOKEN"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)


class FullAuditEmptyCollectionTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _make_worker_with_episodes(self, wid: str, gpu: int, episodes: list[dict]):
        worker_dir = self.root / "workers" / wid
        worker_dir.mkdir(parents=True)
        receipt = {
            "schema": "c2g.r8y.l10_520_worker_receipt.2026-07-12.v1",
            "status": "PASS_C2G_R8Y_L10_520_SHARD_RUN",
            "worker_id": wid,
            "physical_gpu": gpu,
            "assigned_physical_gpu": gpu,
            "cuda_visible_devices": str(gpu),
        }
        (worker_dir / "worker_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        ep_dir = worker_dir / "collection" / "episodes" / "libero_10" / "libero_10"
        for ep in episodes:
            ep_path = ep_dir / f"task_{ep['task_index']}" / f"state_{ep['state_id']}" / ep["cohort"].lower() / f"episode_{ep['local']:03d}"
            ep_path.mkdir(parents=True)
            meta = {
                "parent_key": ep["parent_key"],
                "suite": "libero_10",
                "task_index": ep["task_index"],
                "state_id": ep["state_id"],
                "cohort": ep["cohort"],
                "max_policy_steps": 520,
                "dummy_wait_steps": 10,
                "physical_gpu": gpu,
                "clean_success_observed": ep.get("success", False),
            }
            (ep_path / "episode_metadata.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            (ep_path / "step_records.jsonl").write_text(
                json.dumps({"policy_step": 0, "raw_action_7d": [0.0]*7}) + "\n",
                encoding="utf-8",
            )

    def test_full500_with_complete_data(self):
        # Build 500 episodes across 20 workers
        # Each task (0-9) gets 50 states, each state gets exactly 1 episode
        cohorts = [
            ("DETECTOR_TRAIN", 300),
            ("DETECTOR_VAL", 50),
            ("DETECTOR_TEST_WITHIN_TASK", 50),
            ("ATTACK_EVAL_PREREGISTERED", 100),
        ]
        eps = []
        idx = 0
        for cohort, count in cohorts:
            for local in range(count):
                gpu = [4, 5, 6, 7][idx % 4]
                task_index = idx // 50
                state_id = idx % 50
                eps.append({
                    "parent_key": f"libero_10/task_{task_index}/state_{state_id}/{cohort.lower()}/episode_{local:03d}",
                    "suite": "libero_10",
                    "task_index": task_index,
                    "state_id": state_id,
                    "cohort": cohort,
                    "local": local,
                    "gpu": gpu,
                    "success": idx % 3 == 0,
                })
                idx += 1

        # Distribute to 20 workers (5 per GPU, 25 each)
        for gpu in [4, 5, 6, 7]:
            gpu_eps = [e for e in eps if e["gpu"] == gpu]
            for s in range(5):
                wid = f"g{gpu}_l10_s{s}"
                shard_eps = gpu_eps[s * 25 : (s + 1) * 25]
                self._make_worker_with_episodes(wid, gpu, shard_eps)

        plan = self.root / "plan.json"
        plan.write_text(json.dumps({
            "schema": "c2g.r8y.l10_520_plan.2026-07-12.v1",
            "status": "PASS_C2G_R8Y_L10_520_PLAN",
            "shard_index": str(self.root / "shard_index.json"),
        }))

        old = self.root / "old_l10"
        old.mkdir()

        # This should work when authorized
        report = audit_full500(
            collection_root=self.root,
            plan_report=plan,
            old_l10_source_root=old,
        )
        self.assertEqual(report["episode_count"], 500)
        self.assertEqual(report["unique_identities"], 500)
        self.assertEqual(report["worker_receipt_count"], 20)
        self.assertEqual(report["max_steps_520_count"], 500)
        self.assertEqual(report["dummy_wait_10_count"], 500)
        self.assertEqual(report["runtime_failed"], 0)
        self.assertEqual(report["gpu_migration_count"], 0)

    def test_duplicate_identity_detected(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({
            "schema": "c2g.r8y.l10_520_plan.2026-07-12.v1",
            "status": "PASS_C2G_R8Y_L10_520_PLAN",
        }))
        old = self.root / "old_l10"
        old.mkdir()

        # Create duplicate episode
        ep = {
            "parent_key": "libero_10/task_0/state_0/detector_train/episode_000",
            "task_index": 0,
            "state_id": 0,
            "cohort": "DETECTOR_TRAIN",
            "local": 0,
            "gpu": 4,
        }
        self._make_worker_with_episodes("g4_l10_s0", 4, [ep])
        self._make_worker_with_episodes("g4_l10_s1", 4, [ep])  # same identity!

        report = audit_full500(
            collection_root=self.root,
            plan_report=plan,
            old_l10_source_root=old,
        )
        self.assertGreater(report["duplicate_identities"], 0)
        self.assertFalse(report["status"].startswith("PASS"))


if __name__ == "__main__":
    unittest.main()
