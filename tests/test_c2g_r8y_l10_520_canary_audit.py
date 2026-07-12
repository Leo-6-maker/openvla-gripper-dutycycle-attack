"""Test R8Y canary audit gate logic."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8y_l10_520_canary import (
    PASS_STATUS,
    HOLD_STATUS,
    SCHEMA,
    audit_canary,
)


def _make_canary_worker(
    root: Path, wid: str, gpu: int, status: str = "PASS"
) -> Path:
    worker_dir = root / "workers" / wid
    worker_dir.mkdir(parents=True)
    receipt = {
        "schema": "c2g.r8y.l10_520_worker_receipt.2026-07-12.v1",
        "status": status,
        "worker_id": wid,
        "physical_gpu": gpu,
        "assigned_physical_gpu": gpu,
        "cuda_visible_devices": str(gpu),
    }
    (worker_dir / "worker_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return worker_dir


class CanaryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_all_12_pass_gives_canary_pass(self):
        for gpu in [4, 5, 6, 7]:
            for i in range(3):
                _make_canary_worker(self.root, f"g{gpu}_l10_s{i}", gpu, "PASS_C2G_R8Y_L10_520_SHARD_RUN")

        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"status": "PASS_C2G_R8Y_L10_520_PLAN"}))

        old = self.root / "old_l10"
        old.mkdir()

        report = audit_canary(
            canary_root=self.root,
            old_l10_source_root=old,
            plan_report=plan,
        )
        self.assertEqual(report["runtime_valid"], 12)
        self.assertEqual(report["oom_count"], 0)
        self.assertEqual(report["gpu_migration_count"], 0)
        self.assertTrue(report["canary_pass"])

    def test_missing_worker_fails(self):
        for gpu in [4, 5, 6, 7]:
            for i in range(3):
                _make_canary_worker(self.root, f"g{gpu}_l10_s{i}", gpu)

        # Remove one worker receipt
        receipt = self.root / "workers" / "g4_l10_s0" / "worker_receipt.json"
        receipt.unlink()

        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"status": "PASS_C2G_R8Y_L10_520_PLAN"}))

        old = self.root / "old_l10"
        old.mkdir()

        report = audit_canary(
            canary_root=self.root,
            old_l10_source_root=old,
            plan_report=plan,
        )
        self.assertGreater(report["runtime_failed"], 0)
        self.assertFalse(report["canary_pass"])

    def test_gpu_migration_detected(self):
        worker_dir = _make_canary_worker(self.root, "g4_l10_s0", 4)
        receipt = json.loads(
            (worker_dir / "worker_receipt.json").read_text(encoding="utf-8")
        )
        receipt["cuda_visible_devices"] = "5"  # GPU migration!
        (worker_dir / "worker_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

        for gpu in [4, 5, 6, 7]:
            for i in range(3):
                if gpu == 4 and i == 0:
                    continue
                _make_canary_worker(self.root, f"g{gpu}_l10_s{i}", gpu)

        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"status": "PASS"}))

        old = self.root / "old_l10"
        old.mkdir()

        report = audit_canary(
            canary_root=self.root,
            old_l10_source_root=old,
            plan_report=plan,
        )
        self.assertGreater(report["gpu_migration_count"], 0)

    def test_oom_detected(self):
        _make_canary_worker(
            self.root, "g4_l10_s0", 4, "FAILED_CUDA_OUT_OF_MEMORY"
        )
        for gpu in [4, 5, 6, 7]:
            for i in range(3):
                if gpu == 4 and i == 0:
                    continue
                _make_canary_worker(self.root, f"g{gpu}_l10_s{i}", gpu)

        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"status": "PASS"}))

        old = self.root / "old_l10"
        old.mkdir()

        report = audit_canary(
            canary_root=self.root,
            old_l10_source_root=old,
            plan_report=plan,
        )
        self.assertGreater(report["oom_count"], 0)


if __name__ == "__main__":
    unittest.main()
