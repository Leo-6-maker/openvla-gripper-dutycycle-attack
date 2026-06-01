"""Lightweight tests for the VIS arm-drift diagnostic harness."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = REPO_ROOT / "scripts" / "diagnostics"


def _load_arm_module():
    sys.path.insert(0, str(DIAG_DIR))
    path = DIAG_DIR / "vis_arm_drift_sweep.py"
    spec = importlib.util.spec_from_file_location("vis_arm_drift_sweep_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVisArmDriftSweep(unittest.TestCase):
    def test_dry_run_writes_schema(self):
        module = _load_arm_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "arm.csv"
            argv = [
                "vis_arm_drift_sweep.py",
                "--dry-run",
                "--output_csv",
                str(output_csv),
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = old_argv
            rows = list(csv.DictReader(output_csv.open()))
            self.assertEqual(rows, [])
            header = output_csv.read_text().splitlines()[0].split(",")
            self.assertIn("loss_variant", header)
            self.assertIn("random_baseline_gripper_delta", header)
            self.assertIn("gripper_to_arm_ratio", header)

    def test_ratio_uses_epsilon_denominator(self):
        module = _load_arm_module()
        self.assertAlmostEqual(module._ratio(2.0, 0.0), 2.0 / 1e-6)
        self.assertAlmostEqual(module._ratio(-2.0, 4.0), 0.5)


if __name__ == "__main__":
    unittest.main()
