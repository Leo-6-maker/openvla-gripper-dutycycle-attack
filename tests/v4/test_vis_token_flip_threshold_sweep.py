"""Lightweight tests for VIS threshold diagnostic sweep plumbing.

These tests do not load OpenVLA weights. They verify that the real diagnostic
script reuses a one-frame context across objective/epsilon/step combinations.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = REPO_ROOT / "scripts" / "diagnostics"


def _load_threshold_module():
    sys.path.insert(0, str(DIAG_DIR))
    path = DIAG_DIR / "vis_token_flip_threshold.py"
    spec = importlib.util.spec_from_file_location("vis_token_flip_threshold_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVisTokenFlipThresholdSweep(unittest.TestCase):
    def test_real_sweep_reuses_one_frame_context(self):
        module = _load_threshold_module()
        calls = {"prepare": 0, "attack": 0}

        def fake_prepare(args):
            calls["prepare"] += 1
            return {"context": "one-frame"}

        def fake_attack(context, args):
            self.assertEqual(context, {"context": "one-frame"})
            calls["attack"] += 1
            return {
                "target_ce_before": 2.0,
                "target_ce_after": 1.0,
                "open_bin_prob_mass_before": 0.1,
                "open_bin_prob_mass_after": 0.2,
                "close_bin_prob_mass_before": 0.9,
                "close_bin_prob_mass_after": 0.8,
                "clean_gripper_token": 1,
                "adv_gripper_token": 1,
                "clean_gripper_action": 0.0,
                "adv_gripper_action": 0.0,
                "gripper_token_flipped": "false",
                "arm_l2": 0.0,
                "perturbation_linf": 0.0,
                "perturbation_l2": 0.0,
                "attack_runtime_sec": 0.01,
                "adv_decode_runtime_sec": 0.01,
                "model_dtype": "mock",
                "pixel_values_dtype": "mock",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "threshold.csv"
            argv = [
                "vis_token_flip_threshold.py",
                "--frame",
                "dummy.png",
                "--model_path",
                "dummy-model",
                "--objective",
                "target_action_ce",
                "--objective",
                "gripper_open_region_ce",
                "--eps",
                "4/255",
                "--eps",
                "8/255",
                "--steps",
                "10",
                "--steps",
                "20",
                "--output_csv",
                str(output_csv),
            ]
            with patch.object(module, "prepare_one_frame_context", fake_prepare):
                with patch.object(module, "run_one_frame_attack", fake_attack):
                    with patch.object(sys, "argv", argv):
                        self.assertEqual(module.main(), 0)

            rows = list(csv.DictReader(output_csv.open()))
            self.assertEqual(calls["prepare"], 1)
            self.assertEqual(calls["attack"], 8)
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["objective"] for row in rows}, {"target_action_ce", "gripper_open_region_ce"})


if __name__ == "__main__":
    unittest.main()
