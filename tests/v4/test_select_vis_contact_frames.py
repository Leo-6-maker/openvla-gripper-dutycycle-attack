"""Tests for no-rollout VIS contact-frame selection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "select_vis_contact_frames.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_vis_contact_frames_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSelectVisContactFrames(unittest.TestCase):
    def test_wait_frame_is_not_contact_candidate(self):
        module = _load_module()
        row = {
            "step_idx": 0,
            "policy_step_idx": -1,
            "phase": "wait",
        }
        score, reasons = module.score_row(row)
        self.assertLess(score, 0)
        self.assertIn("wait_or_prepolicy", reasons)

    def test_selects_contact_candidate_and_checks_frame(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = root / "frames"
            frames.mkdir()
            (frames / "step_0003.png").write_bytes(b"fake")
            step_records = root / "step_records.jsonl"
            rows = [
                {"run_id": "r", "step_idx": 0, "policy_step_idx": -1, "phase": "wait"},
                {
                    "run_id": "r",
                    "step_idx": 3,
                    "policy_step_idx": 3,
                    "phase": "carry",
                    "proxy_lift_carry_gate_active": True,
                    "proxy_lift_carry_eef_z_delta_from_min": 0.05,
                },
            ]
            step_records.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            best = module.best_for_file(step_records)
        self.assertEqual(best["step_idx"], 3)
        self.assertEqual(best["selector_status"], "candidate_frame_available")
        self.assertEqual(best["frame_available"], "true")


if __name__ == "__main__":
    unittest.main()
