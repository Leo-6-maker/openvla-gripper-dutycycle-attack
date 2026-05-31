"""Tests for clean-only VIS contact frame collection planning."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "build_vis_contact_frame_collection_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_vis_contact_frame_collection_plan_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildVisContactFrameCollectionPlan(unittest.TestCase):
    def test_selects_only_missing_contact_rows(self):
        module = _load_module()
        rows = [
            {"suite": "libero_object", "selector_status": "candidate_missing_frame", "frame_available": "false"},
            {"suite": "libero_object", "selector_status": "frame_available_but_not_contact", "frame_available": "true"},
            {"suite": "libero_goal", "selector_status": "candidate_missing_frame", "frame_available": "false"},
        ]
        selected = module.select_missing_contact_rows(rows)
        self.assertEqual(len(selected), 1)

    def test_plan_is_clean_only_and_clips_frame_window(self):
        module = _load_module()
        row = {
            "run_id": "obj_ketchup_s0",
            "suite": "libero_object",
            "task_id": "object_pick_up_the_ketchup_and_place_it_in_the_basket",
            "task_name": "pick up the ketchup and place it in the basket",
            "state_id": "",
            "seed": "0",
            "step_records": "/tmp/step_records.jsonl",
            "step_idx": "1",
            "score": "14",
            "reason": "policy_step",
            "frame_available": "false",
        }
        plan = module.build_plan_row(
            row,
            output_root="/tmp/out",
            model_path="/models/object",
            cuda_visible_devices="4,5",
            render_gpu_device_id=0,
            frame_context=2,
            run_id_prefix="vis_contact_frame_clean",
            num_steps_wait=10,
            max_steps=280,
            attention_backend="eager",
        )
        self.assertEqual(plan["libero_task_index"], 4)
        self.assertEqual(plan["state_id"], 0)
        self.assertEqual(plan["frame_window_start"], 0)
        self.assertEqual(plan["frame_window_end"], 3)
        self.assertEqual(plan["attack_enabled"], "false")
        self.assertEqual(plan["vis_enabled"], "false")
        self.assertEqual(plan["sus30_enabled"], "false")
        self.assertEqual(plan["detector_enabled"], "false")
        self.assertIn("--attack_condition clean", plan["collection_command"])
        self.assertIn("--cuda_visible_devices 4,5", plan["collection_command"])
        self.assertNotIn("--cuda_visible_devices 0", plan["collection_command"])
        self.assertEqual(plan["launch_requires_explicit_approval"], "true")

    def test_main_writes_plan_without_execution(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit = root / "audit.csv"
            with audit.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_id",
                        "suite",
                        "task_id",
                        "task_name",
                        "state_id",
                        "seed",
                        "step_records",
                        "step_idx",
                        "score",
                        "selector_status",
                        "reason",
                        "frame_available",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "obj_cream_cheese_s0",
                        "suite": "libero_object",
                        "task_id": "object_pick_up_the_cream_cheese_and_place_it_in_the_basket",
                        "task_name": "pick up the cream cheese and place it in the basket",
                        "state_id": "",
                        "seed": "0",
                        "step_records": "/tmp/step_records.jsonl",
                        "step_idx": "143",
                        "score": "14",
                        "selector_status": "candidate_missing_frame",
                        "reason": "policy_step",
                        "frame_available": "false",
                    }
                )
            out_csv = root / "plan.csv"
            out_md = root / "report.md"
            # Exercise the pure functions used by main instead of mutating sys.argv.
            rows = module.select_missing_contact_rows(module.read_rows(audit))
            plan = [
                module.build_plan_row(
                    rows[0],
                    output_root="/tmp/out",
                    model_path="/models/object",
                    cuda_visible_devices="4,5",
                    render_gpu_device_id=0,
                    frame_context=2,
                    run_id_prefix="vis_contact_frame_clean",
                    num_steps_wait=10,
                    max_steps=280,
                    attention_backend="eager",
                )
            ]
            module.write_rows(out_csv, plan)
            module.write_report(out_md, plan, audit)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())
            self.assertIn("No rollout", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
