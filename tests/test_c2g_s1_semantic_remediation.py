import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from src.gripper_attack.c2g_teacher_v2_contact_identity import (
    analyze_contact_pairs,
    finger_side,
)
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets
from tools.multisuite_detector.audit_c2g_static_assets import audit_static_assets


class S1SemanticRemediationTests(unittest.TestCase):
    def test_strict_inventory_cli_runs_outside_repo(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [sys.executable, str(repo / "tools/multisuite_detector/audit_c2g_static_assets_strict.py"), "--help"],
                cwd=td,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compact_turnon_is_canonicalized_before_target_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turn_on_stove.bddl"
            path.write_text(
                """
                (define (problem p)
                  (:objects flat_stove_1 - stove)
                  (:goal (and (turnon flat_stove_1))))
                """,
                encoding="utf-8",
            )
            metadata = parse_bddl_task_metadata(path)
            self.assertEqual(metadata["goal_predicates"], [["turn_on", "flat_stove_1"]])
            result = resolve_task_targets(metadata)
            self.assertEqual(result.resolved_manipulable_entities, ("flat_stove_1",))
            self.assertNotIn("UNSUPPORTED_OPERATORS", " ".join(result.ambiguities))

    def test_numbered_panda_finger_joint_aliases_are_two_distinct_jaws(self):
        expected = {
            "finger_joint1": "left",
            "finger_joint1_tip": "left",
            "robot0_finger_joint1_tip": "left",
            "finger_joint2": "right",
            "finger_joint2_tip": "right",
            "robot0_finger_joint2_tip": "right",
        }
        self.assertEqual({name: finger_side(name) for name in expected}, expected)

        identity = analyze_contact_pairs(
            [
                ("finger_joint1_tip", "milk_1_collision"),
                ("finger_joint2_tip", "milk_1_visual"),
            ],
            object_names=["milk_1"],
        )
        self.assertTrue(identity.left_finger_contact)
        self.assertTrue(identity.right_finger_contact)
        self.assertTrue(identity.bilateral_grasp_candidate)
        self.assertEqual(identity.contacted_objects, ("milk_1",))

    def test_alias_aware_static_inventory_has_no_semantic_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks = root / "tasks"
            xml = root / "xml"
            tasks.mkdir()
            xml.mkdir()
            (tasks / "task.bddl").write_text(
                "(define (problem p) (:goal (and (turnon flat_stove_1))))",
                encoding="utf-8",
            )
            (xml / "panda.xml").write_text(
                """
                <mujoco><worldbody><body name="robot0">
                  <geom name="finger_joint1_tip"/>
                  <geom name="finger_joint2_tip"/>
                </body></worldbody></mujoco>
                """,
                encoding="utf-8",
            )
            report = audit_static_assets(
                [tasks],
                [xml],
                supported_operators={"turn_on", "turnon"},
                finger_side_fn=finger_side,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["task_inventory"]["unsupported_operators"], [])
            self.assertEqual(report["xml_inventory"]["unresolved_finger_candidates"], [])
            self.assertEqual(
                report["xml_inventory"]["finger_aliases"],
                {
                    "left": ["finger_joint1_tip"],
                    "right": ["finger_joint2_tip"],
                },
            )


if __name__ == "__main__":
    unittest.main()
