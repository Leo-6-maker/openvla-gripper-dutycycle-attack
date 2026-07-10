import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_static_assets import (
    audit_static_assets,
    extract_goal_operators,
    parse_sexpr,
    write_report,
)


def fake_finger_side(name: str) -> str:
    lowered = name.lower()
    if "left" in lowered or "l_finger" in lowered:
        return "left"
    if "right" in lowered or "r_finger" in lowered:
        return "right"
    return ""


class StaticAssetInventoryTests(unittest.TestCase):
    def test_goal_operator_extraction_handles_logic_and_comments(self):
        text = """
        (define (problem demo)
          ; ignored comment
          (:goal (and
            (in milk_1 basket_1)
            (open drawer_1)
            (not (closed cabinet_1))
            (exists (?x - object) (hold ?x)))))
        """
        self.assertEqual(
            extract_goal_operators(text),
            ("closed", "hold", "in", "open"),
        )

    def test_unbalanced_sexpr_fails(self):
        with self.assertRaises(ValueError):
            parse_sexpr("(:goal (and (in a b))")

    def test_inventory_reports_operator_and_finger_gaps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks = root / "tasks"
            models = root / "models"
            tasks.mkdir()
            models.mkdir()
            (tasks / "task.bddl").write_text("""
              (define (problem p)
                (:goal (and
                  (contains basket_1 milk_1)
                  (open drawer_1)
                  (mystery milk_1))))
            """)
            (models / "robot.xml").write_text("""
              <mujoco>
                <worldbody>
                  <body name="robot0">
                    <geom name="robot0_left_finger_collision"/>
                    <geom name="robot0_right_finger_collision"/>
                    <geom name="center_gripper_pad"/>
                  </body>
                </worldbody>
              </mujoco>
            """)
            report = audit_static_assets(
                [tasks], [models],
                supported_operators={"contains", "open"},
                finger_side_fn=fake_finger_side,
            )
            self.assertEqual(report["status"], "HOLD_WITH_GAPS")
            self.assertEqual(report["task_inventory"]["unsupported_operators"], ["mystery"])
            self.assertEqual(
                report["xml_inventory"]["unresolved_finger_candidates"],
                ["center_gripper_pad"],
            )
            self.assertEqual(
                report["xml_inventory"]["finger_aliases"]["left"],
                ["robot0_left_finger_collision"],
            )
            self.assertEqual(report["artifact_manifest"][0]["relative_path"], "task.bddl")
            self.assertEqual(len(report["artifact_manifest_sha256"]), 64)

    def test_clean_inventory_passes_and_report_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks = root / "tasks"
            models = root / "models"
            tasks.mkdir()
            models.mkdir()
            (tasks / "task.bddl").write_text(
                "(define (problem p) (:goal (and (in milk_1 basket_1) (open drawer_1))))"
            )
            (models / "robot.xml").write_text("""
              <mujoco><worldbody><body name="robot0">
                <geom name="robot0_l_finger_collision"/>
                <geom name="robot0_r_finger_collision"/>
              </body></worldbody></mujoco>
            """)
            report1 = audit_static_assets(
                [tasks], [models],
                supported_operators={"in", "open"},
                finger_side_fn=fake_finger_side,
            )
            report2 = audit_static_assets(
                [tasks], [models],
                supported_operators={"in", "open"},
                finger_side_fn=fake_finger_side,
            )
            self.assertEqual(report1["status"], "PASS")
            self.assertEqual(
                report1["artifact_manifest_sha256"],
                report2["artifact_manifest_sha256"],
            )
            output = root / "report.json"
            write_report(report1, output)
            loaded = json.loads(output.read_text())
            self.assertEqual(loaded["status"], "PASS")
            self.assertTrue(output.with_suffix(".json.sha256").is_file())

    def test_missing_roots_fail_closed_unless_optional(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing"
            report = audit_static_assets(
                [missing], [missing],
                supported_operators={"in"},
                finger_side_fn=fake_finger_side,
            )
            self.assertEqual(report["status"], "HOLD_WITH_GAPS")
            optional = audit_static_assets(
                [], [],
                supported_operators={"in"},
                finger_side_fn=fake_finger_side,
                require_bddl=False,
                require_xml=False,
            )
            self.assertEqual(optional["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
