import json
import tempfile
import unittest
from pathlib import Path

from src.gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets
from tools.multisuite_detector.audit_c2g_clean_window_v2 import (
    audit_clean_window_v2,
    select_balanced_dry_run,
)


def metadata(*, episode_key="libero_object/ep0", mechanism="pick_place_transfer"):
    value = {
        "episode_key": episode_key,
        "suite": "libero_object",
        "task_index": 0,
        "object_declarations": ["milk", "ketchup"],
        "receptacle_declarations": ["basket"],
        "structured_goal_metadata": {
            "target_objects": ["milk"],
            "target_receptacles": ["basket"],
        },
        "gripper_command_semantics": "positive_is_close",
    }
    if mechanism:
        value["mechanism_type"] = mechanism
    return value


def contacts(entity="milk"):
    return [
        ["robot0_left_finger_collision", f"{entity}_collision"],
        ["robot0_right_finger_collision", f"{entity}_collision"],
    ]


def positive_rows(count=4):
    return [
        {
            "step": step,
            "contact_pairs": contacts(),
            "gripper_command": 1.0,
            "object_relative_lift": 0.03,
            "near_target": False,
        }
        for step in range(count)
    ]


def write_episode(root: Path, meta, rows):
    episode = root / "libero_object" / Path(meta["episode_key"]).name
    episode.mkdir(parents=True)
    (episode / "episode_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (episode / "step_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return episode


class CleanMechanismTests(unittest.TestCase):
    def test_structured_pick_place_without_task_index_lookup(self):
        meta = metadata(mechanism="")
        resolution = resolve_task_targets(meta)
        self.assertEqual(
            infer_clean_mechanism_type(meta, resolution=resolution),
            "pick_place_transfer",
        )

    def test_multi_object_and_articulated_routes(self):
        multi = metadata(mechanism="")
        multi["structured_goal_metadata"]["target_objects"] = ["milk", "ketchup"]
        self.assertEqual(infer_clean_mechanism_type(multi), "multi_object_transfer")

        articulated = {
            "fixture_declarations": ["drawer"],
            "structured_goal_metadata": {"target_fixtures": ["drawer"]},
        }
        self.assertEqual(infer_clean_mechanism_type(articulated), "articulated_object")

    def test_ambiguous_or_unknown_is_not_guessed_from_language(self):
        meta = {
            "task_language": "put the milk in the basket",
            "object_declarations": ["milk"],
            "receptacle_declarations": ["basket"],
        }
        self.assertEqual(infer_clean_mechanism_type(meta), "unsupported_or_unknown")

    def test_unknown_explicit_mechanism_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown explicit"):
            infer_clean_mechanism_type({"mechanism_type": "magic_task"})


class CleanWindowServerAuditTests(unittest.TestCase):
    def test_cpu_dry_audit_passes_and_writes_external_hash_bound_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            output = Path(td) / "external_audit"
            write_episode(root, metadata(), positive_rows())
            report, episodes, problems = audit_clean_window_v2(
                root,
                episodes_per_suite=2,
                burst_length=3,
                output_dir=output,
                repo_root=Path(td) / "fake_repo",
            )
            self.assertEqual(report["status"], "PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT")
            self.assertEqual(report["processed_episode_count"], 1)
            self.assertEqual(report["critical_positive_row_count"], 4)
            self.assertEqual(report["attack_start_row_count"], 1)
            self.assertEqual(report["input_manifest_file_count"], 2)
            self.assertEqual(len(report["input_manifest_sha256"]), 64)
            self.assertEqual(problems, [])
            self.assertEqual(len(episodes), 1)
            self.assertTrue((output / "clean_window_v2_audit_report.json").is_file())
            self.assertTrue((output / "clean_window_v2_dry_labels.jsonl").is_file())

    def test_strict_four_suite_gate_holds_on_partial_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            write_episode(root, metadata(), positive_rows())
            report, _, _ = audit_clean_window_v2(
                root,
                episodes_per_suite=1,
                burst_length=3,
                strict_four_suites=True,
            )
            self.assertTrue(report["status"].startswith("HOLD_"))
            self.assertEqual(set(report["missing_suites"]), {"libero_10", "libero_goal", "libero_spatial"})

    def test_attacked_field_is_reported_and_holds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            rows = positive_rows()
            rows[0]["vis_success"] = True
            write_episode(root, metadata(), rows)
            report, _, problems = audit_clean_window_v2(root, burst_length=3)
            self.assertTrue(report["status"].startswith("HOLD_"))
            self.assertEqual(report["read_error_count"], 1)
            self.assertTrue(any("attacked" in item["error"] for item in problems))

    def test_eligible_all_unknown_episode_is_a_hold_not_a_negative(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            meta = metadata()
            meta.pop("gripper_command_semantics")
            write_episode(root, meta, positive_rows())
            report, episodes, problems = audit_clean_window_v2(root, burst_length=3)
            self.assertTrue(report["status"].startswith("HOLD_"))
            self.assertEqual(episodes[0]["known_rows"], 0)
            self.assertEqual(episodes[0]["unknown_rows"], 4)
            self.assertTrue(any(item["reason"] == "ELIGIBLE_EPISODE_HAS_ZERO_KNOWN_ROWS" for item in problems))

    def test_absolute_z_alone_never_becomes_positive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            rows = [
                {
                    "step": step,
                    "contact_pairs": contacts(),
                    "gripper_command": 1.0,
                    "eef_z": 1.2,
                    "near_target": False,
                }
                for step in range(4)
            ]
            write_episode(root, metadata(), rows)
            report, episodes, _ = audit_clean_window_v2(root, burst_length=3)
            self.assertTrue(report["status"].startswith("HOLD_"))
            self.assertEqual(episodes[0]["critical_positive_rows"], 0)
            self.assertEqual(episodes[0]["known_rows"], 0)

    def test_output_inside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            root = repo / "input"
            write_episode(root, metadata(), positive_rows())
            with self.assertRaisesRegex(ValueError, "outside"):
                audit_clean_window_v2(
                    root,
                    output_dir=repo / "audit_outputs",
                    repo_root=repo,
                )

    def test_balanced_selection_prefers_one_eligible_and_one_boundary(self):
        records = [
            {"suite": "libero_object", "mechanism_type": "unsupported_or_unknown", "relative_parent": "b"},
            {"suite": "libero_object", "mechanism_type": "pick_place_transfer", "relative_parent": "c"},
            {"suite": "libero_object", "mechanism_type": "pick_place_transfer", "relative_parent": "a"},
        ]
        selected = select_balanced_dry_run(records, episodes_per_suite=2)
        self.assertEqual([item["relative_parent"] for item in selected], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
