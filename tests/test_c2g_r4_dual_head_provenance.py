import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.bind_c2g_collection_model_provenance import (
    collection_artifact_rows,
)
from scripts.stageb.build_c2g_suite_model_map import sha256_file
from tools.multisuite_detector.bind_c2g_r4_dual_head_provenance import (
    PASS_STATUS,
    build_binding,
    verify_binding,
)


class R4DualHeadProvenanceTests(unittest.TestCase):
    def fixture(self, root: Path):
        collection = root / "clean_collection"
        episode = (
            collection
            / "episodes"
            / "libero_10"
            / "task_0"
            / "episode_000"
        )
        episode.mkdir(parents=True)
        collection_head = "a" * 40
        audit_head = "b" * 40
        metadata = episode / "episode_metadata.json"
        steps = episode / "step_records.jsonl"
        metadata.write_text(
            json.dumps(
                {
                    "episode_key": (
                        "libero_10/task_0/state_9/train/episode_000"
                    ),
                    "suite": "libero_10",
                    "task_index": 0,
                    "git_commit": collection_head,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        steps.write_text('{"step": 0}\n', encoding="utf-8")

        artifact_rows = collection_artifact_rows(collection)
        manifest_path = (
            collection / "c2g_clean_collection_input_manifest.jsonl"
        )
        manifest_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in artifact_rows
            ),
            encoding="utf-8",
        )
        collection_report_path = (
            collection / "c2g_clean_collection_report.json"
        )
        collection_report_path.write_text(
            json.dumps(
                {
                    "status": "PASS_CLEAN_COLLECTION",
                    "git_commit": collection_head,
                    "artifact_manifest": str(manifest_path.resolve()),
                    "artifact_manifest_sha256": sha256_file(
                        manifest_path
                    ),
                    "artifact_manifest_entry_count": len(artifact_rows),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifact_manifest = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "entry_count": len(artifact_rows),
            "collection_report_sha256": sha256_file(
                collection_report_path
            ),
        }
        collection_binding_path = root / "collection_binding.json"
        collection_binding_path.write_text(
            json.dumps(
                {
                    "status": (
                        "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING"
                    ),
                    "collection_root": str(collection.resolve()),
                    "artifact_manifest": artifact_manifest,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        canonical_common = {
            "input_root": str(collection.resolve()),
            "label_row_count": 21,
            "known_row_count": 20,
            "unknown_row_count": 1,
            "critical_positive_row_count": 12,
            "known_negative_row_count": 8,
            "release_safe_row_count": 0,
            "distractor_row_count": 0,
            "read_error_count": 0,
            "uses_attack_outcome": False,
            "datasets_materialized": 0,
            "detectors_trained": 0,
        }
        canonical = root / "canonical_pass.json"
        canonical.write_text(
            json.dumps(
                {
                    **canonical_common,
                    "status": (
                        "PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT"
                    ),
                    "attack_start_row_count": 1,
                    "reason_code_counts": {
                        "TARGET_CRITICAL_WINDOW_START": 1
                    },
                    "violation_count": 0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        previous_canonical = root / "canonical_hold.json"
        previous_canonical.write_text(
            json.dumps(
                {
                    **canonical_common,
                    "status": (
                        "HOLD_C2G_CLEAN_WINDOW_V2_DRY_AUDIT"
                    ),
                    "attack_start_row_count": 2,
                    "reason_code_counts": {
                        "TARGET_CRITICAL_WINDOW_START": 2
                    },
                    "violation_count": 1,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        goal_totals = {
            "active_target_known_rows": 20,
            "contacted_unresolved_rows": 0,
            "active_progress_unresolved_rows": 0,
            "known_teacher_rows": 20,
            "critical_positive_rows": 12,
            "burst_feasible_rows": 8,
        }
        goal = root / "goal_pass.json"
        goal.write_text(
            json.dumps(
                {
                    "status": (
                        "PASS_C2G_GOAL_EVENT_TRACKING_AUDIT"
                    ),
                    "input_root": str(collection.resolve()),
                    "violation_count": 0,
                    "violations": [],
                    "totals": {
                        **goal_totals,
                        "attack_start_rows": 1,
                    },
                    "uses_attack_outcomes": False,
                    "openvla_model_loads": 0,
                    "libero_environments_created": 0,
                    "attacks_launched": 0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        previous_goal = root / "goal_hold.json"
        previous_goal.write_text(
            json.dumps(
                {
                    "status": (
                        "HOLD_C2G_GOAL_EVENT_TRACKING_AUDIT"
                    ),
                    "input_root": str(collection.resolve()),
                    "violation_count": 1,
                    "violations": [
                        {
                            "episode_key": (
                                "libero_10/task_0/state_9/"
                                "train/episode_000"
                            ),
                            "reason": (
                                "MULTIPLE_ATTACK_START_ROWS"
                            ),
                            "count": 2,
                        }
                    ],
                    "totals": {
                        **goal_totals,
                        "attack_start_rows": 2,
                    },
                    "uses_attack_outcomes": False,
                    "openvla_model_loads": 0,
                    "libero_environments_created": 0,
                    "attacks_launched": 0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        previous_binding = root / "old_hold_binding.json"
        previous_binding.write_text(
            json.dumps(
                {
                    "status": (
                        "HOLD_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING"
                    ),
                    "collection_root": str(collection.resolve()),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        label_builder = root / "label_builder.py"
        label_builder.write_text(
            "# frozen label builder\n",
            encoding="utf-8",
        )
        return {
            "collection": collection,
            "collection_head": collection_head,
            "audit_head": audit_head,
            "collection_report": collection_report_path,
            "collection_binding": collection_binding_path,
            "canonical": canonical,
            "goal": goal,
            "previous_canonical": previous_canonical,
            "previous_goal": previous_goal,
            "previous_binding": previous_binding,
            "label_builder": label_builder,
            "steps": steps,
        }

    def build(self, fixture, **overrides):
        arguments = {
            "collection_root": fixture["collection"],
            "collection_report_path": fixture[
                "collection_report"
            ],
            "collection_binding_report_path": fixture[
                "collection_binding"
            ],
            "canonical_audit_path": fixture["canonical"],
            "goal_event_audit_path": fixture["goal"],
            "label_builder_path": fixture["label_builder"],
            "collection_head": fixture["collection_head"],
            "audit_head": fixture["audit_head"],
            "previous_canonical_hold_path": fixture[
                "previous_canonical"
            ],
            "previous_goal_event_hold_path": fixture[
                "previous_goal"
            ],
            "previous_hold_binding_path": fixture[
                "previous_binding"
            ],
        }
        arguments.update(overrides)
        return build_binding(**arguments)

    def write_binding(self, root: Path, report):
        binding_path = root / "binding.json"
        binding_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return binding_path

    def test_build_and_verify_one_shot_dual_head_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = self.fixture(root)
            report = self.build(fixture)
            self.assertEqual(report["status"], PASS_STATUS)
            self.assertEqual(
                report["collection_head"],
                fixture["collection_head"],
            )
            self.assertEqual(
                report["audit_head"],
                fixture["audit_head"],
            )
            self.assertTrue(report["source_collection_unchanged"])
            self.assertEqual(
                report["unknown_to_negative_count"],
                0,
            )
            self.assertEqual(
                report["expected_only_change"][
                    "libero_10_attack_start_rows"
                ],
                "2_to_1",
            )
            binding_path = self.write_binding(root, report)
            verified = verify_binding(
                binding_path,
                collection_root=fixture["collection"],
                expected_audit_head=fixture["audit_head"],
            )
            self.assertEqual(verified["status"], PASS_STATUS)

    def test_build_requires_all_previous_hold_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(Path(td))
            with self.assertRaisesRegex(
                ValueError,
                "all three previous HOLD artifacts are required",
            ):
                self.build(
                    fixture,
                    previous_hold_binding_path=None,
                )

    def test_verify_rejects_removed_hold_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = self.fixture(root)
            report = self.build(fixture)
            report.pop("previous_hold_binding_path")
            report.pop("previous_hold_binding_sha256")
            binding_path = self.write_binding(root, report)
            with self.assertRaisesRegex(
                ValueError,
                "previous_hold_binding_path",
            ):
                verify_binding(
                    binding_path,
                    collection_root=fixture["collection"],
                    expected_audit_head=fixture["audit_head"],
                )

    def test_verify_rejects_previous_hold_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = self.fixture(root)
            report = self.build(fixture)
            binding_path = self.write_binding(root, report)
            fixture["previous_binding"].write_text(
                '{"status":"PASS"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "previous_hold_binding_path",
            ):
                verify_binding(
                    binding_path,
                    collection_root=fixture["collection"],
                    expected_audit_head=fixture["audit_head"],
                )

    def test_verify_rejects_collection_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = self.fixture(root)
            report = self.build(fixture)
            binding_path = self.write_binding(root, report)
            fixture["steps"].write_text(
                '{"step": 1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "source manifest changed",
            ):
                verify_binding(
                    binding_path,
                    collection_root=fixture["collection"],
                    expected_audit_head=fixture["audit_head"],
                )

    def test_build_rejects_unexpected_label_drift(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(Path(td))
            canonical = json.loads(
                fixture["canonical"].read_text(encoding="utf-8")
            )
            canonical["critical_positive_row_count"] += 1
            canonical["known_negative_row_count"] -= 1
            fixture["canonical"].write_text(
                json.dumps(canonical),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "unexpected canonical label drift",
            ):
                self.build(fixture)


if __name__ == "__main__":
    unittest.main()
