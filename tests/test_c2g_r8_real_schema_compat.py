import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_clean_source_inventory import (
    PASS_STATUS as SOURCE_PASS_STATUS,
    SCHEMA as SOURCE_SCHEMA,
)
from tools.multisuite_detector.build_c2g_r8_collection_waves import (
    build_collection_waves,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    CohortCounts,
    materialize_plan,
)

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


class R8RealR7SchemaCompatibilityTests(unittest.TestCase):
    def test_registry_parent_key_from_r7_audit_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = CohortCounts(train=1, val=1, test=1, attack_eval=1)
            inventory = [
                {
                    "suite": suite,
                    "task_index": 0,
                    "state_ids": list(range(counts.total)),
                }
                for suite in SUITES
            ]
            plan = materialize_plan(
                inventory,
                output_dir=root / "plan",
                counts=counts,
                seed=42,
                max_steps=300,
                expected_git_commit="a" * 40,
                inventory_source="unit_test",
            )
            registry = read_jsonl(Path(plan["registry"]))
            frozen = registry[0]
            reusable_row = {
                "episode_key": "legacy-source-key",
                "suite": frozen["suite"],
                "task_index": frozen["task_index"],
                "state_id": frozen["state_id"],
                "registry_parent_key": frozen["parent_key"],
                "cohort": frozen["cohort"],
                "split": frozen["split"],
                "registered": True,
                "structurally_eligible": True,
                "reusable": True,
            }
            reusable = root / "r7_reusable.jsonl"
            reusable.write_text(json.dumps(reusable_row, sort_keys=True) + "\n", encoding="utf-8")
            source_report = root / "r7_source_audit.json"
            source_report.write_text(
                json.dumps(
                    {
                        "schema": SOURCE_SCHEMA,
                        "status": SOURCE_PASS_STATUS,
                        "plan_report": plan["report"],
                        "plan_report_sha256": plan["report_sha256"],
                        "registry": plan["registry"],
                        "registry_sha256": plan["registry_sha256"],
                        "reusable_manifest": str(reusable.resolve()),
                        "reusable_manifest_sha256": sha256_file(reusable),
                        "registered_reusable_episode_count": 1,
                        "training_authorization": "HOLD_PENDING_FULL_CORPUS_MATERIALIZATION_AND_AUDIT",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result = build_collection_waves(
                registry_path=Path(plan["registry"]),
                plan_report_path=Path(plan["report"]),
                expected_plan_report_sha256=plan["report_sha256"],
                source_audit_report_path=source_report,
                expected_source_audit_report_sha256=sha256_file(source_report),
                reusable_manifest_path=reusable,
                expected_reusable_manifest_sha256=sha256_file(reusable),
                output_dir=root / "waves",
                expected_git_commit="b" * 40,
                canary_tasks_per_suite=1,
            )
            self.assertEqual(result["registered_parent_count"], 16)
            self.assertEqual(result["reusable_parent_count"], 1)
            self.assertEqual(result["missing_parent_count"], 15)


if __name__ == "__main__":
    unittest.main()
