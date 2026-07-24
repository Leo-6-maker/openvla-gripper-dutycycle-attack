import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_clean_source_inventory import audit_inventory
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    CohortCounts,
    build_fold_plans,
    build_registry,
    materialize_plan,
)


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def inventory(state_count=6):
    return [
        {"suite": suite, "task_index": 0, "state_ids": list(range(state_count))}
        for suite in SUITES
    ]


def write_positive_episode(root: Path, registry_row, *, forbidden=False):
    episode = root / "episodes" / registry_row["suite"] / f"source_{registry_row['state_id']}"
    rgb = episode / "rgb"
    rgb.mkdir(parents=True)
    metadata = {
        "episode_key": f"source/{registry_row['suite']}/task_0/state_{registry_row['state_id']}",
        "parent_key": f"source/{registry_row['suite']}/task_0/state_{registry_row['state_id']}",
        "suite": registry_row["suite"],
        "task_index": 0,
        "state_id": registry_row["state_id"],
        "runtime_valid": True,
        "condition": "CLEAN",
        "task_language": "pick up the milk and place it in the basket",
        "mechanism_type": "pick_place_transfer",
        "object_declarations": ["milk"],
        "receptacle_declarations": ["basket"],
        "structured_goal_metadata": {
            "target_objects": ["milk"],
            "target_receptacles": ["basket"],
        },
        "gripper_command_semantics": "positive_is_close",
    }
    (episode / "episode_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    rows = []
    for step in range(16):
        frame = rgb / f"frame_{step:06d}.png"
        frame.write_bytes(b"not-a-decoded-image-required-for-structural-audit")
        row = {
            "step": step,
            "rgb_path": f"rgb/frame_{step:06d}.png",
            "task_language": metadata["task_language"],
            "features_25d": [float(step)] * 25,
            "clean_policy_intent_9d": [0.1] * 9,
            "clean_close_intent": True,
            "contact_pairs": [
                ["robot0_left_finger_collision", "milk_collision"],
                ["robot0_right_finger_collision", "milk_collision"],
            ],
            "object_relative_lift": 0.03,
            "near_target": False,
        }
        if forbidden and step == 0:
            row["attack_outcome"] = True
        rows.append(row)
    (episode / "step_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return episode


class R7CorpusPlannerTests(unittest.TestCase):
    def test_exact_task_stratified_registry_and_no_overlap(self):
        counts = CohortCounts(train=2, val=1, test=1, attack_eval=1)
        rows, tasks = build_registry(
            inventory(), counts=counts, seed=42, max_steps=300
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len(rows), 20)
        identities = {(r["suite"], r["task_index"], r["state_id"]) for r in rows}
        self.assertEqual(len(identities), len(rows))
        for suite in SUITES:
            local = [row for row in rows if row["suite"] == suite]
            self.assertEqual(
                {cohort: sum(row["cohort"] == cohort for row in local) for cohort in (
                    DETECTOR_TRAIN, DETECTOR_VAL, DETECTOR_TEST, ATTACK_EVAL
                )},
                {
                    DETECTOR_TRAIN: 2,
                    DETECTOR_VAL: 1,
                    DETECTOR_TEST: 1,
                    ATTACK_EVAL: 1,
                },
            )
            self.assertEqual({row["split"] for row in local}, {"train", "val", "test", "attack_eval"})
        repeated, _ = build_registry(
            inventory(), counts=counts, seed=42, max_steps=300
        )
        self.assertEqual(rows, repeated)

    def test_insufficient_states_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires 5"):
            build_registry(
                inventory(state_count=4),
                counts=CohortCounts(train=2, val=1, test=1, attack_eval=1),
                seed=42,
                max_steps=300,
            )

    def test_loto_loso_exclude_attack_eval(self):
        rows, _ = build_registry(
            inventory(),
            counts=CohortCounts(train=2, val=1, test=1, attack_eval=1),
            seed=7,
            max_steps=300,
        )
        loto, loso = build_fold_plans(rows)
        self.assertEqual(len(loto), 4)
        self.assertEqual(len(loso), 4)
        self.assertEqual(loto[0]["train_episode_count"], 6)
        self.assertEqual(loto[0]["val_episode_count"], 3)
        self.assertEqual(loto[0]["test_episode_count"], 4)
        self.assertEqual(loto[0]["excluded_attack_eval_episode_count"], 4)
        self.assertEqual(loso[0]["train_episode_count"], 6)
        self.assertEqual(loso[0]["val_episode_count"], 3)
        self.assertEqual(loso[0]["test_episode_count"], 4)
        self.assertEqual(loso[0]["excluded_attack_eval_episode_count"], 4)


class R7SourceInventoryTests(unittest.TestCase):
    def make_plan(self, root: Path):
        output = root / "plan"
        report = materialize_plan(
            inventory(),
            output_dir=output,
            counts=CohortCounts(train=2, val=1, test=1, attack_eval=1),
            seed=42,
            max_steps=300,
            expected_git_commit="a" * 40,
            inventory_source="unit_test",
        )
        return output, report

    def test_registered_positive_episode_is_reusable_and_triggerable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_root, plan = self.make_plan(root)
            registry_rows = [
                json.loads(line)
                for line in (plan_root / "c2g_parent_registry.jsonl").read_text().splitlines()
            ]
            source = root / "source"
            write_positive_episode(source, registry_rows[0])
            report_path = root / "audit" / "report.json"
            reusable_path = root / "audit" / "reusable.jsonl"
            result = audit_inventory(
                registry_path=plan_root / "c2g_parent_registry.jsonl",
                plan_report_path=Path(plan["report"]),
                expected_plan_report_sha256=plan["report_sha256"],
                source_roots=[source],
                output_report=report_path,
                reusable_manifest=reusable_path,
                hash_rgb=True,
                audit_head="b" * 40,
            )
            self.assertEqual(result["registered_reusable_episode_count"], 1)
            self.assertEqual(result["detector_source_corpus_status"], "HOLD_C2G_R7_SOURCE_CORPUS_INCOMPLETE")
            episode = result["episode_audits"][0]
            self.assertTrue(episode["positive_episode"])
            self.assertTrue(episode["triggerable_positive_episode"])
            self.assertEqual(episode["known_positive_steps"], 16)
            self.assertEqual(len(reusable_path.read_text().splitlines()), 1)

    def test_forbidden_attack_field_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_root, plan = self.make_plan(root)
            registry_row = json.loads(
                (plan_root / "c2g_parent_registry.jsonl").read_text().splitlines()[0]
            )
            source = root / "source"
            write_positive_episode(source, registry_row, forbidden=True)
            result = audit_inventory(
                registry_path=plan_root / "c2g_parent_registry.jsonl",
                plan_report_path=Path(plan["report"]),
                expected_plan_report_sha256=plan["report_sha256"],
                source_roots=[source],
                output_report=root / "audit" / "report.json",
                reusable_manifest=root / "audit" / "reusable.jsonl",
                hash_rgb=False,
            )
            self.assertEqual(result["registered_reusable_episode_count"], 0)
            self.assertIn("forbidden step keys", result["episode_audits"][0]["failure_reason"])

    def test_duplicate_identity_is_fail_closed_for_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_root, plan = self.make_plan(root)
            registry_row = json.loads(
                (plan_root / "c2g_parent_registry.jsonl").read_text().splitlines()[0]
            )
            source_a = root / "source_a"
            source_b = root / "source_b"
            write_positive_episode(source_a, registry_row)
            write_positive_episode(source_b, registry_row)
            result = audit_inventory(
                registry_path=plan_root / "c2g_parent_registry.jsonl",
                plan_report_path=Path(plan["report"]),
                expected_plan_report_sha256=plan["report_sha256"],
                source_roots=[source_a, source_b],
                output_report=root / "audit" / "report.json",
                reusable_manifest=root / "audit" / "reusable.jsonl",
                hash_rgb=False,
            )
            self.assertEqual(result["duplicate_identity_count"], 1)
            self.assertEqual(result["registered_reusable_episode_count"], 0)
            self.assertTrue(all(row["duplicate_identity"] for row in result["episode_audits"]))


if __name__ == "__main__":
    unittest.main()
