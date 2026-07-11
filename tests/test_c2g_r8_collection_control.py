import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_clean_source_inventory import (
    PASS_STATUS as SOURCE_PASS_STATUS,
    SCHEMA as SOURCE_SCHEMA,
)
from tools.multisuite_detector.audit_c2g_r8_collection_wave import (
    OPERATIONAL_HOLD,
    OPERATIONAL_PASS,
    QUALITY_HOLD,
    QUALITY_PASS,
    audit_collection_wave,
)
from tools.multisuite_detector.build_c2g_r8_collection_waves import (
    ATTACK_EVAL_WAVE,
    DETECTOR_CANARY,
    DETECTOR_FULL,
    build_collection_waves,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    CohortCounts,
    materialize_plan,
)

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def inventory(*, task_count=2, state_count=5):
    return [
        {
            "suite": suite,
            "task_index": task_index,
            "state_ids": list(range(state_count)),
        }
        for suite in SUITES
        for task_index in range(task_count)
    ]


def write_source_audit(
    root: Path,
    *,
    plan: dict,
    reusable_rows: list[dict],
):
    root.mkdir(parents=True, exist_ok=True)
    reusable = root / "r7_reusable.jsonl"
    reusable.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reusable_rows),
        encoding="utf-8",
    )
    report = {
        "schema": SOURCE_SCHEMA,
        "status": SOURCE_PASS_STATUS,
        "plan_report": plan["report"],
        "plan_report_sha256": plan["report_sha256"],
        "registry": plan["registry"],
        "registry_sha256": plan["registry_sha256"],
        "reusable_manifest": str(reusable.resolve()),
        "reusable_manifest_sha256": sha256_file(reusable),
        "registered_reusable_episode_count": len(reusable_rows),
        "training_authorization": "HOLD_PENDING_FULL_CORPUS_MATERIALIZATION_AND_AUDIT",
    }
    report_path = root / "r7_source_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, reusable


def make_bound_plan(
    root: Path,
    *,
    task_count=2,
    counts=CohortCounts(train=2, val=1, test=1, attack_eval=1),
    reusable_selector=None,
):
    plan = materialize_plan(
        inventory(task_count=task_count, state_count=counts.total),
        output_dir=root / "plan",
        counts=counts,
        seed=42,
        max_steps=300,
        expected_git_commit="a" * 40,
        inventory_source="unit_test",
    )
    registry_rows = read_jsonl(Path(plan["registry"]))
    reusable_rows = [row for row in registry_rows if reusable_selector and reusable_selector(row)]
    source_report, reusable = write_source_audit(
        root,
        plan=plan,
        reusable_rows=reusable_rows,
    )
    return plan, registry_rows, source_report, reusable


def build_waves(root: Path, *, task_count=2, reusable_selector=None, canary_tasks=1):
    plan, registry, source_report, reusable = make_bound_plan(
        root,
        task_count=task_count,
        reusable_selector=reusable_selector,
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
        canary_tasks_per_suite=canary_tasks,
        canary_shard_size=64,
        detector_full_shard_size=3,
        attack_eval_shard_size=2,
    )
    return result, registry


def write_episode(root: Path, registry_row: dict, *, positive: bool, success: bool):
    episode = root / registry_row["suite"] / f"task_{registry_row['task_index']}" / f"state_{registry_row['state_id']}"
    rgb = episode / "rgb"
    rgb.mkdir(parents=True)
    metadata = {
        "episode_key": registry_row["parent_key"],
        "parent_key": registry_row["parent_key"],
        "suite": registry_row["suite"],
        "task_index": registry_row["task_index"],
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
        "clean_success_observed": success,
    }
    (episode / "episode_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    rows = []
    for step in range(16):
        frame = rgb / f"frame_{step:06d}.png"
        frame.write_bytes(b"r8-structural-rgb")
        row = {
            "step": step,
            "rgb_path": f"rgb/frame_{step:06d}.png",
            "task_language": metadata["task_language"],
            "features_25d": [float(step)] * 25,
            "clean_policy_intent_9d": [0.1] * 9,
            "clean_close_intent": True,
            "contact_pairs": (
                [
                    ["robot0_left_finger_collision", "milk_collision"],
                    ["robot0_right_finger_collision", "milk_collision"],
                ]
                if positive
                else []
            ),
            "object_relative_lift": 0.03 if positive else 0.0,
            "near_target": False,
        }
        rows.append(row)
    (episode / "step_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class R8WavePlannerTests(unittest.TestCase):
    def test_exact_missing_counts_and_canary_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, registry_rows, _, _ = make_bound_plan(root)
            selected = []
            for suite, cohort in (
                ("libero_goal", DETECTOR_TRAIN),
                ("libero_spatial", DETECTOR_TRAIN),
                ("libero_object", ATTACK_EVAL),
                ("libero_10", ATTACK_EVAL),
            ):
                selected.append(
                    next(
                        row
                        for row in registry_rows
                        if row["suite"] == suite and row["cohort"] == cohort
                    )
                )
            source_report, reusable_path = write_source_audit(
                root / "bound",
                plan=plan,
                reusable_rows=selected,
            )
            result = build_collection_waves(
                registry_path=Path(plan["registry"]),
                plan_report_path=Path(plan["report"]),
                expected_plan_report_sha256=plan["report_sha256"],
                source_audit_report_path=source_report,
                expected_source_audit_report_sha256=sha256_file(source_report),
                reusable_manifest_path=reusable_path,
                expected_reusable_manifest_sha256=sha256_file(reusable_path),
                output_dir=root / "waves",
                expected_git_commit="b" * 40,
                canary_tasks_per_suite=1,
                detector_full_shard_size=3,
                attack_eval_shard_size=2,
            )
            self.assertEqual(result["registered_parent_count"], 40)
            self.assertEqual(result["reusable_parent_count"], 4)
            self.assertEqual(result["detector_missing_parent_count"], 30)
            self.assertEqual(result["detector_canary_parent_count"], 12)
            self.assertEqual(result["detector_post_canary_parent_count"], 18)
            self.assertEqual(result["attack_eval_missing_parent_count"], 6)
            canary = read_jsonl(Path(result["waves"][DETECTOR_CANARY]["manifest"]))
            self.assertEqual(len(canary), 12)
            self.assertEqual({row["suite"] for row in canary}, set(SUITES))
            self.assertEqual({row["cohort"] for row in canary}, {DETECTOR_TRAIN, DETECTOR_VAL, DETECTOR_TEST})
            self.assertTrue(all(row["cohort"] != ATTACK_EVAL for row in canary))
            full = read_jsonl(Path(result["waves"][DETECTOR_FULL]["manifest"]))
            attack = read_jsonl(Path(result["waves"][ATTACK_EVAL_WAVE]["manifest"]))
            self.assertEqual(len(full), 18)
            self.assertEqual(len(attack), 6)
            reusable_ids = {(row["suite"], row["task_index"], row["state_id"]) for row in selected}
            canary_ids = {(row["suite"], row["task_index"], row["state_id"]) for row in canary}
            full_ids = {(row["suite"], row["task_index"], row["state_id"]) for row in full}
            self.assertFalse(full_ids & reusable_ids)
            self.assertFalse(canary_ids & full_ids)
            self.assertEqual(len(canary_ids | full_ids), 30)

    def test_tampered_reusable_manifest_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _, source_report, reusable = make_bound_plan(root)
            reusable.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reusable manifest.*hash mismatch"):
                build_collection_waves(
                    registry_path=Path(plan["registry"]),
                    plan_report_path=Path(plan["report"]),
                    expected_plan_report_sha256=plan["report_sha256"],
                    source_audit_report_path=source_report,
                    expected_source_audit_report_sha256=sha256_file(source_report),
                    reusable_manifest_path=reusable,
                    expected_reusable_manifest_sha256="0" * 64,
                    output_dir=root / "waves",
                    expected_git_commit="b" * 40,
                )

    def test_wave_shards_are_suite_closed_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = build_waves(Path(temporary))
            for wave in (DETECTOR_CANARY, DETECTOR_FULL, ATTACK_EVAL_WAVE):
                info = result["waves"][wave]
                self.assertEqual(sha256_file(Path(info["manifest"])), info["manifest_sha256"])
                for shard in info["shards"]:
                    rows = read_jsonl(Path(shard["manifest"]))
                    self.assertEqual(len(rows), shard["episode_count"])
                    self.assertEqual({row["suite"] for row in rows}, {shard["suite"]})
                    self.assertEqual(sha256_file(Path(shard["manifest"])), shard["manifest_sha256"])


class R8WaveAuditTests(unittest.TestCase):
    def make_canary(self, root: Path):
        counts = CohortCounts(train=1, val=1, test=1, attack_eval=1)
        plan = materialize_plan(
            inventory(task_count=1, state_count=counts.total),
            output_dir=root / "plan",
            counts=counts,
            seed=42,
            max_steps=300,
            expected_git_commit="a" * 40,
            inventory_source="unit_test",
        )
        source_report, reusable = write_source_audit(root, plan=plan, reusable_rows=[])
        waves = build_collection_waves(
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
        rows = read_jsonl(Path(waves["waves"][DETECTOR_CANARY]["manifest"]))
        return waves, rows

    def populate(self, root: Path, rows: list[dict], *, success=True, omit_last=False):
        selected = rows[:-1] if omit_last else rows
        for row in selected:
            positive = row["cohort"] != DETECTOR_VAL
            observed_success = bool(success and row["cohort"] == DETECTOR_TRAIN)
            write_episode(root, row, positive=positive, success=observed_success)

    def audit(self, root: Path, waves: dict, new_root: Path):
        baseline = root / "baseline"
        baseline.mkdir(exist_ok=True)
        return audit_collection_wave(
            wave_plan_report=Path(waves["report"]),
            expected_wave_plan_report_sha256=waves["report_sha256"],
            wave=DETECTOR_CANARY,
            baseline_source_roots=[baseline],
            new_source_roots=[new_root],
            output_report=root / "audit" / "report.json",
            output_reusable_manifest=root / "audit" / "reusable.jsonl",
            audit_head="c" * 40,
            hash_rgb=True,
        )

    def test_complete_canary_passes_operational_and_quality_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            waves, rows = self.make_canary(root)
            new_root = root / "new"
            self.populate(new_root, rows)
            result = self.audit(root, waves, new_root)
            self.assertEqual(result["operational_status"], OPERATIONAL_PASS)
            self.assertEqual(result["canary_quality_status"], QUALITY_PASS)
            self.assertEqual(result["completed_reusable_episode_count"], 12)
            self.assertTrue(all(result["per_suite"][suite]["clean_success_observed_count"] == 1 for suite in SUITES))
            self.assertTrue(result["training_authorization"].startswith("HOLD_"))

    def test_missing_expected_episode_is_successful_hold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            waves, rows = self.make_canary(root)
            new_root = root / "new"
            self.populate(new_root, rows, omit_last=True)
            result = self.audit(root, waves, new_root)
            self.assertEqual(result["operational_status"], OPERATIONAL_HOLD)
            self.assertEqual(result["missing_episode_count"], 1)
            self.assertEqual(result["canary_quality_status"], QUALITY_HOLD)

    def test_out_of_wave_attack_eval_episode_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            waves, rows = self.make_canary(root)
            new_root = root / "new"
            self.populate(new_root, rows)
            attack_row = read_jsonl(Path(waves["waves"][ATTACK_EVAL_WAVE]["manifest"]))[0]
            write_episode(new_root, attack_row, positive=True, success=True)
            with self.assertRaisesRegex(ValueError, "out-of-wave identities"):
                self.audit(root, waves, new_root)

    def test_zero_clean_success_is_quality_hold_not_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            waves, rows = self.make_canary(root)
            new_root = root / "new"
            self.populate(new_root, rows, success=False)
            result = self.audit(root, waves, new_root)
            self.assertEqual(result["operational_status"], OPERATIONAL_PASS)
            self.assertEqual(result["canary_quality_status"], QUALITY_HOLD)
            self.assertTrue(any("no clean success observed" in value for value in result["canary_quality_violations"]))


if __name__ == "__main__":
    unittest.main()
