import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector import build_c2g_r8t_teacher_v2_canary as planner
from tools.multisuite_detector import audit_c2g_r8t_teacher_v2_canary as audit
from scripts.stageb.run_c2g_r8t_dynamic_gpu_canary import parse_gpu_list
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    DETECTOR_TRAIN,
    PASS_STATUS as R7_PASS,
    SCHEMA as R7_SCHEMA,
    SUITES,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(root: Path):
    rows = []
    for suite in SUITES:
        for task in range(3):
            for state in range(6):
                rows.append({
                    "suite": suite,
                    "task_index": task,
                    "state_id": state,
                    "parent_key": f"{suite}/task_{task}/state_{state}/parent",
                    "cohort": DETECTOR_TRAIN,
                    "split": "train",
                })
            rows.append({
                "suite": suite,
                "task_index": task,
                "state_id": 40,
                "parent_key": f"{suite}/task_{task}/state_40/val",
                "cohort": "DETECTOR_VAL",
                "split": "val",
            })
            rows.append({
                "suite": suite,
                "task_index": task,
                "state_id": 41,
                "parent_key": f"{suite}/task_{task}/state_41/test",
                "cohort": "DETECTOR_TEST_WITHIN_TASK",
                "split": "test",
            })
            rows.append({
                "suite": suite,
                "task_index": task,
                "state_id": 42,
                "parent_key": f"{suite}/task_{task}/state_42/attack",
                "cohort": "ATTACK_EVAL_PREREGISTERED",
                "split": "attack_eval",
            })
    registry = root / "registry.jsonl"
    write_jsonl(registry, rows)
    plan = root / "plan.json"
    write_json(plan, {
        "schema": R7_SCHEMA,
        "status": R7_PASS,
        "registry": str(registry.resolve()),
        "registry_sha256": planner.sha256_file(registry),
    })
    reusable_rows = [
        next(row for row in rows if row["suite"] == suite and row["task_index"] == 2 and row["state_id"] == 0)
        for suite in SUITES
    ]
    reusable = root / "reusable.jsonl"
    write_jsonl(reusable, reusable_rows)
    r8s = root / "r8s.json"
    write_json(r8s, {
        "final_decision": planner.EXPECTED_R8S_DECISION,
        "episode_count": 2000,
        "strict_replay_ready_count": 0,
        "exact_equivalent_mapping_count": 0,
    })
    return registry, plan, reusable, r8s


def isolated_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    return repo


class R8TTeacherV2CanaryTests(unittest.TestCase):
    def test_plan_is_exactly_24_and_train_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = isolated_repo(root)
            registry, plan, reusable, r8s = fixture(root / "evidence")
            result = planner.build_plan(
                repo=repo,
                expected_git_commit="a" * 40,
                registry_path=registry,
                plan_report_path=plan,
                expected_plan_report_sha256=planner.sha256_file(plan),
                reusable_manifest_path=reusable,
                expected_reusable_manifest_sha256=planner.sha256_file(reusable),
                r8s_report_path=r8s,
                expected_r8s_report_sha256=planner.sha256_file(r8s),
                output_dir=root / "out",
            )
            self.assertEqual(result["episode_count"], 24)
            selected = planner.read_jsonl(Path(result["manifest"]))
            self.assertEqual(len(selected), 24)
            self.assertTrue(all(row["cohort"] == DETECTOR_TRAIN for row in selected))
            self.assertTrue(all(row["split"] == "train" for row in selected))
            self.assertEqual({row["suite"] for row in selected}, set(SUITES))
            self.assertEqual(len(result["shards"]), 4)
            self.assertTrue(all(row["episode_count"] == 6 for row in result["shards"]))
            self.assertEqual(result["invariants"]["validation_parent_count"], 0)
            self.assertEqual(result["invariants"]["attack_eval_parent_count"], 0)

    def test_wrong_r8s_decision_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = isolated_repo(root)
            registry, plan, reusable, r8s = fixture(root / "evidence")
            value = json.loads(r8s.read_text())
            value["final_decision"] = "GO_DETERMINISTIC_REPLAY_CANARY"
            write_json(r8s, value)
            with self.assertRaisesRegex(ValueError, "auxiliary-only"):
                planner.build_plan(
                    repo=repo,
                    expected_git_commit="a" * 40,
                    registry_path=registry,
                    plan_report_path=plan,
                    expected_plan_report_sha256=planner.sha256_file(plan),
                    reusable_manifest_path=reusable,
                    expected_reusable_manifest_sha256=planner.sha256_file(reusable),
                    r8s_report_path=r8s,
                    expected_r8s_report_sha256=planner.sha256_file(r8s),
                    output_dir=root / "out",
                )

    def test_gpu_list_contract(self):
        self.assertEqual(parse_gpu_list("4,5,6,7"), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            parse_gpu_list("4,4")
        with self.assertRaises(ValueError):
            parse_gpu_list("")

    def test_metadata_contract_requires_reproducibility_fields(self):
        metadata = {key: "bound" for key in audit.META_REQUIRED_KEYS}
        metadata["runtime_versions"] = {
            "python": "3.10", "torch": "2", "transformers": "4",
            "libero": "local", "robosuite": "1.4",
        }
        metadata["controller_config"] = {"controller_class": "OSC_POSE"}
        ok, missing = audit.metadata_complete(metadata)
        self.assertTrue(ok, missing)
        del metadata["bddl_sha256"]
        ok, missing = audit.metadata_complete(metadata)
        self.assertFalse(ok)
        self.assertIn("bddl_sha256", missing)

    def test_action_and_trigger_contracts(self):
        self.assertTrue(audit.finite_vector([0.0] * 7, 7))
        self.assertFalse(audit.finite_vector([0.0] * 4, 7))
        self.assertTrue(audit.triggerable([False, True, True]))
        self.assertFalse(audit.triggerable([True, False, False]))


from scripts.stageb.run_c2g_r8t_teacher_v2_canary_shard import _validate_plan_invariants


class TypedInvariantTests(unittest.TestCase):
    def _invariants(self, **overrides):
        base = {
            "train_only": True,
            "episode_cardinality_closed": True,
            "validation_parent_count": 0,
            "clean_test_parent_count": 0,
            "attack_eval_parent_count": 0,
            "suite_count": 4,
        }
        base.update(overrides)
        return base

    def test_valid_counts_pass(self):
        _validate_plan_invariants({"invariants": self._invariants()})

    def test_string_zero_fails(self):
        for key in ("validation_parent_count", "clean_test_parent_count", "attack_eval_parent_count", "suite_count"):
            inv = self._invariants(**{key: "0"})
            with self.assertRaisesRegex(ValueError, "must be an exact int"):
                _validate_plan_invariants({"invariants": inv})

    def test_float_zero_fails(self):
        for key in ("validation_parent_count", "clean_test_parent_count", "attack_eval_parent_count", "suite_count"):
            inv = self._invariants(**{key: 0.0})
            with self.assertRaisesRegex(ValueError, "must be an exact int"):
                _validate_plan_invariants({"invariants": inv})

    def test_bool_false_fails(self):
        for key in ("validation_parent_count", "clean_test_parent_count", "attack_eval_parent_count", "suite_count"):
            inv = self._invariants(**{key: False})
            with self.assertRaises(ValueError):
                _validate_plan_invariants({"invariants": inv})

    def test_wrong_count_fails(self):
        for key in ("validation_parent_count", "clean_test_parent_count", "attack_eval_parent_count"):
            inv = self._invariants(**{key: 1})
            with self.assertRaisesRegex(ValueError, "must equal 0"):
                _validate_plan_invariants({"invariants": inv})

    def test_wrong_suite_count_fails(self):
        inv = self._invariants(suite_count=3)
        with self.assertRaisesRegex(ValueError, "must equal 4"):
            _validate_plan_invariants({"invariants": inv})

    def test_train_only_false_fails(self):
        inv = self._invariants(train_only=False)
        with self.assertRaises(ValueError):
            _validate_plan_invariants({"invariants": inv})

    def test_missing_key_fails(self):
        inv = self._invariants()
        del inv["suite_count"]
        with self.assertRaisesRegex(ValueError, "must be an exact int"):
            _validate_plan_invariants({"invariants": inv})

    def test_null_fails(self):
        inv = self._invariants(validation_parent_count=None)
        with self.assertRaisesRegex(ValueError, "must be an exact int"):
            _validate_plan_invariants({"invariants": inv})


if __name__ == "__main__":
    unittest.main()
