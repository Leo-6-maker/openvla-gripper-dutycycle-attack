import csv
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.multisuite_detector import build_c2g_r8w_full_clean_2000_plan as plan
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    COHORT_TO_SPLIT,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    SUITES,
)


def registry_rows():
    rows = []
    layout = (
        (DETECTOR_TRAIN, 30),
        (DETECTOR_VAL, 5),
        (DETECTOR_TEST, 5),
        (ATTACK_EVAL, 10),
    )
    for suite in SUITES:
        for task in range(10):
            state = 0
            for cohort, count in layout:
                flags = plan.expected_flags(cohort)
                for local_index in range(count):
                    rows.append({
                        "suite": suite,
                        "task_index": task,
                        "state_id": state,
                        "parent_key": f"{suite}/task_{task}/state_{state}/{cohort.lower()}/episode_{local_index:03d}",
                        "cohort": cohort,
                        "split": COHORT_TO_SPLIT[cohort],
                        "max_steps": 300,
                        "selection_seed": 42,
                        **flags,
                    })
                    state += 1
    return rows


class FullCleanPlanTests(unittest.TestCase):
    def test_exact_closure_and_fixed_worker_mapping(self):
        rows = plan.validate_registry(registry_rows())
        assigned, shards = plan.build_plan_data(rows, 20260712)
        self.assertEqual(len(assigned), 2000)
        self.assertEqual(len({plan.identity(row) for row in assigned}), 2000)
        self.assertEqual(len(shards), 16)
        self.assertEqual(Counter(row["suite"] for row in assigned), Counter({suite: 500 for suite in SUITES}))
        self.assertEqual(Counter(row["assigned_physical_gpu"] for row in assigned), Counter({gpu: 500 for gpu in plan.GPUS}))
        self.assertEqual(Counter(row["assigned_worker_id"] for row in assigned), Counter({row["worker_id"]: 125 for row in shards}))
        self.assertEqual(plan.assignment_balance(assigned), [])
        for shard in shards:
            self.assertEqual(shard["episode_count"], 125)
            self.assertEqual(len(shard["max_steps"]), 1)

    def test_wrong_cohort_relabel_fails(self):
        rows = registry_rows()
        rows[0]["cohort"] = DETECTOR_VAL
        with self.assertRaisesRegex(ValueError, "cohort/split mismatch"):
            plan.validate_registry(rows)

    def test_duplicate_identity_fails(self):
        rows = registry_rows()
        rows[-1]["suite"] = rows[0]["suite"]
        rows[-1]["task_index"] = rows[0]["task_index"]
        rows[-1]["state_id"] = rows[0]["state_id"]
        with self.assertRaisesRegex(ValueError, "duplicate R7 identity"):
            plan.validate_registry(rows)

    def test_missing_identity_fails(self):
        rows = registry_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "2000 rows"):
            plan.validate_registry(rows)

    def test_outside_suite_identity_fails(self):
        rows = registry_rows()
        rows[0]["suite"] = "libero_outside"
        with self.assertRaisesRegex(ValueError, "invalid identity"):
            plan.validate_registry(rows)

    def test_eligibility_leakage_fails(self):
        rows = registry_rows()
        row = next(row for row in rows if row["cohort"] == DETECTOR_VAL)
        row["eligible_for_detector_fit"] = True
        with self.assertRaisesRegex(ValueError, "eligibility mismatch"):
            plan.validate_registry(rows)

    def test_shadow_canary_selects_success_and_failure_per_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry.jsonl"
            rows = registry_rows()
            registry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            repo = Path(__file__).resolve().parents[1]
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            r7_report = root / "r7.json"
            r7_report.write_text(json.dumps({
                "schema": plan.R7_SCHEMA,
                "status": plan.R7_PASS_STATUS,
                "registry": str(registry.resolve()),
                "registry_sha256": plan.sha256_file(registry),
            }), encoding="utf-8")
            ledger = root / "r8u_success_replay_episode_ledger.csv"
            fields = ["suite", "task_index", "state_id", "parent_key", "classification", "canonical_success"]
            with ledger.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                success_counts = {"libero_object": 5, "libero_spatial": 4, "libero_goal": 4, "libero_10": 2}
                for suite in SUITES:
                    local = [row for row in rows if row["suite"] == suite and row["cohort"] == DETECTOR_TRAIN][:6]
                    for index, row in enumerate(local):
                        writer.writerow({
                            **{key: row[key] for key in ("suite", "task_index", "state_id", "parent_key")},
                            "classification": "REPLAY_EXACT",
                            "canonical_success": str(index < success_counts[suite]),
                        })
            step_ledger = root / "r8u_success_replay_step_ledger.jsonl"
            step_ledger.write_text('{"placeholder": true}\n', encoding="utf-8")
            report = root / "r8u_postcanary_report.json"
            report.write_text(json.dumps({
                "status": "PASS_C2G_R8U_SUCCESS_REPLAY_INTEGRITY",
                "episode_count": 24,
                "replay_exact_count": 24,
                "replay_numerically_equivalent_count": 0,
                "replay_diverged_count": 0,
                "replay_failed_count": 0,
                "canonical_clean_success_count": 15,
                "per_suite_clean_success": {
                    suite: {"success": success, "total": 6}
                    for suite, success in success_counts.items()
                },
            }), encoding="utf-8")
            sums = root / "SHA256SUMS"
            sums.write_text("".join(
                f"{plan.sha256_file(path)}  {path.name}\n"
                for path in (report, ledger, step_ledger)
            ), encoding="utf-8")
            result = plan.build_shadow_canary_plan(
                mode="canary-preview",
                repo=repo,
                expected_git_commit=head,
                registry_path=registry,
                expected_registry_sha256=plan.sha256_file(registry),
                plan_report_path=r7_report,
                expected_plan_report_sha256=plan.sha256_file(r7_report),
                r8u_report_path=report,
                expected_r8u_report_sha256=plan.sha256_file(report),
                r8u_episode_ledger_path=ledger,
                expected_r8u_episode_ledger_sha256=plan.sha256_file(ledger),
                r8u_step_ledger_path=step_ledger,
                expected_r8u_step_ledger_sha256=plan.sha256_file(step_ledger),
                r8u_sha256s_path=sums,
                expected_r8u_sha256s_sha256=plan.sha256_file(sums),
                output_root=root / "out",
                authorization="",
            )
            self.assertEqual(result["episode_count"], 8)
            self.assertEqual(result["worker_count"], 4)
            for suite in SUITES:
                selected = [row for row in result["shards"] if row["suite"] == suite]
                self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
