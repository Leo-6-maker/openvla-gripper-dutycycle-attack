from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.table1_audit.metrics.aggregate_table1_metrics import aggregate
from tools.table1_audit.metrics.compute_table1_cqfr import compute as compute_cqfr
from tools.table1_audit.metrics.compute_table1_rnad import compute as compute_rnad


class Table1MetricFreezeTests(unittest.TestCase):
    def test_rnad_keeps_no_emission_in_itt_and_uses_env_vs_clean_env(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "telemetry.csv"
            with p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "job_key", "condition_id", "detector_emitted", "attack_active",
                    "clean_policy_action_7d", "policy_action_7d",
                    "clean_env_action_7d", "env_action_7d",
                ])
                w.writeheader()
                base = json.dumps([0, 0, 0, 0, 0, 0, 0])
                arm = json.dumps([1, 1, 1, 1, 1, 1, 0])
                grip = json.dumps([0, 0, 0, 0, 0, 0, 1])
                w.writerow({"job_key": "a", "condition_id": "TRUE_T10", "detector_emitted": "true", "attack_active": "true", "clean_policy_action_7d": base, "policy_action_7d": arm, "clean_env_action_7d": base, "env_action_7d": grip})
                w.writerow({"job_key": "b", "condition_id": "TRUE_T10", "detector_emitted": "false", "attack_active": "false", "clean_policy_action_7d": base, "policy_action_7d": base, "clean_env_action_7d": base, "env_action_7d": base})
            result = compute_rnad(p)["conditions"]["TRUE_T10"]
            self.assertEqual(result["itt_count"], 2)
            self.assertEqual(result["attack_active_count"], 1)
            self.assertEqual(result["policy_arm_attack_active"], 1.0)
            self.assertEqual(result["execution_gripper_attack_active"], 1.0)
            self.assertEqual(result["execution_arm_attack_active"], 0.0)

    def test_cqfr_preserves_no_emission_in_itt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            p.write_text(
                "job_key,condition_id,detector_emitted,contact_quality_failure\n"
                "a,TRUE_T10,true,true\n"
                "b,TRUE_T10,false,false\n",
                encoding="utf-8",
            )
            result = compute_cqfr(p)["conditions"]["TRUE_T10"]
            self.assertEqual(result["itt_count"], 2)
            self.assertEqual(result["emitted_count"], 1)
            self.assertEqual(result["no_emission_count"], 1)
            self.assertEqual(result["cqfr_itt"], 0.5)
            self.assertEqual(result["cqfr_conditional_emitted"], 1.0)

    def test_aggregation_keeps_condition_group(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rnad = root / "rnad.json"
            cqfr = root / "cqfr.json"
            rnad.write_text(json.dumps({"conditions": {"TRUE_T10": {"itt_count": 2}}}), encoding="utf-8")
            cqfr.write_text(json.dumps({"conditions": {"TRUE_T10": {"itt_count": 2, "cqfr_itt": 0.5}}}), encoding="utf-8")
            result = aggregate(rnad, cqfr)
            self.assertEqual(result["conditions"]["TRUE_T10"]["row_count"], 2)
            self.assertEqual(result["conditions"]["TRUE_T10"]["cqfr"]["cqfr_itt"], 0.5)


if __name__ == "__main__":
    unittest.main()
