import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.build_c2g_matched_load_jobs_release import main


class ReleaseJobBuilderTests(unittest.TestCase):
    def test_clean_seed_matches_preregistered_eval_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parents = root / "parents.jsonl"
            timing = root / "timing.jsonl"
            checkpoint = root / "checkpoint.pt"
            config = root / "training_report.json"
            output = root / "jobs.jsonl"
            parent_key = "libero_object/task_0/state_0/eval/episode_000"
            parents.write_text(
                json.dumps(
                    {
                        "parent_key": parent_key,
                        "suite": "libero_object",
                        "task_index": 0,
                        "state_id": 0,
                        "eval_seed": 12345,
                        "clean_parent_sha256": "1" * 64,
                        "initial_state_sha256": "2" * 64,
                        "max_steps": 100,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            timing.write_text(
                json.dumps(
                    {
                        "parent_key": parent_key,
                        "detector_start_step": 20,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("{}", encoding="utf-8")
            rc = main(
                [
                    "--parents", str(parents),
                    "--detector-timing", str(timing),
                    "--checkpoint", str(checkpoint),
                    "--detector-config", str(config),
                    "--output", str(output),
                    "--burst-length", "10",
                ]
            )
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in output.read_text().splitlines() if line]
            clean = next(row for row in rows if row["condition"] == "CLEAN")
            self.assertEqual(clean["objective_seed"], 12345)
            gripper = {
                row["objective_seed"]
                for row in rows
                if row["objective_family"] == "GRIPPER_TARGETED_VIS_PGD"
            }
            shuffled = {
                row["objective_seed"]
                for row in rows
                if row["objective_family"] == "SHUFFLED_GRIPPER_GRADIENT"
            }
            self.assertEqual(len(gripper), 1)
            self.assertEqual(len(shuffled), 1)
            report = json.loads((root / "jobs.jsonl.report.json").read_text())
            self.assertEqual(report["status"], "PASS_C2G_RELEASE_MATCHED_LOAD_JOBS_BUILT")
            self.assertEqual(report["clean_seed_binding"], "objective_seed_equals_eval_seed")


if __name__ == "__main__":
    unittest.main()
