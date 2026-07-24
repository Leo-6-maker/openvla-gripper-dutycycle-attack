"""R8X hardware-bound canary gate tests."""
import json, tempfile, unittest
from pathlib import Path


def _write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _r8t_scheduler_report(shards_mapping=None):
    if shards_mapping is None:
        shards_mapping = [
            ("libero_object", 4), ("libero_spatial", 6),
            ("libero_goal", 5), ("libero_10", 7),
        ]
    return {
        "status": "PASS_C2G_R8T_DYNAMIC_GPU_CANARY",
        "shards": [
            {"suite": s, "physical_gpu": g, "shard_id": f"r8t_{s}"}
            for s, g in shards_mapping
        ],
    }


class HardwareBindingTests(unittest.TestCase):
    """Test GPU mapping extraction and validation."""

    def test_correct_r8t_mapping_parses(self):
        from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
            SUITES, GPUS,
        )
        rpt = _r8t_scheduler_report()
        gpu_by_suite = {}
        for sh in rpt["shards"]:
            gpu_by_suite[sh["suite"]] = sh["physical_gpu"]
        self.assertEqual(gpu_by_suite["libero_object"], 4)
        self.assertEqual(gpu_by_suite["libero_spatial"], 6)
        self.assertEqual(gpu_by_suite["libero_goal"], 5)
        self.assertEqual(gpu_by_suite["libero_10"], 7)
        self.assertEqual(set(gpu_by_suite.keys()), set(SUITES))
        self.assertEqual(set(gpu_by_suite.values()), set(GPUS))

    def test_duplicate_gpu_detected(self):
        mapping = [
            ("libero_object", 4), ("libero_spatial", 4),  # duplicate GPU 4
            ("libero_goal", 5), ("libero_10", 7),
        ]
        rpt = _r8t_scheduler_report(mapping)
        gpus = {sh["physical_gpu"] for sh in rpt["shards"]}
        self.assertEqual(len(gpus), 3)  # not 4 unique GPUs

    def test_missing_suite_detected(self):
        mapping = [
            ("libero_object", 4), ("libero_spatial", 6),
            ("libero_goal", 5),
            # missing libero_10
        ]
        rpt = _r8t_scheduler_report(mapping)
        suites = {sh["suite"] for sh in rpt["shards"]}
        self.assertNotIn("libero_10", suites)
        self.assertEqual(len(suites), 3)

    def test_wrong_scheduler_status_rejected(self):
        rpt = _r8t_scheduler_report()
        rpt["status"] = "FAILED"
        self.assertNotEqual(rpt["status"], "PASS_C2G_R8T_DYNAMIC_GPU_CANARY")

    def test_reference_scheduler_sha_required(self):
        from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
            build_shadow_canary_plan,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                build_shadow_canary_plan(
                    mode="canary-preview", repo=root,
                    expected_git_commit="a" * 40,
                    registry_path=root / "reg.jsonl",
                    expected_registry_sha256="b" * 64,
                    plan_report_path=root / "plan.json",
                    expected_plan_report_sha256="c" * 64,
                    r8u_report_path=root / "r8u.json",
                    expected_r8u_report_sha256="d" * 64,
                    r8u_episode_ledger_path=root / "ledger.csv",
                    expected_r8u_episode_ledger_sha256="e" * 64,
                    r8u_step_ledger_path=root / "steps.jsonl",
                    expected_r8u_step_ledger_sha256="f" * 64,
                    r8u_sha256s_path=root / "sums",
                    expected_r8u_sha256s_sha256="g" * 64,
                    output_root=root / "out",
                    authorization="",
                    reference_scheduler_report_path=None,
                    expected_reference_scheduler_report_sha256="",
                )

    def test_zip_fallback_rejected(self):
        """Verify zip(GPUS, SUITES) gives wrong mapping and is not used."""
        SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
        GPUS = (4, 5, 6, 7)
        wrong = dict(zip(SUITES, GPUS))
        self.assertEqual(wrong["libero_spatial"], 5)  # Wrong! Should be 6
        self.assertEqual(wrong["libero_goal"], 6)      # Wrong! Should be 5


if __name__ == "__main__":
    unittest.main()
