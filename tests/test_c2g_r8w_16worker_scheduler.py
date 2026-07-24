import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb import run_c2g_r8w_full_clean_shard as runner
from scripts.stageb import run_c2g_r8w_gpu4567_16worker as scheduler
from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
    CANARY_PURPOSE,
    GPUS,
    SUITES,
    worker_id,
)


def plan_fixture():
    shards = []
    for gpu in GPUS:
        for suite in SUITES:
            shards.append({
                "worker_id": worker_id(gpu, suite),
                "suite": suite,
                "physical_gpu": gpu,
                "shard_id": f"{suite}__shard_{GPUS.index(gpu)}",
                "episode_count": 125,
            })
    return {"shards": shards}


class SchedulerTests(unittest.TestCase):
    def test_exact_layout_and_serialized_wave_order(self):
        order = scheduler.validate_worker_layout(plan_fixture())
        self.assertEqual(len(order), 16)
        self.assertEqual(
            [row["worker_id"] for row in order[:4]],
            [worker_id(gpu, "libero_object") for gpu in GPUS],
        )
        self.assertEqual(
            [row["worker_id"] for row in order[-4:]],
            [worker_id(gpu, "libero_10") for gpu in GPUS],
        )

    def test_wrong_worker_count_and_gpu_assignment_fail(self):
        value = plan_fixture()
        value["shards"].pop()
        with self.assertRaisesRegex(ValueError, "16 workers"):
            scheduler.validate_worker_layout(value)

    def test_canary_layout_is_one_suite_worker_per_gpu(self):
        shards = [
            {
                "worker_id": f"canary_{worker_id(gpu, suite)}",
                "suite": suite,
                "physical_gpu": gpu,
                "episode_count": 2,
            }
            for gpu, suite in zip(GPUS, SUITES)
        ]
        order = scheduler.validate_canary_layout({"plan_kind": CANARY_PURPOSE, "shards": shards})
        self.assertEqual([row["physical_gpu"] for row in order], list(GPUS))
        self.assertEqual([row["suite"] for row in order], list(SUITES))
        value = plan_fixture()
        value["shards"][0]["physical_gpu"] = 7
        with self.assertRaisesRegex(ValueError, "worker layout mismatch"):
            scheduler.validate_worker_layout(value)

    def test_memory_budget_and_concurrency_degradation(self):
        self.assertEqual(scheduler.measured_worker_budget_mib(16000), 18432)
        self.assertEqual(scheduler.measured_worker_budget_mib(20000), 23552)
        self.assertEqual(scheduler.safe_resident_workers(81920, 18432), 4)
        self.assertEqual(scheduler.safe_resident_workers(70000, 20000), 3)
        self.assertEqual(scheduler.safe_resident_workers(50000, 22000), 1)

    def test_busy_gpu_and_stability_admission(self):
        good = scheduler.GpuSnapshot(4, 81920, 20000, 61920, 10, 40)
        busy = scheduler.GpuSnapshot(4, 81920, 20000, 61920, 41, 40)
        self.assertEqual(scheduler.memory_admission(good, 20000), (True, "PASS"))
        self.assertEqual(scheduler.memory_admission(busy, 20000)[1], "GPU_UTILIZATION_ABOVE_40_PERCENT")
        self.assertEqual(scheduler.stable_admission([good, good, good], 20000), (True, "PASS"))
        unstable = scheduler.GpuSnapshot(4, 81920, 40000, 40000, 10, 40)
        self.assertEqual(scheduler.stable_admission([good, good, unstable], 20000)[1], "FREE_MEMORY_NOT_STABLE")

    def test_worker_failure_preserves_evidence_and_stops_new_launches(self):
        policy = scheduler.worker_failure_policy(1, False)
        self.assertTrue(policy["worker_failed"])
        self.assertTrue(policy["stop_new_launches"])
        self.assertTrue(policy["preserve_worker_output"])
        self.assertTrue(policy["allow_other_running_workers_to_finish"])

    def test_worker_environment_binds_physical_gpu_and_cpu_threads(self):
        env = runner.worker_environment(6)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(env["C2G_PHYSICAL_GPU"], "6")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_resume_receipt_hash_mismatch_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            episode = Path(td)
            (episode / "rgb").mkdir()
            frame = episode / "rgb" / "frame_000000.png"
            frame.write_bytes(b"rgb")
            metadata = episode / "episode_metadata.json"
            metadata.write_text(json.dumps({"runtime_valid": True, "clean_success_observed": False}), encoding="utf-8")
            steps = episode / "step_records.jsonl"
            steps.write_text('{"step": 0}\n', encoding="utf-8")
            rgb_manifest = episode / "rgb_manifest.jsonl"
            rgb_manifest.write_text(
                json.dumps({"path": frame.name, "bytes": 3, "sha256": runner.sha256_file(frame)}) + "\n",
                encoding="utf-8",
            )
            receipt = {
                "schema": runner.EPISODE_RECEIPT_SCHEMA,
                "parent_key": "parent",
                "worker_id": "g4_object",
                "shard_id": "libero_object__shard_0",
                "git_head": "a" * 40,
                "manifest_sha256": "b" * 64,
                "metadata_sha256": runner.sha256_file(metadata),
                "step_records_sha256": runner.sha256_file(steps),
                "rgb_manifest_sha256": runner.sha256_file(rgb_manifest),
                "runtime_valid": True,
            }
            (episode / "episode_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            valid, _ = runner.validate_episode_receipt(
                episode,
                expected_parent_key="parent",
                expected_worker_id="g4_object",
                expected_shard_id="libero_object__shard_0",
                expected_git_head="a" * 40,
                expected_manifest_sha="b" * 64,
            )
            self.assertTrue(valid)
            valid, reason = runner.validate_episode_receipt(
                episode,
                expected_parent_key="parent",
                expected_worker_id="g4_object",
                expected_shard_id="libero_object__shard_0",
                expected_git_head="a" * 40,
                expected_manifest_sha="c" * 64,
            )
            self.assertFalse(valid)
            self.assertIn("manifest_sha256 mismatch", reason)


if __name__ == "__main__":
    unittest.main()
