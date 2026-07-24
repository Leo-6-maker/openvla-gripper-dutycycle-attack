"""R8U adaptive GPU preview tests — CPU-only, no nvidia-smi needed."""
import json, unittest
from pathlib import Path

from scripts.stageb.run_c2g_r8u_adaptive_gpu_preview import (
    AUTHORIZATION_DENIED,
    GpuSnapshot,
    Microshard,
    WorkerPlan,
    build_preview,
    calculate_slots,
)


class SlotCalculationTests(unittest.TestCase):
    def test_80gb_free_2_slots(self):
        snap = GpuSnapshot(0, 81920, 80000, 10)
        available, target = calculate_slots(snap)
        self.assertEqual(available, 3)  # (80000-12000)//22000 = 3
        self.assertEqual(target, 2)  # capped at max_workers_per_gpu

    def test_55gb_free_1_slot(self):
        snap = GpuSnapshot(0, 81920, 55000, 10)
        available, target = calculate_slots(snap)
        self.assertEqual(available, 1)  # (55000-12000)//22000 = 1
        self.assertEqual(target, 1)

    def test_under_34gb_free_0_slots(self):
        snap = GpuSnapshot(0, 81920, 33000, 10)
        available, target = calculate_slots(snap)
        self.assertEqual(available, 0)
        self.assertEqual(target, 0)

    def test_busy_utilization_still_reports_slots(self):
        snap = GpuSnapshot(0, 81920, 70000, 90)
        available, target = calculate_slots(snap)
        self.assertEqual(available, 2)  # slots based on memory only in preview


class WorkerPlanTests(unittest.TestCase):
    def test_worker_has_unique_output_root(self):
        w1 = WorkerPlan("w1", 4, "s1", "/out/w1", "/out/l1", "/out/e1")
        w2 = WorkerPlan("w2", 5, "s2", "/out/w2", "/out/l2", "/out/e2")
        self.assertNotEqual(w1.output_root, w2.output_root)
        self.assertNotEqual(w1.worker_id, w2.worker_id)

    def test_loading_order_increments(self):
        workers = [
            WorkerPlan("w1", 4, "s1", "/o1", "/l1", "/e1", load_order=0),
            WorkerPlan("w2", 5, "s2", "/o2", "/l2", "/e2", load_order=1),
            WorkerPlan("w3", 6, "s3", "/o3", "/l3", "/e3", load_order=2),
        ]
        self.assertEqual(workers[0].load_order, 0)
        self.assertEqual(workers[1].load_order, 1)
        self.assertEqual(workers[2].load_order, 2)


class PreviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.gpu_indices = [4, 5, 6, 7]
        self.snapshots = {
            4: GpuSnapshot(4, 81920, 70000, 10),
            5: GpuSnapshot(5, 81920, 70000, 5),
            6: GpuSnapshot(6, 81920, 70000, 0),
            7: GpuSnapshot(7, 81920, 70000, 15),
        }
        self.shards = [
            Microshard("libero_object", "libero_object", 6),
            Microshard("libero_spatial", "libero_spatial", 6),
            Microshard("libero_goal", "libero_goal", 6),
            Microshard("libero_10", "libero_10", 6),
        ]

    def test_preview_has_correct_slot_counts(self):
        result = build_preview(self.gpu_indices, self.snapshots, self.shards, "/tmp/test")
        for idx in self.gpu_indices:
            self.assertIn(idx, result.slot_counts)
            self.assertEqual(result.slot_counts[idx]["target_workers"], 2)

    def test_preview_assigns_all_shards(self):
        result = build_preview(self.gpu_indices, self.snapshots, self.shards, "/tmp/test")
        self.assertEqual(len(result.workers), 4)
        self.assertEqual(len(result.loading_order), 4)

    def test_preview_workers_have_unique_ids(self):
        result = build_preview(self.gpu_indices, self.snapshots, self.shards, "/tmp/test")
        ids = [w["worker_id"] for w in result.workers]
        self.assertEqual(len(ids), len(set(ids)))

    def test_preview_workers_have_unique_roots(self):
        result = build_preview(self.gpu_indices, self.snapshots, self.shards, "/tmp/test")
        roots = [w["output_root"] for w in result.workers]
        self.assertEqual(len(roots), len(set(roots)))

    def test_global_loading_slots_is_one(self):
        result = build_preview(self.gpu_indices, self.snapshots, self.shards, "/tmp/test")
        # Loading is serialized in build_preview by sequential delays
        delays = result.planned_start_delays
        for i in range(1, len(delays)):
            self.assertGreater(delays[i], delays[i - 1])

    def test_run_mode_denied(self):
        from scripts.stageb.run_c2g_r8u_adaptive_gpu_preview import main
        import sys
        old = sys.argv[:]
        try:
            sys.argv = ["test", "run"]
            with self.assertRaises(PermissionError) as ctx:
                main()
            self.assertIn(AUTHORIZATION_DENIED, str(ctx.exception))
        finally:
            sys.argv = old

    def test_empty_gpu_skips(self):
        snapshots = {4: GpuSnapshot(4, 81920, 33000, 5)}  # < 34GB free
        result = build_preview([4, 5], snapshots, self.shards[:1], "/tmp/test")
        self.assertEqual(result.slot_counts[4]["available_slots"], 0)
        self.assertEqual(result.slot_counts[4]["target_workers"], 0)


if __name__ == "__main__":
    unittest.main()
