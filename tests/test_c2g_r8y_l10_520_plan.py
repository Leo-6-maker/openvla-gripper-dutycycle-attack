"""Test R8Y L10-520 plan builder identity preservation and shard closure."""
import json
import subprocess
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.multisuite_detector.build_c2g_r8y_l10_520_plan import (
    CANONICAL_MAX_STEPS,
    EPISODES_PER_GPU,
    EPISODES_PER_LOGICAL_SHARD,
    GPUS,
    LOGICAL_SHARDS_PER_GPU,
    TARGET_SUITE,
    build_r8y_plan,
    extract_l10_rows,
    identity,
    validate_l10_identity_closure,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    SUITES,
)


def _make_r8w_l10_rows(n: int = 500) -> list[dict[str, Any]]:
    """Make synthetic L10 rows with verified per-GPU per-cohort counts.

    Each GPU gets exactly: TRAIN=75, VAL=13 or 12, TEST=12 or 13, ATTACK=25.
    GPUs 4,6: VAL=13, TEST=12. GPUs 5,7: VAL=12, TEST=13.
    """
    from tools.multisuite_detector.build_c2g_r8y_l10_520_plan import (
        GPU46_COHORT_QUOTA,
        GPU57_COHORT_QUOTA,
        LOGICAL_SHARDS_PER_GPU,
    )

    def _cohort_totals(quota_dict):
        totals: dict[str, int] = {}
        for s in range(LOGICAL_SHARDS_PER_GPU):
            for cohort, count in quota_dict[s].items():
                totals[cohort] = totals.get(cohort, 0) + count
        return totals

    gpu_quotas = {
        4: _cohort_totals(GPU46_COHORT_QUOTA),
        5: _cohort_totals(GPU57_COHORT_QUOTA),
        6: _cohort_totals(GPU46_COHORT_QUOTA),
        7: _cohort_totals(GPU57_COHORT_QUOTA),
    }

    cohort_order = [DETECTOR_TRAIN, DETECTOR_VAL, DETECTOR_TEST, ATTACK_EVAL]
    rows = []
    task_cursor: dict[int, int] = defaultdict(int)  # task_index -> next state_id
    gpu_cohort_count: dict[tuple[int, str], int] = defaultdict(int)
    gpu_total: dict[int, int] = defaultdict(int)

    # Round-robin tasks (0-9) with limited states per task (max 50 each)
    # Distribute across GPUs, filling per-GPU cohort quotas
    task_idx = 0
    while len(rows) < 500:
        task_index = task_idx % 10
        state_id = task_cursor[task_index]
        if state_id >= 50:
            task_idx += 1
            continue

        # Determine which GPU still needs this cohort
        placed = False
        for cohort in cohort_order:
            for gpu in GPUS:
                quota = gpu_quotas[gpu]
                if gpu_cohort_count[(gpu, cohort)] < quota[cohort]:
                    # Check this GPU isn't full
                    if gpu_total[gpu] >= 125:
                        continue
                    row = {
                        "parent_key": (
                            f"libero_10/task_{task_index}/state_{state_id}/"
                            f"{cohort.lower()}/episode_{gpu_cohort_count[(gpu, cohort)]:03d}"
                        ),
                        "suite": "libero_10",
                        "task_index": task_index,
                        "state_id": state_id,
                        "cohort": cohort,
                        "split": {
                            DETECTOR_TRAIN: "train",
                            DETECTOR_VAL: "val",
                            DETECTOR_TEST: "test",
                            ATTACK_EVAL: "attack_eval",
                        }[cohort],
                        "max_steps": 300,
                        "selection_seed": 42,
                        "eligible_for_detector_fit": cohort == DETECTOR_TRAIN,
                        "eligible_for_checkpoint_selection": cohort == DETECTOR_VAL,
                        "eligible_for_threshold_calibration": cohort == DETECTOR_VAL,
                        "eligible_for_clean_test": cohort == DETECTOR_TEST,
                        "eligible_for_attack_evaluation": cohort == ATTACK_EVAL,
                        "assigned_physical_gpu": gpu,
                        "assigned_worker_id": f"g{gpu}_l10",
                        "assigned_shard_id": f"libero_10__shard_{gpu % 4}",
                        "collection_purpose": "FULL_CLEAN_2000",
                    }
                    rows.append(row)
                    gpu_cohort_count[(gpu, cohort)] += 1
                    gpu_total[gpu] += 1
                    task_cursor[task_index] += 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            task_idx += 1  # all quotas filled for this task's state

    assert len(rows) == 500, f"expected 500, got {len(rows)}"
    for gpu in GPUS:
        assert gpu_total[gpu] == 125, f"GPU{gpu}: {gpu_total[gpu]}"
    return rows


def _make_r8w_plan_report(manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    return {
        "schema": "c2g.r8w.full_clean_2000_plan.2026-07-12.v1",
        "status": "PASS_C2G_R8W_FULL_CLEAN_2000_PLAN",
        "mode": "run",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "episode_count": 2000,
    }


class IdentityClosureTests(unittest.TestCase):
    def test_extract_exactly_500_l10(self):
        rows = _make_r8w_l10_rows()
        self.assertEqual(len(rows), 500)
        suites = Counter(r["suite"] for r in rows)
        self.assertEqual(suites, Counter({"libero_10": 500}))

    def test_all_500_unique(self):
        rows = _make_r8w_l10_rows()
        ids = [identity(r) for r in rows]
        self.assertEqual(len(set(ids)), 500)

    def test_validate_closure_passes(self):
        rows = _make_r8w_l10_rows()
        validate_l10_identity_closure(rows)

    def test_wrong_count_fails(self):
        rows = _make_r8w_l10_rows()[:499]
        with self.assertRaises(ValueError):
            validate_l10_identity_closure(rows)

    def test_duplicate_fails(self):
        rows = _make_r8w_l10_rows()
        rows[1] = dict(rows[0])
        with self.assertRaises(ValueError):
            validate_l10_identity_closure(rows)

    def test_wrong_suite_rejected(self):
        rows = _make_r8w_l10_rows()
        rows[0]["suite"] = "libero_object"
        with self.assertRaises(ValueError):
            validate_l10_identity_closure(rows)

    def test_wrong_max_steps_rejected(self):
        rows = _make_r8w_l10_rows()
        rows[0]["max_steps"] = 400
        with self.assertRaises(ValueError):
            validate_l10_identity_closure(rows)


class ShardClosureTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()

    def tearDown(self):
        self.td.cleanup()

    def _setup_plan(self, l10_rows=None):
        if l10_rows is None:
            l10_rows = _make_r8w_l10_rows()

        # Write a synthetic R8W master manifest with all 2000 rows
        all_rows = list(l10_rows)
        # Add non-L10 rows (placeholder)
        for suite in ["libero_object", "libero_spatial", "libero_goal"]:
            for i in range(500):
                all_rows.append({
                    "parent_key": f"{suite}/task_{i%10}/state_{i%50}/detector_train/episode_{i:03d}",
                    "suite": suite,
                    "task_index": i % 10,
                    "state_id": i % 50,
                    "cohort": DETECTOR_TRAIN if i < 300 else DETECTOR_VAL if i < 350 else DETECTOR_TEST if i < 400 else ATTACK_EVAL,
                    "split": "train" if i < 300 else "val" if i < 350 else "test" if i < 400 else "attack_eval",
                    "max_steps": 300,
                    "selection_seed": 42,
                    "eligible_for_detector_fit": i < 300,
                    "eligible_for_checkpoint_selection": False,
                    "eligible_for_threshold_calibration": False,
                    "eligible_for_clean_test": 350 <= i < 400,
                    "eligible_for_attack_evaluation": i >= 400,
                    "assigned_physical_gpu": 4 + (i % 4),
                    "assigned_worker_id": f"g{4 + (i % 4)}_{suite}",
                    "assigned_shard_id": f"{suite}__shard_{i % 4}",
                    "collection_purpose": "FULL_CLEAN_2000",
                })
        manifest_path = self.root / "r8w_manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows),
            encoding="utf-8",
        )

        import hashlib
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        plan_report_path = self.root / "r8w_plan.json"
        plan_report = _make_r8w_plan_report(manifest_path, manifest_sha)
        plan_report_path.write_text(
            json.dumps(plan_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        plan_sha = hashlib.sha256(plan_report_path.read_bytes()).hexdigest()

        return manifest_path, manifest_sha, plan_report_path, plan_sha

    def test_20_shard_closure(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        output = self.root / "plan_out"
        report = build_r8y_plan(
            mode="preview",
            repo=self.repo,
            expected_git_commit=self.head,
            source_r8w_plan_report=plan_path,
            expected_source_r8w_plan_report_sha256=plan_sha,
            source_r8w_master_manifest=manifest_path,
            expected_source_r8w_master_manifest_sha256=manifest_sha,
            output_root=output,
            authorization="",
            selection_seed=20260712,
        )
        self.assertEqual(report["episode_count"], 500)
        self.assertEqual(report["unique_identity_count"], 500)
        self.assertEqual(report["logical_shard_count"], 20)
        self.assertEqual(report["shards_per_gpu"], 5)
        self.assertEqual(report["episodes_per_shard"], 25)
        self.assertTrue(report["status"].startswith("PASS"))

        manifest = output / "c2g_r8y_l10_520_master_manifest.jsonl"
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 500)

        # All max_steps = 520
        for row in rows:
            self.assertEqual(row["max_steps"], 520)
            self.assertEqual(row["suite"], "libero_10")
            self.assertTrue(row.get("horizon_repair"))
            self.assertEqual(row.get("source_r8w_max_steps"), 300)
            self.assertEqual(row.get("canonical_max_steps"), 520)

        # Per-GPU: 125 each
        for gpu in GPUS:
            gpu_rows = [r for r in rows if r["assigned_physical_gpu"] == gpu]
            self.assertEqual(len(gpu_rows), 125, f"GPU {gpu} has {len(gpu_rows)}")

        # 20 shards, 25 each
        workers = Counter(r["assigned_worker_id"] for r in rows)
        self.assertEqual(len(workers), 20)
        for wid, count in workers.items():
            self.assertEqual(count, 25, f"{wid} has {count}")

        # Cohort closure
        cohorts = Counter(r["cohort"] for r in rows)
        self.assertEqual(cohorts[DETECTOR_TRAIN], 300)
        self.assertEqual(cohorts[DETECTOR_VAL], 50)
        self.assertEqual(cohorts[DETECTOR_TEST], 50)
        self.assertEqual(cohorts[ATTACK_EVAL], 100)

        # Identity preservation: all original identities preserved
        src_rows = [r for r in (
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ) if r["suite"] == "libero_10"]
        src_ids = {identity(r) for r in src_rows}
        new_ids = {identity(r) for r in rows}
        self.assertEqual(src_ids, new_ids)

    def test_shard_index_exists(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        output = self.root / "plan_out2"
        build_r8y_plan(
            mode="preview",
            repo=self.repo,
            expected_git_commit=self.head,
            source_r8w_plan_report=plan_path,
            expected_source_r8w_plan_report_sha256=plan_sha,
            source_r8w_master_manifest=manifest_path,
            expected_source_r8w_master_manifest_sha256=manifest_sha,
            output_root=output,
            authorization="",
            selection_seed=20260712,
        )
        si = output / "c2g_r8y_l10_520_shard_index.json"
        self.assertTrue(si.is_file())
        data = json.loads(si.read_text(encoding="utf-8"))
        self.assertEqual(len(data["shards"]), 20)

    def test_gpu_identity_preserved(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        # Verify original GPU assignments are preserved
        l10_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["suite"] == "libero_10"
        ]
        src_gpu_by_id = {identity(r): r["assigned_physical_gpu"] for r in l10_rows}

        output = self.root / "plan_out3"
        build_r8y_plan(
            mode="preview",
            repo=self.repo,
            expected_git_commit=self.head,
            source_r8w_plan_report=plan_path,
            expected_source_r8w_plan_report_sha256=plan_sha,
            source_r8w_master_manifest=manifest_path,
            expected_source_r8w_master_manifest_sha256=manifest_sha,
            output_root=output,
            authorization="",
            selection_seed=20260712,
        )

        new_rows = [
            json.loads(line)
            for line in (output / "c2g_r8y_l10_520_master_manifest.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in new_rows:
            self.assertEqual(
                row["assigned_physical_gpu"],
                src_gpu_by_id[identity(row)],
                f"GPU changed for {identity(row)}",
            )

    def test_sha256sums_created(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        output = self.root / "plan_out4"
        build_r8y_plan(
            mode="preview",
            repo=self.repo,
            expected_git_commit=self.head,
            source_r8w_plan_report=plan_path,
            expected_source_r8w_plan_report_sha256=plan_sha,
            source_r8w_master_manifest=manifest_path,
            expected_source_r8w_master_manifest_sha256=manifest_sha,
            output_root=output,
            authorization="",
            selection_seed=20260712,
        )
        self.assertTrue((output / "SHA256SUMS").is_file())
        self.assertTrue((output / "SHA256SUMS.sha256").is_file())

    def test_output_exists_fails(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        output = self.root / "plan_out5"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            build_r8y_plan(
                mode="preview",
                repo=self.repo,
                expected_git_commit=self.head,
                source_r8w_plan_report=plan_path,
                expected_source_r8w_plan_report_sha256=plan_sha,
                source_r8w_master_manifest=manifest_path,
                expected_source_r8w_master_manifest_sha256=manifest_sha,
                output_root=output,
                authorization="",
            )

    def test_task_balance_per_gpu(self):
        manifest_path, manifest_sha, plan_path, plan_sha = self._setup_plan()
        output = self.root / "plan_out6"
        build_r8y_plan(
            mode="preview",
            repo=self.repo,
            expected_git_commit=self.head,
            source_r8w_plan_report=plan_path,
            expected_source_r8w_plan_report_sha256=plan_sha,
            source_r8w_master_manifest=manifest_path,
            expected_source_r8w_master_manifest_sha256=manifest_sha,
            output_root=output,
            authorization="",
            selection_seed=20260712,
        )
        rows = [
            json.loads(line)
            for line in (output / "c2g_r8y_l10_520_master_manifest.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Per-GPU task balance: within each GPU, each task spread across 5 shards
        # with count difference <= 1
        for gpu in GPUS:
            gpu_rows = [r for r in rows if r["assigned_physical_gpu"] == gpu]
            task_shard = Counter()
            for r in gpu_rows:
                task_shard[(r["task_index"], r["assigned_shard_index"])] += 1
            for task in range(10):
                counts = [
                    task_shard.get((task, s), 0)
                    for s in range(LOGICAL_SHARDS_PER_GPU)
                ]
                diff = max(counts) - min(counts)
                self.assertLessEqual(
                    diff, 1,
                    f"GPU{gpu} task {task}: shard imbalance {diff}",
                )


if __name__ == "__main__":
    unittest.main()
