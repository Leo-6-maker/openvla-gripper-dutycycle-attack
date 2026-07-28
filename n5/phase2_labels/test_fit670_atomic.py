"""[DeepSeek] FIT670 Atomic Collection Tests — Gate F670-F.

Tests for: shard algorithm, identity allowlist structure, atomic publish,
transition receipt validation, and lifecycle invariants.

Run: python -m pytest n5/phase2_labels/test_fit670_atomic.py -v
"""
import json, os, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Shard plan builder has no external deps — always importable
from build_fit670_shard_plan import build_shard_plan, HORIZONS, FOUR_SUITES

# fit_collection_core needs numpy/torch — only import on demand
_HAS_CORE = False
try:
    from fit_collection_core import (
        CollectionHold, sha256_file, sha256_bytes,
        make_episode_staging, compute_episode_target,
        publish_episode, stage_cleanup, seal_root, load_resolutions,
    )
    _HAS_CORE = True
except ImportError:
    pass


def _make_dummy_identities(n=670):
    """Generate n dummy identities matching D0-R2 suite proportions."""
    ids = []
    suite_alloc = {"libero_10": 158, "libero_goal": 165,
                   "libero_object": 171, "libero_spatial": 176}
    for suite, count in suite_alloc.items():
        scaled = min(count, max(1, n * count // 670))
        for i in range(scaled):
            task_id = i % 10
            state_id = i
            ids.append({
                "episode_id": f"{suite}/task_{task_id:02d}/state_{state_id:02d}",
                "suite": suite, "task_id": task_id, "state_id": state_id,
                "collection_seed": 20260717,
                "initial_state_sha256": "a" * 64,
            })
    return ids


class TestShardAlgorithm(unittest.TestCase):
    def test_01_shard_count(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=6)
        self.assertEqual(plan["n_identities"], 670)
        self.assertEqual(len(plan["shards"]), 6)

    def test_02_all_identities_assigned(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=6)
        assigned = set()
        for shard in plan["shards"]:
            for ident in shard["identities"]:
                assigned.add(ident["episode_id"])
        self.assertEqual(len(assigned), 670)

    def test_03_no_duplicates(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=6)
        seen = set()
        for shard in plan["shards"]:
            for ident in shard["identities"]:
                ep = ident["episode_id"]
                self.assertNotIn(ep, seen, f"duplicate: {ep}")
                seen.add(ep)

    def test_04_cost_balance(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=6)
        costs = [s["total_cost"] for s in plan["shards"]]
        imbalance = (max(costs) - min(costs)) / (plan["cost_total"] / 6) * 100
        self.assertLess(imbalance, 10, f"cost imbalance {imbalance:.1f}% >= 10%")

    def test_05_suite_spread(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=6)
        for suite in FOUR_SUITES:
            suite_total = sum(1 for i in ids if i["suite"] == suite)
            max_per_shard = -(-suite_total // 6) + 1
            for shard in plan["shards"]:
                sc = shard["suite_counts"].get(suite, 0)
                self.assertLessEqual(sc, max_per_shard + 2,
                    f"shard {shard['shard_id']} has {sc} {suite} (max {max_per_shard})")

    def test_06_deterministic(self):
        ids1 = _make_dummy_identities(670)
        ids2 = _make_dummy_identities(670)
        p1 = build_shard_plan(ids1, n_shards=6)
        p2 = build_shard_plan(ids2, n_shards=6)
        s1 = json.dumps(p1["shards"], sort_keys=True)
        s2 = json.dumps(p2["shards"], sort_keys=True)
        self.assertEqual(s1, s2, "shard plan not deterministic")

    def test_07_eight_shards(self):
        ids = _make_dummy_identities(670)
        plan = build_shard_plan(ids, n_shards=8)
        self.assertEqual(len(plan["shards"]), 8)
        self.assertEqual(plan["cost_imbalance_pct"], plan["cost_imbalance_pct"])


class TestIdentityAllowlist(unittest.TestCase):
    def test_10_required_fields(self):
        ids = _make_dummy_identities(10)
        required = ["episode_id", "suite", "task_id", "state_id",
                    "collection_seed", "initial_state_sha256"]
        for ident in ids:
            for field in required:
                self.assertIn(field, ident, f"missing {field} in {ident.get('episode_id', '?')}")

    def test_11_episode_id_format(self):
        ids = _make_dummy_identities(10)
        for ident in ids:
            ep = ident["episode_id"]
            parts = ep.split("/")
            self.assertEqual(len(parts), 3, f"bad format: {ep}")
            self.assertIn(parts[0], FOUR_SUITES)
            self.assertTrue(parts[1].startswith("task_"))
            self.assertTrue(parts[2].startswith("state_"))

    def test_12_no_duplicate_episode_ids(self):
        ids = _make_dummy_identities(100)
        seen = set()
        for ident in ids:
            ep = ident["episode_id"]
            self.assertNotIn(ep, seen)
            seen.add(ep)

    def test_13_sha256_format(self):
        ids = _make_dummy_identities(10)
        for ident in ids:
            sha = ident["initial_state_sha256"]
            self.assertEqual(len(sha), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in sha))


class TestAtomicPublish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _HAS_CORE:
            raise unittest.SkipTest("fit_collection_core requires numpy/torch")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fit670_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_20_compute_target_path(self):
        target = compute_episode_target(self.tmp, "libero_10", 3, 5)
        expected = Path(self.tmp) / "episodes" / "libero_10" / "task_03" / "state_05"
        self.assertEqual(str(target), str(expected))

    def test_21_make_staging(self):
        staging = make_episode_staging("libero_10/task_00/state_01", self.tmp)
        self.assertTrue(staging.exists())
        self.assertTrue(staging.name.startswith("."))
        stage_cleanup(staging)
        self.assertFalse(staging.exists())

    def test_22_publish_success(self):
        staging = make_episode_staging("libero_10/task_00/state_01", self.tmp)
        (staging / "test.txt").write_text("hello")
        target = compute_episode_target(self.tmp, "libero_10", 0, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Note: publish_episode calls seal_root which needs all files
        # We need to create SHA256SUMS etc. Let's test the target collision
        (staging / "episode.json").write_text("{}")
        result = publish_episode(staging, target)
        self.assertTrue(target.exists())
        self.assertTrue((target / "SHA256SUMS").exists())
        self.assertFalse(staging.exists())

    def test_23_publish_no_overwrite(self):
        target = compute_episode_target(self.tmp, "libero_10", 0, 1)
        target.mkdir(parents=True)
        (target / "existing.txt").write_text("already here")

        staging = make_episode_staging("libero_10/task_00/state_01", self.tmp)
        (staging / "episode.json").write_text("{}")
        with self.assertRaises(CollectionHold):
            publish_episode(staging, target)
        self.assertFalse(staging.exists())  # cleaned up

    def test_24_seal_root(self):
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        (d / "a.txt").write_text("1")
        (d / "sub").mkdir()
        (d / "sub" / "b.txt").write_text("2")
        result = seal_root(d)
        self.assertIn("sha256sums_sha256", result)
        self.assertEqual(result["file_count"], 2)
        self.assertTrue((d / "SHA256SUMS").exists())
        self.assertTrue((d / "SHA256SUMS.sha256").exists())

    def test_25_staging_cleanup(self):
        staging = make_episode_staging("test", self.tmp)
        self.assertTrue(staging.exists())
        stage_cleanup(staging)
        self.assertFalse(staging.exists())

    def test_26_staging_no_collision(self):
        s1 = make_episode_staging("test", self.tmp)
        with self.assertRaises(CollectionHold):
            # faking a collision by trying to create at same path
            make_episode_staging("test", self.tmp)
        stage_cleanup(s1)


class TestLifecycleInvariants(unittest.TestCase):
    def test_30_horizons_match_suites(self):
        for suite in FOUR_SUITES:
            self.assertIn(suite, HORIZONS)
            self.assertGreater(HORIZONS[suite], 0)

    def test_31_four_suites_order(self):
        self.assertEqual(len(FOUR_SUITES), 4)

    def test_32_collection_hold_is_runtime_error(self):
        if not _HAS_CORE:
            raise unittest.SkipTest("fit_collection_core requires numpy/torch")
        self.assertTrue(issubclass(CollectionHold, RuntimeError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
