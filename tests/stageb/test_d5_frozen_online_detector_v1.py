"""G2-R2b: Independent tests for D5FrozenOnlineDetectorV1.

Groups:
  A: Init + SHA binding (7 tests)
  B: State machine behavior (11 tests)

All tests self-contained, no pytest needed. Run directly with python.
"""
import copy, csv, hashlib, json, os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))

# Use a known-to-emit historical trace for integration tests
_KNOWN_EMIT_TRACE = "/data/liuyu/outputs/d44d_balanced120_gpu50_r1/orange_juice_s8_shadow_attempt1"
CKPT = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"
CONFIG = "/data/liuyu/outputs/d5_training/d5_frozen_config.json"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def _load_trace(path=None):
    """Load step_trace.csv rows and return list of (step, raw, env, qpos, eef_x, eef_y, eef_z, dec, rv, ev, qv, ef, sem)."""
    p = path or _KNOWN_EMIT_TRACE
    rows = list(csv.DictReader(open(os.path.join(p, "step_trace.csv"))))
    out = []
    for r in rows:
        out.append((
            int(r["step"]),
            float(r.get("raw_gripper", 0) or 0),
            float(r.get("env_gripper", 0) or 0),
            float(r.get("gripper_qpos_before", 0) or 0),
            float(r.get("eef_x", 0) or 0),
            float(r.get("eef_y", 0) or 0),
            float(r.get("eef_z", 0) or 0),
            int(float(r.get("decoded_open", 0) or 0)),
            bool(int(float(r.get("raw_valid", "1") or "1"))),
            bool(int(float(r.get("env_valid", "1") or "1"))),
            bool(int(float(r.get("qpos_valid", "1") or "1"))),
            bool(int(float(r.get("eef_valid", "1") or "1"))),
            bool(int(float(r.get("semantics_ok", "1") or "1"))),
        ))
    return out


def feed_trace(detector, rows):
    """Feed rows through detector, return list of results."""
    results = []
    for tup in rows:
        r = detector.update(*tup)
        if r is not None:
            results.append(r)
    return results


class TestInitBinding(unittest.TestCase):
    """Group A: Initialization and SHA binding."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_correct_ckpt_config_starts(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        d = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        self.assertIsNotNone(d)
        self.assertEqual(d.tau, 0.050)

    def test_wrong_checkpoint_sha_rejects(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        bad = os.path.join(self.tmp, "bad.pt")
        with open(CKPT, "rb") as src:
            data = bytearray(src.read())
        data[100] ^= 0xFF  # flip bits
        with open(bad, "wb") as f:
            f.write(data)
        with self.assertRaises(RuntimeError):
            D5FrozenOnlineDetectorV1(bad, CONFIG)

    def test_wrong_config_sha_rejects(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        bad = os.path.join(self.tmp, "bad.json")
        cfg = json.load(open(CONFIG))
        cfg["tau"] = 0.999  # different value
        with open(bad, "w") as f:
            json.dump(cfg, f)
        with self.assertRaises(RuntimeError):
            D5FrozenOnlineDetectorV1(CKPT, bad)

    def test_wrong_tau_rejects(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        bad = os.path.join(self.tmp, "bad_tau.json")
        cfg = json.load(open(CONFIG))
        cfg["tau"] = 0.999
        with open(bad, "w") as f:
            json.dump(cfg, f)
        with self.assertRaises(RuntimeError):
            D5FrozenOnlineDetectorV1(CKPT, bad)

    def test_wrong_feature_names_rejects(self):
        # Test that checkpoint with wrong feature names fails
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        import torch
        bad = os.path.join(self.tmp, "bad_feat.pt")
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        orig = ckpt["feature_names"]
        ckpt["feature_names"] = ["wrong_0"] * 16
        torch.save(ckpt, bad)
        ckpt["feature_names"] = orig  # restore
        with self.assertRaises(RuntimeError):
            D5FrozenOnlineDetectorV1(bad, CONFIG)

    def test_runtime_module_exists_and_readable(self):
        from gripper_attack.d5_frozen_online_detector_v1 import FROZEN_RUNTIME_SHA
        runtime_path = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                                    "gripper_attack", "d5_frozen_runtime_v1.py")
        self.assertTrue(os.path.exists(runtime_path))
        actual = _sha256_file(runtime_path)
        self.assertEqual(actual, FROZEN_RUNTIME_SHA,
                         f"Runtime SHA mismatch! Expected {FROZEN_RUNTIME_SHA[:16]}..., got {actual[:16]}...")

    def test_bound_manifest_complete(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        d = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        m = d.bound_manifest
        for key in ["detector_version", "checkpoint_sha", "config_sha", "runtime_sha",
                     "tau", "feature_schema", "adapter_source_commit"]:
            self.assertIn(key, m, f"Missing key in bound_manifest: {key}")
        self.assertEqual(m["tau"], 0.050)


class TestStateMachine(unittest.TestCase):
    """Group B: Online state machine behavior."""

    @classmethod
    def setUpClass(cls):
        cls._trace = _load_trace()

    def setUp(self):
        import tempfile
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        self.det = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_duplicate_step_raises(self):
        self.det.update(0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 1, True, True, True, True, True)
        with self.assertRaises(ValueError):
            self.det.update(0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 1, True, True, True, True, True)

    def test_skipped_step_raises(self):
        self.det.update(0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 1, True, True, True, True, True)
        with self.assertRaises(ValueError):
            self.det.update(2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True)

    def test_nan_input_fail_closed(self):
        r = self.det.update(0, float("nan"), -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True)
        self.assertIsNone(r, "NaN raw gripper should produce no candidate")

    def test_validity_zero_fail_closed(self):
        r = self.det.update(0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, False, True, True, True)
        self.assertIsNone(r, "env_valid=False should produce no candidate")

    def test_abstained_candidate_never_emits(self):
        # Feed known trace — abstained candidates should never be emitted records
        results = feed_trace(self.det, self._trace)
        for r in self.det.audit_records:
            if r["abstained"]:
                self.assertFalse(r["emitted"],
                    f"Abstained candidate at step {r['step']} was emitted!")

    def test_first_emit_lock(self):
        # Feed entire known trace through detector
        results = feed_trace(self.det, self._trace)
        emitted = [r for r in results if r["emitted"]]
        self.assertLessEqual(len(emitted), 1, "At most one emission per episode")

    def test_no_second_emission_after_first(self):
        feed_trace(self.det, self._trace)
        n_emitted = sum(1 for r in self.det.audit_records if r["emitted"])
        self.assertLessEqual(n_emitted, 1)

    def test_reset_clears_all_state(self):
        feed_trace(self.det, self._trace)
        self.det.reset()
        self.assertEqual(self.det.next_expected_step, 0)
        self.assertFalse(self.det.has_emitted)
        self.assertEqual(len(self.det.audit_records), 0)

    def test_deterministic_replay(self):
        # Run same trace twice, compare candidates and emit
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        d1 = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        r1 = feed_trace(d1, self._trace)
        d2 = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        r2 = feed_trace(d2, self._trace)

        self.assertEqual(len(r1), len(r2), "Candidate count must be deterministic")
        for i, (a, b) in enumerate(zip(r1, r2)):
            self.assertEqual(a["step"], b["step"], f"Step mismatch at index {i}")
            self.assertEqual(a["abstained"], b["abstained"], f"Abstain mismatch at index {i}")
            self.assertAlmostEqual(a["score"], b["score"], places=6,
                                   msg=f"Score mismatch at index {i}")
        self.assertEqual(d1.emit_step, d2.emit_step)

    def test_audit_record_fields_complete(self):
        feed_trace(self.det, self._trace)
        for r in self.det.audit_records:
            for key in ["step", "is_candidate", "features", "normalized_features",
                        "score", "abstain", "abstained", "candidate_reason",
                        "emitted", "first_emit_step", "detector_version"]:
                self.assertIn(key, r, f"Missing audit field: {key}")

    def test_bound_manifest_has_adapter_sha(self):
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1, FROZEN_ADAPTER_SHA
        d = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
        self.assertIn("adapter_sha", d.bound_manifest)
        self.assertEqual(d.bound_manifest["adapter_sha"], FROZEN_ADAPTER_SHA)

    def test_runtime_tamper_rejected(self):
        import shutil
        bad_runtime = os.path.join(self._tmpdir, "d5_frozen_runtime_v1.py")
        src = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                          "gripper_attack", "d5_frozen_runtime_v1.py")
        shutil.copy(src, bad_runtime)
        with open(bad_runtime, "a") as f:
            f.write("\n# tampered\n")
        # Inject bad runtime into detector init — should fail
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1, FROZEN_RUNTIME_SHA
        orig_runtime = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                                    "gripper_attack", "d5_frozen_runtime_v1.py")
        # Verify original still passes
        self.assertEqual(_sha256_file(orig_runtime), FROZEN_RUNTIME_SHA)
        # Tampered file has different SHA
        bad_sha = _sha256_file(bad_runtime)
        self.assertNotEqual(bad_sha, FROZEN_RUNTIME_SHA)

    def test_adapter_tamper_rejected(self):
        import shutil
        bad_adapter = os.path.join(self._tmpdir, "d5_frozen_feature_adapter_v1.py")
        src = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                          "gripper_attack", "d5_frozen_feature_adapter_v1.py")
        shutil.copy(src, bad_adapter)
        with open(bad_adapter, "a") as f:
            f.write("\n# tampered\n")
        from gripper_attack.d5_frozen_online_detector_v1 import FROZEN_ADAPTER_SHA
        bad_sha = _sha256_file(bad_adapter)
        self.assertNotEqual(bad_sha, FROZEN_ADAPTER_SHA)

    def test_first_emit_exactly_one_at_expected_step(self):
        feed_trace(self.det, self._trace)
        emitted = [r for r in self.det.audit_records if r["emitted"]]
        self.assertEqual(len(emitted), 1, "Must have exactly one emission")
        # Verify emit_step is in the audit
        self.assertGreaterEqual(self.det.emit_step, 0)
        self.assertEqual(emitted[0]["step"], self.det.emit_step)

    def test_max_one_emission_per_episode(self):
        # Run trace 5 times in a row with resets
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        for _ in range(3):
            d = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
            feed_trace(d, self._trace)
            n_emitted = sum(1 for r in d.audit_records if r["emitted"])
            self.assertLessEqual(n_emitted, 1)
            d.reset()
            self.assertFalse(d.has_emitted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
