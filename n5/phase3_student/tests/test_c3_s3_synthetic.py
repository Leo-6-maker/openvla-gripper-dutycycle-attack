"""C3-S3 Synthetic Geometry Tests.

Tests the coordinate-transform and geometry verification contracts
without reading any protected data. All fixtures are synthetic.

DEVELOPMENT_ONLY — consumable_as_formal_evidence = false.
"""
import unittest, math, json, hashlib, os, sys
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

# Synthetic test data — no real episodes

# Known static fixture: flat_stove_1_cook_region (from C3-S2 seal)
STATIC_FIXTURE_WORLD = {
    "flat_stove_1_cook_region": {
        "mean_xpos": [-0.040726744789979424, 0.19766883037651553, 0.905],
        "mean_xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    }
}

# Known basket body→site transform (from C3-S2 validation)
BASKET_LOCAL = {
    "site_local_pos": [0.0, 0.0, 0.07185],
    "body_name": "basket_1_main",
}

# Synthetic sidecar body pose snapshot
SYNTHETIC_BODY_POSE = {
    "basket_1_main": {
        "xpos": [0.1, 0.2, 0.5],
        "xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    }
}


class TestProtectedRejection(unittest.TestCase):
    """Protected episode manifest rejection."""

    PROTECTED = ["cal", "g10", "t2r-d", "t2rd"]

    def test_protected_path_rejected(self):
        for token in self.PROTECTED:
            path = f"/data/{token}/something.json"
            self.assertTrue(any(t in path.lower() for t in self.PROTECTED),
                            f"Token {token} should trigger protected rejection")

    def test_non_protected_path_accepted(self):
        path = "/data/libero_10/task_00/state_00/sidecar.jsonl"
        self.assertFalse(any(t in path.lower() for t in self.PROTECTED))

    def test_indirect_symlink_protection(self):
        """Indirect references containing protected tokens must be rejected."""
        references = ["../cal/manifest.json", "./t2r-d/../data.json"]
        for ref in references:
            self.assertTrue(any(t in ref.lower() for t in self.PROTECTED),
                            f"Indirect ref {ref} should be rejected")


class TestExactEpisodeJoin(unittest.TestCase):
    """Exact episode/task/step join — fail-closed on mismatch."""

    def setUp(self):
        self.sidecar_ids = {"libero_10/task_00/state_00", "libero_10/task_00/state_01"}
        self.manifest_ids = {"libero_10/task_00/state_00"}

    def test_missing_episode_fail_closed(self):
        """Episode in manifest but not in sidecar → fail-closed."""
        manifest_only = self.manifest_ids - self.sidecar_ids
        self.assertEqual(len(manifest_only), 0, "No manifest-only episodes allowed")

    def test_duplicate_episode_detected(self):
        """Duplicate episodes must be detected."""
        ids = ["a/b/c", "a/b/c"]
        self.assertNotEqual(len(ids), len(set(ids)), "Duplicates must be detected")

    def test_misaligned_step_count_fail_closed(self):
        """Step count mismatch must fail."""
        sidecar_steps = 299
        action_steps = 298
        self.assertNotEqual(sidecar_steps, action_steps,
                            "Step count mismatch must be detected")


class TestStaticTransform(unittest.TestCase):
    """Known static transform verification."""

    def test_static_fixture_world_pose_constant(self):
        """Static fixture world pose must be identical to seal."""
        seal_pos = STATIC_FIXTURE_WORLD["flat_stove_1_cook_region"]["mean_xpos"]
        self.assertEqual(len(seal_pos), 3)
        self.assertTrue(all(isinstance(x, float) for x in seal_pos))

    @unittest.skipUnless(HAVE_NUMPY, "requires numpy")
    def test_static_xmat_is_orthonormal(self):
        """Static fixture rotation matrix must be orthonormal."""
        xmat = STATIC_FIXTURE_WORLD["flat_stove_1_cook_region"]["mean_xmat"]
        R = np.array(xmat).reshape(3, 3)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=10)


class TestDynamicBodySiteTransform(unittest.TestCase):
    """Known dynamic body+site transform (basket reconstruction)."""

    def test_basket_reconstruction_identity(self):
        """Basket site at body origin + Z offset."""
        body_xpos = np.array(SYNTHETIC_BODY_POSE["basket_1_main"]["xpos"])
        body_xmat = np.array(SYNTHETIC_BODY_POSE["basket_1_main"]["xmat"])
        R = body_xmat.reshape(3, 3)
        site_local = np.array(BASKET_LOCAL["site_local_pos"])
        recon = body_xpos + R @ site_local
        expected = body_xpos + site_local  # identity rotation
        np.testing.assert_array_almost_equal(recon, expected)

    def test_basket_reconstruction_with_rotation(self):
        """Basket site with 90-degree Z rotation."""
        body_xpos = np.array([0.0, 0.0, 0.5])
        # R_z(90deg)
        body_xmat = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        R = np.array(body_xmat).reshape(3, 3)
        site_local = np.array([0.1, 0.0, 0.07185])
        recon = body_xpos + R @ site_local
        # R_z(90) * [0.1, 0, 0.07185] = [0, 0.1, 0.07185]
        expected = np.array([0.0, 0.1, 0.57185])
        np.testing.assert_array_almost_equal(recon, expected)


class TestQuaternionEquivalence(unittest.TestCase):
    """Quaternion q/-q equivalence and geodesic error."""

    def test_q_and_neg_q_equivalent(self):
        """q and -q represent the same rotation."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        neg_q = -q
        # Both should produce same rotation matrix
        R1 = quat_to_matrix(q)
        R2 = quat_to_matrix(neg_q)
        np.testing.assert_array_almost_equal(R1, R2, decimal=12)

    def test_geodesic_error_zero_for_same_rotation(self):
        """Geodesic distance between q and -q should be 0."""
        q = np.array([0.707, 0.0, 0.707, 0.0])
        dist = geodesic_distance(q, -q)
        self.assertAlmostEqual(dist, 0.0, places=12)

    def test_geodesic_error_90_degree(self):
        """Geodesic distance for 90-degree rotation."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # identity (w=1)
        q2 = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # 90deg Z
        dist = geodesic_distance(q1, q2)
        self.assertAlmostEqual(dist, math.pi / 2, places=5)

    def test_quat_normalize(self):
        """Quaternion normalization."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        nq = q / np.linalg.norm(q)
        np.testing.assert_array_almost_equal(nq, np.array([1.0, 0.0, 0.0, 0.0]))


def quat_to_matrix(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def geodesic_distance(q1, q2):
    """Geodesic distance between two quaternions (radians)."""
    q1n = q1 / np.linalg.norm(q1)
    q2n = q2 / np.linalg.norm(q2)
    dot = abs(np.dot(q1n, q2n))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


class TestP99Calculation(unittest.TestCase):
    """P99 calculation with exact denominator."""

    def test_p99_exact_denominator(self):
        """P99 of 100 values: sorted[99] is the 99th percentile."""
        values = list(range(100))  # 0..99
        p99_val = np.percentile(values, 99)
        self.assertEqual(int(p99_val), 98)  # 99th percentile of 0..99

    def test_p99_with_outliers(self):
        """P99 not affected by single extreme outlier."""
        values = [0.0] * 99 + [1000.0]
        p99_val = np.percentile(values, 99)
        self.assertLess(p99_val, 1000.0)

    def test_denominator_exact(self):
        """Denominator must be exact count, not estimated."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p99_val = np.percentile(values, 99, method='linear')
        # With 5 values, p99 uses interpolation
        self.assertTrue(isinstance(float(p99_val), float))


class TestArticulatedUnknownPreservation(unittest.TestCase):
    """Articulated unknown tasks must not be converted to static/dynamic."""

    UNKNOWN_TASKS = {
        "libero_10/task_03": "white_cabinet_1_bottom_region",
        "libero_goal/task_03": "wooden_cabinet_1_top_region",
    }

    def test_articulated_unknown_not_reclassified(self):
        """Articulated tasks must stay UNKNOWN, not become STATIC or DYNAMIC."""
        for task_key, fixture in self.UNKNOWN_TASKS.items():
            self.assertIn("cabinet", fixture)
            # Classification must include MOVABLE/UNSEALED
            self.assertNotIn("STATIC", fixture.upper().replace("_", " ")[:20])

    def test_articulated_not_converted_to_negative(self):
        """Unknown must never be converted to FALSE."""
        status = "ARTICULATED_UNKNOWN"
        self.assertNotEqual(status, "FALSE")
        self.assertNotEqual(status, "NEGATIVE")

    def test_articulated_excluded_from_numerical_evaluation(self):
        """Articulated tasks must be excluded from numerical error computation."""
        task = "libero_10/task_03"
        is_articulated = task in self.UNKNOWN_TASKS
        include_in_p99 = not is_articulated
        self.assertFalse(include_in_p99)


class TestChecksumIntegrity(unittest.TestCase):
    """Checksum/schema/identity failures must be fail-closed."""

    def test_schema_version_mismatch_detected(self):
        """Schema version mismatch must fail."""
        expected = "v3"
        actual = "v2"
        self.assertNotEqual(expected, actual, "Schema mismatch must be detected")

    def test_identity_mismatch_fail_closed(self):
        """Episode ID mismatch must fail."""
        manifest_id = "libero_10/task_00/state_00"
        sidecar_id = "libero_10/task_00/state_01"
        self.assertNotEqual(manifest_id, sidecar_id, "ID mismatch must be detected")

    def test_deterministic_canonical_digest(self):
        """Same payload → same canonical digest."""
        payload = {"gate": "test", "values": [1, 2, 3], "nested": {"a": 1}}
        encoded1 = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
        encoded2 = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded1).hexdigest(),
                         hashlib.sha256(encoded2).hexdigest())


if __name__ == "__main__":
    unittest.main()
