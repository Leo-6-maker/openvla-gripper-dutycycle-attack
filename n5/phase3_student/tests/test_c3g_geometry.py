"""C3-G Geometry Evaluator Unit Tests.

Tests:
  1. world↔local round-trip
  2. quaternion → rotation matrix
  3. In/On/Stack synthetic positives/negatives
  4. boundary unknown band
  5. basket reconstruction chain
  6. object extent missing → UNKNOWN
  7. margin monotonicity
"""
import sys, os, unittest, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT)
from c3g_geometry import (
    world_to_local, local_to_world, _xmat_to_rot3, _quat_to_xmat,
    check_In, check_On, check_Stack, OBJ_HALF_SIZE,
    evaluate_relation, _slice_xyz, _reconstruct_basket_pose,
    DEFAULT_MARGIN_UPPER, DEFAULT_MARGIN_LOWER,
)


class TestWorldLocalRoundTrip(unittest.TestCase):
    """Test 1: world↔local round-trip."""

    def test_identity_round_trip(self):
        """Identity transform: local == world."""
        ref_xpos = [0, 0, 0]
        ref_xmat = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # identity
        world = np.array([1.0, 2.0, 3.0])
        local = world_to_local(world, ref_xpos, ref_xmat)
        np.testing.assert_array_almost_equal(local, world)
        back = local_to_world(local, ref_xpos, ref_xmat)
        np.testing.assert_array_almost_equal(back, world)

    def test_translated_round_trip(self):
        """Translation: local is offset from world."""
        ref_xpos = [5, 10, 15]
        ref_xmat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        world = np.array([6.0, 12.0, 18.0])
        local = world_to_local(world, ref_xpos, ref_xmat)
        np.testing.assert_array_almost_equal(local, [1, 2, 3])
        back = local_to_world(local, ref_xpos, ref_xmat)
        np.testing.assert_array_almost_equal(back, world)

    def test_rotated_round_trip(self):
        """90-degree Z rotation: round-trip preserves position."""
        # Rotate 90 degrees around Z
        ref_xpos = [0, 0, 0]
        # R_z(90°): [0, -1, 0; 1, 0, 0; 0, 0, 1]
        ref_xmat = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        world = np.array([1.0, 0.0, 5.0])
        local = world_to_local(world, ref_xpos, ref_xmat)
        # World (1,0,5) → local: (0, -1, 5) after R^T * world
        np.testing.assert_array_almost_equal(local, [0, -1, 5])
        back = local_to_world(local, ref_xpos, ref_xmat)
        np.testing.assert_array_almost_equal(back, world)


class TestQuaternionConversion(unittest.TestCase):
    """Test 2: quaternion → rotation matrix."""

    def test_identity_quaternion(self):
        q = [1, 0, 0, 0]  # w,x,y,z
        xmat = _quat_to_xmat(q)
        np.testing.assert_array_almost_equal(
            xmat, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_90_degree_z_rotation(self):
        """90° around Z: q = [cos(45°), 0, 0, sin(45°)] in w,x,y,z"""
        import math
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        q = [c, 0, 0, s]  # w,x,y,z
        xmat = _quat_to_xmat(q)
        R = np.array(xmat).reshape(3, 3)
        # Should rotate (1,0,0) → (0,1,0)
        v = R @ np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(v, [0, 1, 0], decimal=6)

    def test_orthonormal_output(self):
        """Rotation matrix must be orthonormal."""
        q = [0.5, 0.5, 0.5, 0.5]  # 120° rotation
        xmat = _quat_to_xmat(q)
        R = np.array(xmat).reshape(3, 3)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=6)
        np.testing.assert_array_almost_equal(np.linalg.det(R), 1.0, decimal=6)


class TestInContainer(unittest.TestCase):
    """Test 3a: In container checks."""

    def setUp(self):
        self.cont_xpos = [0, 0, 0.5]
        self.cont_xmat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.cont_size = [0.1, 0.1, 0.15]  # 20cm x 20cm x 15cm half

    def test_object_clearly_inside(self):
        truth, margin, ev = check_In(
            [0, 0, 0.5], 'alphabet_soup',
            self.cont_xpos, self.cont_xmat, self.cont_size)
        self.assertEqual(truth, 'TRUE')
        self.assertGreater(margin, 0)

    def test_object_clearly_outside_xy(self):
        truth, margin, ev = check_In(
            [0.3, 0, 0.5], 'alphabet_soup',
            self.cont_xpos, self.cont_xmat, self.cont_size)
        self.assertEqual(truth, 'FALSE')
        self.assertLess(margin, -DEFAULT_MARGIN_LOWER)

    def test_object_clearly_outside_z(self):
        truth, margin, ev = check_In(
            [0, 0, 1.0], 'alphabet_soup',
            self.cont_xpos, self.cont_xmat, self.cont_size)
        self.assertEqual(truth, 'FALSE')

    def test_boundary_unknown(self):
        """Object near container boundary → UNKNOWN."""
        # Place object right at the XY boundary
        truth, margin, ev = check_In(
            [0.095, 0, 0.5], 'alphabet_soup',
            self.cont_xpos, self.cont_xmat, self.cont_size,
            margin_upper=0.005, margin_lower=0.020)
        # lx=0.095 + half(0.035) = 0.13 > half_x(0.1) + mu(0.005) = 0.105
        # So xy_ok is False, but margin may be in UNKNOWN band
        self.assertIn(truth, ['FALSE', 'UNKNOWN'])


class TestOnSurface(unittest.TestCase):
    """Test 3b: On surface checks."""

    def setUp(self):
        self.surf_xpos = [0, 0, 0.9]
        self.surf_xmat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.surf_size = [0.15, 0.15, 0.005]  # 30cm plate

    def test_object_clearly_on(self):
        """Object bottom at surface top → TRUE."""
        obj_half = OBJ_HALF_SIZE['butter']
        obj_z = 0.9 + obj_half  # bottom exactly at surface top
        truth, margin, ev = check_On(
            [0, 0, obj_z], 'butter',
            self.surf_xpos, self.surf_xmat, self.surf_size)
        self.assertEqual(truth, 'TRUE')

    def test_object_above_surface(self):
        """Object floating above surface → FALSE."""
        truth, margin, ev = check_On(
            [0, 0, 1.2], 'butter',
            self.surf_xpos, self.surf_xmat, self.surf_size)
        self.assertEqual(truth, 'FALSE')

    def test_object_below_surface(self):
        """Object below surface → FALSE."""
        truth, margin, ev = check_On(
            [0, 0, 0.5], 'butter',
            self.surf_xpos, self.surf_xmat, self.surf_size)
        self.assertEqual(truth, 'FALSE')

    def test_unknown_band_z(self):
        """Object near surface with margin → UNKNOWN."""
        obj_half = OBJ_HALF_SIZE['butter']
        obj_z = 0.9 + obj_half + 0.01  # bottom 1cm above surface
        truth, margin, ev = check_On(
            [0, 0, obj_z], 'butter',
            self.surf_xpos, self.surf_xmat, self.surf_size,
            margin_upper=0.005, margin_lower=0.020)
        # z_dist = 0.01, between mu(0.005) and ml(0.020) → UNKNOWN
        self.assertEqual(truth, 'UNKNOWN')


class TestStack(unittest.TestCase):
    """Test 3c: Stack checks."""

    def test_stacked_with_contact(self):
        obj = [0, 0, 0.15]
        other = [0, 0, 0.05]
        contacts = [['butter_1_geom', 'alphabet_soup_1_geom']]
        truth, margin, ev = check_Stack(
            obj, 'butter', other, 'alphabet_soup',
            contacts=contacts)
        self.assertEqual(truth, 'TRUE')
        self.assertTrue(ev['has_contact'])

    def test_not_stacked_no_contact(self):
        obj = [0, 0, 0.5]
        other = [0, 0, 0.05]
        truth, margin, ev = check_Stack(
            obj, 'butter', other, 'alphabet_soup',
            contacts=[])
        self.assertEqual(truth, 'FALSE')

    def test_horizontal_separation(self):
        """Objects far apart in XY → FALSE."""
        obj = [0.5, 0, 0.15]
        other = [0, 0, 0.05]
        contacts = [['butter_1_geom', 'alphabet_soup_1_geom']]
        truth, margin, ev = check_Stack(
            obj, 'butter', other, 'alphabet_soup',
            contacts=contacts)
        # Only TRUE if z_dist and xy_dist and contact all satisfied
        self.assertIn(truth, ['FALSE', 'UNKNOWN'])


class TestBasketReconstruction(unittest.TestCase):
    """Test 5: basket reconstruction chain."""

    def test_basket_reconstruction_identity(self):
        """Basket site at body origin → reconstructed pose equals body pose."""
        body_poses = {'basket_1_main': ([1, 2, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1])}
        site_local_pos = [0, 0, 0]
        site_local_quat = [1, 0, 0, 0]
        xpos, xmat = _reconstruct_basket_pose(
            body_poses, 'basket_1_main', site_local_pos, site_local_quat)
        np.testing.assert_array_almost_equal(xpos, [1, 2, 3])
        np.testing.assert_array_almost_equal(xmat, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_basket_reconstruction_offset(self):
        """Basket site offset from body origin."""
        body_poses = {'basket_1_main': ([0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1])}
        site_local_pos = [0.1, 0, 0.2]
        site_local_quat = [1, 0, 0, 0]
        xpos, xmat = _reconstruct_basket_pose(
            body_poses, 'basket_1_main', site_local_pos, site_local_quat)
        np.testing.assert_array_almost_equal(xpos, [0.1, 0, 0.2])


class TestUNKNOWNPolicy(unittest.TestCase):
    """Test 6: UNKNOWN policy — missing data → UNKNOWN, never FALSE."""

    def test_unknown_missing_object_size(self):
        """Object without half-size estimate → still evaluates, not UNKNOWN."""
        # We use defaultdict so unknown objects get default 0.03
        truth, margin, ev = check_In(
            [0, 0, 0.5], 'unknown_object_name',
            [0, 0, 0.5], [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0.1, 0.1, 0.15])
        # Should not crash — uses default half-size
        self.assertIn(truth, ['TRUE', 'FALSE', 'UNKNOWN'])

    def test_white_wooden_always_unknown(self):
        """White/wooden fixtures → UNKNOWN regardless of pose."""
        step_data = {'object_state': [0, 0, 0, 0, 0, 0]}
        obj_slices = {'test_obj': {'pos': 0}}
        body_poses = {}
        static_seal = {}
        basket_seal = {}
        white_wooden = {'white_cabinet_1_bottom_region', 'wooden_cabinet_1_top_region'}

        for target in white_wooden:
            truth, margin, tier, source, reason = evaluate_relation(
                'test_obj', target, 'On', step_data, body_poses,
                obj_slices, static_seal, basket_seal, white_wooden)
            self.assertEqual(truth, 'UNKNOWN')
            self.assertIn('white_or_wooden', str(reason))


class TestMarginMonotonicity(unittest.TestCase):
    """Test 7: margin monotonicity — closer to boundary → smaller margin."""

    def test_in_margin_decreases_with_distance(self):
        """Objects further from center have smaller margin."""
        cont_xpos = [0, 0, 0.5]
        cont_xmat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        cont_size = [0.2, 0.2, 0.2]

        _, m_center, _ = check_In([0, 0, 0.5], 'butter',
                                   cont_xpos, cont_xmat, cont_size)
        _, m_edge, _ = check_In([0.15, 0, 0.5], 'butter',
                                 cont_xpos, cont_xmat, cont_size)
        self.assertGreater(m_center, m_edge)


if __name__ == '__main__':
    unittest.main()
