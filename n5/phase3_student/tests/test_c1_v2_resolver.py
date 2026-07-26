"""C1-V2 Resolver Contract Tests.

Tests the pure-function resolver with synthetic entity data.
No MuJoCo dependency — tests are deterministic and repeatable.
"""
import unittest, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from t2rc1_v2_registry import (
    resolve_entity, resolve_relation,
    _is_region, _verify_alias_hierarchy_full,
    VALID_RESOLUTIONS, BLOCKED_RESOLUTIONS,
)


def _make_synthetic_entities():
    """Build synthetic sites/bodies/geoms mimicking a real LIBERO task.

    Pattern: objects like alphabet_soup_1 have body alphabet_soup_1_main
    with geoms alphabet_soup_1_g0 through alphabet_soup_1_gN.
    Regions like basket_1_contain_region are sites on basket_1_main body.
    """
    bodies = {
        'alphabet_soup_1_main': {'id': 1},
        'butter_1_main': {'id': 2},
        'basket_1_main': {'id': 3},
        'plate_1_main': {'id': 4},
        'akita_black_bowl_1_main': {'id': 5},
    }
    sites = {
        'basket_1_contain_region': {
            'id': 10, 'body_id': 3, 'size': [0.06, 0.06, 0.07],
            'pos': [0, 0, 0.07], 'quat': [0, 0, 0, 1], 'type': 0,
        },
        'flat_stove_1_cook_region': {
            'id': 11, 'body_id': 99, 'size': [0.075, 0.075, 0.0025],
            'pos': [0, 0, 0], 'quat': [0, 0, 0, 1], 'type': 0,
        },
    }
    geoms = {}
    for name, body_id in [('alphabet_soup_1', 1), ('butter_1', 2),
                           ('plate_1', 4), ('akita_black_bowl_1', 5)]:
        for i in range(5):
            gn = f'{name}_g{i}'
            geoms[gn] = {'id': 100 + body_id * 100 + i, 'body_id': body_id}
    return sites, bodies, geoms


class TestStructuralRegionDetection(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _make_synthetic_entities()

    def test_region_when_in_sites(self):
        self.assertTrue(_is_region('basket_1_contain_region', self.sites))

    def test_not_region_when_not_in_sites(self):
        self.assertFalse(_is_region('alphabet_soup_1', self.sites))
        self.assertFalse(_is_region('plate_1', self.sites))

    def test_region_detection_is_structural_not_suffix(self):
        """A name with _region suffix but NOT in sites is NOT a region."""
        self.assertFalse(_is_region('fake_contain_region', self.sites))


class TestResolveEntity(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _make_synthetic_entities()

    def test_alias_for_bddl_object(self):
        """BDDL names like alphabet_soup_1 are NOT bodies — only _main is."""
        r = resolve_entity('alphabet_soup_1', False, self.sites, self.bodies, self.geoms)
        self.assertEqual(r['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(r['entity_id'], 1)
        self.assertEqual(r['alias_to'], 'alphabet_soup_1_main')

    def test_exact_body_when_raw_name_is_body(self):
        """If a BDDL name is literally a body name, it resolves EXACT_BODY."""
        bodies2 = dict(self.bodies)
        bodies2['butter_1'] = {'id': 2}  # BDDL name IS the body name
        r = resolve_entity('butter_1', False, self.sites, bodies2, self.geoms)
        self.assertEqual(r['resolution'], 'EXACT_BODY')

    def test_exact_site(self):
        r = resolve_entity('basket_1_contain_region', True, self.sites, self.bodies, self.geoms)
        self.assertEqual(r['resolution'], 'EXACT_SITE')
        self.assertEqual(r['entity_id'], 10)

    def test_alias_main_suffix(self):
        r = resolve_entity('plate_1', False, self.sites, self.bodies, self.geoms)
        self.assertEqual(r['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(r['entity_id'], 4)
        self.assertEqual(r['alias_to'], 'plate_1_main')

    def test_unresolved_unknown(self):
        r = resolve_entity('nonexistent_object', False, self.sites, self.bodies, self.geoms)
        self.assertEqual(r['resolution'], 'UNRESOLVED')

    def test_unresolved_alias_fails_verification(self):
        # Add a body that matches _main but has wrong geoms
        sites2 = dict(self.sites)
        bodies2 = dict(self.bodies)
        bodies2['fake_obj_1_main'] = {'id': 99}
        # No geoms with prefix fake_obj_1_ → hierarchy verification fails
        r = resolve_entity('fake_obj_1', False, sites2, bodies2, self.geoms)
        self.assertEqual(r['resolution'], 'UNRESOLVED')
        self.assertIn('alias_verification_failed', r.get('error_detail', {}).get('reason', ''))

    def test_blocked_region_as_body(self):
        # A region name that matches a body → BLOCKED
        sites2 = dict(self.sites)
        bodies2 = dict(self.bodies)
        bodies2['my_region'] = {'id': 99}
        r = resolve_entity('my_region', True, sites2, bodies2, self.geoms)
        self.assertEqual(r['resolution'], 'BLOCKED_REGION_AS_BODY')

    def test_blocked_region_as_geom(self):
        # A region name that matches a geom → BLOCKED
        sites2 = dict(self.sites)
        geoms2 = dict(self.geoms)
        geoms2['my_region'] = {'id': 99, 'body_id': 1}
        r = resolve_entity('my_region', True, sites2, self.bodies, geoms2)
        self.assertEqual(r['resolution'], 'BLOCKED_REGION_AS_GEOM')

    def test_ambiguous_multi_type(self):
        # Name exists as both site and body → AMBIGUOUS
        sites2 = dict(self.sites)
        bodies2 = dict(self.bodies)
        sites2['ambiguous_name'] = {'id': 50, 'body_id': 1, 'size': [1,1,1],
                                     'pos': [0,0,0], 'quat': [0,0,0,1], 'type': 0}
        bodies2['ambiguous_name'] = {'id': 50}
        r = resolve_entity('ambiguous_name', False, sites2, bodies2, self.geoms)
        self.assertEqual(r['resolution'], 'AMBIGUOUS')

    def test_ambiguous_multiple_alias_candidates(self):
        # Multiple bodies starting with prefix → AMBIGUOUS
        bodies2 = dict(self.bodies)
        bodies2['multi_obj_main'] = {'id': 50}
        bodies2['multi_obj_other'] = {'id': 51}
        r = resolve_entity('multi_obj', False, self.sites, bodies2, self.geoms)
        self.assertEqual(r['resolution'], 'AMBIGUOUS')
        self.assertIn('multiple_alias_candidates', r.get('error_detail', {}).get('reason', ''))


class TestResolveRelation(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _make_synthetic_entities()

    def test_full_relation_ok(self):
        """Both object and target resolve: object via alias, target via EXACT_SITE."""
        rel = resolve_relation('In', 'alphabet_soup_1',
                               'basket_1_contain_region',
                               self.sites, self.bodies, self.geoms)
        self.assertTrue(rel['relation_ok'])
        self.assertEqual(rel['object_resolution']['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(rel['target_resolution']['resolution'], 'EXACT_SITE')

    def test_full_relation_ok_with_alias(self):
        rel = resolve_relation('On', 'plate_1', 'akita_black_bowl_1',
                               self.sites, self.bodies, self.geoms)
        self.assertTrue(rel['relation_ok'])
        self.assertEqual(rel['object_resolution']['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(rel['target_resolution']['resolution'], 'APPROVED_STRUCTURAL_ALIAS')

    def test_relation_fails_on_object_unresolved(self):
        rel = resolve_relation('On', 'nonexistent', 'basket_1_contain_region',
                               self.sites, self.bodies, self.geoms)
        self.assertFalse(rel['relation_ok'])
        self.assertEqual(rel['object_resolution']['resolution'], 'UNRESOLVED')

    def test_relation_fails_on_target_ambiguous(self):
        sites2 = dict(self.sites)
        bodies2 = dict(self.bodies)
        sites2['ambig'] = {'id': 50, 'body_id': 1, 'size': [1,1,1],
                           'pos': [0,0,0], 'quat': [0,0,0,1], 'type': 0}
        bodies2['ambig'] = {'id': 50}
        rel = resolve_relation('On', 'butter_1', 'ambig',
                               sites2, bodies2, self.geoms)
        self.assertFalse(rel['relation_ok'])
        self.assertTrue(rel['target_ambiguous'])

    def test_target_is_region_detected(self):
        rel = resolve_relation('In', 'alphabet_soup_1',
                               'basket_1_contain_region',
                               self.sites, self.bodies, self.geoms)
        self.assertTrue(rel['target_is_region'])

    def test_target_not_region_for_body(self):
        rel = resolve_relation('On', 'plate_1', 'akita_black_bowl_1',
                               self.sites, self.bodies, self.geoms)
        self.assertFalse(rel['target_is_region'])


class TestHierarchyVerificationFull(unittest.TestCase):
    def setUp(self):
        _, self.bodies, self.geoms = _make_synthetic_entities()

    def test_all_geoms_checked(self):
        ok, detail = _verify_alias_hierarchy_full(
            'plate_1', 'plate_1_main', self.bodies, self.geoms, 4)
        self.assertTrue(ok)
        self.assertEqual(detail['total_geoms'], 5)
        self.assertEqual(detail['matched_geoms'], 5)
        self.assertEqual(detail['mismatched_geoms'], 0)

    def test_mismatched_geom_detected(self):
        geoms2 = dict(self.geoms)
        geoms2['plate_1_g3'] = {'id': 999, 'body_id': 999}  # wrong body
        ok, detail = _verify_alias_hierarchy_full(
            'plate_1', 'plate_1_main', self.bodies, geoms2, 4)
        self.assertFalse(ok)
        self.assertGreater(detail['mismatched_geoms'], 0)

    def test_non_unique_bodies_detected(self):
        bodies2 = dict(self.bodies)
        bodies2['plate_1_extra'] = {'id': 99}
        ok, detail = _verify_alias_hierarchy_full(
            'plate_1', 'plate_1_main', bodies2, self.geoms, 4)
        self.assertFalse(ok)
        self.assertIn('non_unique_bodies', detail.get('error', ''))


class TestResolutionEnum(unittest.TestCase):
    def test_valid_not_include_blocked(self):
        self.assertNotIn('STRIP_SUFFIX_BODY', VALID_RESOLUTIONS)
        self.assertNotIn('SUBSTRING', VALID_RESOLUTIONS)

    def test_blocked_set_correct(self):
        self.assertIn('STRIP_SUFFIX_BODY', BLOCKED_RESOLUTIONS)
        self.assertIn('STRIP_SUFFIX_SITE', BLOCKED_RESOLUTIONS)
        self.assertIn('SUBSTRING', BLOCKED_RESOLUTIONS)

    def test_valid_plus_terminal_covers_all(self):
        all_resolutions = VALID_RESOLUTIONS | {'UNRESOLVED', 'AMBIGUOUS',
                                                'BLOCKED_REGION_AS_BODY',
                                                'BLOCKED_REGION_AS_GEOM',
                                                'ENV_ERROR'}
        self.assertGreater(len(all_resolutions), 5)


if __name__ == '__main__':
    unittest.main()
