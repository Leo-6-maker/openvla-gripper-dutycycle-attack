"""H0.3-R6 role-safe resolver contract tests.

These tests exercise semantic role assignment, production relation resolution,
ancestor-chain alias verification, counter aggregation, and fail-closed cases
without requiring MuJoCo.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from t2rc1_v2_registry import (
    MANIPULATED_OBJECT,
    OBJECT_TARGET,
    REGION_TARGET,
    VALID_RESOLUTIONS,
    _body_descends_from,
    _target_semantic_role,
    _verify_alias_hierarchy_full,
    resolve_entity,
    resolve_relation,
    summarize_relation_resolutions,
)


def _entities():
    bodies = {
        'alphabet_soup_1_main': {'id': 1, 'parent_id': 0},
        'butter_1_main': {'id': 2, 'parent_id': 0},
        'basket_1_main': {'id': 3, 'parent_id': 0},
        'plate_1_main': {'id': 4, 'parent_id': 0},
        'akita_black_bowl_1_main': {'id': 5, 'parent_id': 0},
        'nested_obj_main': {'id': 6, 'parent_id': 0},
        'nested_obj_child': {'id': 7, 'parent_id': 6},
    }
    sites = {
        'basket_1_contain_region': {
            'id': 10, 'body_id': 3, 'size': [0.06, 0.06, 0.07],
        },
        'flat_stove_1_cook_region': {
            'id': 11, 'body_id': 99, 'size': [0.075, 0.075, 0.0025],
        },
    }
    geoms = {}
    for name, body_id in [
        ('alphabet_soup_1', 1),
        ('butter_1', 2),
        ('plate_1', 4),
        ('akita_black_bowl_1', 5),
    ]:
        for index in range(5):
            geoms[f'{name}_g{index}'] = {
                'id': 1000 + body_id * 10 + index,
                'body_id': body_id,
            }
    geoms['nested_obj_g0'] = {'id': 2000, 'body_id': 7}
    return sites, bodies, geoms


class TestSemanticRole(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _entities()
        self.object_names = {
            'alphabet_soup_1', 'butter_1', 'plate_1',
            'akita_black_bowl_1', 'nested_obj',
        }

    def test_target_role_comes_from_bddl_not_sites(self):
        self.assertEqual(
            _target_semantic_role('basket_1_contain_region', self.object_names),
            REGION_TARGET,
        )
        self.assertEqual(
            _target_semantic_role('plate_1', self.object_names),
            OBJECT_TARGET,
        )

    def test_declared_region_exact_site(self):
        relation = resolve_relation(
            'In', 'alphabet_soup_1', 'basket_1_contain_region',
            self.object_names, self.sites, self.bodies, self.geoms,
        )
        self.assertTrue(relation['relation_ok'])
        self.assertEqual(relation['target_semantic_role'], REGION_TARGET)
        self.assertEqual(
            relation['target_resolution']['resolution'], 'EXACT_SITE')

    def test_missing_region_site_same_named_body_is_blocked(self):
        bodies = dict(self.bodies)
        bodies['missing_region'] = {'id': 40, 'parent_id': 0}
        relation = resolve_relation(
            'In', 'alphabet_soup_1', 'missing_region',
            self.object_names, self.sites, bodies, self.geoms,
        )
        self.assertFalse(relation['relation_ok'])
        self.assertEqual(
            relation['target_resolution']['resolution'],
            'BLOCKED_REGION_AS_BODY',
        )

    def test_missing_region_site_same_named_geom_is_blocked(self):
        geoms = dict(self.geoms)
        geoms['missing_region'] = {'id': 40, 'body_id': 1}
        relation = resolve_relation(
            'In', 'alphabet_soup_1', 'missing_region',
            self.object_names, self.sites, self.bodies, geoms,
        )
        self.assertFalse(relation['relation_ok'])
        self.assertEqual(
            relation['target_resolution']['resolution'],
            'BLOCKED_REGION_AS_GEOM',
        )

    def test_manipulated_object_site_is_blocked(self):
        sites = dict(self.sites)
        sites['site_only_object'] = {'id': 50, 'body_id': 3, 'size': [1, 1, 1]}
        object_names = set(self.object_names) | {'site_only_object'}
        relation = resolve_relation(
            'On', 'site_only_object', 'plate_1',
            object_names, sites, self.bodies, self.geoms,
        )
        self.assertFalse(relation['relation_ok'])
        self.assertEqual(
            relation['object_resolution']['resolution'],
            'BLOCKED_OBJECT_AS_SITE',
        )

    def test_object_target_site_is_blocked(self):
        sites = dict(self.sites)
        sites['site_only_target'] = {'id': 51, 'body_id': 3, 'size': [1, 1, 1]}
        object_names = set(self.object_names) | {'site_only_target'}
        relation = resolve_relation(
            'On', 'plate_1', 'site_only_target',
            object_names, sites, self.bodies, self.geoms,
        )
        self.assertFalse(relation['relation_ok'])
        self.assertEqual(
            relation['target_resolution']['resolution'],
            'BLOCKED_OBJECT_AS_SITE',
        )

    def test_object_not_declared_in_bddl_is_unresolved(self):
        relation = resolve_relation(
            'On', 'ghost_object', 'plate_1',
            self.object_names, self.sites, self.bodies, self.geoms,
        )
        self.assertFalse(relation['relation_ok'])
        self.assertEqual(
            relation['object_resolution']['resolution'], 'UNRESOLVED')


class TestRoleSafeResolution(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _entities()

    def test_exact_cross_type_conflict_is_ambiguous(self):
        sites = dict(self.sites)
        bodies = dict(self.bodies)
        sites['collision'] = {'id': 60, 'body_id': 3, 'size': [1, 1, 1]}
        bodies['collision'] = {'id': 60, 'parent_id': 0}
        result = resolve_entity(
            'collision', OBJECT_TARGET, sites, bodies, self.geoms)
        self.assertEqual(result['resolution'], 'AMBIGUOUS')

    def test_alias_cross_type_conflict_is_ambiguous(self):
        sites = dict(self.sites)
        sites['plate_1_main'] = {'id': 61, 'body_id': 3, 'size': [1, 1, 1]}
        result = resolve_entity(
            'plate_1', OBJECT_TARGET, sites, self.bodies, self.geoms)
        self.assertEqual(result['resolution'], 'AMBIGUOUS')

    def test_unknown_semantic_role_rejected(self):
        with self.assertRaises(ValueError):
            resolve_entity(
                'plate_1', 'UNKNOWN_ROLE',
                self.sites, self.bodies, self.geoms,
            )

    def test_unresolved_has_reason(self):
        result = resolve_entity(
            'does_not_exist', OBJECT_TARGET,
            self.sites, self.bodies, self.geoms,
        )
        self.assertEqual(result['resolution'], 'UNRESOLVED')
        self.assertIn('reason', result['error_detail'])

    def test_valid_resolutions_exclude_blocked(self):
        self.assertNotIn('BLOCKED_OBJECT_AS_SITE', VALID_RESOLUTIONS)
        self.assertNotIn('BLOCKED_REGION_AS_BODY', VALID_RESOLUTIONS)


class TestAliasAncestry(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _entities()

    def test_direct_alias(self):
        result = resolve_entity(
            'plate_1', MANIPULATED_OBJECT,
            self.sites, self.bodies, self.geoms,
        )
        self.assertEqual(
            result['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(result['alias_to'], 'plate_1_main')

    def test_nested_child_geom_is_accepted(self):
        result = resolve_entity(
            'nested_obj', MANIPULATED_OBJECT,
            self.sites, self.bodies, self.geoms,
        )
        self.assertEqual(
            result['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        verification = result['alias_verification']
        self.assertEqual(verification['direct_geoms'], 0)
        self.assertEqual(verification['descendant_geoms'], 1)

    def test_body_descendant_helper(self):
        self.assertTrue(_body_descends_from(7, 6, self.bodies))
        self.assertTrue(_body_descends_from(6, 6, self.bodies))
        self.assertFalse(_body_descends_from(7, 4, self.bodies))

    def test_geom_outside_ancestry_rejected(self):
        geoms = dict(self.geoms)
        geoms['plate_1_g3'] = {'id': 999, 'body_id': 7}
        result = resolve_entity(
            'plate_1', OBJECT_TARGET,
            self.sites, self.bodies, geoms,
        )
        self.assertEqual(result['resolution'], 'UNRESOLVED')
        self.assertEqual(
            result['error_detail']['reason'], 'alias_verification_failed')

    def test_all_matching_geoms_are_checked(self):
        ok, detail = _verify_alias_hierarchy_full(
            'plate_1', 'plate_1_main',
            self.bodies, self.geoms, 4,
        )
        self.assertTrue(ok)
        self.assertEqual(detail['total_geoms'], 5)
        self.assertEqual(detail['descendant_geoms'], 5)


class TestBlackBookAliasBeforeGeom(unittest.TestCase):
    """R5-D: black_book_1 must resolve to body_origin via APPROVED_STRUCTURAL_ALIAS,
    not geom_center via EXACT_GEOM."""

    def setUp(self):
        self.sites, self.bodies, self.geoms = _entities()

    def test_black_book_alias_wins_over_exact_geom(self):
        """When {name} is a geom AND {name}_main is a body, alias must win."""
        geoms = dict(self.geoms)
        geoms['black_book_1'] = {'id': 3000, 'body_id': 1}
        bodies = dict(self.bodies)
        bodies['black_book_1_main'] = {'id': 50, 'parent_id': 0}
        # Add matching geoms for hierarchy verification
        for idx in range(3):
            geoms[f'black_book_1_g{idx}'] = {'id': 3001 + idx, 'body_id': 50}

        result = resolve_entity(
            'black_book_1', MANIPULATED_OBJECT,
            self.sites, bodies, geoms,
        )
        self.assertEqual(result['resolution'], 'APPROVED_STRUCTURAL_ALIAS')
        self.assertEqual(result['entity_type'], 'body')
        self.assertEqual(result['alias_to'], 'black_book_1_main')
        self.assertTrue(result.get('black_book_applies'))
        self.assertIn('all_candidates', result)
        candidates = result['all_candidates']
        self.assertTrue(any(c['status'] == 'SELECTED' for c in candidates))
        self.assertTrue(any(c['status'] == 'SUPERSEDED_BY_STRUCTURAL_ALIAS' for c in candidates))

    def test_black_book_alias_verification_fails_closed(self):
        """Failed alias verification must return UNRESOLVED, not fallback to geom."""
        geoms = dict(self.geoms)
        geoms['black_book_1'] = {'id': 3000, 'body_id': 1}
        bodies = dict(self.bodies)
        bodies['black_book_1_main'] = {'id': 50, 'parent_id': 0}
        # NO matching geoms → hierarchy verification fails
        result = resolve_entity(
            'black_book_1', MANIPULATED_OBJECT,
            self.sites, bodies, geoms,
        )
        self.assertEqual(result['resolution'], 'UNRESOLVED')
        self.assertEqual(
            result['error_detail']['reason'], 'alias_verification_failed')

    def test_geom_without_alias_still_works(self):
        """When geom exists but no body alias, EXACT_GEOM is still valid."""
        geoms = dict(self.geoms)
        geoms['simple_geom_obj'] = {'id': 4000, 'body_id': 1}
        result = resolve_entity(
            'simple_geom_obj', MANIPULATED_OBJECT,
            self.sites, self.bodies, geoms,
        )
        self.assertEqual(result['resolution'], 'EXACT_GEOM')


class TestProductionAggregation(unittest.TestCase):
    def setUp(self):
        self.sites, self.bodies, self.geoms = _entities()
        self.object_names = {
            'alphabet_soup_1', 'butter_1', 'plate_1',
            'akita_black_bowl_1', 'nested_obj',
        }

    def test_ok_relation_counts_both_roles_and_ledgers(self):
        relation = resolve_relation(
            'On', 'plate_1', 'akita_black_bowl_1',
            self.object_names, self.sites, self.bodies, self.geoms,
        )
        counts, ledger = summarize_relation_resolutions(
            [relation], 'libero_test/task_00')
        self.assertEqual(counts['object_ok'], 1)
        self.assertEqual(counts['target_ok'], 1)
        self.assertEqual(counts['object_ambiguous'], 0)
        self.assertEqual(counts['target_ambiguous'], 0)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(
            {entry['entity_role'] for entry in ledger},
            {'object', 'target'},
        )

    def test_blocked_and_ambiguous_are_aggregated(self):
        sites = dict(self.sites)
        bodies = dict(self.bodies)
        sites['collision'] = {'id': 70, 'body_id': 3, 'size': [1, 1, 1]}
        bodies['collision'] = {'id': 70, 'parent_id': 0}
        object_names = set(self.object_names) | {'collision', 'site_only'}
        sites['site_only'] = {'id': 71, 'body_id': 3, 'size': [1, 1, 1]}
        ambiguous = resolve_relation(
            'On', 'collision', 'plate_1',
            object_names, sites, bodies, self.geoms,
        )
        blocked = resolve_relation(
            'On', 'site_only', 'plate_1',
            object_names, sites, bodies, self.geoms,
        )
        counts, ledger = summarize_relation_resolutions(
            [ambiguous, blocked], 'libero_test/task_01')
        self.assertEqual(counts['object_ambiguous'], 1)
        self.assertEqual(counts['object_blocked'], 1)
        self.assertEqual(counts['target_ok'], 2)
        self.assertEqual(len(ledger), 2)

    def test_unresolved_target_is_counted(self):
        relation = resolve_relation(
            'In', 'alphabet_soup_1', 'unknown_region',
            self.object_names, self.sites, self.bodies, self.geoms,
        )
        counts, _ = summarize_relation_resolutions(
            [relation], 'libero_test/task_02')
        self.assertEqual(counts['target_unresolved'], 1)
        self.assertFalse(relation['relation_ok'])


if __name__ == '__main__':
    unittest.main()
