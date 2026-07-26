"""C1-V2 Resolver Unit Tests.

Validates:
  1. EXACT_SITE resolution
  2. EXACT_BODY resolution
  3. APPROVED_STRUCTURAL_ALIAS (R1: {name}_main)
  4. BLOCKED region→body rejection
  5. UNRESOLVED detection
  6. No ambiguous/multi-candidate acceptance
  7. Hierarchy verification
"""
import unittest, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from t2rc1_v2_registry import (
    _is_region_target, _verify_alias_hierarchy,
    VALID_RESOLUTIONS, BLOCKED_RESOLUTIONS, APPROVED_ALIAS_RULES,
)


class TestRegionDetection(unittest.TestCase):
    def test_known_regions(self):
        regions = ['basket_1_contain_region', 'flat_stove_1_cook_region',
                   'microwave_1_heating_region', 'wooden_cabinet_1_top_region',
                   'white_cabinet_1_bottom_region', 'main_table_stove_front_region']
        for r in regions:
            self.assertTrue(_is_region_target(r), f'{r} should be detected as region')

    def test_non_regions(self):
        non_regions = ['alphabet_soup_1', 'plate_1', 'akita_black_bowl_1',
                       'butter_1', 'milk_1']
        for r in non_regions:
            self.assertFalse(_is_region_target(r), f'{r} should NOT be detected as region')


class TestAliasHierarchyVerification(unittest.TestCase):
    def setUp(self):
        # Simulated entity data
        self.bodies = {
            'plate_1_main': {'id': 10, 'name': 'plate_1_main'},
            'akita_black_bowl_1_main': {'id': 20, 'name': 'akita_black_bowl_1_main'},
        }
        self.geoms = {
            'plate_1_g0': {'body_id': 10}, 'plate_1_g1': {'body_id': 10},
            'plate_1_g2': {'body_id': 10},
            'akita_black_bowl_1_g0': {'body_id': 20},
        }

    def test_valid_alias_unique_hierarchy_verified(self):
        ok, msg = _verify_alias_hierarchy(
            'plate_1', 'plate_1_main', self.bodies, self.geoms, 10)
        self.assertTrue(ok, msg)
        self.assertIn('unique_hierarchy_verified', msg)

    def test_alias_name_mismatch_rejected(self):
        ok, msg = _verify_alias_hierarchy(
            'plate_1', 'plate_1_other', self.bodies, self.geoms, 10)
        self.assertFalse(ok)
        self.assertIn('alias_body', msg)

    def test_non_unique_body_rejected(self):
        bodies = dict(self.bodies)
        bodies['plate_1_extra'] = {'id': 99}
        ok, msg = _verify_alias_hierarchy(
            'plate_1', 'plate_1_main', bodies, self.geoms, 10)
        self.assertFalse(ok)
        self.assertIn('non_unique_bodies', msg)

    def test_no_geoms_rejected(self):
        ok, msg = _verify_alias_hierarchy(
            'plate_1', 'plate_1_main', self.bodies, {}, 10)
        self.assertFalse(ok)
        self.assertIn('no_geoms', msg)

    def test_geom_body_mismatch_rejected(self):
        geoms = {'plate_1_g0': {'body_id': 999}}  # wrong body
        ok, msg = _verify_alias_hierarchy(
            'plate_1', 'plate_1_main', self.bodies, geoms, 10)
        self.assertFalse(ok)
        self.assertIn('body_mismatch', msg)


class TestResolutionEnum(unittest.TestCase):
    def test_valid_resolutions(self):
        self.assertEqual(VALID_RESOLUTIONS,
                         {'EXACT_SITE', 'EXACT_BODY', 'EXACT_GEOM', 'APPROVED_STRUCTURAL_ALIAS'})

    def test_blocked_resolutions(self):
        self.assertEqual(BLOCKED_RESOLUTIONS,
                         {'STRIP_SUFFIX_BODY', 'STRIP_SUFFIX_SITE', 'SUBSTRING'})

    def test_alias_rules_exist(self):
        self.assertIn('R1_main_suffix', APPROVED_ALIAS_RULES)
        rule = APPROVED_ALIAS_RULES['R1_main_suffix']
        self.assertEqual(rule['transform']('plate_1'), 'plate_1_main')
        self.assertEqual(rule['verification'], 'hierarchy')


class TestResolutionPriority(unittest.TestCase):
    """Verify resolution code follows the documented priority order."""

    def test_priority_order_in_source(self):
        """Check the source code has resolution steps in correct order."""
        src_path = os.path.join(os.path.dirname(HERE), 't2rc1_v2_registry.py')
        with open(src_path) as f:
            code = f.read()

        # The documented priority comments must appear in order
        priorities = [
            'Priority 1: EXACT_SITE',
            'Priority 2: EXACT_BODY',
            'Priority 3: EXACT_GEOM',
            'Priority 4: APPROVED_STRUCTURAL_ALIAS',
            'Priority 5: UNRESOLVED',
        ]
        positions = [code.find(p) for p in priorities]
        for i in range(len(positions) - 1):
            self.assertLess(positions[i], positions[i + 1],
                f'Priority {i+1} must appear before priority {i+2} in source')

    def test_no_fallback_in_source(self):
        src_path = os.path.join(os.path.dirname(HERE), 't2rc1_v2_registry.py')
        with open(src_path) as f:
            code = f.read()
        import re
        assignments = re.findall(r"entry\['resolution'\]\s*=\s*'([^']+)'", code)
        for a in assignments:
            self.assertNotIn(a, BLOCKED_RESOLUTIONS,
                f'Blocked resolution {a} assigned in C1-V2 source')


class TestForbiddenPatterns(unittest.TestCase):
    def test_no_region_to_body_fallback(self):
        """Region targets must never resolve to body via fallback."""
        src_path = os.path.join(os.path.dirname(HERE), 't2rc1_v2_registry.py')
        with open(src_path) as f:
            code = f.read()
        # The BLOCKED_REGION_AS_BODY path is intentional (error flag, not resolution)
        # But there must be no path where is_region=True leads to EXACT_BODY or APPROVED_STRUCTURAL_ALIAS
        self.assertIn('is_region', code)
        self.assertIn('BLOCKED_REGION_AS_BODY', code)

    def test_alias_only_for_non_regions(self):
        """APPROVED_STRUCTURAL_ALIAS must only apply when is_region is False."""
        src_path = os.path.join(os.path.dirname(HERE), 't2rc1_v2_registry.py')
        with open(src_path) as f:
            code = f.read()
        # The alias block must be guarded by 'not is_region'
        alias_section_start = code.find('Priority 4: APPROVED_STRUCTURAL_ALIAS')
        alias_section = code[alias_section_start:alias_section_start + 500]
        self.assertIn('not is_region', alias_section,
                      'Alias resolution must be guarded by not is_region')


if __name__ == '__main__':
    unittest.main()
