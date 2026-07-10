import unittest
from collections import OrderedDict

from gripper_attack.c2f_siglip_detector_runtime import CANONICAL_25D_FEATURES
from scripts.stageb.collect_c2g_clean_window_rollouts_strict import canonicalize_stream_result


class StrictCollectorOrderingTests(unittest.TestCase):
    def test_canonicalizes_arbitrary_mapping_order(self):
        reversed_names = list(reversed(CANONICAL_25D_FEATURES))
        result = {
            "features": {name: float(index) for index, name in enumerate(reversed_names)},
            "ready": True,
        }
        canonical = canonicalize_stream_result(result)
        self.assertIsInstance(canonical["features"], OrderedDict)
        self.assertEqual(list(canonical["features"]), list(CANONICAL_25D_FEATURES))
        self.assertEqual(canonical["feature_names"], list(CANONICAL_25D_FEATURES))
        self.assertTrue(canonical["ready"])

    def test_missing_feature_fails_closed(self):
        features = {name: 0.0 for name in CANONICAL_25D_FEATURES[:-1]}
        with self.assertRaisesRegex(ValueError, "missing"):
            canonicalize_stream_result({"features": features})

    def test_unexpected_feature_fails_closed(self):
        features = {name: 0.0 for name in CANONICAL_25D_FEATURES}
        features["task_index"] = 1.0
        with self.assertRaisesRegex(ValueError, "unexpected"):
            canonicalize_stream_result({"features": features})

    def test_nonfinite_value_is_not_silently_reordered(self):
        features = {name: 0.0 for name in CANONICAL_25D_FEATURES}
        features[CANONICAL_25D_FEATURES[0]] = float("nan")
        canonical = canonicalize_stream_result({"features": features})
        self.assertNotEqual(canonical["features"][CANONICAL_25D_FEATURES[0]], 0.0)
        # The collector's existing finite-value gate rejects this after ordering.


if __name__ == "__main__":
    unittest.main()
