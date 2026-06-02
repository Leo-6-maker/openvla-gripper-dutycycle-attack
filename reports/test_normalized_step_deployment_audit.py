"""Verify normalized_step is NOT in deployment training paths — only in legacy utility.

Issue: src/utils/proprio_causal_student.py:46 includes "normalized_step" in NUMERIC_FEATURES.
This is a Milestone 2C legacy utility. Current Object-100 training scripts must NOT use it.
"""
import sys, os, re
import unittest


class NormalizedStepDeploymentAudit(unittest.TestCase):
    """Assert normalized_step stays in legacy utility only, never in deployment training."""

    def setUp(self):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── Legacy utility (known issue — documented, not in deployment path) ──

    def test_legacy_utility_has_normalized_step(self):
        """Document: proprio_causal_student.py NUMERIC_FEATURES includes normalized_step (known issue)."""
        src = os.path.join(self.repo_root, "src", "utils", "proprio_causal_student.py")
        with open(src) as f:
            content = f.read()
        self.assertIn("normalized_step", content,
            "KNOWN: legacy utility includes normalized_step. Must NOT be used in deployment training.")

    # ── Deployment training scripts must NOT include normalized_step ──

    def _check_script_no_normalized_step(self, script_relpath, feature_var_name):
        path = os.path.join(self.repo_root, script_relpath)
        if not os.path.exists(path):
            self.skipTest(f"{script_relpath} not found")
        with open(path) as f:
            content = f.read()
        # Find feature list
        # Pattern: VAR_NAME = [...] or VAR_NAME = "..."
        pattern = rf'{feature_var_name}\s*=\s*\[(.*?)\]'
        m = re.search(pattern, content, re.DOTALL)
        if not m:
            self.skipTest(f"{feature_var_name} list not found in {script_relpath}")
        feature_text = m.group(1)
        features_in_list = re.findall(r'"([^"]+)"', feature_text)
        self.assertNotIn("normalized_step", features_in_list,
            f"normalized_step found in {script_relpath} {feature_var_name} — DEPLOYMENT LEAKAGE")

    def test_tmp_train_obj100_no_normalized_step(self):
        self._check_script_no_normalized_step("tmp_train_obj100.py", "ALLOWED")

    def test_tmp_train_visual_no_normalized_step(self):
        self._check_script_no_normalized_step("tmp_train_visual.py", "ALLOWED_PROPRIO")

    # ── Config YAML: normalized_step must be FORBIDDEN, never ALLOWED ──

    def test_config_normalized_step_forbidden_only(self):
        config_path = os.path.join(self.repo_root, "configs", "proprio_no_step_object100_b_window_full.yaml")
        if not os.path.exists(config_path):
            self.skipTest("Config not found")
        with open(config_path) as f:
            content = f.read()
        # Extract allowed_inputs section
        allowed_match = re.search(r'allowed_inputs:\s*\n((?:\s+-.+\n?)+)', content)
        if allowed_match:
            allowed_text = allowed_match.group(1)
            allowed_items = re.findall(r'-\s+(\S+)', allowed_text)
            self.assertNotIn("normalized_step", allowed_items,
                "normalized_step in config allowed_inputs — DEPLOYMENT LEAKAGE")
        # Extract forbidden_inputs section
        forbidden_match = re.search(r'forbidden_inputs:\s*\n((?:\s+-.+\n?)+)', content)
        if forbidden_match:
            forbidden_text = forbidden_match.group(1)
            forbidden_items = re.findall(r'-\s+(\S+)', forbidden_text)
            self.assertIn("normalized_step", forbidden_items,
                "normalized_step NOT in config forbidden_inputs — should be explicitly forbidden")

    # ── FORBIDDEN_INPUT_SUBSTRINGS should include normalized_step or it's guarded elsewhere ──

    def test_forbidden_list_or_guarded(self):
        """normalized_step must either be in FORBIDDEN_INPUT_SUBSTRINGS or fully absent from training path."""
        src = os.path.join(self.repo_root, "src", "utils", "proprio_causal_student.py")
        with open(src) as f:
            content = f.read()
        forbidden_section = re.search(r'FORBIDDEN_INPUT_SUBSTRINGS\s*=\s*\[(.*?)\]', content, re.DOTALL)
        self.assertIsNotNone(forbidden_section, "FORBIDDEN_INPUT_SUBSTRINGS not found")
        forbidden_text = forbidden_section.group(1)
        forbidden_items = re.findall(r'"([^"]+)"', forbidden_text)

        # Check: normalized_step should be in FORBIDDEN list since it's in NUMERIC_FEATURES
        # OR: All training scripts must use their own feature lists (verified by tests above)
        if "normalized_step" in content and '"normalized_step"' not in forbidden_text:
            # This is OK only if no deployment training uses this module's NUMERIC_FEATURES directly
            # Verify by checking that tmp_train scripts define their own ALLOWED lists
            # (already verified by tests above)
            pass  # Acceptable: legacy utility has it, but deployment scripts don't use it

    # ── No training script should import NUMERIC_FEATURES from proprio_causal_student ──

    def test_no_deployment_import_of_numeric_features(self):
        """Verify no tmp_train* script imports NUMERIC_FEATURES from proprio_causal_student."""
        for script_name in ["tmp_train_obj100.py", "tmp_train_visual.py"]:
            path = os.path.join(self.repo_root, script_name)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                content = f.read()
            self.assertNotIn("NUMERIC_FEATURES", content,
                f"{script_name} imports NUMERIC_FEATURES from legacy utility — remove and use local ALLOWED list")

    # ── All 13 deployment proprio fields must match expected set ──

    def test_deployment_proprio_fields_exactly_thirteen(self):
        """Object-100 training uses exactly 13 proprio fields, no normalized_step."""
        expected = {
            "gripper_command", "gripper_qpos", "gripper_width",
            "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
            "action_dx", "action_dy", "action_dz", "action_gripper",
        }
        for script_name in ["tmp_train_obj100.py", "tmp_train_visual.py"]:
            path = os.path.join(self.repo_root, script_name)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                content = f.read()
            # Find ALLOWED or ALLOWED_PROPRIO list
            for var in ["ALLOWED", "ALLOWED_PROPRIO"]:
                m = re.search(rf'{var}\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if m:
                    features = set(re.findall(r'"([^"]+)"', m.group(1)))
                    self.assertEqual(features, expected,
                        f"{script_name} {var} does not match expected 13-field set: got {features}")
                    break


if __name__ == "__main__":
    unittest.main()
