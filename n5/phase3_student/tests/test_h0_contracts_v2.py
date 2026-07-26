"""Independent H0.1 contract checks.

This is intentionally separate from ``test_h0_contracts.py``.  The older
suite records historical checks; this suite applies an explicit receipt
algorithm and fail-closed provenance rules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import unittest


TESTS = pathlib.Path(__file__).resolve()
N5_ROOT = TESTS.parents[2]
REPO_ROOT = TESTS.parents[3]
BASELINE = N5_ROOT / "phase3_student" / "h0_evidence_baseline"
H0_RECEIPT = BASELINE / "H0_RECEIPT.json"


def strict_json_load(path: pathlib.Path):
    """Parse JSON and reject duplicate keys at every object level."""

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicates)


def canonical_receipt_sha(value: dict) -> str:
    """The explicit H0.1 algorithm for new receipts."""

    payload = dict(value)
    payload.pop("self_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(key)
        result[key] = value
    return result


def _has_numpy():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _git_head():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


class TestStrictReceiptContract(unittest.TestCase):
    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"outer":{"x":1,"x":2}}', object_pairs_hook=lambda pairs: _unique(pairs))

    def test_new_receipt_algorithm_roundtrip(self):
        receipt = {"gate": "H0.1", "status": "HOLD", "nested": {"z": 2, "a": 1}}
        receipt["self_sha256"] = canonical_receipt_sha(receipt)
        self.assertEqual(receipt["self_sha256"], canonical_receipt_sha(receipt))

    def test_historical_h0_receipt_is_rejected_by_explicit_algorithm(self):
        receipt = strict_json_load(H0_RECEIPT)
        self.assertNotEqual(receipt.get("self_sha256"), canonical_receipt_sha(receipt))

    def test_historical_h0_bindings_are_not_current_file_bindings(self):
        receipt = strict_json_load(H0_RECEIPT)
        mismatches = []
        for name, binding in receipt.get("upstream_artifacts", {}).items():
            relative = binding.get("path")
            expected = binding.get("sha256")
            if not relative or not expected:
                continue
            target = REPO_ROOT / relative
            if not target.is_file() or file_sha(target) != expected:
                mismatches.append(name)
        self.assertTrue(mismatches, "stale H0 bindings must be detected, not silently accepted")

    def test_h0_receipt_does_not_bind_current_head(self):
        receipt = strict_json_load(H0_RECEIPT)
        self.assertNotEqual(receipt.get("source_commit"), _git_head())


class TestC2Ordering(unittest.TestCase):
    def test_replay_records_state_before_env_step(self):
        path = N5_ROOT / "phase3_student" / "t2cr0_canonical_replay.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        replay = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "replay_one")
        step_lines = [
            node.lineno
            for node in ast.walk(replay)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        compare_lines = [
            node.lineno
            for node in ast.walk(replay)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "parity_issues"
        ]
        self.assertTrue(step_lines)
        self.assertTrue(compare_lines)
        self.assertLess(min(compare_lines), min(step_lines))


class TestTeacherProvenance(unittest.TestCase):
    @unittest.skipUnless(_has_numpy(), "numpy is required for dynamic Teacher perturbation")
    def test_success_terminal_perturbation_does_not_change_physical_label(self):
        """This is expected to fail until success is evaluation-only."""
        import copy
        import sys

        sys.path.insert(0, str(N5_ROOT / "phase2_labels"))
        from v22_production_v2 import v22_to_label_v2

        snapshot = {
            "factors": {
                "grasp_state": {"grasp_known_mask": True, "grasp_established": True, "grasp_confidence": 1.0},
                "contact_state": {"contact_known_mask": True, "contact_score": 1.0, "contact_confidence": 1.0},
                "comotion_state": {"comotion_known_mask": True, "object_eef_comotion_score": 1.0, "comotion_confidence": 1.0},
                "lift_state": {"lift_known_mask": True, "lift_score": 1.0, "lift_confidence": 1.0},
                "instability_indicators": {
                    "slip_known_mask": True, "slip_detected": False,
                    "contact_loss_known_mask": True, "contact_loss_detected": False,
                    "pose_anomaly_known_mask": True, "pose_anomaly_detected": False,
                    "width_increase_known_mask": True, "unplanned_width_increase": False,
                },
                "terminal_state": {"terminal_known_mask": True, "task_success": False},
                "planned_release": {"planned_release_known_mask": False},
                "placement_state": {},
                "gripper_closing_state": {"gripper_closing_known_mask": True, "gripper_closing_detected": True},
            }
        }
        changed = copy.deepcopy(snapshot)
        changed["factors"]["terminal_state"]["task_success"] = True
        base = v22_to_label_v2(snapshot, 0)
        perturbed = v22_to_label_v2(changed, 0)
        self.assertEqual(
            base["physical_criticality"],
            perturbed["physical_criticality"],
            "success/terminal fields must not alter physical Teacher labels",
        )


class TestCausalPrefix(unittest.TestCase):
    @unittest.skipUnless(_has_torch(), "torch is required for numerical prefix parity")
    def test_rf32_and_dual_prefix_lengths(self):
        import sys
        import torch

        sys.path.insert(0, str(N5_ROOT / "phase3_student"))
        from n5_student_model import N5MultiHeadStudent

        torch.manual_seed(0)
        model = N5MultiHeadStudent(input_dim=51, hidden=8, short_rf=32, long_rf=128, dropout=0.0)
        model.eval()
        x = torch.randn(1, 129, 51)
        lengths = [1, 2, 8, 16, 31, 32, 33, 64, 78, 127, 128, 129]
        with torch.no_grad():
            full = model(x)
            for length in lengths:
                prefix = model(x[:, :length])
                for head in full:
                    self.assertTrue(
                        torch.allclose(prefix[head][:, -1], full[head][:, length - 1], atol=1e-5, rtol=1e-5),
                        f"causal prefix mismatch head={head} length={length}",
                    )


if __name__ == "__main__":
    unittest.main()
