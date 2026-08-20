from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gripper_attack.failure_evidence import build_failure_evidence, write_failure_receipt


ROOT = Path(__file__).resolve().parents[2]


def _row(index: int, *, arm_equal: bool = False) -> dict:
    return {
        "candidate_index": index,
        "candidate_source": "delta0" if index == 0 else f"pgd_iteration_{index}",
        "direct_generated_token_ids": [1, 2, 3, 4, 5, 6 if arm_equal else 9, 20 if index == 5 else 10],
        "clean_arm_token_ids": [1, 2, 3, 4, 5, 6],
        "direct_generated_arm_token_ids": [1, 2, 3, 4, 5, 6 if arm_equal else 9],
        "arm_token_ids_equal": arm_equal,
        "arm_mismatch_dimensions": [] if arm_equal else [5],
        "direct_generated_gripper_token_id": 20 if index == 5 else 10,
        "direct_generated_gripper_is_native_open": index == 5,
        "clean_gripper_token_id": 10,
        "clean_gripper_is_native_open": False,
        "gripper_token_changed": index == 5,
        "processor_input_sha256": f"sha-{index}",
    }


def _diagnostics(*, selected_index=None) -> dict:
    return {
        "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
        "candidate_audit": [_row(index, arm_equal=index == 5) for index in range(6)],
        "selected_candidate_index": selected_index,
        "selected_candidate_source": f"pgd_iteration_{selected_index}" if selected_index is not None else None,
    }


class _Adapter:
    def __init__(self, diagnostics=None):
        self.last_attack_diagnostics = diagnostics


class _Attacker:
    def __init__(self, diagnostics=None):
        self.adapter = _Adapter(diagnostics)


class FailureEvidencePersistenceTests(unittest.TestCase):
    def test_e1_t1_persists_six_candidate_failure_without_fake_success(self):
        diagnostics = _diagnostics()
        exc = RuntimeError("STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE")
        exc.diagnostics = diagnostics
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm_receipt.json"
            receipt = write_failure_receipt(
                path,
                {"status": "HOLD_Q3R3_D_ENGINEERING_ARM", "counters": {"attacked_env_steps": 0}},
                exc,
                _Attacker(json.loads(json.dumps(diagnostics))),
            )
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["selector_error_message"], "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE")
        self.assertTrue(receipt["candidate_audit_complete"])
        self.assertEqual(receipt["observed_candidate_count"], 6)
        self.assertEqual([row["candidate_index"] for row in persisted["candidate_audit"]], list(range(6)))
        self.assertEqual([row["candidate_source"] for row in persisted["candidate_audit"]], ["delta0", *[f"pgd_iteration_{i}" for i in range(1, 6)]])
        self.assertTrue(all(len(row["direct_generated_token_ids"]) == 7 for row in persisted["candidate_audit"]))
        self.assertEqual(persisted["selected_candidate_index"], None)
        self.assertNotIn("attack_result_returned", persisted)

    def test_e1_t2_serialization_preserves_known_success_selection(self):
        diagnostics = _diagnostics(selected_index=5)
        exc = RuntimeError("synthetic selector stop")
        exc.diagnostics = diagnostics
        evidence = build_failure_evidence(exc, _Attacker(json.loads(json.dumps(diagnostics))))
        self.assertEqual(evidence["selected_candidate_index"], 5)
        self.assertEqual(evidence["selected_candidate_source"], "pgd_iteration_5")
        self.assertEqual(evidence["candidate_audit"][5]["candidate_source"], "pgd_iteration_5")

    def test_e1_t3_requires_diagnostics_source_parity(self):
        diagnostics = _diagnostics()
        exc = RuntimeError("selector")
        exc.diagnostics = diagnostics
        self.assertEqual(build_failure_evidence(exc, _Attacker(json.loads(json.dumps(diagnostics))))["diagnostics_consistency_status"], "PASS")
        mismatch = json.loads(json.dumps(diagnostics))
        mismatch["candidate_audit"][2]["arm_mismatch_dimensions"] = [0]
        evidence = build_failure_evidence(exc, _Attacker(mismatch))
        self.assertEqual(evidence["diagnostics_consistency_status"], "HOLD_DIAGNOSTICS_SOURCE_DISAGREEMENT")
        self.assertEqual(evidence["candidate_audit"], [])

    def test_e1_t4_early_failure_does_not_fabricate_candidates(self):
        evidence = build_failure_evidence(RuntimeError("D_INPUT_IDS_CHANGED"), _Attacker())
        self.assertFalse(evidence["candidate_audit_complete"])
        self.assertEqual(evidence["observed_candidate_count"], 0)
        self.assertEqual(evidence["candidate_audit"], [])
        self.assertEqual(evidence["diagnostics_consistency_status"], "NOT_AVAILABLE")

    def test_e1_t5_helper_has_no_forbidden_execution_surface(self):
        source = (ROOT / "src/gripper_attack/failure_evidence.py").read_text(encoding="utf-8").lower()
        for forbidden in ("torch", "cuda", "nvidia-smi", "env.step", "openvla", "backward"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
