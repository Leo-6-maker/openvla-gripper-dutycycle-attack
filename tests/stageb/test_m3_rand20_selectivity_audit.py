from __future__ import annotations

import pytest

from scripts.stageb.audit_m3_step78_rand20_official_candidates import (
    classify_rand20_selectivity,
    verify_candidate_hashes,
)


def test_rand20_selectivity_classifies_selective_match():
    rows = [
        {"official_gripper_token": 31744, "arm_prefix_match_count": 5},
        {"official_gripper_token": 31872, "arm_prefix_match_count": 6},
    ]
    assert classify_rand20_selectivity(rows, arm_gate_min_match_count=5) == "RANDOM_SELECTIVE_MATCH_EXISTS"


def test_rand20_selectivity_classifies_nonselective_match():
    rows = [
        {"official_gripper_token": 31744, "arm_prefix_match_count": 4},
        {"official_gripper_token": 31872, "arm_prefix_match_count": 6},
    ]
    assert classify_rand20_selectivity(rows, arm_gate_min_match_count=5) == "RANDOM_ONLY_NONSELECTIVE_MATCH"


def test_rand20_selectivity_classifies_no_match():
    rows = [
        {"official_gripper_token": 31872, "arm_prefix_match_count": 6},
        {"official_gripper_token": 31871, "arm_prefix_match_count": 5},
    ]
    assert classify_rand20_selectivity(rows, arm_gate_min_match_count=5) == "NO_RANDOM_MATCH"


def test_candidate_hash_verification_rejects_mismatch():
    frozen = [
        {
            "candidate_id": "0",
            "candidate_seed": "11",
            "delta_sha256": "aaa",
            "processor_input_sha256": "bbb",
        }
    ]
    reconstructed = [
        {
            "candidate_id": 0,
            "candidate_seed": 11,
            "delta_sha256": "changed",
            "processor_input_sha256": "bbb",
        }
    ]
    with pytest.raises(RuntimeError, match="candidate 0"):
        verify_candidate_hashes(frozen, reconstructed)
