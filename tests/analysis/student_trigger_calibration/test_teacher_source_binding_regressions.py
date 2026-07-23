"""Focused regressions for authoritative Teacher source binding.

These tests close the two remaining provenance edges needed before Codex runs
Phase B2 materialization on the server:
  1. every row of one canonical identity must bind to one source artifact SHA;
  2. the declared source episode step count must exactly match the sealed rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from validate_factorized_identity_disjointness import check_source_sha_validity


def test_authoritative_source_binding_accepts_exact_episode_closure():
    rows = [
        {
            "canonical_parent_key": "ep",
            "step": step,
            "source_artifact_recursive_sha256": "a" * 64,
            "source_episode_step_count": 2,
        }
        for step in range(2)
    ]
    errors = []
    check_source_sha_validity(
        rows,
        "CALIBRATION",
        "o0_i0",
        errors,
        require_source_step_count=True,
    )
    assert errors == []


def test_authoritative_source_binding_rejects_multiple_source_hashes():
    rows = [
        {
            "canonical_parent_key": "ep",
            "step": 0,
            "source_artifact_recursive_sha256": "a" * 64,
            "source_episode_step_count": 2,
        },
        {
            "canonical_parent_key": "ep",
            "step": 1,
            "source_artifact_recursive_sha256": "b" * 64,
            "source_episode_step_count": 2,
        },
    ]
    errors = []
    check_source_sha_validity(
        rows,
        "POLICY",
        "o0_i0",
        errors,
        require_source_step_count=True,
    )
    assert any("SOURCE_SHA_MULTIPLE" in error for error in errors)


def test_authoritative_source_binding_rejects_bad_step_count():
    for declared in (3, True, None, "2"):
        rows = [
            {
                "canonical_parent_key": "ep",
                "step": step,
                "source_artifact_recursive_sha256": "a" * 64,
                "source_episode_step_count": declared,
            }
            for step in range(2)
        ]
        errors = []
        check_source_sha_validity(
            rows,
            "HELDOUT",
            "o0_i0",
            errors,
            require_source_step_count=True,
        )
        assert any("SOURCE_STEP_COUNT_MISMATCH" in error for error in errors), declared
