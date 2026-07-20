"""Focused CPU tests for the E-R3a receipt builder contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r10_4"))

from build_r10_4e_panel_receipt import (  # noqa: E402
    E_R3A_MANIFEST,
    E_R3A_PHASE,
    build_task_manifest,
    canonical_json_sha,
)


def test_e_r3a_manifest_is_exact_and_ordered() -> None:
    manifest = build_task_manifest(E_R3A_PHASE)
    assert manifest == [
        {"identity": "libero_10/task_00/state_20", "reuse": True},
        {"identity": "libero_10/task_01/state_20", "reuse": False},
    ]
    assert len(manifest) == 2
    assert sum(1 for row in manifest if row["reuse"]) == 1
    assert sum(1 for row in manifest if not row["reuse"]) == 1


def test_e_r3a_manifest_excludes_all_later_tasks() -> None:
    identities = {row["identity"] for row in build_task_manifest(E_R3A_PHASE)}
    for index in range(2, 10):
        assert f"libero_10/task_{index:02d}/state_20" not in identities


def test_unsupported_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="UNSUPPORTED_PHASE"):
        build_task_manifest("E_R3B_TASK02_03")


def test_manifest_builder_returns_copy() -> None:
    first = build_task_manifest(E_R3A_PHASE)
    first[0]["reuse"] = False
    second = build_task_manifest(E_R3A_PHASE)
    assert second == E_R3A_MANIFEST
    assert second[0]["reuse"] is True


def test_manifest_digest_is_deterministic() -> None:
    first = canonical_json_sha(build_task_manifest(E_R3A_PHASE))
    second = canonical_json_sha(build_task_manifest(E_R3A_PHASE))
    assert first == second
    assert len(first) == 64
