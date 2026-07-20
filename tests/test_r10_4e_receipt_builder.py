"""Focused CPU tests for the E-R3a receipt and runner authorization contract."""

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
from run_r10_4e_passive_panel import (  # noqa: E402
    EXPECTED_MANIFEST,
    PHASE,
    validate_receipt,
)


def test_e_r3a_manifest_is_exact_and_ordered() -> None:
    manifest = build_task_manifest(E_R3A_PHASE)
    assert manifest == [
        {"identity": "libero_10/task_00/state_20", "reuse": True},
        {"identity": "libero_10/task_01/state_20", "reuse": False},
    ]
    assert manifest == EXPECTED_MANIFEST
    assert PHASE == E_R3A_PHASE
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


def _receipt(task00_root: Path, **overrides):
    value = {
        "schema": "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1",
        "scope": "R10_4E_E_R3A_TASK01_CANARY",
        "phase": E_R3A_PHASE,
        "source_commit": "a" * 40,
        "authorization_comment_id": 501,
        "episodes_authorized": 2,
        "fresh_executions_authorized": 1,
        "reuse_authorized": 1,
        "task_manifest": build_task_manifest(E_R3A_PHASE),
        "task_manifest_sha256": canonical_json_sha(build_task_manifest(E_R3A_PHASE)),
        "task00_root": str(task00_root.resolve()),
        "task00_root_sha256s": "b" * 64,
        "task00_summary_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "detector_checkpoint_sha256": "e" * 64,
        "bundle_sha256s_sha256": "f" * 64,
        "model_tree_sha256": "1" * 64,
        "model_file_count": 7,
        "model_bytes": 12345,
        "feature_order_sha256": "2" * 64,
        "gpu": 0,
        "render_gpu": 0,
        "passive_only": True,
        "model_load_authorized": True,
        "detector_execution_authorized": True,
        "action_mutation_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "command_open_authorized": False,
        "visual_attack_authorized": False,
        "random_attack_authorized": False,
        "retry_authorized": False,
        "parent_substitution_authorized": False,
        "threshold_or_fsm_change_authorized": False,
        "output_overwrite_authorized": False,
    }
    value.update(overrides)
    return value


def _validate(receipt, task00_root: Path) -> None:
    validate_receipt(
        receipt,
        head="a" * 40,
        expected_comment_id=501,
        protocol_sha="d" * 64,
        task00_root=task00_root,
        task00_seal_sha="b" * 64,
        task00_summary_sha="c" * 64,
        model_tree_sha="1" * 64,
        model_file_count=7,
        model_bytes=12345,
        detector_checkpoint_sha="e" * 64,
        detector_bundle_sha="f" * 64,
        feature_order_sha="2" * 64,
        gpu=0,
        render_gpu=0,
    )


def test_runner_accepts_only_exact_e_r3a_receipt(tmp_path: Path) -> None:
    _validate(_receipt(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "override, expected_error",
    [
        ({"authorization_comment_id": 502}, "authorization_comment_id"),
        ({"phase": "E_R3B_TASK02_03"}, "phase"),
        ({"scope": "R10_4E_TEN_TASK_PASSIVE_PANEL"}, "scope"),
        ({"fresh_executions_authorized": 9}, "fresh_executions_authorized"),
        ({"episodes_authorized": 10}, "episodes_authorized"),
        ({"action_mutation_authorized": True}, "action_mutation_authorized"),
    ],
)
def test_runner_rejects_receipt_scope_expansion(tmp_path: Path, override, expected_error) -> None:
    with pytest.raises(SystemExit, match=expected_error):
        _validate(_receipt(tmp_path, **override), tmp_path)


def test_runner_rejects_task02_manifest_expansion(tmp_path: Path) -> None:
    expanded = build_task_manifest(E_R3A_PHASE) + [
        {"identity": "libero_10/task_02/state_20", "reuse": False}
    ]
    receipt = _receipt(
        tmp_path,
        task_manifest=expanded,
        task_manifest_sha256=canonical_json_sha(expanded),
    )
    with pytest.raises(SystemExit, match="task_manifest"):
        _validate(receipt, tmp_path)


def test_runner_rejects_task00_root_substitution(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="task00_root"):
        _validate(_receipt(other), tmp_path)
