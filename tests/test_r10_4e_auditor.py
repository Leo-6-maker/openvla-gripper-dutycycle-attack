"""End-to-end CPU tests for the E-R3a sealed evidence auditor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r10_4"))

from audit_r10_4e_sealed_roots import main as audit_main  # noqa: E402
from run_r10_4e_passive_panel import (  # noqa: E402
    TASK00,
    TASK01,
    seal_root,
    sha256_file,
    write_json,
    write_jsonl,
    write_ledger_revision,
)

HEAD = "a" * 40
RECEIPT_SHA = "b" * 64
COMMENT_ID = 501


def _build_valid_tree(tmp_path: Path) -> tuple[Path, Path]:
    external = tmp_path / "external_task00"
    external.mkdir()
    write_json(
        external / "episode_summary.json",
        {
            "identity": TASK00,
            "status": "PASS_RUNTIME_NO_EMIT",
            "n_steps": 520,
            "emit_count": 0,
        },
    )
    external_seal = seal_root(external)
    external_summary_sha = sha256_file(external / "episode_summary.json")

    panel = tmp_path / "panel"
    panel.mkdir()

    reuse = panel / TASK00.replace("/", "_")
    reuse.mkdir()
    write_json(
        reuse / "REUSE_BINDING.json",
        {
            "schema": "R10_4E_TASK00_REUSE_BINDING_V1",
            "identity": TASK00,
            "external_root": str(external.resolve()),
            "external_sha256sums_sha256": external_seal["sha256sums_sha256"],
            "external_summary_sha256": external_summary_sha,
            "original_status": "PASS_RUNTIME_NO_EMIT",
            "n_steps": 520,
            "emit_count": 0,
        },
    )
    seal_root(reuse)

    fresh = panel / TASK01.replace("/", "_")
    fresh.mkdir()
    steps = [
        {
            "step": index,
            "generation_passes_per_step": 1,
            "clean_env_action_7d": [0.0] * 7,
            "executed_action_7d": [0.0] * 7,
            "action_max_abs_error": 0.0,
            "features_25d": [0.0] * 25,
            "info": {},
        }
        for index in range(2)
    ]
    detectors = [{"step": index, "emit": False} for index in range(2)]
    privileged = [{"step": index, "detector_input": False} for index in range(2)]
    write_jsonl(fresh / "step_records.jsonl", steps)
    write_jsonl(fresh / "detector_records.jsonl", detectors)
    write_jsonl(fresh / "privileged_teacher_sidecar.jsonl", privileged)
    write_json(
        fresh / "episode_metadata.json",
        {
            "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_METADATA_V1",
            "identity": TASK01,
            "source_commit": HEAD,
            "parent": {"identity": TASK01},
            "panel_receipt_sha256": RECEIPT_SHA,
            "authorization_comment_id": COMMENT_ID,
            "action_mutation": False,
            "attack_enabled": False,
            "command_open_enabled": False,
            "visual_attack_enabled": False,
            "random_attack_enabled": False,
            "privileged_runtime_input": False,
        },
    )
    write_json(
        fresh / "episode_summary.json",
        {
            "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1",
            "identity": TASK01,
            "status": "PASS_RUNTIME_NO_EMIT",
            "n_steps": 2,
            "emit_count": 0,
            "termination_reason": "FULL_LOOP_TASK_FAILURE",
            "task_success": False,
            "violations": [],
            "action_mutation": False,
            "privileged_runtime_input": False,
        },
    )
    write_json(
        fresh / "runtime_audit.json",
        {
            "runtime_valid": True,
            "status": "PASS_RUNTIME_NO_EMIT",
            "termination_reason": "FULL_LOOP_TASK_FAILURE",
            "action_mutation": False,
            "privileged_runtime_input": False,
        },
    )
    write_json(
        fresh / "ROOT_SEAL_RECEIPT.json",
        {
            "schema": "R10_4E_ROOT_SEAL_RECEIPT_V1",
            "identity": TASK01,
            "source_commit": HEAD,
            "panel_receipt_sha256": RECEIPT_SHA,
        },
    )
    fresh_seal = seal_root(fresh)

    attempts0 = [
        {
            "identity": TASK00,
            "status": "PASS_RUNTIME_NO_EMIT",
            "ledger_status": "REUSE_VERIFIED",
            "reuse": True,
            "external_root": str(external.resolve()),
            "sha256sums_sha256": external_seal["sha256sums_sha256"],
        }
    ]
    rev0_sha = write_ledger_revision(panel, attempts0, 0, None)
    attempts1 = attempts0 + [
        {
            "identity": TASK01,
            "status": "PASS_RUNTIME_NO_EMIT",
            "ledger_status": "SEALED_PASS",
            "n_steps": 2,
            "emit_count": 0,
            "termination_reason": "FULL_LOOP_TASK_FAILURE",
            "task_success": False,
            "violations": [],
            "sha256sums_sha256": fresh_seal["sha256sums_sha256"],
            "reuse": False,
        }
    ]
    rev1_sha = write_ledger_revision(panel, attempts1, 1, rev0_sha)
    write_json(
        panel / "panel_ledger.json",
        {
            "schema": "R10_4E_PANEL_LEDGER_V1",
            "revision": 1,
            "previous_ledger_sha256": rev1_sha,
            "attempts": attempts1,
            "n_attempts": 2,
            "all_runtime_valid": True,
            "panel_ok": True,
        },
    )
    write_json(
        panel / "panel_summary.json",
        {
            "panel": "R10_4E",
            "phase": "E_R3A_TASK01_CANARY",
            "source_commit": HEAD,
            "panel_receipt_sha256": RECEIPT_SHA,
            "authorization_comment_id": COMMENT_ID,
            "n_tasks_attempted": 2,
            "n_reuse": 1,
            "n_fresh": 1,
            "all_runtime_valid": True,
            "panel_ok": True,
            "per_task": [
                {"identity": TASK00, "status": "PASS_RUNTIME_NO_EMIT", "reuse": True},
                {"identity": TASK01, "status": "PASS_RUNTIME_NO_EMIT", "reuse": False},
            ],
        },
    )
    seal_root(panel)
    return panel, external


def _run_audit(monkeypatch, panel: Path, external: Path, output: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--panel-root",
            str(panel),
            "--task00-root",
            str(external),
            "--expected-head",
            HEAD,
            "--expected-authorization-comment-id",
            str(COMMENT_ID),
            "--expected-receipt-sha256",
            RECEIPT_SHA,
            "--output",
            str(output),
        ],
    )
    return audit_main()


def test_complete_e_r3a_tree_passes(monkeypatch, tmp_path: Path) -> None:
    panel, external = _build_valid_tree(tmp_path)
    output = tmp_path / "audit.json"
    assert _run_audit(monkeypatch, panel, external, output) == 0
    assert json.loads(output.read_text())["overall"] == "PASS"


def test_tampered_episode_fails_aggregate_and_fresh_audit(monkeypatch, tmp_path: Path) -> None:
    panel, external = _build_valid_tree(tmp_path)
    summary = panel / TASK01.replace("/", "_") / "episode_summary.json"
    summary.write_text(summary.read_text() + "\n", encoding="utf-8")
    output = tmp_path / "audit_tampered.json"
    assert _run_audit(monkeypatch, panel, external, output) == 1
    assert json.loads(output.read_text())["overall"] == "FAIL"


def test_broken_ledger_chain_fails(monkeypatch, tmp_path: Path) -> None:
    panel, external = _build_valid_tree(tmp_path)
    revision = panel / "panel_ledger_rev0001.json"
    value = json.loads(revision.read_text())
    value["previous_ledger_sha256"] = "0" * 64
    revision.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "audit_ledger.json"
    assert _run_audit(monkeypatch, panel, external, output) == 1


def test_audit_output_inside_panel_is_rejected(monkeypatch, tmp_path: Path) -> None:
    panel, external = _build_valid_tree(tmp_path)
    with pytest.raises(SystemExit, match="OUTSIDE_PANEL_ROOT"):
        _run_audit(monkeypatch, panel, external, panel / "audit.json")
