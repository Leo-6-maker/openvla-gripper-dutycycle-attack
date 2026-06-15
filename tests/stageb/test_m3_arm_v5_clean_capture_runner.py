from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gripper_attack.m3_event_panel import select_two_states_per_task
from scripts.stageb.run_m3_arm_v5_clean_capture import (
    load_config,
    run_offline_select,
    select_events_from_clean_record_dir,
    selected_rows_have_exact_binding,
    state_pool_from_config,
    validate_attempt_ledger_policy,
    validate_output_dir_new,
    write_csv,
)


CONFIG = Path("configs/m3_arm_v5_clean_close_event_panel.yaml")
RUNNER = Path("scripts/stageb/run_m3_arm_v5_clean_capture.py")


def _record(step, token, invariant=True, *, task="alphabet_soup", state_id=9, artifact=True):
    row = {
        "task": task,
        "state_id": state_id,
        "step": step,
        "tokens": [1, 2, 3, 4, 5, 6, token],
        "gripper_token": token,
        "score_invariant": {"tie_aware_pass": invariant},
        "official_score_argmax_token_id": token,
    }
    if artifact:
        row.update(
            {
                "raw_image_path": f"{task}_s{state_id}/step_{step}/raw.npy",
                "raw_image_sha256": f"raw-{task}-{state_id}-{step}",
                "processed_tensor_path": f"{task}_s{state_id}/step_{step}/processor.pt",
                "processed_tensor_sha256": f"tensor-{task}-{state_id}-{step}",
                "prompt_token_ids": "[1,2,3]",
                "prompt_token_ids_sha256": f"prompt-{task}-{state_id}-{step}",
                "model_fingerprint": "model-fp",
                "model_checkpoint_sha256": "model-sha",
                "processor_config_sha256": "processor-sha",
                "preprocess_config_sha256": "preprocess-sha",
                "task_state_init_sha256": f"init-{task}-{state_id}",
                "clean_record_source_path": f"{task}_s{state_id}_clean_records.json",
                "clean_record_source_sha256": f"source-{task}-{state_id}",
                "runner_sha256": "runner-sha",
                "config_sha256": "config-sha",
                "commit": "commit-sha",
                "gpu_query": "2, GPU-test, test, 0 MiB, 1 MiB, 0 %, 30",
                "worktree_status": "CLEAN",
            }
        )
    return row


def _write_clean_records(path: Path, records: list[dict]):
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def _write_attempt_ledger(path: Path, records_dir: Path, *, captured_count: int | None = None):
    candidates = select_two_states_per_task()
    limit = len(candidates) if captured_count is None else captured_count
    rows = []
    for candidate in candidates:
        record_path = records_dir / f"{candidate.task}_s{candidate.state_id}_clean_records.json"
        rows.append(
            {
                "task": candidate.task,
                "state_id": candidate.state_id,
                "attempt_index": 0,
                "attempt_status": "CAPTURED" if len(rows) < limit else "NO_EVENT",
                "first_action_taken": "true" if len(rows) < limit else "false",
                "clean_records_path": record_path.name if len(rows) < limit else "",
                "clean_records_sha256": "" if len(rows) >= limit else __import__("hashlib").sha256(record_path.read_bytes()).hexdigest(),
                "failure_reason": "",
            }
        )
    write_csv(
        path,
        rows,
        ["task", "state_id", "attempt_index", "attempt_status", "first_action_taken", "clean_records_path", "clean_records_sha256", "failure_reason"],
    )


def _write_all_candidate_records(directory: Path, *, event_count: int | None = None):
    directory.mkdir(parents=True)
    candidates = select_two_states_per_task()
    limit = len(candidates) if event_count is None else event_count
    for candidate in candidates[:limit]:
        _write_clean_records(
            directory / f"{candidate.task}_s{candidate.state_id}_clean_records.json",
            [
                _record(0, 31744, task=candidate.task, state_id=candidate.state_id),
                _record(1, 31872, task=candidate.task, state_id=candidate.state_id),
            ],
        )


def test_v5_clean_capture_runner_does_not_import_attack_modules():
    text = RUNNER.read_text(encoding="utf-8")
    assert "OpenVLAVisualAttacker" not in text
    assert "TokenPrefixPGDAttacker" not in text
    assert "m3_controls" not in text
    assert "attack_adapter import" not in text


def test_output_dir_must_be_new_or_empty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="new or empty"):
        validate_output_dir_new(out)


def test_state_pool_rejects_replacement_or_duplicate_state():
    cfg = load_config(CONFIG)
    bad = dict(cfg)
    bad["task_state_pool"] = list(cfg["task_state_pool"])
    bad["task_state_pool"][1] = dict(bad["task_state_pool"][0])
    with pytest.raises(ValueError, match="duplicate frozen state"):
        state_pool_from_config(bad)


def test_state_pool_rejects_prior_development_state():
    cfg = load_config(CONFIG)
    bad = dict(cfg)
    bad["task_state_pool"] = list(cfg["task_state_pool"])
    bad["task_state_pool"][4] = {
        "task": "butter",
        "state_id": 2,
        "task_rank": 1,
        "state_hash": __import__("gripper_attack.m3_event_panel", fromlist=["v5_state_hash"]).v5_state_hash("butter", 2),
    }
    records_dir = Path("does-not-matter")
    with pytest.raises(ValueError, match="prior Layer3 development state"):
        select_events_from_clean_record_dir(cfg=bad, clean_records_dir=records_dir)


def test_attempt_ledger_allows_only_first_action_before_infra_retry():
    validate_attempt_ledger_policy(
        [
            {
                "task": "ketchup",
                "state_id": 41,
                "attempt_index": 0,
                "attempt_status": "FIRST_ACTION_BEFORE_INFRA_FAILURE",
                "first_action_taken": "false",
            },
            {
                "task": "ketchup",
                "state_id": 41,
                "attempt_index": 1,
                "attempt_status": "CAPTURED",
                "first_action_taken": "true",
            },
        ]
    )
    with pytest.raises(ValueError, match="retry not allowed"):
        validate_attempt_ledger_policy(
            [
                {
                    "task": "ketchup",
                    "state_id": 41,
                    "attempt_index": 0,
                    "attempt_status": "INFRA_FAILURE_AFTER_ACTION",
                    "first_action_taken": "true",
                },
                {
                    "task": "ketchup",
                    "state_id": 41,
                    "attempt_index": 1,
                    "attempt_status": "CAPTURED",
                    "first_action_taken": "true",
                },
            ]
        )


def test_attempt_ledger_requires_full_pool_coverage_and_sha_binding(tmp_path):
    records_dir = tmp_path / "records"
    _write_all_candidate_records(records_dir)
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir)
    cfg = load_config(CONFIG)
    validate_attempt_ledger_policy(
        __import__("csv").DictReader(ledger.open(encoding="utf-8", newline="")),
        pool=select_two_states_per_task(),
        clean_records_dir=records_dir,
    )

    rows = list(__import__("csv").DictReader(ledger.open(encoding="utf-8", newline="")))
    rows = rows[:-1]
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_attempt_ledger_policy(rows, pool=select_two_states_per_task(), clean_records_dir=records_dir)

    rows = list(__import__("csv").DictReader(ledger.open(encoding="utf-8", newline="")))
    rows[0]["clean_records_sha256"] = "bad"
    with pytest.raises(ValueError, match="sha mismatch"):
        validate_attempt_ledger_policy(rows, pool=select_two_states_per_task(), clean_records_dir=records_dir)
    with pytest.raises(ValueError, match="attempt indices"):
        validate_attempt_ledger_policy(
            [
                {"task": "ketchup", "state_id": 41, "attempt_index": 0},
                {"task": "ketchup", "state_id": 41, "attempt_index": 1},
                {"task": "ketchup", "state_id": 41, "attempt_index": 2},
            ]
        )


def test_offline_select_freezes_first_eight_by_hash(tmp_path):
    records_dir = tmp_path / "records"
    _write_all_candidate_records(records_dir)
    cfg = load_config(CONFIG)
    _rows, selected, status = select_events_from_clean_record_dir(cfg=cfg, clean_records_dir=records_dir)
    expected = sorted(select_two_states_per_task(), key=lambda c: c.state_hash)[:8]
    assert status == "V5_EVENT_PANEL_INPUTS_FROZEN"
    assert [(e.task, e.state_id) for e in selected] == [(c.task, c.state_id) for c in expected]


def test_offline_select_stops_on_insufficient_pool(tmp_path):
    records_dir = tmp_path / "records"
    _write_all_candidate_records(records_dir, event_count=7)
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir, captured_count=7)
    out = tmp_path / "out"
    args = SimpleNamespace(config=str(CONFIG), clean_records_dir=str(records_dir), output_dir=str(out), attempt_ledger=str(ledger))
    with pytest.raises(SystemExit, match="V5_CAPTURE_POOL_INSUFFICIENT"):
        run_offline_select(args)
    summary = json.loads((out / "m3_arm_v5_clean_capture_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "V5_CAPTURE_POOL_INSUFFICIENT"
    assert summary["selected_count"] == 7


def test_offline_select_writes_manifest_for_success(tmp_path):
    records_dir = tmp_path / "records"
    _write_all_candidate_records(records_dir)
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir)
    out = tmp_path / "out"
    args = SimpleNamespace(config=str(CONFIG), clean_records_dir=str(records_dir), output_dir=str(out), attempt_ledger=str(ledger))
    run_offline_select(args)
    assert (out / "m3_arm_v5_clean_capture_manifest.csv").exists()
    assert (out / "m3_arm_v5_clean_event_selection_all_states.csv").exists()
    assert (out / "m3_arm_v5_frozen_event_panel.csv").exists()
    assert (out / "m3_arm_v5_artifact_hash_manifest.csv").exists()


def test_offline_select_requires_attempt_ledger(tmp_path):
    records_dir = tmp_path / "records"
    _write_all_candidate_records(records_dir)
    args = SimpleNamespace(config=str(CONFIG), clean_records_dir=str(records_dir), output_dir=str(tmp_path / "out"), attempt_ledger="")
    with pytest.raises(SystemExit, match="attempt_ledger"):
        run_offline_select(args)


def test_selected_row_blank_exact_binding_fails():
    ok, reason = selected_rows_have_exact_binding(
        [
            {
                "task": "ketchup",
                "state_id": 41,
                "selected_step": 1,
                "raw_image_path": "",
            }
        ]
    )
    assert not ok
    assert "missing_exact_input_field" in reason


def test_config_attack_seed_is_not_legacy_seed85_or_86():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = int(cfg["selection"]["first_attack_seed"]["seed"])
    assert seed == 428198
    assert seed not in {85, 86}
