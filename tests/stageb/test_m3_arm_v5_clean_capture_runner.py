from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from gripper_attack.m3_event_panel import select_two_states_per_task
from scripts.stageb.run_m3_arm_v5_clean_capture import (
    current_branch_value,
    gpu_compute_process_snapshot,
    model_bundle_manifest,
    load_config,
    prepare_generation_inputs,
    require_runtime_gates,
    run_offline_select,
    select_events_from_clean_record_dir,
    selected_rows_have_exact_binding,
    state_pool_from_config,
    validate_attempt_ledger_policy,
    validate_output_dir_new,
    verify_exact_input_binding,
    verify_model_bundle_manifest,
    verify_selected_rows_exact_bindings,
    write_csv,
)
from scripts.stageb.audit_m3_arm_v5_clean_capture import audit_capture_root


CONFIG = Path("configs/m3_arm_v5_clean_close_event_panel.yaml")
RUNNER = Path("scripts/stageb/run_m3_arm_v5_clean_capture.py")


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_artifact(root: Path, rel: str, content: str) -> tuple[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return rel, _sha_file(path)


def _sha_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _make_model_dir(tmp_path: Path) -> tuple[Path, list[dict], str]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"mock"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-00001.safetensors").write_text("weights", encoding="utf-8")
    rows, bundle_sha = model_bundle_manifest(model_dir)
    return model_dir, rows, bundle_sha


def _write_temp_config(tmp_path: Path, model_dir: Path) -> Path:
    cfg = load_config(CONFIG)
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["path"] = str(model_dir)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_model_manifest(records_dir: Path, rows: list[dict]) -> None:
    write_csv(records_dir / "m3_arm_v5_model_bundle_manifest.csv", rows, ["relative_path", "size_bytes", "sha256"])


def _write_capture_manifest(records_dir: Path) -> None:
    write_csv(
        records_dir / "m3_arm_v5_clean_capture_manifest.csv",
        [
            {
                "stage": "M3_ARM_V5_CLEAN_CAPTURE",
                "commit": _head(),
                "dirty_status": "CLEAN",
                "config_path": str(CONFIG),
                "config_sha256": _sha("config"),
                "runner_path": str(RUNNER),
                "runner_sha256": _sha("runner"),
                "model_fingerprint": "{}",
                "gpu_query": "2, GPU-test, test, 0 MiB, 1 MiB, 0 %, 30",
                "hostname": "test",
                "python": "test",
                "cuda_visible_devices": "2,6",
            }
        ],
        [
            "stage",
            "commit",
            "dirty_status",
            "config_path",
            "config_sha256",
            "runner_path",
            "runner_sha256",
            "model_fingerprint",
            "gpu_query",
            "hostname",
            "python",
            "cuda_visible_devices",
        ],
    )


def _write_marker(root: Path, rel: str, phase: str) -> None:
    path = root / rel / f"{phase}.marker"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(phase, encoding="utf-8")


def _write_capture_markers(root: Path, rel: str) -> None:
    for phase in ("ATTEMPT_STARTED", "MODEL_READY", "ENV_READY", "FIRST_ACTION_GENERATED", "FIRST_ACTION_TAKEN", "CAPTURE_COMPLETED"):
        _write_marker(root, rel, phase)


def _record(
    step,
    token,
    invariant=True,
    *,
    task="alphabet_soup",
    state_id=9,
    artifact=True,
    root: Path | None = None,
    model_sha: str | None = None,
):
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
        if root is None:
            raw_path = f"{task}_s{state_id}/step_{step}/raw.npy"
            tensor_path = f"{task}_s{state_id}/step_{step}/processor.pt"
            source_path = f"{task}_s{state_id}/step_{step}/clean_generation_source.json"
            raw_sha = _sha(raw_path)
            tensor_sha = _sha(tensor_path)
            source_sha = _sha(source_path)
        else:
            raw_path, raw_sha = _write_artifact(root, f"{task}_s{state_id}/step_{step}/raw.npy", f"raw-{task}-{state_id}-{step}")
            tensor_path, tensor_sha = _write_artifact(root, f"{task}_s{state_id}/step_{step}/processor.pt", f"tensor-{task}-{state_id}-{step}")
            source_payload = {"task": task, "state_id": state_id, "step": step}
            source_path = f"{task}_s{state_id}/step_{step}/clean_generation_source.json"
            source_file = root / source_path
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text(json.dumps(source_payload), encoding="utf-8")
            source_sha = _sha_file(source_file)
        prompt = "[1,2,3,29871]"
        row.update(
            {
                "raw_image_path": raw_path,
                "raw_image_sha256": raw_sha,
                "processed_tensor_path": tensor_path,
                "processed_tensor_sha256": tensor_sha,
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": _sha(prompt),
                "model_fingerprint": f'{{"model_bundle_sha256":"{model_sha or _sha("model")}","ok":true}}',
                "model_checkpoint_sha256": model_sha or _sha("model"),
                "processor_config_sha256": _sha("processor"),
                "preprocess_config_sha256": _sha("preprocess"),
                "task_state_init_sha256": _sha(f"init-{task}-{state_id}"),
                "clean_record_source_path": source_path,
                "clean_record_source_sha256": source_sha,
                "runner_sha256": _sha("runner"),
                "config_sha256": _sha("config"),
                "commit": _head(),
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
        attempt_dir = f"attempts/{candidate.task}_s{candidate.state_id}/attempt_0"
        if len(rows) < limit:
            _write_capture_markers(records_dir, attempt_dir)
        else:
            _write_marker(records_dir, attempt_dir, "ATTEMPT_STARTED")
        rows.append(
            {
                "task": candidate.task,
                "state_id": candidate.state_id,
                "attempt_index": 0,
                "attempt_status": "CAPTURED" if len(rows) < limit else "NO_EVENT",
                "first_action_taken": "true" if len(rows) < limit else "false",
                "attempt_dir": attempt_dir,
                "clean_records_path": record_path.name if len(rows) < limit else "",
                "clean_records_sha256": "" if len(rows) >= limit else __import__("hashlib").sha256(record_path.read_bytes()).hexdigest(),
                "failure_reason": "",
            }
        )
    write_csv(
        path,
        rows,
        ["task", "state_id", "attempt_index", "attempt_status", "first_action_taken", "attempt_dir", "clean_records_path", "clean_records_sha256", "failure_reason"],
    )


def _write_all_candidate_records(directory: Path, *, event_count: int | None = None, model_sha: str | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    candidates = select_two_states_per_task()
    limit = len(candidates) if event_count is None else event_count
    for candidate in candidates[:limit]:
        _write_clean_records(
            directory / f"{candidate.task}_s{candidate.state_id}_clean_records.json",
            [
                _record(0, 31744, task=candidate.task, state_id=candidate.state_id, root=directory, model_sha=model_sha),
                _record(1, 31872, task=candidate.task, state_id=candidate.state_id, root=directory, model_sha=model_sha),
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


def test_runtime_gate_requires_all_expected_values(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    ledger = tmp_path / "ledger.csv"
    pool = tmp_path / "pool.csv"
    for path in (config, ledger, pool):
        path.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        expected_commit=_head(),
        expected_branch="",
        expected_config_sha256=_sha_file(config),
        expected_ledger_sha256=_sha_file(ledger),
        expected_pool_csv_sha256=_sha_file(pool),
        expected_cuda_visible_devices="2,6",
        expected_gpu_uuids="GPU-a,GPU-b",
    )
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.git_value", lambda argv: _head())
    with pytest.raises(RuntimeError, match="V5_RUNTIME_PROVENANCE_INCOMPLETE"):
        require_runtime_gates(args, config_path=config, ledger_path=ledger, pool_csv_path=pool)


def test_runtime_gate_uses_old_git_compatible_branch_and_rejects_head(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    ledger = tmp_path / "ledger.csv"
    pool = tmp_path / "pool.csv"
    for path in (config, ledger, pool):
        path.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        expected_commit="commit",
        expected_branch="branch",
        expected_config_sha256=_sha_file(config),
        expected_ledger_sha256=_sha_file(ledger),
        expected_pool_csv_sha256=_sha_file(pool),
        expected_cuda_visible_devices="2,6",
        expected_gpu_uuids="GPU-a,GPU-b",
    )
    seen = []

    def fake_git(argv):
        seen.append(tuple(argv))
        if argv == ["rev-parse", "HEAD"]:
            return "commit"
        if argv == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "HEAD"
        return ""

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.git_value", fake_git)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    with pytest.raises(RuntimeError, match="detached HEAD"):
        require_runtime_gates(args, config_path=config, ledger_path=ledger, pool_csv_path=pool)
    assert ("branch", "--show-current") not in seen
    assert ("rev-parse", "--abbrev-ref", "HEAD") in seen


def test_runtime_gate_rejects_wrong_cuda_or_busy_gpu(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    ledger = tmp_path / "ledger.csv"
    pool = tmp_path / "pool.csv"
    for path in (config, ledger, pool):
        path.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        expected_commit="commit",
        expected_branch="branch",
        expected_config_sha256=_sha_file(config),
        expected_ledger_sha256=_sha_file(ledger),
        expected_pool_csv_sha256=_sha_file(pool),
        expected_cuda_visible_devices="2,6",
        expected_gpu_uuids="GPU-a,GPU-b",
    )

    def fake_git(argv):
        return "commit" if argv == ["rev-parse", "HEAD"] else "branch"

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.git_value", fake_git)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.gpu_query_snapshot", lambda: "2, GPU-a, A, 0 MiB, 1 MiB, 0 %, 30\n6, GPU-b, B, 0 MiB, 1 MiB, 0 %, 30")
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.gpu_compute_process_snapshot", lambda: "GPU-b, 123, python")
    with pytest.raises(RuntimeError, match="existing compute process"):
        require_runtime_gates(args, config_path=config, ledger_path=ledger, pool_csv_path=pool)


def test_runtime_gate_rejects_unordered_gpu_uuid_match(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    ledger = tmp_path / "ledger.csv"
    pool = tmp_path / "pool.csv"
    for path in (config, ledger, pool):
        path.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        expected_commit="commit",
        expected_branch="branch",
        expected_config_sha256=_sha_file(config),
        expected_ledger_sha256=_sha_file(ledger),
        expected_pool_csv_sha256=_sha_file(pool),
        expected_cuda_visible_devices="5,4",
        expected_gpu_uuids="GPU-five,GPU-four",
    )

    def fake_git(argv):
        return "commit" if argv == ["rev-parse", "HEAD"] else "branch"

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.git_value", fake_git)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,4")
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.gpu_query_snapshot", lambda: "4, GPU-five, A, 0 MiB, 1 MiB, 0 %, 30\n5, GPU-four, B, 0 MiB, 1 MiB, 0 %, 30")
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.gpu_compute_process_snapshot", lambda: "NVIDIA_SMI_COMPUTE_EMPTY")
    with pytest.raises(RuntimeError, match="ordered GPU UUID binding mismatch"):
        require_runtime_gates(args, config_path=config, ledger_path=ledger, pool_csv_path=pool)


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
        select_events_from_clean_record_dir(cfg=bad, clean_records_dir=records_dir, attempt_rows=[])


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
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir)
    _rows, selected, status = select_events_from_clean_record_dir(
        cfg=cfg,
        clean_records_dir=records_dir,
        attempt_rows=list(__import__("csv").DictReader(ledger.open(encoding="utf-8", newline=""))),
    )
    expected = sorted(select_two_states_per_task(), key=lambda c: c.state_hash)[:8]
    assert status == "V5_EVENT_PANEL_INPUTS_FROZEN"
    assert [(e.task, e.state_id) for e in selected] == [(c.task, c.state_id) for c in expected]


def test_offline_select_stops_on_insufficient_pool(tmp_path):
    records_dir = tmp_path / "records"
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(records_dir, manifest_rows)
    _write_all_candidate_records(records_dir, event_count=7, model_sha=bundle_sha)
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir, captured_count=7)
    out = tmp_path / "out"
    args = SimpleNamespace(config=str(cfg_path), clean_records_dir=str(records_dir), output_dir=str(out), attempt_ledger=str(ledger))
    with pytest.raises(SystemExit, match="V5_CAPTURE_POOL_INSUFFICIENT"):
        run_offline_select(args)
    summary = json.loads((out / "m3_arm_v5_clean_capture_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "V5_CAPTURE_POOL_INSUFFICIENT"
    assert summary["selected_count"] == 7


def test_offline_select_writes_manifest_for_success(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(records_dir, manifest_rows)
    candidates = select_two_states_per_task()
    for candidate in candidates:
        _write_clean_records(
            records_dir / f"{candidate.task}_s{candidate.state_id}_clean_records.json",
            [
                _record(0, 31744, task=candidate.task, state_id=candidate.state_id, root=records_dir, model_sha=bundle_sha),
                _record(1, 31872, task=candidate.task, state_id=candidate.state_id, root=records_dir, model_sha=bundle_sha),
            ],
        )
    ledger = tmp_path / "attempt.csv"
    _write_attempt_ledger(ledger, records_dir)
    out = tmp_path / "out"
    args = SimpleNamespace(config=str(cfg_path), clean_records_dir=str(records_dir), output_dir=str(out), attempt_ledger=str(ledger))
    run_offline_select(args)
    assert (out / "m3_arm_v5_clean_capture_manifest.csv").exists()
    assert (out / "m3_arm_v5_clean_event_selection_all_states.csv").exists()
    assert (out / "m3_arm_v5_frozen_event_panel.csv").exists()
    assert (out / "m3_arm_v5_artifact_hash_manifest.csv").exists()


def test_independent_capture_auditor_passes_producer_style_output(tmp_path):
    capture_root = tmp_path / "capture"
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(capture_root, manifest_rows)
    _write_capture_manifest(capture_root)
    _write_all_candidate_records(capture_root, model_sha=bundle_sha)
    ledger = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    _write_attempt_ledger(ledger, capture_root)
    result = audit_capture_root(capture_root=capture_root, config_path=cfg_path, expected_commit=_head())
    assert result["audit_status"] == "PASS"
    assert result["selected_count"] == 8


def test_independent_capture_auditor_rejects_stale_failed_ledger(tmp_path):
    capture_root = tmp_path / "capture"
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(capture_root, manifest_rows)
    _write_capture_manifest(capture_root)
    _write_all_candidate_records(capture_root, model_sha=bundle_sha)
    ledger = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    _write_attempt_ledger(ledger, capture_root, captured_count=7)
    result = audit_capture_root(capture_root=capture_root, config_path=cfg_path, expected_commit=_head())
    assert result["audit_status"] == "FAIL"


def test_independent_capture_auditor_rejects_model_bundle_exact_set_mismatch(tmp_path):
    capture_root = tmp_path / "capture"
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(capture_root, manifest_rows)
    _write_capture_manifest(capture_root)
    (model_dir / "extra_remote_code.py").write_text("print('extra')", encoding="utf-8")
    _write_all_candidate_records(capture_root, model_sha=bundle_sha)
    ledger = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    _write_attempt_ledger(ledger, capture_root)
    result = audit_capture_root(capture_root=capture_root, config_path=cfg_path, expected_commit=_head())
    assert result["audit_status"] == "FAIL"
    assert "exact-set mismatch" in result["failure_reason"]


def test_independent_capture_auditor_rejects_phase_marker_mismatch(tmp_path):
    capture_root = tmp_path / "capture"
    model_dir, manifest_rows, bundle_sha = _make_model_dir(tmp_path)
    cfg_path = _write_temp_config(tmp_path, model_dir)
    _write_model_manifest(capture_root, manifest_rows)
    _write_capture_manifest(capture_root)
    _write_all_candidate_records(capture_root, model_sha=bundle_sha)
    ledger = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    _write_attempt_ledger(ledger, capture_root)
    marker = capture_root / "attempts" / "alphabet_soup_s9" / "attempt_0" / "FIRST_ACTION_TAKEN.marker"
    marker.unlink()
    result = audit_capture_root(capture_root=capture_root, config_path=cfg_path, expected_commit=_head())
    assert result["audit_status"] == "FAIL"
    assert "missing markers" in result["failure_reason"]


def _one_candidate():
    return select_two_states_per_task()[:1]


def _mock_capture_args(tmp_path: Path, cfg_path: Path):
    return SimpleNamespace(
        config=str(cfg_path),
        output_dir=str(tmp_path / "out"),
        model_gpu_device_id=-1,
        render_gpu_device_id=0,
        max_steps=2,
        num_steps_wait=0,
        expected_commit="commit",
        expected_branch="branch",
        expected_config_sha256="x",
        expected_ledger_sha256="y",
        expected_pool_csv_sha256="z",
        expected_cuda_visible_devices="2,6",
        expected_gpu_uuids="GPU-a,GPU-b",
    )


def test_capture_runner_allows_one_pregeneration_retry(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"path": str(tmp_path / "model")}, "selection": {"prior_layer3_state_ledger": "ledger.csv"}}), encoding="utf-8")
    calls = {"n": 0}
    candidate = _one_candidate()[0]

    class FakeModel:
        config = SimpleNamespace(model_type="mock")
        bin_centers = torch.zeros(1)
        norm_stats = {}

        def parameters(self):
            return iter([torch.zeros(1)])

    def fake_capture(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pre-generation")
        attempt_dir = kwargs["attempt_dir"]
        for phase in ("ENV_READY", "FIRST_ACTION_GENERATED", "FIRST_ACTION_TAKEN"):
            _write_marker(attempt_dir.parent.parent.parent, str(attempt_dir.relative_to(attempt_dir.parent.parent.parent)), phase)
        out = Path(kwargs["output_dir"])
        rel = f"states/{candidate.task}_s{candidate.state_id}/attempt_1/records.json"
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"records":[]}', encoding="utf-8")
        return rel, _sha_file(path)

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_output_dir_new", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_clean_worktree", lambda: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_runtime_gates", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_frozen_pool_sources", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.model_bundle_manifest", lambda _path: ([{"relative_path": "config.json", "size_bytes": 2, "sha256": _sha("m")}], _sha("bundle")))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.load_model", lambda *args, **kwargs: (FakeModel(), object(), "cpu"))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.run_clean_capture_for_state", fake_capture)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_attempt_ledger_policy", lambda *args, **kwargs: None)
    from scripts.stageb.run_m3_arm_v5_clean_capture import run_capture_clean_pool

    run_capture_clean_pool(_mock_capture_args(tmp_path, cfg_path))
    rows = list(__import__("csv").DictReader((tmp_path / "out" / "m3_arm_v5_capture_attempt_ledger.csv").open(encoding="utf-8", newline="")))
    assert [row["attempt_index"] for row in rows] == ["0", "1"]
    assert rows[0]["attempt_status"] == "FIRST_ACTION_BEFORE_INFRA_FAILURE"
    assert rows[1]["attempt_status"] == "CAPTURED"


def test_capture_runner_forbids_post_generation_retry(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"path": str(tmp_path / "model")}, "selection": {"prior_layer3_state_ledger": "ledger.csv"}}), encoding="utf-8")
    candidate = _one_candidate()[0]

    class FakeModel:
        config = SimpleNamespace(model_type="mock")
        bin_centers = torch.zeros(1)
        norm_stats = {}

        def parameters(self):
            return iter([torch.zeros(1)])

    def fake_capture(**kwargs):
        attempt_dir = kwargs["attempt_dir"]
        _write_marker(attempt_dir.parent.parent.parent, str(attempt_dir.relative_to(attempt_dir.parent.parent.parent)), "FIRST_ACTION_GENERATED")
        raise RuntimeError("post-generation")

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_output_dir_new", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_clean_worktree", lambda: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_runtime_gates", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_frozen_pool_sources", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.model_bundle_manifest", lambda _path: ([{"relative_path": "config.json", "size_bytes": 2, "sha256": _sha("m")}], _sha("bundle")))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.load_model", lambda *args, **kwargs: (FakeModel(), object(), "cpu"))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.run_clean_capture_for_state", fake_capture)
    from scripts.stageb.run_m3_arm_v5_clean_capture import run_capture_clean_pool

    with pytest.raises(RuntimeError, match="post-generation"):
        run_capture_clean_pool(_mock_capture_args(tmp_path, cfg_path))
    rows = list(__import__("csv").DictReader((tmp_path / "out" / "m3_arm_v5_capture_attempt_ledger.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["attempt_status"] == "CAPTURE_FAILED_POST_ACTION"


def test_capture_runner_records_sigterm_as_terminal_post_action(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"path": str(tmp_path / "model")}, "selection": {"prior_layer3_state_ledger": "ledger.csv"}}), encoding="utf-8")
    candidate = _one_candidate()[0]

    class FakeModel:
        config = SimpleNamespace(model_type="mock")
        bin_centers = torch.zeros(1)
        norm_stats = {}

        def parameters(self):
            return iter([torch.zeros(1)])

    def fake_capture(**kwargs):
        attempt_dir = kwargs["attempt_dir"]
        root = attempt_dir.parent.parent.parent
        rel = str(attempt_dir.relative_to(root))
        _write_marker(root, rel, "FIRST_ACTION_GENERATED")
        _write_marker(root, rel, "FIRST_ACTION_TAKEN")
        from scripts.stageb.run_m3_arm_v5_clean_capture import CaptureTermination

        raise CaptureTermination("received signal 15")

    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_output_dir_new", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_clean_worktree", lambda: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.require_runtime_gates", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.validate_frozen_pool_sources", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.model_bundle_manifest", lambda _path: ([{"relative_path": "config.json", "size_bytes": 2, "sha256": _sha("m")}], _sha("bundle")))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.load_model", lambda *args, **kwargs: (FakeModel(), object(), "cpu"))
    monkeypatch.setattr("scripts.stageb.run_m3_arm_v5_clean_capture.run_clean_capture_for_state", fake_capture)
    from scripts.stageb.run_m3_arm_v5_clean_capture import run_capture_clean_pool

    with pytest.raises(RuntimeError, match="received signal 15"):
        run_capture_clean_pool(_mock_capture_args(tmp_path, cfg_path))
    rows = list(__import__("csv").DictReader((tmp_path / "out" / "m3_arm_v5_capture_attempt_ledger.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["attempt_status"] == "CAPTURE_FAILED_POST_ACTION"
    assert rows[0]["first_action_taken"] == "true"
    assert "received signal 15" in rows[0]["failure_reason"]


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


def test_verify_exact_input_binding_rejects_forged_sha_and_bad_gpu(tmp_path):
    row = _record(1, 31872, task="ketchup", state_id=41, root=tmp_path)
    prev = _record(0, 31744, task="ketchup", state_id=41, root=tmp_path)
    for key in ("raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256", "prompt_token_ids", "prompt_token_ids_sha256"):
        row[f"previous_{key}"] = prev[key]
    row["previous_official_score_argmax_token_id"] = prev["official_score_argmax_token_id"]
    row["selected_step"] = 1
    verify_exact_input_binding(row, capture_root=tmp_path, expected_commit=_head())
    forged = dict(row, raw_image_sha256=_sha("wrong"))
    with pytest.raises(ValueError, match="raw_image sha mismatch"):
        verify_exact_input_binding(forged, capture_root=tmp_path, expected_commit="commit-sha")
    bad_gpu = dict(row, gpu_query="NVIDIA_SMI_UNAVAILABLE")
    with pytest.raises(Exception, match="invalid GPU"):
        verify_exact_input_binding(bad_gpu, capture_root=tmp_path, expected_commit="commit-sha")


def test_model_bundle_sha_changes_when_shard_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_text("a", encoding="utf-8")
    _rows, first = model_bundle_manifest(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_text("b", encoding="utf-8")
    _rows, second = model_bundle_manifest(tmp_path)
    assert first != second


def test_verify_model_bundle_manifest_rejects_tampered_model_shard(tmp_path):
    model_dir, rows, _bundle_sha = _make_model_dir(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, rows, ["relative_path", "size_bytes", "sha256"])
    assert verify_model_bundle_manifest(manifest, model_dir)
    (model_dir / "model-00001-of-00001.safetensors").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|sha mismatch"):
        verify_model_bundle_manifest(manifest, model_dir)


def test_verify_model_bundle_manifest_rejects_tampered_manifest(tmp_path):
    model_dir, rows, _bundle_sha = _make_model_dir(tmp_path)
    rows[0]["relative_path"] = "../escape"
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, rows, ["relative_path", "size_bytes", "sha256"])
    with pytest.raises(ValueError, match="unsafe relative path"):
        verify_model_bundle_manifest(manifest, model_dir)


def test_selected_rows_reject_mixed_model_bundle_sha(tmp_path):
    row = _record(1, 31872, task="ketchup", state_id=41, root=tmp_path, model_sha=_sha("model-a"))
    prev = _record(0, 31744, task="ketchup", state_id=41, root=tmp_path, model_sha=_sha("model-a"))
    for key in ("raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256", "prompt_token_ids", "prompt_token_ids_sha256"):
        row[f"previous_{key}"] = prev[key]
    row["previous_official_score_argmax_token_id"] = prev["official_score_argmax_token_id"]
    row["selected_step"] = 1
    other = _record(1, 31872, task="ketchup", state_id=42, root=tmp_path, model_sha=_sha("model-b"))
    other_prev = _record(0, 31744, task="ketchup", state_id=42, root=tmp_path, model_sha=_sha("model-b"))
    for key in ("raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256", "prompt_token_ids", "prompt_token_ids_sha256"):
        other[f"previous_{key}"] = other_prev[key]
    other["previous_official_score_argmax_token_id"] = other_prev["official_score_argmax_token_id"]
    other["selected_step"] = 1
    ok, reason = verify_selected_rows_exact_bindings([row, other], capture_root=tmp_path, expected_commit=_head())
    assert not ok
    assert "mixed model bundle" in reason


def test_prepare_generation_inputs_appends_actual_prompt_suffix():
    class DummyProcessor:
        def __call__(self, _prompt, _image, return_tensors):
            return {
                "input_ids": torch.tensor([[10, 20]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            }

    cfg = {
        "preprocess": {
            "libero_official_preprocess": False,
            "libero_preprocess_backend": "none",
            "center_crop": False,
            "resize_size": 2,
        }
    }
    out = prepare_generation_inputs(
        raw=__import__("numpy").zeros((2, 2, 3), dtype="uint8"),
        processor=DummyProcessor(),
        instruction="test",
        cfg=cfg,
        device="cpu",
        model_dtype=torch.float32,
    )
    assert out["input_ids"].tolist() == [[10, 20, 29871]]
    assert "attention_mask" not in out


def test_config_attack_seed_is_not_legacy_seed85_or_86():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = int(cfg["selection"]["first_attack_seed"]["seed"])
    assert seed == 428198
    assert seed not in {85, 86}
