from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import subprocess

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "detector_v5"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_d8_3b_run as auditor
import run_d8_2_cv_parallel as dispatcher
import run_d8_2_cv_unit as unit

BASH = Path(r"C:\msys64\usr\bin\bash.exe")


def _records(
    *,
    auroc_by_seed: dict[int, float] | None = None,
    bacc: float = 0.75,
    mcc: float = 0.10,
    status: str = "COMPLETED",
) -> list[dict]:
    auroc_by_seed = auroc_by_seed or {}
    rows = []
    for seed in dispatcher.D8_3B_SEEDS:
        for fold in dispatcher.D8_3B_FOLDS:
            rows.append(
                {
                    "job_id": f"B3_seed{seed}_fold{fold}",
                    "config": "B3",
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "metrics": {
                        "auroc": auroc_by_seed.get(seed, 0.85),
                        "balanced_accuracy": bacc,
                        "mcc": mcc,
                    },
                }
            )
    return rows


def test_exact_matrix_accepts():
    assert (
        dispatcher.validate_d8_3b_matrix(
            ["B3"],
            dispatcher.D8_3B_SEEDS,
            dispatcher.D8_3B_FOLDS,
            100,
            [0, 1],
        )
        == 50
    )


@pytest.mark.parametrize(
    "configs,seeds,folds,epochs,gpus",
    [
        (["B2"], dispatcher.D8_3B_SEEDS, dispatcher.D8_3B_FOLDS, 100, [0, 1]),
        (["B3", "B3"], dispatcher.D8_3B_SEEDS, dispatcher.D8_3B_FOLDS, 100, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS[:-1], dispatcher.D8_3B_FOLDS, 100, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS[:9] + [dispatcher.D8_3B_SEEDS[0]], dispatcher.D8_3B_FOLDS, 100, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS[:-1] + [99999999], dispatcher.D8_3B_FOLDS, 100, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS, dispatcher.D8_3B_FOLDS, 99, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS, [0, 1, 2, 3, 4, 5], 100, [0, 1]),
        (["B3"], dispatcher.D8_3B_SEEDS, dispatcher.D8_3B_FOLDS, 100, []),
        (["B3"], dispatcher.D8_3B_SEEDS, dispatcher.D8_3B_FOLDS, 100, [0, 0]),
    ],
)
def test_matrix_contract_rejects_mutations(configs, seeds, folds, epochs, gpus):
    with pytest.raises(ValueError):
        dispatcher.validate_d8_3b_matrix(configs, seeds, folds, epochs, gpus)


def test_invalid_matrix_is_rejected_before_python_probe_or_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dispatcher,
        "validate_python_environment",
        lambda _: pytest.fail("python probe must not run"),
    )
    with pytest.raises(ValueError):
        dispatcher.main(
            [
                "--cache-root",
                str(tmp_path / "cache"),
                "--cache-seal",
                "a" * 64,
                "--output-root",
                str(tmp_path / "run"),
                "--log-root",
                str(tmp_path),
                "--python-bin",
                str(Path(sys.executable).resolve()),
                "--gpus",
                "0",
                "--seeds",
                ",".join(str(seed) for seed in dispatcher.D8_3B_SEEDS),
                "--epochs",
                "100",
                "--configs",
                "B2",
            ]
        )
    assert not (tmp_path / "run").exists()


def _valid_main_args(tmp_path: Path, **overrides) -> list[str]:
    values = {
        "cache-root": tmp_path / "cache",
        "cache-seal": "f" * 64,
        "cache-a-seal": "f" * 64,
        "cache-b-seal": "4" * 64,
        "comparator-seal": "5" * 64,
        "p5-artifact-seal": "6" * 64,
        "h1-review-seal": "7" * 64,
        "h1-source-commit": "1" * 40,
        "h1-source-tree": "2" * 40,
        "source-snapshot-sha256": "3" * 64,
        "expected-source-commit": "a" * 40,
        "expected-source-tree": "b" * 40,
        "shell-script-sha256": "8" * 64,
        "output-root": tmp_path / "run",
        "log-root": tmp_path / "logs",
        "python-bin": Path(sys.executable).resolve(),
        "gpus": "0",
        "seeds": ",".join(str(seed) for seed in dispatcher.D8_3B_SEEDS),
        "configs": "B3",
        "epochs": 100,
    }
    values.update(overrides)
    Path(values["cache-root"]).mkdir(parents=True, exist_ok=True)
    Path(values["log-root"]).mkdir(parents=True, exist_ok=True)
    args = []
    for key, value in values.items():
        args.extend([f"--{key}", str(value)])
    return args


def _fake_probe_environment() -> dict:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "cuda_available": True,
        "cuda_device_count": 1,
        "inherited_CUDA_VISIBLE_DEVICES": "",
    }


def test_preflight_only_success_has_no_run_root_manifest_or_dispatch(tmp_path, monkeypatch, capsys):
    args = _valid_main_args(tmp_path) + ["--preflight-only"]
    monkeypatch.setattr(
        dispatcher,
        "git_provenance",
        lambda: {"source_commit": "a" * 40, "source_tree": "b" * 40},
    )
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: _fake_probe_environment())
    monkeypatch.setattr(dispatcher, "validate_cache_seal", lambda *_: "f" * 64)
    monkeypatch.setattr(dispatcher, "dispatch_jobs", lambda *_args, **_kwargs: pytest.fail("dispatch ran"))
    monkeypatch.setattr(dispatcher, "build_jobs", lambda *_args, **_kwargs: pytest.fail("jobs built"))
    assert dispatcher.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "PASS"
    assert report["run_root_created"] is False
    assert report["manifest_created"] is False
    assert report["children_launched"] == 0
    assert report["gpu_training"] == 0
    assert not (tmp_path / "run").exists()


def test_preflight_only_failure_is_structured_and_has_no_run_root(tmp_path, monkeypatch, capsys):
    args = _valid_main_args(tmp_path) + ["--preflight-only"]
    monkeypatch.setattr(
        dispatcher,
        "git_provenance",
        lambda: {"source_commit": "a" * 40, "source_tree": "b" * 40},
    )
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: _fake_probe_environment())
    monkeypatch.setattr(
        dispatcher,
        "validate_cache_seal",
        lambda *_: (_ for _ in ()).throw(RuntimeError("cache mismatch")),
    )
    assert dispatcher.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL"
    assert report["error"]["message"] == "cache mismatch"
    assert report["children_launched"] == 0
    assert not (tmp_path / "run").exists()


def test_expected_source_commit_mismatch_is_before_probe_and_run_root(tmp_path, monkeypatch):
    args = _valid_main_args(tmp_path)
    monkeypatch.setattr(
        dispatcher,
        "git_provenance",
        lambda: {"source_commit": "d" * 40, "source_tree": "b" * 40},
    )
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: pytest.fail("probe ran"))
    with pytest.raises(RuntimeError, match="commit mismatch"):
        dispatcher.main(args)
    assert not (tmp_path / "run").exists()


def test_expected_source_tree_mismatch_is_before_probe_and_run_root(tmp_path, monkeypatch):
    args = _valid_main_args(tmp_path)
    monkeypatch.setattr(
        dispatcher,
        "git_provenance",
        lambda: {"source_commit": "a" * 40, "source_tree": "e" * 40},
    )
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: pytest.fail("probe ran"))
    with pytest.raises(RuntimeError, match="tree mismatch"):
        dispatcher.main(args)
    assert not (tmp_path / "run").exists()


def test_dirty_source_is_fail_closed_before_run_root(tmp_path, monkeypatch):
    args = _valid_main_args(tmp_path)
    monkeypatch.setattr(dispatcher, "git_provenance", lambda: (_ for _ in ()).throw(RuntimeError("dirty")))
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: pytest.fail("probe ran"))
    with pytest.raises(RuntimeError, match="dirty"):
        dispatcher.main(args)
    assert not (tmp_path / "run").exists()


def test_cuda_probe_contract_rejects_false_count_and_inherited_visibility():
    with pytest.raises(RuntimeError, match="not available"):
        dispatcher.validate_cuda_environment({"cuda_available": False, "cuda_device_count": 1}, [0])
    with pytest.raises(RuntimeError, match="device count"):
        dispatcher.validate_cuda_environment({"cuda_available": True, "cuda_device_count": 1}, [1])
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        dispatcher.validate_cuda_environment(
            {
                "cuda_available": True,
                "cuda_device_count": 1,
                "inherited_CUDA_VISIBLE_DEVICES": "0",
            },
            [0],
        )


def test_cache_seal_mismatch_is_before_run_root_or_child(tmp_path, monkeypatch):
    args = _valid_main_args(tmp_path)
    monkeypatch.setattr(
        dispatcher,
        "git_provenance",
        lambda: {"source_commit": "a" * 40, "source_tree": "b" * 40},
    )
    monkeypatch.setattr(dispatcher, "validate_python_environment", lambda *_: _fake_probe_environment())
    monkeypatch.setattr(dispatcher, "verify_seal", lambda _root: {"sha256sums_sha256": "e" * 64})
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("child ran"))
    with pytest.raises(RuntimeError, match="Cache A seal mismatch"):
        dispatcher.main(args)
    assert not (tmp_path / "run").exists()


def test_h1_lineage_requires_all_complete_sha_values():
    with pytest.raises(ValueError, match="complete SHA"):
        dispatcher.validate_h1_lineage({})
    invalid = {
        "h1_source_commit": "z" * 40,
        "h1_source_tree": "2" * 40,
        "source_snapshot_sha256": "3" * 64,
        "cache_a_seal": "f" * 64,
        "cache_b_seal": "4" * 64,
        "comparator_seal": "5" * 64,
        "p5_artifact_seal": "6" * 64,
        "h1_review_seal": "7" * 64,
    }
    with pytest.raises(ValueError, match="complete SHA"):
        dispatcher.validate_h1_lineage(invalid)


def test_shell_separates_log_and_run_roots_and_requires_explicit_python():
    text = (SCRIPTS / "d8_launch_3b_safe.sh").read_text(encoding="utf-8")
    assert "D8_LOG_ROOT" in text
    assert "D8_RUN_ROOT" in text
    assert "D8_PYTHON_BIN" in text
    assert "which python" not in text
    assert '> "${DISPATCH_LOG}"' in text
    for required in (
        "D8_EXPECTED_SOURCE_COMMIT",
        "D8_EXPECTED_SOURCE_TREE",
        "D8_CACHE_A_SEAL",
        "D8_H1_SOURCE_COMMIT",
        "D8_H1_REVIEW_SEAL",
        "--shell-script-sha256",
        "kill -0",
        "EXECUTION_RECEIPT.json",
    ):
        assert required in text


def test_manifest_atomic_update_has_no_temporary_leftover(tmp_path):
    path = tmp_path / "JOB_MANIFEST.json"
    dispatcher.atomic_write_json(path, {"state": "PENDING"})
    dispatcher.atomic_write_json(path, {"state": "RUNNING"})
    assert json.loads(path.read_text())["state"] == "RUNNING"
    assert not list(tmp_path.glob("*.tmp"))


def test_initial_job_plan_is_exactly_50_pending_records_without_output_creation(tmp_path):
    output_root = tmp_path / "run"
    environment = {"executable": str(Path(sys.executable).resolve()), "cuda_available": False}
    jobs = dispatcher.build_jobs(
        configs=["B3"],
        seeds=dispatcher.D8_3B_SEEDS,
        folds=dispatcher.D8_3B_FOLDS,
        gpu_ids=[0, 1],
        epochs=100,
        cache_root=str(tmp_path / "cache"),
        cache_seal="a" * 64,
        output_root=output_root,
        python_bin=str(Path(sys.executable).resolve()),
        source_commit="b" * 40,
        source_tree="c" * 40,
        python_environment=environment,
        lineage_digest="d" * 64,
    )
    assert len(jobs) == 50
    assert all(job["status"] == "PENDING" for job in jobs)
    assert all({"job_id", "planned_index", "gpu", "pid", "command", "metrics_path", "checkpoint_path", "predictions_path", "log_path"} <= set(job) for job in jobs)
    assert not output_root.exists()


class _FakeProcess:
    next_pid = 1000

    def __init__(self, finish_after=2):
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.poll_count = 0
        self.finish_after = finish_after
        self.returncode = None
        self.terminated = False

    def poll(self):
        self.poll_count += 1
        if self.terminated:
            self.returncode = -15
        elif self.poll_count >= self.finish_after:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.terminated = True
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def _dispatch_manifest(tmp_path, count=4):
    jobs = []
    for index in range(count):
        jobs.append(
            {
                "job_id": f"job{index}",
                "config": "B3",
                "seed": dispatcher.D8_3B_SEEDS[0],
                "fold": index,
                "planned_index": index,
                "status": "PENDING",
                "gpu": None,
                "pid": None,
                "started_utc": None,
                "finished_utc": None,
                "exit_code": None,
                "log_path": str(tmp_path / f"job{index}.log"),
                "failure_reason": None,
            }
        )
    manifest = {
        "matrix": {"gpus": [0, 1]},
        "jobs": jobs,
        "abort_reason": None,
    }
    path = tmp_path / "JOB_MANIFEST.json"
    dispatcher.atomic_write_json(path, manifest)
    return path, manifest


def test_mocked_dispatch_never_exceeds_gpu_slots(tmp_path):
    path, manifest = _dispatch_manifest(tmp_path)
    active = set()
    max_active = 0
    processes = []

    def launch(job):
        nonlocal max_active
        process = _FakeProcess()
        processes.append(process)
        active.add(process.pid)
        max_active = max(max_active, len(active))
        return process, io.StringIO()

    def validate(job, returncode):
        active.discard(job["pid"])
        return True, "ok", {"auroc": 0.85, "balanced_accuracy": 0.75, "mcc": 0.1}

    dispatcher.dispatch_jobs(
        path,
        tmp_path / "STOP_D8_3B",
        manifest,
        launch_fn=launch,
        validate_fn=validate,
        sleep_fn=lambda _: None,
        poll_interval=0,
    )
    assert max_active <= 2
    assert all(job["status"] == "COMPLETED" for job in manifest["jobs"])
    assert all(process.returncode == 0 for process in processes)


def test_first_formal_job_failure_aborts_without_launching_second(tmp_path):
    path, manifest = _dispatch_manifest(tmp_path, count=3)
    manifest["matrix"]["gpus"] = [0]
    manifest["dispatcher_pid"] = 1
    manifest["created_utc"] = "2026-08-04T00:00:00+00:00"
    launched = []

    def launch(job):
        launched.append(job["job_id"])
        return _FakeProcess(finish_after=1), io.StringIO()

    def validate(job, _returncode):
        if job["job_id"] == "job0":
            return False, "checkpoint_schema", {}
        return True, "ok", {"auroc": 0.85, "balanced_accuracy": 0.75, "mcc": 0.1}

    dispatcher.dispatch_jobs(
        path,
        tmp_path / "STOP_D8_3B",
        manifest,
        launch_fn=launch,
        validate_fn=validate,
        sleep_fn=lambda _: None,
        poll_interval=0,
    )
    assert launched == ["job0"]
    assert manifest["jobs"][0]["status"] == "FAILED"
    assert manifest["jobs"][1]["status"] == "ABORTED"
    assert manifest["jobs"][2]["status"] == "ABORTED"
    assert manifest["abort_reason"].startswith("JOB_FAILURE:job0:checkpoint_schema")
    assert dispatcher._finalize_run(
        tmp_path,
        manifest,
        {},
        abort_reason=manifest["abort_reason"],
        audit_fn=lambda _root, **_kwargs: {"verdict": "FAIL", "errors": []},
        seal_writer=lambda _root: None,
        seal_verifier=lambda _root: {},
    ) == 1


def test_kill_switch_skips_pending_and_is_not_pass(tmp_path):
    path, manifest = _dispatch_manifest(tmp_path)
    (tmp_path / "STOP_D8_3B").touch()
    dispatcher.dispatch_jobs(
        path,
        tmp_path / "STOP_D8_3B",
        manifest,
        sleep_fn=lambda _: None,
        poll_interval=0,
    )
    assert manifest["abort_reason"] == "KILL_SWITCH"
    assert all(job["status"] == "SKIPPED_KILL_SWITCH" for job in manifest["jobs"])
    assert not dispatcher.final_gate(_records()[:0])["pass"]


def test_signal_terminates_inflight_and_records_nonzero(tmp_path, monkeypatch):
    path, manifest = _dispatch_manifest(tmp_path, count=2)
    calls = 0
    processes = []

    monkeypatch.setattr(
        dispatcher,
        "_signal_process",
        lambda process, kill=False: process.kill() if kill else process.terminate(),
    )

    def launch(job):
        process = _FakeProcess(finish_after=1000)
        processes.append(process)
        return process, io.StringIO()

    def signal_reason():
        return "SIGTERM" if calls > 0 else None

    def sleep(_):
        nonlocal calls
        calls += 1

    dispatcher.dispatch_jobs(
        path,
        tmp_path / "STOP_D8_3B",
        manifest,
        launch_fn=launch,
        validate_fn=lambda _job, _rc: (True, "ok", {}),
        abort_reason_fn=signal_reason,
        sleep_fn=sleep,
        poll_interval=0,
        grace_seconds=1,
    )
    assert manifest["abort_reason"] == "SIGTERM"
    assert all(job["status"] == "ABORTED" for job in manifest["jobs"])
    assert all(process.terminated for process in processes)
    assert all(job["exit_code"] != 0 for job in manifest["jobs"])


def test_gate_rejects_incomplete_and_failed_closure():
    assert not dispatcher.final_gate(_records()[:49])["pass"]
    nine_seeds = [row for row in _records() if row["seed"] != dispatcher.D8_3B_SEEDS[-1]]
    assert not dispatcher.final_gate(nine_seeds)["pass"]
    failed = _records()
    failed[0]["status"] = "FAILED"
    assert not dispatcher.final_gate(failed)["pass"]


def test_gate_rejects_nonfinite_and_per_seed_metric_failure():
    for bad_value in (float("nan"), float("inf")):
        nonfinite = _records()
        nonfinite[0]["metrics"]["auroc"] = bad_value
        with np.errstate(all="ignore"):
            assert not dispatcher.final_gate(nonfinite)["pass"]
    low = _records()
    for row in low[-5:]:
        row["metrics"]["mcc"] = 0.0
    assert not dispatcher.final_gate(low)["pass"]


def test_gate_uses_sample_std_ddof1_and_rejects_large_std():
    means = {seed: 0.80 + index * 0.001 for index, seed in enumerate(dispatcher.D8_3B_SEEDS)}
    result = dispatcher.final_gate(_records(auroc_by_seed=means))
    assert np.isclose(
        result["stability_std_ddof1"],
        np.std(list(means.values()), ddof=1),
    )
    assert result["pass"]
    too_wide = {seed: 0.80 + index * 0.01 for index, seed in enumerate(dispatcher.D8_3B_SEEDS)}
    result = dispatcher.final_gate(_records(auroc_by_seed=too_wide))
    assert result["stability_std_ddof1"] > 0.03
    assert not result["pass"]


def test_job_validator_rejects_nonzero_and_missing_or_nonfinite_artifacts(tmp_path):
    unit = tmp_path / "B3_unit"
    job = {
        "config": "B3",
        "seed": dispatcher.D8_3B_SEEDS[0],
        "fold": 0,
        "metrics_path": str(unit / "metrics.json"),
        "checkpoint_path": str(unit / "checkpoint.pt"),
        "predictions_path": str(unit / "predictions.json"),
        "expected_provenance": {},
    }
    ok, reason, _ = dispatcher.validate_job_artifacts(job, 7)
    assert not ok and reason.startswith("nonzero_returncode")
    unit.mkdir()
    (unit / "metrics.json").write_text("{}")
    (unit / "predictions.json").write_text("[]")
    ok, reason, _ = dispatcher.validate_job_artifacts(job, 0)
    assert not ok and "checkpoint" in reason
    for missing_name in ("metrics_path", "checkpoint_path", "predictions_path"):
        missing = tmp_path / missing_name
        missing.mkdir()
        missing_job = {**job}
        missing_job["metrics_path"] = str(missing / "metrics.json")
        missing_job["checkpoint_path"] = str(missing / "checkpoint.pt")
        missing_job["predictions_path"] = str(missing / "predictions.json")
        ok, _, _ = dispatcher.validate_job_artifacts(missing_job, 0)
        assert not ok


def _synthetic_unit_entries():
    def row(episode_id, fold_id, target, step):
        return {
            "episode_id": episode_id,
            "fold_id": fold_id,
            "step": step,
            "effective_mask": True,
            "physical_target": float(target),
            "D8_weight": 1.0,
            "features_25d_raw": [float(target) + (index / 100.0) for index in range(25)],
        }

    return [
        row("train_neg", 1, 0, 0),
        row("train_pos", 1, 1, 0),
        row("val_neg", 0, 0, 0),
        row("val_pos", 0, 1, 0),
    ]


def _synthetic_lineage():
    lineage = {
        "h1_source_commit": "1" * 40,
        "h1_source_tree": "2" * 40,
        "source_snapshot_sha256": "3" * 64,
        "cache_a_seal": "f" * 64,
        "cache_b_seal": "4" * 64,
        "comparator_seal": "5" * 64,
        "p5_artifact_seal": "6" * 64,
        "h1_review_seal": "7" * 64,
    }
    lineage["lineage_digest"] = dispatcher.canonical_lineage_digest(lineage)
    return lineage


def test_provenance_has_no_schema_and_payload_collision_is_rejected(tmp_path):
    lineage = _synthetic_lineage()
    provenance = unit._provenance(
        cache_root=tmp_path / "cache",
        cache_seal="f" * 64,
        config="B3",
        fold=0,
        seed=20260719,
        epochs=1,
        started_utc="2026-08-04T00:00:00+00:00",
        source_commit="a" * 40,
        source_tree="b" * 40,
        launcher_sha256="c" * 64,
        train_core_sha256="d" * 64,
        receipt={"lineage_digest": lineage["lineage_digest"]},
    )
    assert "schema" not in provenance
    with pytest.raises(RuntimeError, match="schema"):
        unit._artifact_payload("D8_3B_CHECKPOINT_V2", {"schema": "wrong"}, provenance)
    with pytest.raises(RuntimeError, match="schema"):
        unit._artifact_payload("D8_3B_CHECKPOINT_V2", {}, {**provenance, "schema": "wrong"})


def test_real_run_unit_producer_roundtrip_and_strict_canary_audit(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    lineage = _synthetic_lineage()
    monkeypatch.setattr(
        unit,
        "load_cache",
        lambda _root, _seal: (_synthetic_unit_entries(), {"sha256sums_sha256": "f" * 64}),
    )
    monkeypatch.setattr(unit.torch.cuda, "is_available", lambda: False)
    receipt = tmp_path / "EXECUTION_RECEIPT.json"
    receipt.write_text(json.dumps({"lineage_digest": lineage["lineage_digest"]}))
    output_root = tmp_path / "bundle"
    unit_dir = output_root / "seed20260719" / "B3_fold0"
    metrics = unit.run_unit(
        cache_root,
        "B3",
        0,
        20260719,
        1,
        unit_dir,
        expected_cache_seal="f" * 64,
        source_commit="a" * 40,
        source_tree="b" * 40,
        launcher_sha256=dispatcher.sha256_file(SCRIPTS / "run_d8_2_cv_parallel.py"),
        train_core_sha256=dispatcher.sha256_file(SCRIPTS / "d8_train_core.py"),
        execution_receipt=receipt,
    )
    checkpoint = torch.load(unit_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    metrics_file = json.loads((unit_dir / "metrics.json").read_text())
    predictions = json.loads((unit_dir / "predictions.json").read_text())
    assert metrics["schema"] == metrics_file["schema"] == "D8_3B_UNIT_METRICS_V2"
    assert checkpoint["schema"] == "D8_3B_CHECKPOINT_V2"
    assert checkpoint["schema"] != metrics_file["schema"]
    assert checkpoint["model_state"]
    assert checkpoint["normalization"]["schema"] == "D8_NORMALIZATION_V2"
    assert checkpoint["normalization"]["feature_dim"] == 25
    assert predictions
    expected = {key: metrics_file[key] for key in dispatcher.PROVENANCE_FIELDS}
    job = {
        "job_id": "B3_seed20260719_fold0",
        "config": "B3",
        "seed": 20260719,
        "fold": 0,
        "status": "COMPLETED",
        "exit_code": 0,
        "metrics_path": str(unit_dir / "metrics.json"),
        "checkpoint_path": str(unit_dir / "checkpoint.pt"),
        "predictions_path": str(unit_dir / "predictions.json"),
        "expected_provenance": expected,
    }
    ok, reason, observed = dispatcher.validate_job_artifacts(job, 0)
    assert (ok, reason) == (True, "ok")
    assert observed["schema"] == "D8_3B_UNIT_METRICS_V2"
    global_provenance = {
        **expected,
        "shell_script_sha256": auditor.sha256_file(SCRIPTS / "d8_launch_3b_safe.sh"),
        "h1_lineage": lineage,
    }
    canary_audit = auditor.audit_unit_bundle(
        output_root,
        job,
        global_provenance=global_provenance,
        verify_source_scripts=True,
    )
    assert canary_audit["verdict"] == "PASS", canary_audit

    checkpoint["schema"] = "D8_3B_UNIT_METRICS_V2"
    torch.save(checkpoint, unit_dir / "checkpoint.pt")
    ok, reason, _ = dispatcher.validate_job_artifacts(job, 0)
    assert not ok and "checkpoint" in reason
    rejected = auditor.audit_unit_bundle(output_root, job, global_provenance=global_provenance)
    assert rejected["verdict"] == "FAIL"
    assert any("checkpoint_schema" in error for error in rejected["errors"])


def _make_auditor_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    environment = {
        "python_version": "synthetic",
        "python_implementation": "CPython",
        "executable": "/synthetic/python",
        "torch_version": "synthetic",
        "cuda_version": None,
        "cuda_available": True,
        "cuda_device_count": 2,
        "inherited_CUDA_VISIBLE_DEVICES": "",
        "numpy_version": "synthetic",
        "sklearn_version": "synthetic",
    }
    lineage = {
        "h1_source_commit": "1" * 40,
        "h1_source_tree": "2" * 40,
        "source_snapshot_sha256": "3" * 64,
        "cache_a_seal": "f" * 64,
        "cache_b_seal": "4" * 64,
        "comparator_seal": "5" * 64,
        "p5_artifact_seal": "6" * 64,
        "h1_review_seal": "7" * 64,
    }
    lineage["lineage_digest"] = dispatcher.canonical_lineage_digest(lineage)
    shell_path = SCRIPTS / "d8_launch_3b_safe.sh"
    source = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "expected_source_commit": "a" * 40,
        "expected_source_tree": "b" * 40,
        "unit_script_sha256": auditor.sha256_file(SCRIPTS / "run_d8_2_cv_unit.py"),
        "parallel_launcher_sha256": auditor.sha256_file(SCRIPTS / "run_d8_2_cv_parallel.py"),
        "train_core_sha256": auditor.sha256_file(SCRIPTS / "d8_train_core.py"),
        "shell_script_sha256": auditor.sha256_file(shell_path),
        "cache_root": str(root / "cache"),
        "cache_seal": "f" * 64,
        "python_environment": environment,
        "h1_lineage": lineage,
        "lineage_digest": lineage["lineage_digest"],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    jobs = []
    for seed in auditor.D8_3B_SEEDS:
        for fold in auditor.D8_3B_FOLDS:
            unit = root / f"seed{seed}" / f"B3_fold{fold}"
            unit.mkdir(parents=True)
            started = f"2026-08-03T00:00:{fold:02d}+00:00"
            finished = f"2026-08-03T00:01:{fold:02d}+00:00"
            provenance = {
                "config": "B3",
                "seed": seed,
                "fold": fold,
                "epochs": 100,
                "threshold": 0.0,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "weight_normalization": "mean_to_one",
                "cache_root": source["cache_root"],
                "cache_seal": source["cache_seal"],
                "source_commit": source["source_commit"],
                "source_tree": source["source_tree"],
                "unit_script_sha256": source["unit_script_sha256"],
                "parallel_launcher_sha256": source["parallel_launcher_sha256"],
                "train_core_sha256": source["train_core_sha256"],
                "python_environment": environment,
                "lineage_digest": source["lineage_digest"],
                "started_utc": started,
                "finished_utc": finished,
            }
            metrics = {
                "schema": "D8_3B_UNIT_METRICS_V2",
                "n": 2,
                "tp": 1,
                "tn": 1,
                "fp": 0,
                "fn": 0,
                "accuracy": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "auroc": 1.0,
                "balanced_accuracy": 1.0,
                "mcc": 1.0,
                **provenance,
            }
            (unit / "metrics.json").write_text(json.dumps(metrics))
            (unit / "predictions.json").write_text(
                json.dumps(
                    [
                        {"episode_id": "ep0", "step": 0, "target": 0, "logit": -1.0, "pred": 0},
                        {"episode_id": "ep1", "step": 0, "target": 1, "logit": 1.0, "pred": 1},
                    ]
                )
            )
            normalization = {
                "schema": "D8_NORMALIZATION_V2",
                "feature_dim": 25,
                "mean": [0.0] * 25,
                "std": [1.0] * 25,
            }
            torch.save(
                {
                    "schema": "D8_3B_CHECKPOINT_V2",
                    "model_state": {},
                    "normalization": normalization,
                    **provenance,
                },
                unit / "checkpoint.pt",
            )
            jobs.append(
                {
                    "job_id": f"B3_seed{seed}_fold{fold}",
                    "config": "B3",
                    "seed": seed,
                    "fold": fold,
                    "planned_index": len(jobs),
                    "status": "COMPLETED",
                    "gpu": 0,
                    "pid": 1,
                    "exit_code": 0,
                    "metrics_path": str(unit / "metrics.json"),
                    "checkpoint_path": str(unit / "checkpoint.pt"),
                    "predictions_path": str(unit / "predictions.json"),
                    "expected_provenance": provenance,
                }
            )
    manifest = {
        "schema": "D8_3B_JOB_MANIFEST_V2",
        "dispatcher_pid": 1,
        "created_utc": "2026-08-03T00:00:00+00:00",
        "matrix": {
            "configs": auditor.D8_3B_CONFIGS,
            "seeds": auditor.D8_3B_SEEDS,
            "folds": auditor.D8_3B_FOLDS,
            "epochs": 100,
            "gpus": [0, 1],
            "planned_jobs": 50,
        },
        "provenance": source,
        "abort_reason": None,
        "jobs": jobs,
    }
    (root / "JOB_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "D8_3B_SUMMARY.json").write_text(json.dumps({"verdict": "PASS"}))
    (root / "EXECUTION_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema": "synthetic",
                "matrix": manifest["matrix"],
                "provenance": source,
                "h1_lineage": lineage,
                "lineage_digest": lineage["lineage_digest"],
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "attack_rollouts": 0,
            }
        )
    )
    return root


def _first_job_path(root: Path, name: str) -> Path:
    return root / "seed20260720" / "B3_fold0" / name


def _edit_json(path: Path, edit) -> None:
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_manifest_and_receipt_lineage_must_match(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(
        root / "JOB_MANIFEST.json",
        lambda value: value["provenance"].update({"lineage_digest": "0" * 64}),
    )
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("lineage" in error for error in result["errors"])

    root = _make_auditor_fixture(tmp_path / "receipt")
    _edit_json(root / "EXECUTION_RECEIPT.json", lambda value: value.update({"lineage_digest": "0" * 64}))
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("lineage" in error for error in result["errors"])


def test_eval160_protected_and_attack_counters_must_be_zero(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(
        root / "JOB_MANIFEST.json",
        lambda value: value["provenance"].update({"attack_rollouts": 1}),
    )
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert "manifest:attack_rollouts_not_zero" in result["errors"]


@pytest.mark.parametrize("field", ["auroc", "balanced_accuracy", "mcc"])
def test_independent_metrics_reject_mismatch(tmp_path, field):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(_first_job_path(root, "metrics.json"), lambda value: value.update({field: 0.9}))
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any(f"metric_mismatch:{field}" in error for error in result["errors"])


def test_independent_metrics_reject_duplicate_episode_step(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(
        _first_job_path(root, "predictions.json"),
        lambda value: value.append(dict(value[0])),
    )
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("duplicate episode-step" in error for error in result["errors"])


def test_independent_metrics_reject_threshold_mismatch(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(
        _first_job_path(root, "predictions.json"),
        lambda value: value[0].update({"logit": 1.0}),
    )
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("threshold mismatch" in error for error in result["errors"])


@pytest.mark.parametrize("bad_logit", [float("nan"), float("inf"), float("-inf")])
def test_independent_metrics_reject_nonfinite_logit(tmp_path, bad_logit):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(
        _first_job_path(root, "predictions.json"),
        lambda value: value[0].update({"logit": bad_logit}),
    )
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("non-finite logit" in error for error in result["errors"])


def test_independent_metrics_reject_row_count_mismatch(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    _edit_json(_first_job_path(root, "metrics.json"), lambda value: value.update({"n": 3}))
    result = auditor.audit_run(root, write_artifacts=False)
    assert result["verdict"] == "FAIL"
    assert any("metric_mismatch:n" in error for error in result["errors"])


def test_final_readonly_auditor_failure_controls_exit_code(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    manifest = {
        "dispatcher_pid": 1,
        "created_utc": "2026-08-03T00:00:00+00:00",
        "matrix": {"gpus": [0, 1]},
        "jobs": _records(),
        "abort_reason": None,
    }
    receipt = {}
    calls = []

    def audit_fn(_root, *, write_artifacts):
        calls.append(write_artifacts)
        return {"verdict": "PASS" if write_artifacts else "FAIL"}

    code = dispatcher._finalize_run(
        root,
        manifest,
        receipt,
        abort_reason=None,
        audit_fn=audit_fn,
        seal_writer=lambda _root: None,
        seal_verifier=lambda _root: {},
    )
    assert code == 1
    assert True in calls and False in calls
    assert json.loads((root / "D8_3B_SUMMARY.json").read_text())["verdict"] == "FAIL"



def test_finalizer_emits_receipt_summary_manifest_audit_and_seal(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    manifest = json.loads((root / "JOB_MANIFEST.json").read_text())
    receipt = json.loads((root / "EXECUTION_RECEIPT.json").read_text())
    assert dispatcher._finalize_run(root, manifest, receipt, abort_reason=None) == 0
    assert {path.name for path in root.iterdir()} >= {
        "D8_3B_SUMMARY.json",
        "D8_3B_AUDIT.json",
        "EXECUTION_RECEIPT.json",
        "JOB_MANIFEST.json",
        "SHA256SUMS",
        "SHA256SUMS.sha256",
    }
    assert json.loads((root / "D8_3B_SUMMARY.json").read_text())["auditor_agreement"]
    assert auditor.verify_sha256_seal(root)["sha256sums_sha256"]


def test_independent_auditor_passes_and_detects_tamper(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    first = auditor.audit_run(root, write_artifacts=True)
    assert first["verdict"] == "PASS"
    auditor.write_sha256_seal(root)
    assert auditor.audit_run(root, write_artifacts=False)["verdict"] == "PASS"
    metrics_path = root / "seed20260720" / "B3_fold0" / "metrics.json"
    metrics_path.write_text(metrics_path.read_text().replace('"auroc": 1.0', '"auroc": 0.95'))
    tampered = auditor.audit_run(root, write_artifacts=False)
    assert tampered["verdict"] == "FAIL"
    assert tampered["errors"]


def _fake_python_for_shell(tmp_path: Path) -> Path:
    script = tmp_path / "fake-python.sh"
    script.write_text(
        """#!/usr/bin/env bash
set -u
MODE="__DOLLAR__{FAKE_D8_MODE:-success}"
ROOT=""
PREFLIGHT=0
for arg in "__DOLLAR__@"; do
    if [[ "__DOLLAR__{arg}" == "--preflight-only" ]]; then
        PREFLIGHT=1
    fi
done
for ((i=1; i<=__DOLLAR__#; i++)); do
    if [[ "__DOLLAR__{!i}" == "--output-root" ]]; then
        j=__DOLLAR__((i + 1))
        ROOT="__DOLLAR__{!j}"
    fi
done
mkdir -p "__DOLLAR__(dirname "__DOLLAR__{ROOT}")"
if (( PREFLIGHT == 1 )); then
    if [[ "__DOLLAR__{MODE}" == "preflight_fail" ]]; then
        echo "synthetic preflight failure" >&2
        exit 17
    fi
    if [[ "__DOLLAR__{MODE}" == "slow_preflight" ]]; then
        sleep 1.2
    fi
    printf '%s\n' '{"schema":"D8_3B_PREFLIGHT_V1","verdict":"PASS","run_root_created":false}'
    exit 0
fi
printf '%s\n' "$$" > "__DOLLAR__{ROOT}.fakepid"
printf '%s\n' formal >> "__DOLLAR__{ROOT}.formal_count"
trap 'printf "%s\n" terminated > "__DOLLAR__{ROOT}.terminated"; exit 143' TERM INT
write_artifacts() {
    local count="__DOLLAR__1"
    mkdir -p "__DOLLAR__{ROOT}"
    printf '%s\n' '{}' > "__DOLLAR__{ROOT}/EXECUTION_RECEIPT.json"
    {
        printf '%s\n' '{"matrix": {"planned_jobs": 50}, "jobs": ['
        for n in __DOLLAR__(seq 1 "__DOLLAR__{count}"); do
            printf '%s\n' "{\\\"job_id\\\": \\"job__DOLLAR__{n}\\\", \\"status\\\": \\"PENDING\\\"},"
        done
        printf '%s\n' '{}]}'
    } > "__DOLLAR__{ROOT}/JOB_MANIFEST.json"
}
case "__DOLLAR__{MODE}" in
    delay) sleep 1.5; write_artifacts 50; sleep 1 ;;
    success|slow_preflight) write_artifacts 50; sleep 1 ;;
    early_fail) echo "synthetic dispatcher early failure" >&2; exit 23 ;;
    timeout) sleep 20 ;;
    receipt_only) mkdir -p "__DOLLAR__{ROOT}"; printf '%s\n' '{}' > "__DOLLAR__{ROOT}/EXECUTION_RECEIPT.json"; sleep 20 ;;
    wrong_manifest) write_artifacts 49; sleep 20 ;;
esac
""".replace("__DOLLAR__", "$"),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _run_shell_launcher(tmp_path: Path, mode: str, *, timeout: int = 4):
    if not BASH.is_file():
        pytest.skip("requires Windows MSYS2 bash")
    fake_python = _fake_python_for_shell(tmp_path)
    cache = tmp_path / "cache"
    logs = tmp_path / "logs"
    shim_bin = tmp_path / "bin"
    cache.mkdir()
    logs.mkdir()
    shim_bin.mkdir()
    setsid_shim = shim_bin / "setsid"
    setsid_shim.write_text("#!/usr/bin/env bash\nexec \"$@\"\n", encoding="utf-8")
    setsid_shim.chmod(setsid_shim.stat().st_mode | stat.S_IXUSR)
    nohup_shim = shim_bin / "nohup"
    nohup_shim.write_text("#!/usr/bin/env bash\nexec \"$@\"\n", encoding="utf-8")
    nohup_shim.chmod(nohup_shim.stat().st_mode | stat.S_IXUSR)
    def msys_path(path: Path) -> str:
        value = path.as_posix()
        return f"/{value[0].lower()}{value[2:]}" if len(value) > 1 and value[1] == ":" else value

    env = os.environ.copy()
    env["PATH"] = ":".join([shim_bin.as_posix(), "/usr/bin", "/bin", "/c/Program Files/Git/cmd"])
    env.update(
        {
            "D8_CACHE_ROOT": msys_path(cache),
            "D8_CACHE_SEAL": "f" * 64,
            "D8_CACHE_A_SEAL": "f" * 64,
            "D8_CACHE_B_SEAL": "4" * 64,
            "D8_COMPARATOR_SEAL": "5" * 64,
            "D8_P5_ARTIFACT_SEAL": "6" * 64,
            "D8_H1_REVIEW_SEAL": "7" * 64,
            "D8_H1_SOURCE_COMMIT": "1" * 40,
            "D8_H1_SOURCE_TREE": "2" * 40,
            "D8_SOURCE_SNAPSHOT_SHA256": "3" * 64,
            "D8_EXPECTED_SOURCE_COMMIT": "a" * 40,
            "D8_EXPECTED_SOURCE_TREE": "b" * 40,
            "D8_LOG_ROOT": msys_path(logs),
            "D8_RUN_ROOT": msys_path(tmp_path / "run"),
            "D8_PYTHON_BIN": msys_path(fake_python),
            "D8_GPUS": "0",
            "D8_STARTUP_TIMEOUT_SECONDS": str(timeout),
            "D8_STARTUP_POLL_SECONDS": "0.1",
            "D8_STARTUP_GRACE_SECONDS": "1",
            "FAKE_D8_MODE": mode,
        }
    )
    result = subprocess.run(
        [str(BASH), str(SCRIPTS / "d8_launch_3b_safe.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, tmp_path / "run"


def _assert_shell_pid_dead(pid: int):
    probe = subprocess.run(
        [str(BASH), "-lc", f"kill -0 {int(pid)}"],
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0, probe.stderr


def test_shell_waits_for_late_receipt_and_manifest(tmp_path):
    result, root = _run_shell_launcher(tmp_path, "delay")
    assert result.returncode == 0, result.stderr
    assert (root / "EXECUTION_RECEIPT.json").is_file()
    assert (root / "JOB_MANIFEST.json").is_file()
    assert (root.with_suffix(".formal_count")).read_text().splitlines() == ["formal"]


def test_shell_slow_preflight_does_not_duplicate_formal_dispatch(tmp_path):
    result, root = _run_shell_launcher(tmp_path, "slow_preflight")
    assert result.returncode == 0, result.stderr
    assert (root.with_suffix(".formal_count")).read_text().splitlines() == ["formal"]


def test_shell_early_dispatcher_exit_is_nonzero_and_logs_tail(tmp_path):
    result, _root = _run_shell_launcher(tmp_path, "early_fail")
    assert result.returncode == 23
    assert "synthetic dispatcher early failure" in result.stderr


def test_shell_timeout_reaps_dispatcher_process_group(tmp_path):
    result, root = _run_shell_launcher(tmp_path, "timeout", timeout=2)
    assert result.returncode != 0
    assert root.with_suffix(".fakepid").is_file()
    pid = int(root.with_suffix(".fakepid").read_text())
    _assert_shell_pid_dead(pid)


def test_shell_receipt_without_manifest_is_not_startup_success(tmp_path):
    result, root = _run_shell_launcher(tmp_path, "receipt_only", timeout=2)
    assert result.returncode != 0
    assert (root / "EXECUTION_RECEIPT.json").is_file()
    assert not (root / "JOB_MANIFEST.json").is_file()
    _assert_shell_pid_dead(int(root.with_suffix(".fakepid").read_text()))


def test_shell_non_exact_manifest_is_not_startup_success(tmp_path):
    result, root = _run_shell_launcher(tmp_path, "wrong_manifest", timeout=2)
    assert result.returncode != 0
    assert (root / "EXECUTION_RECEIPT.json").is_file()
    assert (root / "JOB_MANIFEST.json").is_file()
    _assert_shell_pid_dead(int(root.with_suffix(".fakepid").read_text()))
