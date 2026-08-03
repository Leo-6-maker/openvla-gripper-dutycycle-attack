from __future__ import annotations

import io
import json
from pathlib import Path

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


def test_shell_separates_log_and_run_roots_and_requires_explicit_python():
    text = (SCRIPTS / "d8_launch_3b_safe.sh").read_text(encoding="utf-8")
    assert "D8_LOG_ROOT" in text
    assert "D8_RUN_ROOT" in text
    assert "D8_PYTHON_BIN" in text
    assert "which python" not in text
    assert '> "${DISPATCH_LOG}"' in text


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


def test_signal_terminates_inflight_and_records_nonzero(tmp_path):
    path, manifest = _dispatch_manifest(tmp_path, count=2)
    calls = 0
    processes = []

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


def _make_auditor_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    environment = {
        "python_version": "synthetic",
        "python_implementation": "CPython",
        "executable": "/synthetic/python",
        "torch_version": "synthetic",
        "cuda_version": None,
        "cuda_available": False,
        "numpy_version": "synthetic",
        "sklearn_version": "synthetic",
    }
    source = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "unit_script_sha256": "c" * 64,
        "parallel_launcher_sha256": "d" * 64,
        "train_core_sha256": "e" * 64,
        "cache_root": str(root / "cache"),
        "cache_seal": "f" * 64,
        "python_environment": environment,
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
                "started_utc": started,
                "finished_utc": finished,
            }
            metrics = {
                "schema": "D8_3B_UNIT_METRICS_V2",
                "auroc": 0.85,
                "balanced_accuracy": 0.75,
                "mcc": 0.1,
                **provenance,
            }
            (unit / "metrics.json").write_text(json.dumps(metrics))
            (unit / "predictions.json").write_text(json.dumps([{"target": 0, "pred": 0}]))
            torch.save(
                {"schema": "D8_3B_CHECKPOINT_V2", "model_state": {}, **provenance},
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
            }
        )
    )
    return root


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


def test_independent_auditor_passes_and_detects_tamper(tmp_path):
    root = _make_auditor_fixture(tmp_path)
    first = auditor.audit_run(root, write_artifacts=True)
    assert first["verdict"] == "PASS"
    auditor.write_sha256_seal(root)
    assert auditor.audit_run(root, write_artifacts=False)["verdict"] == "PASS"
    metrics_path = root / "seed20260720" / "B3_fold0" / "metrics.json"
    metrics_path.write_text(metrics_path.read_text().replace("0.85", "0.95"))
    tampered = auditor.audit_run(root, write_artifacts=False)
    assert tampered["verdict"] == "FAIL"
    assert tampered["errors"]
