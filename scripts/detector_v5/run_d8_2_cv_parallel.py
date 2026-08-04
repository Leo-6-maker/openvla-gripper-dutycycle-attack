"""Fail-closed D8-3B dispatcher.

This file only dispatches the frozen B3 matrix.  It does not select GPUs,
change training definitions, or authorize downstream evaluation/attack work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from audit_r3_contact_input import verify_seal

ROOT = Path(__file__).resolve().parents[2]

D8_3B_CONFIGS = ["B3"]
D8_3B_SEEDS = list(range(20260720, 20260730))
D8_3B_FOLDS = [0, 1, 2, 3, 4]
D8_3B_EPOCHS = 100
D8_3B_TOTAL_JOBS = 50
FOLDS = D8_3B_FOLDS

ALLOWED_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED_KILL_SWITCH",
    "ABORTED",
}
METRIC_FIELDS = ("auroc", "balanced_accuracy", "mcc")
H1_LINEAGE_FIELDS = (
    "h1_source_commit",
    "h1_source_tree",
    "source_snapshot_sha256",
    "cache_a_seal",
    "cache_b_seal",
    "comparator_seal",
    "p5_artifact_seal",
    "h1_review_seal",
)
PROVENANCE_FIELDS = (
    "config",
    "seed",
    "fold",
    "epochs",
    "threshold",
    "optimizer",
    "learning_rate",
    "weight_normalization",
    "cache_root",
    "cache_seal",
    "source_commit",
    "source_tree",
    "unit_script_sha256",
    "parallel_launcher_sha256",
    "train_core_sha256",
    "python_environment",
    "lineage_digest",
    "started_utc",
    "finished_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _split_csv(raw: str, cast: Callable[[str], Any] = str) -> list[Any]:
    if raw is None:
        raise ValueError("CSV value is required")
    pieces = [item.strip() for item in raw.split(",")]
    if not pieces or any(not item for item in pieces):
        raise ValueError(f"empty CSV item: {raw!r}")
    try:
        return [cast(item) for item in pieces]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid CSV value: {raw!r}") from exc


def validate_gpu_ids(gpu_ids: list[int]) -> list[int]:
    if not gpu_ids:
        raise ValueError("GPU list must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gpu_ids):
        raise ValueError(f"GPU IDs must be unique non-negative integers: {gpu_ids!r}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU IDs are forbidden: {gpu_ids!r}")
    return list(gpu_ids)


def validate_d8_3b_matrix(
    configs: list[str],
    seeds: list[int],
    folds: list[int],
    epochs: int,
    gpu_ids: list[int],
) -> int:
    """Validate every frozen matrix dimension before creating output or children."""

    if list(configs) != D8_3B_CONFIGS:
        raise ValueError(f"D8-3B requires configs exactly {D8_3B_CONFIGS!r}, got {configs!r}")
    if list(seeds) != D8_3B_SEEDS:
        raise ValueError(f"D8-3B requires seeds exactly {D8_3B_SEEDS!r}, got {seeds!r}")
    if list(folds) != D8_3B_FOLDS:
        raise ValueError(f"D8-3B requires folds exactly {D8_3B_FOLDS!r}, got {folds!r}")
    if epochs != D8_3B_EPOCHS:
        raise ValueError(f"D8-3B requires epochs={D8_3B_EPOCHS}, got {epochs!r}")
    validate_gpu_ids(gpu_ids)
    total_jobs = len(configs) * len(seeds) * len(folds)
    if total_jobs != D8_3B_TOTAL_JOBS:
        raise ValueError(f"D8-3B requires total_jobs={D8_3B_TOTAL_JOBS}, got {total_jobs}")
    return total_jobs


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def canonical_lineage_digest(lineage: Mapping[str, Any]) -> str:
    payload = {key: str(lineage[key]).lower() for key in H1_LINEAGE_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_h1_lineage(lineage: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(lineage, Mapping):
        raise ValueError("H1 lineage must be a mapping")
    for key in H1_LINEAGE_FIELDS:
        length = 40 if key in {"h1_source_commit", "h1_source_tree"} else 64
        if not _is_hex(lineage.get(key), length):
            raise ValueError(f"H1 lineage field is not a complete SHA: {key}")
    normalized = {key: str(lineage[key]).lower() for key in H1_LINEAGE_FIELDS}
    digest = canonical_lineage_digest(normalized)
    supplied = lineage.get("lineage_digest")
    if supplied is not None and supplied != digest:
        raise ValueError("H1 lineage digest mismatch")
    normalized["lineage_digest"] = digest
    return normalized


def validate_expected_source(
    expected_commit: str,
    expected_tree: str,
    actual: Mapping[str, str],
) -> dict[str, str]:
    if not _is_hex(expected_commit, 40) or not _is_hex(expected_tree, 40):
        raise ValueError("expected source commit/tree must be complete 40-character SHA-1 values")
    if actual.get("source_commit") != expected_commit.lower():
        raise RuntimeError(
            f"expected source commit mismatch: {expected_commit} != {actual.get('source_commit')}"
        )
    if actual.get("source_tree") != expected_tree.lower():
        raise RuntimeError(
            f"expected source tree mismatch: {expected_tree} != {actual.get('source_tree')}"
        )
    return {
        "expected_source_commit": expected_commit.lower(),
        "expected_source_tree": expected_tree.lower(),
    }


def validate_cuda_environment(environment: Mapping[str, Any], gpu_ids: list[int]) -> None:
    validate_gpu_ids(gpu_ids)
    if environment.get("cuda_available") is not True:
        raise RuntimeError("CUDA is not available; refusing D8-3B launch")
    count = environment.get("cuda_device_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= max(gpu_ids):
        raise RuntimeError(
            f"CUDA device count {count!r} cannot cover selected physical GPUs {gpu_ids!r}"
        )
    inherited = environment.get("inherited_CUDA_VISIBLE_DEVICES", "")
    if inherited:
        raise RuntimeError(
            "inherited CUDA_VISIBLE_DEVICES is non-empty; refusing implicit GPU remapping"
        )


def validate_cache_seal(cache_root: Path, expected_seal: str) -> str:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root is not a directory: {cache_root}")
    try:
        actual = str(verify_seal(cache_root)["sha256sums_sha256"]).lower()
    except Exception as exc:
        raise RuntimeError(f"Cache A seal verification failed: {cache_root}") from exc
    if actual != expected_seal.lower():
        raise RuntimeError(f"Cache A seal mismatch: expected {expected_seal.lower()}, got {actual}")
    return actual


def validate_python_environment(python_bin: str, gpu_ids: list[int]) -> dict[str, Any]:
    if not python_bin:
        raise ValueError("--python-bin or D8_PYTHON_BIN is required")
    executable = Path(python_bin).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError(f"python executable must be an existing absolute file: {python_bin!r}")
    executable = executable.resolve()
    probe = (
        "import json, os, platform, sys\n"
        "import numpy, sklearn, torch\n"
        "print(json.dumps({"
        "'python_version': sys.version,"
        "'python_implementation': platform.python_implementation(),"
        "'executable': os.path.abspath(sys.executable),"
        "'torch_version': torch.__version__,"
        "'cuda_version': torch.version.cuda,"
        "'cuda_available': bool(torch.cuda.is_available()),"
        "'cuda_device_count': int(torch.cuda.device_count()),"
        "'inherited_CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),"
        "'numpy_version': numpy.__version__,"
        "'sklearn_version': sklearn.__version__"
        "}, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", probe],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"python environment probe failed: {executable}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"python environment probe returned {result.returncode}: {result.stderr[-1000:]}"
        )
    try:
        environment = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("python environment probe did not return JSON") from exc
    if environment.get("executable") != str(executable):
        raise RuntimeError(
            f"python executable binding mismatch: {environment.get('executable')!r} != {str(executable)!r}"
        )
    required = (
        "python_version",
        "torch_version",
        "cuda_version",
        "cuda_available",
        "cuda_device_count",
        "inherited_CUDA_VISIBLE_DEVICES",
        "numpy_version",
        "sklearn_version",
        "executable",
    )
    if any(key not in environment for key in required):
        raise RuntimeError(f"python environment probe missing keys: {required!r}")
    validate_cuda_environment(environment, gpu_ids)
    return environment


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git provenance failed: {args!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_provenance() -> dict[str, str]:
    if _git_value("status", "--porcelain"):
        raise RuntimeError("source worktree is dirty; refusing to launch")
    commit = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    if not _is_hex(commit, 40) or not _is_hex(tree, 40):
        raise RuntimeError("source commit/tree are not full SHA-1 values")
    return {"source_commit": commit, "source_tree": tree}


def _job_identity(config: str, seed: int, fold: int) -> str:
    return f"{config}_seed{seed}_fold{fold}"


def _expected_provenance(
    *,
    config: str,
    seed: int,
    fold: int,
    epochs: int,
    cache_root: str,
    cache_seal: str,
    source_commit: str,
    source_tree: str,
    unit_script_sha256: str,
    parallel_launcher_sha256: str,
    train_core_sha256: str,
    python_environment: Mapping[str, Any],
    lineage_digest: str,
    started_utc: str,
    finished_utc: str | None,
) -> dict[str, Any]:
    return {
        "config": config,
        "seed": seed,
        "fold": fold,
        "epochs": epochs,
        "threshold": 0.0,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_normalization": "mean_to_one",
        "cache_root": cache_root,
        "cache_seal": cache_seal,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "unit_script_sha256": unit_script_sha256,
        "parallel_launcher_sha256": parallel_launcher_sha256,
        "train_core_sha256": train_core_sha256,
        "python_environment": dict(python_environment),
        "lineage_digest": lineage_digest,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }


def build_jobs(
    *,
    configs: list[str],
    seeds: list[int],
    folds: list[int],
    gpu_ids: list[int],
    epochs: int,
    cache_root: str,
    cache_seal: str,
    output_root: Path,
    python_bin: str,
    source_commit: str,
    source_tree: str,
    python_environment: Mapping[str, Any],
    lineage_digest: str,
) -> list[dict[str, Any]]:
    unit_script = ROOT / "scripts" / "detector_v5" / "run_d8_2_cv_unit.py"
    launcher_script = Path(__file__).resolve()
    train_core = ROOT / "scripts" / "detector_v5" / "d8_train_core.py"
    unit_sha = sha256_file(unit_script)
    launcher_sha = sha256_file(launcher_script)
    train_core_sha = sha256_file(train_core)
    receipt_path = output_root / "EXECUTION_RECEIPT.json"
    jobs: list[dict[str, Any]] = []
    planned_index = 0
    for seed in seeds:
        for config in configs:
            for fold in folds:
                unit_dir = output_root / f"seed{seed}" / f"{config}_fold{fold}"
                metrics_path = unit_dir / "metrics.json"
                checkpoint_path = unit_dir / "checkpoint.pt"
                predictions_path = unit_dir / "predictions.json"
                log_path = unit_dir / "train.log"
                command = [
                    str(Path(python_bin).resolve()),
                    "-u",
                    str(unit_script),
                    "--cache-root",
                    cache_root,
                    "--config",
                    config,
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(epochs),
                    "--output-dir",
                    str(unit_dir),
                    "--expected-cache-seal",
                    cache_seal,
                    "--source-commit",
                    source_commit,
                    "--source-tree",
                    source_tree,
                    "--launcher-sha256",
                    launcher_sha,
                    "--train-core-sha256",
                    train_core_sha,
                    "--execution-receipt",
                    str(receipt_path),
                ]
                started_utc = utc_now()
                job = {
                    "job_id": _job_identity(config, seed, fold),
                    "config": config,
                    "seed": seed,
                    "fold": fold,
                    "planned_index": planned_index,
                    "status": "PENDING",
                    "gpu": None,
                    "planned_gpu": gpu_ids[planned_index % len(gpu_ids)],
                    "pid": None,
                    "command": command,
                    "started_utc": None,
                    "finished_utc": None,
                    "exit_code": None,
                    "metrics_path": str(metrics_path),
                    "checkpoint_path": str(checkpoint_path),
                    "predictions_path": str(predictions_path),
                    "log_path": str(log_path),
                    "failure_reason": None,
                    "expected_provenance": _expected_provenance(
                        config=config,
                        seed=seed,
                        fold=fold,
                        epochs=epochs,
                        cache_root=cache_root,
                        cache_seal=cache_seal,
                        source_commit=source_commit,
                        source_tree=source_tree,
                        unit_script_sha256=unit_sha,
                        parallel_launcher_sha256=launcher_sha,
                        train_core_sha256=train_core_sha,
                        python_environment=python_environment,
                        lineage_digest=lineage_digest,
                        started_utc=started_utc,
                        finished_utc=None,
                    ),
                }
                jobs.append(job)
                planned_index += 1
    return jobs


def new_manifest(
    *,
    jobs: list[dict[str, Any]],
    configs: list[str],
    seeds: list[int],
    folds: list[int],
    epochs: int,
    gpu_ids: list[int],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "D8_3B_JOB_MANIFEST_V2",
        "state_machine": sorted(ALLOWED_STATES),
        "dispatcher_pid": os.getpid(),
        "created_utc": utc_now(),
        "finished_utc": None,
        "abort_reason": None,
        "verdict": None,
        "matrix": {
            "configs": list(configs),
            "seeds": list(seeds),
            "folds": list(folds),
            "epochs": epochs,
            "gpus": list(gpu_ids),
            "planned_jobs": len(jobs),
        },
        "provenance": dict(source),
        "jobs": jobs,
    }


def _check_path(path_value: str, expected: Path) -> bool:
    path = Path(path_value)
    return path == expected and not path.is_symlink() and path.is_file()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def validate_job_artifacts(job: Mapping[str, Any], returncode: int) -> tuple[bool, str, dict[str, Any]]:
    """Return strict completion status; nonzero children never become complete."""

    if returncode != 0:
        return False, f"nonzero_returncode:{returncode}", {}
    identity = f"{job['config']}_seed{job['seed']}_fold{job['fold']}"
    unit_dir = Path(job["metrics_path"]).parent
    expected_paths = {
        "metrics_path": unit_dir / "metrics.json",
        "checkpoint_path": unit_dir / "checkpoint.pt",
        "predictions_path": unit_dir / "predictions.json",
    }
    for field, expected in expected_paths.items():
        if not _check_path(str(job[field]), expected):
            return False, f"invalid_or_missing_{field}", {}
    try:
        metrics = json.loads(Path(job["metrics_path"]).read_text(encoding="utf-8"))
        predictions = json.loads(Path(job["predictions_path"]).read_text(encoding="utf-8"))
        checkpoint = _load_checkpoint(Path(job["checkpoint_path"]))
    except Exception as exc:
        return False, f"artifact_parse_error:{type(exc).__name__}", {}
    if not isinstance(metrics, dict) or not isinstance(predictions, list) or not isinstance(checkpoint, Mapping):
        return False, "artifact_types_invalid", {}
    required = {"schema", *METRIC_FIELDS, *PROVENANCE_FIELDS}
    missing = sorted(key for key in required if key not in metrics)
    if missing:
        return False, f"metrics_missing:{','.join(missing)}", {}
    if metrics.get("schema") != "D8_3B_UNIT_METRICS_V2":
        return False, "metrics_schema_mismatch", {}
    expected = dict(job["expected_provenance"])
    for key, expected_value in expected.items():
        if key in {"started_utc", "finished_utc"}:
            if not isinstance(metrics.get(key), str) or not metrics.get(key):
                return False, f"metrics_timestamp_missing:{key}", {}
            continue
        if metrics.get(key) != expected_value:
            return False, f"metrics_provenance_mismatch:{key}", {}
    for key in METRIC_FIELDS:
        try:
            if not math.isfinite(float(metrics[key])):
                return False, f"nonfinite_metric:{key}", {}
        except (TypeError, ValueError):
            return False, f"invalid_metric:{key}", {}
    for key, expected_value in expected.items():
        expected_value = metrics[key] if key in {"started_utc", "finished_utc"} else expected_value
        if checkpoint.get(key) != expected_value:
            return False, f"checkpoint_provenance_mismatch:{key}", {}
    if checkpoint.get("schema") != "D8_3B_CHECKPOINT_V2" or "model_state" not in checkpoint:
        return False, "checkpoint_schema_or_model_missing", {}
    if metrics.get("config") != job["config"] or metrics.get("seed") != job["seed"] or metrics.get("fold") != job["fold"]:
        return False, f"identity_mismatch:{identity}", {}
    return True, "ok", metrics


def _metrics_for_gate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    inline = job.get("metrics")
    if isinstance(inline, dict):
        return inline
    path = Path(str(job.get("metrics_path", "")))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def final_gate(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (config, seed, fold)
        for config in D8_3B_CONFIGS
        for seed in D8_3B_SEEDS
        for fold in D8_3B_FOLDS
    }
    identities = [
        (record.get("config"), record.get("seed"), record.get("fold"))
        for record in records
    ]
    identity_set = set(identities)
    duplicates = len(identities) != len(identity_set)
    unexpected = sorted(identity_set - expected, key=str)
    missing = sorted(expected - identity_set, key=str)
    completed = [record for record in records if record.get("status") == "COMPLETED"]
    failed = [record for record in records if record.get("status") == "FAILED"]
    skipped = [record for record in records if record.get("status") == "SKIPPED_KILL_SWITCH"]
    aborted = [record for record in records if record.get("status") == "ABORTED"]
    seed_summary: dict[str, dict[str, Any]] = {}
    all_metrics_finite = True
    for seed in D8_3B_SEEDS:
        rows = [
            record
            for record in completed
            if record.get("config") == "B3" and record.get("seed") == seed
        ]
        values: dict[str, list[float]] = {key: [] for key in METRIC_FIELDS}
        for row in rows:
            metrics = _metrics_for_gate(row)
            if metrics is None:
                all_metrics_finite = False
                continue
            for key in METRIC_FIELDS:
                try:
                    value = float(metrics[key])
                except (KeyError, TypeError, ValueError):
                    all_metrics_finite = False
                    continue
                if not math.isfinite(value):
                    all_metrics_finite = False
                values[key].append(value)
        if len(rows) == len(D8_3B_FOLDS) and all(len(values[key]) == len(D8_3B_FOLDS) for key in METRIC_FIELDS):
            seed_summary[str(seed)] = {
                "folds_completed": len(rows),
                "mean_auroc": float(np.mean(values["auroc"])),
                "mean_bacc": float(np.mean(values["balanced_accuracy"])),
                "mean_mcc": float(np.mean(values["mcc"])),
                "auroc_std_folds": float(np.std(values["auroc"], ddof=1)),
            }
        else:
            seed_summary[str(seed)] = {
                "folds_completed": len(rows),
                "mean_auroc": float("nan"),
                "mean_bacc": float("nan"),
                "mean_mcc": float("nan"),
                "auroc_std_folds": float("nan"),
            }
    seed_means = [seed_summary[str(seed)]["mean_auroc"] for seed in D8_3B_SEEDS]
    stability_std = float(np.std(seed_means, ddof=1)) if len(seed_means) == 10 and all(math.isfinite(value) for value in seed_means) else float("nan")
    per_seed_pass = all(
        math.isfinite(seed_summary[str(seed)]["mean_auroc"])
        and math.isfinite(seed_summary[str(seed)]["mean_bacc"])
        and math.isfinite(seed_summary[str(seed)]["mean_mcc"])
        and seed_summary[str(seed)]["mean_auroc"] >= 0.80
        and seed_summary[str(seed)]["mean_bacc"] >= 0.70
        and seed_summary[str(seed)]["mean_mcc"] > 0.0
        for seed in D8_3B_SEEDS
    )
    checks = {
        "planned_jobs_50": len(records) == D8_3B_TOTAL_JOBS,
        "completed_50": len(completed) == D8_3B_TOTAL_JOBS,
        "failed_0": len(failed) == 0,
        "skipped_0": len(skipped) == 0,
        "aborted_0": len(aborted) == 0,
        "identity_exact": not duplicates and not missing and not unexpected,
        "seed_coverage_10": all(seed_summary[str(seed)]["folds_completed"] == 5 for seed in D8_3B_SEEDS),
        "config_b3_only": all(record.get("config") == "B3" for record in records),
        "metrics_finite": all_metrics_finite,
        "per_seed_metrics": per_seed_pass,
        "auroc_sample_std_le_003": math.isfinite(stability_std) and stability_std <= 0.03,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "planned_jobs": len(records),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "skipped_jobs": len(skipped),
        "aborted_jobs": len(aborted),
        "missing_identities": [list(item) for item in missing],
        "unexpected_identities": [list(item) for item in unexpected],
        "duplicate_identity": duplicates,
        "seed_summary": seed_summary,
        "seed_means_auroc": seed_means,
        "stability_std_ddof1": stability_std,
        "stability_threshold": 0.03,
    }


def _signal_process(proc: subprocess.Popen, kill: bool = False) -> None:
    requested = signal.SIGKILL if kill else signal.SIGTERM
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), requested)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        (proc.kill if kill else proc.terminate)()
    except (OSError, ProcessLookupError):
        pass


def launch_unit(job: Mapping[str, Any]) -> tuple[subprocess.Popen, Any]:
    log_path = Path(str(job["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    gpu = job.get("gpu")
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "scripts" / "detector_v5"),
            str(ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    kwargs: dict[str, Any] = {
        "env": environment,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(ROOT),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(list(job["command"]), **kwargs)
    except Exception:
        log_handle.close()
        raise
    return process, log_handle


def _close_log(handle: Any) -> None:
    try:
        handle.close()
    except Exception:
        pass


def dispatch_jobs(
    manifest_path: Path,
    stop_path: Path,
    manifest: dict[str, Any],
    *,
    launch_fn: Callable[[Mapping[str, Any]], tuple[Any, Any]] = launch_unit,
    validate_fn: Callable[[Mapping[str, Any], int], tuple[bool, str, dict[str, Any]]] = validate_job_artifacts,
    abort_reason_fn: Callable[[], str | None] = lambda: None,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.25,
    job_timeout_seconds: float = 24 * 60 * 60,
    grace_seconds: float = 30.0,
) -> dict[str, Any]:
    jobs = manifest["jobs"]
    gpu_ids = list(manifest["matrix"]["gpus"])
    slots: dict[int, dict[str, Any]] = {}
    abort_reason: str | None = None

    def persist() -> None:
        atomic_write_json(manifest_path, manifest)

    def mark_abort(reason: str) -> None:
        nonlocal abort_reason
        if abort_reason is not None:
            return
        abort_reason = reason
        manifest["abort_reason"] = reason
        pending_state = "SKIPPED_KILL_SWITCH" if reason == "KILL_SWITCH" else "ABORTED"
        for job in jobs:
            if job["status"] == "PENDING":
                job["status"] = pending_state
                job["finished_utc"] = utc_now()
                job["failure_reason"] = reason
        persist()
        for slot in slots.values():
            _signal_process(slot["process"], kill=False)
        deadline = time.monotonic() + max(grace_seconds, 0.0)
        while slots and time.monotonic() < deadline:
            for gpu, slot in list(slots.items()):
                if slot["process"].poll() is not None:
                    _close_log(slot["log_handle"])
                    job = slot["job"]
                    job["status"] = "ABORTED"
                    job["finished_utc"] = utc_now()
                    job["exit_code"] = slot["process"].poll()
                    job["failure_reason"] = reason
                    del slots[gpu]
            if slots:
                sleep_fn(0.1)
        for gpu, slot in list(slots.items()):
            process = slot["process"]
            if process.poll() is None:
                _signal_process(process, kill=True)
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
            _close_log(slot["log_handle"])
            job = slot["job"]
            job["status"] = "ABORTED"
            job["finished_utc"] = utc_now()
            job["exit_code"] = process.poll()
            job["failure_reason"] = f"{reason}:terminated"
            del slots[gpu]
        persist()

    try:
        while True:
            if abort_reason is None:
                external_reason = abort_reason_fn()
                if external_reason:
                    mark_abort(external_reason)
                elif stop_path.exists():
                    mark_abort("KILL_SWITCH")
            if abort_reason is not None:
                break

            changed = False
            for gpu, slot in list(slots.items()):
                process = slot["process"]
                elapsed = time.monotonic() - slot["started_monotonic"]
                if process.poll() is None and elapsed > job_timeout_seconds:
                    mark_abort(f"WATCHDOG_TIMEOUT:{slot['job']['job_id']}")
                    break
                returncode = process.poll()
                if returncode is None:
                    continue
                _close_log(slot["log_handle"])
                job = slot["job"]
                ok, reason, metrics = validate_fn(job, returncode)
                job["status"] = "COMPLETED" if ok else "FAILED"
                job["finished_utc"] = utc_now()
                job["exit_code"] = returncode
                job["failure_reason"] = None if ok else reason
                if metrics:
                    job["metrics"] = metrics
                del slots[gpu]
                changed = True
            if abort_reason is not None:
                break
            if changed:
                persist()

            pending = [job for job in jobs if job["status"] == "PENDING"]
            while pending and len(slots) < len(gpu_ids):
                external_reason = abort_reason_fn()
                if external_reason:
                    mark_abort(external_reason)
                    break
                if stop_path.exists():
                    mark_abort("KILL_SWITCH")
                    break
                available = [gpu for gpu in gpu_ids if gpu not in slots]
                if not available:
                    break
                job = pending.pop(0)
                gpu = available[0]
                job["gpu"] = gpu
                try:
                    process, log_handle = launch_fn(job)
                except Exception as exc:
                    job["status"] = "FAILED"
                    job["gpu"] = gpu
                    job["finished_utc"] = utc_now()
                    job["failure_reason"] = f"launch_error:{type(exc).__name__}"
                    job["exit_code"] = None
                    persist()
                    pending = [item for item in jobs if item["status"] == "PENDING"]
                    continue
                job["status"] = "RUNNING"
                job["pid"] = process.pid
                job["started_utc"] = utc_now()
                slots[gpu] = {
                    "job": job,
                    "process": process,
                    "log_handle": log_handle,
                    "started_monotonic": time.monotonic(),
                }
                persist()
                pending = [item for item in jobs if item["status"] == "PENDING"]
            if abort_reason is not None:
                break
            if not slots and not any(job["status"] == "PENDING" for job in jobs):
                break
            sleep_fn(poll_interval)
    except Exception as exc:
        if abort_reason is None:
            mark_abort(f"DISPATCHER_EXCEPTION:{type(exc).__name__}")
    finally:
        for slot in slots.values():
            _signal_process(slot["process"], kill=True)
            _close_log(slot["log_handle"])
        slots.clear()
        for job in jobs:
            if job["status"] in {"PENDING", "RUNNING"}:
                job["status"] = "ABORTED"
                job["finished_utc"] = utc_now()
                job["failure_reason"] = abort_reason or "DISPATCHER_FINALIZE"
        persist()
    return manifest


def _write_hash_seal(root: Path) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    content = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    atomic_write_text(root / "SHA256SUMS", content)
    atomic_write_text(
        root / "SHA256SUMS.sha256",
        f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n",
    )


def _finalize_run(
    output_root: Path,
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    *,
    abort_reason: str | None,
    audit_fn: Callable[..., dict[str, Any]] | None = None,
    seal_writer: Callable[[Path], None] | None = None,
    seal_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
) -> int:
    gate = final_gate(manifest["jobs"])
    manifest["finished_utc"] = utc_now()
    manifest["gate"] = gate
    gate_verdict = (
        "ABORTED_INCOMPLETE"
        if abort_reason
        else ("PASS" if gate["pass"] else "FAIL")
    )
    summary = {
        "schema": "D8_3B_SUMMARY_V2",
        "verdict": gate_verdict,
        "abort_reason": abort_reason,
        "matrix": manifest["matrix"],
        "gate": gate,
        "dispatcher_pid": manifest["dispatcher_pid"],
        "created_utc": manifest["created_utc"],
        "finished_utc": manifest["finished_utc"],
        "auditor_verdict": None,
        "auditor_agreement": False,
    }
    receipt["state"] = "PROVISIONAL"

    def write_bundle(verdict: str, audit_verdict: str | None, agreement: bool) -> None:
        manifest["verdict"] = verdict
        manifest["auditor_verdict"] = audit_verdict
        manifest["auditor_agreement"] = agreement
        summary["verdict"] = verdict
        summary["auditor_verdict"] = audit_verdict
        summary["auditor_agreement"] = agreement
        receipt.update(
            {
                "state": "FINISHED",
                "verdict": verdict,
                "abort_reason": abort_reason,
                "finished_utc": manifest["finished_utc"],
                "auditor_verdict": audit_verdict,
                "auditor_agreement": agreement,
                "exit_code": 0 if verdict == "PASS" else 1,
            }
        )
        atomic_write_json(output_root / "JOB_MANIFEST.json", manifest)
        atomic_write_json(output_root / "D8_3B_SUMMARY.json", summary)
        atomic_write_json(output_root / "EXECUTION_RECEIPT.json", receipt)

    atomic_write_json(output_root / "JOB_MANIFEST.json", manifest)
    atomic_write_json(output_root / "D8_3B_SUMMARY.json", summary)
    atomic_write_json(output_root / "EXECUTION_RECEIPT.json", receipt)

    if audit_fn is None:
        try:
            from audit_d8_3b_run import audit_run
        except Exception as exc:
            error_name = type(exc).__name__
            audit_fn = lambda _root, **_kwargs: {
                "verdict": "FAIL",
                "errors": [f"auditor_import:{error_name}"],
            }
        else:
            audit_fn = audit_run
    if seal_writer is None or seal_verifier is None:
        try:
            from audit_d8_3b_run import verify_sha256_seal, write_sha256_seal
        except Exception:
            seal_writer = seal_writer or _write_hash_seal
            seal_verifier = seal_verifier or (lambda _root: {"fallback": True})
        else:
            seal_writer = seal_writer or write_sha256_seal
            seal_verifier = seal_verifier or verify_sha256_seal

    try:
        first_audit = audit_fn(output_root, write_artifacts=True)
    except Exception as exc:
        first_audit = {"verdict": "FAIL", "errors": [f"auditor_exception:{type(exc).__name__}"]}
        atomic_write_json(output_root / "D8_3B_AUDIT.json", first_audit)

    def resolve_verdict(audit_verdict: Any) -> str:
        if abort_reason:
            return "ABORTED_INCOMPLETE"
        return "PASS" if gate["pass"] and audit_verdict == "PASS" else "FAIL"

    current_verdict = resolve_verdict(first_audit.get("verdict"))
    current_audit = first_audit
    for _ in range(3):
        write_bundle(
            current_verdict,
            current_audit.get("verdict"),
            current_audit.get("verdict") == current_verdict,
        )
        try:
            audit_fn(output_root, write_artifacts=True)
            readonly_audit = audit_fn(output_root, write_artifacts=False)
        except Exception as exc:
            readonly_audit = {
                "verdict": "FAIL",
                "errors": [f"final_auditor_exception:{type(exc).__name__}"],
            }
        desired_verdict = resolve_verdict(readonly_audit.get("verdict"))
        write_bundle(
            desired_verdict,
            readonly_audit.get("verdict"),
            readonly_audit.get("verdict") == desired_verdict,
        )
        try:
            final_audit = audit_fn(output_root, write_artifacts=False)
        except Exception as exc:
            final_audit = {
                "verdict": "FAIL",
                "errors": [f"post_summary_auditor_exception:{type(exc).__name__}"],
            }
        stable_verdict = resolve_verdict(final_audit.get("verdict"))
        if stable_verdict == desired_verdict:
            write_bundle(
                stable_verdict,
                final_audit.get("verdict"),
                final_audit.get("verdict") == stable_verdict,
            )
            current_verdict = stable_verdict
            current_audit = final_audit
            break
        current_verdict = stable_verdict
        current_audit = final_audit
    else:
        write_bundle(
            current_verdict,
            current_audit.get("verdict"),
            current_audit.get("verdict") == current_verdict,
        )

    try:
        seal_writer(output_root)
        seal_verifier(output_root)
    except Exception:
        return 1
    return 0 if current_verdict == "PASS" and current_audit.get("verdict") == "PASS" else 1


def _run_preflight(
    args: argparse.Namespace,
    configs: list[str],
    seeds: list[int],
    gpu_ids: list[int],
) -> dict[str, Any]:
    validate_d8_3b_matrix(configs, seeds, FOLDS, args.epochs, gpu_ids)
    if not _is_hex(args.cache_seal, 64):
        raise ValueError("--cache-seal or D8_CACHE_SEAL must be a 64-character SHA256")
    source = git_provenance()
    expected_source = validate_expected_source(
        args.expected_source_commit,
        args.expected_source_tree,
        source,
    )
    lineage = validate_h1_lineage(
        {
            "h1_source_commit": args.h1_source_commit,
            "h1_source_tree": args.h1_source_tree,
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "cache_a_seal": args.cache_a_seal,
            "cache_b_seal": args.cache_b_seal,
            "comparator_seal": args.comparator_seal,
            "p5_artifact_seal": args.p5_artifact_seal,
            "h1_review_seal": args.h1_review_seal,
        }
    )
    if lineage["cache_a_seal"] != args.cache_seal.lower():
        raise RuntimeError(
            f"H1 cache_a_seal mismatch: {lineage['cache_a_seal']} != {args.cache_seal.lower()}"
        )
    if not _is_hex(args.shell_script_sha256, 64):
        raise ValueError("--shell-script-sha256 or D8_SHELL_SCRIPT_SHA256 must be a 64-character SHA256")
    python_environment = validate_python_environment(args.python_bin, gpu_ids)

    cache_root = args.cache_root.resolve()
    actual_cache_seal = validate_cache_seal(cache_root, args.cache_seal)
    log_root = args.log_root.resolve()
    if not log_root.is_dir():
        raise FileNotFoundError(f"log root must pre-exist: {log_root}")
    output_root = args.output_root.resolve()
    if output_root == log_root:
        raise ValueError("run root and dispatcher log root must be different paths")
    if output_root.exists():
        raise FileExistsError(f"run root already exists; refusing to clobber: {output_root}")

    unit_script_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "run_d8_2_cv_unit.py")
    launcher_sha = sha256_file(Path(__file__).resolve())
    train_core_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "d8_train_core.py")
    source_receipt = {
        "source_commit": source["source_commit"],
        "source_tree": source["source_tree"],
        **expected_source,
        "unit_script_sha256": unit_script_sha,
        "parallel_launcher_sha256": launcher_sha,
        "train_core_sha256": train_core_sha,
        "shell_script_sha256": args.shell_script_sha256.lower(),
        "cache_root": str(cache_root),
        "cache_seal": actual_cache_seal,
        "python_environment": python_environment,
        "h1_lineage": lineage,
        "lineage_digest": lineage["lineage_digest"],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    return {
        "cache_root": cache_root,
        "cache_seal": actual_cache_seal,
        "log_root": log_root,
        "output_root": output_root,
        "source_receipt": source_receipt,
        "lineage": lineage,
        "python_environment": python_environment,
    }


def _preflight_report(
    args: argparse.Namespace,
    configs: list[str],
    seeds: list[int],
    gpu_ids: list[int],
    *,
    context: Mapping[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "D8_3B_PREFLIGHT_V1",
        "verdict": "PASS" if error is None else "FAIL",
        "run_root": str(args.output_root.resolve()),
        "log_root": str(args.log_root.resolve()),
        "run_root_created": False,
        "manifest_created": False,
        "children_launched": 0,
        "gpu_training": 0,
        "matrix": {
            "configs": configs,
            "seeds": seeds,
            "folds": FOLDS,
            "epochs": args.epochs,
            "gpus": gpu_ids,
            "planned_jobs": D8_3B_TOTAL_JOBS,
        },
    }
    if context is not None:
        report["provenance"] = context["source_receipt"]
    if error is not None:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-seal", default=os.environ.get("D8_CACHE_SEAL"))
    parser.add_argument("--cache-a-seal", default=os.environ.get("D8_CACHE_A_SEAL"))
    parser.add_argument("--cache-b-seal", default=os.environ.get("D8_CACHE_B_SEAL"))
    parser.add_argument("--comparator-seal", default=os.environ.get("D8_COMPARATOR_SEAL"))
    parser.add_argument("--p5-artifact-seal", default=os.environ.get("D8_P5_ARTIFACT_SEAL"))
    parser.add_argument("--h1-review-seal", default=os.environ.get("D8_H1_REVIEW_SEAL"))
    parser.add_argument("--h1-source-commit", default=os.environ.get("D8_H1_SOURCE_COMMIT"))
    parser.add_argument("--h1-source-tree", default=os.environ.get("D8_H1_SOURCE_TREE"))
    parser.add_argument("--source-snapshot-sha256", default=os.environ.get("D8_SOURCE_SNAPSHOT_SHA256"))
    parser.add_argument("--expected-source-commit", default=os.environ.get("D8_EXPECTED_SOURCE_COMMIT"))
    parser.add_argument("--expected-source-tree", default=os.environ.get("D8_EXPECTED_SOURCE_TREE"))
    parser.add_argument("--shell-script-sha256", default=os.environ.get("D8_SHELL_SCRIPT_SHA256"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--python-bin", default=os.environ.get("D8_PYTHON_BIN"))
    parser.add_argument("--gpus", default=os.environ.get("D8_GPUS"))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--job-timeout-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--terminate-grace-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    configs: list[str] = []
    seeds: list[int] = []
    gpu_ids: list[int] = []
    try:
        configs = _split_csv(args.configs)
        seeds = _split_csv(args.seeds, int)
        gpu_ids = validate_gpu_ids(_split_csv(args.gpus, int) if args.gpus else [])
        context = _run_preflight(args, configs, seeds, gpu_ids)
    except Exception as exc:
        if args.preflight_only:
            print(
                json.dumps(
                    _preflight_report(
                        args,
                        configs,
                        seeds,
                        gpu_ids,
                        error=exc,
                    ),
                    sort_keys=True,
                )
            )
            return 1
        raise
    if args.preflight_only:
        print(json.dumps(_preflight_report(args, configs, seeds, gpu_ids, context=context), sort_keys=True))
        return 0

    cache_root = context["cache_root"]
    actual_cache_seal = context["cache_seal"]
    log_root = context["log_root"]
    output_root = context["output_root"]
    source_receipt = context["source_receipt"]
    python_environment = context["python_environment"]
    lineage = context["lineage"]
    started_utc = utc_now()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    receipt = {
        "schema": "D8_3B_EXECUTION_RECEIPT_V2",
        "state": "RUNNING",
        "run_root": str(output_root),
        "log_root": str(log_root),
        "dispatcher_pid": os.getpid(),
        "dispatcher_command": [sys.executable, *sys.argv],
        "started_utc": started_utc,
        "matrix": {
            "configs": configs,
            "seeds": seeds,
            "folds": FOLDS,
            "epochs": args.epochs,
            "gpus": gpu_ids,
            "planned_jobs": D8_3B_TOTAL_JOBS,
        },
        "provenance": source_receipt,
        "h1_lineage": lineage,
        "lineage_digest": lineage["lineage_digest"],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    atomic_write_json(output_root / "EXECUTION_RECEIPT.json", receipt)
    jobs = build_jobs(
        configs=configs,
        seeds=seeds,
        folds=FOLDS,
        gpu_ids=gpu_ids,
        epochs=args.epochs,
        cache_root=str(cache_root),
        cache_seal=actual_cache_seal,
        output_root=output_root,
        python_bin=args.python_bin,
        source_commit=source_receipt["source_commit"],
        source_tree=source_receipt["source_tree"],
        python_environment=python_environment,
        lineage_digest=lineage["lineage_digest"],
    )
    manifest = new_manifest(
        jobs=jobs,
        configs=configs,
        seeds=seeds,
        folds=FOLDS,
        epochs=args.epochs,
        gpu_ids=gpu_ids,
        source=source_receipt,
    )
    atomic_write_json(output_root / "JOB_MANIFEST.json", manifest)

    abort_state: dict[str, str | None] = {"reason": None}
    previous_handlers: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        if abort_state["reason"] is None:
            abort_state["reason"] = signal.Signals(signum).name

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, on_signal)
    try:
        manifest = dispatch_jobs(
            output_root / "JOB_MANIFEST.json",
            output_root / "STOP_D8_3B",
            manifest,
            abort_reason_fn=lambda: abort_state["reason"],
            job_timeout_seconds=args.job_timeout_seconds,
            grace_seconds=args.terminate_grace_seconds,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return _finalize_run(
        output_root,
        manifest,
        receipt,
        abort_reason=manifest.get("abort_reason"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
