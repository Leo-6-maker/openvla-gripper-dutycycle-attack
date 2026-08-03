"""Independent, read-only-first auditor for a D8-3B run root."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

D8_3B_CONFIGS = ["B3"]
D8_3B_SEEDS = list(range(20260720, 20260730))
D8_3B_FOLDS = [0, 1, 2, 3, 4]
D8_3B_EPOCHS = 100
D8_3B_TOTAL_JOBS = 50
ALLOWED_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED_KILL_SWITCH",
    "ABORTED",
}
METRIC_FIELDS = ("auroc", "balanced_accuracy", "mcc")
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
    "started_utc",
    "finished_utc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_sha256_seal(root: Path) -> None:
    root = root.resolve()
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    _atomic_text(root / "SHA256SUMS", sums)
    _atomic_text(
        root / "SHA256SUMS.sha256",
        f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n",
    )


def _verify_existing_seal(root: Path) -> list[str]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.exists() and not sidecar.exists():
        return []
    errors: list[str] = []
    if not sums.is_file() or not sidecar.is_file():
        return ["seal_pair_incomplete"]
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        errors.append("seal_sidecar_mismatch")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        relative = Path(name)
        if (
            not separator
            or len(digest) != 64
            or name in listed
            or relative.is_absolute()
            or ".." in relative.parts
            or name in {"SHA256SUMS", "SHA256SUMS.sha256"}
        ):
            errors.append(f"seal_row_invalid:{name}")
            continue
        path = root / relative
        listed[name] = digest.lower()
        if not path.is_file() or sha256_file(path).lower() != digest.lower():
            errors.append(f"seal_file_mismatch:{name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if actual != set(listed):
        errors.append("seal_file_closure_mismatch")
    return errors


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint is not a mapping")
    return value


def _safe_expected_path(root: Path, value: Any, expected: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        actual = Path(value).resolve()
    except OSError:
        return False
    return actual == expected.resolve() and root.resolve() in actual.parents and not actual.is_symlink()


def _expected_identities() -> set[tuple[str, int, int]]:
    return {
        (config, seed, fold)
        for config in D8_3B_CONFIGS
        for seed in D8_3B_SEEDS
        for fold in D8_3B_FOLDS
    }


def _audit_job(
    root: Path,
    job: Mapping[str, Any],
    global_provenance: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    config, seed, fold = job.get("config"), job.get("seed"), job.get("fold")
    job_id = str(job.get("job_id", "UNKNOWN"))
    if job.get("status") != "COMPLETED":
        return None
    unit_dir = root / f"seed{seed}" / f"{config}_fold{fold}"
    expected_paths = {
        "metrics_path": unit_dir / "metrics.json",
        "checkpoint_path": unit_dir / "checkpoint.pt",
        "predictions_path": unit_dir / "predictions.json",
    }
    for field, expected_path in expected_paths.items():
        if not _safe_expected_path(root, job.get(field), expected_path) or not expected_path.is_file():
            errors.append(f"{job_id}:{field}_missing_or_rebound")
            return None
    try:
        metrics = _load_json(expected_paths["metrics_path"])
        predictions = _load_json(expected_paths["predictions_path"])
        checkpoint = _load_checkpoint(expected_paths["checkpoint_path"])
    except Exception as exc:
        errors.append(f"{job_id}:artifact_parse:{type(exc).__name__}")
        return None
    if not isinstance(metrics, Mapping) or not isinstance(predictions, list):
        errors.append(f"{job_id}:artifact_type")
        return None
    if job.get("exit_code") != 0:
        errors.append(f"{job_id}:returncode_not_zero")
    if metrics.get("schema") != "D8_3B_UNIT_METRICS_V2":
        errors.append(f"{job_id}:metrics_schema")
    for key in (*METRIC_FIELDS, *PROVENANCE_FIELDS):
        if key not in metrics:
            errors.append(f"{job_id}:metrics_missing:{key}")
    expected_provenance = job.get("expected_provenance", {})
    if not isinstance(expected_provenance, Mapping):
        expected_provenance = {}
    for key in PROVENANCE_FIELDS:
        if key in {"started_utc", "finished_utc"}:
            if not isinstance(metrics.get(key), str) or not metrics.get(key):
                errors.append(f"{job_id}:timestamp_missing:{key}")
            continue
        expected = expected_provenance.get(key, global_provenance.get(key))
        if metrics.get(key) != expected:
            errors.append(f"{job_id}:metrics_provenance:{key}")
    for key in METRIC_FIELDS:
        try:
            if not math.isfinite(float(metrics[key])):
                errors.append(f"{job_id}:nonfinite:{key}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{job_id}:invalid_metric:{key}")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != "D8_3B_CHECKPOINT_V2":
        errors.append(f"{job_id}:checkpoint_schema")
    elif "model_state" not in checkpoint:
        errors.append(f"{job_id}:checkpoint_model_missing")
    else:
        for key in PROVENANCE_FIELDS:
            if checkpoint.get(key) != metrics.get(key):
                errors.append(f"{job_id}:checkpoint_provenance:{key}")
    return dict(metrics)


def audit_run(run_root: Path, *, write_artifacts: bool = True) -> dict[str, Any]:
    root = run_root.resolve()
    errors = _verify_existing_seal(root)
    manifest_path = root / "JOB_MANIFEST.json"
    summary_path = root / "D8_3B_SUMMARY.json"
    receipt_path = root / "EXECUTION_RECEIPT.json"
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        manifest = {}
        errors.append(f"manifest_unreadable:{type(exc).__name__}")
    if not isinstance(manifest, Mapping):
        manifest = {}
        errors.append("manifest_not_mapping")
    matrix = manifest.get("matrix", {})
    if not isinstance(matrix, Mapping):
        matrix = {}
    exact_matrix = (
        matrix.get("configs") == D8_3B_CONFIGS
        and matrix.get("seeds") == D8_3B_SEEDS
        and matrix.get("folds") == D8_3B_FOLDS
        and matrix.get("epochs") == D8_3B_EPOCHS
        and matrix.get("planned_jobs") == D8_3B_TOTAL_JOBS
    )
    if not exact_matrix:
        errors.append("matrix_not_exact")
    gpus = matrix.get("gpus", [])
    if not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus) or any(
        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpus
    ):
        errors.append("gpu_contract_invalid")
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        jobs = []
        errors.append("jobs_not_list")
    expected = _expected_identities()
    actual = [(job.get("config"), job.get("seed"), job.get("fold")) for job in jobs if isinstance(job, Mapping)]
    if len(jobs) != D8_3B_TOTAL_JOBS:
        errors.append("job_count_not_50")
    if len(actual) != len(set(actual)):
        errors.append("duplicate_job_identity")
    if set(actual) != expected:
        errors.append("job_identity_closure")
    if any(job.get("status") not in ALLOWED_STATES for job in jobs if isinstance(job, Mapping)):
        errors.append("unknown_job_state")
    global_provenance = manifest.get("provenance", {})
    if not isinstance(global_provenance, Mapping):
        global_provenance = {}
        errors.append("manifest_provenance_invalid")
    completed_metrics: dict[int, list[dict[str, Any]]] = {seed: [] for seed in D8_3B_SEEDS}
    for job in jobs:
        if not isinstance(job, Mapping):
            errors.append("job_not_mapping")
            continue
        metrics = _audit_job(root, job, global_provenance, errors)
        if metrics is not None and job.get("seed") in completed_metrics:
            completed_metrics[int(job["seed"])].append(metrics)

    seed_summary: dict[str, dict[str, Any]] = {}
    for seed in D8_3B_SEEDS:
        rows = completed_metrics[seed]
        values = {key: [float(row[key]) for row in rows if key in row] for key in METRIC_FIELDS}
        complete = len(rows) == len(D8_3B_FOLDS) and all(len(values[key]) == 5 for key in METRIC_FIELDS)
        seed_summary[str(seed)] = {
            "folds_completed": len(rows),
            "mean_auroc": float(np.mean(values["auroc"])) if complete else float("nan"),
            "mean_bacc": float(np.mean(values["balanced_accuracy"])) if complete else float("nan"),
            "mean_mcc": float(np.mean(values["mcc"])) if complete else float("nan"),
        }
    seed_means = [seed_summary[str(seed)]["mean_auroc"] for seed in D8_3B_SEEDS]
    stability_std = (
        float(np.std(seed_means, ddof=1))
        if len(seed_means) == 10 and all(math.isfinite(value) for value in seed_means)
        else float("nan")
    )
    completed_count = sum(job.get("status") == "COMPLETED" for job in jobs if isinstance(job, Mapping))
    failed_count = sum(job.get("status") == "FAILED" for job in jobs if isinstance(job, Mapping))
    skipped_count = sum(job.get("status") == "SKIPPED_KILL_SWITCH" for job in jobs if isinstance(job, Mapping))
    aborted_count = sum(job.get("status") == "ABORTED" for job in jobs if isinstance(job, Mapping))
    per_seed_pass = all(
        math.isfinite(seed_summary[str(seed)]["mean_auroc"])
        and math.isfinite(seed_summary[str(seed)]["mean_bacc"])
        and math.isfinite(seed_summary[str(seed)]["mean_mcc"])
        and seed_summary[str(seed)]["mean_auroc"] >= 0.80
        and seed_summary[str(seed)]["mean_bacc"] >= 0.70
        and seed_summary[str(seed)]["mean_mcc"] > 0
        for seed in D8_3B_SEEDS
    )
    checks = {
        "matrix_exact": exact_matrix,
        "planned_50": len(jobs) == 50,
        "completed_50": completed_count == 50,
        "failed_0": failed_count == 0,
        "skipped_0": skipped_count == 0,
        "aborted_0": aborted_count == 0,
        "identity_exact": set(actual) == expected and len(actual) == len(set(actual)),
        "seed_fold_coverage": all(seed_summary[str(seed)]["folds_completed"] == 5 for seed in D8_3B_SEEDS),
        "per_seed_metrics": per_seed_pass,
        "auroc_std_ddof1_le_003": math.isfinite(stability_std) and stability_std <= 0.03,
        "artifacts_and_provenance": not any(
            ":artifact_" in error
            or ":metrics_" in error
            or ":checkpoint_" in error
            or ":nonfinite" in error
            or ":invalid_metric" in error
            for error in errors
        ),
    }
    summary = {}
    try:
        summary = _load_json(summary_path)
    except Exception as exc:
        errors.append(f"summary_unreadable:{type(exc).__name__}")
    try:
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            errors.append("receipt_not_mapping")
        else:
            if receipt.get("matrix") != matrix:
                errors.append("receipt_matrix_mismatch")
            if receipt.get("provenance") != global_provenance:
                errors.append("receipt_provenance_mismatch")
    except Exception as exc:
        errors.append(f"receipt_unreadable:{type(exc).__name__}")

    explicit_abort = bool(manifest.get("abort_reason")) or skipped_count > 0 or aborted_count > 0
    computed_pass = all(checks.values()) and not errors
    verdict = "ABORTED_INCOMPLETE" if explicit_abort else ("PASS" if computed_pass else "FAIL")
    launcher_verdict = summary.get("verdict") if isinstance(summary, Mapping) else None
    if launcher_verdict != verdict:
        errors.append(f"launcher_verdict_mismatch:{launcher_verdict!r}!={verdict!r}")
        if verdict == "PASS":
            verdict = "FAIL"
    audit = {
        "schema": "D8_3B_AUDIT_V2",
        "verdict": verdict,
        "launcher_verdict": launcher_verdict,
        "checks": checks,
        "errors": errors,
        "matrix": {
            "configs": D8_3B_CONFIGS,
            "seeds": D8_3B_SEEDS,
            "folds": D8_3B_FOLDS,
            "epochs": D8_3B_EPOCHS,
            "planned_jobs": D8_3B_TOTAL_JOBS,
        },
        "counts": {
            "planned": len(jobs),
            "completed": completed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "aborted": aborted_count,
        },
        "seed_summary": seed_summary,
        "seed_means_auroc": seed_means,
        "stability_std_ddof1": stability_std,
        "stability_threshold": 0.03,
    }
    if write_artifacts:
        _atomic_json(root / "D8_3B_AUDIT.json", audit)
    return audit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = audit_run(args.run_root, write_artifacts=True)
    write_sha256_seal(args.run_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)
