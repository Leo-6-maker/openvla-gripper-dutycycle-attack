#!/usr/bin/env python3
"""Preregister and train the clean-only R3-A matched full-data ensemble."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
try:
    import resource
except ImportError:  # pragma: no cover - server runs Linux
    resource = None
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_d8_3b_run import verify_sha256_seal
from audit_r3_contact_input import sha256_file
from d8_train_core import apply_normalization, compute_loss, compute_normalization, create_model
from run_detector_clean_freeze import (
    EPOCHS,
    LEARNING_RATE,
    CONFIG,
    WEIGHT_NORMALIZATION,
    cache_effective_rows,
    load_cache,
    metric_summary,
    seal_directory,
    sha256_json,
    utc_now,
)

SEEDS = tuple(range(20260720, 20260730))
CACHE_A_SEAL = "929a0a666a867c93094b13752f4c2f848640bbedb2dadc9a20d834f3ee8b6814"
CORE_BLOB = "bd4c505ada3696913b061f3132b7ea67622b3cad"
FEATURE_SCHEMA_BLOB = "3f6c62dd7b263d4d1faf42e6c6eae5e7d52196ab"
H1_COMMIT = "9dd324ad70a9be17548f72437da8454356abfd28"
H1_TREE = "0333510e291f8ec0c5b8738136019f30c5de17aa"
SOURCE_SNAPSHOT = "99648bdee45cde6411159f6d6586b8b7e46b626ea000f07a6cff0b38251efdbd"
LINEAGE_DIGEST = "d42b1fd9a2e511facb71faaedb84c575ff5fa649e16071685626547c96a61833"
R2_ROOT_SEAL = "f36ac1e18fca516fb8fae82e1290d034848ad572f6fad6908beb2d40b9bc9277"
R2_CHECKPOINT_SHA = "ce7f03088d84a796d38fbdc107cea7f21bdb4808e35f7dc754e1b52e48bce1d4"
R2_SCHEDULER = {
    "threshold": 0.43356089500710393,
    "persistence": 5,
    "hysteresis": 1.0,
    "cooldown": 0,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def python_environment() -> dict[str, Any]:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
        "numpy_version": np.__version__,
    }


def require_clean_gpu_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be unset or empty")
    if not torch.cuda.is_available():
        raise RuntimeError("R3-A is GPU-only; CUDA is unavailable")


def checkpoint_provenance(
    args: argparse.Namespace,
    commit: str,
    tree: str,
    cache_seal: str,
    formal_root: str,
    formal_seal: str,
) -> dict[str, Any]:
    return {
        "r3_source_commit": commit,
        "r3_source_tree": tree,
        "d8_train_core_blob": CORE_BLOB,
        "feature_schema_blob": FEATURE_SCHEMA_BLOB,
        "cache_a_root": str(args.cache_a.resolve()),
        "cache_a_seal": cache_seal,
        "student_formal_root": formal_root,
        "student_formal_root_seal": formal_seal,
        "h1_source_commit": H1_COMMIT,
        "h1_source_tree": H1_TREE,
        "source_snapshot_sha256": SOURCE_SNAPSHOT,
        "lineage_digest": LINEAGE_DIGEST,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def build_r3b_plan(args: argparse.Namespace, r3: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "R3B_CLEAN_HOLDOUT_PREREGISTRATION_V1",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "trigger": "execute only if R3A clean transfer hard Gate fails",
        "r3_source": r3,
        "split": {
            "salt": "D8_STAGE2_R3B_SPLIT_V1_20260805",
            "unit": "episode_id",
            "development_fraction": 0.80,
            "holdout_fraction": 0.20,
            "within_each_suite": True,
            "positive_event_stratification": True,
            "ordering": "SHA256(salt + :: + episode_id)",
            "episode_cross_split_forbidden": True,
            "all_suites_and_positive_strata_required": True,
        },
        "development_oof": {
            "seeds": list(SEEDS),
            "folds": 5,
            "config": CONFIG,
            "epochs": EPOCHS,
            "architecture": "25->32->16->1",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_normalization": WEIGHT_NORMALIZATION,
        },
        "scheduler_search": {
            "score": "mean_logit_over_10_development_oof_models",
            "thresholds": "401 quantiles plus 0",
            "persistence": [1, 2, 3, 4, 5],
            "hysteresis": [0.0, 0.25, 0.5, 1.0],
            "cooldown": [0, 2, 5, 10],
            "s1": {"false_onset_episode_rate_max": 0.10, "negative_active_step_rate_max": 0.05},
            "s2": {"active_overlap_event_recall_min": 0.70, "median_delay_max": 2},
            "tie_break": ["s1_pass", "max_active_overlap_recall", "min_false_onset", "min_delay", "min_persistence", "threshold_closest_to_zero"],
            "holdout_or_attack_tuning_forbidden": True,
        },
        "deployment": {"scope": "development_episodes_only", "scorer": "mean_of_10_full_development_models", "same_seeds": list(SEEDS)},
        "decision": {"s1_holdout_pass_required_for_shadow": True, "s2_holdout_report_only_for_shadow": True, "third_detector_forbidden": True},
        "thresholds_changed": False,
        "attack_informed_tuning": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "created_utc": utc_now(),
    }


def build_r3a_plan(args: argparse.Namespace, r3: dict[str, Any], r2_receipt: dict[str, Any], r2_receipt_sha: str, stage_t_sha: str) -> dict[str, Any]:
    return {
        "schema": "R3A_MATCHED_ENSEMBLE_PLAN_V1",
        "status": "PREREGISTERED",
        "created_utc": utc_now(),
        "r3_source": r3,
        "stage_t_audit": {"path": str(args.stage_t_audit.resolve()), "sha256": stage_t_sha, "verdict": "PASS"},
        "training": {
            "seeds": list(SEEDS),
            "model_count": 10,
            "fit_scope": "full_clean_cache_A",
            "config": CONFIG,
            "feature_dim": 25,
            "architecture": "25->32->16->1",
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_normalization": WEIGHT_NORMALIZATION,
            "full_data_normalization_per_model": True,
            "seed_selection": "none; all ten fixed seeds required",
        },
        "cache_a": {"root": str(args.cache_a.resolve()), "seal": r3["cache_a_seal"]},
        "r2_scheduler": {
            "root": str(args.r2_root.resolve()),
            "root_seal": R2_ROOT_SEAL,
            "freeze_receipt_sha256": r2_receipt_sha,
            "freeze_receipt_schema": r2_receipt.get("schema"),
            "scheduler": dict(R2_SCHEDULER),
            "parameters_may_change": False,
        },
        "deployment_scorer": {"definition": "mean of all ten full-data model logits at each effective Cache A identity", "missing_model_fallback": False, "partial_ensemble_forbidden": True},
        "hard_gate": {"false_onset_episode_rate_max": 0.10, "negative_active_step_rate_max": 0.05, "all_scores_finite": True, "all_suites_covered": True, "identity_closure": True, "score_deterministic": True, "scheduler_deterministic": True},
        "s2_reporting_only": ["active_overlap_event_recall", "median_first_activation_delay"],
        "failure_route": "R3B_CLEAN_HOLDOUT_PREREGISTRATION_V1",
        "thresholds_changed": False,
        "attack_informed_tuning": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    existing: list[list[str]] = []
    fieldnames = ["updated_utc", "stage", "kind", "status", "root", "details"]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or fieldnames
            existing = [dict(item) for item in reader]
    existing.append({key: str(row.get(key, "")) for key in fieldnames})
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(path)


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    goal = args.goal_root.resolve()
    goal.mkdir(parents=True, exist_ok=True)
    if (goal / "R3A_MATCHED_ENSEMBLE_PLAN.json").exists():
        raise FileExistsError("R3-A plan already exists; preregistration is immutable")
    repo = args.repo_root.resolve(strict=True)
    commit, tree = git_value(repo, "rev-parse", "HEAD"), git_value(repo, "rev-parse", "HEAD^{tree}")
    if commit != args.expected_source_commit or tree != args.expected_source_tree:
        raise RuntimeError("R3 source commit/tree mismatch")
    stage_t = read_json(args.stage_t_audit.resolve(strict=True))
    if stage_t.get("status") != "PASS" or stage_t.get("input_audit_verdict") != "PASS":
        raise RuntimeError("Stage T is not PASS")
    rows, _, cache_seal = load_cache(args.cache_a.resolve(strict=True), CACHE_A_SEAL)
    if not cache_effective_rows(rows):
        raise RuntimeError("Cache A has no effective rows")
    r2_root = args.r2_root.resolve(strict=True)
    if verify_sha256_seal(r2_root)["sha256sums_sha256"].lower() != R2_ROOT_SEAL:
        raise RuntimeError("R2 root seal mismatch")
    r2_receipt_path = r2_root / "DETECTOR_FREEZE_RECEIPT_R2.json"
    r2_receipt = read_json(r2_receipt_path)
    scheduler = r2_receipt.get("scheduler") or {}
    if any(float(scheduler.get(key)) != float(value) for key, value in R2_SCHEDULER.items()):
        raise RuntimeError("R2 scheduler binding mismatch")
    r3 = {
        "commit": commit,
        "tree": tree,
        "d8_train_core_blob": CORE_BLOB,
        "feature_schema_blob": FEATURE_SCHEMA_BLOB,
        "cache_a_seal": cache_seal["sha256sums_sha256"],
        "formal_root": stage_t.get("oof_producer", {}).get("formal_root"),
        "formal_root_seal": stage_t.get("oof_producer", {}).get("formal_root_seal"),
        "h1_source_commit": H1_COMMIT,
        "h1_source_tree": H1_TREE,
        "source_snapshot_sha256": SOURCE_SNAPSHOT,
        "lineage_digest": LINEAGE_DIGEST,
    }
    r2_sha = sha256_file(r2_receipt_path)
    stage_t_sha = sha256_file(args.stage_t_audit.resolve())
    r3a = build_r3a_plan(args, r3, r2_receipt, r2_sha, stage_t_sha)
    r3b = build_r3b_plan(args, r3)
    atomic_json(goal / "R3A_MATCHED_ENSEMBLE_PLAN.json", r3a)
    atomic_json(goal / "R3B_PRE_REGISTERED_PLAN.json", r3b)
    atomic_json(goal / "GOAL_STATUS_R3.json", {
        "schema": "TEACHER_STUDENT_DETECTOR_GOAL_STATUS_R3_V1",
        "status": "R3A_PREREGISTERED",
        "stage_t": "PASS",
        "r3a": "PREREGISTERED_NOT_EXECUTED",
        "r3b": "PREREGISTERED_FALLBACK_NOT_EXECUTED",
        "stage3a_authorized": False,
        "stage3a_rollouts": 0,
        "stage3b_authorized": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "stage1_retrained": False,
        "stage1_artifact_modified": False,
        "stage2_r2_artifact_modified": False,
        "thresholds_changed": False,
        "attack_informed_tuning": False,
        "updated_utc": utc_now(),
    })
    progress = """# Teacher–Student Detector Goal R3\n\n- Stage T: PASS\n- R3-A: preregistered; training not started\n- R3-B: preregistered fallback; not executed\n- Stage3A / Eval160 / attack: not authorized\n\nBoundary counters: `Eval160=0`, `protected_eval=0`, `attack_rollouts=0`.\n"""
    atomic_bytes(goal / "NIGHTLY_PROGRESS_R3.md", progress.encode())
    append_csv(goal / "RESOURCE_LEDGER_R3.csv", {"updated_utc": utc_now(), "stage": "R3A", "kind": "preregistration", "status": "PASS", "root": str(goal), "details": "GPU training not started"})
    atomic_json(goal / "ARTIFACT_INDEX_R3.json", {"schema": "TEACHER_STUDENT_DETECTOR_ARTIFACT_INDEX_R3_V1", "artifacts": [{"kind": "stage_t_audit", "path": str(args.stage_t_audit.resolve()), "sha256": stage_t_sha}, {"kind": "r3a_plan", "path": str(goal / "R3A_MATCHED_ENSEMBLE_PLAN.json"), "sha256": sha256_file(goal / "R3A_MATCHED_ENSEMBLE_PLAN.json")}, {"kind": "r3b_plan", "path": str(goal / "R3B_PRE_REGISTERED_PLAN.json"), "sha256": sha256_file(goal / "R3B_PRE_REGISTERED_PLAN.json")}], "updated_utc": utc_now()})
    atomic_json(goal / "DECISION_LEDGER_R3.json", {"schema": "TEACHER_STUDENT_DECISION_LEDGER_R3_V1", "decisions": [{"decision_id": "R3-T-PASS", "stage": "Stage T", "input_artifacts": [str(args.stage_t_audit.resolve())], "preregistered_rule": "all frozen Teacher/Student inputs and provenance checks pass", "observed_result": "PASS", "decision": "proceed to R3-A preregistration", "next_stage": "R3-A", "thresholds_changed": False, "attack_informed_tuning": False, "updated_utc": utc_now()}, {"decision_id": "R3A-PREREG", "stage": "R3-A", "input_artifacts": [str(goal / "R3A_MATCHED_ENSEMBLE_PLAN.json"), str(goal / "R3B_PRE_REGISTERED_PLAN.json")], "preregistered_rule": "fixed 10-model full-data ensemble; R3-B fallback is fixed before results", "observed_result": "plans written before training", "decision": "training authorized after resource gate", "next_stage": "R3-A training", "thresholds_changed": False, "attack_informed_tuning": False, "updated_utc": utc_now()}]})
    return {"goal_root": str(goal), "r3a_plan_sha256": sha256_file(goal / "R3A_MATCHED_ENSEMBLE_PLAN.json"), "r3b_plan_sha256": sha256_file(goal / "R3B_PRE_REGISTERED_PLAN.json"), "source_commit": commit, "source_tree": tree}


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> str:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def train_seed(
    seed: int,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    w_cpu: torch.Tensor,
    norm: dict[str, Any],
    output: Path,
    provenance: dict[str, Any],
    gpu_index: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = utc_now()
    timer = time.perf_counter()
    model = create_model(seed).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x, y, w = x_cpu.to(device), y_cpu.to(device), w_cpu.to(device)
    losses: list[float] = []
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(apply_normalization(x, norm)), y, w)
        if not torch.isfinite(loss):
            raise RuntimeError(f"seed {seed} loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    model.eval()
    with torch.no_grad():
        logits = model(apply_normalization(x, norm)).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError(f"seed {seed} produced non-finite logits")
    effective_count = int(x_cpu.shape[0])
    training_identity_digest = sha256_json({"seed": seed, "identities": provenance["training_identities"]})
    checkpoint = {
        "schema": "D8_R3A_FULLDATA_CHECKPOINT_V1",
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "normalization": norm,
        "normalization_sha256": sha256_json(norm),
        "config": CONFIG,
        "seed": seed,
        "fold": "FULL_CLEAN_CACHE_A",
        "epochs": EPOCHS,
        "architecture": "25->32->16->1",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_normalization": WEIGHT_NORMALIZATION,
        "training_identity_digest": training_identity_digest,
        "effective_sample_count": effective_count,
        **{key: value for key, value in provenance.items() if key != "training_identities"},
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    checkpoint_sha = save_checkpoint(output, checkpoint)
    atomic_bytes(output.with_suffix(output.suffix + ".sha256"), f"{checkpoint_sha}  {output.name}\n".encode())
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource else 0
    gpu_peak = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
    metrics = {
        "schema": "D8_R3A_FULLDATA_METRICS_V1",
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "fit_metrics": metric_summary(y_cpu.tolist(), logits.tolist()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "effective_sample_count": effective_count,
        "normalization_sha256": checkpoint["normalization_sha256"],
        "wall_seconds": time.perf_counter() - timer,
        "host_max_rss_raw": rss,
        "gpu_peak_memory_allocated_bytes": gpu_peak,
        "gpu_index": gpu_index,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    atomic_json(output.with_suffix(".json"), metrics)
    del model, optimizer, x, y, w
    torch.cuda.empty_cache()
    return {"seed": seed, "checkpoint": output.name, **metrics}


def run_training(args: argparse.Namespace) -> int:
    require_clean_gpu_env()
    repo = args.repo_root.resolve(strict=True)
    commit, tree = git_value(repo, "rev-parse", "HEAD"), git_value(repo, "rev-parse", "HEAD^{tree}")
    if commit != args.expected_source_commit or tree != args.expected_source_tree:
        raise RuntimeError("R3 source commit/tree mismatch")
    plan_path = args.goal_root.resolve(strict=True) / "R3A_MATCHED_ENSEMBLE_PLAN.json"
    plan = read_json(plan_path)
    if plan.get("status") != "PREREGISTERED" or plan.get("r3_source", {}).get("commit") != commit or plan.get("r3_source", {}).get("tree") != tree:
        raise RuntimeError("R3-A plan/source binding mismatch")
    stage_t = read_json(args.stage_t_audit.resolve(strict=True))
    if stage_t.get("status") != "PASS":
        raise RuntimeError("Stage T is not PASS")
    rows, _, cache_seal = load_cache(args.cache_a.resolve(strict=True), CACHE_A_SEAL)
    effective = cache_effective_rows(rows)
    if not effective:
        raise RuntimeError("Cache A has no effective rows")
    x_cpu = torch.tensor([row["features_25d_raw"] for row in effective], dtype=torch.float32)
    y_cpu = torch.tensor([float(row["physical_target"]) for row in effective], dtype=torch.float32)
    w_cpu = torch.tensor([float(row["D8_weight"]) for row in effective], dtype=torch.float32)
    w_cpu = w_cpu / w_cpu.mean()
    identities = sorted(f"{row['episode_id']}::{row['step']}" for row in effective)
    identity_digest = sha256_json(identities)
    norm = compute_normalization(x_cpu, source_identity_digest=identity_digest)
    norm["fit_on"] = "full_clean_cache_A"
    norm["source_identity_digest"] = identity_digest
    formal = stage_t.get("oof_producer", {})
    provenance = checkpoint_provenance(args, commit, tree, cache_seal["sha256sums_sha256"], str(formal.get("formal_root")), str(formal.get("formal_root_seal")))
    provenance["training_identities"] = identities
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(str(staging))
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        checkpoint_dir = staging / "R3A_CHECKPOINTS"
        checkpoint_dir.mkdir()
        members = []
        for seed in SEEDS:
            members.append(train_seed(seed, x_cpu, y_cpu, w_cpu, norm, checkpoint_dir / f"seed{seed}.pt", provenance, args.gpu_index))
        manifest = {
            "schema": "R3A_ENSEMBLE_MANIFEST_V1",
            "status": "PASS_10_OF_10_COMPLETED",
            "seeds": list(SEEDS),
            "members": members,
            "model_count": len(members),
            "normalization_sha256": sha256_json(norm),
            "training_identity_digest": identity_digest,
            "scorer": "mean_logits_all_10_fixed_full_data_models",
            "provenance": {key: value for key, value in provenance.items() if key != "training_identities"},
            "plan_sha256": sha256_file(plan_path),
            "started_utc": started,
            "finished_utc": utc_now(),
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
        }
        atomic_json(staging / "R3A_ENSEMBLE_MANIFEST.json", manifest)
        atomic_json(staging / "R3A_ENSEMBLE_RECEIPT.json", {
            "schema": "R3A_ENSEMBLE_RECEIPT_V1",
            "status": "PASS_10_OF_10_COMPLETED",
            "manifest": "R3A_ENSEMBLE_MANIFEST.json",
            "plan_sha256": manifest["plan_sha256"],
            "gpu_index": args.gpu_index,
            "worker_count": 1,
            "dispatcher_count": 1,
            "python_environment": python_environment(),
            "provenance": manifest["provenance"],
            "started_utc": started,
            "finished_utc": utc_now(),
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
        })
        seal = seal_directory(staging)
        os.rename(staging, output_root)
        print(json.dumps({"status": "PASS", "output_root": str(output_root), "seal": seal}, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(staging / "R3A_TRAINING_FAILURE.json", {"schema": "R3A_TRAINING_FAILURE_V1", "error": f"{type(exc).__name__}: {exc}", "started_utc": started, "finished_utc": utc_now(), "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--stage-t-audit", type=Path, required=True)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--preregister-only", action="store_true")
    args = parser.parse_args()
    if not args.preregister_only and args.output_root is None:
        parser.error("--output-root is required for training")
    return args


def main() -> int:
    args = parse_args()
    if args.preregister_only:
        print(json.dumps(preregister(args), sort_keys=True))
        return 0
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
