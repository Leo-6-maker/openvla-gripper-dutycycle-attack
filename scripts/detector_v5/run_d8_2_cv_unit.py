"""D8 CV unit: one frozen (config, seed, fold) training job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_train_core import (
    D8StudentDetector,
    apply_normalization,
    compute_loss,
    compute_normalization,
    create_model,
    FEATURE_DIM,
)
from audit_r3_contact_input import sha256_file, verify_seal

THRESHOLD = 0.0
OPTIMIZER_NAME = "Adam"
LEARNING_RATE = 1e-3
WEIGHT_NORMALIZATION = "mean_to_one"
UNIT_METRICS_SCHEMA = "D8_3B_UNIT_METRICS_V2"
CHECKPOINT_SCHEMA = "D8_3B_CHECKPOINT_V2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    def json_safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, Mapping):
            return {str(key): json_safe(value) for key, value in item.items()}
        if isinstance(item, list):
            return [json_safe(value) for value in item]
        return item

    _atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_payload(
    schema: str,
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if "schema" in payload or "schema" in provenance:
        raise RuntimeError("artifact schema must be owned by the artifact envelope")
    overlap = set(payload).intersection(provenance)
    if overlap:
        raise RuntimeError(f"artifact payload/provenance collision: {sorted(overlap)!r}")
    return {"schema": schema, **dict(payload), **dict(provenance)}


def _local_python_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }


def _git_value(*args: str) -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_receipt(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_cache(cache_root: Path, expected_seal: str | None = None) -> tuple[list[dict], dict[str, Any]]:
    seal = verify_seal(cache_root)
    actual_seal = str(seal["sha256sums_sha256"]).lower()
    if expected_seal is not None and actual_seal != expected_seal.lower():
        raise RuntimeError(
            f"cache seal mismatch: expected {expected_seal.lower()}, got {actual_seal}"
        )
    entries = []
    for ep_file in sorted((cache_root / "per_episode").iterdir()):
        if ep_file.suffix == ".json":
            entries.extend(json.loads(ep_file.read_text("utf-8")))
    return entries, {**seal, "sha256sums_sha256": actual_seal}


def _provenance(
    *,
    cache_root: Path,
    cache_seal: str,
    config: str,
    fold: int,
    seed: int,
    epochs: int,
    started_utc: str,
    source_commit: str | None,
    source_tree: str | None,
    launcher_sha256: str | None,
    train_core_sha256: str | None,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_provenance = receipt.get("provenance", {})
    if not isinstance(receipt_provenance, Mapping):
        receipt_provenance = {}
    source_commit = source_commit or receipt_provenance.get("source_commit") or _git_value("rev-parse", "HEAD")
    source_tree = source_tree or receipt_provenance.get("source_tree") or _git_value("rev-parse", "HEAD^{tree}")
    launcher_sha256 = launcher_sha256 or receipt_provenance.get("parallel_launcher_sha256")
    train_core_sha256 = train_core_sha256 or receipt_provenance.get("train_core_sha256")
    if not source_commit or not source_tree or not launcher_sha256 or not train_core_sha256:
        raise RuntimeError("D8-3B provenance binding is incomplete")
    lineage_digest = receipt.get("lineage_digest") or receipt_provenance.get("lineage_digest")
    if (
        not isinstance(lineage_digest, str)
        or len(lineage_digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in lineage_digest)
    ):
        raise RuntimeError("D8-3B execution receipt lineage binding is incomplete")
    environment = receipt_provenance.get("python_environment")
    if not isinstance(environment, Mapping):
        environment = _local_python_environment()
    return {
        "config": config,
        "seed": seed,
        "fold": fold,
        "epochs": epochs,
        "threshold": THRESHOLD,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "weight_normalization": WEIGHT_NORMALIZATION,
        "cache_root": str(cache_root.resolve()),
        "cache_seal": cache_seal,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "unit_script_sha256": sha256_file(Path(__file__).resolve()),
        "parallel_launcher_sha256": launcher_sha256,
        "train_core_sha256": train_core_sha256,
        "python_environment": dict(environment),
        "lineage_digest": lineage_digest.lower(),
        "started_utc": started_utc,
    }


def run_unit(
    cache_root: Path,
    config: str,
    fold: int,
    seed: int,
    epochs: int,
    output_dir: Path,
    *,
    expected_cache_seal: str | None = None,
    source_commit: str | None = None,
    source_tree: str | None = None,
    launcher_sha256: str | None = None,
    train_core_sha256: str | None = None,
    execution_receipt: Path | None = None,
):
    started_utc = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries, seal = load_cache(cache_root, expected_cache_seal)
    metadata = _provenance(
        cache_root=cache_root,
        cache_seal=str(seal["sha256sums_sha256"]),
        config=config,
        fold=fold,
        seed=seed,
        epochs=epochs,
        started_utc=started_utc,
        source_commit=source_commit,
        source_tree=source_tree,
        launcher_sha256=launcher_sha256,
        train_core_sha256=train_core_sha256,
        receipt=_read_receipt(execution_receipt),
    )

    train_entries = [e for e in entries if e["fold_id"] != fold]
    val_entries = [e for e in entries if e["fold_id"] == fold]
    effective_train = [e for e in train_entries if e["effective_mask"]]
    effective_val = [e for e in val_entries if e["effective_mask"]]

    train_ids = sorted(set(e["episode_id"] for e in effective_train))
    val_ids = sorted(set(e["episode_id"] for e in effective_val))
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/val identity overlap")

    checkpoint_payload: dict[str, Any] | None = None
    if config == "B0":
        n_pos = sum(1 for e in effective_train if e["physical_target"] == 1.0)
        n_neg = sum(1 for e in effective_train if e["physical_target"] == 0.0)
        pred_class = 1.0 if n_pos >= n_neg else 0.0
        preds = [
            {
                "episode_id": e["episode_id"],
                "step": e["step"],
                "target": e["physical_target"],
                "logit": 1.0 if pred_class == 1.0 else -1.0,
                "pred": pred_class,
            }
            for e in effective_val
        ]
        metrics = compute_metrics(preds)
        metrics["train_pos"] = n_pos
        metrics["train_neg"] = n_neg
    elif config == "B1":
        preds = []
        metrics = {"note": "heuristic placeholder"}
    else:
        X_tr = torch.tensor([e["features_25d_raw"] for e in effective_train], dtype=torch.float32)
        y_tr = torch.tensor([e["physical_target"] for e in effective_train], dtype=torch.float32)
        X_va = torch.tensor([e["features_25d_raw"] for e in effective_val], dtype=torch.float32)
        y_va = torch.tensor([e["physical_target"] for e in effective_val], dtype=torch.float32)

        if config == "B2":
            # Uniform per-step weights (legacy)
            w_tr = torch.ones(len(effective_train), dtype=torch.float32)
        elif config == "B3":
            # Teacher-event weights from cache (G=3 consolidation)
            w_tr = torch.tensor([e["D8_weight"] for e in effective_train], dtype=torch.float32)
        elif config == "B4":
            # B3 + suite balancing: normalize by suite prevalence
            w_raw = torch.tensor([e["D8_weight"] for e in effective_train], dtype=torch.float32)
            suite_counts = defaultdict(float)
            for e in effective_train:
                suite = e["episode_id"].split("/")[0]
                suite_counts[suite] += 1.0
            suite_weights = {s: 1.0 / max(c, 1) for s, c in suite_counts.items()}
            w_tr = w_raw.clone()
            for i, e in enumerate(effective_train):
                suite = e["episode_id"].split("/")[0]
                w_tr[i] *= suite_weights.get(suite, 1.0)
        else:
            w_tr = torch.ones(len(effective_train), dtype=torch.float32)

        # Global weight normalization: mean=1 so effective LR is stable.
        w_tr = w_tr / w_tr.mean()

        norm = compute_normalization(X_tr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_model(seed).to(device)
        X_tr, y_tr, w_tr = X_tr.to(device), y_tr.to(device), w_tr.to(device)
        X_va, y_va = X_va.to(device), y_va.to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        torch.manual_seed(seed)
        losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(apply_normalization(X_tr, norm))
            loss = compute_loss(logits, y_tr, w_tr)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        model.eval()
        with torch.no_grad():
            val_logits = model(apply_normalization(X_va, norm))

        preds = []
        for i, e in enumerate(effective_val):
            logit = float(val_logits[i])
            preds.append(
                {
                    "episode_id": e["episode_id"],
                    "step": e["step"],
                    "target": e["physical_target"],
                    "logit": logit,
                    "pred": 1.0 if logit > THRESHOLD else 0.0,
                }
            )
        metrics = compute_metrics(preds)
        metrics["train_losses"] = losses
        metrics["train_ids"] = len(train_ids)
        metrics["val_ids"] = len(val_ids)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "normalization": norm,
        }

    finished_utc = utc_now()
    metadata["finished_utc"] = finished_utc
    metrics = _artifact_payload(UNIT_METRICS_SCHEMA, metrics, metadata)
    if checkpoint_payload is not None:
        _atomic_torch_save(
            output_dir / "checkpoint.pt",
            _artifact_payload(CHECKPOINT_SCHEMA, checkpoint_payload, metadata),
        )
    _atomic_json(output_dir / "predictions.json", preds)
    _atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def compute_metrics(preds):
    if not preds:
        return {"n": 0}
    y_true = np.array([p["target"] for p in preds])
    y_pred = np.array([p["pred"] for p in preds])
    y_logit = np.array([p.get("logit", p["pred"]) for p in preds])
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    mcc_num = tp * tn - fp * fn
    mcc_den = max(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1)
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y_true, y_logit))
    except ImportError:
        auroc = float("nan")
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(total, 1),
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": mcc_num / mcc_den,
        "precision": tp / max(tp + fp, 1),
        "recall": tpr,
        "specificity": tnr,
        "auroc": auroc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-seal")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    parser.add_argument("--launcher-sha256")
    parser.add_argument("--train-core-sha256")
    parser.add_argument("--execution-receipt", type=Path)
    args = parser.parse_args()
    metrics = run_unit(
        args.cache_root,
        args.config,
        args.fold,
        args.seed,
        args.epochs,
        args.output_dir,
        expected_cache_seal=args.expected_cache_seal,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
        launcher_sha256=args.launcher_sha256,
        train_core_sha256=args.train_core_sha256,
        execution_receipt=args.execution_receipt,
    )
    print(
        f"DONE {args.config} fold {args.fold} seed {args.seed}: "
        f"BACC={metrics.get('balanced_accuracy', 0):.4f} "
        f"MCC={metrics.get('mcc', 0):.4f}"
    )
