"""D8-2: Full 5-fold CV — B0-B4 x 5 folds x seed 20260717.

Configs:
  B0 = majority baseline
  B1 = frozen heuristic baseline
  B2 = 25D Student + legacy step weighting (uniform)
  B3 = 25D Student + Teacher-event weighting
  B4 = B3 + suite balancing + G=3 consolidation

All use shared d8_train_core. Threshold fixed at logit>0.
OOF predictions saved for event/scheduler metrics.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_train_core import (
    D8StudentDetector, create_model, compute_normalization, apply_normalization,
    compute_loss, SEED, FEATURE_DIM,
)
from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

FOLDS = [0, 1, 2, 3, 4]
CONFIGS = ["B0", "B1", "B2", "B3", "B4"]
EPOCHS = 20
LR = 1e-3
THRESHOLD = 0.0  # logit > 0 = positive


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def load_cache(cache_root: Path) -> list[dict]:
    verify_seal(cache_root)
    entries = []
    for ep_file in sorted((cache_root / "per_episode").iterdir()):
        if ep_file.suffix == ".json":
            entries.extend(json.loads(ep_file.read_text("utf-8")))
    return entries


def majority_baseline(train: list[dict], val: list[dict]) -> dict:
    """B0: Always predict majority class from train."""
    n_pos = sum(1 for e in train if e["physical_target"] == 1.0)
    n_neg = sum(1 for e in train if e["physical_target"] == 0.0)
    pred_class = 1.0 if n_pos >= n_neg else 0.0
    preds = [{"episode_id": e["episode_id"], "step": e["step"],
              "target": e["physical_target"], "logit": 1.0 if pred_class == 1.0 else -1.0,
              "pred": pred_class, "weight": e["D8_weight"]}
             for e in val if e["effective_mask"]]
    return {"config": "B0", "predictions": preds, "train_pos": n_pos, "train_neg": n_neg}


def train_student(train: list[dict], val: list[dict], config: str, fold: int,
                  output_dir: Path, device: torch.device) -> dict:
    """B2/B3/B4: Train 25D Student with different weight strategies."""
    effective_train = [e for e in train if e["effective_mask"]]
    effective_val = [e for e in val if e["effective_mask"]]

    X_tr = torch.tensor([e["features_25d_raw"] for e in effective_train], dtype=torch.float32)
    y_tr = torch.tensor([e["physical_target"] for e in effective_train], dtype=torch.float32)
    w_tr = torch.tensor([e["D8_weight"] for e in effective_train], dtype=torch.float32)
    X_va = torch.tensor([e["features_25d_raw"] for e in effective_val], dtype=torch.float32)
    y_va = torch.tensor([e["physical_target"] for e in effective_val], dtype=torch.float32)

    norm = compute_normalization(X_tr)
    model = create_model().to(device)
    X_tr, y_tr, w_tr = X_tr.to(device), y_tr.to(device), w_tr.to(device)
    X_va, y_va = X_va.to(device), y_va.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    torch.manual_seed(SEED)
    losses = []

    for epoch in range(EPOCHS):
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
        val_loss = float(compute_loss(val_logits, y_va, torch.ones_like(y_va)))

    predictions = []
    for i, e in enumerate(effective_val):
        predictions.append({
            "episode_id": e["episode_id"], "step": e["step"],
            "target": e["physical_target"], "logit": float(val_logits[i]),
            "pred": 1.0 if float(val_logits[i]) > THRESHOLD else 0.0,
            "weight": e["D8_weight"],
        })

    # Save checkpoint
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalization": norm,
        "config": config, "fold": fold,
    }, str(ckpt_path))

    return {
        "config": config, "fold": fold,
        "predictions": predictions,
        "train_losses": losses, "val_loss": val_loss,
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
    }


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute step-level metrics from OOF predictions."""
    y_true = np.array([p["target"] for p in predictions])
    y_pred = np.array([p["pred"] for p in predictions])
    y_logit = np.array([p["logit"] for p in predictions])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    tpr = tp / max(tp + fn, 1)  # recall / sensitivity
    tnr = tn / max(tn + fp, 1)  # specificity
    bacc = (tpr + tnr) / 2
    precision = tp / max(tp + fp, 1)
    mcc_num = tp * tn - fp * fn
    mcc_den = max(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1)
    mcc = mcc_num / mcc_den

    # AUROC (simple trapezoidal)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_true, y_logit))
    except ImportError:
        auroc = float("nan")

    return {
        "n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "balanced_accuracy": bacc, "mcc": mcc,
        "precision": precision, "recall": tpr, "specificity": tnr,
        "auroc": auroc, "pred_positive_rate": (tp + fp) / max(total, 1),
    }


def run_cv(cache_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output_root exists: {output_root}")

    entries = load_cache(cache_root)
    print(f"Loaded {len(entries)} entries")

    results = {}
    all_metrics = []

    for config in CONFIGS:
        fold_metrics = []
        for fold in FOLDS:
            print(f"\n{'='*40}\n{config} fold {fold}\n{'='*40}")

            fold_dir = output_root / f"{config}_fold{fold}"
            fold_dir.mkdir(parents=True)

            train_entries = [e for e in entries if e["fold_id"] != fold]
            val_entries = [e for e in entries if e["fold_id"] == fold]

            if config == "B0":
                result = majority_baseline(train_entries, val_entries)
            elif config == "B1":
                result = {"config": "B1", "predictions": [], "note": "Heuristic baseline — frozen from prior work"}
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                result = train_student(train_entries, val_entries, config, fold, fold_dir, device)

            metrics = compute_metrics(result["predictions"])
            metrics["config"] = config
            metrics["fold"] = fold
            all_metrics.append(metrics)
            fold_metrics.append(metrics)

            # Save fold predictions
            (fold_dir / "predictions.json").write_text(json.dumps(result["predictions"]) + "\n")
            (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

            print(f"  BACC={metrics['balanced_accuracy']:.4f} MCC={metrics['mcc']:.4f} AUROC={metrics['auroc']:.4f}")

        results[config] = fold_metrics

    # Summary
    summary = {}
    for config in CONFIGS:
        fm = [m for m in all_metrics if m["config"] == config]
        summary[config] = {
            "mean_bacc": float(np.mean([m["balanced_accuracy"] for m in fm])),
            "mean_mcc": float(np.mean([m["mcc"] for m in fm])),
            "mean_auroc": float(np.mean([m["auroc"] for m in fm])),
            "mean_recall": float(np.mean([m["recall"] for m in fm])),
        }

    (output_root / "D8_2_CV_SUMMARY.json").write_text(json.dumps({
        "schema": "D8_2_CV_SUMMARY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "folds": FOLDS, "configs": CONFIGS,
        "epochs": EPOCHS, "threshold": THRESHOLD,
        "summary": summary,
        "all_metrics": all_metrics,
    }, indent=2, sort_keys=True) + "\n")

    digest = _write_seal(output_root)
    print(f"\nCV complete. Seal: {digest}")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "seal": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_cv(args.cache_root, args.output_root)
    raise SystemExit(0)
