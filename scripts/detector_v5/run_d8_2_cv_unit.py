"""D8-2 CV unit: single (config, fold) training job. Invoked by parallel launcher."""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import defaultdict
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
    compute_loss, FEATURE_DIM,
)
from audit_r3_contact_input import sha256_file, verify_seal

THRESHOLD = 0.0


def load_cache(cache_root: Path) -> list[dict]:
    verify_seal(cache_root)
    entries = []
    for ep_file in sorted((cache_root / "per_episode").iterdir()):
        if ep_file.suffix == ".json":
            entries.extend(json.loads(ep_file.read_text("utf-8")))
    return entries


def run_unit(cache_root: Path, config: str, fold: int, seed: int, epochs: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_cache(cache_root)

    train_entries = [e for e in entries if e["fold_id"] != fold]
    val_entries = [e for e in entries if e["fold_id"] == fold]
    effective_train = [e for e in train_entries if e["effective_mask"]]
    effective_val = [e for e in val_entries if e["effective_mask"]]

    train_ids = sorted(set(e["episode_id"] for e in effective_train))
    val_ids = sorted(set(e["episode_id"] for e in effective_val))
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/val identity overlap")

    if config == "B0":
        n_pos = sum(1 for e in effective_train if e["physical_target"] == 1.0)
        n_neg = sum(1 for e in effective_train if e["physical_target"] == 0.0)
        pred_class = 1.0 if n_pos >= n_neg else 0.0
        preds = [{"episode_id": e["episode_id"], "step": e["step"],
                  "target": e["physical_target"], "logit": 1.0 if pred_class == 1.0 else -1.0,
                  "pred": pred_class} for e in effective_val]
        metrics = compute_metrics(preds)
        metrics["config"] = config; metrics["fold"] = fold
        metrics["train_pos"] = n_pos; metrics["train_neg"] = n_neg
    elif config == "B1":
        preds = []
        metrics = {"config": config, "fold": fold, "note": "heuristic placeholder"}
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

        # Global weight normalization: mean=1 so effective LR is stable
        w_tr = w_tr / w_tr.mean()

        norm = compute_normalization(X_tr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_model(seed).to(device)
        X_tr, y_tr, w_tr = X_tr.to(device), y_tr.to(device), w_tr.to(device)
        X_va, y_va = X_va.to(device), y_va.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            preds.append({"episode_id": e["episode_id"], "step": e["step"],
                          "target": e["physical_target"], "logit": logit,
                          "pred": 1.0 if logit > THRESHOLD else 0.0})

        torch.save({"model_state": model.state_dict(), "normalization": norm,
                     "config": config, "fold": fold},
                    str(output_dir / "checkpoint.pt"))

        metrics = compute_metrics(preds)
        metrics["config"] = config; metrics["fold"] = fold
        metrics["train_losses"] = losses
        metrics["train_ids"] = len(train_ids); metrics["val_ids"] = len(val_ids)

    (output_dir / "predictions.json").write_text(json.dumps(preds) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def compute_metrics(preds):
    if not preds: return {"n": 0}
    y_true = np.array([p["target"] for p in preds])
    y_pred = np.array([p["pred"] for p in preds])
    y_logit = np.array([p.get("logit", p["pred"]) for p in preds])
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    mcc_num = tp * tn - fp * fn
    mcc_den = max(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_true, y_logit))
    except ImportError:
        auroc = float("nan")
    return {"n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / max(total, 1),
            "balanced_accuracy": (tpr + tnr) / 2,
            "mcc": mcc_num / mcc_den, "precision": tp / max(tp + fp, 1),
            "recall": tpr, "specificity": tnr, "auroc": auroc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = run_unit(args.cache_root, args.config, args.fold, args.seed, args.epochs, args.output_dir)
    print(f"DONE {args.config} fold {args.fold}: BACC={metrics.get('balanced_accuracy',0):.4f} MCC={metrics.get('mcc',0):.4f}")
