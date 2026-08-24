"""D8-3A2: Resume training from epoch 50 checkpoint, continue to epoch 100."""
from __future__ import annotations

import argparse, json, sys
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


def continue_training(input_dir: Path, output_dir: Path, cache_entries: list[dict], fold: int, epochs: int):
    """Resume from checkpoint in input_dir, train epochs more, save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(str(input_dir / "checkpoint.pt"), map_location="cpu", weights_only=False)

    train_entries = [e for e in cache_entries if e["fold_id"] != fold and e["effective_mask"]]
    val_entries = [e for e in cache_entries if e["fold_id"] == fold and e["effective_mask"]]

    X_tr = torch.tensor([e["features_25d_raw"] for e in train_entries], dtype=torch.float32)
    y_tr = torch.tensor([e["physical_target"] for e in train_entries], dtype=torch.float32)
    w_tr = torch.tensor([e["D8_weight"] for e in train_entries], dtype=torch.float32)
    w_tr = w_tr / w_tr.mean()  # global normalization
    X_va = torch.tensor([e["features_25d_raw"] for e in val_entries], dtype=torch.float32)
    y_va = torch.tensor([e["physical_target"] for e in val_entries], dtype=torch.float32)

    norm = ckpt["normalization"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    X_tr, y_tr, w_tr = X_tr.to(device), y_tr.to(device), w_tr.to(device)
    X_va, y_va = X_va.to(device), y_va.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = ckpt.get("epoch", 50)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(apply_normalization(X_tr, norm))
        loss = compute_loss(logits, y_tr, w_tr)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {start_epoch + epoch + 1}: loss={float(loss):.1f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(apply_normalization(X_va, norm))

    from run_d8_2_cv_unit import compute_metrics
    preds = []
    for i, e in enumerate(val_entries):
        logit = float(val_logits[i])
        preds.append({"episode_id": e["episode_id"], "step": e["step"],
                      "target": e["physical_target"], "logit": logit,
                      "pred": 1.0 if logit > THRESHOLD else 0.0})

    metrics = compute_metrics(preds)
    metrics["config"] = "B3"; metrics["fold"] = fold
    metrics["train_losses"] = ckpt.get("train_losses", []) + losses
    metrics["continued_from_epoch"] = start_epoch

    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "normalization": norm, "config": "B3", "fold": fold, "epoch": start_epoch + epochs},
                str(output_dir / "checkpoint.pt"))
    (output_dir / "predictions.json").write_text(json.dumps(preds) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    entries = load_cache(args.cache_root)
    metrics = continue_training(args.input_dir, args.output_dir, entries, args.fold, args.epochs)
    print(f"DONE: AUROC={metrics.get('auroc',0):.4f} BACC={metrics.get('balanced_accuracy',0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
