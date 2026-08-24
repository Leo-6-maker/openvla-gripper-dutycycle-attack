"""D8-2R: Learnability diagnosis — tiny overfit, train/val metrics, AUROC, logit quantiles."""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_train_core import (
    D8StudentDetector, create_model, compute_normalization, apply_normalization,
    compute_loss, FEATURE_DIM,
)
from audit_r3_contact_input import sha256_file, verify_seal

FOLD = 0


def load_cache(cache_root: Path) -> list[dict]:
    verify_seal(cache_root)
    entries = []
    for ep_file in sorted((cache_root / "per_episode").iterdir()):
        if ep_file.suffix == ".json":
            entries.extend(json.loads(ep_file.read_text("utf-8")))
    return entries


def run_diagnosis(cache_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_cache(cache_root)

    train_entries = [e for e in entries if e["fold_id"] != FOLD and e["effective_mask"]]
    val_entries = [e for e in entries if e["fold_id"] == FOLD and e["effective_mask"]]

    pos_train = [e for e in train_entries if e["physical_target"] == 1.0]
    neg_train = [e for e in train_entries if e["physical_target"] == 0.0]
    print(f"Train: {len(pos_train)} pos, {len(neg_train)} neg (ratio 1:{len(neg_train)/max(len(pos_train),1):.1f})")

    # ── R1: Tiny overfit ──
    print("\n=== R1: Tiny Overfit ===")
    np.random.seed(42)
    tiny_pos = np.random.choice(pos_train, min(500, len(pos_train)), replace=False)
    tiny_neg = np.random.choice(neg_train, 500, replace=False)
    tiny = list(tiny_pos) + list(tiny_neg)
    np.random.shuffle(tiny)

    X_tiny = torch.tensor([e["features_25d_raw"] for e in tiny], dtype=torch.float32)
    y_tiny = torch.tensor([e["physical_target"] for e in tiny], dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_tiny, y_tiny = X_tiny.to(device), y_tiny.to(device)

    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(42)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tiny)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tiny)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 49:
            with torch.no_grad():
                acc = ((logits > 0) == y_tiny.bool()).float().mean()
            print(f"  epoch {epoch+1}: loss={float(loss):.4f} acc={float(acc):.4f}")

    model.eval()
    with torch.no_grad():
        logits_tiny = model(X_tiny)
    auroc = roc_auc_score(y_tiny.cpu(), logits_tiny.cpu())
    auprc = average_precision_score(y_tiny.cpu(), logits_tiny.cpu())
    bacc = ((logits_tiny > 0).cpu() == y_tiny.cpu().bool()).float().mean()
    print(f"  R1 AUROC={auroc:.4f} AUPRC={auprc:.4f} BACC={float(bacc):.4f}")
    r1_pass = auroc > 0.99 and float(bacc) > 0.95

    # ── R2: Per-fold train/val metrics ──
    print("\n=== R2: Train/Val Split Metrics ===")
    norm = compute_normalization(torch.tensor([e["features_25d_raw"] for e in train_entries], dtype=torch.float32))

    configs_to_test = {
        "uniform": torch.ones(len(train_entries)),
        "event_weight": torch.tensor([e["D8_weight"] for e in train_entries], dtype=torch.float32),
    }

    for cfg_name, w_tr in configs_to_test.items():
        print(f"\n  Config: {cfg_name}")
        X_tr = torch.tensor([e["features_25d_raw"] for e in train_entries], dtype=torch.float32)
        y_tr = torch.tensor([e["physical_target"] for e in train_entries], dtype=torch.float32)
        X_va = torch.tensor([e["features_25d_raw"] for e in val_entries], dtype=torch.float32)
        y_va = torch.tensor([e["physical_target"] for e in val_entries], dtype=torch.float32)

        # Normalize the weight to mean=1 for stability
        w_tr = w_tr / w_tr.mean()

        model = create_model().to(device)
        X_tr, y_tr, w_tr_dev = X_tr.to(device), y_tr.to(device), w_tr.to(device)
        X_va, y_va = X_va.to(device), y_va.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(42)

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            logits = model(apply_normalization(X_tr, norm))
            loss = compute_loss(logits, y_tr, w_tr_dev)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(apply_normalization(X_tr, norm))
            val_logits = model(apply_normalization(X_va, norm))

        for split, logits_t, y_t in [("train", train_logits, y_tr), ("val", val_logits, y_va)]:
            y_np = y_t.cpu().numpy()
            logits_np = logits_t.cpu().numpy()
            auroc_val = roc_auc_score(y_np, logits_np) if len(set(y_np)) > 1 else float("nan")
            auprc_val = average_precision_score(y_np, logits_np) if len(set(y_np)) > 1 else float("nan")
            preds = (logits_np > 0).astype(float)
            tp = int(((y_np == 1) & (preds == 1)).sum())
            tn = int(((y_np == 0) & (preds == 0)).sum())
            fp = int(((y_np == 0) & (preds == 1)).sum())
            fn = int(((y_np == 1) & (preds == 0)).sum())
            bacc_val = ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2
            pp = (tp + fp) / max(tp + tn + fp + fn, 1)

            q10, q50, q90 = np.percentile(logits_np, [10, 50, 90])
            p_pos = np.percentile(logits_np[y_np == 1], [10, 50, 90]) if (y_np == 1).sum() > 0 else [np.nan]*3
            p_neg = np.percentile(logits_np[y_np == 0], [10, 50, 90]) if (y_np == 0).sum() > 0 else [np.nan]*3

            print(f"    {split:5s}: AUROC={auroc_val:.4f} AUPRC={auprc_val:.4f} BACC={bacc_val:.4f} "
                  f"pred_pos={pp:.3f} TP={tp} TN={tn} FP={fp} FN={fn}")
            print(f"    {split:5s} logit q10/50/90: {q10:.3f} / {q50:.3f} / {q90:.3f}")
            print(f"    {split:5s} pos logit q10/50/90: {p_pos[0]:.3f} / {p_pos[1]:.3f} / {p_pos[2]:.3f}")
            print(f"    {split:5s} neg logit q10/50/90: {p_neg[0]:.3f} / {p_neg[1]:.3f} / {p_neg[2]:.3f}")

    # ── Save results ──
    results = {"r1": {"auroc": auroc, "auprc": auprc, "bacc": float(bacc), "pass": r1_pass}}
    (output_dir / "diagnosis.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*50}")
    print(f"R1 tiny overfit: {'PASS' if r1_pass else 'FAIL'} (AUROC={auroc:.4f})")
    print(f"{'='*50}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnosis(args.cache_root, args.output_dir)
