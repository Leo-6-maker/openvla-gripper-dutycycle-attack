#!/usr/bin/env python3
"""D8F1 — Selective abstention detector v0.

Three-head model: emit_p, suppress_p, abstain_p.
Objective: maximize recall subject to FP <= 25–30% via abstention gate.

Unlike C2e3 (which emits whenever emit_p >= tau AND suppress_p <= tau_suppress),
this model can abstain on ambiguous windows, trading coverage for precision.

CPU-only training. No env.step, no OpenVLA, no MuJoCo.
Uses C2e1/C2e2 clean-only dataset with teacher labels.
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stats(path: Path) -> dict:
    obj = json.loads(path.read_text())
    return {
        "temporal_mean": np.asarray(obj["temporal_feature_mean"], dtype=np.float32),
        "temporal_std": np.asarray(obj["temporal_feature_std"], dtype=np.float32),
        "context_mean": np.asarray(obj["context_feature_mean"], dtype=np.float32),
        "context_std": np.asarray(obj["context_feature_std"], dtype=np.float32),
    }


def normalize_data(xt: np.ndarray, xc: np.ndarray, stats: dict) -> Tuple[np.ndarray, np.ndarray]:
    tm = stats["temporal_mean"].reshape(1, 1, -1)
    ts = np.maximum(stats["temporal_std"].reshape(1, 1, -1), 1e-8)
    xt = (xt.astype(np.float32) - tm) / ts
    if xc.shape[1] > 0:
        cm = stats["context_mean"].reshape(1, -1)
        cs = np.maximum(stats["context_std"].reshape(1, -1), 1e-8)
        xc = (xc.astype(np.float32) - cm) / cs
    return xt, xc


class ThreeHeadGRU(nn.Module):
    """GRU with emit/suppress/abstain heads."""
    def __init__(self, nf: int = 25, nc: int = 108, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_emit = nn.Linear(hidden + nc, 1)
        self.head_suppress = nn.Linear(hidden + nc, 1)
        self.head_abstain = nn.Linear(hidden + nc, 1)

    def forward(self, xt: torch.Tensor, xc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h = self.gru(xt)
        feats = self.drop(torch.cat([h[-1], xc], dim=1))
        emit_logit = self.head_emit(feats)
        suppress_logit = self.head_suppress(feats)
        abstain_logit = self.head_abstain(feats)
        return emit_logit, suppress_logit, abstain_logit


def abstention_loss(
    emit_logit: torch.Tensor,
    suppress_logit: torch.Tensor,
    abstain_logit: torch.Tensor,
    y: torch.Tensor,
    fp_penalty_weight: float = 0.5,
    abstain_penalty: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss with FP penalty and abstain regularization.

    - BCE on emit (hazard=1, safe=0)
    - BCE on suppress (release/unsafe=1, stable=0)
    - FP penalty: when y=0 but emit_p is high, penalize
    - Abstain: when model is uncertain (abstain_p high), reduce emit pressure
    """
    bce_emit = nn.functional.binary_cross_entropy_with_logits(emit_logit.squeeze(-1), y.float())
    bce_suppress = nn.functional.binary_cross_entropy_with_logits(
        suppress_logit.squeeze(-1), (1 - y.float()))  # suppress for non-hazard

    # FP penalty: penalize high emit probability on negative examples
    emit_prob = torch.sigmoid(emit_logit.squeeze(-1))
    fp_penalty = (emit_prob * (1 - y.float())).mean() * fp_penalty_weight

    # Abstain regularization: encourage abstain to be low overall
    abstain_prob = torch.sigmoid(abstain_logit.squeeze(-1))
    abstain_reg = abstain_prob.mean() * abstain_penalty

    loss = bce_emit + 0.5 * bce_suppress + fp_penalty + abstain_reg

    with torch.no_grad():
        info = {
            "bce_emit": float(bce_emit),
            "bce_suppress": float(bce_suppress),
            "fp_penalty": float(fp_penalty),
            "abstain_reg": float(abstain_reg),
            "loss": float(loss),
        }
    return loss, info


def compute_metrics(
    model: nn.Module,
    xt: np.ndarray, xc: np.ndarray, y: np.ndarray,
    suites: np.ndarray,
    tau_emit: float = 0.33, tau_suppress: float = 0.67, tau_abstain: float = 0.5,
    batch_size: int = 256, device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Compute detection metrics with abstention gate.

    Decision: emit if emit_p >= tau AND suppress_p <= tau_suppress AND abstain_p < tau_abstain
    """
    model.eval()
    all_emit_p = []
    all_suppress_p = []
    all_abstain_p = []

    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            bx = torch.from_numpy(xt[i:i+batch_size]).to(device)
            bc = torch.from_numpy(xc[i:i+batch_size]).to(device)
            el, sl, al = model(bx, bc)
            all_emit_p.append(torch.sigmoid(el).cpu().numpy())
            all_suppress_p.append(torch.sigmoid(sl).cpu().numpy())
            all_abstain_p.append(torch.sigmoid(al).cpu().numpy())

    ep = np.concatenate(all_emit_p).flatten()
    sp = np.concatenate(all_suppress_p).flatten()
    ap = np.concatenate(all_abstain_p).flatten()

    # Decision: emit if all three conditions met
    emitted = (ep >= tau_emit) & (sp <= tau_suppress) & (ap < tau_abstain)
    abstained = ap >= tau_abstain

    tp = int(np.sum(emitted & (y == 1)))
    fp = int(np.sum(emitted & (y == 0)))
    fn = int(np.sum((~emitted) & (y == 1)))
    tn = int(np.sum((~emitted) & (y == 0)))
    n_abstain = int(np.sum(abstained))

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    fp_rate = fp / max(1, fp + tn)
    f1 = 2 * recall * precision / max(1e-8, recall + precision)

    # Per-suite
    suite_metrics = {}
    for s in np.unique(suites):
        sm = suites == s
        tp_s = int(np.sum(emitted[sm] & (y[sm] == 1)))
        fp_s = int(np.sum(emitted[sm] & (y[sm] == 0)))
        fn_s = int(np.sum((~emitted)[sm] & (y[sm] == 1)))
        tn_s = int(np.sum((~emitted)[sm] & (y[sm] == 0)))
        suite_metrics[str(s)] = {
            "recall": tp_s / max(1, tp_s + fn_s),
            "fp_rate": fp_s / max(1, fp_s + tn_s),
            "f1": 2 * tp_s / max(1, 2 * tp_s + fp_s + fn_s),
        }

    return {
        "recall": recall, "precision": precision, "fp_rate": fp_rate, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_abstain": n_abstain, "abstain_rate": n_abstain / max(1, len(y)),
        "emission_rate": float(np.mean(emitted)),
        "per_suite": suite_metrics,
    }


def main():
    ap = argparse.ArgumentParser(description="D8F1 Selective Abstention Detector v0")
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fp-penalty", type=float, default=0.5)
    ap.add_argument("--abstain-penalty", type=float, default=0.3)
    ap.add_argument("--tau-emit", type=float, default=0.33)
    ap.add_argument("--tau-suppress", type=float, default=0.67)
    ap.add_argument("--tau-abstain", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    # ── Load data ──
    c2e1 = Path(args.c2e1_root)
    w = args.window
    npz = np.load(c2e1 / f"c2e1_w{w:02d}_temporal_dataset.npz", allow_pickle=True)
    stats = load_stats(c2e1 / f"c2e1_w{w:02d}_normalization_stats_train_only.json")

    xt_all, xc_all = normalize_data(
        np.asarray(npz["X_temporal"], dtype=np.float32),
        np.asarray(npz["X_context"], dtype=np.float32),
        stats,
    )
    y_all = npz["y"].astype(np.int64)
    suite_all = np.asarray(npz["suite"]).astype(str)
    split_all = np.asarray(npz["split"]).astype(str)

    print(f"D8F1: {len(xt_all)} windows, n_context={xc_all.shape[1]}")

    train_mask = split_all == "train"
    val_mask = split_all == "val"

    xt_train, xc_train, y_train = xt_all[train_mask], xc_all[train_mask], y_all[train_mask]
    xt_val, xc_val, y_val = xt_all[val_mask], xc_all[val_mask], y_all[val_mask]
    suite_val = suite_all[val_mask]

    # ── Model ──
    model = ThreeHeadGRU(nf=25, nc=xc_all.shape[1], hidden=args.hidden,
                         dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    # ── Training ──
    best_val_f1 = 0.0
    best_state = None
    n_train = len(xt_train)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        indices = np.random.permutation(n_train)

        for bi in range(0, n_train, args.batch_size):
            batch_idx = indices[bi:bi + args.batch_size]
            bx = torch.from_numpy(xt_train[batch_idx]).to(device)
            bc = torch.from_numpy(xc_train[batch_idx]).to(device)
            by = torch.from_numpy(y_train[batch_idx]).to(device)

            el, sl, al = model(bx, bc)
            loss, loss_info = abstention_loss(el, sl, al, by,
                                              args.fp_penalty, args.abstain_penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation
        val_m = compute_metrics(model, xt_val, xc_val, y_val, suite_val,
                                tau_emit=args.tau_emit, tau_suppress=args.tau_suppress,
                                tau_abstain=args.tau_abstain, device=device)

        is_best = val_m["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or is_best:
            print(f"  epoch {epoch:3d}: loss={avg_loss:.4f} "
                  f"val_recall={val_m['recall']:.4f} val_fp={val_m['fp_rate']:.4f} "
                  f"val_f1={val_m['f1']:.4f} abstain={val_m['abstain_rate']:.3f}"
                  f"{' *' if is_best else ''}",
                  flush=True)

    # ── Save ──
    val_final = compute_metrics(model, xt_val, xc_val, y_val, suite_val,
                                tau_emit=args.tau_emit, tau_suppress=args.tau_suppress,
                                tau_abstain=args.tau_abstain, device=device)

    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "model": "ThreeHeadGRU",
            "window": w, "hidden": args.hidden, "dropout": args.dropout,
            "lr": args.lr, "fp_penalty": args.fp_penalty, "abstain_penalty": args.abstain_penalty,
            "tau_emit": args.tau_emit, "tau_suppress": args.tau_suppress, "tau_abstain": args.tau_abstain,
            "seed": args.seed, "n_features": 25, "n_context": int(xc_all.shape[1]),
        },
        "val_metrics": {k: v for k, v in val_final.items() if k != "per_suite"},
    }
    torch.save(checkpoint, str(out / "d8f1_selective_abstention.pt"))

    report = {
        "gate": "D8F1_SELECTIVE_ABSTENTION_V0",
        "status": "PASS_D8F1_TRAINED",
        "val_recall": val_final["recall"],
        "val_fp_rate": val_final["fp_rate"],
        "val_f1": val_final["f1"],
        "val_abstain_rate": val_final["abstain_rate"],
        "val_per_suite": val_final["per_suite"],
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "note": "three_head_abstention — can refuse to emit on ambiguous windows, trading FP for recall",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    with open(out / "d8f1_training_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    print(f"\nD8F1 done: val_recall={val_final['recall']:.4f} val_fp={val_final['fp_rate']:.4f} "
          f"val_f1={val_final['f1']:.4f}  saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
