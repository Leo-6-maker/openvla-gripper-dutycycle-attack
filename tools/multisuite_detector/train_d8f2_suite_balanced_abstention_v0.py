#!/usr/bin/env python3
"""D8F2 — Suite-balanced selective abstention detector v0.

Same three-head architecture as D8F1, but with:
  - Equal suite sampling per batch (macro-objective)
  - L10 positive window weight boost (rescue L10 recall)
  - Suite-stratified hard negative mining
  - Constrained macro-F1 checkpoint selection (L10 recall >= 45.6%)

Goal: diagnose whether D8F1's L10 failure is recoverable in 25D space.

CPU-only. No env.step, no OpenVLA, no MuJoCo.
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import defaultdict
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
    def __init__(self, nf=25, nc=108, hidden=128, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_emit = nn.Linear(hidden + nc, 1)
        self.head_suppress = nn.Linear(hidden + nc, 1)
        self.head_abstain = nn.Linear(hidden + nc, 1)

    def forward(self, xt, xc):
        _, h = self.gru(xt)
        feats = self.drop(torch.cat([h[-1], xc], dim=1))
        return self.head_emit(feats), self.head_suppress(feats), self.head_abstain(feats)


def abstention_loss(el, sl, al, y, suite_mask_l10, fp_weight=0.5, abstain_w=0.3, l10_pos_w=2.0):
    """Suite-aware abstention loss: extra weight on L10 positives."""
    bce_emit = nn.functional.binary_cross_entropy_with_logits(el.squeeze(-1), y.float(), reduction="none")
    # Boost L10 positive emit loss
    l10_pos = suite_mask_l10 & (y == 1)
    if l10_pos.any():
        bce_emit[l10_pos] *= l10_pos_w
    bce_emit = bce_emit.mean()

    bce_suppress = nn.functional.binary_cross_entropy_with_logits(sl.squeeze(-1), (1 - y.float()))

    emit_prob = torch.sigmoid(el.squeeze(-1))
    fp_penalty = (emit_prob * (1 - y.float())).mean() * fp_weight

    with torch.no_grad():
        hard_neg = ((y == 0) & (emit_prob > 0.25)).float()
    loss_abstain = nn.functional.binary_cross_entropy_with_logits(al.squeeze(-1), hard_neg)

    loss = bce_emit + 0.5 * bce_suppress + fp_penalty + abstain_w * loss_abstain
    return loss


def compute_metrics(model, xt, xc, y, suites, tau_emit=0.33, tau_suppress=0.67, tau_abstain=0.5,
                    batch_size=256, device=torch.device("cpu")):
    model.eval()
    all_ep, all_sp, all_ap = [], [], []
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            el, sl, al = model(torch.from_numpy(xt[i:i+batch_size]).to(device),
                               torch.from_numpy(xc[i:i+batch_size]).to(device))
            all_ep.append(torch.sigmoid(el).cpu().numpy())
            all_sp.append(torch.sigmoid(sl).cpu().numpy())
            all_ap.append(torch.sigmoid(al).cpu().numpy())
    ep = np.concatenate(all_ep).flatten()
    sp = np.concatenate(all_sp).flatten()
    ap = np.concatenate(all_ap).flatten()
    emitted = (ep >= tau_emit) & (sp <= tau_suppress) & (ap < tau_abstain)
    abstained = ap >= tau_abstain

    def _per_suite(sm):
        tp = int(np.sum(emitted[sm] & (y[sm] == 1)))
        fp = int(np.sum(emitted[sm] & (y[sm] == 0)))
        fn = int(np.sum((~emitted)[sm] & (y[sm] == 1)))
        tn = int(np.sum((~emitted)[sm] & (y[sm] == 0)))
        return {"recall": tp/max(1,tp+fn), "fp_rate": fp/max(1,fp+tn),
                "f1": 2*tp/max(1,2*tp+fp+fn), "tp": tp, "fp": fp}

    tp = int(np.sum(emitted & (y == 1)))
    fp = int(np.sum(emitted & (y == 0)))
    fn = int(np.sum((~emitted) & (y == 1)))
    tn = int(np.sum((~emitted) & (y == 0)))
    per_suite = {}
    suite_recalls = []
    for s in np.unique(suites):
        sm = suites == s
        per_suite[str(s)] = _per_suite(sm)
        suite_recalls.append(per_suite[str(s)]["recall"])
    return {
        "recall": tp/max(1,tp+fn), "fp_rate": fp/max(1,fp+tn),
        "f1": 2*tp/max(1,2*tp+fp+fn),
        "abstain_rate": float(np.mean(abstained)),
        "emission_rate": float(np.mean(emitted)),
        "macro_recall": float(np.mean(suite_recalls)),
        "per_suite": per_suite,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser(description="D8F2 Suite-Balanced Abstention v0")
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fp-penalty", type=float, default=0.5)
    ap.add_argument("--abstain-weight", type=float, default=0.3)
    ap.add_argument("--l10-pos-weight", type=float, default=2.0)
    ap.add_argument("--tau-emit", type=float, default=0.33)
    ap.add_argument("--tau-suppress", type=float, default=0.67)
    ap.add_argument("--tau-abstain", type=float, default=0.5)
    ap.add_argument("--min-l10-recall", type=float, default=0.456)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    # Load data
    c2e1 = Path(args.c2e1_root); w = args.window
    npz = np.load(c2e1 / f"c2e1_w{w:02d}_temporal_dataset.npz", allow_pickle=True)
    stats = load_stats(c2e1 / f"c2e1_w{w:02d}_normalization_stats_train_only.json")
    xt_all, xc_all = normalize_data(np.asarray(npz["X_temporal"], dtype=np.float32),
                                     np.asarray(npz["X_context"], dtype=np.float32), stats)
    y_all = npz["y"].astype(np.int64)
    suite_all = np.asarray(npz["suite"]).astype(str)
    split_all = np.asarray(npz["split"]).astype(str)

    print(f"D8F2: {len(xt_all)} windows, n_context={xc_all.shape[1]}")

    train_mask = split_all == "train"; val_mask = split_all == "val"
    xt_train, xc_train, y_train = xt_all[train_mask], xc_all[train_mask], y_all[train_mask]
    xt_val, xc_val, y_val = xt_all[val_mask], xc_all[val_mask], y_all[val_mask]
    suite_train = suite_all[train_mask]; suite_val = suite_all[val_mask]

    # Suite-stratified indices for balanced sampling
    suite_indices = defaultdict(list)
    for i, s in enumerate(suite_train):
        suite_indices[str(s)].append(i)
    suite_names = sorted(suite_indices.keys())
    min_suite_size = min(len(v) for v in suite_indices.values())

    model = ThreeHeadGRU(25, xc_all.shape[1], args.hidden, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    best_state = None; best_state_fp25 = None; best_state_fp30 = None; best_state_l10 = None
    best_f1 = 0.0; best_recall_fp25 = 0.0; best_recall_fp30 = 0.0; best_l10_recall = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0

        # Suite-balanced sampling
        for bi in range(0, min_suite_size, args.batch_size // len(suite_names)):
            batch_indices = []
            for s in suite_names:
                perm = np.random.permutation(suite_indices[s])
                n_take = min(args.batch_size // len(suite_names), len(perm))
                batch_indices.extend(perm[:n_take])
            np.random.shuffle(batch_indices)

            bx = torch.from_numpy(xt_train[batch_indices]).to(device)
            bc = torch.from_numpy(xc_train[batch_indices]).to(device)
            by = torch.from_numpy(y_train[batch_indices]).to(device)
            bs_l10 = torch.from_numpy((suite_train[batch_indices] == "libero_10")).to(device)

            el, sl, al = model(bx, bc)
            loss = abstention_loss(el, sl, al, by, bs_l10,
                                   args.fp_penalty, args.abstain_weight, args.l10_pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        val_m = compute_metrics(model, xt_val, xc_val, y_val, suite_val,
                                args.tau_emit, args.tau_suppress, args.tau_abstain, device=device)
        l10_recall = val_m["per_suite"].get("libero_10", {}).get("recall", 0.0)

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if val_m["fp_rate"] <= 0.25 and val_m["recall"] > best_recall_fp25:
            best_recall_fp25 = val_m["recall"]; best_state_fp25 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if val_m["fp_rate"] <= 0.30 and val_m["recall"] > best_recall_fp30:
            best_recall_fp30 = val_m["recall"]; best_state_fp30 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if l10_recall >= args.min_l10_recall and l10_recall > best_l10_recall:
            best_l10_recall = l10_recall; best_state_l10 = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  epoch {epoch:3d}: loss={avg_loss:.4f} val_f1={val_m['f1']:.4f} "
                  f"fp={val_m['fp_rate']:.4f} rec={val_m['recall']:.4f} "
                  f"L10_rec={l10_recall:.4f} macro_rec={val_m['macro_recall']:.4f} "
                  f"abstain={val_m['abstain_rate']:.3f}", flush=True)

    # Final evaluation
    def _eval(st):
        if st is not None: model.load_state_dict(st)
        return compute_metrics(model, xt_val, xc_val, y_val, suite_val,
                               args.tau_emit, args.tau_suppress, args.tau_abstain, device=device)

    v_f1 = _eval(best_state); v_fp25 = _eval(best_state_fp25)
    v_fp30 = _eval(best_state_fp30); v_l10 = _eval(best_state_l10)

    checkpoint = {
        "model_state_dict": best_state,
        "model_state_dict_l10": best_state_l10,
        "config": {"model": "ThreeHeadGRU_suite_balanced", "window": w, "hidden": args.hidden,
                   "l10_pos_weight": args.l10_pos_weight, "min_l10_recall": args.min_l10_recall,
                   "seed": args.seed, "n_features": 25, "n_context": int(xc_all.shape[1])},
        "val_best_f1": {k: v for k, v in v_f1.items() if k != "per_suite"},
        "val_best_l10": {k: v for k, v in v_l10.items() if k != "per_suite"} if best_state_l10 else None,
    }
    torch.save(checkpoint, str(out / "d8f2_suite_balanced.pt"))

    report = {
        "gate": "D8F2_SUITE_BALANCED_ABSTENTION_V0",
        "status": "PASS_D8F2_TRAINED",
        "best_f1": {"recall": v_f1["recall"], "fp": v_f1["fp_rate"], "f1": v_f1["f1"],
                     "macro_recall": v_f1["macro_recall"], "per_suite": v_f1["per_suite"]},
        "best_l10": {"recall": v_l10["recall"], "fp": v_l10["fp_rate"], "f1": v_l10["f1"],
                      "macro_recall": v_l10["macro_recall"], "per_suite": v_l10["per_suite"]} if best_state_l10 else None,
        "l10_rescue_possible": best_state_l10 is not None,
        "created_at_unix": time.time(), "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
    }
    with open(out / "d8f2_training_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    print(f"\nD8F2 done: best_f1 rec={v_f1['recall']:.4f} fp={v_f1['fp_rate']:.4f} "
          f"L10_rec={v_f1['per_suite'].get('libero_10',{}).get('recall',0):.4f} "
          f"macro_rec={v_f1['macro_recall']:.4f}")
    print(f"  L10 rescue: {'POSSIBLE' if best_state_l10 else 'NOT POSSIBLE'} "
          f"(min L10 recall threshold={args.min_l10_recall})")
    for s in sorted(v_f1["per_suite"].keys()):
        print(f"  {s}: rec={v_f1['per_suite'][s]['recall']:.4f} fp={v_f1['per_suite'][s]['fp_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
