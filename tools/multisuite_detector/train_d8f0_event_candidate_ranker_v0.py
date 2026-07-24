#!/usr/bin/env python3
"""D8F0 — Event candidate ranker v0.

Instead of per-frame binary classification (C2e3), this trains a model to
rank candidate events within each trajectory:

  positive candidate = teacher-labelled attackable window (primary event)
  negative candidate = non-attackable window (setup/auxiliary/distractor)

Loss: margin ranking loss — positive score > all negative scores + margin.

CPU-only training. No env.step, no OpenVLA, no MuJoCo.
Uses C2e1/C2e2 clean-only dataset with teacher labels.
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


class WindowEncoder(nn.Module):
    """Encodes a W×25 window into a fixed-size embedding."""
    def __init__(self, nf: int = 25, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, 25)
        _, h = self.gru(x)  # h: (1, B, hidden)
        return self.drop(h[-1])  # (B, hidden)


class CandidateScorer(nn.Module):
    """Scores a candidate window given its embedding and context."""
    def __init__(self, window_dim: int = 64, context_dim: int = 108, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, window_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # window_emb: (B, window_dim), context: (B, context_dim)
        return self.net(torch.cat([window_emb, context], dim=1)).squeeze(-1)  # (B,)


class EventCandidateRanker(nn.Module):
    """Full event candidate ranking model."""
    def __init__(self, nf: int = 25, nc: int = 108, window_hidden: int = 64,
                 scorer_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = WindowEncoder(nf, window_hidden, dropout)
        self.scorer = CandidateScorer(window_hidden, nc, scorer_hidden, dropout)

    def forward(self, x_windows: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """Score a batch of candidate windows. Returns scalar scores (B,)."""
        emb = self.encoder(x_windows)
        return self.scorer(emb, x_context)


def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """For each positive, loss = max(0, margin - pos_score + max_neg_score)."""
    if neg_scores.numel() == 0:
        return pos_scores.sum() * 0.0  # keep grad, zero loss
    max_neg = neg_scores.max()
    # For each positive, penalize if max_neg_score + margin > pos_score
    losses = torch.clamp(margin + max_neg - pos_scores, min=0.0)
    return losses.mean()


def build_candidate_groups(
    xt: np.ndarray, xc: np.ndarray, y: np.ndarray,
    row_indices: np.ndarray, suites: np.ndarray,
) -> List[Dict[str, Any]]:
    """Group windows by trajectory and extract positive/negative candidates.

    Returns list of dicts: {pos_windows, pos_contexts, neg_windows, neg_contexts, suite, row_index}
    """
    groups: Dict[int, Dict[str, list]] = defaultdict(lambda: {
        "pos_windows": [], "pos_contexts": [],
        "neg_windows": [], "neg_contexts": [],
        "suite": "",
    })

    for i in range(len(xt)):
        ri = int(row_indices[i])
        g = groups[ri]
        g["suite"] = str(suites[i])
        if y[i] == 1:
            g["pos_windows"].append(xt[i])
            g["pos_contexts"].append(xc[i])
        else:
            g["neg_windows"].append(xt[i])
            g["neg_contexts"].append(xc[i])

    candidates = []
    for ri, g in groups.items():
        if len(g["pos_windows"]) == 0:
            continue  # skip trajectories with no positive events
        candidates.append({
            "row_index": ri,
            "suite": g["suite"],
            "pos_windows": np.stack(g["pos_windows"]),
            "pos_contexts": np.stack(g["pos_contexts"]),
            "neg_windows": np.stack(g["neg_windows"]) if g["neg_windows"] else np.zeros((0, 16, 25), dtype=np.float32),
            "neg_contexts": np.stack(g["neg_contexts"]) if g["neg_contexts"] else np.zeros((0, 108), dtype=np.float32),
            "n_pos": len(g["pos_windows"]),
            "n_neg": len(g["neg_windows"]),
        })
    return candidates


def evaluate_ranking(
    model: nn.Module,
    candidates: List[Dict],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ranking metrics: recall@1, mean reciprocal rank, precision@k."""
    model.eval()
    total_pos = 0
    recalled_at_1 = 0
    reciprocal_ranks = []

    with torch.no_grad():
        for c in candidates:
            pos_w = torch.from_numpy(c["pos_windows"]).to(device)
            pos_c = torch.from_numpy(c["pos_contexts"]).to(device)
            neg_w = torch.from_numpy(c["neg_windows"]).to(device)
            neg_c = torch.from_numpy(c["neg_contexts"]).to(device)

            pos_scores = model(pos_w, pos_c)
            all_w = torch.cat([pos_w, neg_w], dim=0) if neg_w.shape[0] > 0 else pos_w
            all_c = torch.cat([pos_c, neg_c], dim=0) if neg_c.shape[0] > 0 else pos_c
            all_scores = model(all_w, all_c)

            n_pos = pos_w.shape[0]
            total_pos += n_pos

            # Recall@1: top-scoring candidate is positive
            top_idx = int(torch.argmax(all_scores).item())
            if top_idx < n_pos:
                recalled_at_1 += 1

            # MRR
            sorted_indices = torch.argsort(all_scores, descending=True).cpu().numpy()
            for pi in range(n_pos):
                rank = int(np.where(sorted_indices == pi)[0][0]) + 1
                reciprocal_ranks.append(1.0 / rank)

    return {
        "recall_at_1": recalled_at_1 / max(1, len(candidates)),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "total_candidates": len(candidates),
        "total_pos_windows": total_pos,
    }


def main():
    ap = argparse.ArgumentParser(description="D8F0 Event Candidate Ranker v0")
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--window-hidden", type=int, default=64)
    ap.add_argument("--scorer-hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-groups", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.5)
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
    row_all = npz["row_index"].astype(np.int64)
    suite_all = np.asarray(npz["suite"]).astype(str)
    split_all = np.asarray(npz["split"]).astype(str)

    print(f"D8F0: {len(xt_all)} windows, {len(np.unique(row_all))} trajectories")

    # ── Split ──
    train_mask = split_all == "train"
    val_mask = split_all == "val"

    train_candidates = build_candidate_groups(
        xt_all[train_mask], xc_all[train_mask], y_all[train_mask],
        row_all[train_mask], suite_all[train_mask],
    )
    val_candidates = build_candidate_groups(
        xt_all[val_mask], xc_all[val_mask], y_all[val_mask],
        row_all[val_mask], suite_all[val_mask],
    )
    print(f"  Train: {len(train_candidates)} trajectories with positive events")
    print(f"  Val:   {len(val_candidates)} trajectories with positive events")

    # ── Model ──
    model = EventCandidateRanker(
        nf=25, nc=xc_all.shape[1],
        window_hidden=args.window_hidden,
        scorer_hidden=args.scorer_hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    # ── Training ──
    best_val_mrr = 0.0
    best_state = None
    history: List[Dict] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        indices = np.random.permutation(len(train_candidates))

        for bi in range(0, len(indices), args.batch_groups):
            batch_indices = indices[bi:bi + args.batch_groups]
            losses = []
            n_valid = 0

            for idx in batch_indices:
                c = train_candidates[idx]
                if c["n_pos"] == 0:
                    continue
                pos_w = torch.from_numpy(c["pos_windows"]).to(device)
                pos_c = torch.from_numpy(c["pos_contexts"]).to(device)
                neg_w = torch.from_numpy(c["neg_windows"]).to(device)
                neg_c = torch.from_numpy(c["neg_contexts"]).to(device)

                pos_scores = model(pos_w, pos_c)
                neg_scores = model(neg_w, neg_c) if neg_w.shape[0] > 0 else torch.tensor([], device=device)
                loss = margin_ranking_loss(pos_scores, neg_scores, args.margin)
                losses.append(loss)
                n_valid += 1

            if n_valid > 0:
                batch_loss = torch.stack(losses).sum() / n_valid
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation
        val_metrics = evaluate_ranking(model, val_candidates, device)
        history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

        is_best = val_metrics["mrr"] > best_val_mrr
        if is_best:
            best_val_mrr = val_metrics["mrr"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or is_best:
            print(f"  epoch {epoch:3d}: loss={avg_loss:.4f} "
                  f"val_recall@1={val_metrics['recall_at_1']:.4f} "
                  f"val_mrr={val_metrics['mrr']:.4f}"
                  f"{' *' if is_best else ''}",
                  flush=True)

    # ── Save ──
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "model": "EventCandidateRanker",
            "window": w,
            "window_hidden": args.window_hidden,
            "scorer_hidden": args.scorer_hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "margin": args.margin,
            "seed": args.seed,
            "n_features": 25,
            "n_context": xc_all.shape[1],
        },
        "best_val_mrr": best_val_mrr,
        "history": history,
    }
    torch.save(checkpoint, str(out / "d8f0_event_candidate_ranker.pt"))

    # Report
    report = {
        "gate": "D8F0_EVENT_CANDIDATE_RANKER_V0",
        "status": "PASS_D8F0_TRAINED",
        "best_val_mrr": best_val_mrr,
        "best_val_recall_at_1": float(max(h["recall_at_1"] for h in history)) if history else 0.0,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "note": "event_candidate_ranker_not_frame_classifier — ranking formulation for L10 multi-event structure",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    with open(out / "d8f0_training_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    print(f"\nD8F0 done: best_val_mrr={best_val_mrr:.4f}  saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
