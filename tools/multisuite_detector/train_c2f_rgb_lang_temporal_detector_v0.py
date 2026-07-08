#!/usr/bin/env python3
"""Train C2f RGB+language+25D temporal student detector v0.

Input: NPZ produced by materialize_c2f_frozen_embeddings.py
Model: 25D GRU + RGB embedding MLP + language embedding MLP + context MLP
Heads: emit/hazard, suppress/release, abstain, primary_attackable, event_role.

This is offline clean-only training. It does not run LIBERO/OpenVLA and does not
read D7B2 outcomes. Do not use this detector in D7 Table 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn, optim

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


class C2fDetector(nn.Module):
    def __init__(self, nf: int, nv: int, nl: int, nc: int, hidden: int = 128, proj: int = 128, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.GRU(nf, hidden, 1, batch_first=True)
        self.visual = nn.Sequential(nn.Linear(nv, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, proj), nn.ReLU())
        self.lang = nn.Sequential(nn.Linear(nl, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, proj), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(nc, 64), nn.ReLU())
        fused = hidden + proj + proj + 64
        self.drop = nn.Dropout(dropout)
        self.emit = nn.Linear(fused, 1)
        self.suppress = nn.Linear(fused, 1)
        self.abstain = nn.Linear(fused, 1)
        self.primary = nn.Linear(fused, 1)
        self.role = nn.Linear(fused, 4)

    def forward(self, xt, xv, xl, xc):
        _, h = self.temporal(xt)
        z = torch.cat([h[-1], self.visual(xv), self.lang(xl), self.context(xc)], dim=1)
        z = self.drop(z)
        return {
            "emit": self.emit(z).squeeze(-1),
            "suppress": self.suppress(z).squeeze(-1),
            "abstain": self.abstain(z).squeeze(-1),
            "primary": self.primary(z).squeeze(-1),
            "role": self.role(z),
        }


def batch_iter(indices: np.ndarray, batch_size: int):
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def compute_loss(out: Dict[str, torch.Tensor], y_hazard, y_primary, y_release, y_role, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = nn.functional.binary_cross_entropy_with_logits
    ce = nn.functional.cross_entropy
    loss_emit = bce(out["emit"], y_hazard.float())
    # suppress head learns release/unsafe/non-hazard pressure; release label is the clean teacher proxy
    loss_suppress = bce(out["suppress"], y_release.float())
    loss_primary = bce(out["primary"], y_primary.float())
    loss_role = ce(out["role"], y_role.long())

    emit_prob = torch.sigmoid(out["emit"])
    primary_prob = torch.sigmoid(out["primary"])
    hard_neg = ((y_hazard == 0) & (emit_prob.detach() > args.hard_neg_emit)).float()
    # C2f should abstain on false-positive-prone negatives, not on positives/primary events.
    abstain_target = hard_neg * (1 - y_primary.float())
    loss_abstain = bce(out["abstain"], abstain_target)

    fp_penalty = (emit_prob * (1 - y_hazard.float())).mean() * args.fp_penalty
    primary_fp_penalty = (primary_prob * (1 - y_primary.float())).mean() * args.primary_fp_penalty

    loss = (
        loss_emit
        + args.suppress_weight * loss_suppress
        + args.primary_weight * loss_primary
        + args.role_weight * loss_role
        + args.abstain_weight * loss_abstain
        + fp_penalty
        + primary_fp_penalty
    )
    info = {
        "loss_emit": float(loss_emit.detach()),
        "loss_suppress": float(loss_suppress.detach()),
        "loss_primary": float(loss_primary.detach()),
        "loss_role": float(loss_role.detach()),
        "loss_abstain": float(loss_abstain.detach()),
        "fp_penalty": float(fp_penalty.detach()),
        "loss": float(loss.detach()),
    }
    return loss, info


def metrics_from_probs(ep, sp, ap, pp, y_h, y_p, suites, tau_emit, tau_suppress, tau_abstain, tau_primary):
    emitted = (ep >= tau_emit) & (sp <= tau_suppress) & (ap < tau_abstain) & (pp >= tau_primary)
    tp = int(np.sum(emitted & (y_h == 1)))
    fp = int(np.sum(emitted & (y_h == 0)))
    fn = int(np.sum((~emitted) & (y_h == 1)))
    tn = int(np.sum((~emitted) & (y_h == 0)))
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    fp_rate = fp / max(1, fp + tn)
    f1 = 2 * recall * precision / max(1e-8, recall + precision)
    per_suite = {}
    for s in sorted(set(map(str, suites))):
        m = np.asarray(suites).astype(str) == s
        tp_s = int(np.sum(emitted[m] & (y_h[m] == 1)))
        fp_s = int(np.sum(emitted[m] & (y_h[m] == 0)))
        fn_s = int(np.sum((~emitted[m]) & (y_h[m] == 1)))
        tn_s = int(np.sum((~emitted[m]) & (y_h[m] == 0)))
        per_suite[s] = {
            "recall": tp_s / max(1, tp_s + fn_s),
            "fp_rate": fp_s / max(1, fp_s + tn_s),
            "f1": 2 * tp_s / max(1, 2 * tp_s + fp_s + fn_s),
            "emission_rate": float(np.mean(emitted[m])) if np.any(m) else 0.0,
            "abstain_rate": float(np.mean(ap[m] >= tau_abstain)) if np.any(m) else 0.0,
            "primary_positive_rate": float(np.mean(pp[m] >= tau_primary)) if np.any(m) else 0.0,
        }
    macro_recall = float(np.mean([v["recall"] for v in per_suite.values()])) if per_suite else 0.0
    macro_fp = float(np.mean([v["fp_rate"] for v in per_suite.values()])) if per_suite else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1": f1,
        "macro_recall": macro_recall,
        "macro_fp_rate": macro_fp,
        "abstain_rate": float(np.mean(ap >= tau_abstain)),
        "emission_rate": float(np.mean(emitted)),
        "per_suite": per_suite,
    }


def evaluate(model, data, mask, args, device) -> Dict[str, Any]:
    model.eval()
    probs = {"emit": [], "suppress": [], "abstain": [], "primary": []}
    idx = np.where(mask)[0]
    with torch.no_grad():
        for bi in batch_iter(idx, args.batch_size):
            out = model(
                torch.from_numpy(data["X_temporal"][bi]).to(device),
                torch.from_numpy(data["X_visual"][bi]).to(device),
                torch.from_numpy(data["X_language"][bi]).to(device),
                torch.from_numpy(data["X_context"][bi]).to(device),
            )
            for k in probs:
                probs[k].append(torch.sigmoid(out[k]).cpu().numpy())
    ep = np.concatenate(probs["emit"])
    sp = np.concatenate(probs["suppress"])
    ap = np.concatenate(probs["abstain"])
    pp = np.concatenate(probs["primary"])
    return metrics_from_probs(
        ep, sp, ap, pp,
        data["y_hazard"][idx], data["y_primary"][idx], data["suite"][idx],
        args.tau_emit, args.tau_suppress, args.tau_abstain, args.tau_primary,
    )


def threshold_sweep(model, data, mask, args, device) -> Dict[str, Any]:
    base = []
    for te in [0.25, 0.30, 0.33, 0.40, 0.50]:
        for ts in [0.50, 0.67, 0.75]:
            for ta in [0.30, 0.50, 0.70]:
                for tp in [0.30, 0.50, 0.70]:
                    old = (args.tau_emit, args.tau_suppress, args.tau_abstain, args.tau_primary)
                    args.tau_emit, args.tau_suppress, args.tau_abstain, args.tau_primary = te, ts, ta, tp
                    m = evaluate(model, data, mask, args, device)
                    m.update({"tau_emit": te, "tau_suppress": ts, "tau_abstain": ta, "tau_primary": tp})
                    l10 = m["per_suite"].get("libero_10", {})
                    m["l10_recall"] = l10.get("recall", 0.0)
                    m["l10_fp_rate"] = l10.get("fp_rate", 0.0)
                    base.append(m)
                    args.tau_emit, args.tau_suppress, args.tau_abstain, args.tau_primary = old
    best_f1 = max(base, key=lambda x: x["f1"])
    feasible = [m for m in base if m["fp_rate"] <= 0.30 and m["l10_recall"] >= 0.456]
    best_c2f_gate = max(feasible, key=lambda x: (x["l10_recall"], x["macro_recall"], -x["fp_rate"]), default=None)
    return {"best_f1": best_f1, "best_c2f_gate": best_c2f_gate, "n_sweep": len(base)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train C2f RGB+language+temporal detector v0")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fp-penalty", type=float, default=0.5)
    ap.add_argument("--primary-fp-penalty", type=float, default=0.5)
    ap.add_argument("--suppress-weight", type=float, default=0.5)
    ap.add_argument("--primary-weight", type=float, default=1.0)
    ap.add_argument("--role-weight", type=float, default=0.5)
    ap.add_argument("--abstain-weight", type=float, default=0.3)
    ap.add_argument("--hard-neg-emit", type=float, default=0.25)
    ap.add_argument("--tau-emit", type=float, default=0.33)
    ap.add_argument("--tau-suppress", type=float, default=0.67)
    ap.add_argument("--tau-abstain", type=float, default=0.5)
    ap.add_argument("--tau-primary", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    npz = np.load(args.dataset, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    for k in ["X_temporal", "X_visual", "X_language", "X_context"]:
        data[k] = data[k].astype(np.float32)
    for k in ["y_hazard", "y_primary", "y_release", "y_role"]:
        data[k] = data[k].astype(np.int64)
    split = data["split"].astype(str)
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"

    model = C2fDetector(
        nf=data["X_temporal"].shape[-1],
        nv=data["X_visual"].shape[-1],
        nl=data["X_language"].shape[-1],
        nc=data["X_context"].shape[-1],
        hidden=args.hidden,
        proj=args.proj,
        dropout=args.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_score = -1.0
    history = []
    train_idx = np.where(train_mask)[0]

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_idx)
        losses = []
        for bi in batch_iter(train_idx, args.batch_size):
            out_pred = model(
                torch.from_numpy(data["X_temporal"][bi]).to(device),
                torch.from_numpy(data["X_visual"][bi]).to(device),
                torch.from_numpy(data["X_language"][bi]).to(device),
                torch.from_numpy(data["X_context"][bi]).to(device),
            )
            loss, info = compute_loss(
                out_pred,
                torch.from_numpy(data["y_hazard"][bi]).to(device),
                torch.from_numpy(data["y_primary"][bi]).to(device),
                torch.from_numpy(data["y_release"][bi]).to(device),
                torch.from_numpy(data["y_role"][bi]).to(device),
                args,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val = evaluate(model, data, val_mask, args, device)
        l10 = val["per_suite"].get("libero_10", {"recall": 0.0, "fp_rate": 1.0})
        # C2f selection prioritizes L10 rescue under FP control, not overall F1 only.
        feasible_bonus = 1.0 if val["fp_rate"] <= 0.30 else 0.0
        score = feasible_bonus + l10["recall"] + 0.25 * val["macro_recall"] - 0.25 * val["fp_rate"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{k: v for k, v in val.items() if k != "per_suite"}, "l10_recall": l10["recall"], "l10_fp_rate": l10["fp_rate"]}
        history.append(row)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:03d}: loss={row['loss']:.4f} recall={val['recall']:.3f} fp={val['fp_rate']:.3f} l10_rec={l10['recall']:.3f} macro_rec={val['macro_recall']:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    val_final = evaluate(model, data, val_mask, args, device)
    test_final = evaluate(model, data, test_mask, args, device) if np.any(test_mask) else {}
    sweep = threshold_sweep(model, data, val_mask, args, device)

    ckpt = {
        "model_state_dict": best_state,
        "config": vars(args),
        "dims": {
            "temporal": int(data["X_temporal"].shape[-1]),
            "visual": int(data["X_visual"].shape[-1]),
            "language": int(data["X_language"].shape[-1]),
            "context": int(data["X_context"].shape[-1]),
        },
        "history": history,
        "val_final": val_final,
        "test_final": test_final,
        "threshold_sweep": sweep,
    }
    ckpt_path = out / "c2f_rgb_lang_temporal_detector_v0.pt"
    torch.save(ckpt, str(ckpt_path))
    report = {
        "gate": "C2F_RGB_LANG_TEMPORAL_DETECTOR_V0",
        "status": "TRAINED_NEEDS_AUDIT",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(Path(args.dataset)),
        "val_final": val_final,
        "test_final": test_final,
        "threshold_sweep_summary": {
            "best_f1": {k: v for k, v in sweep["best_f1"].items() if k != "per_suite"},
            "best_c2f_gate": None if sweep["best_c2f_gate"] is None else {k: v for k, v in sweep["best_c2f_gate"].items() if k != "per_suite"},
            "n_sweep": sweep["n_sweep"],
        },
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "boundaries": {
            "attack": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "D7_Table1": "NOT_MODIFIED",
        },
    }
    write_json(out / "c2f_training_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
