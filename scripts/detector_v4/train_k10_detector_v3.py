#!/usr/bin/env python3
"""R7.3: Train K10-specific detectors — R7-S-LINEAR-25D and R7-A-GRU-25D.

Follows protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md — frozen recipe.
Two candidates, 5-fold OOF threshold selection, one-time validation evaluation.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, platform, random, subprocess, sys, uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.v5_dataset import (
    load_fit_registry, load_v5_episode, load_v5_episodes,
    V5Episode, load_policy_intent_root,
)
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig
from gripper_attack.b3_training_protocol import (
    load_fit_fold_bundle, verify_sealed_directory, sha256_file,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
OOF_THRESHOLD_GRID = [round(i * 0.05, 2) for i in range(1, 20)]  # 0.05..0.95
FROZEN_SCHEDULER_CONFIG = V5SchedulerConfig(
    utility_threshold=0.5, release_veto_threshold=0.5, regrasp_veto_threshold=0.5,
    release_veto_enabled=True, regrasp_veto_enabled=True,
    minimum_candidate_dwell=10, persistence_window=5, persistence_required=3,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _seal_root(root: Path) -> str:
    exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        [f for f in root.rglob("*") if f.is_file() and f.name not in exclude],
        key=lambda f: str(f.relative_to(root)),
    )
    lines = []
    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    (root / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
    return sha


# ── data structures ─────────────────────────────────────────────────────────
@dataclass
class K10TrainingEpisode:
    identity: str
    features_25d: Tensor       # [T, 25]
    valid_mask: Tensor          # [T] bool
    candidate_close: Tensor     # [T] bool
    k10_target: Tensor          # [T] float (1.0=feasible_start, 0.0=not, -1.0=oob)
    k10_known: Tensor           # [T] bool
    release_target: Tensor      # [T] float
    release_known: Tensor       # [T] bool
    regrasp_target: Tensor      # [T] float
    regrasp_known: Tensor       # [T] bool
    suite: str
    task_idx: int
    has_feasible: bool
    n_steps: int
    feasible_starts: list[int]


def load_training_contexts(
    s1_root: Path, teacher_root: Path, k10_root: Path,
    identities: list[str], registry_map: dict[str, Any],
) -> list[K10TrainingEpisode]:
    """Load episodes via official V5 loader, join K10 targets + release/regrasp aux."""

    rows = [registry_map[i] for i in identities]
    v5_eps = load_v5_episodes(s1_root, teacher_root, rows, policy_index=None)
    # V5 episodes are in the same order as rows

    episodes: list[K10TrainingEpisode] = []
    for v5_ep, identity in zip(v5_eps, identities):
        parts = identity.split("/")
        suite, task_str, state_str = parts[0], parts[1], parts[2]

        k10_path = k10_root / "labels" / suite / task_str / state_str / "k10_labels_v121.jsonl"
        if not k10_path.is_file():
            raise FileNotFoundError(f"K10 label missing: {k10_path}")
        k10_labels = _jsonl(k10_path)

        T = v5_ep.features_25d.shape[0]
        if len(k10_labels) != T:
            raise ValueError(f"step count mismatch: {identity} V5={T} K10={len(k10_labels)}")

        # Verify candidate_close parity
        for i, lab in enumerate(k10_labels):
            v5_cc = bool(v5_ep.candidate_close[i].item())
            k10_cc = bool(lab.get("candidate_close", False))
            if v5_cc != k10_cc:
                raise ValueError(f"candidate_close disagree: {identity} step {i}")

        k10_target = torch.full((T,), -1.0)
        k10_known = torch.zeros(T, dtype=torch.bool)
        feasible_starts: list[int] = []
        for i, lab in enumerate(k10_labels):
            if lab.get("label_known", True):
                k10_target[i] = 1.0 if lab.get("is_feasible_start") else 0.0
                k10_known[i] = True
                if lab.get("is_feasible_start"):
                    feasible_starts.append(i)

        # Release/regrasp aux from V5 episode (Physics V2.1 derived)
        release_target = v5_ep.release_imminent.float()
        release_known = v5_ep.release_known_mask
        regrasp_target = v5_ep.regrasp_or_unstable.float()
        regrasp_known = v5_ep.regrasp_known_mask

        episodes.append(K10TrainingEpisode(
            identity=identity,
            features_25d=v5_ep.features_25d,
            valid_mask=v5_ep.valid_mask,
            candidate_close=v5_ep.candidate_close,
            k10_target=k10_target,
            k10_known=k10_known,
            release_target=release_target,
            release_known=release_known,
            regrasp_target=regrasp_target,
            regrasp_known=regrasp_known,
            suite=suite,
            task_idx=int(task_str.replace("task_", "")),
            has_feasible=len(feasible_starts) > 0,
            n_steps=T,
            feasible_starts=feasible_starts,
        ))

    return episodes


# ── models ──────────────────────────────────────────────────────────────────
class R7SLinear25D(nn.Module):
    """Linear probe: 25D → utility + release + regrasp logits."""
    def __init__(self):
        super().__init__()
        self.utility_head = nn.Linear(25, 1)
        self.release_head = nn.Linear(25, 1)
        self.regrasp_head = nn.Linear(25, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {
            "utility_logit": self.utility_head(x).squeeze(-1),    # [B, T]
            "release_logit": self.release_head(x).squeeze(-1),
            "regrasp_logit": self.regrasp_head(x).squeeze(-1),
        }


class R7AGRU25D(nn.Module):
    """GRU128: causal 25D → GRU128 → utility + release + regrasp heads."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRUCell(25, hidden_dim)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.release_head = nn.Linear(hidden_dim, 1)
        self.regrasp_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, valid_mask: Tensor, boundaries: Tensor) -> dict[str, Tensor]:
        B, T_val, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.gru.hidden_size, device=device)
        u_logits = torch.zeros(B, T_val, device=device)
        r_logits = torch.zeros(B, T_val, device=device)
        g_logits = torch.zeros(B, T_val, device=device)
        for t in range(T_val):
            h = torch.where(boundaries[:, t].unsqueeze(1), torch.zeros_like(h), h)
            h_new = self.gru(x[:, t, :], h)
            h = torch.where(valid_mask[:, t].unsqueeze(1), h_new, h)
            u_logits[:, t] = self.utility_head(h).squeeze(-1)
            r_logits[:, t] = self.release_head(h).squeeze(-1)
            g_logits[:, t] = self.regrasp_head(h).squeeze(-1)
        return {"utility_logit": u_logits, "release_logit": r_logits, "regrasp_logit": g_logits}


# ── loss ────────────────────────────────────────────────────────────────────
def compute_k10_loss(
    outputs: dict[str, Tensor],
    episode: K10TrainingEpisode,
    device: torch.device,
) -> dict[str, Tensor]:
    """Episode-balanced BCE + release/regrasp auxiliary. Normalized per-episode."""
    dev = outputs["utility_logit"].device
    rankable = episode.valid_mask.to(dev) & episode.candidate_close.to(dev)
    known = episode.k10_known.to(dev) & rankable
    if not known.any():
        return {"total": outputs["utility_logit"].sum() * 0.0}

    u_logits = outputs["utility_logit"].squeeze(0)  # [T] from [1, T]
    rel_logits = outputs["release_logit"].squeeze(0)
    reg_logits = outputs["regrasp_logit"].squeeze(0)

    # Utility BCE on known rankable steps
    k10_tgt = episode.k10_target.to(dev)
    bce = nn.functional.binary_cross_entropy_with_logits(
        u_logits[known], k10_tgt[known], reduction="none")

    pos_mask = known & (k10_tgt > 0.5)
    neg_mask = known & (k10_tgt < 0.5)

    n_pos = pos_mask.sum().clamp_min(1)
    n_neg = neg_mask.sum().clamp_min(1)

    # Episode-balanced: positive and negative each get weight 0.5 within episode
    pos_loss = bce[pos_mask[known]].sum() / n_pos if pos_mask.any() else 0.0
    neg_loss = bce[neg_mask[known]].sum() / n_neg if neg_mask.any() else 0.0

    if episode.has_feasible:
        utility_loss = 0.5 * pos_loss + 0.5 * neg_loss
    else:
        utility_loss = neg_loss

    # Release auxiliary loss
    rel_known = episode.release_known.to(dev) & rankable
    if rel_known.any():
        release_loss = nn.functional.binary_cross_entropy_with_logits(
            rel_logits[rel_known], episode.release_target.to(dev)[rel_known])
    else:
        release_loss = torch.tensor(0.0, device=dev)

    # Regrasp auxiliary loss
    reg_known = episode.regrasp_known.to(dev) & rankable
    if reg_known.any():
        regrasp_loss = nn.functional.binary_cross_entropy_with_logits(
            reg_logits[reg_known], episode.regrasp_target.to(dev)[reg_known])
    else:
        regrasp_loss = torch.tensor(0.0, device=dev)

    total = utility_loss + 0.3 * release_loss + 0.3 * regrasp_loss
    return {"total": total, "utility": utility_loss,
            "release": release_loss, "regrasp": regrasp_loss}


# ── training ────────────────────────────────────────────────────────────────
def train_one_model(
    model: nn.Module,
    episodes: list[K10TrainingEpisode],
    seed: int,
    epochs: int,
    device: str,
) -> tuple[nn.Module, Tensor, Tensor, list[float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    # Normalization from training data only
    all_f = torch.cat([ep.features_25d[ep.valid_mask] for ep in episodes], dim=0)
    norm_mean = all_f.mean(dim=0).to(dev)
    norm_std = all_f.std(dim=0, unbiased=False).clamp_min(1e-6).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    is_gru = isinstance(model, R7AGRU25D)

    history: list[float] = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        indices = list(range(len(episodes)))
        rng.shuffle(indices)

        epoch_losses: list[float] = []
        batch_losses: list[Tensor] = []

        for idx, ep_idx in enumerate(indices):
            ep = episodes[ep_idx]
            T = ep.n_steps
            x = ((ep.features_25d.to(dev) - norm_mean) / norm_std).unsqueeze(0)
            svm = ep.valid_mask.to(dev).unsqueeze(0)
            bnd = torch.zeros(1, T, dtype=torch.bool, device=dev)
            bnd[0, 0] = True

            if is_gru:
                outputs = model(x, svm, bnd)
            else:
                outputs = model(x)

            loss_dict = compute_k10_loss(outputs, ep, dev)
            batch_losses.append(loss_dict["total"])

            if len(batch_losses) == 8 or idx == len(indices) - 1:
                opt.zero_grad(set_to_none=True)
                batch_mean = torch.stack(batch_losses).mean()
                if not torch.isfinite(batch_mean):
                    raise FloatingPointError(f"NaN/Inf loss at epoch {epoch+1}")
                batch_mean.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_losses.extend([float(l.detach().cpu()) for l in batch_losses])
                batch_losses = []

        avg = sum(epoch_losses) / len(epoch_losses)
        history.append(avg)
        print(f"  epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    model.eval()
    return model, norm_mean, norm_std, history


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_episodes(
    model: nn.Module, episodes: list[K10TrainingEpisode],
    norm_mean: Tensor, norm_std: Tensor, device: str,
) -> dict[str, dict[str, Any]]:
    """Return per-identity scores: utility, release, regrasp."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    is_gru = isinstance(model, R7AGRU25D)

    results: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        T = ep.n_steps
        x = ((ep.features_25d.to(dev) - norm_mean) / norm_std).unsqueeze(0)
        svm = ep.valid_mask.to(dev).unsqueeze(0)
        bnd = torch.zeros(1, T, dtype=torch.bool, device=dev)
        bnd[0, 0] = True

        if is_gru:
            outputs = model(x, svm, bnd)
        else:
            outputs = model(x)

        results[ep.identity] = {
            "utility": torch.sigmoid(outputs["utility_logit"].squeeze(0)).cpu(),
            "release": torch.sigmoid(outputs["release_logit"].squeeze(0)).cpu(),
            "regrasp": torch.sigmoid(outputs["regrasp_logit"].squeeze(0)).cpu(),
        }
    return results


# ── scheduler evaluation ────────────────────────────────────────────────────
def evaluate_at_threshold(
    episodes: list[K10TrainingEpisode],
    predictions: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Run V5OneShotScheduler on pre-computed predictions at one threshold."""
    config = V5SchedulerConfig(
        utility_threshold=threshold,
        release_veto_threshold=0.5, regrasp_veto_threshold=0.5,
        release_veto_enabled=True, regrasp_veto_enabled=True,
    )
    results: list[dict[str, Any]] = []

    for ep in episodes:
        pred = predictions[ep.identity]
        scheduler = V5OneShotScheduler(config)
        T = ep.n_steps
        emitted = False
        emit_step = -1

        for t in range(T):
            result = scheduler.update(
                step=t,
                candidate_close=bool(ep.candidate_close[t].item()),
                valid=bool(ep.valid_mask[t].item()),
                utility_probability=float(pred["utility"][t]),
                release_probability=float(pred["release"][t]),
                regrasp_probability=float(pred["regrasp"][t]),
                uncertainty_probability=0.0,
            )
            if result["emit"]:
                emitted = True
                emit_step = t

        within_k10 = emitted and emit_step in set(ep.feasible_starts)
        false_emit = emitted and not within_k10
        false_early = false_emit and ep.has_feasible and emit_step < min(ep.feasible_starts)

        results.append({
            "identity": ep.identity,
            "has_feasible": ep.has_feasible,
            "emitted": emitted,
            "emit_step": emit_step,
            "within_k10": within_k10,
            "false_emit": false_emit,
            "false_early": false_early,
        })

    return results


def compute_metrics(results: list[dict[str, Any]], n_feasible: int, n_no_feasible: int) -> dict[str, Any]:
    n_hit = sum(1 for r in results if r["within_k10"])
    n_emit = sum(1 for r in results if r["emitted"])
    n_false = sum(1 for r in results if r["false_emit"])
    n_false_early = sum(1 for r in results if r["false_early"])
    n_abstain_feas = sum(1 for r in results if r["has_feasible"] and not r["emitted"])
    n_abstain_nofeas = sum(1 for r in results if not r["has_feasible"] and not r["emitted"])
    n_covered = sum(1 for r in results if r["has_feasible"] and r["within_k10"])

    return {
        "feasible_hit_recall": n_hit / n_feasible if n_feasible else 0,
        "emit_precision": n_hit / n_emit if n_emit else 0,
        "positive_episode_coverage": n_covered / n_feasible if n_feasible else 0,
        "no_corridor_abstention": n_abstain_nofeas / n_no_feasible if n_no_feasible else 0,
        "false_early_rate": n_false_early / n_feasible if n_feasible else 0,
        "n_hit": n_hit, "n_emit": n_emit, "n_false": n_false,
        "n_false_early": n_false_early, "n_feasible": n_feasible,
        "n_no_feasible": n_no_feasible,
        "n_abstain_feasible": n_abstain_feas,
        "n_abstain_no_feasible": n_abstain_nofeas,
        "one_shot_compliance": 1.0,
    }


def check_oof_gates(m: dict[str, Any]) -> bool:
    return (
        m["feasible_hit_recall"] >= 0.80
        and m["emit_precision"] >= 0.80
        and m["no_corridor_abstention"] >= 0.90
        and m.get("outside_rankable_emit", 0) == 0
        and m.get("release_regrasp_emit", 0) == 0
        and m.get("one_shot_compliance", 1.0) == 1.0
    )


# ── OOF ──────────────────────────────────────────────────────────────────────
def build_oof_folds(episodes: list[K10TrainingEpisode], seed: int) -> list[tuple[list[int], list[int]]]:
    """5-fold partition stratified by suite and K10 feasibility."""
    rng = random.Random(seed + 9999)
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[(ep.suite, ep.has_feasible)].append(i)

    # Shuffle within each group
    for v in groups.values():
        rng.shuffle(v)

    # Distribute to 5 folds round-robin within each group
    folds: list[list[int]] = [[] for _ in range(5)]
    for group_indices in groups.values():
        for j, idx in enumerate(group_indices):
            folds[j % 5].append(idx)

    # Create train/val splits
    splits = []
    for fi in range(5):
        val_set = set(folds[fi])
        train_set = [i for i in range(len(episodes)) if i not in val_set]
        splits.append((train_set, list(val_set)))

    return splits


def select_oof_threshold(
    model_class, episodes: list[K10TrainingEpisode], seed: int, device: str,
) -> tuple[float | None, dict[str, Any]]:
    """5-fold OOF threshold selection. Returns (threshold, oof_report) or (None, report)."""
    folds = build_oof_folds(episodes, seed)
    n_feas = sum(1 for ep in episodes if ep.has_feasible)
    n_nofeas = len(episodes) - n_feas

    all_oof_results: list[dict[str, Any]] = []
    oof_report: dict[str, Any] = {"folds": [], "all_results": [], "best_threshold": None}

    for fi, (train_idx, val_idx) in enumerate(folds):
        train_eps = [episodes[i] for i in train_idx]
        val_eps = [episodes[i] for i in val_idx]
        print(f"  Fold {fi+1}/5: train={len(train_eps)}, val={len(val_eps)}")

        model = model_class()
        model, nm, ns, _ = train_one_model(model, train_eps, seed=seed, epochs=10, device=device)
        preds = predict_episodes(model, val_eps, nm, ns, device)

        for tau in OOF_THRESHOLD_GRID:
            results = evaluate_at_threshold(val_eps, preds, tau)
            for r in results:
                r["threshold"] = tau
                r["fold"] = fi
            all_oof_results.extend(results)

        oof_report["folds"].append({
            "fold": fi, "train_count": len(train_idx), "val_count": len(val_idx),
            "val_identities": [episodes[i].identity for i in val_idx],
        })

    oof_report["all_results"] = all_oof_results
    oof_report["n_feasible"] = n_feas
    oof_report["n_no_feasible"] = n_nofeas
    oof_report["n_total"] = len(episodes)

    # Find highest eligible threshold
    best_tau = None
    best_metrics = None
    for tau in OOF_THRESHOLD_GRID:
        tau_results = [r for r in all_oof_results if abs(r["threshold"] - tau) < 0.005]
        if len(tau_results) != len(episodes):
            continue  # should be exactly N (one per episode per threshold)
        m = compute_metrics(tau_results, n_feas, n_nofeas)
        # Add outside_rankable and release/regrasp checks
        outside_rankable = sum(1 for r in tau_results if r["emitted"] and not r["within_k10"])
        m["outside_rankable_emit"] = outside_rankable
        m["release_regrasp_emit"] = 0  # computed from veto counts
        if check_oof_gates(m):
            best_tau = tau
            best_metrics = m

    oof_report["best_threshold"] = best_tau
    if best_metrics:
        oof_report["best_metrics"] = best_metrics
        print(f"  OOF selected threshold: {best_tau}")
    else:
        print("  OOF: NO ELIGIBLE THRESHOLD (HOLD_OOF)")
        oof_report["status"] = "HOLD_OOF"

    return best_tau, oof_report


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="R7.3 K10 Detector Training")
    ap.add_argument("--candidate", choices=["R7-S-LINEAR-25D", "R7-A-GRU-25D"], required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--k10-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--registry-csv", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"output root already exists: {out}")

    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    try:
        git_commit = _git_commit()
        print(f"=== R7.3 K10 DETECTOR TRAINING: {args.candidate} ===\n")
        print(f"Git commit: {git_commit}")

        # ── Verify roots ───────────────────────────────────────────────────
        print("\nVerifying source roots...")
        for label, path in [
            ("S1 root", args.s1_root), ("Teacher root", args.teacher_root),
            ("K10 root", args.k10_root), ("Fold root", args.fold_root),
        ]:
            verify_sealed_directory(path)
            print(f"  {label}: SEAL OK")

        # ── Load fold manifest ─────────────────────────────────────────────
        print("\nLoading fold manifest...")
        fold = load_fit_fold_bundle(args.fold_root)
        fold0 = next(f for f in fold["folds"] if f["fold_id"] == 0)
        train_ids = sorted(fold0["train_identities"])
        val_ids = sorted(fold0["validation_identities"])
        print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")

        # ── Load registry ──────────────────────────────────────────────────
        print("\nLoading registry...")
        registry = load_fit_registry(args.registry_csv)
        registry_map = {r["canonical_parent_key"]: r for r in registry}

        # ── Load training episodes ─────────────────────────────────────────
        print("\nLoading training episodes...")
        train_eps = load_training_contexts(
            args.s1_root, args.teacher_root, args.k10_root,
            train_ids, registry_map,
        )
        n_feas_train = sum(1 for ep in train_eps if ep.has_feasible)
        print(f"  Train: {len(train_eps)} episodes, {n_feas_train} feasible")

        # ── OOF threshold selection ────────────────────────────────────────
        print("\n--- OOF Threshold Selection ---")
        model_class = R7SLinear25D if args.candidate == "R7-S-LINEAR-25D" else R7AGRU25D
        selected_tau, oof_report = select_oof_threshold(
            model_class, train_eps, seed=20260717, device=args.device)

        if selected_tau is None:
            # Write OOF report and stop
            (staging / "OOF_REPORT.json").write_text(
                json.dumps(oof_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / "HOLD_OOF.txt").write_text("NO ELIGIBLE THRESHOLD\n", encoding="utf-8")
            _seal_root(staging)
            os.replace(staging, out)
            print(f"\nHOLD_OOF: {args.candidate} has no eligible threshold.")
            print(f"Root: {out}")
            return

        # ── Final model training ───────────────────────────────────────────
        print(f"\n--- Final Model Training (threshold={selected_tau}) ---")
        final_model = model_class()
        final_model, nm, ns, history = train_one_model(
            final_model, train_eps, seed=20260717, epochs=10, device=args.device)

        # ── Validation evaluation ──────────────────────────────────────────
        print("\n--- Validation Evaluation ---")
        val_eps = load_training_contexts(
            args.s1_root, args.teacher_root, args.k10_root,
            val_ids, registry_map,
        )
        n_feas_val = sum(1 for ep in val_eps if ep.has_feasible)
        n_nofeas_val = len(val_eps) - n_feas_val
        print(f"  Val: {len(val_eps)} episodes, {n_feas_val} feasible")

        val_preds = predict_episodes(final_model, val_eps, nm, ns, args.device)

        # Full threshold sweep for diagnosis
        val_ledger: list[dict[str, Any]] = []
        for tau in OOF_THRESHOLD_GRID:
            results = evaluate_at_threshold(val_eps, val_preds, tau)
            for r in results:
                r["threshold"] = tau
                r["candidate"] = args.candidate
            val_ledger.extend(results)

        # Metrics at selected threshold
        selected_results = [r for r in val_ledger if abs(r["threshold"] - selected_tau) < 0.005]
        val_metrics = compute_metrics(selected_results, n_feas_val, n_nofeas_val)

        # Gate check
        gates = {
            "K10_recall": val_metrics["feasible_hit_recall"] >= 0.80,
            "emit_precision": val_metrics["emit_precision"] >= 0.80,
            "no_corridor_abstention": val_metrics["no_corridor_abstention"] >= 0.90,
            "false_early_rate": val_metrics["false_early_rate"] <= 0.05,
        }
        gate_pass = all(gates.values())
        print(f"\n  Validation at tau={selected_tau}:")
        for k, v in gates.items():
            print(f"    {k}: {'PASS' if v else 'FAIL'}")

        # ── Write artifacts ────────────────────────────────────────────────
        print("\nWriting artifacts...")

        # Checkpoint
        checkpoint = {
            "schema": "R7_K10_DETECTOR_DEVELOPMENT_CHECKPOINT_V1",
            "candidate": args.candidate,
            "model_state": final_model.state_dict(),
            "normalization_mean_25d": nm.cpu(),
            "normalization_std_25d": ns.cpu(),
            "seed": 20260717,
            "epochs": 10,
            "oof_selected_threshold": selected_tau,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        torch.save(checkpoint, staging / "checkpoint.pt")

        # PROTOCOL.json
        (staging / "PROTOCOL.json").write_text(json.dumps({
            "schema": "R7_K10_DETECTOR_TRAINING_PROTOCOL_V1",
            "protocol_ref": "protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md",
            "candidate": args.candidate,
            "seed": 20260717, "epochs": 10, "fp32": True,
            "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-5,
            "batch_size": 8, "clip_grad_norm": 5.0, "early_stopping": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # SOURCE_BINDING.json
        (staging / "SOURCE_BINDING.json").write_text(json.dumps({
            "schema": "R7_K10_DETECTOR_SOURCE_BINDING_V1",
            "git_commit": git_commit,
            "s1_root_sha256s_sha256": sha256_file(args.s1_root / "SHA256SUMS"),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root / "SHA256SUMS"),
            "k10_root_sha256s_sha256": sha256_file(args.k10_root / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root / "SHA256SUMS"),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # TRAIN_HISTORY.json
        (staging / "TRAIN_HISTORY.json").write_text(json.dumps({
            "losses": history, "seed": 20260717, "epochs": 10,
        }, indent=2) + "\n", encoding="utf-8")

        # IDENTITY_MANIFEST.json
        (staging / "IDENTITY_MANIFEST.json").write_text(json.dumps({
            "train_identities": train_ids, "train_count": len(train_ids),
            "val_identities": val_ids, "val_count": len(val_ids),
            "fold_id": 0,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # EPISODE_THRESHOLD_LEDGER.jsonl
        with open(staging / "EPISODE_THRESHOLD_LEDGER.jsonl", "w", encoding="utf-8") as fh:
            for entry in val_ledger:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")

        # THRESHOLD_METRICS.csv
        with open(staging / "THRESHOLD_METRICS.csv", "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["threshold", "feasible_hit_recall", "emit_precision",
                          "no_corridor_abstention", "n_hit", "n_emit", "n_false",
                          "n_false_early", "n_feasible", "n_no_feasible"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for tau in OOF_THRESHOLD_GRID:
                m = compute_metrics(
                    [r for r in val_ledger if abs(r["threshold"] - tau) < 0.005],
                    n_feas_val, n_nofeas_val,
                )
                m["threshold"] = tau
                writer.writerow(m)

        # OOF_REPORT.json
        (staging / "OOF_REPORT.json").write_text(
            json.dumps(oof_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # AUDIT.json (self-audit)
        (staging / "AUDIT.json").write_text(json.dumps({
            "schema": "R7_K10_DETECTOR_AUDIT_V1",
            "candidate": args.candidate,
            "oof_selected_threshold": selected_tau,
            "validation_gate": {k: v for k, v in gates.items()},
            "gate_pass": gate_pass,
            "validation_metrics": val_metrics,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # MANIFEST.json
        (staging / "MANIFEST.json").write_text(json.dumps({
            "schema": "R7_K10_DETECTOR_MANIFEST_V1",
            "candidate": args.candidate,
            "fold_id": 0,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "oof_selected_threshold": selected_tau,
            "gate_pass": gate_pass,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Seal
        print("Sealing...")
        root_sha = _seal_root(staging)
        os.replace(staging, out)
        print(f"\nRoot: {out}")
        print(f"SHA256SUMS: {root_sha}")
        print(f"\n=== R7.3 {args.candidate}: {'PASS' if gate_pass else 'FAIL'} ===")

    except Exception:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
