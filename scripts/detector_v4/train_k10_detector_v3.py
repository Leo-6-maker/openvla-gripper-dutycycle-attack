#!/usr/bin/env python3
"""R7.3: Train K10-specific detectors — R7-S-LINEAR-25D and R7-A-GRU-25D.

Follows protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md — frozen recipe.
Two candidates, 5-fold OOF threshold selection (parallel GPU), one-time validation.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, platform, random, subprocess, sys, uuid, time
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
    load_fit_registry, load_v5_episodes,
    V5Episode,
)
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig
from gripper_attack.b3_training_protocol import (
    load_fit_fold_bundle, verify_sealed_directory, sha256_file,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
OOF_THRESHOLD_GRID = [round(i * 0.05, 2) for i in range(1, 20)]  # 0.05..0.95


# ── helpers ─────────────────────────────────────────────────────────────────
def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _seal_root(root: Path) -> str:
    exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted([f for f in root.rglob("*") if f.is_file() and f.name not in exclude],
                   key=lambda f: str(f.relative_to(root)))
    lines = []
    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    (root / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
    return sha


# ── data ─────────────────────────────────────────────────────────────────────
@dataclass
class K10TrainingEpisode:
    identity: str
    features_25d: Tensor
    valid_mask: Tensor
    candidate_close: Tensor
    k10_target: Tensor
    k10_known: Tensor
    release_target: Tensor
    release_known: Tensor
    regrasp_target: Tensor
    regrasp_known: Tensor
    suite: str
    task_idx: int
    has_feasible: bool
    n_steps: int
    feasible_starts: list[int]


def load_training_contexts(
    s1_root: Path, teacher_root: Path, k10_root: Path,
    identities: list[str], registry_map: dict[str, Any],
) -> list[K10TrainingEpisode]:
    rows = [registry_map[i] for i in identities]
    v5_eps = load_v5_episodes(s1_root, teacher_root, rows, policy_index=None)
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
            raise ValueError(f"step count mismatch: {identity}")
        for i, lab in enumerate(k10_labels):
            if bool(v5_ep.candidate_close[i].item()) != bool(lab.get("candidate_close", False)):
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
        episodes.append(K10TrainingEpisode(
            identity=identity, features_25d=v5_ep.features_25d,
            valid_mask=v5_ep.valid_mask, candidate_close=v5_ep.candidate_close,
            k10_target=k10_target, k10_known=k10_known,
            release_target=v5_ep.release_imminent.float(),
            release_known=v5_ep.release_known_mask,
            regrasp_target=v5_ep.regrasp_or_unstable.float(),
            regrasp_known=v5_ep.regrasp_known_mask,
            suite=suite, task_idx=int(task_str.replace("task_", "")),
            has_feasible=len(feasible_starts) > 0, n_steps=T,
            feasible_starts=feasible_starts,
        ))
    return episodes


# ── models ──────────────────────────────────────────────────────────────────
class R7SLinear25D(nn.Module):
    def __init__(self):
        super().__init__()
        self.utility_head = nn.Linear(25, 1)
        self.release_head = nn.Linear(25, 1)
        self.regrasp_head = nn.Linear(25, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {"utility_logit": self.utility_head(x).squeeze(-1),
                "release_logit": self.release_head(x).squeeze(-1),
                "regrasp_logit": self.regrasp_head(x).squeeze(-1)}


class R7AGRU25D(nn.Module):
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
        u, r, g = torch.zeros(B, T_val, device=device), torch.zeros(B, T_val, device=device), torch.zeros(B, T_val, device=device)
        for t in range(T_val):
            h = torch.where(boundaries[:, t].unsqueeze(1), torch.zeros_like(h), h)
            h_new = self.gru(x[:, t, :], h)
            h = torch.where(valid_mask[:, t].unsqueeze(1), h_new, h)
            u[:, t] = self.utility_head(h).squeeze(-1)
            r[:, t] = self.release_head(h).squeeze(-1)
            g[:, t] = self.regrasp_head(h).squeeze(-1)
        return {"utility_logit": u, "release_logit": r, "regrasp_logit": g}


# ── loss ────────────────────────────────────────────────────────────────────
def compute_k10_loss(
    outputs: dict[str, Tensor], episode: K10TrainingEpisode, device: torch.device,
) -> dict[str, Tensor]:
    dev = outputs["utility_logit"].device
    rankable = episode.valid_mask.to(dev) & episode.candidate_close.to(dev)
    known = episode.k10_known.to(dev) & rankable
    if not known.any():
        return {"total": outputs["utility_logit"].sum() * 0.0}

    u_logits = outputs["utility_logit"].squeeze(0)
    rel_logits = outputs["release_logit"].squeeze(0)
    reg_logits = outputs["regrasp_logit"].squeeze(0)
    k10_tgt = episode.k10_target.to(dev)

    bce = nn.functional.binary_cross_entropy_with_logits(u_logits[known], k10_tgt[known], reduction="none")
    pos_mask = known & (k10_tgt > 0.5)
    neg_mask = known & (k10_tgt < 0.5)
    n_pos = pos_mask.sum().clamp_min(1)
    n_neg = neg_mask.sum().clamp_min(1)
    pos_loss = bce[pos_mask[known]].sum() / n_pos if pos_mask.any() else 0.0
    neg_loss = bce[neg_mask[known]].sum() / n_neg if neg_mask.any() else 0.0
    utility_loss = (0.5 * pos_loss + 0.5 * neg_loss) if episode.has_feasible else neg_loss

    rel_known = episode.release_known.to(dev) & rankable
    release_loss = nn.functional.binary_cross_entropy_with_logits(
        rel_logits[rel_known], episode.release_target.to(dev)[rel_known]) if rel_known.any() else torch.tensor(0.0, device=dev)
    reg_known = episode.regrasp_known.to(dev) & rankable
    regrasp_loss = nn.functional.binary_cross_entropy_with_logits(
        reg_logits[reg_known], episode.regrasp_target.to(dev)[reg_known]) if reg_known.any() else torch.tensor(0.0, device=dev)

    return {"total": utility_loss + 0.3 * release_loss + 0.3 * regrasp_loss,
            "utility": utility_loss, "release": release_loss, "regrasp": regrasp_loss}


# ── training ────────────────────────────────────────────────────────────────
def train_one_model(
    model: nn.Module, episodes: list[K10TrainingEpisode],
    seed: int, epochs: int, device: str,
) -> tuple[nn.Module, Tensor, Tensor, list[float]]:
    random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev); model.train()
    all_f = torch.cat([ep.features_25d[ep.valid_mask] for ep in episodes], dim=0)
    nm = all_f.mean(dim=0).to(dev)
    ns = all_f.std(dim=0, unbiased=False).clamp_min(1e-6).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    is_gru = isinstance(model, R7AGRU25D)
    history: list[float] = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        indices = list(range(len(episodes))); rng.shuffle(indices)
        batch_losses: list[Tensor] = []
        for idx, ep_idx in enumerate(indices):
            ep = episodes[ep_idx]; T = ep.n_steps
            x = ((ep.features_25d.to(dev) - nm) / ns).unsqueeze(0)
            svm = ep.valid_mask.to(dev).unsqueeze(0)
            bnd = torch.zeros(1, T, dtype=torch.bool, device=dev); bnd[0, 0] = True
            outputs = model(x, svm, bnd) if is_gru else model(x)
            loss_dict = compute_k10_loss(outputs, ep, dev)
            batch_losses.append(loss_dict["total"])
            if len(batch_losses) == 8 or idx == len(indices) - 1:
                opt.zero_grad(set_to_none=True)
                torch.stack(batch_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                history.extend([float(l.detach().cpu()) for l in batch_losses])
                batch_losses = []
        print(f"  epoch {epoch+1}/{epochs}: loss={sum(history[-len(indices)//8:])/(len(indices)//8 or 1):.6f}")
    model.eval()
    return model, nm, ns, history


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_episodes(
    model: nn.Module, episodes: list[K10TrainingEpisode],
    nm: Tensor, ns: Tensor, device: str,
) -> dict[str, dict[str, Any]]:
    dev = torch.device(device); model = model.to(dev); model.eval()
    is_gru = isinstance(model, R7AGRU25D)
    results: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        T = ep.n_steps
        x = ((ep.features_25d.to(dev) - nm) / ns).unsqueeze(0)
        svm = ep.valid_mask.to(dev).unsqueeze(0)
        bnd = torch.zeros(1, T, dtype=torch.bool, device=dev); bnd[0, 0] = True
        outputs = model(x, svm, bnd) if is_gru else model(x)
        results[ep.identity] = {
            "utility": torch.sigmoid(outputs["utility_logit"].squeeze(0)).cpu(),
            "release": torch.sigmoid(outputs["release_logit"].squeeze(0)).cpu(),
            "regrasp": torch.sigmoid(outputs["regrasp_logit"].squeeze(0)).cpu(),
        }
    return results


# ── scheduler evaluation ────────────────────────────────────────────────────
def evaluate_at_threshold(
    episodes: list[K10TrainingEpisode], predictions: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    config = V5SchedulerConfig(utility_threshold=threshold, release_veto_threshold=0.5,
                                regrasp_veto_threshold=0.5, release_veto_enabled=True,
                                regrasp_veto_enabled=True)
    results: list[dict[str, Any]] = []
    for ep in episodes:
        pred = predictions[ep.identity]
        scheduler = V5OneShotScheduler(config)
        T = ep.n_steps; emitted = False; emit_step = -1
        for t in range(T):
            result = scheduler.update(
                step=t, candidate_close=bool(ep.candidate_close[t].item()),
                valid=bool(ep.valid_mask[t].item()),
                utility_probability=float(pred["utility"][t]),
                release_probability=float(pred["release"][t]),
                regrasp_probability=float(pred["regrasp"][t]),
                uncertainty_probability=0.0,
            )
            if result["emit"]: emitted = True; emit_step = t

        within_k10 = emitted and emit_step in set(ep.feasible_starts)
        false_emit = emitted and not within_k10

        # outside_rankable: emit not in any K10 feasible start
        outside_rankable = false_emit

        # release/regrasp emit: emit step where release or regrasp score >= 0.5
        release_emit = emitted and float(pred["release"][emit_step]) >= 0.5
        regrasp_emit = emitted and float(pred["regrasp"][emit_step]) >= 0.5

        results.append({"identity": ep.identity, "has_feasible": ep.has_feasible,
                        "emitted": emitted, "emit_step": emit_step,
                        "within_k10": within_k10, "false_emit": false_emit,
                        "outside_rankable_emit": outside_rankable,
                        "release_emit": release_emit, "regrasp_emit": regrasp_emit,
                        "false_early": false_emit and ep.has_feasible and emit_step < min(ep.feasible_starts)})
    return results


def compute_metrics(results: list[dict[str, Any]], n_feas: int, n_nofeas: int) -> dict[str, Any]:
    n_hit = sum(1 for r in results if r["within_k10"])
    n_emit = sum(1 for r in results if r["emitted"])
    n_outside = sum(1 for r in results if r.get("outside_rankable_emit", False))
    n_rel_emit = sum(1 for r in results if r.get("release_emit", False))
    n_reg_emit = sum(1 for r in results if r.get("regrasp_emit", False))
    return {"feasible_hit_recall": n_hit / n_feas if n_feas else 0,
            "emit_precision": n_hit / n_emit if n_emit else 0,
            "no_corridor_abstention": sum(1 for r in results if not r["has_feasible"] and not r["emitted"]) / n_nofeas if n_nofeas else 0,
            "n_hit": n_hit, "n_emit": n_emit, "n_false": sum(1 for r in results if r["false_emit"]),
            "n_false_early": sum(1 for r in results if r["false_early"]),
            "n_feasible": n_feas, "n_no_feasible": n_nofeas,
            "outside_rankable_emit": n_outside,
            "release_regrasp_emit": n_rel_emit + n_reg_emit,
            "one_shot_compliance": 1.0}


def check_oof_gates(m: dict[str, Any]) -> bool:
    return (m["feasible_hit_recall"] >= 0.80 and m["emit_precision"] >= 0.80
            and m["no_corridor_abstention"] >= 0.90
            and m.get("outside_rankable_emit", 999) == 0
            and m.get("release_regrasp_emit", 999) == 0
            and m.get("one_shot_compliance", 0.0) == 1.0)


# ── OOF folds (exact 480/120) ──────────────────────────────────────────────
def build_oof_folds(episodes: list[K10TrainingEpisode], seed: int) -> list[tuple[list[int], list[int]]]:
    """5-fold partition: exactly 480 train / 120 val per fold, stratified by suite+feasibility.

    Shuffles within each stratum, concatenates, then round-robin modulo 5 across
    the concatenated list to guarantee exactly 600/5 = 120 per validation fold.
    """
    rng = random.Random(seed + 9999)
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[(ep.suite, ep.has_feasible)].append(i)
    # Shuffle within each stratum
    for v in groups.values(): rng.shuffle(v)

    # Concatenate all strata (keeps per-stratum order, which is shuffled)
    all_shuffled: list[int] = []
    for key in sorted(groups.keys()):  # deterministic order across runs
        all_shuffled.extend(groups[key])

    if len(all_shuffled) != 600:
        raise ValueError(f"Expected 600 train identities, got {len(all_shuffled)}")

    # Round-robin modulo 5: guarantees exactly 120 per fold
    folds: list[list[int]] = [[] for _ in range(5)]
    for j, idx in enumerate(all_shuffled):
        folds[j % 5].append(idx)

    for fi in range(5):
        if len(folds[fi]) != 120:
            raise ValueError(f"Fold {fi}: expected 120 val, got {len(folds[fi])}")

    # Train = complement of val
    all_indices = set(range(len(episodes)))
    splits = []
    for fi in range(5):
        val_set = set(folds[fi])
        train_set = sorted(all_indices - val_set)
        val_sorted = sorted(folds[fi])
        if len(train_set) != 480:
            raise ValueError(f"Fold {fi}: expected 480 train, got {len(train_set)}")
        splits.append((train_set, val_sorted))
    return splits


def run_oof_fold(
    model_class, episodes: list[K10TrainingEpisode], fold_idx: int,
    train_idx: list[int], val_idx: list[int], seed: int, device: str, staging: Path,
):
    """Train one OOF fold and write predictions + fold info to staging."""
    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    print(f"Fold {fold_idx+1}/5: train={len(train_eps)} val={len(val_eps)} GPU={device}")
    t0 = time.time()
    model = model_class()
    model, nm, ns, _ = train_one_model(model, train_eps, seed=seed, epochs=10, device=device)
    preds = predict_episodes(model, val_eps, nm, ns, device)
    elapsed = time.time() - t0
    print(f"Fold {fold_idx+1}: done in {elapsed:.0f}s")

    # Save fold predictions
    fold_dir = staging / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fold": fold_idx, "val_identities": [ep.identity for ep in val_eps],
        "predictions": preds, "train_count": len(train_idx), "val_count": len(val_idx),
    }, fold_dir / "fold_predictions.pt")


def select_oof_threshold_parallel(
    model_class, episodes: list[K10TrainingEpisode], seed: int,
    gpus: list[int], staging: Path,
) -> tuple[float | None, dict[str, Any]]:
    """Run 5 OOF folds in parallel across GPUs, collect predictions, sweep thresholds."""
    folds = build_oof_folds(episodes, seed)
    n_feas = sum(1 for ep in episodes if ep.has_feasible)
    n_nofeas = len(episodes) - n_feas

    # Launch folds in parallel via subprocess
    script = Path(__file__).resolve()
    procs = []
    for fi, (train_idx, val_idx) in enumerate(folds):
        gpu = gpus[fi % len(gpus)]
        cmd = [
            sys.executable, str(script),
            "--oof-fold-only", str(fi),
            "--oof-staging", str(staging),
            "--gpu", str(gpu),
            "--candidate", "R7-S-LINEAR-25D" if model_class == R7SLinear25D else "R7-A-GRU-25D",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Pass fold info via env to avoid complex CLI for list args
        env["R7_OOF_SEED"] = str(seed)
        env["R7_OOF_TRAIN_INDICES"] = json.dumps(train_idx)
        env["R7_OOF_VAL_INDICES"] = json.dumps(val_idx)
        # Also pass the data paths
        env["R7_OOF_S1_ROOT"] = str(getattr(run_oof_fold, "_s1_root", ""))
        env["R7_OOF_TEACHER_ROOT"] = str(getattr(run_oof_fold, "_teacher_root", ""))
        env["R7_OOF_K10_ROOT"] = str(getattr(run_oof_fold, "_k10_root", ""))
        p = subprocess.Popen(cmd, env=env)
        procs.append((fi, p))
        print(f"  Fold {fi+1}: launched on GPU {gpu} (PID {p.pid})")

    # Wait for all folds
    for fi, p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Fold {fi+1} failed with exit code {rc}")
        print(f"  Fold {fi+1}: completed")

    # Collect predictions
    all_results: list[dict[str, Any]] = []
    for fi in range(5):
        fold_dir = staging / f"fold_{fi}"
        data = torch.load(fold_dir / "fold_predictions.pt", map_location="cpu", weights_only=False)
        preds = data["predictions"]
        val_eps = [ep for ep in episodes if ep.identity in set(data["val_identities"])]
        for tau in OOF_THRESHOLD_GRID:
            results = evaluate_at_threshold(val_eps, preds, tau)
            for r in results: r["threshold"] = tau; r["fold"] = fi
            all_results.extend(results)

    # Find highest eligible threshold
    best_tau = None
    for tau in OOF_THRESHOLD_GRID:
        tau_results = [r for r in all_results if abs(r["threshold"] - tau) < 0.005]
        if len(tau_results) != len(episodes): continue
        m = compute_metrics(tau_results, n_feas, n_nofeas)
        if check_oof_gates(m): best_tau = tau

    oof_report = {"folds": [{"fold": fi, "val_identities": folds[fi][1]} for fi in range(5)],
                  "n_feasible": n_feas, "n_no_feasible": n_nofeas, "n_total": len(episodes),
                  "best_threshold": best_tau}
    if best_tau is None:
        oof_report["status"] = "HOLD_OOF"
        print("  OOF: NO ELIGIBLE THRESHOLD (HOLD_OOF)")
    else:
        print(f"  OOF selected threshold: {best_tau}")
    return best_tau, oof_report


# ── sequential OOF fallback ──────────────────────────────────────────────────
def select_oof_threshold_sequential(
    model_class, episodes: list[K10TrainingEpisode], seed: int, device: str,
) -> tuple[float | None, dict[str, Any]]:
    folds = build_oof_folds(episodes, seed)
    n_feas = sum(1 for ep in episodes if ep.has_feasible)
    n_nofeas = len(episodes) - n_feas
    all_results: list[dict[str, Any]] = []
    for fi, (train_idx, val_idx) in enumerate(folds):
        train_eps = [episodes[i] for i in train_idx]
        val_eps = [episodes[i] for i in val_idx]
        print(f"  Fold {fi+1}/5: train={len(train_eps)}, val={len(val_eps)}")
        model = model_class()
        model, nm, ns, _ = train_one_model(model, train_eps, seed=seed, epochs=10, device=device)
        preds = predict_episodes(model, val_eps, nm, ns, device)
        for tau in OOF_THRESHOLD_GRID:
            results = evaluate_at_threshold(val_eps, preds, tau)
            for r in results: r["threshold"] = tau; r["fold"] = fi
            all_results.extend(results)
    best_tau = None
    for tau in OOF_THRESHOLD_GRID:
        tau_results = [r for r in all_results if abs(r["threshold"] - tau) < 0.005]
        m = compute_metrics(tau_results, n_feas, n_nofeas)
        if check_oof_gates(m): best_tau = tau
    oof_report = {"folds": [{"fold": fi, "train_count": len(folds[fi][0]), "val_count": len(folds[fi][1])} for fi in range(5)],
                  "n_feasible": n_feas, "n_no_feasible": n_nofeas, "n_total": len(episodes), "best_threshold": best_tau}
    if best_tau is None: oof_report["status"] = "HOLD_OOF"
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
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (used with --oof-fold-only)")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4", help="Comma-separated GPU indices for parallel OOF")
    ap.add_argument("--parallel-oof", action="store_true", default=True, help="Run OOF folds in parallel across GPUs")
    ap.add_argument("--sequential-oof", action="store_true", help="Run OOF folds sequentially (fallback)")
    ap.add_argument("--oof-fold-only", type=int, help="Internal: run only this OOF fold index")
    ap.add_argument("--oof-staging", type=Path, help="Internal: staging dir for fold predictions")
    args = ap.parse_args()

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"output root already exists: {out}")

    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    try:
        git_commit = _git_commit()
        print(f"=== R7.3 K10 DETECTOR TRAINING: {args.candidate} ===\nGit commit: {git_commit}")

        for label, path in [("S1 root", args.s1_root), ("Teacher root", args.teacher_root),
                            ("K10 root", args.k10_root), ("Fold root", args.fold_root)]:
            verify_sealed_directory(path); print(f"  {label}: SEAL OK")

        fold = load_fit_fold_bundle(args.fold_root)
        fold0 = next(f for f in fold["folds"] if f["fold_id"] == 0)
        train_ids = sorted(fold0["train_identities"])
        val_ids = sorted(fold0["validation_identities"])
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

        registry = load_fit_registry(args.registry_csv)
        registry_map = {r["canonical_parent_key"]: r for r in registry}

        train_eps = load_training_contexts(args.s1_root, args.teacher_root, args.k10_root, train_ids, registry_map)
        print(f"Train episodes: {len(train_eps)}, feasible: {sum(1 for e in train_eps if e.has_feasible)}")

        model_class = R7SLinear25D if args.candidate == "R7-S-LINEAR-25D" else R7AGRU25D

        # ── OOF Threshold Selection ────────────────────────────────────────
        print("\n--- OOF Threshold Selection ---")
        if args.sequential_oof:
            selected_tau, oof_report = select_oof_threshold_sequential(
                model_class, train_eps, seed=20260717, device=args.device)
        else:
            gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
            print(f"Using GPUs: {gpu_list}")

            # Run folds in parallel via subprocess
            folds = build_oof_folds(train_eps, seed=20260717)
            n_feas = sum(1 for ep in train_eps if ep.has_feasible)
            n_nofeas = len(train_eps) - n_feas
            script = Path(__file__).resolve()
            oof_staging = staging / "oof_staging"
            oof_staging.mkdir(parents=True, exist_ok=True)

            # Write episode data to disk for subprocesses
            # (can't pass large objects via env; write to staging)
            import pickle
            with open(oof_staging / "episodes.pkl", "wb") as fh:
                pickle.dump(train_eps, fh)
            with open(oof_staging / "registry_map.pkl", "wb") as fh:
                pickle.dump(registry_map, fh)

            procs = []
            for fi, (train_idx, val_idx) in enumerate(folds):
                gpu = gpu_list[fi % len(gpu_list)]
                cmd = [
                    sys.executable, str(script),
                    "--oof-fold-only", str(fi),
                    "--oof-staging", str(oof_staging),
                    "--gpu", str(gpu),
                    "--candidate", args.candidate,
                ]
                p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
                procs.append((fi, p))
                print(f"  Fold {fi+1}: GPU {gpu}, PID {p.pid}")

            for fi, p in procs:
                rc = p.wait()
                if rc != 0: raise RuntimeError(f"Fold {fi+1} failed (exit {rc})")
                print(f"  Fold {fi+1}: done")

            # Collect predictions
            all_oor: list[dict[str, Any]] = []
            for fi in range(5):
                data = torch.load(oof_staging / f"fold_{fi}" / "fold_predictions.pt", map_location="cpu", weights_only=False)
                preds = data["predictions"]
                val_eps = [ep for ep in train_eps if ep.identity in set(data["val_identities"])]
                for tau in OOF_THRESHOLD_GRID:
                    results = evaluate_at_threshold(val_eps, preds, tau)
                    for r in results: r["threshold"] = tau; r["fold"] = fi
                    all_oor.extend(results)

            selected_tau = None
            print("  OOF per-threshold sweep:")
            for tau in OOF_THRESHOLD_GRID:
                tau_results = [r for r in all_oor if abs(r["threshold"] - tau) < 0.005]
                if len(tau_results) != len(train_eps): continue
                m = compute_metrics(tau_results, n_feas, n_nofeas)
                gate_pass = check_oof_gates(m)
                failed = []
                if m["feasible_hit_recall"] < 0.80: failed.append("REC")
                if m["emit_precision"] < 0.80: failed.append("PREC")
                if m["no_corridor_abstention"] < 0.90: failed.append("ABST")
                if m["outside_rankable_emit"] != 0: failed.append("OUTSIDE")
                if m["release_regrasp_emit"] != 0: failed.append("RELREG")
                print(f"    tau={tau:.2f}: rec={m['feasible_hit_recall']:.3f} prec={m['emit_precision']:.3f} "
                      f"abst={m['no_corridor_abstention']:.3f} outside={m['outside_rankable_emit']} "
                      f"relreg={m['release_regrasp_emit']} -> {'PASS' if gate_pass else '/'.join(failed)}")
                if gate_pass and selected_tau is None: selected_tau = tau

            oof_report = {"folds": [{"fold": fi, "val_count": len(folds[fi][1])} for fi in range(5)],
                          "n_feasible": n_feas, "n_no_feasible": n_nofeas, "n_total": len(train_eps),
                          "best_threshold": selected_tau, "all_results": all_oor}
            print(f"  OOF selected threshold: {selected_tau}" if selected_tau else "  OOF: HOLD_OOF")

        if selected_tau is None:
            (staging / "OOF_REPORT.json").write_text(json.dumps(oof_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / "HOLD_OOF.txt").write_text("NO ELIGIBLE THRESHOLD\n", encoding="utf-8")
            _seal_root(staging); os.replace(staging, out)
            print(f"\nHOLD_OOF: {args.candidate} has no eligible threshold.\nRoot: {out}")
            return

        # ── Final Model Training ───────────────────────────────────────────
        print(f"\n--- Final Model (threshold={selected_tau}) ---")
        final_model = model_class()
        final_model, nm, ns, history = train_one_model(final_model, train_eps, seed=20260717, epochs=10, device=args.device)

        # ── Validation ─────────────────────────────────────────────────────
        print("\n--- Validation ---")
        val_eps = load_training_contexts(args.s1_root, args.teacher_root, args.k10_root, val_ids, registry_map)
        n_feas_val = sum(1 for ep in val_eps if ep.has_feasible)
        n_nofeas_val = len(val_eps) - n_feas_val
        print(f"Val: {len(val_eps)} episodes, {n_feas_val} feasible")
        val_preds = predict_episodes(final_model, val_eps, nm, ns, args.device)
        val_ledger: list[dict[str, Any]] = []
        for tau in OOF_THRESHOLD_GRID:
            results = evaluate_at_threshold(val_eps, val_preds, tau)
            for r in results: r["threshold"] = tau; r["candidate"] = args.candidate
            val_ledger.extend(results)
        selected_results = [r for r in val_ledger if abs(r["threshold"] - selected_tau) < 0.005]
        val_metrics = compute_metrics(selected_results, n_feas_val, n_nofeas_val)
        gates = {"recall": val_metrics["feasible_hit_recall"] >= 0.80,
                 "precision": val_metrics["emit_precision"] >= 0.80,
                 "abstention": val_metrics["no_corridor_abstention"] >= 0.90,
                 "false_early": val_metrics.get("n_false_early", 0) / n_feas_val <= 0.05 if n_feas_val else True}
        gate_pass = all(gates.values())
        for k, v in gates.items(): print(f"    {k}: {'PASS' if v else 'FAIL'}")

        # ── Write artifacts ────────────────────────────────────────────────
        print("\nWriting artifacts...")
        torch.save({"schema": "R7_K10_DETECTOR_DEVELOPMENT_CHECKPOINT_V1", "candidate": args.candidate,
                    "model_state": final_model.state_dict(), "normalization_mean_25d": nm.cpu(),
                    "normalization_std_25d": ns.cpu(), "seed": 20260717, "epochs": 10,
                    "oof_selected_threshold": selected_tau, "formal_training_authorized": False,
                    "formal_attack_authorized": False}, staging / "checkpoint.pt")
        (staging / "PROTOCOL.json").write_text(json.dumps({"schema": "R7_K10_DETECTOR_TRAINING_PROTOCOL_V1",
            "candidate": args.candidate, "seed": 20260717, "epochs": 10, "optimizer": "AdamW",
            "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 8, "clip_grad_norm": 5.0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "SOURCE_BINDING.json").write_text(json.dumps({"schema": "R7_K10_DETECTOR_SOURCE_BINDING_V1",
            "git_commit": git_commit, "s1_root_sha256s_sha256": sha256_file(args.s1_root / "SHA256SUMS"),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root / "SHA256SUMS"),
            "k10_root_sha256s_sha256": sha256_file(args.k10_root / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root / "SHA256SUMS")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "TRAIN_HISTORY.json").write_text(json.dumps({"losses": history}, indent=2) + "\n", encoding="utf-8")
        (staging / "IDENTITY_MANIFEST.json").write_text(json.dumps({"train_identities": train_ids,
            "train_count": len(train_ids), "val_identities": val_ids, "val_count": len(val_ids), "fold_id": 0},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with open(staging / "EPISODE_THRESHOLD_LEDGER.jsonl", "w", encoding="utf-8") as fh:
            for e in val_ledger: fh.write(json.dumps(e, sort_keys=True) + "\n")
        with open(staging / "THRESHOLD_METRICS.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["threshold", "feasible_hit_recall", "emit_precision",
                "no_corridor_abstention", "n_hit", "n_emit", "n_false", "n_false_early", "n_feasible", "n_no_feasible"],
                extrasaction="ignore")
            w.writeheader()
            for tau in OOF_THRESHOLD_GRID:
                m = compute_metrics([r for r in val_ledger if abs(r["threshold"] - tau) < 0.005], n_feas_val, n_nofeas_val)
                m["threshold"] = tau; w.writerow(m)
        (staging / "OOF_REPORT.json").write_text(json.dumps(oof_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "AUDIT.json").write_text(json.dumps({"schema": "R7_K10_DETECTOR_AUDIT_V1",
            "candidate": args.candidate, "oof_selected_threshold": selected_tau,
            "validation_gate": {k: v for k, v in gates.items()}, "gate_pass": gate_pass,
            "validation_metrics": val_metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "MANIFEST.json").write_text(json.dumps({"schema": "R7_K10_DETECTOR_MANIFEST_V1",
            "candidate": args.candidate, "fold_id": 0, "train_count": len(train_ids),
            "val_count": len(val_ids), "oof_selected_threshold": selected_tau, "gate_pass": gate_pass},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        root_sha = _seal_root(staging); os.replace(staging, out)
        print(f"\nRoot: {out}\nSHA256SUMS: {root_sha}")
        print(f"\n=== R7.3 {args.candidate}: {'PASS' if gate_pass else 'FAIL'} ===")

    except Exception:
        import shutil
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)
        raise


# ── OOF fold subprocess entry ───────────────────────────────────────────────
def run_single_oof_fold():
    """Entry point for subprocess: train one OOF fold and save to staging."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-fold-only", type=int, required=True)
    ap.add_argument("--oof-staging", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--candidate", choices=["R7-S-LINEAR-25D", "R7-A-GRU-25D"], required=True)
    args, _ = ap.parse_known_args()

    # Load episode data from staging
    import pickle
    with open(args.oof_staging / "episodes.pkl", "rb") as fh:
        train_eps = pickle.load(fh)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"OOF Fold {args.oof_fold_only}: device={device}")

    model_class = R7SLinear25D if args.candidate == "R7-S-LINEAR-25D" else R7AGRU25D
    folds = build_oof_folds(train_eps, seed=20260717)
    train_idx, val_idx = folds[args.oof_fold_only]

    train_subset = [train_eps[i] for i in train_idx]
    val_subset = [train_eps[i] for i in val_idx]
    print(f"Fold {args.oof_fold_only+1}: train={len(train_subset)} val={len(val_subset)}")

    t0 = time.time()
    model = model_class()
    model, nm, ns, history = train_one_model(model, train_subset, seed=20260717, epochs=10, device=device)
    preds = predict_episodes(model, val_subset, nm, ns, device)
    elapsed = time.time() - t0
    print(f"Fold {args.oof_fold_only+1}: done in {elapsed:.0f}s")

    fold_dir = args.oof_staging / f"fold_{args.oof_fold_only}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"fold": args.oof_fold_only, "val_identities": [ep.identity for ep in val_subset],
                "train_identities": [train_subset[i].identity for i in range(len(train_subset))],
                "predictions": preds, "train_count": len(train_idx), "val_count": len(val_idx),
                "normalization_mean_25d": nm.cpu(), "normalization_std_25d": ns.cpu(),
                "train_history": history},
               fold_dir / "fold_predictions.pt")
    torch.save({"model_state": model.state_dict(), "fold": args.oof_fold_only,
                "normalization_mean_25d": nm.cpu(), "normalization_std_25d": ns.cpu()},
               fold_dir / "checkpoint.pt")


if __name__ == "__main__":
    # Check if this is a subprocess call for a single fold
    if "--oof-fold-only" in sys.argv:
        run_single_oof_fold()
    else:
        main()
