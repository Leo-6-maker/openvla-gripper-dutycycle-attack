#!/usr/bin/env python3
"""Q1/Q2 Detector V2.1: Single quality-head GRU with clean V2.1 labels.

Architecture:
  candidate_close = deterministic rule (close event membership)
  Model: P(valid_retention_quality | features, causal history)
  EMIT = candidate_close AND quality_score >= threshold

Q1: View B (33D) + GRU128 + single quality head
Q2: View C (39D) + GRU128 + single quality head (+ auxiliary release head)

Key fixes over V4 V1:
  1. V2.1 labels: quality_valid AND veto_invalid = 0 (hard invariant)
  2. Single quality head (no conflicting veto head)
  3. Cross-episode window ranking loss
  4. Proper two-stage evaluator with threshold sweep
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, random, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Reuse feature derivation from V4 V1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v4_detector import (
    derive_dynamic_features, ALL_VIEWS, jsonl as read_jsonl, sha256_text,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FIT_STATES = list(range(0, 20))


# ── data loading ───────────────────────────────────────────────────────
@dataclass
class QEpisode:
    identity: str
    suite: str
    task_idx: int
    state_id: int
    fold_id: int
    features: Tensor  # [T, F]
    quality_target: Tensor  # [T] 1=quality_valid, 0=veto_invalid or background
    known_mask: Tensor  # [T] bool
    candidate_close: Tensor  # [T] bool
    release_target: Tensor  # [T] 1=release_imminent
    n_steps: int


def load_q_episode(s1_root: Path, v21_root: Path,
                   suite: str, task: int, state: int,
                   view: str) -> Optional[QEpisode]:
    s1_path = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "student_input_records.jsonl"
    v21_path = v21_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_v21_labels.jsonl"
    if not s1_path.exists() or not v21_path.exists():
        return None

    students = read_jsonl(s1_path)
    v21_labels = read_jsonl(v21_path)
    if not students:
        return None

    features_25d = torch.tensor(
        [[float(v) for v in r["features_25d"]] for r in students], dtype=torch.float32)
    features = derive_dynamic_features(features_25d, view)
    T = features.shape[0]

    quality = torch.zeros(T)
    known = torch.zeros(T, dtype=torch.bool)
    cand_close = torch.zeros(T, dtype=torch.bool)
    release = torch.zeros(T)

    for i, label in enumerate(v21_labels):
        if i >= T:
            break
        if label["known_mask"]:
            known[i] = True
            if label["quality_valid"]:
                quality[i] = 1.0
            elif label["veto_invalid"]:
                quality[i] = 0.0
            else:
                quality[i] = -1.0  # will be masked
                known[i] = False
        # candidate_close from deterministic rule
        cand_close[i] = label["candidate_close"]
        release[i] = float(label["release_imminent"] and label["known_mask"])

    # Only supervise on known candidate_close steps
    supervise_mask = known & cand_close

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
    return QEpisode(
        identity=cid, suite=suite, task_idx=task, state_id=state,
        fold_id=state // 5, features=features,
        quality_target=quality, known_mask=supervise_mask,
        candidate_close=cand_close, release_target=release,
        n_steps=T,
    )


# ── model ──────────────────────────────────────────────────────────────
class QualityGRU(nn.Module):
    """Single quality-head GRU. No veto head, no conflicting labels."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 aux_release: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.aux_release = aux_release
        if aux_release:
            self.release_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, mask: Tensor
                ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        """Returns: quality_logits [B,T], release_logits [B,T]|None, final_hidden"""
        B, T_val, _ = x.shape
        lengths = mask.sum(dim=1).cpu().long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out_packed, final_hidden = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out_packed, batch_first=True, total_length=T_val)

        quality_logits = self.quality_head(gru_out).squeeze(-1)  # [B, T]
        release_logits = None
        if self.aux_release:
            release_logits = self.release_head(gru_out).squeeze(-1)
        return quality_logits, release_logits, final_hidden


# ── loss ────────────────────────────────────────────────────────────────
def loss_bce_quality(quality_logits: Tensor, targets: Tensor, masks: Tensor,
                     **kwargs) -> Tensor:
    """Masked BCE on quality target (only candidate_close known steps)."""
    loss = F.binary_cross_entropy_with_logits(quality_logits, targets, reduction="none")
    n = masks.sum().clamp_min(1)
    return (loss * masks.float()).sum() / n


def loss_cross_window_ranking(quality_logits: Tensor, targets: Tensor,
                              masks: Tensor, episode_ids: list[str],
                              margin: float = 0.3, weight: float = 0.5,
                              cand_close: Tensor = None,
                              **kwargs) -> Tensor:
    """BCE + DIFFERENTIABLE cross-episode window ranking loss.

    For each episode, compute mean quality score over VALID_RETENTION
    steps (target=1) vs INVALID steps (target=0), then enforce
    score(valid_window) > score(invalid_window) + margin across episodes.
    All scores stay in the computation graph (no .item() calls).
    """
    base = loss_bce_quality(quality_logits, targets, masks)

    B = quality_logits.shape[0]
    if B < 2:
        return base

    q_probs = torch.sigmoid(quality_logits)  # [B, T]

    # Compute per-episode mean quality scores over supervised steps,
    # split by positive (target=1) and negative (target=0)
    pos_scores = []  # list of scalar tensors
    neg_scores = []
    for b in range(B):
        m = masks[b]
        if m.sum() == 0:
            continue
        t = targets[b][m]
        s = q_probs[b][m]
        pos_mask = t > 0.5
        neg_mask = t < 0.5
        if pos_mask.any():
            pos_scores.append(s[pos_mask].mean())  # stays in graph
        if neg_mask.any():
            neg_scores.append(s[neg_mask].mean())

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return base

    # Cross-episode: pair each positive episode with a different negative episode
    pos_t = torch.stack(pos_scores)  # [N_pos]
    neg_t = torch.stack(neg_scores)  # [N_neg]

    # All pairwise hinge losses
    # pos_t[i] should be > neg_t[j] + margin for all i,j
    # Expand to [N_pos, N_neg]
    pos_exp = pos_t.unsqueeze(1).expand(-1, len(neg_t))   # [N_pos, N_neg]
    neg_exp = neg_t.unsqueeze(0).expand(len(pos_t), -1)    # [N_pos, N_neg]
    hinge = F.relu(margin - (pos_exp - neg_exp))            # [N_pos, N_neg]
    rank_loss = hinge.mean()

    return base + weight * rank_loss


def loss_with_release(quality_logits: Tensor, targets: Tensor, masks: Tensor,
                      release_logits: Tensor = None, release_targets: Tensor = None,
                      release_weight: float = 0.3, **kwargs) -> Tensor:
    """BCE quality + BCE release auxiliary loss."""
    base = loss_bce_quality(quality_logits, targets, masks)
    if release_logits is None or release_targets is None:
        return base

    # Release loss on all known steps (not just candidate_close)
    release_mask = (release_targets >= 0)  # valid label exists
    if release_mask.sum() == 0:
        return base
    # release_targets: 1=release_imminent, 0=not
    rel_loss = F.binary_cross_entropy_with_logits(
        release_logits, release_targets.clamp(0, 1), reduction="none")
    n_rel = release_mask.sum().clamp_min(1)
    rel_term = (rel_loss * release_mask.float()).sum() / n_rel
    return base + release_weight * rel_term


LOSS_FNS = {
    "BCE": loss_bce_quality,
    "RANK": loss_cross_window_ranking,
    "BCE_RELEASE": loss_with_release,
}


# ── training ───────────────────────────────────────────────────────────
def train_q_model(
    episodes: list[QEpisode],
    view: str,
    loss_name: str,
    aux_release: bool = False,
    seed: int = 20260717,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
):
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(device)

    input_dim = ALL_VIEWS[view].feature_count
    model = QualityGRU(input_dim, aux_release=aux_release).to(device_obj)
    model.train()

    # Normalization from training episodes
    all_f = torch.cat([ep.features for ep in episodes], dim=0)
    norm_mean = all_f.mean(dim=0)
    norm_std = all_f.std(dim=0, unbiased=False).clamp_min(1e-6)

    loss_fn = LOSS_FNS[loss_name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Group for balanced sampling
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[(ep.suite, ep.task_idx)].append(i)

    epoch_losses = []
    for epoch in range(epochs):
        indices = []
        rng = random.Random(seed + epoch * 1000)
        queues = {k: list(v) for k, v in sorted(groups.items())}
        for v in queues.values():
            rng.shuffle(v)
        while any(queues.values()):
            for k in sorted(queues):
                if queues[k]:
                    indices.append(queues[k].pop(0))

        epoch_terms = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_eps = [episodes[i] for i in batch_idx]
            max_T = max(ep.n_steps for ep in batch_eps)
            B = len(batch_eps)
            F = batch_eps[0].features.shape[1]

            x = torch.zeros(B, max_T, F)
            padding = torch.zeros(B, max_T, dtype=torch.bool)
            q_target = torch.zeros(B, max_T)
            q_mask = torch.zeros(B, max_T, dtype=torch.bool)
            r_target = torch.full((B, max_T), -1.0)  # -1 = no label
            cand_close_batch = torch.zeros(B, max_T, dtype=torch.bool)

            for b, ep in enumerate(batch_eps):
                T_ep = ep.n_steps
                x[b, :T_ep] = (ep.features - norm_mean) / norm_std
                padding[b, :T_ep] = True
                q_target[b, :T_ep] = ep.quality_target
                q_mask[b, :T_ep] = ep.known_mask
                r_target[b, :T_ep] = ep.release_target
                cand_close_batch[b, :T_ep] = ep.candidate_close

            x = x.to(device_obj)
            padding = padding.to(device_obj)
            q_target = q_target.to(device_obj)
            q_mask = q_mask.to(device_obj)
            r_target = r_target.to(device_obj)
            cand_close_batch = cand_close_batch.to(device_obj)

            optimizer.zero_grad(set_to_none=True)
            q_logits, r_logits, _ = model(x, padding)
            ep_ids = [ep.identity for ep in batch_eps]
            loss = loss_fn(q_logits, q_target, q_mask,
                          episode_ids=ep_ids, cand_close=cand_close_batch,
                          release_logits=r_logits, release_targets=r_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_terms.append(float(loss.detach()))

        avg = sum(epoch_terms) / len(epoch_terms) if epoch_terms else 0
        epoch_losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    return model, epoch_losses, (norm_mean, norm_std)


# ── evaluation ─────────────────────────────────────────────────────────
def evaluate_model(
    model: QualityGRU,
    episodes: list[QEpisode],
    norm_mean: Tensor, norm_std: Tensor,
    device: str = "cpu",
):
    """Phase-aware evaluation with threshold sweep.

    Reports four metric layers:
      1. Pure-negative episode any-emit
      2. Invalid-phase any-emit (including from mixed episodes)
      3. Valid-window hit rate
      4. Release-overlap emit
    """
    model.eval()
    device_obj = torch.device(device)

    all_results = []
    with torch.no_grad():
        for ep in episodes:
            x = (ep.features - norm_mean) / norm_std
            x = x.unsqueeze(0).to(device_obj)
            padding = torch.ones(1, ep.n_steps, dtype=torch.bool).to(device_obj)
            q_logits, r_logits, _ = model(x, padding)
            q_probs = torch.sigmoid(q_logits.squeeze(0)).cpu()
            r_probs = torch.sigmoid(r_logits.squeeze(0)).cpu() if r_logits is not None else None

            has_quality = (ep.known_mask & (ep.quality_target > 0.5)).any().item()
            has_veto = (ep.known_mask & (ep.quality_target < 0.5)).any().item()
            is_pure_hard_negative = has_veto and not has_quality
            is_mixed = has_quality and has_veto

            # Threshold sweep with phase-level metrics
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            sweep = []
            for tau in thresholds:
                pred_emit = (q_probs >= tau) & ep.candidate_close

                # Valid-window hit: emit on at least one quality_valid step
                valid_hit = (pred_emit & (ep.quality_target > 0.5)).any().item() if has_quality else None

                # Invalid-phase false emit: emit on veto_invalid step
                veto_mask = (ep.quality_target < 0.5) & ep.known_mask
                invalid_emit = (pred_emit & veto_mask).any().item() if veto_mask.any() else False

                # Pure-negative episode false emit
                pure_false = pred_emit.any().item() and is_pure_hard_negative

                # Release overlap: emit on release_imminent step
                release_mask = ep.release_target > 0.5
                release_emit = (pred_emit & release_mask).any().item()

                sweep.append({
                    "threshold": tau,
                    "any_emit": pred_emit.any().item(),
                    "valid_hit": valid_hit,
                    "pure_negative_false_emit": pure_false,
                    "invalid_phase_false_emit": invalid_emit,
                    "release_overlap_emit": release_emit,
                    "n_emit_steps": int(pred_emit.sum()),
                    "n_invalid_emit_steps": int((pred_emit & veto_mask).sum()) if veto_mask.any() else 0,
                })

            all_results.append({
                "identity": ep.identity,
                "fold_id": ep.fold_id,
                "has_quality": has_quality,
                "has_veto": has_veto,
                "is_pure_hard_negative": is_pure_hard_negative,
                "is_mixed": is_mixed,
                "max_quality_prob": float(q_probs.max()),
                "mean_quality_prob_valid": float(q_probs[(ep.quality_target > 0.5) & ep.known_mask].mean()) if has_quality else 0,
                "mean_quality_prob_invalid": float(q_probs[(ep.quality_target < 0.5) & ep.known_mask].mean()) if has_veto else 0,
                "threshold_sweep": sweep,
            })

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--v21-root", type=Path, required=True)
    ap.add_argument("--view", choices=["A", "B", "C"], default="B")
    ap.add_argument("--loss", choices=["BCE", "RANK", "BCE_RELEASE"], default="BCE")
    ap.add_argument("--aux-release", action="store_true")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-fold", type=int, default=None)
    ap.add_argument("--eval-fold", type=int, default=None)
    args = ap.parse_args()

    model_name = "Q2" if args.aux_release else "Q1"
    print(f"=== {model_name} V2.1: view={args.view} loss={args.loss} seed={args.seed} ===")

    # Load training episodes
    if args.train_fold is not None:
        fold = args.train_fold
        val_set = set(range(fold * 5, (fold + 1) * 5))
        train_states = [s for s in FIT_STATES if s not in val_set]
    else:
        train_states = FIT_STATES

    scope = [(s, t, st) for s in SUITES for t in range(10) for st in train_states]
    train_eps = []
    for suite, task, state in scope:
        ep = load_q_episode(args.s1_root, args.v21_root, suite, task, state, args.view)
        if ep is not None:
            train_eps.append(ep)

    print(f"Training episodes: {len(train_eps)}")
    if not train_eps:
        print("ERROR: no training episodes"); return

    # Train
    model, losses, (norm_mean, norm_std) = train_q_model(
        train_eps, args.view, args.loss,
        aux_release=args.aux_release,
        seed=args.seed, epochs=args.epochs, device=args.device,
    )
    print(f"Final training loss: {losses[-1]:.6f}")

    # Save checkpoint
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "model_state": model.state_dict(),
        "view": args.view, "loss": args.loss,
        "aux_release": args.aux_release,
        "seed": args.seed, "losses": losses,
        "norm_mean": norm_mean, "norm_std": norm_std,
        "n_train_episodes": len(train_eps),
        "schema": "DETECTOR_V4_Q_TRAINED_V21_V1",
    }, args.output / "checkpoint.pt")

    # Evaluate if eval-fold specified
    if args.eval_fold is not None:
        eval_states = list(range(args.eval_fold * 5, (args.eval_fold + 1) * 5))
        eval_scope = [(s, t, st) for s in SUITES for t in range(10) for st in eval_states]
        eval_eps = []
        for suite, task, state in eval_scope:
            ep = load_q_episode(args.s1_root, args.v21_root, suite, task, state, args.view)
            if ep is not None:
                eval_eps.append(ep)

        print(f"Evaluation episodes: {len(eval_eps)}")
        results = evaluate_model(model, eval_eps, norm_mean, norm_std, args.device)

        # Phase-aware aggregate metrics
        n_q = sum(1 for r in results if r["has_quality"])
        n_pure_neg = sum(1 for r in results if r["is_pure_hard_negative"])
        n_mixed = sum(1 for r in results if r["is_mixed"])
        n_total_invalid_phases = sum(1 for r in results if r["has_veto"])

        print(f"\n=== Fold {args.eval_fold} Results (Phase-Aware) ===")
        print(f"  quality episodes: {n_q}")
        print(f"  pure-negative episodes: {n_pure_neg}")
        print(f"  mixed episodes (quality+invalid): {n_mixed}")
        print(f"  total episodes with invalid phases: {n_total_invalid_phases}")

        for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
            q_hits = 0; pure_false = 0; invalid_false = 0; release_emit = 0
            for r in results:
                sweep_entry = next((s for s in r["threshold_sweep"] if abs(s["threshold"] - tau) < 0.01), None)
                if sweep_entry:
                    if sweep_entry["valid_hit"]:
                        q_hits += 1
                    if sweep_entry["pure_negative_false_emit"]:
                        pure_false += 1
                    if sweep_entry["invalid_phase_false_emit"]:
                        invalid_false += 1
                    if sweep_entry["release_overlap_emit"]:
                        release_emit += 1
            q_rate = q_hits / n_q if n_q else 0
            pn_rate = pure_false / n_pure_neg if n_pure_neg else 0
            inv_rate = invalid_false / n_total_invalid_phases if n_total_invalid_phases else 0
            print(f"  tau={tau:.1f}: valid_hit={q_rate:.4f} ({q_hits}/{n_q})  "
                  f"pure-neg-false={pn_rate:.4f} ({pure_false}/{n_pure_neg})  "
                  f"invalid-phase-false={inv_rate:.4f} ({invalid_false}/{n_total_invalid_phases})  "
                  f"release-overlap={release_emit}")

        # Save detailed results
        with open(args.output / "eval_results.json", "w") as fh:
            json.dump({
                "fold": args.eval_fold,
                "view": args.view, "loss": args.loss,
                "model": model_name,
                "n_quality_episodes": n_q,
                "n_pure_negative_episodes": n_pure_neg,
                "n_mixed_episodes": n_mixed,
                "n_total_invalid_phase_episodes": n_total_invalid_phases,
                "results": results,
            }, fh, indent=2)

    # SHA256SUMS
    out_files = sorted(args.output.rglob("*"))
    file_list = [f for f in out_files if f.is_file()]
    with open(args.output / "SHA256SUMS", "w") as fh:
        for fp in file_list:
            rel = fp.relative_to(args.output)
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel}\n")
    sha = hashlib.sha256((args.output / "SHA256SUMS").read_bytes()).hexdigest()
    with open(args.output / "SHA256SUMS.sha256", "w") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")
    print(f"SHA256SUMS: {sha}")


if __name__ == "__main__":
    main()
