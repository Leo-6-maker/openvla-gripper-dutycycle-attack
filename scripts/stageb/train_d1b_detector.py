#!/usr/bin/env python3
"""D1b.1: Learned critical-CLOSE candidate ranker — training runner.

Frozen protocol:
  - Offline per-trace candidate ranker (NOT online trigger/detector).
  - 100% coverage: always emits highest-scoring candidate per trace.
  - Zero-stdev features → normalized to 0.0 (not NaN/Inf).
  - Missing values → train mean imputation.
  - Pairwise margin ranking loss, per-trace grouped batches.
  - Single seed=42 (feasibility canary). Multi-seed deferred to D2.
  - Early stop on val per-trace top-1 accuracy.
  - Checkpoint: best val top-1 → lower MAE → earlier epoch.

All SHAs frozen before execution. Training only when audited.
"""

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Frozen constants ──
TRAINING_SEED = 42
BATCH_SIZE_TRACES = 8  # per-trace grouped (all candidates from same trace)
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 30
LR = 0.001
WEIGHT_DECAY = 1e-4
MARGIN = 0.5
TIE_TOLERANCE = 0.001  # score difference below this → tie
ZERO_STDEV_THRESHOLD = 1e-8
CLIP_RANGE = 3.0

# All 16 features in frozen order
FEATURE_NAMES = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
    "eef_deceleration_bonus", "qpos_ready_bonus", "eef_speed_now", "eef_speed_prev",
    "eef_deceleration_delta", "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]

torch.manual_seed(TRAINING_SEED)
np.random.seed(TRAINING_SEED)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


class CandidateRanker(nn.Module):
    """MLP: 16 → 128 → 64 → 32 → 1 (scalar score per candidate)."""
    def __init__(self, n_features=16, hidden=(128, 64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_normalization(norm_csv):
    """Load z-score mean/stdev. Zero-stdev features → write 0.0 during normalize."""
    rows = list(csv.DictReader(open(norm_csv)))
    means = {}; stdevs = {}; impute = {}
    for r in rows:
        fn = r["feature"]
        m = float(r["mean"]) if r["mean"] else 0.0
        s = float(r["stdev"]) if r["stdev"] else 0.0
        means[fn] = m
        stdevs[fn] = s
        impute[fn] = m  # missing → train mean
    return means, stdevs, impute


def normalize_features(candidates, means, stdevs, impute):
    """Normalize with frozen stats. Zero-stdev → 0.0. Missing → impute. Clip [-3,3]."""
    X = []
    for c in candidates:
        row = []
        for fn in FEATURE_NAMES:
            v = c.get(fn, "")
            if v == "" or v is None:
                v = impute[fn]
            else:
                try: v = float(v)
                except: v = impute[fn]
            s = stdevs[fn]
            if s < ZERO_STDEV_THRESHOLD:
                nv = 0.0
            else:
                nv = (v - means[fn]) / s
            row.append(max(-CLIP_RANGE, min(CLIP_RANGE, nv)))
        X.append(row)
    X = torch.tensor(X, dtype=torch.float32)
    assert torch.isfinite(X).all(), f"Non-finite values after normalization"
    return X


def per_trace_top1_accuracy(model, traces, means, stdevs, impute, device):
    """Per-trace: candidate with highest score == Teacher-P? Tie → incorrect."""
    model.eval()
    correct = 0; total = 0; errors = []
    with torch.no_grad():
        for tid, candidates in traces.items():
            X = normalize_features(candidates, means, stdevs, impute).to(device)
            scores = model(X).cpu().numpy()
            # Find argmax with tie tolerance
            best_idx = int(np.argmax(scores))
            max_score = scores[best_idx]
            ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]
            total += 1
            is_correct = False
            if len(ties) == 1 and int(candidates[best_idx].get("is_teacher_p", 0)) == 1:
                correct += 1; is_correct = True
            # MAE
            tp_step = None
            for c in candidates:
                if int(c.get("is_teacher_p", 0)) == 1:
                    tp_step = int(c["candidate_step"]); break
            pred_step = int(candidates[best_idx]["candidate_step"])
            err = abs(pred_step - tp_step) if tp_step is not None else -1
            errors.append({"trace_id": tid, "correct": is_correct, "abs_error": err, "n_ties": len(ties)})
    return correct / total if total > 0 else 0, errors


def train_epoch(model, optimizer, train_traces, means, stdevs, impute, device):
    """One epoch: per-trace pairwise margin ranking loss."""
    model.train()
    total_loss = 0.0; n_batches = 0
    trace_ids = list(train_traces.keys())
    np.random.shuffle(trace_ids)
    for i in range(0, len(trace_ids), BATCH_SIZE_TRACES):
        batch_ids = trace_ids[i:i + BATCH_SIZE_TRACES]
        batch_loss = 0.0; n_pairs = 0
        for tid in batch_ids:
            candidates = train_traces[tid]
            X = normalize_features(candidates, means, stdevs, impute).to(device)
            scores = model(X)
            # Find positive index
            pos_idx = None
            for j, c in enumerate(candidates):
                if int(c.get("is_teacher_p", 0)) == 1:
                    pos_idx = j; break
            if pos_idx is None: continue
            pos_score = scores[pos_idx]
            # Pairwise loss: margin(max(0, neg - pos + margin)) across all negatives
            neg_mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
            neg_mask[pos_idx] = False
            if neg_mask.sum() == 0: continue
            neg_scores = scores[neg_mask]
            losses = torch.clamp(neg_scores - pos_score + MARGIN, min=0)
            batch_loss += losses.mean(); n_pairs += 1
        if n_pairs > 0:
            batch_loss = batch_loss / n_pairs
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item(); n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="d1b_training_manifest.csv")
    ap.add_argument("--candidate-table", required=True, help="l12_e4c2b_close_candidates.csv")
    ap.add_argument("--norm-csv", required=True, help="d1b_feature_normalization.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # Runtime provenance
    artifact_hashes = {
        "runner_sha": sha256_file(__file__),
        "manifest_sha": sha256_file(args.manifest),
        "candidate_table_sha": sha256_file(args.candidate_table),
        "norm_csv_sha": sha256_file(args.norm_csv),
    }
    print("=== RUNTIME ARTIFACTS ===")
    for k, v in artifact_hashes.items():
        print(f"  {k}: {v[:16]}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    manifest = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}
    candidates = list(csv.DictReader(open(args.candidate_table)))
    means, stdevs, impute = load_normalization(args.norm_csv)

    # Group candidates by trace, filter to training manifest
    by_trace = defaultdict(list)
    for c in candidates:
        tid = c["trace_id"]
        if tid in manifest:
            by_trace[tid].append(c)

    # Split
    train_traces = {tid: cands for tid, cands in by_trace.items() if manifest[tid]["split"] == "train"}
    val_traces = {tid: cands for tid, cands in by_trace.items() if manifest[tid]["split"] == "val"}
    print(f"Train traces: {len(train_traces)}  Val traces: {len(val_traces)}")

    # Verify all train values finite after normalization
    for tid, cands in train_traces.items():
        X = normalize_features(cands, means, stdevs, impute)
        assert torch.isfinite(X).all(), f"Non-finite in train trace {tid}"

    model = CandidateRanker(n_features=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0; best_epoch = -1; best_state = None
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, train_traces, means, stdevs, impute, device)
        val_acc, val_errors = per_trace_top1_accuracy(model, val_traces, means, stdevs, impute, device)
        val_mae = np.mean([e["abs_error"] for e in val_errors if e["abs_error"] >= 0])

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_top1_acc": round(val_acc, 6), "val_mae": round(val_mae, 2)})

        # Checkpoint rule: best val top-1 → lower MAE → earlier epoch
        is_better = False
        if val_acc > best_val_acc + TIE_TOLERANCE:
            is_better = True
        elif abs(val_acc - best_val_acc) < TIE_TOLERANCE and val_mae < best_val_mae - TIE_TOLERANCE:
            is_better = True
        elif abs(val_acc - best_val_acc) < TIE_TOLERANCE and abs(val_mae - best_val_mae) < TIE_TOLERANCE and epoch < best_epoch:
            is_better = True

        if is_better:
            best_val_acc = val_acc; best_val_mae = val_mae; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: loss={train_loss:.4f} val_acc={val_acc:.4f} val_mae={val_mae:.1f} best_ep={best_epoch} patience={patience_counter}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    # Save checkpoint
    ckpt_path = out / "d1b_detector_checkpoint.pt"
    torch.save({
        "model_state": best_state,
        "epoch": best_epoch,
        "val_top1_acc": best_val_acc,
        "val_mae": best_val_mae,
        "config": {"n_features": 16, "hidden": [128, 64, 32], "dropout": 0.1, "seed": TRAINING_SEED},
        "normalization": {"means": means, "stdevs": stdevs, "impute": impute},
        "artifact_hashes": artifact_hashes,
    }, ckpt_path)

    # Final evaluation on val
    val_acc_final, val_errors_final = per_trace_top1_accuracy(model, val_traces, means, stdevs, impute, device)
    val_mae_final = np.mean([e["abs_error"] for e in val_errors_final if e["abs_error"] >= 0]) if val_errors_final else 0

    print(f"\nBest checkpoint: epoch={best_epoch} val_acc={best_val_acc:.4f} val_mae={best_val_mae:.1f}")
    print(f"Final val: acc={val_acc_final:.4f} mae={val_mae_final:.1f}")

    # Save training history
    with open(out / "d1b_training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_top1_acc", "val_mae"])
        w.writeheader(); w.writerows(history)

    # Run log
    end_time = datetime.now(timezone.utc)
    with open(out / "d1b_training_run_log.txt", "w") as f:
        f.write(f"D1b TRAINING RUN LOG\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        f.write(f"runner_sha: {artifact_hashes['runner_sha']}\n")
        f.write(f"manifest_sha: {artifact_hashes['manifest_sha']}\n")
        f.write(f"candidate_table_sha: {artifact_hashes['candidate_table_sha']}\n")
        f.write(f"norm_csv_sha: {artifact_hashes['norm_csv_sha']}\n")
        f.write(f"device: {device}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_top1_acc: {best_val_acc}\n")
        f.write(f"best_val_mae: {best_val_mae}\n")
        f.write(f"checkpoint_path: {ckpt_path}\n")
        ckpt_sha = sha256_file(str(ckpt_path))
        f.write(f"checkpoint_sha256: {ckpt_sha}\n")
        artifact_hashes["checkpoint_sha"] = ckpt_sha

    print(f"\nTRAINING COMPLETE — checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
