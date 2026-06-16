#!/usr/bin/env python3
"""D5: Extract candidate dataset from accepted episodes, then train online detector.

Reuses:
  - train_d1b_detector.py: CandidateRanker, FEATURE_NAMES, normalize_features
  - Accepted episode manifest for exact binding
  - Teacher-P labels v2 for supervision
"""
import csv, hashlib, json, math, os, sys, time
from collections import defaultdict, Counter
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "stageb"))

# ── Config ──
SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 30
LR = 0.001
WEIGHT_DECAY = 1e-4
MARGIN = 0.5
TIE_TOLERANCE = 0.001

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(SEED)
np.random.seed(SEED)

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features, load_normalization

ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}
ACCEPTED = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
OUT = "/data/liuyu/outputs/d5_training"
os.makedirs(OUT, exist_ok=True)


def main():
    # ── Load accepted manifest ──
    accepted = {}
    for r in csv.DictReader(open(ACCEPTED)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r
    print("Accepted episodes: {}".format(len(accepted)))

    # ── Load Teacher-P labels ──
    labels = {}
    for r in csv.DictReader(open(LABELS)):
        key = (r["task"], int(r["state_id"]))
        labels[key] = {"status": r["status"], "anchor": int(r["anchor"]), "ws": int(r["ws"]), "we": int(r["we"])}

    # ── Extract candidates with Teacher-P labels ──
    candidate_rows = []
    manifest_rows = []
    trace_id_map = {}

    for (task, sid), acc in sorted(accepted.items()):
        sp = acc["split"]
        rname = acc["accepted_root"]
        edir_name = acc["accepted_episode_dir"]
        rpath = ROOTS.get(rname, "")
        edir = os.path.join(rpath, edir_name) if rpath else ""

        if not os.path.isdir(edir):
            continue

        tid = "d5_{}_s{}".format(task, sid)
        trace_id_map[tid] = {"task": task, "state_id": sid, "split": sp}

        # Teacher-P
        lp = labels.get((task, sid), {})
        p_anchor = lp.get("anchor", -1)
        p_status = lp.get("status", "UNKNOWN")

        # Only include labeled episodes for training; abstain as separate report
        if p_status == "VALID_TEACHER_P_ABSTAIN":
            manifest_rows.append({"trace_id": tid, "task": task, "state_id": sid,
                                  "split": sp, "teacher_p_anchor": -1, "teacher_p_status": "ABSTAIN"})
            continue
        if p_status != "VALID_LABELED":
            manifest_rows.append({"trace_id": tid, "task": task, "state_id": sid,
                                  "split": sp, "teacher_p_anchor": -1, "teacher_p_status": p_status})
            continue

        manifest_rows.append({"trace_id": tid, "task": task, "state_id": sid,
                              "split": sp, "teacher_p_anchor": p_anchor, "teacher_p_status": "LABELED"})

        # Read detector candidates
        ccf = os.path.join(edir, "detector_candidates.csv")
        if not os.path.exists(ccf):
            continue

        cands = list(csv.DictReader(open(ccf)))
        for c in cands:
            step = int(c["step"])
            dist = abs(step - p_anchor)
            is_positive = 1 if dist == 0 else 0
            abstained = int(c.get("abstained", 0) or 0) == 1
            row = {"trace_id": tid, "step": step, "is_teacher_p": is_positive,
                   "distance_to_p": step - p_anchor, "abstained": int(abstained)}
            for fn in FEATURE_NAMES:
                row[fn] = c.get("feat_" + fn, c.get(fn, ""))
            # Abstained candidates cannot be positive (predictor refused)
            if abstained and is_positive:
                row["is_teacher_p"] = 0
            candidate_rows.append(row)

    # ── Write manifest ──
    man_path = os.path.join(OUT, "d5_training_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace_id", "task", "state_id", "split",
                                          "teacher_p_anchor", "teacher_p_status"])
        w.writeheader()
        w.writerows(manifest_rows)

    # ── Write candidate table ──
    ct_path = os.path.join(OUT, "d5_close_candidates.csv")
    with open(ct_path, "w", newline="") as f:
        fields = ["trace_id", "step", "is_teacher_p", "distance_to_p", "abstained"] + FEATURE_NAMES
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(candidate_rows)

    # ── Split ──
    train_tids = set(r["trace_id"] for r in manifest_rows if r["split"] == "train" and r["teacher_p_status"] == "LABELED")
    val_tids = set(r["trace_id"] for r in manifest_rows if r["split"] == "val" and r["teacher_p_status"] == "LABELED")
    test_tids = set(r["trace_id"] for r in manifest_rows if r["split"] == "test" and r["teacher_p_status"] == "LABELED")

    print("\nSplit: train={} val={} test={} abstain={}".format(
        len(train_tids), len(val_tids), len(test_tids),
        sum(1 for r in manifest_rows if r["teacher_p_status"] == "ABSTAIN")))

    # ── Verify: each included trace has exactly 1 teacher_p positive ──
    pos_counts = defaultdict(int)
    for c in candidate_rows:
        if c["is_teacher_p"] == 1:
            pos_counts[c["trace_id"]] += 1
    bad = {tid: n for tid, n in pos_counts.items() if n != 1}
    if bad:
        print("WARNING: traces with !=1 positive:")
        for tid, n in sorted(bad.items())[:5]:
            print("  {}: {} positives".format(tid, n))

    # ── Training ──
    device = torch.device("cpu")
    print("\nDevice: {}".format(device))

    # Group candidates by trace (all labeled for evaluation)
    by_trace = defaultdict(list)
    for c in candidate_rows:
        tid = c["trace_id"]
        if tid in train_tids or tid in val_tids or tid in test_tids:
            by_trace[tid].append(c)

    train_cands = {tid: cs for tid, cs in by_trace.items() if tid in train_tids}
    val_cands = {tid: cs for tid, cs in by_trace.items() if tid in val_tids}
    print("Train traces: {}  Val traces: {}".format(len(train_cands), len(val_cands)))

    # Compute normalization from train
    feat_sums = {fn: 0.0 for fn in FEATURE_NAMES}
    feat_sqsums = {fn: 0.0 for fn in FEATURE_NAMES}
    feat_counts = {fn: 0 for fn in FEATURE_NAMES}
    for tid, cands in train_cands.items():
        for c in cands:
            for fn in FEATURE_NAMES:
                v = c.get(fn, "")
                if v != "" and v is not None:
                    try:
                        vf = float(v)
                        if math.isfinite(vf):
                            feat_sums[fn] += vf
                            feat_sqsums[fn] += vf * vf
                            feat_counts[fn] += 1
                    except:
                        pass

    means = {}; stdevs = {}; impute = {}
    for fn in FEATURE_NAMES:
        n = feat_counts.get(fn, 0)
        if n > 0:
            m = feat_sums[fn] / n
            var = max(0, feat_sqsums[fn] / n - m * m)
            s = math.sqrt(var) if var > 1e-8 else 0.0
        else:
            m = 0.0; s = 0.0
        means[fn] = m; stdevs[fn] = s; impute[fn] = m

    # Save normalization
    norm_path = os.path.join(OUT, "d5_feature_normalization.csv")
    with open(norm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean", "stdev"])
        for fn in FEATURE_NAMES:
            w.writerow([fn, means[fn], stdevs[fn]])

    # Build model
    model = CandidateRanker(n_features=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def train_epoch():
        model.train()
        tids = list(train_cands.keys())
        np.random.shuffle(tids)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(tids), BATCH_SIZE):
            batch_tids = tids[i:i + BATCH_SIZE]
            all_cands = []
            offsets = []
            for tid in batch_tids:
                offsets.append(len(all_cands))
                all_cands.extend(train_cands[tid])
            if len(all_cands) < 2:
                continue
            X = normalize_features(all_cands, means, stdevs, impute)
            scores = model(X.to(device))
            losses = []
            for j, tid in enumerate(batch_tids):
                start = offsets[j]
                end = offsets[j + 1] if j + 1 < len(offsets) else len(all_cands)
                if start >= end:
                    continue
                local_scores = scores[start:end]
                local_cands = all_cands[start:end]
                pos_idx = [k for k, c in enumerate(local_cands) if int(c.get("is_teacher_p", 0)) == 1]
                neg_idx = [k for k, c in enumerate(local_cands) if int(c.get("is_teacher_p", 0)) == 0]
                if not pos_idx or not neg_idx:
                    continue
                pos_scores = local_scores[pos_idx].unsqueeze(1)
                neg_scores = local_scores[neg_idx].unsqueeze(0)
                margin_loss = torch.clamp(MARGIN - pos_scores + neg_scores, min=0)
                losses.append(margin_loss.mean())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def eval_top1(cands_dict):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for tid, cands in cands_dict.items():
                X = normalize_features(cands, means, stdevs, impute)
                scores = model(X.to(device))
                best = torch.argmax(scores).item()
                if int(cands[best].get("is_teacher_p", 0)) == 1:
                    correct += 1
                total += 1
        return correct / max(1, total)

    # Train
    best_val_acc = -1.0
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tloss = train_epoch()
        vacc = eval_top1(val_cands)
        train_acc = eval_top1(train_cands)
        history.append({"epoch": epoch, "train_loss": round(tloss, 6),
                        "train_acc": round(train_acc, 4), "val_acc": round(vacc, 4)})
        if epoch <= 3 or epoch % 10 == 0 or vacc > best_val_acc:
            print("Epoch {:3d}: loss={:.4f} train_acc={:.4f} val_acc={:.4f}".format(
                epoch, tloss, train_acc, vacc))
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stop at epoch {}".format(epoch))
                break

    # Restore best
    model.load_state_dict(best_state)
    final_train_acc = eval_top1(train_cands)
    final_val_acc = eval_top1(val_cands)

    # Test evaluation
    test_cands = {tid: cs for tid, cs in by_trace.items() if tid in test_tids}
    final_test_acc = eval_top1(test_cands) if test_cands else 0.0

    print("\n=== Final ===")
    print("Train top-1: {:.4f}".format(final_train_acc))
    print("Val top-1:   {:.4f}".format(final_val_acc))
    print("Test top-1:  {:.4f} ({} traces)".format(final_test_acc, len(test_cands)))

    # Save checkpoint
    ckpt = {
        "model_state": best_state,
        "means": means, "stdevs": stdevs, "impute": impute,
        "normalization": {"means": means, "stdevs": stdevs, "impute": impute},
        "n_features": 16, "feature_names": FEATURE_NAMES,
        "best_val_acc": best_val_acc, "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "history": history,
    }
    ckpt_path = os.path.join(OUT, "d5_candidate_best.pt")
    torch.save(ckpt, ckpt_path)

    # Save history
    hist_path = os.path.join(OUT, "d5_training_history.csv")
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_acc"])
        w.writeheader()
        w.writerows(history)

    print("Checkpoint: {}".format(ckpt_path))
    print("Manifest:   {}".format(man_path))
    print("Candidates: {}".format(ct_path))
    print("History:    {}".format(hist_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
