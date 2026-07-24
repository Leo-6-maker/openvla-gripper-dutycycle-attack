"""Hidden-state probe v2: Direct V2.1 label + S1 feature loading, no sklearn."""
import json, sys, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v4_detector import (
    derive_dynamic_features, ALL_VIEWS,
    CandidateAGRU, CandidateBGRU, CandidateCGRU,
    jsonl as read_jsonl,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def compute_auc(scores, labels):
    """Trapezoidal AUC for binary classification."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    y_sorted = [labels[i] for i in order]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0; fp = 0; tpr = [0.0]; fpr = [0.0]
    for yi in y_sorted:
        if yi == 1: tp += 1
        else: fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    auc = 0.0
    for i in range(len(fpr)-1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i] + tpr[i+1]) / 2.0
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--v21-root", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    view = ckpt["view"]
    candidate = ckpt["candidate"]
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    # Build model
    if candidate == "B":
        model = CandidateBGRU()
    elif candidate == "A":
        model = CandidateAGRU()
    else:
        model = CandidateCGRU()
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device)
    model.eval()

    # Extract GRU for manual forward with hidden state capture
    if hasattr(model, 'backbone'):
        gru = model.backbone.gru
        head_dict = model.backbone.heads
    elif hasattr(model, 'gru'):
        gru = model.gru
        head_dict = model.heads
    else:
        raise ValueError("Cannot find GRU in model")

    val_states = list(range(args.fold * 5, (args.fold + 1) * 5))
    X_list = []; y_list = []; head_scores = []; head_labels = []; ep_ids = []

    with torch.no_grad():
        for suite in SUITES:
            for task in range(10):
                for state in val_states:
                    cid_short = f"{suite}/task_{task:02d}/state_{state:02d}"
                    v21_path = args.v21_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_v21_labels.jsonl"
                    s1_path = args.s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "student_input_records.jsonl"

                    if not v21_path.exists() or not s1_path.exists():
                        continue

                    v21_labels = read_jsonl(v21_path)
                    students = read_jsonl(s1_path)
                    if len(students) == 0:
                        continue

                    features_25d = torch.tensor(
                        [[float(v) for v in r["features_25d"]] for r in students],
                        dtype=torch.float32
                    )
                    features = derive_dynamic_features(features_25d, view)
                    n_steps = features.shape[0]

                    x = features.unsqueeze(0).to(args.device)
                    x = (x - norm_mean.to(x.device)) / norm_std.to(x.device)

                    # Manual GRU forward to capture hidden states
                    gru.flatten_parameters()
                    gru_out, _ = gru(x)  # [1, T, 128]
                    hs = gru_out.squeeze(0).cpu()  # [T, 128]

                    # Compute head logits manually
                    logits = {}
                    for name, head in head_dict.items():
                        logits[name] = head(gru_out).squeeze(-1)

                    crit_head = "criticality" if "criticality" in logits else "valid_retention"
                    crit_probs = torch.sigmoid(logits[crit_head].squeeze(0)).cpu()

                    for t in range(min(n_steps, len(hs), len(v21_labels))):
                            label = v21_labels[t]
                            if not label["known_mask"] or not label["candidate_close"]:
                                continue
                            if label["quality_valid"]:
                                X_list.append(hs[t].numpy())
                                y_list.append(1)
                                ep_ids.append(cid_short)
                                head_scores.append(float(crit_probs[t]))
                                head_labels.append(1)
                            elif label["veto_invalid"]:
                                X_list.append(hs[t].numpy())
                                y_list.append(0)
                                ep_ids.append(cid_short)
                                head_scores.append(float(crit_probs[t]))
                                head_labels.append(0)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    print(f"Probe data: {len(X)} close-window known steps from {len(set(ep_ids))} episodes")
    print(f"  quality_valid (class 1): {n_pos}")
    print(f"  veto_invalid (class 0): {n_neg}")

    if len(X) < 10 or n_pos == 0 or n_neg == 0:
        print("ERROR: insufficient data")
        return

    # Linear probe
    pos_weight = n_neg / max(n_pos, 1)
    probe = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for epoch in range(500):
        opt.zero_grad()
        loss = bce(probe(X).squeeze(), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probe_scores = torch.sigmoid(probe(X).squeeze()).tolist()
    probe_auc = compute_auc(probe_scores, y_list)
    head_auc = compute_auc(head_scores, head_labels)

    print(f"\nResults:")
    print(f"  Linear probe ROC-AUC:  {probe_auc:.4f}")
    print(f"  Original head ROC-AUC: {head_auc:.4f}")
    print(f"  DELTA (probe - head):  {probe_auc - head_auc:.4f}")

    if probe_auc - head_auc > 0.05:
        print("  → HEAD UNDERFITTING: representation can separate, head/optimization fails")
    elif probe_auc > 0.75:
        print("  → REPRESENTATION OK: linear probe separates; fix labels/loss first")
    elif probe_auc > 0.60:
        print("  → MARGINAL: representation has weak signal; add dynamic features")
    else:
        print("  → REPRESENTATION FAILS: hidden state cannot linearly separate; need View C or architecture change")


if __name__ == "__main__":
    main()
