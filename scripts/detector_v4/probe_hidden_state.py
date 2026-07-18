"""Hidden-state separability probe: train logistic regression on frozen GRU states.

Tests whether the GRU hidden representation can linearly separate
VALID_RETENTION close steps from hard-negative close steps.

Interpretation:
- probe AUC high, original head poor → optimization/loss/head problem
- probe AUC low → representation/features problem (need View C or architecture change)
"""
import json, sys, argparse, pickle
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detector_v4"))
from train_v4_detector import (
    load_v4_episode, derive_dynamic_features, ALL_VIEWS,
    CandidateAGRU, CandidateBGRU, CandidateCGRU,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


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

    # Build model and register hook to capture hidden states
    if candidate == "B":
        model = CandidateBGRU()
    elif candidate == "A":
        model = CandidateAGRU()
    else:
        model = CandidateCGRU()
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device)
    model.eval()

    hidden_states = []
    def hook(module, input, output):
        # output from GRU is (output, h_n); capture output
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach().cpu())
        else:
            hidden_states.append(output.detach().cpu())

    # Register hook on GRU
    if hasattr(model, 'backbone'):
        model.backbone.gru.register_forward_hook(hook)
    elif hasattr(model, 'gru'):
        model.gru.register_forward_hook(hook)

    # Collect hidden states and labels for validation episodes
    val_states = list(range(args.fold * 5, (args.fold + 1) * 5))
    X_list = []
    y_list = []  # 1 = quality_valid, 0 = veto_invalid (hard negative)
    episode_ids = []

    with torch.no_grad():
        for suite in SUITES:
            for task in range(10):
                for state in val_states:
                    # Load V2.1 labels
                    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
                    v21_path = args.v21_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_v21_labels.jsonl"
                    if not v21_path.exists():
                        continue

                    v21_labels = [json.loads(l) for l in open(v21_path)]

                    # Load features
                    ep = load_v4_episode(args.s1_root, args.v21_root,
                                        suite, task, state, view)
                    if ep is None:
                        continue

                    x = ep.features.unsqueeze(0)
                    x = (x - norm_mean.to(x.device)) / norm_std.to(x.device)
                    padding = torch.ones(1, ep.n_steps, dtype=torch.bool).to(args.device)

                    hidden_states.clear()
                    _ = model(x.to(args.device), padding)

                    if hidden_states:
                        hs = hidden_states[0].squeeze(0)  # [T, 128]
                        for t in range(min(ep.n_steps, len(hs), len(v21_labels))):
                            label = v21_labels[t]
                            if not label["known_mask"] or not label["candidate_close"]:
                                continue  # only probe close-window known steps
                            if label["quality_valid"]:
                                X_list.append(hs[t].numpy())
                                y_list.append(1)
                                episode_ids.append(cid)
                            elif label["veto_invalid"]:
                                X_list.append(hs[t].numpy())
                                y_list.append(0)
                                episode_ids.append(cid)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    print(f"Probe data: {len(X)} close-window known steps")
    print(f"  quality_valid (class 1): {int(y.sum())}")
    print(f"  veto_invalid (class 0): {len(y) - int(y.sum())}")
    print(f"  Unique episodes: {len(set(episode_ids))}")

    if len(X) < 10 or y.sum() == 0 or y.sum() == len(y):
        print("ERROR: insufficient data for probe (need both classes)")
        return

    # Simple PyTorch logistic regression probe
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    probe = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    for epoch in range(500):
        opt.zero_grad()
        logits = probe(X).squeeze()
        loss = bce(logits, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_score = torch.sigmoid(probe(X).squeeze()).numpy()
    y_np = y.numpy()

    # Simple AUC using trapezoidal rule
    order = y_score.argsort()[::-1]
    y_sorted = y_np[order]
    y_score_sorted = y_score[order]

    tp = 0; fp = 0; tpr = [0.0]; fpr = [0.0]
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    auc = 0.0
    for i in range(len(fpr)-1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i] + tpr[i+1]) / 2.0

    # AUPRC
    precisions = []; recalls = []
    tp = 0; fp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1: tp += 1
        else: fp += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / n_pos
        precisions.append(prec)
        recalls.append(rec)

    auprc = 0.0
    for i in range(len(recalls)-1):
        auprc += (recalls[i+1] - recalls[i]) * max(precisions[i], precisions[i+1])
    auprc = abs(auprc)

    print(f"\nProbe results (linear logistic regression on frozen GRU states):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  AUPRC:   {auprc:.4f}")
    print(f"  Final BCE loss: {loss.item():.4f}")

    # Compare with original head scores
    # Get head probabilities from the model
    print(f"\nHead comparison (from original model):")
    # Reload and get head outputs
    model2 = CandidateBGRU() if candidate == "B" else CandidateAGRU()
    model2.load_state_dict(ckpt["model_state"])
    model2 = model2.to(args.device)
    model2.eval()

    head_scores = []
    head_labels = []
    with torch.no_grad():
        for suite in SUITES:
            for task in range(10):
                for state in val_states:
                    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
                    v21_path = args.v21_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_v21_labels.jsonl"
                    if not v21_path.exists():
                        continue
                    v21_labels = [json.loads(l) for l in open(v21_path)]
                    ep = load_v4_episode(args.s1_root, args.v21_root, suite, task, state, view)
                    if ep is None:
                        continue
                    x = ep.features.unsqueeze(0)
                    x = (x - norm_mean.to(x.device)) / norm_std.to(x.device)
                    padding = torch.ones(1, ep.n_steps, dtype=torch.bool).to(args.device)
                    logits, _ = model2(x.to(args.device), padding)
                    crit_head = "criticality" if "criticality" in logits else "valid_retention"
                    crit_probs = torch.sigmoid(logits[crit_head].squeeze(0)).cpu()
                    for t in range(min(ep.n_steps, len(v21_labels))):
                        label = v21_labels[t]
                        if not label["known_mask"] or not label["candidate_close"]:
                            continue
                        if label["quality_valid"]:
                            head_scores.append(float(crit_probs[t]))
                            head_labels.append(1)
                        elif label["veto_invalid"]:
                            head_scores.append(float(crit_probs[t]))
                            head_labels.append(0)

    if head_scores:
        # Compute AUC for original head
        hs_arr = torch.tensor(head_scores)
        hl_arr = torch.tensor(head_labels, dtype=torch.float32)
        n_pos_head = int(hl_arr.sum())
        n_neg_head = len(hl_arr) - n_pos_head
        order_h = hs_arr.argsort(descending=True)
        hl_sorted = hl_arr[order_h].numpy()
        tp_h = 0; fp_h = 0; tpr_h = [0.0]; fpr_h = [0.0]
        for i in range(len(hl_sorted)):
            if hl_sorted[i] == 1: tp_h += 1
            else: fp_h += 1
            tpr_h.append(tp_h / n_pos_head)
            fpr_h.append(fp_h / n_neg_head)
        head_auc = 0.0
        for i in range(len(fpr_h)-1):
            head_auc += (fpr_h[i+1] - fpr_h[i]) * (tpr_h[i] + tpr_h[i+1]) / 2.0

        print(f"  Original head ROC-AUC: {head_auc:.4f}")
        print(f"\n  DELTA (probe - head) AUC: {auc - head_auc:.4f}")
        if auc - head_auc > 0.05:
            print("  → Head is underfitting relative to representation capacity")
        elif auc > 0.80:
            print("  → Representation + linear head can separate; label/loss issue dominates")
        else:
            print("  → Representation cannot linearly separate; need better features")


if __name__ == "__main__":
    main()
