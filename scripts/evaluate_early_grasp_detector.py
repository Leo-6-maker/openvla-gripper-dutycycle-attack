#!/usr/bin/env python3
"""evaluate_early_grasp_detector.py — Evaluate trained causal TCN detector.

Computes per-step classification metrics and trigger-level T_pred vs T_gform errors.
Compares against rule-based baseline if available.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v1.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_task_holdout")
    ap.add_argument("--rule-csv", default="tables/object_rule_based_trigger_eval.csv")
    ap.add_argument("--output-csv", default="tables/object_detector_predictions.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_DETECTOR_EVAL.md")
    ap.add_argument("--trigger-K", type=int, default=2)
    ap.add_argument("--trigger-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def find_trigger_from_probs(probs, grasp_class=1, K=2, threshold=0.5):
    """T_pred = first step where P(grasp_formation) > threshold for K consecutive steps."""
    streak = 0
    for t in range(len(probs)):
        if probs[t, grasp_class] >= threshold:
            streak += 1
            if streak >= K:
                return t - K + 1
        else:
            streak = 0
    return None


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    input_dim = config.get("input_dim", 13)
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 3)

    # Import model class
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN, SequenceDataset

    model = EarlyGraspTCN(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load data
    ds_kwargs = dict(npz_path=args.npz_path, split_csv=args.split_csv, split_col=args.split_col)
    test_ds = SequenceDataset(**ds_kwargs, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test episodes: {len(test_ds)}")

    # Load rule baseline
    rule_baseline = {}
    if os.path.exists(args.rule_csv):
        with open(args.rule_csv, newline="") as f:
            for r in csv.DictReader(f):
                rule_baseline[r["episode_id"]] = r
        print(f"Loaded {len(rule_baseline)} rule baseline entries")

    # Evaluate
    results = []
    per_task_tcn = defaultdict(list)
    per_task_rule = defaultdict(list)

    for i, (X_batch, y_batch, mask_batch) in enumerate(test_loader):
        X_batch = X_batch.to(device)
        eid = test_ds.episode_ids[i]

        with torch.no_grad():
            logits = model(X_batch)  # [1, T, C]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        T = int(mask_batch.sum())
        probs = probs[:T]

        # Find T_pred
        T_pred = find_trigger_from_probs(probs, K=args.trigger_K, threshold=args.trigger_threshold)

        # T_gform from meta (need to load separately)
        meta_csv = args.npz_path.replace(".npz", "_meta.csv")
        tg = None
        task = "unknown"
        if os.path.exists(meta_csv):
            with open(meta_csv, newline="") as f:
                for r in csv.DictReader(f):
                    if r["episode_id"] == eid:
                        tg = int(r["T_gform"]) if r.get("T_gform") else None
                        task = r.get("task_name", "unknown")
                        break

        # Errors
        tcn_error = None
        tcn_abs = None
        rule_error = None
        rule_abs = None

        if T_pred is not None and tg is not None:
            tcn_error = T_pred - tg
            tcn_abs = abs(tcn_error)
            per_task_tcn[task].append(tcn_abs)

        rb = rule_baseline.get(eid, {})
        T_rule_str = rb.get("T_rule", "")
        if T_rule_str:
            T_rule = int(T_rule_str)
            if tg is not None:
                rule_error = T_rule - tg
                rule_abs = abs(rule_error)
                per_task_rule[task].append(rule_abs)

        results.append({
            "episode_id": eid,
            "task_name": task,
            "T_gform": tg if tg is not None else "",
            "T_pred": T_pred if T_pred is not None else "",
            "T_rule": T_rule_str,
            "tcn_error": tcn_error if tcn_error is not None else "",
            "tcn_abs_error": tcn_abs if tcn_abs is not None else "",
            "rule_error": rule_error if rule_error is not None else "",
            "rule_abs_error": rule_abs if rule_abs is not None else "",
        })

    # Write CSV
    csv_fields = ["episode_id", "task_name", "T_gform", "T_pred", "T_rule",
                  "tcn_error", "tcn_abs_error", "rule_error", "rule_abs_error"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # Aggregate
    valid_tcn = [r for r in results if r["tcn_abs_error"] != ""]
    valid_rule = [r for r in results if r["rule_abs_error"] != ""]

    tcn_mae = np.mean([r["tcn_abs_error"] for r in valid_tcn]) if valid_tcn else None
    rule_mae = np.mean([r["rule_abs_error"] for r in valid_rule]) if valid_rule else None
    tcn_med = np.median([r["tcn_abs_error"] for r in valid_tcn]) if valid_tcn else None
    rule_med = np.median([r["rule_abs_error"] for r in valid_rule]) if valid_rule else None

    print(f"\nTest results:")
    print(f"  TCN:  n={len(valid_tcn)} MAE={tcn_mae:.2f} MedAE={tcn_med:.2f}" if tcn_mae else "  TCN: no valid")
    print(f"  Rule: n={len(valid_rule)} MAE={rule_mae:.2f} MedAE={rule_med:.2f}" if rule_mae else "  Rule: no valid")

    if tcn_mae is not None and rule_mae is not None:
        delta = rule_mae - tcn_mae
        print(f"  Improvement: {delta:+.2f} steps ({100*delta/rule_mae:+.1f}%)")

    # Report
    report = f"""# Early-Grasp Detector Evaluation

**Checkpoint**: {args.checkpoint}
**Trigger config**: K={args.trigger_K}, threshold={args.trigger_threshold}

## Test Set Results

| Metric | TCN Detector | Rule Baseline |
|--------|-------------|---------------|
| n | {len(valid_tcn)} | {len(valid_rule)} |
| MAE | {tcn_mae:.2f} | {rule_mae:.2f} |
| MedAE | {tcn_med:.2f} | {rule_med:.2f} |
"""
    if tcn_mae is not None and rule_mae is not None:
        delta = rule_mae - tcn_mae
        report += f"| Delta | {delta:+.2f} ({100*delta/rule_mae:+.1f}%) | -- |\n"

    report += f"""

## Per-Task (TCN)

| Task | n | MAE |
|------|---|-----|
""" + "\n".join(
        f"| {t} | {len(e)} | {np.mean(e):.2f} |"
        for t, e in sorted(per_task_tcn.items())
    ) + f"""

## Verdict

"""
    if tcn_mae is not None and rule_mae is not None and tcn_mae < rule_mae:
        report += (
            f"TCN detector improves over rule-based baseline by {rule_mae - tcn_mae:.1f} steps. "
            "Learned detector is justified."
        )
    else:
        report += (
            "TCN detector does not beat rule-based baseline. "
            "Rule-based trigger should be the primary approach."
        )

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
