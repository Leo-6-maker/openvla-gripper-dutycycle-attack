#!/usr/bin/env python3
"""evaluate_early_grasp_detector.py — Evaluate trained causal TCN detector with threshold sweep. v2.

Fixes (v2):
  - Sweep thresholds {0.05,0.1,0.15,0.2,0.3,0.5} × K={1,2,3}
  - Select best on val split, report on test.
  - Load full model config from checkpoint.
  - Compare against rule baseline.
  - OPEN naming: raw_gripper < 0.5 = OPEN (canonical semantics).
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
K_VALUES = [1, 2, 3]
GRASP_CLASS = 1  # grasp_formation


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_state_holdout")
    ap.add_argument("--rule-csv", default="tables/object_rule_based_trigger_eval.csv")
    ap.add_argument("--output-csv", default="tables/object_detector_predictions.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_DETECTOR_EVAL.md")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def find_trigger(probs, K=2, threshold=0.5):
    streak = 0
    for t in range(len(probs)):
        if probs[t, GRASP_CLASS] >= threshold:
            streak += 1
            if streak >= K:
                return t - K + 1
        else: streak = 0
    return None


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN, SequenceDataset, compute_detailed_metrics

    model = EarlyGraspTCN(
        input_dim=config.get("input_dim", 13),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 3),
        kernel_size=config.get("kernel_size", 3),
        num_classes=config.get("num_classes", 3),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: best_val_f1_grasp={config.get('best_val_f1_grasp','?')}")

    # Load data
    ds_kwargs = dict(npz_path=args.npz_path, split_csv=args.split_csv, split_col=args.split_col)
    val_ds = SequenceDataset(**ds_kwargs, split="val")
    test_ds = SequenceDataset(**ds_kwargs, split="test")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Load meta
    meta_csv = args.npz_path.replace(".npz", "_meta.csv")
    meta = {}
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f):
                meta[r["episode_id"]] = r

    # Load rule baseline
    rule_base = {}
    if os.path.exists(args.rule_csv):
        with open(args.rule_csv, newline="") as f:
            for r in csv.DictReader(f):
                rule_base[r["episode_id"]] = r

    # ── Threshold sweep on val ──
    def evaluate_split(loader, threshold, K):
        results = []
        for i, (Xb, yb, mb) in enumerate(loader):
            Xb = Xb.to(device); eid = loader.dataset.episode_ids[i]
            with torch.no_grad():
                probs = torch.softmax(model(Xb), dim=-1).squeeze(0).cpu().numpy()
            T = int(mb.sum()); probs = probs[:T]
            T_pred = find_trigger(probs, K=K, threshold=threshold)
            ep_m = meta.get(eid, {})
            tg_str = ep_m.get("T_gform", ""); tg = int(tg_str) if tg_str else None
            task = ep_m.get("task_name", "?")
            error = (T_pred - tg) if (T_pred is not None and tg is not None) else None
            results.append(dict(episode_id=eid, task_name=task, T_gform=tg, T_pred=T_pred,
                                error=error, abs_error=abs(error) if error is not None else None,
                                triggered=T_pred is not None))
        return results

    # Sweep val
    MIN_TRIGGER_RATE = 0.8
    best_config = None; best_mae = float("inf"); best_score = float("inf")
    sweep_results = []
    for th in THRESHOLDS:
        for k in K_VALUES:
            val_res = evaluate_split(val_loader, th, k)
            errs = [r["abs_error"] for r in val_res if r["abs_error"] is not None]
            triggered = sum(1 for r in val_res if r["triggered"])
            n_val = len(val_res)
            trig_rate = triggered / n_val if n_val > 0 else 0.0
            mae = np.mean(errs) if errs else float("inf")
            early = sum(1 for r in val_res if r["error"] is not None and r["error"] < -5)
            late = sum(1 for r in val_res if r["error"] is not None and r["error"] > 5)
            within5 = sum(1 for r in val_res if r["abs_error"] is not None and r["abs_error"] <= 5)
            within10 = sum(1 for r in val_res if r["abs_error"] is not None and r["abs_error"] <= 10)
            no_trig_rate = 1.0 - trig_rate
            score = mae + 100 * no_trig_rate if errs else float("inf")
            sweep_results.append(dict(threshold=th, K=k,
                val_mae=round(mae,2) if errs else None,
                val_triggered=triggered, val_n=n_val,
                val_trigger_rate=round(100*trig_rate,1),
                val_early_pct=round(100*early/max(triggered,1),1),
                val_late_pct=round(100*late/max(triggered,1),1),
                val_within5=within5, val_within10=within10,
                score=round(score,2) if errs else None))
            if trig_rate >= MIN_TRIGGER_RATE and errs and mae < best_mae:
                best_mae = mae; best_config = (th, k)

    print("\nThreshold sweep (val):")
    print(f"  {'th':>6s}  {'K':1s}  {'MAE':>8s}  {'triggered':>9s}  {'rate':>6s}")
    for sr in sweep_results:
        mae_str = f"{sr['val_mae']:.2f}" if sr['val_mae'] is not None else "N/A"
        print(f"  {sr['threshold']:6.2f}  {sr['K']:1d}  {mae_str:>8s}  {sr['val_triggered']:4d}/{sr['val_n']:<4d}  {sr['val_trigger_rate']:5.1f}%")

    # Evaluate test with best config
    if best_config is None:
        print(f"No config passed min_trigger_rate={MIN_TRIGGER_RATE:.0%} — using best score")
        valid_sweep = [s for s in sweep_results if s["score"] is not None]
        if valid_sweep:
            best = min(valid_sweep, key=lambda s: s["score"])
            best_config = (best["threshold"], best["K"])
            print(f"  Fallback: th={best_config[0]}, K={best_config[1]}, score={best['score']}")
        else:
            print("  No valid config at all — using threshold=0.15, K=2")
            best_config = (0.15, 2)

    best_th, best_k = best_config
    print(f"\nBest config: threshold={best_th}, K={best_k}, val_MAE={best_mae:.2f}")
    test_res = evaluate_split(test_loader, best_th, best_k)

    # Per-episode results
    results = []
    per_task_tcn = defaultdict(list); per_task_rule = defaultdict(list)
    for tr in test_res:
        eid = tr["episode_id"]
        rb = rule_base.get(eid, {})
        T_rule_str = rb.get("T_rule", ""); T_rule = int(T_rule_str) if T_rule_str else None
        tg = tr["T_gform"]
        rule_err = (T_rule - tg) if (T_rule is not None and tg is not None) else None
        r = dict(episode_id=eid, task_name=tr["task_name"], T_gform=tg if tg is not None else "",
                 T_pred=tr["T_pred"] if tr["T_pred"] is not None else "",
                 T_rule=T_rule if T_rule is not None else "",
                 tcn_error=tr["error"] if tr["error"] is not None else "",
                 tcn_abs_error=tr["abs_error"] if tr["abs_error"] is not None else "",
                 rule_error=rule_err if rule_err is not None else "",
                 rule_abs_error=abs(rule_err) if rule_err is not None else "")
        results.append(r)
        if tr["abs_error"] is not None: per_task_tcn[tr["task_name"]].append(tr["abs_error"])
        if rule_err is not None: per_task_rule[tr["task_name"]].append(abs(rule_err))

    tcn_valid = [r for r in results if r["tcn_abs_error"] != ""]
    rule_valid = [r for r in results if r["rule_abs_error"] != ""]
    tcn_mae = np.mean([r["tcn_abs_error"] for r in tcn_valid]) if tcn_valid else None
    rule_mae = np.mean([r["rule_abs_error"] for r in rule_valid]) if rule_valid else None
    tcn_med = np.median([r["tcn_abs_error"] for r in tcn_valid]) if tcn_valid else None
    rule_med = np.median([r["rule_abs_error"] for r in rule_valid]) if rule_valid else None
    tcn_trig = sum(1 for r in tcn_valid)

    # Write CSV
    csv_fields = ["episode_id","task_name","T_gform","T_pred","T_rule",
                  "tcn_error","tcn_abs_error","rule_error","rule_abs_error"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)

    # Save eval metrics JSON for proposal script
    eval_metrics = {
        "best_threshold": best_th, "best_K": best_k,
        "min_trigger_rate": MIN_TRIGGER_RATE,
        "best_val_MAE": best_mae if best_mae != float("inf") else None,
        "test_MAE_tcn": round(tcn_mae, 2) if tcn_mae is not None else None,
        "test_MAE_rule": round(rule_mae, 2) if rule_mae is not None else None,
        "tcn_triggers_test": tcn_trig,
    }
    eval_json_path = args.output_csv.replace(".csv", "_eval_metrics.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved eval metrics to {eval_json_path}")

    # Report
    imp = ""
    if tcn_mae is not None and rule_mae is not None:
        delta = rule_mae - tcn_mae
        imp = f"| Delta | {delta:+.2f} ({100*delta/rule_mae:+.1f}%) | -- |\n"

    report = f"""# Early-Grasp Detector Evaluation v3

**Checkpoint**: {args.checkpoint}
**Best config**: threshold={best_th}, K={best_k} (selected on val)
**Val MAE**: {best_mae:.2f}

## Test Set Results

| Metric | TCN Detector | Rule Baseline |
|--------|-------------|---------------|
| n triggered | {tcn_trig} | {len(rule_valid)} |
| MAE | {tcn_mae:.2f} | {rule_mae:.2f} |
| MedAE | {tcn_med:.2f} | {rule_med:.2f} |
{imp}
## Per-Task (TCN)

| Task | n | MAE |
|------|---|-----|
""" + "\n".join(f"| {t} | {len(e)} | {np.mean(e):.2f} |" for t, e in sorted(per_task_tcn.items())) + f"""

## Threshold Sweep (Val)

| Threshold | K | MAE | Trigger Rate |
|-----------|----|-----|-------------|
""" + "\n".join(f"| {sr['threshold']} | {sr['K']} | {sr['val_mae']} | {sr['val_trigger_rate']}% |" for sr in sweep_results) + f"""

## Verdict

"""

    if tcn_mae is not None and rule_mae is not None and tcn_mae < rule_mae:
        report += f"TCN improves over rule baseline by {rule_mae-tcn_mae:.1f} steps. Learned detector justified."
    elif tcn_trig == 0:
        report += "TCN still cannot trigger. Detector training needs further improvement (auxiliary head, more data, etc.)."
    else:
        report += "TCN does not beat rule baseline. Rule-based trigger should be primary approach."

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report: {args.output_report}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
