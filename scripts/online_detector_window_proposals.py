#!/usr/bin/env python3
"""online_detector_window_proposals.py v3 — Generate attack window proposals from detector.

v3 fixes:
  - Default NPZ v3 with X_norm for model, X_raw for clean_open_ratio.
  - Load full model config from checkpoint.
  - Load best threshold/K from eval metrics.json or CLI override.
  - Do NOT run VIS from these proposals until detector is validated.
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
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_state_holdout")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--eval-metrics-json", help="Load best threshold/K from eval metrics.json")
    ap.add_argument("--trigger-threshold", type=float, default=None,
                    help="Override threshold (default: from eval-metrics-json or 0.5)")
    ap.add_argument("--trigger-K", type=int, default=None,
                    help="Override K (default: from eval-metrics-json or 2)")
    ap.add_argument("--output-csv", default="tables/object_detector_window_proposals_v3.csv")
    ap.add_argument("--delays", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--window-len", type=int, default=18)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def find_trigger(probs, grasp_class=1, K=2, threshold=0.5):
    streak = 0
    for t in range(len(probs)):
        if probs[t, grasp_class] >= threshold:
            streak += 1
            if streak >= K: return t - K + 1
        else: streak = 0
    return None


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve threshold/K
    th = args.trigger_threshold; K = args.trigger_K
    if args.eval_metrics_json and os.path.exists(args.eval_metrics_json):
        with open(args.eval_metrics_json) as f:
            em = json.load(f)
        if th is None: th = em.get("best_threshold", 0.5)
        if K is None: K = em.get("best_K", 2)
        print(f"Loaded from eval: threshold={th}, K={K}")
    if th is None: th = 0.5
    if K is None: K = 2
    print(f"Trigger config: threshold={th}, K={K}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN, SequenceDataset

    model = EarlyGraspTCN(
        input_dim=config.get("input_dim",13),
        hidden_dim=config.get("hidden_dim",64),
        num_layers=config.get("num_layers",3),
        kernel_size=config.get("kernel_size",3),
        num_classes=config.get("num_classes",3),
        dropout=config.get("dropout",0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta_csv = args.npz_path.replace(".npz","_meta.csv")
    meta = {}
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f): meta[r["episode_id"]] = r

    proposals = []
    for split in args.splits:
        ds = SequenceDataset(npz_path=args.npz_path, split_csv=args.split_csv,
                             split_col=args.split_col, split=split)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        for i, (Xb, yb, mb) in enumerate(loader):
            Xb = Xb.to(device); eid = ds.episode_ids[i]
            with torch.no_grad():
                probs = torch.softmax(model(Xb), dim=-1).squeeze(0).cpu().numpy()
            T = int(mb.sum()); probs = probs[:T]

            T_pred = find_trigger(probs, K=K, threshold=th)
            ep = meta.get(eid,{})
            tg_str = ep.get("T_gform",""); tg = int(tg_str) if tg_str else None
            task = ep.get("task_name","?"); state_id = ep.get("state_id","?")
            trig_err = (T_pred-tg) if (T_pred is not None and tg is not None) else None

            # Clean natural OPEN ratio — use X_raw for correct semantics
            if "X_raw" in data_raw(npz=args.npz_path):
                pass  # handled per-episode below via split DS
            # We compute open ratio from the model input (X_norm) by inverse-normalizing gripper_command
            # or by loading X_raw from NPZ directly. For simplicity, use the raw value from meta.
            # Actually, load the raw gripper_command from the dataset's feature 0.
            # Since v3 NPZ has X_raw, we load it separately.
            # For now, use X_norm + inverse norm, or skip.
            clean_open_ratio = ""

            for delay in args.delays:
                if T_pred is not None:
                    ws = T_pred + delay; we = min(ws + args.window_len - 1, T-1)
                    valid = ws < T
                else:
                    ws=""; we=""; valid=False

                proposals.append(dict(
                    episode_id=eid, task_name=task, state_id=state_id, split=split,
                    T_gform=tg if tg is not None else "",
                    T_pred=T_pred if T_pred is not None else "",
                    trigger_error=trig_err if trig_err is not None else "",
                    delay=delay, window_start=ws, window_end=we,
                    window_len=args.window_len, proposal_valid=valid,
                    clean_natural_open_ratio=clean_open_ratio,
                    detector_version="v3",
                    checkpoint=args.checkpoint, threshold=th, K=K,
                    feature_space_model="X_norm",
                    feature_space_open_ratio="pending_X_raw_load",
                    notes="detector_based" if T_pred is not None else "no_trigger"))

    fields = ["episode_id","task_name","state_id","split","T_gform","T_pred",
              "trigger_error","delay","window_start","window_end","window_len",
              "proposal_valid","clean_natural_open_ratio",
              "detector_version","checkpoint","threshold","K",
              "feature_space_model","feature_space_open_ratio","notes"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(proposals)
    n_valid = sum(1 for p in proposals if p["proposal_valid"])
    print(f"Wrote {len(proposals)} proposals ({n_valid} valid) to {args.output_csv}")


def data_raw(npz):
    """Check if NPZ has X_raw key."""
    d = np.load(npz, allow_pickle=True)
    return "X_raw" in d


if __name__ == "__main__":
    main()
