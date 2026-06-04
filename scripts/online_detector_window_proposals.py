#!/usr/bin/env python3
"""online_detector_window_proposals.py — Generate attack window proposals from detector predictions.

Applies positive delay Delta to T_pred to produce attack window [T_pred+Delta, T_pred+Delta+17].
"""

from __future__ import annotations
import argparse, csv, os, sys
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
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--output-csv", default="tables/object_detector_window_proposals.csv")
    ap.add_argument("--delays", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--window-len", type=int, default=18)
    ap.add_argument("--trigger-K", type=int, default=2)
    ap.add_argument("--trigger-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def find_trigger_from_probs(probs, grasp_class=1, K=2, threshold=0.5):
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN, SequenceDataset

    model = EarlyGraspTCN(
        input_dim=config.get("input_dim", 13),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    proposals = []
    meta_csv = args.npz_path.replace(".npz", "_meta.csv")
    meta = {}
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f):
                meta[r["episode_id"]] = r

    for split in args.splits:
        ds = SequenceDataset(
            npz_path=args.npz_path, split_csv=args.split_csv,
            split_col=args.split_col, split=split,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        for i, (X_batch, y_batch, mask_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            eid = ds.episode_ids[i]

            with torch.no_grad():
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            T = int(mask_batch.sum())
            probs = probs[:T]

            T_pred = find_trigger_from_probs(
                probs, K=args.trigger_K, threshold=args.trigger_threshold,
            )

            ep_meta = meta.get(eid, {})
            tg_str = ep_meta.get("T_gform", "")
            T_gform = int(tg_str) if tg_str else None
            task = ep_meta.get("task_name", "unknown")
            state_id = ep_meta.get("state_id", "?")

            trigger_error = (T_pred - T_gform) if (T_pred is not None and T_gform is not None) else None

            for delay in args.delays:
                if T_pred is not None:
                    ws = T_pred + delay
                    we = min(ws + args.window_len - 1, T - 1)
                    valid = ws < T
                else:
                    ws = ""
                    we = ""
                    valid = False

                # Check clean natural open ratio in proposed window
                clean_open_ratio = ""
                if T_pred is not None and ws < T:
                    gc_idx = 0  # gripper_command is feature 0
                    w_gc = X_batch[0, ws:we+1, gc_idx].cpu().numpy()
                    n_win = len(w_gc)
                    n_open = int((w_gc < 0.5).sum())
                    clean_open_ratio = round(n_open / max(n_win, 1), 4)

                proposals.append({
                    "episode_id": eid,
                    "task_name": task,
                    "state_id": state_id,
                    "split": split,
                    "T_gform": T_gform if T_gform is not None else "",
                    "T_pred": T_pred if T_pred is not None else "",
                    "trigger_error": trigger_error if trigger_error is not None else "",
                    "delay": delay,
                    "window_start": ws,
                    "window_end": we,
                    "window_len": args.window_len,
                    "proposal_valid": valid,
                    "clean_natural_open_ratio": clean_open_ratio,
                    "notes": "detector_based" if T_pred is not None else "no_trigger",
                })

    csv_fields = ["episode_id", "task_name", "state_id", "split", "T_gform", "T_pred",
                  "trigger_error", "delay", "window_start", "window_end", "window_len",
                  "proposal_valid", "clean_natural_open_ratio", "notes"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(proposals)

    n_valid = sum(1 for p in proposals if p["proposal_valid"])
    print(f"Wrote {len(proposals)} proposals ({n_valid} valid) to {args.output_csv}")
    for delay in args.delays:
        n_d = sum(1 for p in proposals if p["delay"] == delay and p["proposal_valid"])
        print(f"  delay={delay}: {n_d} valid proposals")


if __name__ == "__main__":
    main()
