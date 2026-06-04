#!/usr/bin/env python3
"""online_detector_window_proposals.py v4 — Generate attack window proposals from detector.

Fixes from v3:
  - Load full NPZ once, access X_raw and X_norm from matching original indices.
  - Compute clean_natural_open_ratio from X_raw with raw_gripper<0.5=OPEN.
  - Load best threshold/K from eval metrics.json.
  - --dry-run mode.
  - Do NOT run VIS from these proposals without gate approval.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
import torch

GRIPPER_IDX = 0  # gripper_command is feature 0
GRASP_CLASS = 1  # grasp_formation


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_state_holdout")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--eval-metrics-json", help="Load best threshold/K from eval metrics.json")
    ap.add_argument("--trigger-threshold", type=float, default=None)
    ap.add_argument("--trigger-K", type=int, default=None)
    ap.add_argument("--output-csv", default="tables/object_detector_window_proposals_v4.csv")
    ap.add_argument("--delays", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--window-len", type=int, default=18)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def find_trigger(probs, K=2, threshold=0.5):
    streak = 0
    for t in range(len(probs)):
        if probs[t, GRASP_CLASS] >= threshold:
            streak += 1
            if streak >= K: return t - K + 1
        else: streak = 0
    return None


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Resolve threshold/K ──
    th = args.trigger_threshold; K = args.trigger_K
    eval_path = args.eval_metrics_json
    if eval_path and os.path.exists(eval_path):
        with open(eval_path) as f: em = json.load(f)
        if th is None: th = em.get("best_threshold")
        if K is None: K = em.get("best_K")
        print(f"Loaded from eval: threshold={th}, K={K}")
    if th is None: th = 0.5
    if K is None: K = 2
    print(f"Trigger config: threshold={th}, K={K}")

    if args.dry_run:
        print("DRY RUN: would load checkpoint and generate proposals.")
        return

    # ── Load NPZ once ──
    print(f"Loading NPZ: {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)
    has_raw = "X_raw" in data
    X_model = data["X_norm"] if "X_norm" in data else data["X"]
    X_raw_all = data["X_raw"] if has_raw else None
    y_all = data["y"]; mask_all = data["mask"]
    ep_ids_all = list(data.get("episode_ids", [f"ep_{i}" for i in range(len(X_model))]))
    print(f"  {len(ep_ids_all)} episodes, X_raw={'present' if has_raw else 'MISSING'}")

    # ── Load split ──
    split_map = {}
    if os.path.exists(args.split_csv):
        with open(args.split_csv, newline="") as f:
            split_map = {r["episode_id"]: r[args.split_col] for r in csv.DictReader(f)}

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN

    model = EarlyGraspTCN(
        input_dim=config.get("input_dim",13), hidden_dim=config.get("hidden_dim",64),
        num_layers=config.get("num_layers",3), kernel_size=config.get("kernel_size",3),
        num_classes=config.get("num_classes",3), dropout=config.get("dropout",0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Meta ──
    meta_csv = args.npz_path.replace(".npz","_meta.csv")
    meta = {}
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f): meta[r["episode_id"]] = r

    # ── Generate proposals per split ──
    proposals = []
    for split in args.splits:
        for orig_idx, eid in enumerate(ep_ids_all):
            eid_str = str(eid)
            if split_map.get(eid_str, "train") != split:
                continue

            X_norm_ep = X_model[orig_idx]
            X_raw_ep = X_raw_all[orig_idx] if has_raw else None
            m = mask_all[orig_idx]
            T = int(m.sum())

            # Model inference
            Xt = torch.from_numpy(X_norm_ep[:T]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(Xt), dim=-1).squeeze(0).cpu().numpy()

            T_pred = find_trigger(probs, K=K, threshold=th)
            ep = meta.get(eid_str, {})
            tg_str = ep.get("T_gform",""); tg = int(tg_str) if tg_str else None
            task = ep.get("task_name","?"); state_id = ep.get("state_id","?")
            trig_err = (T_pred - tg) if (T_pred is not None and tg is not None) else None

            for delay in args.delays:
                if T_pred is not None and T_pred < T:
                    ws = T_pred + delay
                    we = min(ws + args.window_len - 1, T - 1)
                    valid = ws <= we < T
                else:
                    ws = ""; we = ""; valid = False

                # Compute clean_natural_open_ratio from X_raw
                clean_open_ratio = ""; clean_open_count = ""; actual_win_len = ""
                raw_open_semantics = ""
                if has_raw and valid and isinstance(ws, int):
                    win_end = int(we) + 1
                    raw_gc = X_raw_ep[int(ws):win_end, GRIPPER_IDX]
                    actual_win_len = len(raw_gc)
                    clean_open_count = int((raw_gc < 0.5).sum())
                    clean_open_ratio = round(clean_open_count / max(actual_win_len, 1), 4)
                    raw_open_semantics = "raw_gripper<0.5=OPEN"
                elif not has_raw:
                    raw_open_semantics = "X_raw_missing_in_NPZ"

                proposals.append(dict(
                    episode_id=eid_str, task_name=task, state_id=state_id, split=split,
                    T_gform=tg if tg is not None else "",
                    T_pred=T_pred if T_pred is not None else "",
                    trigger_error=trig_err if trig_err is not None else "",
                    delay=delay, window_start=ws, window_end=we,
                    window_len=args.window_len, proposal_valid=valid,
                    clean_natural_open_ratio=clean_open_ratio,
                    clean_open_count=clean_open_count,
                    actual_window_len=actual_win_len,
                    raw_open_semantics=raw_open_semantics,
                    detector_version="v4",
                    checkpoint=os.path.basename(args.checkpoint),
                    threshold=th, K=K,
                    eval_metrics_json=eval_path or "",
                    feature_space_model="X_norm",
                    feature_space_open_ratio="X_raw" if has_raw else "missing",
                    notes="detector_based" if T_pred is not None else "no_trigger"))

    # ── Write ──
    fields = ["episode_id","task_name","state_id","split","T_gform","T_pred",
              "trigger_error","delay","window_start","window_end","window_len",
              "proposal_valid","clean_natural_open_ratio","clean_open_count",
              "actual_window_len","raw_open_semantics",
              "detector_version","checkpoint","threshold","K","eval_metrics_json",
              "feature_space_model","feature_space_open_ratio","notes"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(proposals)

    n_valid = sum(1 for p in proposals if p["proposal_valid"])
    n_no_trigger = sum(1 for p in proposals if "no_trigger" in str(p.get("notes","")))
    n_with_open_ratio = sum(1 for p in proposals if p["clean_natural_open_ratio"] != "")
    print(f"Wrote {len(proposals)} proposals: {n_valid} valid, "
          f"{n_no_trigger} no-trigger, {n_with_open_ratio} with clean_open_ratio")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
