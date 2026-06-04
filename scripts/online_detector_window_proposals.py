#!/usr/bin/env python3
"""online_detector_window_proposals.py v5 — Hardened attack window proposals from detector.

v5 hardening:
  - proposal_valid requires full 18-step window (no clipped-short).
  - proposal_eligible gate: clean_natural_open_ratio <= max_clean_open_ratio.
  - Consistency check: checkpoint/NPZ/split must match eval metrics.
  - SystemExit if X_raw missing (required for clean_open_ratio).
  - --allow-mismatch to override consistency checks.
  - Do NOT run VIS from proposals without gate approval.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
import torch

GRIPPER_IDX = 0; GRASP_CLASS = 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--split-csv", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--split-col", default="split_state_holdout")
    ap.add_argument("--splits", nargs="+", default=["val","test"])
    ap.add_argument("--eval-metrics-json", required=True)
    ap.add_argument("--trigger-threshold", type=float, default=None)
    ap.add_argument("--trigger-K", type=int, default=None)
    ap.add_argument("--max-clean-open-ratio", type=float, default=0.1,
                    help="Max clean natural OPEN ratio for eligibility")
    ap.add_argument("--output-csv", default="tables/object_detector_window_proposals_v5.csv")
    ap.add_argument("--delays", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--window-len", type=int, default=18)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-mismatch", action="store_true")
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

    # ── Load eval metrics ──
    if not os.path.exists(args.eval_metrics_json):
        print(f"ERROR: eval_metrics_json not found: {args.eval_metrics_json}")
        sys.exit(1)
    with open(args.eval_metrics_json) as f:
        em = json.load(f)

    # Consistency checks
    mismatches = []
    em_npz = em.get("npz_path",""); em_ckpt = em.get("checkpoint_basename","")
    em_split = em.get("split_col",""); em_label = em.get("label_schema_path","")

    if em_npz and os.path.basename(args.npz_path) != os.path.basename(em_npz):
        mismatches.append(f"NPZ: args={os.path.basename(args.npz_path)} vs eval={os.path.basename(em_npz)}")
    if em_ckpt and os.path.basename(args.checkpoint) != em_ckpt:
        mismatches.append(f"checkpoint: args={os.path.basename(args.checkpoint)} vs eval={em_ckpt}")
    if em_split and args.split_col != em_split:
        mismatches.append(f"split_col: args={args.split_col} vs eval={em_split}")
    if mismatches:
        msg = "Consistency mismatch:\n  " + "\n  ".join(mismatches)
        if args.allow_mismatch:
            print(f"WARNING: {msg}")
        else:
            print(f"ERROR: {msg}\n  Use --allow-mismatch to override.")
            sys.exit(1)

    # Resolve threshold/K
    th = args.trigger_threshold or em.get("best_threshold", 0.5)
    K = args.trigger_K or em.get("best_K", 2)
    print(f"Trigger: threshold={th}, K={K} (min_trigger_rate={em.get('min_trigger_rate','?')})")

    if args.dry_run:
        print("DRY RUN complete."); return

    # ── Load NPZ ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading NPZ: {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)
    if "X_raw" not in data:
        print("ERROR: X_raw not found in NPZ — required for clean_open_ratio with raw gripper semantics.")
        sys.exit(1)
    X_norm_all = data["X_norm"] if "X_norm" in data else data["X"]
    X_raw_all = data["X_raw"]
    mask_all = data["mask"]
    ep_ids_all = list(data.get("episode_ids", [f"ep_{i}" for i in range(len(X_norm_all))]))
    print(f"  {len(ep_ids_all)} episodes, X_raw present, X_norm present")

    # ── Split ──
    split_map = {}
    if os.path.exists(args.split_csv):
        with open(args.split_csv, newline="") as f:
            split_map = {r["episode_id"]: r[args.split_col] for r in csv.DictReader(f)}

    # ── Model ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_early_grasp_detector import EarlyGraspTCN
    model = EarlyGraspTCN(
        input_dim=config.get("input_dim",13), hidden_dim=config.get("hidden_dim",64),
        num_layers=config.get("num_layers",3), kernel_size=config.get("kernel_size",3),
        num_classes=config.get("num_classes",3), dropout=config.get("dropout",0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    # ── Meta ──
    meta_csv = args.npz_path.replace(".npz","_meta.csv")
    meta = {}
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f): meta[r["episode_id"]] = r

    # ── Generate ──
    proposals = []
    for split in args.splits:
        for orig_idx, eid in enumerate(ep_ids_all):
            eid_str = str(eid)
            if split_map.get(eid_str, "train") != split: continue

            Xn = X_norm_all[orig_idx]; Xr = X_raw_all[orig_idx]
            m = mask_all[orig_idx]; T = int(m.sum())

            Xt = torch.from_numpy(Xn[:T]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(Xt), dim=-1).squeeze(0).cpu().numpy()

            T_pred = find_trigger(probs, K=K, threshold=th)
            ep = meta.get(eid_str,{})
            tg_str = ep.get("T_gform",""); tg = int(tg_str) if tg_str else None
            task = ep.get("task_name","?"); state_id = ep.get("state_id","?")
            trig_err = (T_pred - tg) if (T_pred is not None and tg is not None) else None

            for delay in args.delays:
                ws = ""; we = ""; actual_len = 0
                structural_valid = False; inv_reason = ""
                clean_open_ratio = ""; clean_open_count = 0
                strict_eligible = False; relaxed_eligible = False
                strict_reason = ""; relaxed_reason = ""
                online_feasible = False; online_reason = ""

                if T_pred is None:
                    inv_reason = "no_trigger"
                elif T_pred >= T:
                    inv_reason = "trigger_out_of_bounds"
                else:
                    ws = T_pred + delay; we = ws + args.window_len - 1
                    if ws < 0:
                        actual_len = args.window_len - abs(ws)
                        inv_reason = f"window_before_start_len_{actual_len}"
                    elif we >= T:
                        actual_len = T - ws
                        inv_reason = f"clipped_short_len_{actual_len}"
                    else:
                        actual_len = args.window_len
                        structural_valid = True

                # Compute clean_open_ratio from X_raw
                STRICT_THRESH = 0.1
                if structural_valid and isinstance(ws, int) and ws >= 0 and we < T:
                    raw_gc = Xr[int(ws):int(we)+1, GRIPPER_IDX]
                    clean_open_count = int((raw_gc < 0.5).sum())
                    clean_open_ratio = round(clean_open_count / args.window_len, 4)
                    if clean_open_ratio <= STRICT_THRESH:
                        strict_eligible = True
                    else:
                        strict_reason = f"natural_open_confound_{clean_open_ratio}"
                    if clean_open_ratio <= args.max_clean_open_ratio:
                        relaxed_eligible = True
                    else:
                        relaxed_reason = f"natural_open_confound_{clean_open_ratio}"
                elif structural_valid and isinstance(ws, int) and ws < 0:
                    # Window starts before episode start — check partial
                    start = max(0, int(ws)); end_val = int(we)
                    if start <= end_val:
                        raw_gc = Xr[start:end_val+1, GRIPPER_IDX]
                        clean_open_count = int((raw_gc < 0.5).sum())
                        clean_open_ratio = round(clean_open_count / (end_val-start+1), 4)

                # online_feasible: delay >= 0 (no future knowledge needed)
                online_feasible = delay >= 0
                if not online_feasible:
                    online_reason = "requires_future_T_pred"

                proposals.append(dict(
                    episode_id=eid_str, task_name=task, state_id=state_id, split=split,
                    T_gform=tg if tg is not None else "",
                    T_pred=T_pred if T_pred is not None else "",
                    trigger_error=trig_err if trig_err is not None else "",
                    delay=delay, window_start=ws, window_end=we,
                    actual_window_len=actual_len,
                    structural_valid=structural_valid, invalid_reason=inv_reason,
                    attack_eligible_strict=strict_eligible, strict_reason=strict_reason,
                    attack_eligible_relaxed=relaxed_eligible, relaxed_reason=relaxed_reason,
                    online_feasible=online_feasible, online_reason=online_reason,
                    clean_natural_open_ratio=clean_open_ratio,
                    clean_open_count=clean_open_count,
                    clean_open_threshold_strict=STRICT_THRESH,
                    clean_open_threshold_relaxed=args.max_clean_open_ratio,
                    raw_open_semantics="raw_gripper<0.5=OPEN",
                    detector_version="v6", checkpoint=os.path.basename(args.checkpoint),
                    threshold=th, K=K,
                    eval_metrics_json=args.eval_metrics_json,
                    npz_path=args.npz_path, split_col=args.split_col,
                    feature_space_model="X_norm", feature_space_open_ratio="X_raw",
                    notes="detector_based"))

    # ── Write ──
    fields = ["episode_id","task_name","state_id","split","T_gform","T_pred",
              "trigger_error","delay","window_start","window_end","actual_window_len",
              "structural_valid","invalid_reason",
              "attack_eligible_strict","strict_reason",
              "attack_eligible_relaxed","relaxed_reason",
              "online_feasible","online_reason",
              "clean_natural_open_ratio","clean_open_count",
              "clean_open_threshold_strict","clean_open_threshold_relaxed",
              "raw_open_semantics",
              "detector_version","checkpoint","threshold","K",
              "eval_metrics_json","npz_path","split_col",
              "feature_space_model","feature_space_open_ratio","notes"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(proposals)

    n = len(proposals); n_struct = sum(1 for p in proposals if p["structural_valid"])
    n_strict = sum(1 for p in proposals if p["attack_eligible_strict"])
    n_relaxed = sum(1 for p in proposals if p["attack_eligible_relaxed"])
    n_online = sum(1 for p in proposals if p["online_feasible"])
    n_no_trig = sum(1 for p in proposals if "no_trigger" in str(p.get("invalid_reason","")))
    n_clipped = sum(1 for p in proposals if "clipped" in str(p.get("invalid_reason","")))
    print(f"Wrote {n} proposals: {n_struct} structural, "
          f"{n_strict} strict-eligible, {n_relaxed} relaxed-eligible, "
          f"{n_online} online-feasible")
    print(f"  {n_no_trig} no-trigger, {n_clipped} clipped")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
