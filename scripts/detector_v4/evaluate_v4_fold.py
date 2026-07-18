#!/usr/bin/env python3
"""Evaluate a trained V4 detector on validation fold, producing per-step predictions."""

import json, sys, argparse, hashlib
from pathlib import Path
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

# Reuse from train_v4_detector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v4_detector import (
    load_v4_episode, derive_dynamic_features, ALL_VIEWS,
    CandidateAGRU, CandidateBGRU, CandidateCGRU,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--windows-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    view = ckpt["view"]
    candidate = ckpt["candidate"]
    seed = ckpt["seed"]
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    print(f"Evaluating: view={view} candidate={candidate} seed={seed} fold={args.fold}")

    # Build model
    input_dim = ALL_VIEWS[view].feature_count
    if candidate == "A":
        model = CandidateAGRU()
    elif candidate == "B":
        model = CandidateBGRU()
    elif candidate == "C":
        model = CandidateCGRU()
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device)
    model.eval()

    # Validation states
    val_states = list(range(args.fold * 5, (args.fold + 1) * 5))
    episodes = []
    for suite in SUITES:
        for task in range(10):
            for state in val_states:
                ep = load_v4_episode(args.s1_root, args.windows_root,
                                    suite, task, state, view)
                if ep is not None:
                    episodes.append(ep)

    print(f"Loaded {len(episodes)} validation episodes")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    episode_metrics = []

    with torch.no_grad():
        for ep in episodes:
            x = ep.features.unsqueeze(0)  # [1, T, F]
            x = (x - norm_mean.to(x.device)) / norm_std.to(x.device)
            padding = torch.ones(1, ep.n_steps, dtype=torch.bool).to(args.device)

            logits, _ = model(x.to(args.device), padding)

            crit_head = "criticality" if "criticality" in logits else "valid_retention"
            if crit_head not in logits:
                continue

            crit_logits = logits[crit_head].squeeze(0).cpu()  # [T]
            crit_probs = torch.sigmoid(crit_logits)

            veto_logits = logits.get("veto", None)
            veto_probs = torch.sigmoid(veto_logits.squeeze(0).cpu()) if veto_logits is not None else None

            # Compute per-step emission (threshold 0.5 for now)
            pred_emit = crit_probs >= 0.5

            # Move targets to CPU for comparison
            ep_crit_target = ep.targets["criticality"].cpu()
            ep_veto_target = ep.targets["veto"].cpu()

            for t in range(ep.n_steps):
                rec = {
                    "identity": ep.identity, "step": t,
                    "fold_id": args.fold, "seed": seed,
                    "criticality_prob": float(crit_probs[t]),
                    "pred_emit_criticality": bool(pred_emit[t]),
                    "target_criticality": float(ep.targets["criticality"][t]),
                    "target_veto": float(ep.targets["veto"][t]),
                    "candidate_close": bool(ep.targets["veto"][t] > 0.5 or ep.targets["criticality"][t] > 0.5),
                }
                if veto_probs is not None:
                    rec["veto_prob"] = float(veto_probs[t])
                all_records.append(rec)

            # Episode-level metrics
            has_positive = ep_crit_target.sum() > 0
            has_veto_target = ep_veto_target.sum() > 0
            is_negative = has_veto_target and not has_positive

            ep_emit = pred_emit.any().item()
            ep_hit = (pred_emit & (ep_crit_target > 0.5)).any().item() if has_positive else False
            ep_false = ep_emit and is_negative

            episode_metrics.append({
                "identity": ep.identity,
                "has_positive": has_positive,
                "is_hard_negative": is_negative,
                "any_emit": ep_emit,
                "hit": ep_hit,
                "false_emit": ep_false,
                "max_criticality_prob": float(crit_probs.max()),
                "n_emits": int(pred_emit.sum()),
            })

    # Write prediction records
    with open(out_dir / "prediction_records.jsonl", "w") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec) + "\n")

    # Compute aggregate metrics
    n_pos = sum(1 for e in episode_metrics if e["has_positive"])
    n_neg = sum(1 for e in episode_metrics if e["is_hard_negative"])
    n_hit = sum(1 for e in episode_metrics if e["hit"])
    n_false = sum(1 for e in episode_metrics if e["false_emit"])
    n_any_emit = sum(1 for e in episode_metrics if e["any_emit"])

    results = {
        "fold": args.fold,
        "view": view, "candidate": candidate, "seed": seed,
        "n_validation_episodes": len(episodes),
        "n_positive_episodes": n_pos,
        "n_hard_negative_episodes": n_neg,
        "full_criticality_hit_rate": n_hit / n_pos if n_pos else 0,
        "hard_negative_false_emit_rate": n_false / n_neg if n_neg else 0,
        "episode_any_emit_rate": n_any_emit / len(episodes) if episodes else 0,
        "threshold": 0.5,
    }

    print(json.dumps(results, indent=2))

    with open(out_dir / "evaluation_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # SHA256SUMS
    files = sorted(out_dir.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    with open(out_dir / "SHA256SUMS", "w") as fh:
        for fp in file_list:
            rel = fp.relative_to(out_dir)
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel}\n")


if __name__ == "__main__":
    main()
