#!/usr/bin/env python3
"""Train one Factorized Student fold×seed (OOF run).

Usage:
  python train_factorized_oof.py --model-type 25D9D --fold-id 0 --seed 42
      --gpu 1 --output-root <path> --s1-root <path> --teacher-root <path>
      --fold-root <path>
"""
import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes, compute_factorized_normalization,
    verify_factorized_source_roots,
)
from gripper_attack.v5_factorized_student import FactorizedStudent
from gripper_attack.v5_factorized_loss import FactorizedLoss
from gripper_attack.b3_training_protocol import load_fit_fold_bundle


def sha256_file(p): d=hashlib.sha256(); [d.update(b) for b in iter(lambda: p.open("rb").read(1048576), b"")]; return d.hexdigest()

def _atomic_text(p, v):
    t = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    with t.open("x") as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)

def write_seal(root):
    excl = {"SHA256SUMS", "SHA256SUMS.sha256"}
    fs = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in fs)
    _atomic_text(root / "SHA256SUMS", c)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["25D9D", "25D"], required=True)
    ap.add_argument("--fold-id", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # Load fold
    folds = load_fit_fold_bundle(args.fold_root.resolve())
    fold = [f for f in folds["folds"] if f["fold_id"] == args.fold_id][0]
    train_ids = set(fold["train_identities"])
    val_ids = set(fold["validation_identities"])

    # Verify source seals
    verify_factorized_source_roots(args.s1_root, args.teacher_root)

    # Load all episodes via registry
    import csv
    reg = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv")
    rows = list(csv.DictReader(open(reg)))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]
    train_rows = [r for r in fit if r["canonical_parent_key"] in train_ids]
    val_rows = [r for r in fit if r["canonical_parent_key"] in val_ids]

    print(f"Fold {args.fold_id}: train={len(train_rows)} val={len(val_rows)} seed={args.seed} gpu={args.gpu}")

    train_eps = load_factorized_episodes(args.s1_root, args.teacher_root, train_rows)
    val_eps = load_factorized_episodes(args.s1_root, args.teacher_root, val_rows)

    # Normalization from train only
    mean_25d, std_25d = compute_factorized_normalization(train_eps)
    print(f"Norm: mean={mean_25d[:3].tolist()}... std={std_25d[:3].tolist()}...")

    # Model
    use_9d = args.model_type == "25D9D"
    model = FactorizedStudent(use_9d=use_9d).to(device)
    loss_fn = FactorizedLoss(consistency_weight=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_grasp": [], "val_manip": [], "val_release": []}
    best_val = float("inf")

    for epoch in range(30):
        model.train()
        train_losses = []
        for ep in train_eps:
            if not ep.route_supported:
                continue
            T = len(ep.features_25d)
            x25 = ((ep.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
            mask = ep.valid_mask.unsqueeze(0).to(device)
            x9 = None
            m9 = None

            opt.zero_grad()
            logits = model.forward_logits(x25, x9, mask, m9, ep.mechanism_route)
            loss, _ = loss_fn(logits, [ep])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        head_metrics = defaultdict(list)
        with torch.no_grad():
            for ep in val_eps:
                if not ep.route_supported:
                    continue
                T = len(ep.features_25d)
                x25 = ((ep.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
                mask = ep.valid_mask.unsqueeze(0).to(device)
                logits = model.forward_logits(x25, None, mask, None, ep.mechanism_route)
                loss, m = loss_fn(logits, [ep])
                val_losses.append(loss.item())
                for k in ["grasp", "manipulation", "release"]:
                    head_metrics[k].append(m[k])

        avg_train = sum(train_losses) / max(1, len(train_losses))
        avg_val = sum(val_losses) / max(1, len(val_losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        for k in ["grasp", "manipulation", "release"]:
            history[f"val_{k}"].append(sum(head_metrics[k]) / max(1, len(head_metrics[k])))

        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"g={history['val_grasp'][-1]:.4f} m={history['val_manip'][-1]:.4f} r={history['val_release'][-1]:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save sealed output
    ckpt = {"state_dict": best_state, "config": {"hidden_dim": 128, "use_9d": use_9d},
            "fold_id": args.fold_id, "seed": args.seed, "model_type": args.model_type}
    torch.save(ckpt, staging / "checkpoint.pt")
    _atomic_text(staging / "history.json", json.dumps(history, indent=2))
    _atomic_text(staging / "normalization.json", json.dumps({
        "mean_25d": mean_25d.tolist(), "std_25d": std_25d.tolist(),
    }))
    _atomic_text(staging / "run_config.json", json.dumps({
        "model_type": args.model_type, "fold_id": args.fold_id, "seed": args.seed,
        "gpu": args.gpu, "epochs": 30, "lr": 1e-3, "weight_decay": 1e-5, "grad_clip": 5.0,
        "train_identities": len(train_rows), "val_identities": len(val_rows),
    }, indent=2))
    _atomic_text(staging / "source_binding.json", json.dumps({
        "s1_root": str(args.s1_root), "teacher_root": str(args.teacher_root),
        "fold_root": str(args.fold_root),
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f"  Sealed: {out} val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
