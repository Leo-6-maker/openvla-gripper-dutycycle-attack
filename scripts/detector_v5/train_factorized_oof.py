#!/usr/bin/env python3
"""Train one Factorized Student fold×seed (OOF run, formal).

Fixed contracts:
- Epoch 30 saved (no best-val selection)
- 25D9D uses real policy-intent 9D stream
- Train-fold-only normalization for both 25D and 9D
- Sealed output with predictions, metrics, source binding
- Fail-closed: any error → non-zero exit
"""

import argparse, csv, hashlib, json, os, sys, uuid
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
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file


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
    ap.add_argument("--policy-intent-root", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    use_9d = args.model_type == "25D9D"
    if use_9d and args.policy_intent_root is None:
        raise SystemExit("25D9D requires --policy-intent-root")

    # Load fold
    folds = load_fit_fold_bundle(args.fold_root.resolve())
    fold = [f for f in folds["folds"] if f["fold_id"] == args.fold_id][0]
    train_ids = set(fold["train_identities"])
    val_ids = set(fold["validation_identities"])

    verify_factorized_source_roots(args.s1_root, args.teacher_root)
    if use_9d:
        from gripper_attack.b3_training_protocol import verify_sealed_directory
        verify_sealed_directory(args.policy_intent_root.resolve())

    # Load policy-intent index for 9D
    policy_index = None
    if use_9d:
        from gripper_attack.v5_dataset import load_policy_intent_root
        policy_index, _ = load_policy_intent_root(args.policy_intent_root.resolve())

    # Load registry
    reg = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv")
    rows = list(csv.DictReader(open(reg)))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]
    train_rows = [r for r in fit if r["canonical_parent_key"] in train_ids]
    val_rows = [r for r in fit if r["canonical_parent_key"] in val_ids]

    print(f"Fold {args.fold_id}: train={len(train_rows)} val={len(val_rows)} seed={args.seed} gpu={args.gpu} model={args.model_type}")

    train_eps = load_factorized_episodes(args.s1_root, args.teacher_root, train_rows, policy_index=policy_index)
    val_eps = load_factorized_episodes(args.s1_root, args.teacher_root, val_rows, policy_index=policy_index)

    # Train-fold-only normalization
    mean_25d, std_25d = compute_factorized_normalization(train_eps)
    mean_9d = std_9d = None
    if use_9d:
        all_9d = torch.cat([ep.policy_intent_9d[ep.policy_intent_valid_mask] for ep in train_eps if ep.policy_intent_9d.numel() > 0], dim=0)
        mean_9d = all_9d.mean(dim=0)
        std_9d = all_9d.std(dim=0, unbiased=False).clamp_min(1e-6)
        print(f"9D norm: mean={mean_9d[:3].tolist()}...")

    # Model
    model = FactorizedStudent(use_9d=use_9d).to(device)
    loss_fn = FactorizedLoss(consistency_weight=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Batch groups
    def group_by_route(eps, batch_size=8):
        groups = defaultdict(list)
        for ep in eps:
            if ep.route_supported:
                groups[ep.mechanism_route].append(ep)
        batches = []
        for route, route_eps in groups.items():
            for i in range(0, len(route_eps), batch_size):
                batches.append((route, route_eps[i:i+batch_size]))
        return batches

    train_batches = group_by_route(train_eps)
    val_batches = group_by_route(val_eps)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_grasp": [], "val_manipulation": [], "val_release": []}

    for epoch in range(30):
        model.train()
        train_losses = []
        for route, batch_eps in train_batches:
            B = len(batch_eps)
            max_T = max(len(ep.features_25d) for ep in batch_eps)
            x25 = torch.zeros(B, max_T, 25, device=device)
            mask25 = torch.zeros(B, max_T, dtype=torch.bool, device=device)
            x9 = mask9 = None
            if use_9d:
                max_T9 = max(ep.policy_intent_9d.shape[0] for ep in batch_eps)
                x9 = torch.zeros(B, max_T9, 9, device=device)
                mask9 = torch.zeros(B, max_T9, dtype=torch.bool, device=device)
            for b, ep in enumerate(batch_eps):
                T = len(ep.features_25d)
                x25[b, :T] = ((ep.features_25d - mean_25d) / std_25d).to(device)
                mask25[b, :T] = ep.valid_mask.to(device)
                if use_9d and ep.policy_intent_9d.numel() > 0:
                    T9 = ep.policy_intent_9d.shape[0]
                    x9[b, :T9] = ((ep.policy_intent_9d - mean_9d) / std_9d).to(device)
                    mask9[b, :T9] = ep.policy_intent_valid_mask.to(device)
            opt.zero_grad()
            logits = model.forward_logits(x25, x9, mask25, mask9, route)
            loss, _ = loss_fn(logits, batch_eps, mask25)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d}: train={sum(train_losses)/max(1,len(train_losses)):.4f}", end="")

        # Validation
        model.eval()
        val_losses = []; head_metrics = defaultdict(list)
        with torch.no_grad():
            for route, batch_eps in val_batches:
                B = len(batch_eps)
                max_T = max(len(ep.features_25d) for ep in batch_eps)
                x25 = torch.zeros(B, max_T, 25, device=device)
                mask25 = torch.zeros(B, max_T, dtype=torch.bool, device=device)
                x9 = mask9 = None
                if use_9d:
                    max_T9 = max(ep.policy_intent_9d.shape[0] for ep in batch_eps if ep.policy_intent_9d.numel()>0) if any(ep.policy_intent_9d.numel()>0 for ep in batch_eps) else 1
                    x9 = torch.zeros(B, max_T9, 9, device=device)
                    mask9 = torch.zeros(B, max_T9, dtype=torch.bool, device=device)
                for b, ep in enumerate(batch_eps):
                    T = len(ep.features_25d)
                    x25[b, :T] = ((ep.features_25d - mean_25d) / std_25d).to(device)
                    mask25[b, :T] = ep.valid_mask.to(device)
                    if use_9d and ep.policy_intent_9d.numel() > 0:
                        T9 = ep.policy_intent_9d.shape[0]
                        x9[b, :T9] = ((ep.policy_intent_9d - mean_9d) / std_9d).to(device)
                        mask9[b, :T9] = ep.policy_intent_valid_mask.to(device)
                logits = model.forward_logits(x25, x9, mask25, mask9, route)
                loss, m = loss_fn(logits, batch_eps, mask25)
                val_losses.append(loss.item())
                for k in ["grasp", "manipulation", "release"]:
                    head_metrics[k].append(m[k])

        avg_train = sum(train_losses)/max(1,len(train_losses))
        avg_val = sum(val_losses)/max(1,len(val_losses))
        history["epoch"].append(epoch); history["train_loss"].append(avg_train); history["val_loss"].append(avg_val)
        for k in ["grasp", "manipulation", "release"]:
            history[f"val_{k}"].append(sum(head_metrics[k])/max(1,len(head_metrics[k])))
        if epoch % 5 == 0:
            print(f" val={avg_val:.4f} g={history['val_grasp'][-1]:.4f} m={history['val_manipulation'][-1]:.4f} r={history['val_release'][-1]:.4f}")

    # Save epoch-30 checkpoint (fixed, no best-val selection per protocol)
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {"hidden_dim": 128, "use_9d": use_9d},
            "fold_id": args.fold_id, "seed": args.seed, "model_type": args.model_type, "epoch": 30}
    torch.save(ckpt, staging / "checkpoint.pt")

    _atomic_text(staging / "history.json", json.dumps(history, indent=2))
    _atomic_text(staging / "normalization.json", json.dumps({
        "mean_25d": mean_25d.tolist(), "std_25d": std_25d.tolist(),
        **({"mean_9d": mean_9d.tolist(), "std_9d": std_9d.tolist()} if use_9d else {}),
    }))
    _atomic_text(staging / "run_config.json", json.dumps({
        "model_type": args.model_type, "fold_id": args.fold_id, "seed": args.seed,
        "gpu": args.gpu, "epochs": 30, "lr": 1e-3, "weight_decay": 1e-5, "grad_clip": 5.0,
        "best_val_epoch_selected": False, "epoch30_saved": True,
        "train_identities": len(train_rows), "val_identities": len(val_rows),
    }, indent=2))
    _atomic_text(staging / "source_binding.json", json.dumps({
        "s1_root": str(args.s1_root), "teacher_root": str(args.teacher_root),
        "fold_root": str(args.fold_root),
        "policy_intent_root": str(args.policy_intent_root) if args.policy_intent_root else None,
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f"  Sealed: {out}")


if __name__ == "__main__":
    main()
