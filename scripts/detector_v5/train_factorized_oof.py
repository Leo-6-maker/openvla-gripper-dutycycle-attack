#!/usr/bin/env python3
"""Train one Factorized Student fold×seed (formal OOF run).

Contracts:
- Formal authorization verification (seal, commit, protocol SHA)
- Route-balanced epoch (alternating single/multi, deterministic upsampling)
- Fold-only class weights per head per route
- Full outputs: predictions, metrics, environment, authorization receipt
- Fixed epoch 30 checkpoint
"""

import argparse, csv, hashlib, json, os, sys, uuid, platform
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes, compute_factorized_normalization,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.v5_factorized_student import FactorizedStudent
from gripper_attack.v5_factorized_loss import FactorizedLoss
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory


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
    ap.add_argument("--authorization-root", type=Path, required=True)
    ap.add_argument("--f3-root", type=Path, required=True)
    ap.add_argument("--expected-source-commit", type=str, required=True)
    ap.add_argument("--policy-intent-root", type=Path, default=None)
    args = ap.parse_args()
    use_9d = args.model_type == "25D9D"
    if use_9d and args.policy_intent_root is None:
        raise SystemExit("25D9D requires --policy-intent-root")

    # ── Authorization verification ────────────────────────────────────
    auth_root = args.authorization_root.resolve()
    verify_sealed_directory(auth_root)
    auth = json.loads((auth_root / "authorization.json").read_text())
    if auth.get("formal_oof_training_authorized") is not True:
        raise RuntimeError("formal OOF training is not authorized")
    if auth.get("full_fit_authorized") is not False:
        raise RuntimeError("full-FIT must not be authorized")
    if auth.get("attack_authorized") is not False:
        raise RuntimeError("attack must not be authorized")
    if auth.get("source_commit") != args.expected_source_commit:
        raise RuntimeError(f"auth commit {auth.get('source_commit','?')[:8]} != expected {args.expected_source_commit[:8]}")
    for key in ["teacher_root_seal", "s1_root_seal", "f3_root_seal"]:
        if key not in auth:
            raise RuntimeError(f"authorization missing {key}")

    # Full F3 verification
    f3_root = args.f3_root.resolve()
    verify_sealed_directory(f3_root)
    f3_audit = json.loads((f3_root / "geometry_audit.json").read_text())
    if f3_audit.get("status") != "PASS_FINAL_STUDENT_TRAINING":
        raise RuntimeError(f"F3 status is {f3_audit.get('status')}, not PASS_FINAL_STUDENT_TRAINING")
    if f3_audit.get("formal_training_authorized") is not True:
        raise RuntimeError("F3 does not authorize training")
    f3_seal = sha256_file(f3_root / "SHA256SUMS")
    if f3_seal != auth.get("f3_root_seal", ""):
        raise RuntimeError("F3 seal mismatch")

    s1_seal = sha256_file(args.s1_root / "SHA256SUMS")
    teacher_seal = sha256_file(args.teacher_root / "SHA256SUMS")
    if s1_seal != auth.get("s1_root_seal", ""):
        raise RuntimeError("S1 seal mismatch")
    if teacher_seal != auth.get("teacher_root_seal", ""):
        raise RuntimeError("Teacher seal mismatch")

    # Fold root binding
    fold_root_resolved = args.fold_root.resolve()
    verify_sealed_directory(fold_root_resolved)
    fold_seal = sha256_file(fold_root_resolved / "SHA256SUMS")
    if fold_seal != auth.get("fold_root_seal", ""):
        raise RuntimeError("Fold seal mismatch")

    # Policy-intent binding (required for 25D9D, optional for 25D)
    if use_9d:
        pi_root = args.policy_intent_root.resolve()
        verify_sealed_directory(pi_root)
        pi_seal = sha256_file(pi_root / "SHA256SUMS")
        if pi_seal != auth.get("policy_intent_root_seal", ""):
            raise RuntimeError("Policy-intent seal mismatch")

    # Source code SHAs
    for key, rel_path in [
        ("dataset_sha", "src/gripper_attack/v5_factorized_dataset.py"),
        ("model_sha", "src/gripper_attack/v5_factorized_student.py"),
        ("loss_sha", "src/gripper_attack/v5_factorized_loss.py"),
        ("trainer_sha", "scripts/detector_v5/train_factorized_oof.py"),
    ]:
        if key not in auth:
            raise RuntimeError(f"Authorization missing {key}")
        actual = sha256_file(ROOT / rel_path)
        if actual != auth[key]:
            raise RuntimeError(f"{key} mismatch: {actual[:16]} != {auth[key][:16]}")

    # Student protocol SHA
    proto_path = ROOT / "configs/DETECTOR_V5_FACTORIZED_STUDENT_PROTOCOL_V1.json"
    if sha256_file(proto_path) != auth.get("student_protocol_sha", ""):
        raise RuntimeError("Student protocol SHA mismatch")

    # Git commit (source code SHAs verified against authorization below)
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_source_commit:
        raise RuntimeError(f"git HEAD {head[:8]} != expected {args.expected_source_commit[:8]}")

    # Environment fingerprint
    env_info = {
        "python": sys.executable, "python_version": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "host": platform.node(),
    }
    print(f"Auth: PASS | commit={head[:8]}")

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

    verify_factorized_source_roots(args.s1_root, args.teacher_root)
    if use_9d:
        verify_sealed_directory(args.policy_intent_root.resolve())
        from gripper_attack.v5_dataset import load_policy_intent_root
        policy_index, _ = load_policy_intent_root(args.policy_intent_root.resolve())
    else:
        policy_index = None

    reg = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv")
    rows = list(csv.DictReader(open(reg)))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]
    train_rows = [r for r in fit if r["canonical_parent_key"] in train_ids]
    val_rows = [r for r in fit if r["canonical_parent_key"] in val_ids]

    train_eps = load_factorized_episodes(args.s1_root, args.teacher_root, train_rows, policy_index=policy_index)
    val_eps = load_factorized_episodes(args.s1_root, args.teacher_root, val_rows, policy_index=policy_index)

    # Normalization (train only)
    mean_25d, std_25d = compute_factorized_normalization(train_eps)
    mean_9d = std_9d = None
    if use_9d:
        all_9d = torch.cat([ep.policy_intent_9d[ep.policy_intent_valid_mask] for ep in train_eps if ep.policy_intent_9d.numel() > 0], dim=0)
        mean_9d = all_9d.mean(dim=0); std_9d = all_9d.std(dim=0, unbiased=False).clamp_min(1e-6)

    # ── Fold-only class weights ──────────────────────────────────────
    def compute_class_weights(eps_list):
        """Per-route, per-head positive weight from training fold only."""
        counts = defaultdict(lambda: defaultdict(lambda: {"pos": 0, "neg": 0}))
        for ep in eps_list:
            r = ep.mechanism_route
            if r not in SUPPORTED_ROUTES:
                continue
            eids = ep.event_id
            for eid in eids.unique().tolist():
                em = eids == eid
                for head, tgt, km in [("grasp", ep.grasp_target, ep.grasp_known_mask),
                                       ("manipulation", ep.manipulation_target, ep.manipulation_known_mask),
                                       ("release", ep.release_target, ep.release_known_mask)]:
                    if km[em].any():
                        counts[r][head]["pos"] += int(tgt[em].any())
                        counts[r][head]["neg"] += int(not tgt[em].any() and km[em].any())
        weights = {}
        for r, heads in counts.items():
            weights[r] = {}
            for h, c in heads.items():
                pos, neg = c["pos"], c["neg"]
                w_pos = (pos + neg) / max(1, 2 * pos)
                w_neg = (pos + neg) / max(1, 2 * neg)
                weights[r][h] = {"pos_weight": round(w_pos, 4), "neg_weight": round(w_neg, 4), "pos_count": pos, "neg_count": neg}
        return weights

    class_weights = compute_class_weights(train_eps)
    print(f"Class weights: single grasp pos={class_weights.get('single_object_pick_place',{}).get('grasp',{}).get('pos_count',0)}")

    # ── Route-balanced batch construction ─────────────────────────────
    def build_batches(eps_list, batch_size=8):
        groups = defaultdict(list)
        for ep in eps_list:
            if ep.route_supported:
                groups[ep.mechanism_route].append(ep)
        single_eps = groups.get("single_object_pick_place", [])
        multi_eps = groups.get("multi_object_transfer", [])

        def make_batches(eps, route):
            batches = []
            for i in range(0, len(eps), batch_size):
                batches.append((route, eps[i:i+batch_size]))
            return batches

        single_batches = make_batches(single_eps, "single_object_pick_place")
        multi_batches = make_batches(multi_eps, "multi_object_transfer")

        # Route-balanced: alternate, upsample minority deterministically
        N = max(len(single_batches), len(multi_batches))
        rng = __import__("random").Random(42)
        if len(single_batches) < N:
            single_batches = single_batches + [single_batches[i % len(single_batches)] for i in range(N - len(single_batches))]
        if len(multi_batches) < N:
            multi_batches = multi_batches + [multi_batches[i % len(multi_batches)] for i in range(N - len(multi_batches))]
        # Interleave
        balanced = []
        for i in range(N):
            balanced.append(single_batches[i])
            balanced.append(multi_batches[i])
        rng.shuffle(balanced)
        return balanced

    train_batches = build_batches(train_eps)
    # Validation: no upsampling, report per-route macro
    val_batches_single = [("single_object_pick_place", val_eps_single[i:i+8])
                          for i in range(0, len(val_eps_single), 8)] if (val_eps_single := [e for e in val_eps if e.mechanism_route == "single_object_pick_place"]) else []
    val_batches_multi = [("multi_object_transfer", val_eps_multi[i:i+8])
                         for i in range(0, len(val_eps_multi), 8)] if (val_eps_multi := [e for e in val_eps if e.mechanism_route == "multi_object_transfer"]) else []

    # ── Helper: batch to device ───────────────────────────────────────
    def batch_to_device(batch_eps, route):
        B = len(batch_eps)
        max_T = max(len(ep.features_25d) for ep in batch_eps)
        x25 = torch.zeros(B, max_T, 25, device=device)
        mask25 = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        x9 = mask9 = None
        if use_9d:
            max_T9 = max((ep.policy_intent_9d.shape[0] for ep in batch_eps if ep.policy_intent_9d.numel()>0), default=1)
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
        return x25, x9, mask25, mask9

    # ── Model ─────────────────────────────────────────────────────────
    model = FactorizedStudent(use_9d=use_9d).to(device)
    loss_fn = FactorizedLoss(consistency_weight=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_grasp": [], "val_manipulation": [], "val_release": []}

    for epoch in range(30):
        model.train()
        train_losses = []
        for route, batch_eps in train_batches:
            x25, x9, mask25, mask9 = batch_to_device(batch_eps, route)
            opt.zero_grad()
            logits = model.forward_logits(x25, x9, mask25, mask9, route)
            cw = class_weights.get(route, {})
            loss, _ = loss_fn(logits, batch_eps, mask25, class_weights=cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []; head_metrics = defaultdict(list)
        route_losses = defaultdict(list)
        with torch.no_grad():
            for route, batch_eps in (val_batches_single + val_batches_multi):
                x25, x9, mask25, mask9 = batch_to_device(batch_eps, route)
                logits = model.forward_logits(x25, x9, mask25, mask9, route)
                cw = class_weights.get(route, {})
                loss, m = loss_fn(logits, batch_eps, mask25, class_weights=cw)
                val_losses.append(loss.item())
                route_losses[route].append(loss.item())
                for k in ["grasp", "manipulation", "release"]: head_metrics[k].append(m[k])

        avg_train = sum(train_losses)/max(1,len(train_losses))
        # Route macro: (single_mean + multi_mean) / 2
        single_m = sum(route_losses["single_object_pick_place"])/max(1,len(route_losses["single_object_pick_place"]))
        multi_m = sum(route_losses["multi_object_transfer"])/max(1,len(route_losses["multi_object_transfer"]))
        avg_val = (single_m + multi_m) / 2
        history["epoch"].append(epoch); history["train_loss"].append(avg_train); history["val_loss"].append(avg_val)
        for k in ["grasp", "manipulation", "release"]:
            history[f"val_{k}"].append(sum(head_metrics[k])/max(1,len(head_metrics[k])))
        history.setdefault("val_single", []).append(single_m)
        history.setdefault("val_multi", []).append(multi_m)
        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d}: train={avg_train:.4f} val={avg_val:.4f} g={history['val_grasp'][-1]:.4f} m={history['val_manipulation'][-1]:.4f} r={history['val_release'][-1]:.4f}")

    # ── Save epoch-30 checkpoint ──────────────────────────────────────
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {"hidden_dim": 128, "use_9d": use_9d},
            "fold_id": args.fold_id, "seed": args.seed, "model_type": args.model_type, "epoch": 30}
    torch.save(ckpt, staging / "checkpoint.pt")

    # ── Formal outputs ────────────────────────────────────────────────
    _atomic_text(staging / "history.json", json.dumps(history, indent=2))
    _atomic_text(staging / "normalization.json", json.dumps({
        "mean_25d": mean_25d.tolist(), "std_25d": std_25d.tolist(),
        **({"mean_9d": mean_9d.tolist(), "std_9d": std_9d.tolist()} if use_9d else {}),
    }))
    _atomic_text(staging / "class_weights.json", json.dumps(class_weights, indent=2))
    _atomic_text(staging / "environment.json", json.dumps(env_info, indent=2))
    _atomic_text(staging / "authorization_receipt.json", json.dumps({
        "authorization_root": str(auth_root), "authorization_seal": sha256_file(auth_root / "SHA256SUMS"),
        "source_commit": head, "f3_seal": f3_seal,
    }, indent=2))
    _atomic_text(staging / "source_binding.json", json.dumps({
        "s1_root": str(args.s1_root), "s1_seal": s1_seal,
        "teacher_root": str(args.teacher_root), "teacher_seal": teacher_seal,
        "fold_root": str(args.fold_root), "fold_id": args.fold_id, "seed": args.seed,
        "model_type": args.model_type, "source_commit": head,
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f"  Sealed: {out}")


if __name__ == "__main__":
    main()
