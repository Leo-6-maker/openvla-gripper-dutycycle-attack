#!/usr/bin/env python3
"""SC5 MLP training on canonical dataset — 3 seeds, best checkpoint, Gate D2 evaluation.

Reuses: train_sc5_v4.py architecture and training loop.
Changes:
  - Canonical dataset path (multi-source, dedup'd)
  - Optional mechanism filter (train on pick-and-place only)
  - Per-mechanism held-out evaluation
  - Save full evaluation manifest
"""
import csv, json, os, sys, random, numpy as np, torch, copy, argparse
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

SC5_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]


class SC5MLP(torch.nn.Module):
    def __init__(self, n_feat, hidden=128):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(n_feat, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
        )
        self.phase_head = torch.nn.Linear(hidden // 2, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(hidden // 2, 1)
        self.release_head = torch.nn.Linear(hidden // 2, 1)
        self.confidence_head = torch.nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "phase_logits": self.phase_head(h),
            "corridor_logit": self.corridor_head(h),
            "release_logit": self.release_head(h),
            "confidence_logit": self.confidence_head(h),
        }


def load_data(csv_path, seed, mechanism_filter=None):
    """Load canonical dataset with proper split isolation.
    mechanism_filter: if set, only include this mechanism (e.g. 'pick_and_place').
    """
    rows = []; held_rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ok = all(r.get(fn, "") not in ("", "nan", None) for fn in SC5_FEATURES)
            if not ok:
                continue
            if mechanism_filter and r.get("mechanism") != mechanism_filter:
                continue
            is_h = r.get("is_held_out", "False") in ("True", "true", "1")
            if is_h:
                held_rows.append(r)
            else:
                rows.append(r)

    # Split non-held episodes 75/25 train/val
    eps = sorted(set(r.get("run_id", "") for r in rows))
    random.seed(seed); random.shuffle(eps)
    n_tr = int(len(eps) * 0.75)
    tr_set = set(eps[:n_tr]); vl_set = set(eps[n_tr:])
    tr_rows = [r for r in rows if r.get("run_id", "") in tr_set]
    vl_rows = [r for r in rows if r.get("run_id", "") in vl_set]

    def mk(rl):
        X = np.array([[float(r[fn]) for fn in SC5_FEATURES] for r in rl], dtype=np.float32)
        yp = np.array([SC5_PHASES.index(r.get("teacher_phase", "abstain_unsupported"))
                       if r.get("teacher_phase", "") in SC5_PHASES else len(SC5_PHASES) - 1
                       for r in rl], dtype=np.int64)
        yc = np.array([float(r.get("teacher_sc5_corridor_active", 0)) for r in rl], dtype=np.float32)
        yr = np.array([float(r.get("teacher_phase", "") == "release_safe") for r in rl], dtype=np.float32)
        return X, {"phase": yp, "corridor": yc, "release": yr}

    Xtr, Ytr = mk(tr_rows); Xvl, Yvl = mk(vl_rows); Xte, Yte = mk(held_rows)
    return Xtr, Ytr, Xvl, Yvl, Xte, Yte, len(held_rows), len(tr_rows), len(vl_rows)


def train(model, Xtr, Ytr, Xvl, Yvl, epochs=120, lr=0.001, device="cpu"):
    model = model.to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xvl, dtype=torch.float32, device=device)
    counts = Counter(Ytr["phase"].tolist())
    total = sum(counts.values())
    cw = torch.tensor([total / max(counts.get(i, 1), 1) for i in range(len(SC5_PHASES))],
                      dtype=torch.float32, device=device)
    pl = torch.nn.CrossEntropyLoss(weight=cw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_vl = float("inf"); best_state = None; patience = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device); tl = 0.0; nb = 0
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i+128]; xb = Xt[idx]
            yp = torch.tensor(Ytr["phase"][idx.cpu().numpy()], dtype=torch.long, device=device)
            yc = torch.tensor(Ytr["corridor"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            yr = torch.tensor(Ytr["release"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            out = model(xb)
            loss = (pl(out["phase_logits"], yp) +
                    0.5 * bce(out["corridor_logit"], yc) +
                    0.3 * bce(out["release_logit"], yr))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            ov = model(Xv); ypv = torch.tensor(Yvl["phase"], dtype=torch.long, device=device)
            vl = pl(ov["phase_logits"], ypv).item()
            va = (ov["phase_logits"].argmax(1) == ypv).float().mean().item()
        if vl < best_vl - 1e-6:
            best_vl = vl; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
        if patience >= 25:
            break
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  e{ep}: tr={tl / max(nb, 1):.3f} vl={vl:.3f} pa={va:.3f}")
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tables/v2_sc5_canonical_dataset.csv")
    ap.add_argument("--output_dir", default="outputs/sc5_canonical")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mechanism", default=None,
                    help="Only use this mechanism (e.g. 'pick_and_place')")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    mech_label = f" (mechanism={args.mechanism})" if args.mechanism else " (all)"
    print(f"Loading {args.dataset}{mech_label}...")
    Xtr, Ytr, Xvl, Yvl, Xte, Yte, n_h, n_tr, n_vl = load_data(
        args.dataset, args.seed, mechanism_filter=args.mechanism)
    print(f"  train={n_tr} val={n_vl} held_test={n_h}")

    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mean) / std; Xvl = (Xvl - mean) / std
    if len(Xte) > 0:
        Xte = (Xte - mean) / std

    model = SC5MLP(n_feat=len(SC5_FEATURES), hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} params (hidden={args.hidden})")
    model = train(model, Xtr, Ytr, Xvl, Yvl, device=args.device)

    # Evaluation
    model.eval()
    eval_results = {}
    with torch.no_grad():
        if len(Xte) > 0:
            Xt_t = torch.tensor(Xte, dtype=torch.float32, device=args.device)
            out = model(Xt_t)
            ypt = torch.tensor(Yte["phase"], dtype=torch.long, device=args.device)
            held_phase_acc = (out["phase_logits"].argmax(1) == ypt).float().mean().item()
            eval_results["held_out_phase_acc"] = round(held_phase_acc, 4)
            print(f"Held-out phase acc: {held_phase_acc:.3f}")

        # Val phase acc
        Xv_t = torch.tensor(Xvl, dtype=torch.float32, device=args.device)
        ov = model(Xv_t); ypv = torch.tensor(Yvl["phase"], dtype=torch.long, device=args.device)
        val_acc = (ov["phase_logits"].argmax(1) == ypv).float().mean().item()
        eval_results["val_phase_acc"] = round(val_acc, 4)

    eval_results["n_params"] = n_params
    eval_results["n_train_rows"] = len(Xtr)
    eval_results["n_val_rows"] = len(Xvl)
    eval_results["n_held_rows"] = len(Xte)
    eval_results["hidden_dim"] = args.hidden
    eval_results["mechanism_filter"] = args.mechanism

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_s{args.seed}"
    if args.mechanism:
        suffix = f"_{args.mechanism}_s{args.seed}"

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(), "mean": mean, "std": std,
        "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
        "eval_results": eval_results,
    }, out_dir / f"sc5_mlp{suffix}.pt")

    # Save evaluation manifest
    with open(out_dir / f"sc5_mlp{suffix}_eval.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Saved to {out_dir}/sc5_mlp{suffix}.pt")
    return eval_results


if __name__ == '__main__':
    main()
