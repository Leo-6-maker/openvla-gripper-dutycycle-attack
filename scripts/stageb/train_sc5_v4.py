#!/usr/bin/env python3
"""SC5 v4: fixed held-out split, best checkpoint, real corridor labels."""
import csv, hashlib, json, math, os, sys, random, numpy as np, torch, copy, argparse
from collections import Counter
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
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(n_feat, hidden), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(hidden, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(hidden, 1)
        self.release_head = torch.nn.Linear(hidden, 1)
        self.confidence_head = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h), "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h), "confidence_logit": self.confidence_head(h)}

def load_data(csv_path, seed, legacy_random_split=False):
    """Load SC5 dataset. Returns (Xtr, Ytr, Xvl, Yvl, Xte, Yte, n_h, n_tr, n_vl, metadata).

    Canonical mode (default): requires split column, fail-closed on any issue.
    Legacy mode (--legacy_random_split): random 75/25 with silent row filtering.
    """
    all_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            all_rows.append(r)

    has_split_col = 'split' in fieldnames
    metadata = {'split_mode': 'unknown', 'train_eps': [], 'val_eps': [], 'heldout_eps': []}

    if legacy_random_split:
        metadata['split_mode'] = 'legacy_random'
        # Legacy: filter valid rows, random shuffle 75/25 split
        rows = []; held_rows = []
        for r in all_rows:
            ok = all(r.get(fn,"") not in ("","nan",None) for fn in SC5_FEATURES)
            if not ok: continue
            is_h = r.get("is_held_out","False") in ("True","true","1")
            if is_h: held_rows.append(r)
            else: rows.append(r)
        eps = sorted(set(r.get("run_id","") for r in rows))
        random.seed(seed); random.shuffle(eps)
        n_tr = int(len(eps) * 0.75)
        tr_set = set(eps[:n_tr]); vl_set = set(eps[n_tr:])
        tr_rows = [r for r in rows if r.get("run_id","") in tr_set]
        vl_rows = [r for r in rows if r.get("run_id","") in vl_set]
    else:
        # Canonical: frozen split column required
        if not has_split_col:
            raise ValueError(
                f"Canonical mode requires 'split' column. "
                f"Found columns: {list(fieldnames)[:15]}. "
                f"Use --legacy_random_split for old datasets without split column.")
        metadata['split_mode'] = 'frozen'
        # Route by frozen split column
        tr_rows = []; vl_rows = []; held_rows = []
        for r in all_rows:
            sp = r.get('split', '')
            if sp == 'train': tr_rows.append(r)
            elif sp == 'val': vl_rows.append(r)
            elif sp == 'held_out': held_rows.append(r)
            else:
                raise ValueError(f"Unknown split '{sp}' in row step_idx={r.get('step_idx','?')}")

        # is_held_out ↔ split biconditional: no empty/missing values allowed
        for r in tr_rows + vl_rows:
            is_h_val = r.get('is_held_out', '')
            if is_h_val not in ('False', 'false', '0'):
                raise ValueError(f"train/val row has is_held_out='{is_h_val}': "
                                 f"split={r.get('split')} episode={r.get('episode_id','?')}")
        for r in held_rows:
            is_h_val = r.get('is_held_out', '')
            if is_h_val not in ('True', 'true', '1'):
                raise ValueError(f"held_out row has is_held_out='{is_h_val}': "
                                 f"episode={r.get('episode_id','?')}")

        if not tr_rows: raise ValueError("No train rows found")
        if not vl_rows: raise ValueError("No validation rows found")
        if not held_rows: raise ValueError("No held-out rows found — strict held-out required")

        # Episode/group split consistency
        ep_splits = {}
        for r in all_rows:
            eid = r.get('episode_id', r.get('run_id', ''))
            sp = r.get('split', '')
            if eid in ep_splits and ep_splits[eid] != sp:
                raise ValueError(f"Episode {eid} has multiple splits: {ep_splits[eid]} and {sp}")
            ep_splits[eid] = sp
        # Group split consistency for both hash types
        for group_key in ['initial_state_sha256', 'trajectory_content_sha256']:
            grp_splits = {}
            for r in all_rows:
                gk = r.get(group_key, '')
                if not gk: continue
                sp = r.get('split', '')
                if gk in grp_splits and grp_splits[gk] != sp:
                    raise ValueError(f"Group {gk[:16]} ({group_key}) has multiple splits: "
                                     f"{grp_splits[gk]} and {sp}")
                grp_splits[gk] = sp

        metadata['train_eps'] = sorted(set(r.get('episode_id', r.get('run_id','')) for r in tr_rows))
        metadata['val_eps'] = sorted(set(r.get('episode_id', r.get('run_id','')) for r in vl_rows))
        metadata['heldout_eps'] = sorted(set(r.get('episode_id', r.get('run_id','')) for r in held_rows))

    # Fail-closed: reject rows with missing/invalid feature values
    MISSING_SENTINEL = -999999.0
    def _check_row(r):
        for fn in SC5_FEATURES:
            v = r.get(fn, '')
            if v in ('', 'nan', 'NaN', None):
                raise ValueError(f"Missing feature '{fn}' in row step_idx={r.get('step_idx','?')}")
            try:
                fv = float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Non-float feature '{fn}'={v} in row step_idx={r.get('step_idx','?')}")
            if math.isnan(fv) or math.isinf(fv) or abs(fv - MISSING_SENTINEL) < 1e-6:
                raise ValueError(f"Invalid feature '{fn}'={fv} in row step_idx={r.get('step_idx','?')}")

    for r in tr_rows + vl_rows + held_rows:
        _check_row(r)

    def mk(rl):
        X = np.array([[float(r[fn]) for fn in SC5_FEATURES] for r in rl], dtype=np.float32)
        yp = np.array([SC5_PHASES.index(r.get("teacher_phase","abstain_unsupported"))
                       if r.get("teacher_phase","") in SC5_PHASES else len(SC5_PHASES)-1
                       for r in rl], dtype=np.int64)
        yc = np.array([float(r.get("teacher_sc5_corridor_active",0)) for r in rl], dtype=np.float32)
        yr = np.array([float(r.get("teacher_phase","")=="release_safe") for r in rl], dtype=np.float32)
        return X, {"phase": yp, "corridor": yc, "release": yr}

    # Label validation: reject unknown teacher_phase
    for r in tr_rows + vl_rows + held_rows:
        phase = r.get('teacher_phase', '')
        if phase not in SC5_PHASES:
            raise ValueError(f"Invalid teacher_phase '{phase}' in row "
                             f"step_idx={r.get('step_idx','?')} episode={r.get('episode_id','?')}")

    Xtr, Ytr = mk(tr_rows); Xvl, Yvl = mk(vl_rows); Xte, Yte = mk(held_rows)
    return Xtr, Ytr, Xvl, Yvl, Xte, Yte, len(held_rows), len(tr_rows), len(vl_rows), metadata

def train(model, Xtr, Ytr, Xvl, Yvl, epochs=80, lr=0.001, device="cpu"):
    model = model.to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xvl, dtype=torch.float32, device=device)
    counts = Counter(Ytr["phase"].tolist())
    total = sum(counts.values())
    cw = torch.tensor([total/max(counts.get(i,1),1) for i in range(len(SC5_PHASES))],
                      dtype=torch.float32, device=device)
    pl = torch.nn.CrossEntropyLoss(weight=cw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_vl = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device); tl = 0.0; nb = 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]; xb = Xt[idx]
            yp = torch.tensor(Ytr["phase"][idx.cpu().numpy()], dtype=torch.long, device=device)
            yc = torch.tensor(Ytr["corridor"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            yr = torch.tensor(Ytr["release"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            out = model(xb)
            loss = pl(out["phase_logits"], yp) + 0.5*bce(out["corridor_logit"], yc) + 0.3*bce(out["release_logit"], yr)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            ov = model(Xv); ypv = torch.tensor(Yvl["phase"], dtype=torch.long, device=device)
            vl = pl(ov["phase_logits"], ypv).item()
            va = (ov["phase_logits"].argmax(1)==ypv).float().mean().item()
        if vl < best_vl: best_vl = vl; best_state = copy.deepcopy(model.state_dict())
        if ep % 20 == 0 or ep == epochs-1:
            print(f"  e{ep}: tr={tl/max(nb,1):.3f} vl={vl:.3f} pa={va:.3f}")
    model.load_state_dict(best_state)
    return model

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="tables/v2_sc5_canonical_dataset.csv")
ap.add_argument("--output_dir", default="outputs/sc5_canonical")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--legacy_random_split", action="store_true",
                help="Disable frozen split auto-detection; use legacy random 75/25")
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

print(f"Loading {args.dataset}...")
Xtr, Ytr, Xvl, Yvl, Xte, Yte, n_h, n_tr, n_vl, meta = load_data(
    args.dataset, args.seed, legacy_random_split=args.legacy_random_split)
print(f"  train={n_tr} val={n_vl} held_test={n_h} ({meta['split_mode']} split)")
mean = Xtr.mean(0); std = Xtr.std(0) + 1e-8
Xtr = (Xtr-mean)/std; Xvl = (Xvl-mean)/std; Xte = (Xte-mean)/std

model = SC5MLP(n_feat=len(SC5_FEATURES))
print(f"Model: {sum(p.numel() for p in model.parameters())} params")
model = train(model, Xtr, Ytr, Xvl, Yvl, device=args.device)

model.eval()
with torch.no_grad():
    Xt = torch.tensor(Xte, dtype=torch.float32, device=args.device)
    out = model(Xt); ypt = torch.tensor(Yte["phase"], dtype=torch.long, device=args.device)
    ta = (out["phase_logits"].argmax(1)==ypt).float().mean().item()
print(f"Held-out phase acc: {ta:.3f}")

out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

# Compute dataset SHA for provenance
with open(args.dataset, 'rb') as f:
    dataset_sha = hashlib.sha256(f.read()).hexdigest()

torch.save({
    "model_state": model.state_dict(), "mean": mean, "std": std,
    "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
    "dataset_sha256": dataset_sha,
    "dataset_path": args.dataset,
    "split_mode": meta['split_mode'],
    "normalization_source": "train_only",
    "n_train": n_tr, "n_val": n_vl, "n_held_test": n_h,
    "train_episode_ids": meta.get('train_eps', []),
    "val_episode_ids": meta.get('val_eps', []),
    "heldout_episode_ids": meta.get('heldout_eps', []),
}, out_dir / f"sc5_mlp_s{args.seed}.pt")
print(f"Saved to {out_dir}/sc5_mlp_s{args.seed}.pt (dataset SHA: {dataset_sha[:16]})")
