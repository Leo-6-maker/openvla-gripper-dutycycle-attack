#!/usr/bin/env python3
"""Train SC5 student MLP — thin wrapper reusing ProprioCausalMLP and training loop.

Reuses: src/utils/proprio_causal_student.py (ProprioCausalMLP, split logic, normalization)
New: SC5-specific features (no normalized_step), corridor/release heads, sc5 dataset.
"""
import csv, json, os, sys, random, numpy as np, torch
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

# Reuse existing MLP and utilities
from src.utils.proprio_causal_student import ProprioCausalMLP as _BaseMLP, assign_splits

# --- SC5-specific feature list (NO normalized_step) ---
SC5_NUMERIC_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

# Forbidden features (must NOT be in model input)
FORBIDDEN_SUBSTRINGS = [
    "normalized_step", "step_idx", "state_id", "task_name",
    "run_id", "episode_key", "teacher_", "eval_",
    "is_butter", "is_held_out", "mechanism_type", "parse_confidence",
    "object_pose", "target_pose", "object_to_target", "attack", "manual",
]

# --- SC5 phase mapping (aligned with Teacher) ---
SC5_PHASE_CLASSES = [
    "approach", "grasp_close", "stable_grasp", "first_lift",
    "stable_carry", "pre_place_unsupported", "release_safe",
    "recovery_or_regrasp", "abstain_unsupported",
]


class SC5ProprioMLP(torch.nn.Module):
    """MLP with shared backbone + SC5 heads: phase, corridor, release_safe, confidence."""

    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
        )
        self.phase_head = torch.nn.Linear(hidden_dim, len(SC5_PHASE_CLASSES))
        self.corridor_head = torch.nn.Linear(hidden_dim, 1)  # SC5 corridor active
        self.release_head = torch.nn.Linear(hidden_dim, 1)  # release_safe
        self.confidence_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "phase_logits": self.phase_head(h),
            "corridor_logit": self.corridor_head(h),
            "release_logit": self.release_head(h),
            "confidence_logit": self.confidence_head(h),
        }


def load_dataset(csv_path: str):
    """Load SC5 dataset, filter valid feature rows, return X, y dict, meta."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Filter: only rows with all required numeric features present
    valid_rows = []
    for r in rows:
        ok = True
        for fn in SC5_NUMERIC_FEATURES:
            v = r.get(fn, "")
            if v == "" or v is None:
                ok = False; break
        if ok:
            valid_rows.append(r)

    X = np.array([[float(r[fn]) for fn in SC5_NUMERIC_FEATURES] for r in valid_rows], dtype=np.float32)

    # Labels
    y_phase = np.array([SC5_PHASE_CLASSES.index(r.get("teacher_phase", "abstain_unsupported"))
                        if r.get("teacher_phase", "abstain_unsupported") in SC5_PHASE_CLASSES
                        else len(SC5_PHASE_CLASSES) - 1 for r in valid_rows], dtype=np.int64)
    y_corridor = np.array([float(r.get("teacher_sc5_ready", 0)) for r in valid_rows], dtype=np.float32)
    y_release = np.array([float(r.get("teacher_release_safe", 0)) if "teacher_release_safe" in r
                          else float(r.get("teacher_phase", "") == "release_safe") for r in valid_rows], dtype=np.float32)
    # Confidence: 1.0 for stable_carry/release_safe, 0.5 otherwise
    y_conf = np.array([1.0 if r.get("teacher_phase", "") in ("stable_carry", "release_safe", "stable_grasp")
                       else 0.5 for r in valid_rows], dtype=np.float32)

    # Episode keys for split
    episode_keys = [r.get("run_id", r.get("task_name", "unknown")) for r in valid_rows]

    meta = {"n_rows": len(valid_rows), "n_features": len(SC5_NUMERIC_FEATURES),
            "feature_names": SC5_NUMERIC_FEATURES,
            "phase_distribution": dict(Counter(y_phase.tolist()))}
    return X, {"phase": y_phase, "corridor": y_corridor, "release": y_release, "confidence": y_conf}, episode_keys, meta


def train_model(model, X_train, y_train, X_val, y_val, epochs=80, lr=0.001, batch_size=64, device="cpu"):
    """Train with multi-head loss."""
    model = model.to(device)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    # Phase class weights (inverse frequency)
    phase_counts = Counter(y_train["phase"].tolist())
    total = sum(phase_counts.values())
    class_weights = torch.tensor(
        [total / max(phase_counts.get(i, 1), 1) for i in range(len(SC5_PHASE_CLASSES))],
        dtype=torch.float32, device=device)

    phase_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))  # imbalance

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n_train = X_train.shape[0]
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0; n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            yp = torch.tensor(y_train["phase"][idx.cpu().numpy()], dtype=torch.long, device=device)
            yc = torch.tensor(y_train["corridor"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            yr = torch.tensor(y_train["release"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            yf = torch.tensor(y_train["confidence"][idx.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)

            out = model(xb)
            loss = (phase_loss_fn(out["phase_logits"], yp) +
                    0.5 * bce_loss_fn(out["corridor_logit"], yc) +
                    0.3 * bce_loss_fn(out["release_logit"], yr) +
                    0.1 * torch.nn.functional.mse_loss(torch.sigmoid(out["confidence_logit"]), yf))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        # Val
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            ypv = torch.tensor(y_val["phase"], dtype=torch.long, device=device)
            val_loss = phase_loss_fn(out_v["phase_logits"], ypv).item()
            val_acc = (out_v["phase_logits"].argmax(1) == ypv).float().mean().item()
            # Corridor: only evaluate on steps where SC5 is actually valid
            corridor_mask = torch.tensor(
                [r.get("teacher_sc5_corridor_valid", "0") == "1" for r in
                 [dict(zip(SC5_NUMERIC_FEATURES, X_val[j])) for j in range(len(X_val))]],
                dtype=torch.bool, device=device)
            if corridor_mask.sum() > 0:
                ycv = torch.tensor(y_val["corridor"], dtype=torch.float32, device=device).unsqueeze(1)
                corridor_acc = ((torch.sigmoid(out_v["corridor_logit"]) > 0.5) == (ycv > 0.5)).float()[corridor_mask.squeeze(1)].mean().item()
            else:
                corridor_acc = 0.0

        history.append({"epoch": epoch, "train_loss": total_loss / max(n_batches, 1),
                        "val_loss": val_loss, "val_phase_acc": val_acc, "val_corridor_acc": corridor_acc})
        if val_loss < best_val_loss: best_val_loss = val_loss

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: train_loss={total_loss/max(n_batches,1):.4f} val_loss={val_loss:.4f} phase_acc={val_acc:.3f} corridor_acc={corridor_acc:.3f}")

    return history


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tables/v2_sc5_student_dataset.csv")
    ap.add_argument("--output_dir", default="outputs/sc5_student_v2")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda": torch.cuda.manual_seed_all(args.seed)

    print(f"Loading {args.dataset}...")
    X, y, episode_keys, meta = load_dataset(args.dataset)
    print(f"  {meta['n_rows']} rows, {meta['n_features']} features")
    print(f"  phase dist: {meta['phase_distribution']}")

    # Split by episode key (not by row)
    unique_eps = sorted(set(episode_keys))
    random.shuffle(unique_eps)
    n_train = int(len(unique_eps) * 0.70)
    n_val = int(len(unique_eps) * 0.15)
    train_eps = set(unique_eps[:n_train])
    val_eps = set(unique_eps[n_train:n_train + n_val])
    test_eps = set(unique_eps[n_train + n_val:])

    train_idx = [i for i, ek in enumerate(episode_keys) if ek in train_eps]
    val_idx = [i for i, ek in enumerate(episode_keys) if ek in val_eps]
    test_idx = [i for i, ek in enumerate(episode_keys) if ek in test_eps]
    print(f"  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    X_train, y_train = X[train_idx], {k: v[train_idx] for k, v in y.items()}
    X_val, y_val = X[val_idx], {k: v[val_idx] for k, v in y.items()}

    # Normalize (train stats only)
    mean = X_train.mean(0); std = X_train.std(0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    model = SC5ProprioMLP(n_features=len(SC5_NUMERIC_FEATURES))
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=args.epochs, lr=args.lr, device=args.device)

    # Save
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "feature_names": SC5_NUMERIC_FEATURES, "phase_classes": SC5_PHASE_CLASSES,
                "history": history, "config": vars(args)},
               out / "sc5_student_mlp.pt")

    # Final metrics
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor((X[test_idx] - mean) / std, dtype=torch.float32, device=args.device)
        out_t = model(X_test)
        ypt = torch.tensor(y["phase"][test_idx], dtype=torch.long, device=args.device)
        test_acc = (out_t["phase_logits"].argmax(1) == ypt).float().mean().item()
    print(f"Test phase accuracy: {test_acc:.3f}")
    print(f"Model saved to {out / 'sc5_student_mlp.pt'}")


if __name__ == "__main__":
    main()
