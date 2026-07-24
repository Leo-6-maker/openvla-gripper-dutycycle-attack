#!/usr/bin/env python3
"""
Generic LOTO Trainer V2 — parameterized by fold, NOT hardcoded to Fold 0.

Reads per-fold artifacts produced by build_sc5_loto_fold_v2.py.
Identical training logic to train_sc5_strict_fold0_v2.py (commit b0168a2).
"""
import csv, hashlib, json, math, os, sys, random, argparse, copy
import numpy as np; import torch
from collections import Counter, defaultdict
from pathlib import Path
from datetime import timezone, datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.sc5_detector_runtime import SC5MLP, SC5DetectorRuntime
from gripper_attack.v2_privileged_teacher import (
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
)

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]
SOURCE_COMMIT = "0280c8564773a5e6ca0482c740891d8f9eddad84"
K_SC5 = 10; GUARD_SC5 = 5


def validate_label_support(support, split_label):
    assert support["corridor_positive_rows"] > 0, "FATAL: 0 corridor-positive rows in %s" % split_label
    assert support["corridor_negative_rows"] > 0, "FATAL: 0 corridor-negative rows in %s" % split_label
    assert support["release_positive_rows"] > 0, "FATAL: 0 release-positive rows in %s" % split_label
    assert support["release_negative_rows"] > 0, "FATAL: 0 release-negative rows in %s" % split_label
    assert support["phase_unique_classes"] >= 2


def build_labels(rows, teacher_labels_raw, split_label, enforce_support_gates=True):
    ep_groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (int(r["task_idx"]), int(r["state_id"]))
        ep_groups[key].append((i, r))
    for key in ep_groups:
        ep_groups[key] = sorted(ep_groups[key], key=lambda x: int(x[1]["step"]))

    n_rows = len(rows)
    yp = np.zeros(n_rows, dtype=np.int64); yc = np.zeros(n_rows, dtype=np.float32)
    yr = np.zeros(n_rows, dtype=np.float32)
    corridor_pos = 0; release_pos = 0; corridor_audit = []

    for ep_key in sorted(ep_groups.keys()):
        t, s = ep_key; ep_rows = ep_groups[ep_key]
        row_steps = [int(r["step"]) for _, r in ep_rows]
        assert row_steps == list(range(len(row_steps))), "Non-contiguous steps in t%d s%d" % (t, s)

        ep_labels = []
        for i, r in ep_rows:
            step = int(r["step"])
            lab_key = (int(r["task_idx"]), int(r["state_id"]), step)
            lab = teacher_labels_raw.get(lab_key)
            if lab is None: raise KeyError("Missing label t%d s%d step %d" % (t, s, step))
            _ = lab["phase"]; _ = int(lab["step_idx"])
            ep_labels.append(lab)

        for idx, (i, r) in enumerate(ep_rows):
            lab = ep_labels[idx]; phase = lab["phase"]
            if phase not in SC5_PHASES: raise ValueError("Unknown phase: %s" % phase)
            yp[i] = SC5_PHASES.index(phase)
            yr[i] = 1.0 if phase == "release_safe" else 0.0
            if yr[i] > 0: release_pos += 1

        sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD_SC5)
        anchor = sc5["anchor"]; assert isinstance(anchor, int)
        if sc5["valid"]:
            assert anchor >= 0
            corridor_info = compute_sc5_valid_start_corridor(ep_labels, anchor, K=K_SC5)
            corridor_active = corridor_info["corridor_active_at_t"]
        else:
            corridor_active = set()

        ep_corr_pos = 0
        for idx, (i, r) in enumerate(ep_rows):
            step = ep_labels[idx]["step_idx"]
            if step in corridor_active: yc[i] = 1.0; corridor_pos += 1; ep_corr_pos += 1

        corridor_audit.append({"task_idx": t, "state_id": s, "n_steps": len(ep_rows),
            "sc5_valid": sc5["valid"], "sc5_anchor": anchor,
            "corridor_positive_rows": ep_corr_pos})

    support = {"total_rows": n_rows, "corridor_positive_rows": corridor_pos,
        "corridor_negative_rows": n_rows - corridor_pos,
        "release_positive_rows": release_pos, "release_negative_rows": n_rows - release_pos,
        "phase_unique_classes": len(set(yp.tolist()))}
    if enforce_support_gates: validate_label_support(support, split_label)
    return yp, yc, yr, support, corridor_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--teacher_labels", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--protocol_freeze", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.protocol_freeze) as f: protocol = json.load(f)
    training_cfg = protocol["training"]
    assert args.seed in training_cfg["seeds"], "Seed not in protocol"
    assert training_cfg["checkpoint_selection_metric"] == "phase_cross_entropy_val_only"
    epochs = int(training_cfg["epochs"]); lr = float(training_cfg["learning_rate"])
    batch_size = int(training_cfg["batch_size"])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        try: torch.use_deterministic_algorithms(True)
        except RuntimeError: pass

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load teacher labels
    teacher_labels_raw = {}
    with open(args.teacher_labels) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line); key = (lab["task_idx"], lab["state_id"], lab["step_idx"])
            if key in teacher_labels_raw: raise ValueError("Duplicate key: %s" % str(key))
            teacher_labels_raw[key] = lab

    # Load normalization
    with open(args.normalization) as f: norm = json.load(f)
    saved_mean = np.array([norm["mean"]["f_" + n] for n in SC5_FEATURES], dtype=np.float32)
    saved_std = np.array([norm["std"]["f_" + n] for n in SC5_FEATURES], dtype=np.float32)

    # Load dataset
    with open(args.dataset, "rb") as f: dataset_sha = hashlib.sha256(f.read()).hexdigest()
    all_rows = []
    with open(args.dataset) as f:
        for r in csv.DictReader(f): all_rows.append(r)

    tr_rows = [r for r in all_rows if r["split"] == "train"]
    vl_rows = [r for r in all_rows if r["split"] == "val"]
    te_rows = [r for r in all_rows if r["split"] not in ("train", "val")]
    assert len(te_rows) == 0, "FATAL: %d held-out rows!" % len(te_rows)

    tr_eps = set((int(r["task_idx"]), int(r["state_id"])) for r in tr_rows)
    vl_eps = set((int(r["task_idx"]), int(r["state_id"])) for r in vl_rows)
    assert len(tr_eps & vl_eps) == 0
    print("Train: %d rows %d eps  Val: %d rows %d eps  Test rows: 0" % (
        len(tr_rows), len(tr_eps), len(vl_rows), len(vl_eps)))

    # Build labels
    Yp_tr, Yc_tr, Yr_tr, tr_sup, tr_audit = build_labels(tr_rows, teacher_labels_raw, "train")
    Yp_vl, Yc_vl, Yr_vl, vl_sup, vl_audit = build_labels(vl_rows, teacher_labels_raw, "val")
    print("Train: corr_pos=%d corr_neg=%d  Val: corr_pos=%d corr_neg=%d" % (
        tr_sup["corridor_positive_rows"], tr_sup["corridor_negative_rows"],
        vl_sup["corridor_positive_rows"], vl_sup["corridor_negative_rows"]))

    # Features
    def extract_X(rows):
        X = np.zeros((len(rows), 25), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, name in enumerate(SC5_FEATURES):
                v = float(r["f_" + name])
                assert not (math.isnan(v) or math.isinf(v))
                X[i, j] = v
        return X
    Xtr = extract_X(tr_rows); Xvl = extract_X(vl_rows)

    # Norm parity
    rmean = Xtr.astype(np.float64).mean(0); rstd = Xtr.astype(np.float64).std(0)
    assert np.abs(saved_mean - rmean).max() < 1e-4 and np.abs(saved_std - rstd).max() < 1e-2
    saved_std_safe = np.maximum(saved_std, 1e-8)
    Xtr_n = (Xtr - saved_mean) / saved_std_safe
    Xvl_n = (Xvl - saved_mean) / saved_std_safe

    # Training
    model = SC5MLP(n_feat=25).to(device)
    Xt_t = torch.tensor(Xtr_n, dtype=torch.float32, device=device)
    Xv_t = torch.tensor(Xvl_n, dtype=torch.float32, device=device)
    counts = Counter(Yp_tr.tolist()); total = sum(counts.values())
    cw = torch.tensor([total/max(counts.get(i,1),1) for i in range(len(SC5_PHASES))],
                      dtype=torch.float32, device=device)
    pl = torch.nn.CrossEntropyLoss(weight=cw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator(device=device); g.manual_seed(args.seed)

    best_vl = float("inf"); best_state = None; best_epoch = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_n), generator=g, device=device)
        tl = 0.0; nb = 0
        for i in range(0, len(Xtr_n), batch_size):
            idx = perm[i:i+batch_size]; idx_np = idx.cpu().numpy()
            xb = Xt_t[idx]
            yp_b = torch.tensor(Yp_tr[idx_np], dtype=torch.long, device=device)
            yc_b = torch.tensor(Yc_tr[idx_np], dtype=torch.float32, device=device).unsqueeze(1)
            yr_b = torch.tensor(Yr_tr[idx_np], dtype=torch.float32, device=device).unsqueeze(1)
            out = model(xb)
            loss = pl(out["phase_logits"], yp_b) + 0.5*bce(out["corridor_logit"], yc_b) + 0.3*bce(out["release_logit"], yr_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            ov = model(Xv_t)
            ypv = torch.tensor(Yp_vl, dtype=torch.long, device=device)
            vl_loss = pl(ov["phase_logits"], ypv).item()
        if vl_loss < best_vl: best_vl = vl_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = ep
        if ep % 20 == 0 or ep == epochs-1: print("  e%d: tr=%.3f vl=%.3f" % (ep, tl/max(nb,1), vl_loss))

    model.load_state_dict(best_state)

    # SHAs
    with open(args.teacher_labels, "rb") as f: tl_sha = hashlib.sha256(f.read()).hexdigest()
    with open(args.normalization, "rb") as f: nm_sha = hashlib.sha256(f.read()).hexdigest()

    # Atomic checkpoint
    ckpt = {"model_state": best_state, "mean": saved_mean, "std": saved_std_safe,
        "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
        "split_mode": "frozen", "dataset_sha256": dataset_sha,
        "normalization_sha256": nm_sha, "teacher_labels_sha256": tl_sha,
        "seed": args.seed, "best_epoch": best_epoch, "best_val_phase_loss": best_vl,
        "selection_metric": "phase_cross_entropy_val_only",
        "n_train_rows": len(tr_rows), "n_val_rows": len(vl_rows),
        "label_support": {"train": tr_sup, "val": vl_sup},
        "test_accessed": False, "source_commit": SOURCE_COMMIT}

    tmp_path = out_dir / "best_model.unvalidated.pt"
    final_path = out_dir / "best_model.pt"
    torch.save(ckpt, tmp_path)

    rt = SC5DetectorRuntime(str(tmp_path), tau_corridor=0.3, tau_release=0.3, guard=GUARD_SC5)
    rt_state = rt.model.state_dict()
    assert max((best_state[k] - rt_state[k].cpu()).abs().max().item() for k in best_state) < 1e-12
    os.replace(tmp_path, final_path)
    print("  Runtime: PASS  Saved: %s" % final_path)
    print("Best epoch: %d  Val loss: %.4f" % (best_epoch, best_vl))

if __name__ == "__main__":
    main()
