#!/usr/bin/env python3
"""
Strict Fold 0 Test-Blind Trainer V2 — CORRIDOR LABEL FIX (TRAINING_AMENDMENT_02).

P0-1 FIX: runtime checkpoint now includes dataset_sha256 (saved BEFORE rt check).
P0-2 FIX: build_labels(enforce_support_gates=False) for unit-test single episodes.
P1-1 FIX: assert teacher labels contain only train+val tasks, detect duplicate keys.
P1-2 FIX: read PROTOCOL_FREEZE.json, enforce seeds/epochs/lr/batch/metric.
P1-3 FIX: sort episode rows by step before processing.
"""
import csv, hashlib, json, math, os, sys, random, argparse, copy
import numpy as np
import torch
from collections import Counter, defaultdict
from pathlib import Path
from datetime import timezone, datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.sc5_detector_runtime import SC5MLP, SC5DetectorRuntime
from gripper_attack.v2_privileged_teacher import (
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
)

# ═══════════════════════════════════════════════
# Frozen constants
# ═══════════════════════════════════════════════
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
TRAIN_TASKS = {0,1,2,3,4,5,7,9}
VAL_TASKS = {6}
TEST_TASK = 8
SOURCE_COMMIT = "0280c8564773a5e6ca0482c740891d8f9eddad84"
K_SC5 = 10
GUARD_SC5 = 5


def validate_label_support(support, split_label):
    """Fail-closed gates on label support. Called only for full train/val datasets."""
    assert support["corridor_positive_rows"] > 0, \
        "FATAL: 0 corridor-positive rows in %s" % split_label
    assert support["corridor_negative_rows"] > 0, \
        "FATAL: 0 corridor-negative rows in %s" % split_label
    assert support["release_positive_rows"] > 0, \
        "FATAL: 0 release-positive rows in %s" % split_label
    assert support["release_negative_rows"] > 0, \
        "FATAL: 0 release-negative rows in %s" % split_label
    assert support["phase_unique_classes"] >= 2, \
        "FATAL: only %d unique phases in %s" % (support["phase_unique_classes"], split_label)


def build_labels(rows, teacher_labels_raw, split_label, enforce_support_gates=True):
    """Build phase/corridor/release labels using frozen SC5 protocol.

    P0-2 FIX: enforce_support_gates=False allows single-episode testing.
    P1-3 FIX: rows are sorted by step within each episode.
    """
    # Group rows by episode, SORTED by step (P1-3)
    ep_groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (int(r["task_idx"]), int(r["state_id"]))
        ep_groups[key].append((i, r))
    for key in ep_groups:
        ep_groups[key] = sorted(ep_groups[key], key=lambda x: int(x[1]["step"]))

    n_rows = len(rows)
    yp = np.zeros(n_rows, dtype=np.int64)
    yc = np.zeros(n_rows, dtype=np.float32)
    yr = np.zeros(n_rows, dtype=np.float32)

    corridor_pos = 0; release_pos = 0
    corridor_audit = []

    for ep_key in sorted(ep_groups.keys()):
        t, s = ep_key
        ep_rows = ep_groups[ep_key]  # already sorted by step
        split = "train" if t in TRAIN_TASKS else ("val" if t in VAL_TASKS else "test")

        # ── Step continuity check on sorted rows (P1-3) ──
        row_steps = [int(r["step"]) for _, r in ep_rows]
        assert row_steps == list(range(len(row_steps))), \
            "Non-contiguous or unsorted steps in t%d s%d: %s" % (t, s, str(row_steps[:10]))

        # ── Collect teacher labels (FAIL-CLOSED) ──
        ep_labels = []
        for i, r in ep_rows:
            step = int(r["step"])
            lab_key = (int(r["task_idx"]), int(r["state_id"]), step)
            lab = teacher_labels_raw.get(lab_key)
            if lab is None:
                raise KeyError(
                    "Missing teacher label: t%d s%d step %d in %s" % (t, s, step, split_label))
            _ = lab["phase"]
            _ = int(lab["step_idx"])
            ep_labels.append(lab)

        # ── Phase and release labels ──
        for idx, (i, r) in enumerate(ep_rows):
            lab = ep_labels[idx]
            phase = lab["phase"]
            if phase not in SC5_PHASES:
                raise ValueError("Unknown phase '%s' in t%d s%d" % (phase, t, s))
            yp[i] = SC5_PHASES.index(phase)
            is_release = (phase == "release_safe")
            yr[i] = 1.0 if is_release else 0.0
            if is_release: release_pos += 1

        # ── Corridor labels: frozen SC5 protocol ──
        sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD_SC5)
        anchor = sc5["anchor"]
        assert isinstance(anchor, int), \
            "anchor must be int, got %s" % type(anchor)

        if sc5["valid"]:
            assert anchor >= 0
            corridor_info = compute_sc5_valid_start_corridor(ep_labels, anchor, K=K_SC5)
            corridor_active = corridor_info["corridor_active_at_t"]
        else:
            assert anchor == -1 or sc5["reason"] != ""
            corridor_active = set()

        ep_corridor_pos = 0
        for idx, (i, r) in enumerate(ep_rows):
            step = ep_labels[idx]["step_idx"]
            if step in corridor_active:
                yc[i] = 1.0; corridor_pos += 1; ep_corridor_pos += 1

        corridor_audit.append({
            "task_idx": t, "state_id": s, "split": split,
            "n_steps": len(ep_rows),
            "stable_carry_start": sc5.get("stable_carry_start", -1),
            "sc5_anchor": anchor, "sc5_valid": sc5["valid"],
            "sc5_reason": sc5.get("reason", ""),
            "corridor_positive_rows": ep_corridor_pos,
        })

    corridor_neg = n_rows - corridor_pos
    release_neg = n_rows - release_pos

    support = {
        "total_rows": n_rows,
        "corridor_positive_rows": corridor_pos,
        "corridor_negative_rows": corridor_neg,
        "release_positive_rows": release_pos,
        "release_negative_rows": release_neg,
        "phase_unique_classes": len(set(yp.tolist())),
    }

    if enforce_support_gates:
        validate_label_support(support, split_label)

    return yp, yc, yr, support, corridor_audit


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--teacher_labels", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--protocol_freeze", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label_audit_only", action="store_true",
                    help="Build labels and report support, skip training")
    args = ap.parse_args()

    # ── P1-2: Read and enforce PROTOCOL_FREEZE.json ──
    with open(args.protocol_freeze) as f:
        protocol = json.load(f)
    protocol_sha = hashlib.sha256(open(args.protocol_freeze, "rb").read()).hexdigest()
    training_cfg = protocol.get("training", {})
    runtime_cfg = protocol.get("runtime", {})

    assert args.seed in training_cfg.get("seeds", []), \
        "Seed %d not in protocol freeze seeds %s" % (args.seed, training_cfg.get("seeds"))
    assert training_cfg.get("checkpoint_selection_metric") == "phase_cross_entropy_val_only", \
        "Selection metric mismatch"
    assert runtime_cfg.get("tau_corridor") == 0.3
    assert runtime_cfg.get("tau_release") == 0.3
    assert runtime_cfg.get("guard") == GUARD_SC5
    assert runtime_cfg.get("K") == K_SC5

    epochs = int(training_cfg["epochs"])
    lr = float(training_cfg["learning_rate"])
    batch_size = int(training_cfg["batch_size"])
    print("Protocol freeze enforced: seeds=%s epochs=%d lr=%.4f batch=%d metric=%s" % (
        training_cfg.get("seeds"), epochs, lr, batch_size, training_cfg.get("checkpoint_selection_metric")))

    # Determinism
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try: torch.use_deterministic_algorithms(True)
        except RuntimeError: pass

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── P1-1: Load teacher labels, assert train/val only, detect duplicates ──
    print("Loading teacher labels: %s" % args.teacher_labels)
    teacher_labels_raw = {}
    label_tasks = set()
    with open(args.teacher_labels) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line)
            key = (lab["task_idx"], lab["state_id"], lab["step_idx"])
            if key in teacher_labels_raw:
                raise ValueError("Duplicate teacher label key: %s" % str(key))
            teacher_labels_raw[key] = lab
            label_tasks.add(lab["task_idx"])
    print("  %d label rows, tasks=%s" % (len(teacher_labels_raw), sorted(label_tasks)))
    assert label_tasks == (TRAIN_TASKS | VAL_TASKS), \
        "Label tasks %s != expected train+val %s" % (sorted(label_tasks), sorted(TRAIN_TASKS | VAL_TASKS))
    assert TEST_TASK not in label_tasks, "Test task %d in teacher labels!" % TEST_TASK

    # ── Load normalization ──
    with open(args.normalization) as f:
        norm = json.load(f)
    saved_mean = np.array([norm["mean"]["f_" + n] for n in SC5_FEATURES], dtype=np.float32)
    saved_std = np.array([norm["std"]["f_" + n] for n in SC5_FEATURES], dtype=np.float32)

    # ── Load feature dataset ──
    print("Loading dataset: %s" % args.dataset)
    with open(args.dataset, "rb") as f:
        dataset_sha = hashlib.sha256(f.read()).hexdigest()
    all_rows = []
    with open(args.dataset) as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
    print("  %d rows, SHA256: %s" % (len(all_rows), dataset_sha[:16]))

    # ── Split ──
    tr_rows = [r for r in all_rows if r["split"] == "train"]
    vl_rows = [r for r in all_rows if r["split"] == "val"]
    te_rows = [r for r in all_rows if r["split"] not in ("train", "val")]
    assert len(te_rows) == 0, "FATAL: %d held-out rows in train+val dataset!" % len(te_rows)
    tr_tasks = set(int(r["task_idx"]) for r in tr_rows)
    vl_tasks = set(int(r["task_idx"]) for r in vl_rows)
    assert tr_tasks == TRAIN_TASKS, "Train task mismatch: %s" % tr_tasks
    assert vl_tasks == VAL_TASKS, "Val task mismatch: %s" % vl_tasks
    tr_eps = set((int(r["task_idx"]), int(r["state_id"])) for r in tr_rows)
    vl_eps = set((int(r["task_idx"]), int(r["state_id"])) for r in vl_rows)
    assert len(tr_eps & vl_eps) == 0
    assert len(tr_eps) == 400 and len(vl_eps) == 50

    print("  Train: %d rows, %d eps  Val: %d rows, %d eps  Test rows: 0" % (
        len(tr_rows), len(tr_eps), len(vl_rows), len(vl_eps)))

    # Key-set equality: every feature row must have exactly one teacher label
    dataset_keys = {(int(r["task_idx"]), int(r["state_id"]), int(r["step"])) for r in all_rows}
    label_keys = set(teacher_labels_raw.keys())
    assert label_keys == dataset_keys, (
        "Key-set mismatch: extra_labels=%d missing_labels=%d" % (
            len(label_keys - dataset_keys), len(dataset_keys - label_keys)))

    # ── Build labels ──
    print("\nBuilding labels (SC5 corridor protocol)...")
    Yp_tr, Yc_tr, Yr_tr, train_support, train_audit = \
        build_labels(tr_rows, teacher_labels_raw, "train", enforce_support_gates=True)
    Yp_vl, Yc_vl, Yr_vl, val_support, val_audit = \
        build_labels(vl_rows, teacher_labels_raw, "val", enforce_support_gates=True)

    print("  Train: corr_pos=%d corr_neg=%d rel_pos=%d rel_neg=%d phases=%d" % (
        train_support["corridor_positive_rows"], train_support["corridor_negative_rows"],
        train_support["release_positive_rows"], train_support["release_negative_rows"],
        train_support["phase_unique_classes"]))
    print("  Val:   corr_pos=%d corr_neg=%d rel_pos=%d rel_neg=%d phases=%d" % (
        val_support["corridor_positive_rows"], val_support["corridor_negative_rows"],
        val_support["release_positive_rows"], val_support["release_negative_rows"],
        val_support["phase_unique_classes"]))

    # Per-task audit
    task_corridor = defaultdict(lambda: {"eps":0,"corridor_eps":0,"corridor_rows":0,"total_rows":0})
    for rec in train_audit + val_audit:
        tk = rec["task_idx"]
        task_corridor[tk]["eps"] += 1; task_corridor[tk]["total_rows"] += rec["n_steps"]
        if rec["corridor_positive_rows"] > 0:
            task_corridor[tk]["corridor_eps"] += 1
            task_corridor[tk]["corridor_rows"] += rec["corridor_positive_rows"]
    print("\n  Per-task corridor:")
    for tk in sorted(task_corridor):
        tc = task_corridor[tk]
        print("    task %d: %d eps, %d corridor_eps, %d corridor_rows/%d rows" % (
            tk, tc["eps"], tc["corridor_eps"], tc["corridor_rows"], tc["total_rows"]))

    if args.label_audit_only:
        print("\nLabel audit complete. Skipping training (--label_audit_only).")
        return

    # ── Extract features ──
    print("\nExtracting features...")
    def extract_X(rows):
        X = np.zeros((len(rows), 25), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, name in enumerate(SC5_FEATURES):
                v = float(r["f_" + name])
                assert not (math.isnan(v) or math.isinf(v))
                X[i, j] = v
        return X
    Xtr = extract_X(tr_rows); Xvl = extract_X(vl_rows)

    # ── Norm parity ──
    rmean = Xtr.astype(np.float64).mean(0); rstd = Xtr.astype(np.float64).std(0)
    assert np.abs(saved_mean - rmean).max() < 1e-4 and np.abs(saved_std - rstd).max() < 1e-2
    saved_std_safe = np.maximum(saved_std, 1e-8)
    Xtr_n = (Xtr - saved_mean) / saved_std_safe
    Xvl_n = (Xvl - saved_mean) / saved_std_safe

    # ── Training ──
    print("\nTraining (seed=%d epochs=%d lr=%.4f batch=%d)..." % (args.seed, epochs, lr, batch_size))
    model = SC5MLP(n_feat=25).to(device)
    print("  Params: %d" % sum(p.numel() for p in model.parameters()))
    Xt_t = torch.tensor(Xtr_n, dtype=torch.float32, device=device)
    Xv_t = torch.tensor(Xvl_n, dtype=torch.float32, device=device)

    counts = Counter(Yp_tr.tolist()); total = sum(counts.values())
    cw = torch.tensor([total/max(counts.get(i,1),1) for i in range(len(SC5_PHASES))],
                      dtype=torch.float32, device=device)
    pl = torch.nn.CrossEntropyLoss(weight=cw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator(device=device); g.manual_seed(args.seed)

    best_vl = float("inf"); best_state = None; best_epoch = 0; epoch_metrics = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_n), generator=g, device=device)
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
            vl_phase_loss = pl(ov["phase_logits"], ypv).item()
            vl_phase_acc = (ov["phase_logits"].argmax(1)==ypv).float().mean().item()

        epoch_metrics.append({"epoch":ep,"train_loss":tl/max(nb,1),
                              "val_phase_loss":vl_phase_loss,"val_phase_acc":float(vl_phase_acc)})
        if vl_phase_loss < best_vl:
            best_vl = vl_phase_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
        if ep % 20 == 0 or ep == epochs-1:
            print("  e%d: tr=%.3f vl=%.3f acc=%.3f" % (ep, tl/max(nb,1), vl_phase_loss, vl_phase_acc))

    model.load_state_dict(best_state)
    print("  Best: epoch=%d vl=%.4f" % (best_epoch, best_vl))

    # ── SHA computations ──
    with open(args.teacher_labels, "rb") as f: tl_sha = hashlib.sha256(f.read()).hexdigest()
    with open(args.normalization, "rb") as f: nm_sha = hashlib.sha256(f.read()).hexdigest()
    training_script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()

    # ── Atomic checkpoint: save to .unvalidated.pt, validate, then rename ──
    ckpt = {
        "model_state": best_state, "mean": saved_mean, "std": saved_std_safe,
        "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
        "split_mode": "frozen",
        "dataset_sha256": dataset_sha,
        "normalization_sha256": nm_sha,
        "teacher_labels_sha256": tl_sha,
        "protocol_freeze_sha256": protocol_sha,
        "training_script_sha256": training_script_sha,
        "source_commit": SOURCE_COMMIT,
        "seed": args.seed, "best_epoch": best_epoch,
        "best_val_phase_loss": best_vl,
        "selection_metric": "phase_cross_entropy_val_only",
        "n_train_rows": len(tr_rows), "n_val_rows": len(vl_rows),
        "n_train_episodes": len(tr_eps), "n_val_episodes": len(vl_eps),
        "label_support": {"train": train_support, "val": val_support},
        "test_accessed": False,
        "training_version": "V2_corridor_label_fixed",
    }
    tmp_path = out_dir / "best_model.unvalidated.pt"
    final_path = out_dir / "best_model.pt"
    torch.save(ckpt, tmp_path)

    # Runtime validation
    rt = SC5DetectorRuntime(str(tmp_path), tau_corridor=0.3, tau_release=0.3, guard=GUARD_SC5)
    rt_state = rt.model.state_dict()
    max_sd_err = max((best_state[k] - rt_state[k].cpu()).abs().max().item() for k in best_state)
    assert max_sd_err < 1e-12, "Runtime load FAIL: sd_err=%.2e" % max_sd_err
    print("  Runtime load: PASS (sd_err=%.2e)" % max_sd_err)

    os.replace(tmp_path, final_path)
    print("  Saved: %s" % final_path)

    # ── Outputs ──
    with open(out_dir / "TRAIN_EPOCH_METRICS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_phase_loss","val_phase_acc"])
        w.writeheader(); w.writerows(epoch_metrics)
    with open(out_dir / "LABEL_SUPPORT.json", "w") as f:
        json.dump({"train": train_support, "val": val_support}, f, indent=2)
    with open(out_dir / "CORRIDOR_LABEL_AUDIT.json", "w") as f:
        json.dump({"train": train_audit, "val": val_audit}, f, indent=2)
    with open(out_dir / "CORRIDOR_LABEL_BY_TASK.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_idx","eps","corridor_eps","corridor_rows","total_rows","corridor_rate"])
        for tk in sorted(task_corridor):
            tc = task_corridor[tk]
            w.writerow([tk,tc["eps"],tc["corridor_eps"],tc["corridor_rows"],
                       tc["total_rows"],tc["corridor_rows"]/max(tc["total_rows"],1)])
    with open(out_dir / "VAL_ONLY_METRICS.json", "w") as f:
        json.dump({"seed":args.seed,"best_epoch":best_epoch,"best_val_phase_loss":best_vl,
                   "selection_metric":"phase_cross_entropy_val_only",
                   "test_metrics_generated":False,"heldout_file_open_count":0}, f, indent=2)
    with open(out_dir / "SHA256SUMS.txt", "w") as f:
        for fn in ["best_model.pt","TRAIN_EPOCH_METRICS.csv","LABEL_SUPPORT.json",
                   "CORRIDOR_LABEL_AUDIT.json","CORRIDOR_LABEL_BY_TASK.csv"]:
            fp = out_dir / fn
            if fp.exists():
                f.write("%s  %s\n" % (hashlib.sha256(fp.read_bytes()).hexdigest(), fn))

    print("\n=== TRAINING COMPLETE (V2) ===")
    print("Seed: %d  Best epoch: %d  Val phase loss: %.4f" % (args.seed, best_epoch, best_vl))

if __name__ == "__main__":
    main()
