#!/usr/bin/env python3
"""SC5-V2 trainer — frozen protocol, multi-task loss, train/val split, checkpoint export."""
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
    def __init__(self, n_feat=25, hidden=64):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(n_feat, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(hidden, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(hidden, 1)
        self.release_head = torch.nn.Linear(hidden, 1)
        self.confidence_head = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h), "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h), "confidence_logit": self.confidence_head(h)}

EXPECTED_DATASET_SHA = "a3f9a388e94e7fab295160b11ebaa811cbd3fc4e8e254f7a909a28580e696bc2"
EXPECTED_TRAIN_STEPS = 59773
EXPECTED_VAL_STEPS = 24242
EXPECTED_TOTAL_STEPS = 84015
EXPECTED_TRAIN_EPS = 280
EXPECTED_VAL_EPS = 90

def load_data(csv_path, expected_sha=None, exp_train_steps=None, exp_val_steps=None,
              exp_train_eps=None, exp_val_eps=None):
    """Load SC5-V2 step dataset. Returns (Xtr,Ytr,Xvl,Yvl,n_tr,n_vl,metadata).
    If expected_* parameters are None, uses hardcoded defaults (full dataset).
    """
    actual_sha = hashlib.sha256(open(csv_path,"rb").read()).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError("DATASET_SHA_MISMATCH: actual=%s expected=%s" % (actual_sha, expected_sha))

    all_rows = list(csv.DictReader(open(csv_path)))
    if 'split' not in (all_rows[0].keys() if all_rows else []):
        raise ValueError("Dataset missing 'split' column")

    VALID_SPLITS = {'train', 'val'}
    tr_rows = []; vl_rows = []
    for r in all_rows:
        sp = r.get('split','')
        if sp not in VALID_SPLITS:
            raise ValueError("Unknown split '%s' in row step_idx=%s" % (sp, r.get('step_idx','?')))
        eid = r.get('episode_id','')
        if not eid:
            raise ValueError("Empty episode_id in row step_idx=%s" % r.get('step_idx','?'))
        if sp == 'train': tr_rows.append(r)
        else: vl_rows.append(r)

    if not tr_rows: raise ValueError("No train rows")
    if not vl_rows: raise ValueError("No val rows")

    _etr = exp_train_steps if exp_train_steps is not None else EXPECTED_TRAIN_STEPS
    _evl = exp_val_steps if exp_val_steps is not None else EXPECTED_VAL_STEPS
    if len(tr_rows) != _etr:
        raise RuntimeError("TRAIN_STEP_COUNT_MISMATCH: %d != %d" % (len(tr_rows), _etr))
    if len(vl_rows) != _evl:
        raise RuntimeError("VAL_STEP_COUNT_MISMATCH: %d != %d" % (len(vl_rows), _evl))

    # Fail-closed: check all features present and valid
    for r in tr_rows + vl_rows:
        for fn in SC5_FEATURES:
            v = r.get(fn, '')
            if v in ('', 'nan', 'NaN', None):
                raise ValueError("Missing feature '%s' in row step_idx=%s" % (fn, r.get('step_idx','?')))
            try:
                fv = float(v)
            except (ValueError, TypeError):
                raise ValueError("Non-float feature '%s'=%s" % (fn, v))
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError("Invalid feature '%s'=%s" % (fn, v))
        phase = r.get('teacher_phase','')
        if phase not in SC5_PHASES:
            raise ValueError("Invalid teacher_phase '%s'" % phase)

    # Episode consistency
    ep_splits = {}
    for r in all_rows:
        eid = r.get('episode_id','')
        sp = r.get('split','')
        if eid in ep_splits and ep_splits[eid] != sp:
            raise ValueError("Episode %s has multiple splits" % eid)
        ep_splits[eid] = sp

    tr_eps = sorted(set(r['episode_id'] for r in tr_rows))
    vl_eps = sorted(set(r['episode_id'] for r in vl_rows))

    _etr_eps = exp_train_eps if exp_train_eps is not None else EXPECTED_TRAIN_EPS
    _evl_eps = exp_val_eps if exp_val_eps is not None else EXPECTED_VAL_EPS
    if len(tr_eps) != _etr_eps:
        raise RuntimeError("TRAIN_EPISODE_COUNT_MISMATCH: %d != %d" % (len(tr_eps), _etr_eps))
    if len(vl_eps) != _evl_eps:
        raise RuntimeError("VAL_EPISODE_COUNT_MISMATCH: %d != %d" % (len(vl_eps), _evl_eps))

    def mk(rl):
        X = np.array([[float(r[fn]) for fn in SC5_FEATURES] for r in rl], dtype=np.float32)
        yp = np.array([SC5_PHASES.index(r['teacher_phase']) for r in rl], dtype=np.int64)
        yc = np.array([float(r.get('teacher_sc5_corridor_active',0)) for r in rl], dtype=np.float32)
        yr = np.array([float(r['teacher_phase']=='release_safe') for r in rl], dtype=np.float32)
        return X, {"phase": yp, "corridor": yc, "release": yr}

    Xtr, Ytr = mk(tr_rows); Xvl, Yvl = mk(vl_rows)
    meta = {'train_eps': tr_eps, 'val_eps': vl_eps, 'n_train_eps': len(tr_eps), 'n_val_eps': len(vl_eps)}
    return Xtr, Ytr, Xvl, Yvl, len(tr_rows), len(vl_rows), meta

def train(model, Xtr, Ytr, Xvl, Yvl, lr=0.001, weight_decay=1e-4, epochs=80, batch_size=64, device="cpu"):
    model = model.to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xvl, dtype=torch.float32, device=device)

    # Class weights: inverse frequency
    counts = Counter(Ytr["phase"].tolist())
    total = sum(counts.values())
    cw = torch.tensor([total/max(counts.get(i,1),1) for i in range(len(SC5_PHASES))],
                      dtype=torch.float32, device=device)

    pl = torch.nn.CrossEntropyLoss(weight=cw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_vl = float("inf"); best_state = None; best_epoch = 0
    log = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tl = 0.0; nb = 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i+batch_size]; xb = Xt[idx]
            cpu_idx = idx.cpu().numpy()
            yp = torch.tensor(Ytr["phase"][cpu_idx], dtype=torch.long, device=device)
            yc = torch.tensor(Ytr["corridor"][cpu_idx], dtype=torch.float32, device=device).unsqueeze(1)
            out = model(xb)
            # Phase + corridor only (release head uses frozen V1 weights — release_safe=0 in 05f6)
            loss = pl(out["phase_logits"], yp) + 0.5*bce(out["corridor_logit"], yc)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            ov = model(Xv)
            ypv = torch.tensor(Yvl["phase"], dtype=torch.long, device=device)
            vl = pl(ov["phase_logits"], ypv).item()
            va = (ov["phase_logits"].argmax(1)==ypv).float().mean().item()

        if vl < best_vl:
            best_vl = vl; best_state = copy.deepcopy(model.state_dict()); best_epoch = ep

        entry = {"epoch": ep, "train_loss": tl/max(nb,1), "val_phase_ce": vl, "val_phase_acc": va}
        log.append(entry)
        if ep % 10 == 0 or ep == epochs-1:
            print("  e%d: tr=%.3f vl_ce=%.3f vl_acc=%.3f" % (ep, entry["train_loss"], vl, va))

    model.load_state_dict(best_state)
    print("Best: epoch=%d vl_ce=%.3f" % (best_epoch, best_vl))
    return model, best_epoch, log

def export_checkpoint(model, ckpt_path, dataset_sha, mean, std, metadata, v1_ckpt_path=None):
    """Export checkpoint compatible with SC5DetectorRuntime strict=True.
    If v1_ckpt_path is provided, inject V1 release head + confidence head weights
    (release_safe=0 in 05f6 dataset, so release head cannot be trained on V2 data).
    """
    state = model.state_dict()

    if v1_ckpt_path:
        v1 = torch.load(v1_ckpt_path, map_location="cpu", weights_only=False)
        for k in ["release_head.weight", "release_head.bias",
                   "confidence_head.weight", "confidence_head.bias"]:
            if k in v1["model_state"]:
                state[k] = v1["model_state"][k]
        print("Injected V1 release + confidence heads from %s" % v1_ckpt_path)

    payload = {
        "model_state": state,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "feature_names": SC5_FEATURES,
        "phase_classes": SC5_PHASES,
        "dataset_sha256": dataset_sha,
        "dataset_path": str(Path(ckpt_path).parent.parent / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"),
        "split_mode": "frozen",
        "normalization_source": "train_only",
        "n_train_steps": metadata["n_train_steps"],
        "n_val_steps": metadata["n_val_steps"],
        "n_held_test": 0,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_episode_ids": metadata["train_eps"],
        "val_episode_ids": metadata["val_eps"],
        "heldout_episode_ids": [],
        "training_metadata": metadata,
        "release_head_source": "SC5-V1 (frozen, not trained on V2 data)",
        "note_release_head": "05f6 dataset has release_safe=0 (post-emit steps lack valid streaming features). Release head weights copied from V1 checkpoint.",
    }
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(payload, ckpt_path)
    ckpt_sha = hashlib.sha256(open(ckpt_path,"rb").read()).hexdigest()
    print("Checkpoint: %s (sha=%s)" % (ckpt_path, ckpt_sha[:16]))
    return ckpt_sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expected_dataset_sha256", default=EXPECTED_DATASET_SHA,
                    help="Fail-closed if dataset SHA does not match")
    ap.add_argument("--expected_train_steps", type=int, default=None,
                    help="Override expected train step count (for ablation datasets)")
    ap.add_argument("--expected_val_steps", type=int, default=None,
                    help="Override expected val step count")
    ap.add_argument("--expected_train_eps", type=int, default=None,
                    help="Override expected train episode count")
    ap.add_argument("--expected_val_eps", type=int, default=None,
                    help="Override expected val episode count")
    ap.add_argument("--v1_checkpoint", default="artifacts/detector/sc5_mlp_s2.pt",
                    help="SC5-V1 checkpoint for release head injection")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("Loading %s..." % args.dataset)
    Xtr, Ytr, Xvl, Yvl, n_tr, n_vl, meta = load_data(
        args.dataset, expected_sha=args.expected_dataset_sha256,
        exp_train_steps=args.expected_train_steps, exp_val_steps=args.expected_val_steps,
        exp_train_eps=args.expected_train_eps, exp_val_eps=args.expected_val_eps)
    print("  train=%d steps (%d eps)  val=%d steps (%d eps)" % (n_tr, meta['n_train_eps'], n_vl, meta['n_val_eps']))

    # Normalize
    _mean = Xtr.mean(0); _std = Xtr.std(0) + 1e-8
    Xtr_norm = (Xtr-_mean)/_std; Xvl_norm = (Xvl-_mean)/_std

    model = SC5MLP(n_feat=len(SC5_FEATURES))
    n_params = sum(p.numel() for p in model.parameters())
    print("Model: %d params" % n_params)

    dataset_sha = hashlib.sha256(open(args.dataset,"rb").read()).hexdigest()

    model, best_epoch, log = train(model, Xtr_norm, Ytr, Xvl_norm, Yvl, device=args.device)

    metadata = {
        "seed": args.seed, "dataset_sha256": dataset_sha,
        "n_train_steps": n_tr, "n_val_steps": n_vl,
        "n_train_eps": meta['n_train_eps'], "n_val_eps": meta['n_val_eps'],
        "train_eps": meta['train_eps'], "val_eps": meta['val_eps'],
        "best_epoch": best_epoch, "n_params": n_params,
    }

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sc5_mlp_v2.pt"
    ckpt_sha = export_checkpoint(model, str(ckpt_path), dataset_sha, _mean, _std, metadata,
                                  v1_ckpt_path=args.v1_checkpoint)

    # ── Runtime strict-load verification (subprocess, avoids CUDA context issues) ──
    print("Verifying runtime strict-load...")
    import subprocess
    verify_script = (
        "import sys; sys.path.insert(0, '{repo}'); sys.path.insert(0, '{repo}/src')\n"
        "from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime\n"
        "import traceback, sys as _sys\n"
        "try:\n"
        "    rt = SC5DetectorRuntime('{ckpt}', tau_corridor=0.3, tau_release=0.3, guard=5)\n"
        "    print('RUNTIME_STRICT_LOAD=PASS')\n"
        "    print('dataset_sha256=' + rt.dataset_sha256[:16])\n"
        "except Exception as e:\n"
        "    traceback.print_exc()\n"
        "    print('RUNTIME_STRICT_LOAD=FAIL')\n"
        "    _sys.exit(1)\n"
    ).format(repo=REPO, ckpt=ckpt_path)
    proc = subprocess.run(
        [sys.executable, "-c", verify_script],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr[-500:])
    if "RUNTIME_STRICT_LOAD=PASS" not in proc.stdout:
        print("RUNTIME_LOAD_FAILED")
        sys.exit(1)

    # Save training log
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)

    # Summary JSON
    summary = {
        "seed": args.seed, "n_params": n_params,
        "best_epoch": best_epoch, "best_val_phase_ce": log[best_epoch]["val_phase_ce"],
        "final_val_phase_acc": log[best_epoch]["val_phase_acc"],
        "dataset_sha256": dataset_sha, "checkpoint_sha256": ckpt_sha,
        "trainer_sha256": hashlib.sha256(open(__file__,"rb").read()).hexdigest(),
        "runtime_strict_load": "PASS",
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Checkpoint: %s" % ckpt_path)

if __name__ == "__main__":
    main()
