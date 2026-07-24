#!/usr/bin/env python3
"""C2e2D: small causal TCN / GRU sequence model training gate.

Replaces pooling MLP with causal sequence models that preserve temporal order:
  - CausalTCN: dilated Conv1d with causal padding
  - GRU: recurrent readout from final hidden state

Inputs: C2e1 materialized temporal datasets (W=8,16,32).
Selection: val-only (model, window, hyperparams, threshold).
Test: final report only.

CPU-only. No OpenVLA, no LIBERO, no env.step, no rollout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PASS = "PASS_C2E2D_SEQUENCE_MODEL_TRAINED"
HOLD = "HOLD_C2E2D_SEQUENCE_MODEL_TRAINING"

# ===================== Models =====================
class CausalTCN(nn.Module):
    """Small causal TCN for [W, 25] temporal windows."""
    def __init__(self, n_features=25, n_context=0, channels=64, kernel_size=3, dilations=(1,2,4), dropout=0.0):
        super().__init__()
        layers = []
        in_ch = n_features
        for d in dilations:
            pad = (kernel_size - 1) * d  # causal: pad only left side
            layers.extend([
                nn.Conv1d(in_ch, channels, kernel_size, dilation=d,
                         padding=pad, padding_mode='zeros'),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            # Trim to causal: remove right-side padding
            self._trim = pad
            in_ch = channels
        self.conv = nn.Sequential(*layers) if layers else nn.Identity()
        self._trim = (kernel_size - 1) * max(dilations) if dilations else 0
        self.input_dim = channels + n_context
        self.head = nn.Linear(channels + n_context, 2)

    def forward(self, x_temporal, x_context):
        # x_temporal: [B, W, F]
        x = x_temporal.permute(0, 2, 1)  # [B, F, W]
        x = self.conv(x)
        if self._trim > 0:
            x = x[:, :, :-self._trim] if x.shape[2] > self._trim else x[:, :, -1:]
        last = x[:, :, -1]  # [B, channels]
        if x_context.shape[1] > 0:
            last = torch.cat([last, x_context], dim=1)
        return self.head(last)


class GRUSmall(nn.Module):
    """Small GRU readout for [W, 25] temporal windows."""
    def __init__(self, n_features=25, n_context=0, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.input_dim = hidden + n_context
        self.head = nn.Linear(hidden + n_context, 2)

    def forward(self, x_temporal, x_context):
        _, h = self.gru(x_temporal)  # h: [layers, B, hidden]
        last = h[-1]  # [B, hidden]
        if x_context.shape[1] > 0:
            last = torch.cat([last, x_context], dim=1)
        return self.head(last)


@dataclass
class TrainConfig:
    window: int; model_type: str; channels: int; dropout: float
    lr: float; weight_decay: float; seed: int

# ===================== Helpers (shared with C2e2 v2) =====================
def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+"\n")
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

def load_stats(path):
    obj = json.loads(open(path).read())
    return {"temporal_mean": np.asarray(obj.get("temporal_feature_mean",[]), dtype=np.float32),
            "temporal_std": np.asarray(obj.get("temporal_feature_std",[]), dtype=np.float32),
            "context_mean": np.asarray(obj.get("context_feature_mean",[]), dtype=np.float32),
            "context_std": np.asarray(obj.get("context_feature_std",[]), dtype=np.float32)}

def normalize_data(xt, xc, stats):
    tm, ts = stats["temporal_mean"].reshape(1,1,-1), stats["temporal_std"].reshape(1,1,-1)
    xt = (xt.astype(np.float32) - tm) / np.maximum(ts, 1e-8)
    if xc.shape[1] > 0:
        cm, cs = stats["context_mean"].reshape(1,-1), stats["context_std"].reshape(1,-1)
        xc = (xc.astype(np.float32) - cm) / np.maximum(cs, 1e-8)
    return xt, xc

def load_window_dataset(c2e1_root, window):
    npz = np.load(Path(c2e1_root) / f"c2e1_w{window:02d}_temporal_dataset.npz", allow_pickle=True)
    stats = load_stats(Path(c2e1_root) / f"c2e1_w{window:02d}_normalization_stats_train_only.json")
    xt, xc = normalize_data(np.asarray(npz["X_temporal"], dtype=np.float32),
                            np.asarray(npz["X_context"], dtype=np.float32), stats)
    return {"xt": xt, "xc": xc, "y": npz["y"].astype(np.int64),
            "split": np.asarray(npz["split"]).astype(str),
            "suite": np.asarray(npz["suite"]).astype(str),
            "row_index": npz["row_index"].astype(np.int64),
            "n_context": xc.shape[1]}

def predict_logits(model, xt, xc, batch_size=512):
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(xt), batch_size):
            outs.append(model(torch.from_numpy(xt[s:s+batch_size]).float(),
                            torch.from_numpy(xc[s:s+batch_size]).float()).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,2), dtype=np.float32)

def metrics_from_pred(y, pred):
    pos, neg = y==1, y==0
    tp, fn = int((pred&pos).sum()), int(((~pred)&pos).sum())
    fp, tn = int((pred&neg).sum()), int(((~pred)&neg).sum())
    rec = tp/max(1, tp+fn); fpr = fp/max(1, fp+tn); prec = tp/max(1, tp+fp)
    f1 = 2*prec*rec/max(1e-12, prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1,
            "acc":(tp+tn)/max(1,len(y))}

def threshold_sweep(y, ep, sp, tau_vals, min_recall, max_fp):
    rows = []
    for te in tau_vals:
        for ts in tau_vals:
            pred = (ep>=te)&(sp<=ts); m = metrics_from_pred(y, pred)
            feasible = m["recall"]>=min_recall and m["fp_rate"]<=max_fp
            score = m["f1"]+0.5*m["recall"]-3*max(0,min_recall-m["recall"])-3*max(0,m["fp_rate"]-max_fp)
            rows.append({"tau_emit":te,"tau_suppress":ts,"feasible":feasible,"score":score,**m})
    rows.sort(key=lambda r: (bool(r["feasible"]), r["score"], r["f1"], r["recall"], -r["fp_rate"]), reverse=True)
    return rows

def suite_metrics(y, pred, suite, split_name):
    return [{"split":split_name,"suite":s,"n":int((suite==s).sum()),
             **metrics_from_pred(y[suite==s], pred[suite==s])} for s in sorted(set(suite))]

def bce_two_head(logits, y, pw_emit, pw_suppress):
    et, st = y.float(), 1.0 - y.float()
    return (nn.functional.binary_cross_entropy_with_logits(logits[:,0], et, pos_weight=torch.tensor(pw_emit, device=logits.device)) +
            nn.functional.binary_cross_entropy_with_logits(logits[:,1], st, pos_weight=torch.tensor(pw_suppress, device=logits.device)))

# ===================== Training =====================
def train_one_config(dataset, cfg, epochs, batch_size, patience, torch_threads):
    set_seed(cfg.seed); torch.set_num_threads(max(1, torch_threads))
    xt, xc, y, split = dataset["xt"], dataset["xc"], dataset["y"], dataset["split"]
    tm, vm = split=="train", split=="val"
    n_pos, n_neg = max(1, int((y[tm]==1).sum())), max(1, int((y[tm]==0).sum()))
    pw_emit, pw_suppress = n_neg/n_pos, n_pos/n_neg

    if cfg.model_type == "tcn":
        model = CausalTCN(xt.shape[2], xc.shape[1], cfg.channels, dropout=cfg.dropout)
    else:
        model = GRUSmall(xt.shape[2], xc.shape[1], cfg.channels, dropout=cfg.dropout)
    model.cpu()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(xt[tm]).float(), torch.from_numpy(xc[tm]).float(),
                        torch.from_numpy(y[tm]).float()), batch_size=batch_size, shuffle=True, drop_last=False)
    xt_val = torch.from_numpy(xt[vm]).float(); xc_val = torch.from_numpy(xc[vm]).float()
    y_val = torch.from_numpy(y[vm]).float()
    best_state, best_val, best_epoch, stale = None, float("inf"), -1, 0

    for epoch in range(1, epochs+1):
        model.train(); losses = []
        for xtb, xcb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = bce_two_head(model(xtb, xcb), yb, pw_emit, pw_suppress); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(bce_two_head(model(xt_val, xc_val), y_val, pw_emit, pw_suppress).detach().cpu())
        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": epoch, "train_loss": float(np.mean(losses))}


def evaluate_config(dataset, model, min_recall, max_fp, tau_vals):
    xt, xc, y, split = dataset["xt"], dataset["xc"], dataset["y"], dataset["split"]
    logits = predict_logits(model, xt, xc)
    ep, sp = sigmoid_np(logits[:,0]), sigmoid_np(logits[:,1])
    vm, tm_test = split=="val", split=="test"
    sweep = threshold_sweep(y[vm], ep[vm], sp[vm], tau_vals, min_recall, max_fp)
    best = sweep[0]; te, ts = best["tau_emit"], best["tau_suppress"]
    pred = (ep>=te)&(sp<=ts)
    return {"logits":logits,"emit_p":ep,"suppress_p":sp,"pred":pred,"sweep":sweep,
            "selected_threshold":{"tau_emit":te,"tau_suppress":ts},
            "val_metrics":metrics_from_pred(y[vm], pred[vm]),
            "test_metrics":metrics_from_pred(y[tm_test], pred[tm_test]),
            "suite_test":suite_metrics(y[tm_test], pred[tm_test], dataset["suite"][tm_test], "test")}


def _train_worker(args_dict):
    cfg = TrainConfig(**args_dict["config"])
    dataset = load_window_dataset(args_dict["c2e1_root"], cfg.window)
    model, ti = train_one_config(dataset, cfg, args_dict["epochs"], args_dict["batch_size"],
                                  args_dict["patience"], args_dict.get("torch_threads",4))
    ev = evaluate_config(dataset, model, args_dict["min_recall"], args_dict["max_fp"], args_dict["tau_values"])
    row = asdict(cfg)
    row.update({"best_epoch":ti["best_epoch"],"epochs_run":ti["epochs_run"],"best_val_loss":ti["best_val_loss"],
                "tau_emit":ev["selected_threshold"]["tau_emit"],"tau_suppress":ev["selected_threshold"]["tau_suppress"]})
    for k,v in ev["val_metrics"].items(): row[f"val_{k}"] = v
    for k,v in ev["test_metrics"].items(): row[f"test_{k}"] = v
    row["test_suite_fp_max"] = max((float(r["fp_rate"]) for r in ev["suite_test"]), default=0.0)
    vr, vf1, vfp = float(row["val_recall"]), float(row["val_f1"]), float(row["val_fp_rate"])
    row["selection_score"] = vf1 + 0.5*vr - 3*max(0, args_dict["min_recall"]-vr) - 3*max(0, vfp-args_dict["max_fp"])
    row["feasible"] = bool(vr >= args_dict["min_recall"] and vfp <= args_dict["max_fp"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e1-root", required=True); ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--windows", default="8,16,32"); ap.add_argument("--models", default="tcn,gru")
    ap.add_argument("--channels-grid", default="64,128"); ap.add_argument("--dropout-grid", default="0.0,0.1")
    ap.add_argument("--lr-grid", default="0.001,0.0003"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-val-recall", type=float, default=0.70)
    ap.add_argument("--max-val-fp", type=float, default=0.30)
    ap.add_argument("--min-test-recall", type=float, default=0.70)
    ap.add_argument("--max-test-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-test-fp", type=float, default=0.50)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    c2e1_root = str(Path(args.c2e1_root).resolve())
    tau_vals = [round(x,3) for x in np.linspace(0.01,0.99,99).tolist()]

    windows = [int(x) for x in args.windows.split(",")]
    models = [x.strip() for x in args.models.split(",")]
    channels_grid = [int(x) for x in args.channels_grid.split(",")]
    dropout_grid = [float(x) for x in args.dropout_grid.split(",")]
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    all_configs = []
    for w in windows:
        for mt in models:
            for ch in channels_grid:
                for do in dropout_grid:
                    for lr in lr_grid:
                        for sd in seeds:
                            all_configs.append({"c2e1_root": c2e1_root,
                                "config": {"window":w,"model_type":mt,"channels":ch,"dropout":do,
                                           "lr":lr,"weight_decay":1e-4,"seed":sd},
                                "epochs":args.epochs,"batch_size":args.batch_size,"patience":args.patience,
                                "min_recall":args.min_val_recall,"max_fp":args.max_val_fp,
                                "tau_values":tau_vals,"torch_threads":args.torch_threads})

    nw = min(args.workers, len(all_configs))
    print(f"C2e2D: {len(all_configs)} configs × {nw} workers (W={windows} models={models})")

    all_rows = []; best_row = None; done = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures = {pool.submit(_train_worker, c): c for c in all_configs}
        for fut in as_completed(futures):
            row = fut.result(); all_rows.append(row); done += 1
            sel = float(row["selection_score"]); feasible = bool(row["feasible"])
            key = (feasible, sel, float(row["val_f1"]), float(row["val_recall"]), -float(row["val_fp_rate"]))
            if best_row is None or key > best_row["_key"]:
                row["_key"] = key; best_row = row
            if done % 24 == 0 or done == len(all_configs):
                print(f"  {done}/{len(all_configs)} configs...")

    if best_row is None: raise RuntimeError("no config completed")

    # Retrain best in main process
    best_cfg = TrainConfig(window=int(best_row["window"]), model_type=str(best_row["model_type"]),
        channels=int(best_row["channels"]), dropout=float(best_row["dropout"]),
        lr=float(best_row["lr"]), weight_decay=float(best_row["weight_decay"]), seed=int(best_row["seed"]))
    torch.set_num_threads(max(1, args.torch_threads * 4))
    dataset = load_window_dataset(c2e1_root, best_cfg.window)
    model, ti = train_one_config(dataset, best_cfg, args.epochs, args.batch_size, args.patience, args.torch_threads*4)
    ev = evaluate_config(dataset, model, args.min_val_recall, args.max_val_fp, tau_vals)
    selected_row = best_row; test_mask = dataset["split"] == "test"

    # Test predictions
    pred_rows = []
    for i in np.where(test_mask)[0]:
        pred_rows.append({"row_index":int(dataset["row_index"][i]),"suite":str(dataset["suite"][i]),
            "y":int(dataset["y"][i]),"pred":int(ev["pred"][i]),
            "emit_p":float(ev["emit_p"][i]),"suppress_p":float(ev["suppress_p"][i])})

    # Violations
    violations = []
    if float(selected_row["test_recall"]) < args.min_test_recall:
        violations.append(f"LOW_TEST_RECALL:{selected_row['test_recall']:.4f}")
    if float(selected_row["test_fp_rate"]) > args.max_test_fp:
        violations.append(f"HIGH_TEST_FP:{selected_row['test_fp_rate']:.4f}")
    sfpm = float(selected_row["test_suite_fp_max"])
    if sfpm > args.max_suite_test_fp:
        violations.append(f"HIGH_SUITE_FP_MAX:{sfpm:.4f}")
    status = PASS if not violations else HOLD

    # Save model
    model_path = out / "c2e2d_selected_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_class": best_cfg.model_type,
        "config": asdict(best_cfg), "threshold": ev["selected_threshold"],
        "n_features": dataset["xt"].shape[2], "n_context": dataset["n_context"]}, model_path)

    # Outputs
    write_csv(out / "c2e2d_all_config_metrics.csv", all_rows,
              ["window","model_type","channels","dropout","lr","weight_decay","seed",
               "best_epoch","epochs_run","best_val_loss","tau_emit","tau_suppress",
               "val_tp","val_fn","val_fp","val_tn","val_recall","val_fp_rate","val_f1",
               "test_tp","test_fn","test_fp","test_tn","test_recall","test_fp_rate","test_f1",
               "test_suite_fp_max","selection_score","feasible"])
    write_csv(out / "c2e2d_selected_threshold_sweep.csv", ev["sweep"],
              ["tau_emit","tau_suppress","feasible","score","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])
    write_csv(out / "c2e2d_selected_test_predictions.csv", pred_rows,
              ["row_index","suite","y","pred","emit_p","suppress_p"])
    write_csv(out / "c2e2d_selected_test_metrics_by_suite.csv", ev["suite_test"],
              ["split","suite","n","tp","fn","fp","tn","recall","fp_rate","precision","f1","acc"])
    write_json(out / "c2e2d_selected_model_config.json",
               {"selected_config": asdict(best_cfg), "selected_threshold": ev["selected_threshold"]})
    write_csv(out / "c2e2d_violations.csv", [{"violation":v} for v in violations], ["violation"])

    report = {
        "gate": "C2E2D_CAUSAL_SEQUENCE_MODEL_TRAINING",
        "status": status, "reason": "violations=0" if not violations else f"violations={len(violations)}",
        "created_at_unix": time.time(), "runtime_seconds": time.time()-t0, "git_commit": args.git_commit,
        "selection_rule": "model/config/threshold selected on validation split only; test split used only for final report",
        "selected_config": asdict(best_cfg), "selected_threshold": ev["selected_threshold"],
        "selected_row_metrics": selected_row,
        "selected_test_metrics_by_suite": ev["suite_test"],
        "violations": violations,
        "recommendation": ("proceed_to_C2E2E_post_training_audit" if not violations else
                          "hold_debug_or_proceed_to_C2F_observation_features"),
        "boundaries": {
            "CUDA_required":"NOT_REQUIRED","device":"cpu","OpenVLA_model":"NOT_LOADED",
            "LIBERO_runtime":"NOT_PERFORMED","detector_training":"CPU_ONLY_SEQUENCE_MODEL_ON_C2E1",
            "env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED",
        },
    }
    write_json(out / "c2e2d_sequence_model_report.json", report)

    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and fn != "checksum_report.json": csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)
    with open(out/"SHA256SUMS","w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS","SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status":status,"selected":asdict(best_cfg),"test_recall":selected_row["test_recall"],
        "test_fp":selected_row["test_fp_rate"],"test_f1":selected_row["test_f1"],
        "violations":violations,"runtime":report["runtime_seconds"]}, indent=2, sort_keys=True))
    return 0 if not violations else 1

if __name__ == "__main__":
    raise SystemExit(main())
