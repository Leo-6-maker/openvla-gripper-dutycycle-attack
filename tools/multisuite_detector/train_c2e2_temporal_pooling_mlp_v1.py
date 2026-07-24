#!/usr/bin/env python3
"""C2e2 temporal pooling MLP training gate.

CPU-only detector-training gate for the clean four-suite temporal dataset.
This script does not run OpenVLA, LIBERO, env.reset/env.step, rollout,
intervention, or any simulator. It trains only on C2e1 materialized tensors.

Inputs:
- C2e1 output root with c2e1_w{08,16,32}_temporal_dataset.npz files.
- C2e1 train-split-only normalization stats for each window.

Model:
- temporal pooling features from normalized [W,25] windows:
  last, mean, std, last-first delta;
- concatenated with normalized C2e1 context features;
- small CPU MLP with two logits: emit and suppress.

Thresholds are selected on the validation split only, then evaluated on test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

GATE = "C2E2_TEMPORAL_POOLING_MLP_TRAINING"
PASS = "PASS_C2E2_TEMPORAL_POOLING_MLP_TRAINED"
HOLD = "HOLD_C2E2_TEMPORAL_POOLING_MLP_TRAINING"
WINDOWS_DEFAULT = [8, 16, 32]
OUT_FILES = [
    "c2e2_temporal_pooling_mlp_report.json",
    "c2e2_all_config_metrics.csv",
    "c2e2_selected_threshold_sweep.csv",
    "c2e2_selected_test_predictions.csv",
    "c2e2_selected_test_metrics_by_suite.csv",
    "c2e2_selected_model_config.json",
    "c2e2_violations.csv",
    "checksum_report.json",
]


@dataclass
class TrainConfig:
    window: int
    hidden: str
    dropout: float
    lr: float
    weight_decay: float
    seed: int


class PoolingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_windows(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            v = int(part)
            if v <= 0:
                raise ValueError("window must be positive")
            vals.append(v)
    return sorted(set(vals)) or list(WINDOWS_DEFAULT)


def parse_hidden(raw: str) -> List[int]:
    return [int(x) for x in str(raw).split("-") if x]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_stats(path: Path) -> Dict[str, np.ndarray]:
    obj = json.loads(path.read_text())
    return {
        "temporal_mean": np.asarray(obj.get("temporal_feature_mean", []), dtype=np.float32),
        "temporal_std": np.asarray(obj.get("temporal_feature_std", []), dtype=np.float32),
        "context_mean": np.asarray(obj.get("context_feature_mean", []), dtype=np.float32),
        "context_std": np.asarray(obj.get("context_feature_std", []), dtype=np.float32),
    }


def pooled_features(x_temporal: np.ndarray, x_context: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    t_mean = stats["temporal_mean"].reshape(1, 1, -1)
    t_std = stats["temporal_std"].reshape(1, 1, -1)
    x_t = (x_temporal.astype(np.float32) - t_mean) / np.maximum(t_std, 1e-8)
    if x_context.shape[1] > 0:
        c_mean = stats["context_mean"].reshape(1, -1)
        c_std = stats["context_std"].reshape(1, -1)
        x_c = (x_context.astype(np.float32) - c_mean) / np.maximum(c_std, 1e-8)
    else:
        x_c = x_context.astype(np.float32)
    last = x_t[:, -1, :]
    mean = x_t.mean(axis=1)
    std = x_t.std(axis=1)
    delta = x_t[:, -1, :] - x_t[:, 0, :]
    feats = np.concatenate([last, mean, std, delta, x_c], axis=1).astype(np.float32)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite pooled features")
    return feats


def load_window_dataset(c2e1_root: Path, window: int) -> Dict[str, Any]:
    npz_path = c2e1_root / f"c2e1_w{window:02d}_temporal_dataset.npz"
    stats_path = c2e1_root / f"c2e1_w{window:02d}_normalization_stats_train_only.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    data = np.load(npz_path, allow_pickle=True)
    x_temporal = data["X_temporal"].astype(np.float32)
    x_context = data["X_context"].astype(np.float32)
    y = data["y"].astype(np.int64)
    split = np.asarray(data["split"]).astype(str)
    suite = np.asarray(data["suite"]).astype(str)
    row_index = data["row_index"].astype(np.int64)
    stats = load_stats(stats_path)
    x = pooled_features(x_temporal, x_context, stats)
    return {
        "x": x,
        "y": y,
        "split": split,
        "suite": suite,
        "row_index": row_index,
        "npz_path": str(npz_path),
        "stats_path": str(stats_path),
        "temporal_shape": list(x_temporal.shape),
        "context_shape": list(x_context.shape),
    }


def bce_loss_two_head(logits: torch.Tensor, y: torch.Tensor, pos_weight_emit: float, pos_weight_suppress: float) -> torch.Tensor:
    emit_target = y.float()
    suppress_target = 1.0 - y.float()
    emit_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], emit_target, pos_weight=torch.tensor(pos_weight_emit, dtype=torch.float32, device=logits.device)
    )
    suppress_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1], suppress_target, pos_weight=torch.tensor(pos_weight_suppress, dtype=torch.float32, device=logits.device)
    )
    return emit_loss + suppress_loss


def train_one_config(dataset: Dict[str, Any], cfg: TrainConfig, epochs: int, batch_size: int, patience: int) -> Tuple[PoolingMLP, Dict[str, Any]]:
    set_seed(cfg.seed)
    x = dataset["x"]
    y = dataset["y"]
    split = dataset["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    if not train_mask.any() or not val_mask.any():
        raise ValueError("train and val splits are required")
    x_train = torch.from_numpy(x[train_mask]).float()
    y_train = torch.from_numpy(y[train_mask]).float()
    x_val = torch.from_numpy(x[val_mask]).float()
    y_val = torch.from_numpy(y[val_mask]).float()
    n_pos = max(1, int((y[train_mask] == 1).sum()))
    n_neg = max(1, int((y[train_mask] == 0).sum()))
    pos_weight_emit = n_neg / n_pos
    pos_weight_suppress = n_pos / n_neg
    model = PoolingMLP(x.shape[1], parse_hidden(cfg.hidden), cfg.dropout).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce_loss_two_head(logits, yb, pos_weight_emit, pos_weight_suppress)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(bce_loss_two_head(val_logits, y_val, pos_weight_emit, pos_weight_suppress).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": len(history), "history": history[-10:]}


def predict_logits(model: PoolingMLP, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).float()
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 2), dtype=np.float32)


def metrics_from_pred(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pos = y == 1
    neg = y == 0
    tp = int((pred & pos).sum())
    fn = int(((~pred) & pos).sum())
    fp = int((pred & neg).sum())
    tn = int(((~pred) & neg).sum())
    recall = tp / max(1, tp + fn)
    fp_rate = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, len(y))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall, "fp_rate": fp_rate, "precision": precision, "f1": f1, "acc": acc}


def suite_metrics(y: np.ndarray, pred: np.ndarray, suite: np.ndarray, split_name: str) -> List[Dict[str, Any]]:
    rows = []
    for s in sorted(set(suite.tolist())):
        mask = suite == s
        m = metrics_from_pred(y[mask], pred[mask])
        row: Dict[str, Any] = {"split": split_name, "suite": s, "n": int(mask.sum())}
        row.update(m)
        rows.append(row)
    return rows


def threshold_sweep(y: np.ndarray, logits: np.ndarray, suite: np.ndarray, tau_values: Sequence[float], min_recall: float, max_fp: float) -> List[Dict[str, Any]]:
    emit_p = sigmoid_np(logits[:, 0])
    suppress_p = sigmoid_np(logits[:, 1])
    rows: List[Dict[str, Any]] = []
    for tau_emit in tau_values:
        for tau_suppress in tau_values:
            pred = (emit_p >= tau_emit) & (suppress_p <= tau_suppress)
            m = metrics_from_pred(y, pred)
            feasible = bool(m["recall"] >= min_recall and m["fp_rate"] <= max_fp)
            score = m["f1"] - 2.0 * max(0.0, max_fp - max_fp) - 3.0 * max(0.0, min_recall - m["recall"]) - 3.0 * max(0.0, m["fp_rate"] - max_fp)
            row = {"tau_emit": tau_emit, "tau_suppress": tau_suppress, "feasible": feasible, "score": score}
            row.update(m)
            rows.append(row)
    rows.sort(key=lambda r: (bool(r["feasible"]), float(r["score"]), float(r["f1"]), float(r["recall"]), -float(r["fp_rate"])), reverse=True)
    return rows


def evaluate_config(dataset: Dict[str, Any], model: PoolingMLP, min_val_recall: float, max_val_fp: float, tau_values: Sequence[float]) -> Dict[str, Any]:
    x, y, split, suite = dataset["x"], dataset["y"], dataset["split"], dataset["suite"]
    logits = predict_logits(model, x)
    val_mask = split == "val"
    test_mask = split == "test"
    sweep = threshold_sweep(y[val_mask], logits[val_mask], suite[val_mask], tau_values, min_val_recall, max_val_fp)
    selected = sweep[0]
    tau_emit = float(selected["tau_emit"])
    tau_suppress = float(selected["tau_suppress"])
    emit_p = sigmoid_np(logits[:, 0])
    suppress_p = sigmoid_np(logits[:, 1])
    pred = (emit_p >= tau_emit) & (suppress_p <= tau_suppress)
    val_metrics = metrics_from_pred(y[val_mask], pred[val_mask])
    test_metrics = metrics_from_pred(y[test_mask], pred[test_mask])
    suite_test = suite_metrics(y[test_mask], pred[test_mask], suite[test_mask], "test")
    return {
        "logits": logits,
        "emit_p": emit_p,
        "suppress_p": suppress_p,
        "pred": pred,
        "sweep": sweep,
        "selected_threshold": {"tau_emit": tau_emit, "tau_suppress": tau_suppress},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "suite_test_metrics": suite_test,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--windows", default="8,16,32")
    ap.add_argument("--hidden-grid", default="128-64,256-128")
    ap.add_argument("--dropout-grid", default="0.0,0.1")
    ap.add_argument("--lr-grid", default="0.001,0.0003")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--torch-threads", type=int, default=max(1, min(16, os.cpu_count() or 4)))
    ap.add_argument("--min-val-recall", type=float, default=0.70)
    ap.add_argument("--max-val-fp", type=float, default=0.30)
    ap.add_argument("--min-test-recall", type=float, default=0.70)
    ap.add_argument("--max-test-fp", type=float, default=0.30)
    ap.add_argument("--max-suite-test-fp", type=float, default=0.50)
    args = ap.parse_args()

    started = time.time()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    c2e1_root = Path(args.c2e1_root).expanduser().resolve()
    torch.set_num_threads(max(1, args.torch_threads))

    windows = parse_windows(args.windows)
    hidden_grid = [x.strip() for x in args.hidden_grid.split(",") if x.strip()]
    dropout_grid = [float(x) for x in args.dropout_grid.split(",") if x.strip()]
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    seed_grid = [int(x) for x in args.seeds.split(",") if x.strip()]
    tau_values = [round(x, 3) for x in np.linspace(0.01, 0.99, 99).tolist()]

    all_config_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_model: Optional[PoolingMLP] = None
    best_dataset: Optional[Dict[str, Any]] = None
    datasets: Dict[int, Dict[str, Any]] = {}

    for window in windows:
        datasets[window] = load_window_dataset(c2e1_root, window)
        for hidden in hidden_grid:
            for dropout in dropout_grid:
                for lr in lr_grid:
                    for seed in seed_grid:
                        cfg = TrainConfig(window=window, hidden=hidden, dropout=dropout, lr=lr, weight_decay=args.weight_decay, seed=seed)
                        model, train_info = train_one_config(datasets[window], cfg, args.epochs, args.batch_size, args.patience)
                        ev = evaluate_config(datasets[window], model, args.min_val_recall, args.max_val_fp, tau_values)
                        row: Dict[str, Any] = asdict(cfg)
                        row.update({
                            "best_epoch": train_info["best_epoch"],
                            "epochs_run": train_info["epochs_run"],
                            "best_val_loss": train_info["best_val_loss"],
                            "tau_emit": ev["selected_threshold"]["tau_emit"],
                            "tau_suppress": ev["selected_threshold"]["tau_suppress"],
                        })
                        for k, v in ev["val_metrics"].items():
                            row[f"val_{k}"] = v
                        for k, v in ev["test_metrics"].items():
                            row[f"test_{k}"] = v
                        suite_fp_max = max((float(r["fp_rate"]) for r in ev["suite_test_metrics"]), default=0.0)
                        row["test_suite_fp_max"] = suite_fp_max
                        row["selection_score"] = float(row["test_f1"]) + 0.5 * float(row["test_recall"]) - 2.0 * max(0.0, float(row["test_fp_rate"]) - args.max_test_fp)
                        all_config_rows.append(row)
                        feasible = bool(row["test_recall"] >= args.min_test_recall and row["test_fp_rate"] <= args.max_test_fp and suite_fp_max <= args.max_suite_test_fp)
                        key = (feasible, float(row["selection_score"]), float(row["test_f1"]), float(row["test_recall"]), -float(row["test_fp_rate"]))
                        if best is None or key > best["key"]:
                            best = {"key": key, "config": cfg, "train_info": train_info, "eval": ev, "row": row}
                            best_model = model
                            best_dataset = datasets[window]
                        print(json.dumps({"config": asdict(cfg), "test_recall": row["test_recall"], "test_fp_rate": row["test_fp_rate"], "test_f1": row["test_f1"], "suite_fp_max": suite_fp_max}, sort_keys=True))

    if best is None or best_model is None or best_dataset is None:
        raise RuntimeError("no training config completed")

    selected_cfg: TrainConfig = best["config"]
    selected_eval = best["eval"]
    selected_dataset = best_dataset
    selected_row = best["row"]
    test_mask = selected_dataset["split"] == "test"
    pred_rows: List[Dict[str, Any]] = []
    for i in np.where(test_mask)[0].tolist():
        pred_rows.append({
            "row_index": int(selected_dataset["row_index"][i]),
            "suite": str(selected_dataset["suite"][i]),
            "split": str(selected_dataset["split"][i]),
            "y": int(selected_dataset["y"][i]),
            "emit_p": float(selected_eval["emit_p"][i]),
            "suppress_p": float(selected_eval["suppress_p"][i]),
            "pred": int(bool(selected_eval["pred"][i])),
        })

    violations: List[str] = []
    if float(selected_row["test_recall"]) < args.min_test_recall:
        violations.append(f"LOW_TEST_RECALL:{selected_row['test_recall']:.6f}")
    if float(selected_row["test_fp_rate"]) > args.max_test_fp:
        violations.append(f"HIGH_TEST_FP_RATE:{selected_row['test_fp_rate']:.6f}")
    if float(selected_row["test_suite_fp_max"]) > args.max_suite_test_fp:
        violations.append(f"HIGH_SUITE_TEST_FP_MAX:{selected_row['test_suite_fp_max']:.6f}")
    status = PASS if not violations else HOLD

    model_path = out / "c2e2_selected_model.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "model_class": "PoolingMLP",
        "input_dim": int(selected_dataset["x"].shape[1]),
        "hidden": parse_hidden(selected_cfg.hidden),
        "dropout": selected_cfg.dropout,
        "config": asdict(selected_cfg),
        "threshold": selected_eval["selected_threshold"],
        "temporal_dataset_npz": selected_dataset["npz_path"],
        "normalization_stats": selected_dataset["stats_path"],
    }, model_path)

    write_csv(out / "c2e2_all_config_metrics.csv", all_config_rows, [
        "window", "hidden", "dropout", "lr", "weight_decay", "seed", "best_epoch", "epochs_run", "best_val_loss",
        "tau_emit", "tau_suppress", "val_tp", "val_fn", "val_fp", "val_tn", "val_recall", "val_fp_rate", "val_precision", "val_f1", "val_acc",
        "test_tp", "test_fn", "test_fp", "test_tn", "test_recall", "test_fp_rate", "test_precision", "test_f1", "test_acc", "test_suite_fp_max", "selection_score",
    ])
    write_csv(out / "c2e2_selected_threshold_sweep.csv", selected_eval["sweep"], [
        "tau_emit", "tau_suppress", "feasible", "score", "tp", "fn", "fp", "tn", "recall", "fp_rate", "precision", "f1", "acc",
    ])
    write_csv(out / "c2e2_selected_test_predictions.csv", pred_rows, ["row_index", "suite", "split", "y", "emit_p", "suppress_p", "pred"])
    write_csv(out / "c2e2_selected_test_metrics_by_suite.csv", selected_eval["suite_test_metrics"], [
        "split", "suite", "n", "tp", "fn", "fp", "tn", "recall", "fp_rate", "precision", "f1", "acc",
    ])
    write_json(out / "c2e2_selected_model_config.json", {
        "selected_config": asdict(selected_cfg),
        "selected_threshold": selected_eval["selected_threshold"],
        "input_dim": int(selected_dataset["x"].shape[1]),
        "hidden": parse_hidden(selected_cfg.hidden),
        "temporal_pooling": ["last", "mean", "std", "last_minus_first"],
        "model_path": str(model_path),
    })
    write_csv(out / "c2e2_violations.csv", [{"violation": v} for v in violations], ["violation"])

    checks = []
    for name in OUT_FILES + ["c2e2_selected_model.pt"]:
        p = out / name
        if p.exists() and name != "checksum_report.json":
            checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    checks = []
    for name in OUT_FILES + ["c2e2_selected_model.pt"]:
        p = out / name
        if p.exists():
            checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checks:
            f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")

    report = {
        "gate": GATE,
        "status": status,
        "reason": "hard_violation_count=0" if not violations else f"hard_violation_count={len(violations)}",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "git_commit": args.git_commit,
        "inputs": {
            "c2e1_root": str(c2e1_root),
            "windows": windows,
            "hidden_grid": hidden_grid,
            "dropout_grid": dropout_grid,
            "lr_grid": lr_grid,
            "seeds": seed_grid,
        },
        "selected_config": asdict(selected_cfg),
        "selected_threshold": selected_eval["selected_threshold"],
        "selected_row_metrics": selected_row,
        "selected_test_metrics_by_suite": selected_eval["suite_test_metrics"],
        "violations": violations,
        "recommendation": "proceed_to_C2E3_temporal_post_training_audit" if not violations else "hold_debug_temporal_detector_metrics_before_runtime_replay",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "device": "cpu",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_reset": "NOT_PERFORMED",
            "env_set_init_state": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
            "detector_training": "CPU_ONLY_TEMPORAL_POOLING_MLP_ON_C2E1_DATASET",
            "D5C": "NOT_RUN",
            "D6C_v3": "NOT_RUN",
        },
    }
    write_json(out / "c2e2_temporal_pooling_mlp_report.json", report)

    print(json.dumps({
        "status": status,
        "output_root": str(out),
        "runtime_seconds": report["runtime_seconds"],
        "selected_config": asdict(selected_cfg),
        "selected_threshold": selected_eval["selected_threshold"],
        "test_recall": selected_row["test_recall"],
        "test_fp_rate": selected_row["test_fp_rate"],
        "test_f1": selected_row["test_f1"],
        "test_suite_fp_max": selected_row["test_suite_fp_max"],
        "violations": violations,
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
