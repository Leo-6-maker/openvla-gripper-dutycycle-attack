#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import (  # noqa: E402
    SC5_V2_FEATURES,
    validate_no_forbidden_inputs,
)

GATE = "D4B0_CLEAN2000_RUNTIME_OBJECTIVE_25D_SEPARABILITY_AUDIT"
PASS = "PASS_D4B0_RUNTIME_OBJECTIVE_25D_SEPARABILITY_AUDITED"
HOLD = "HOLD_D4B0_RUNTIME_OBJECTIVE_25D_SEPARABILITY_AUDIT"
OUT_FILES = [
    "d4b0_runtime_objective_separability_report.json",
    "d4b0_row_labels.csv",
    "d4b0_metrics_by_subset.csv",
    "d4b0_threshold_sweep.csv",
    "d4b0_nearest_neighbor_overlap.csv",
    "d4b0_hard_negative_manifest.csv",
    "d4b0_violations.csv",
    "d4b0_upper_bound_checkpoint.pt",
    "checksum_report.json",
]
POSITIVE_LABELS = {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"}
NEGATIVE_LABELS = {"VALID_AUXILIARY", "VALID_AUXILIARY_CANDIDATE", "NO_EVENT"}
NEGATIVE_ROLES = {"auxiliary_manipulation", "unsupported_or_abstain", "distractor_or_setup"}
POSITIVE_ROLES = {"primary_attackable"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stable_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def group_key(row: Dict[str, Any]) -> str:
    for key in ["group_key", "episode_key", "parent_id", "run_id", "record_id", "episode_id"]:
        if row.get(key):
            return str(row[key])
    return str(row.get("record_id", ""))


def label_status(row: Dict[str, Any]) -> str:
    return str(row.get("teacher_label_status", row.get("label_status", "")))


def event_role(row: Dict[str, Any]) -> str:
    return str(row.get("event_role", row.get("event_role_true", "")))


def has_imputation(row: Dict[str, Any]) -> bool:
    if truthy(row.get("has_imputation")):
        return True
    try:
        return int(float(row.get("imputed_feature_count", 0) or 0)) > 0
    except Exception:
        return False


def runtime_label(row: Dict[str, Any]) -> Tuple[int, str]:
    status = label_status(row)
    role = event_role(row)
    if status in POSITIVE_LABELS or role in POSITIVE_ROLES:
        return 1, "runtime_emit_eligible"
    if status in NEGATIVE_LABELS or role in NEGATIVE_ROLES:
        return 0, "no_primary_suppress"
    return -1, "ignored_unknown_label"


def finite_feature_vector(row: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    xs: List[float] = []
    bad: List[str] = []
    for fn in SC5_V2_FEATURES:
        v = stable_float(row.get(fn))
        if not math.isfinite(v):
            bad.append(fn)
        xs.append(v)
    return np.asarray(xs, dtype=np.float32), bad


def load_false_emit_group_keys(paths: List[str]) -> set[str]:
    keys: set[str] = set()
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        for row in read_csv(path):
            g = str(row.get("group_key", ""))
            if g:
                keys.add(g)
    return keys


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_ranks = ranks[y == 1]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    pred = (np.asarray(p, dtype=np.float64) >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    acc = (tp + tn) / max(1, len(y))
    return {"n": int(len(y)), "tp": tp, "fn": fn, "fp": fp, "tn": tn, "acc": acc, "precision": precision, "recall": recall, "f1": f1, "no_primary_fp_rate": fpr, "pred_positive_rate": float(pred.mean()) if len(pred) else 0.0, "auc": auc_score(y, p)}


def threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    return [{"threshold": float(t), **binary_metrics(y, p, float(t))} for t in thresholds]


def choose_threshold(y: np.ndarray, p: np.ndarray, min_recall: float) -> Tuple[float, Dict[str, Any]]:
    candidates = threshold_sweep(y, p, [round(x, 3) for x in np.linspace(0.01, 0.99, 99)])
    feasible = [m for m in candidates if float(m["recall"]) >= min_recall]
    if not feasible:
        best = sorted(candidates, key=lambda m: (float(m["f1"]), float(m["recall"]), -float(m["no_primary_fp_rate"])), reverse=True)[0]
        return float(best["threshold"]), best
    best = sorted(feasible, key=lambda m: (float(m["no_primary_fp_rate"]), -float(m["recall"]), -float(m["f1"]))) [0]
    return float(best["threshold"]), best


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class UpperBoundMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> Tuple[UpperBoundMLP, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = UpperBoundMLP(x_train.shape[1], args.hidden)
    labels = y_train.astype(np.int64)
    class_counts = Counter(labels.tolist())
    weights = np.asarray([1.0 / max(1, class_counts[int(v)]) for v in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(ArrayDataset(x_train, y_train.astype(np.float32)), batch_size=args.batch_size, sampler=sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([max(1, class_counts.get(0, 0)) / max(1, class_counts.get(1, 0))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best = {"epoch": -1, "val_auc": -1.0, "state": None}
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(x_val, dtype=torch.float32))).numpy().reshape(-1)
        auc = auc_score(y_val, pv)
        score = -1.0 if math.isnan(auc) else auc
        if score > float(best["val_auc"]):
            best = {"epoch": epoch, "val_auc": score, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_auc": best["val_auc"], "class_counts": dict(class_counts)}


def predict(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / (std + 1e-8)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(z, dtype=torch.float32))).numpy().reshape(-1)


def nearest_neighbor_overlap(rows: List[Dict[str, Any]], x: np.ndarray, y: np.ndarray, split_mask: np.ndarray, limit: int = 1000) -> List[Dict[str, Any]]:
    idx = np.where(split_mask)[0]
    pos_idx = idx[y[idx] == 1]
    neg_idx = idx[y[idx] == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return []
    if len(neg_idx) > limit:
        neg_idx = neg_idx[:limit]
    xp = x[pos_idx]
    xn = x[neg_idx]
    mean = x[idx].mean(axis=0)
    std = x[idx].std(axis=0) + 1e-8
    xp = (xp - mean) / std
    xn = (xn - mean) / std
    out: List[Dict[str, Any]] = []
    for j, ni in enumerate(neg_idx):
        d = ((xp - xn[j:j+1]) ** 2).sum(axis=1)
        k = int(np.argmin(d))
        pi = int(pos_idx[k])
        out.append({
            "negative_label_row_id": rows[int(ni)].get("label_row_id", ""),
            "negative_group_key": rows[int(ni)].get("group_key", group_key(rows[int(ni)])),
            "negative_suite": rows[int(ni)].get("suite", ""),
            "negative_status": label_status(rows[int(ni)]),
            "negative_role": event_role(rows[int(ni)]),
            "nearest_primary_label_row_id": rows[pi].get("label_row_id", ""),
            "nearest_primary_group_key": rows[pi].get("group_key", group_key(rows[pi])),
            "nearest_primary_suite": rows[pi].get("suite", ""),
            "distance_sq": float(d[k]),
        })
    return sorted(out, key=lambda r: float(r["distance_sq"]))[:limit]


def add_violation(rows: List[Dict[str, Any]], code: str, severity: str, detail: str) -> None:
    rows.append({"violation_code": code, "severity": severity, "detail": detail})


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("D4B0 is CPU-only; use --device cpu")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    dataset_path = Path(args.frozen_dataset)
    rows_raw = read_csv(dataset_path)
    if len(rows_raw) != args.expected_rows:
        raise ValueError(f"dataset rows={len(rows_raw)} expected={args.expected_rows}")
    false_emit_keys = load_false_emit_group_keys(args.d6d_false_emit_groups or [])
    labeled_rows: List[Dict[str, Any]] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    skipped_nonfinite = 0
    ignored_unknown = 0
    for i, row in enumerate(rows_raw):
        y, cls = runtime_label(row)
        if y < 0:
            ignored_unknown += 1
            continue
        x, bad = finite_feature_vector(row)
        if bad:
            skipped_nonfinite += 1
            continue
        g = group_key(row)
        split = str(row.get("split", "")) or "train"
        rec = {
            "source_row_index": i,
            "label_row_id": row.get("label_row_id", f"row_{i:06d}"),
            "record_id": row.get("record_id", ""),
            "group_key": g,
            "suite": row.get("suite", ""),
            "split": split,
            "teacher_label_status": label_status(row),
            "event_role": event_role(row),
            "runtime_objective_label": y,
            "runtime_objective_class": cls,
            "is_d6d_false_emit_group": int(g in false_emit_keys),
            "has_imputation": int(has_imputation(row)),
            "imputed_feature_count": row.get("imputed_feature_count", ""),
        }
        labeled_rows.append(rec)
        features.append(x)
        labels.append(y)
    if not features:
        raise ValueError("no labeled finite rows")
    x_all = np.stack(features).astype(np.float32)
    y_all = np.asarray(labels, dtype=np.int64)
    splits = np.asarray([r["split"] for r in labeled_rows])
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(f"missing split rows: {Counter(splits.tolist())}")
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0) + 1e-8
    x_train = (x_all[train_mask] - mean) / std
    x_val = (x_all[val_mask] - mean) / std
    model, train_info = train_model(x_train, y_all[train_mask], x_val, y_all[val_mask], args)
    probs = predict(model, x_all, mean, std)
    val_threshold, val_choice = choose_threshold(y_all[val_mask], probs[val_mask], args.min_val_recall)
    metrics_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask), ("all", np.ones(len(y_all), dtype=bool))]:
        metrics_rows.append({"subset": f"split={split_name}", "threshold": val_threshold, **binary_metrics(y_all[mask], probs[mask], val_threshold)})
        for m in threshold_sweep(y_all[mask], probs[mask], [round(v, 3) for v in np.linspace(0.05, 0.95, 19)]):
            sweep_rows.append({"subset": f"split={split_name}", **m})
    suites = sorted({str(r.get("suite", "")) for r in labeled_rows})
    for suite in suites:
        mask = np.asarray([r["suite"] == suite for r in labeled_rows], dtype=bool)
        metrics_rows.append({"subset": f"suite={suite}", "threshold": val_threshold, **binary_metrics(y_all[mask], probs[mask], val_threshold)})
        tmask = mask & test_mask
        if tmask.any():
            metrics_rows.append({"subset": f"split=test|suite={suite}", "threshold": val_threshold, **binary_metrics(y_all[tmask], probs[tmask], val_threshold)})
    hn_mask = np.asarray([int(r["is_d6d_false_emit_group"]) == 1 for r in labeled_rows], dtype=bool)
    if hn_mask.any():
        metrics_rows.append({"subset": "d6d_false_emit_groups", "threshold": val_threshold, **binary_metrics(y_all[hn_mask], probs[hn_mask], val_threshold)})
    imputed_mask = np.asarray([int(r["has_imputation"]) == 1 for r in labeled_rows], dtype=bool)
    if imputed_mask.any():
        metrics_rows.append({"subset": "imputed=true", "threshold": val_threshold, **binary_metrics(y_all[imputed_mask], probs[imputed_mask], val_threshold)})
        metrics_rows.append({"subset": "imputed=false", "threshold": val_threshold, **binary_metrics(y_all[~imputed_mask], probs[~imputed_mask], val_threshold)})
    for r, p in zip(labeled_rows, probs.tolist()):
        r["runtime_emit_prob"] = p
        r["runtime_emit_pred_at_val_threshold"] = int(p >= val_threshold)
    hard_negative_rows = [r for r in labeled_rows if int(r["runtime_objective_label"]) == 0 and (int(r["runtime_emit_pred_at_val_threshold"]) == 1 or int(r["is_d6d_false_emit_group"]) == 1)]
    for r in hard_negative_rows:
        r["hard_negative_reason"] = "d6d_false_emit_group" if int(r["is_d6d_false_emit_group"]) == 1 else "upper_bound_false_positive"
        r["recommended_weight"] = args.hard_negative_weight
    nn_rows = nearest_neighbor_overlap(labeled_rows, x_all, y_all, test_mask, args.nn_limit)
    violations: List[Dict[str, Any]] = []
    test_metrics = next(r for r in metrics_rows if r["subset"] == "split=test")
    if float(test_metrics["recall"]) < args.min_test_recall:
        add_violation(violations, "LOW_TEST_RUNTIME_RECALL", "HOLD", f"recall={test_metrics['recall']}")
    if float(test_metrics["no_primary_fp_rate"]) > args.max_test_no_primary_fp_rate:
        add_violation(violations, "HIGH_TEST_NO_PRIMARY_FP_RATE", "HOLD", f"fp_rate={test_metrics['no_primary_fp_rate']}")
    for suite in suites:
        row = next((m for m in metrics_rows if m["subset"] == f"split=test|suite={suite}"), None)
        if row and int(row.get("no_primary_groups", row.get("tn", 0) + row.get("fp", 0)) or 0) == 0:
            add_violation(violations, "SUITE_TEST_HAS_NO_NEGATIVES", "WARN", f"suite={suite}")
        if row and float(row.get("no_primary_fp_rate", 0.0)) > args.max_suite_no_primary_fp_rate:
            add_violation(violations, "HIGH_SUITE_NO_PRIMARY_FP_RATE", "HOLD", f"suite={suite} fp_rate={row.get('no_primary_fp_rate')}")
    if hn_mask.any():
        hn_metrics = next(r for r in metrics_rows if r["subset"] == "d6d_false_emit_groups")
        if float(hn_metrics["no_primary_fp_rate"]) > args.max_hard_negative_fp_rate:
            add_violation(violations, "HARD_NEGATIVES_NOT_SEPARABLE", "HOLD", f"fp_rate={hn_metrics['no_primary_fp_rate']}")
    hard = [v for v in violations if v["severity"] == "HOLD"]
    status = PASS if not hard else HOLD
    reason = "" if not hard else f"hard_violation_count={len(hard)}"
    write_csv(out / "d4b0_row_labels.csv", labeled_rows, sorted({k for r in labeled_rows for k in r.keys()}))
    write_csv(out / "d4b0_metrics_by_subset.csv", metrics_rows, sorted({k for r in metrics_rows for k in r.keys()}))
    write_csv(out / "d4b0_threshold_sweep.csv", sweep_rows, sorted({k for r in sweep_rows for k in r.keys()}))
    write_csv(out / "d4b0_nearest_neighbor_overlap.csv", nn_rows, ["negative_label_row_id", "negative_group_key", "negative_suite", "negative_status", "negative_role", "nearest_primary_label_row_id", "nearest_primary_group_key", "nearest_primary_suite", "distance_sq"])
    write_csv(out / "d4b0_hard_negative_manifest.csv", hard_negative_rows, sorted({k for r in hard_negative_rows for k in r.keys()}))
    write_csv(out / "d4b0_violations.csv", violations, ["violation_code", "severity", "detail"])
    ckpt = {"model_state": model.state_dict(), "feature_names": list(SC5_V2_FEATURES), "mean": mean.astype(np.float32), "std": std.astype(np.float32), "threshold": val_threshold, "dataset_sha256": sha256_file(dataset_path), "train_info": train_info, "gate": GATE}
    torch.save(ckpt, out / "d4b0_upper_bound_checkpoint.pt")
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "frozen_dataset": str(dataset_path),
        "frozen_dataset_sha256": sha256_file(dataset_path),
        "expected_rows": args.expected_rows,
        "source_row_count": len(rows_raw),
        "labeled_finite_row_count": len(labeled_rows),
        "ignored_unknown_label_count": ignored_unknown,
        "skipped_nonfinite_count": skipped_nonfinite,
        "feature_count": len(SC5_V2_FEATURES),
        "runtime_objective_counts": dict(Counter(int(y) for y in labels)),
        "suite_counts": dict(Counter(r["suite"] for r in labeled_rows)),
        "split_counts": dict(Counter(r["split"] for r in labeled_rows)),
        "d6d_false_emit_group_key_count": len(false_emit_keys),
        "d6d_false_emit_labeled_row_count": int(hn_mask.sum()) if len(labeled_rows) else 0,
        "selected_threshold": val_threshold,
        "val_threshold_choice": val_choice,
        "test_metrics": test_metrics,
        "train_info": train_info,
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "audit_thresholds": {"min_val_recall": args.min_val_recall, "min_test_recall": args.min_test_recall, "max_test_no_primary_fp_rate": args.max_test_no_primary_fp_rate, "max_suite_no_primary_fp_rate": args.max_suite_no_primary_fp_rate, "max_hard_negative_fp_rate": args.max_hard_negative_fp_rate},
        "interpretation": "CPU-only upper-bound separability audit for explicit runtime_emit_eligible vs no_primary_suppress objective. This does not train the project detector, run LIBERO, load OpenVLA, execute rollout, intervention, or attack.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "device": "cpu", "OpenVLA_model": "NOT_LOADED", "LIBERO_runtime": "NOT_PERFORMED", "env_reset": "NOT_PERFORMED", "env_set_init_state": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "DIAGNOSTIC_UPPER_BOUND_ONLY_NOT_PROJECT_DETECTOR"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "d4b0_runtime_objective_separability_report.json", report)
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-dataset", required=True)
    p.add_argument("--d6d-false-emit-groups", action="append", default=[])
    p.add_argument("--expected-rows", type=int, default=3717)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-val-recall", type=float, default=0.80)
    p.add_argument("--min-test-recall", type=float, default=0.70)
    p.add_argument("--max-test-no-primary-fp-rate", type=float, default=0.30)
    p.add_argument("--max-suite-no-primary-fp-rate", type=float, default=0.40)
    p.add_argument("--max-hard-negative-fp-rate", type=float, default=0.40)
    p.add_argument("--hard-negative-weight", type=float, default=4.0)
    p.add_argument("--nn-limit", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
