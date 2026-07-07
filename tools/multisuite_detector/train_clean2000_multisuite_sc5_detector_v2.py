#!/usr/bin/env python3
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import (
    SC5MultiSuiteMLP,
    SC5_V2_EVENT_ROLES,
    SC5_V2_FEATURES,
    SC5_V2_PHASES,
    validate_no_forbidden_inputs,
)

GATE = "D4_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_CPU_TRAINING"
PASS = "PASS_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_CPU_TRAINED"
OUT_FILES = [
    "clean2000_multisuite_sc5_detector_v2_training_report.json",
    "metrics_by_split.csv",
    "metrics_by_suite_split.csv",
    "split_manifest.csv",
    "checkpoint.pt",
    "checksum_report.json",
]
LABEL_STATUS_ALLOWED = {"VALID_PRIMARY", "VALID_AUXILIARY", "NO_EVENT", "CLEAN_FAILURE_NO_POSITIVE", "UNSUPPORTED_MECHANISM", "AMBIGUOUS"}
TRAINABLE_LABEL_STATUS = {"VALID_PRIMARY", "VALID_AUXILIARY", "NO_EVENT", "CLEAN_FAILURE_NO_POSITIVE"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stable_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def bool_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "positive", "valid"}:
        return 1
    if s in {"0", "false", "no", "n", "negative", "invalid"}:
        return 0
    raise ValueError(f"cannot parse binary label from {value!r}")


def read_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def row_group_key(row: Dict[str, Any]) -> str:
    for key in ["episode_key", "parent_id", "run_id", "record_id", "episode_id"]:
        if row.get(key):
            return str(row[key])
    raise KeyError("dataset row missing group key: expected one of episode_key,parent_id,run_id,record_id,episode_id")


def infer_split_for_group(group_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16 ** 12)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def validate_and_build(rows: List[Dict[str, Any]], seed: int, split_column: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_no_forbidden_inputs(SC5_V2_FEATURES)
    group_to_split: Dict[str, str] = {}
    split_manifest: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        missing_features = [fn for fn in SC5_V2_FEATURES if fn not in row]
        if missing_features:
            raise KeyError(f"row {i} missing features: {missing_features[:5]}")
        label_status = str(row.get("teacher_label_status", row.get("label_status", ""))).strip()
        if label_status and label_status not in LABEL_STATUS_ALLOWED:
            raise ValueError(f"row {i} invalid label_status={label_status!r}")
        if label_status and label_status not in TRAINABLE_LABEL_STATUS:
            continue
        phase = str(row.get("phase_label", row.get("phase", ""))).strip()
        if phase not in SC5_V2_PHASES:
            raise ValueError(f"row {i} invalid phase_label={phase!r}")
        event_role = str(row.get("event_role", "primary_attackable" if label_status == "VALID_PRIMARY" else "unsupported_or_abstain")).strip()
        if event_role not in SC5_V2_EVENT_ROLES:
            raise ValueError(f"row {i} invalid event_role={event_role!r}")
        X = [stable_float(row.get(fn)) for fn in SC5_V2_FEATURES]
        if not np.all(np.isfinite(np.asarray(X, dtype=np.float32))):
            raise ValueError(f"row {i} has nonfinite feature values")
        g = row_group_key(row)
        split = str(row.get(split_column, "")).strip() if split_column else ""
        if not split:
            split = infer_split_for_group(g, seed)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"row {i} invalid split={split!r}")
        if g in group_to_split and group_to_split[g] != split:
            raise ValueError(f"group leakage: group={g!r} appears in both {group_to_split[g]} and {split}")
        group_to_split[g] = split
        new = dict(row)
        new["__split"] = split
        new["__group_key"] = g
        out_rows.append(new)
    for g, split in sorted(group_to_split.items()):
        split_manifest.append({"group_key": g, "split": split})
    return out_rows, split_manifest


class SC5Rows(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        x = np.asarray([stable_float(row.get(fn)) for fn in SC5_V2_FEATURES], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-8)
        phase = SC5_V2_PHASES.index(str(row.get("phase_label", row.get("phase", ""))))
        event_role = SC5_V2_EVENT_ROLES.index(str(row.get("event_role", "unsupported_or_abstain")))
        corridor = bool_label(row.get("corridor_label", row.get("hazard_label", 0)))
        release = bool_label(row.get("release_safe_label", 0))
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "phase": torch.tensor(phase, dtype=torch.long),
            "event_role": torch.tensor(event_role, dtype=torch.long),
            "corridor": torch.tensor([corridor], dtype=torch.float32),
            "release": torch.tensor([release], dtype=torch.float32),
        }


def split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        out[row["__split"]].append(row)
    return out


def compute_mean_std(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[stable_float(row.get(fn)) for fn in SC5_V2_FEATURES] for row in rows], dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray, batch_size: int, shuffle: bool, suite_balanced: bool) -> DataLoader:
    ds = SC5Rows(rows, mean, std)
    if suite_balanced and rows:
        suite_counts = Counter(str(r.get("suite", "UNKNOWN")) for r in rows)
        weights = [1.0 / max(1, suite_counts[str(r.get("suite", "UNKNOWN"))]) for r in rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(model: SC5MultiSuiteMLP, rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray, batch_size: int) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    loader = make_loader(rows, mean, std, batch_size, shuffle=False, suite_balanced=False)
    model.eval()
    phase_true: List[int] = []
    phase_pred: List[int] = []
    role_true: List[int] = []
    role_pred: List[int] = []
    corr_true: List[float] = []
    corr_prob: List[float] = []
    rel_true: List[float] = []
    rel_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"])
            phase_true.extend(batch["phase"].numpy().tolist())
            phase_pred.extend(out["phase_logits"].argmax(dim=1).cpu().numpy().tolist())
            role_true.extend(batch["event_role"].numpy().tolist())
            role_pred.extend(out["event_role_logits"].argmax(dim=1).cpu().numpy().tolist())
            corr_true.extend(batch["corridor"].reshape(-1).numpy().tolist())
            corr_prob.extend(torch.sigmoid(out["corridor_logit"]).reshape(-1).cpu().numpy().tolist())
            rel_true.extend(batch["release"].reshape(-1).numpy().tolist())
            rel_prob.extend(torch.sigmoid(out["release_logit"]).reshape(-1).cpu().numpy().tolist())
    phase_acc = float(np.mean(np.asarray(phase_true) == np.asarray(phase_pred)))
    role_acc = float(np.mean(np.asarray(role_true) == np.asarray(role_pred)))
    cm = binary_metrics(np.asarray(corr_true), np.asarray(corr_prob))
    rm = binary_metrics(np.asarray(rel_true), np.asarray(rel_prob))
    return {
        "n": len(rows),
        "phase_acc": phase_acc,
        "event_role_acc": role_acc,
        "corridor_acc": cm["acc"],
        "corridor_precision": cm["precision"],
        "corridor_recall": cm["recall"],
        "corridor_f1": cm["f1"],
        "release_acc": rm["acc"],
        "release_precision": rm["precision"],
        "release_recall": rm["recall"],
        "release_f1": rm["f1"],
    }


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        raise ValueError("This gated trainer is CPU-only. Use --device cpu.")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    raw_rows = read_dataset(Path(args.dataset_csv))
    dataset_sha = sha256_file(Path(args.dataset_csv))
    rows, split_manifest = validate_and_build(raw_rows, args.seed, args.split_column)
    splits = split_rows(rows)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError({k: len(v) for k, v in splits.items()})
    mean, std = compute_mean_std(splits["train"])
    model = SC5MultiSuiteMLP(n_feat=len(SC5_V2_FEATURES), hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(splits["train"], mean, std, args.batch_size, shuffle=True, suite_balanced=args.suite_balanced_sampler)
    best_val = -1.0
    best_state = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            opt.zero_grad()
            outp = model(batch["x"])
            loss = (
                F.cross_entropy(outp["phase_logits"], batch["phase"]) * args.phase_weight
                + F.binary_cross_entropy_with_logits(outp["corridor_logit"], batch["corridor"]) * args.corridor_weight
                + F.binary_cross_entropy_with_logits(outp["release_logit"], batch["release"]) * args.release_weight
                + F.cross_entropy(outp["event_role_logits"], batch["event_role"]) * args.event_role_weight
            )
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch["x"].shape[0]
            total_n += int(batch["x"].shape[0])
        val_metrics = evaluate(model, splits["val"], mean, std, args.batch_size)
        score = float(val_metrics.get("corridor_f1", 0.0)) + 0.25 * float(val_metrics.get("event_role_acc", 0.0))
        hist = {"epoch": epoch, "train_loss": total_loss / max(1, total_n), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(hist)
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metric_rows: List[Dict[str, Any]] = []
    suite_metric_rows: List[Dict[str, Any]] = []
    for split, split_data in splits.items():
        metrics = evaluate(model, split_data, mean, std, args.batch_size)
        metric_rows.append({"split": split, **metrics})
        by_suite: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in split_data:
            by_suite[str(row.get("suite", "UNKNOWN"))].append(row)
        for suite, suite_rows in sorted(by_suite.items()):
            suite_metric_rows.append({"split": split, "suite": suite, **evaluate(model, suite_rows, mean, std, args.batch_size)})

    ckpt = {
        "model_name": "SC5MultiSuiteMLP",
        "model_state": model.state_dict(),
        "feature_names": list(SC5_V2_FEATURES),
        "phase_classes": list(SC5_V2_PHASES),
        "event_role_classes": list(SC5_V2_EVENT_ROLES),
        "mean": mean,
        "std": std,
        "hidden": int(args.hidden),
        "dataset_sha256": dataset_sha,
        "split_mode": "suite_stratified_grouped_or_provided",
        "thresholds": {"tau_corridor": args.tau_corridor, "tau_release": args.tau_release, "tau_primary": args.tau_primary, "guard": args.guard},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(ckpt, out / "checkpoint.pt")
    write_csv(out / "metrics_by_split.csv", metric_rows, sorted({k for row in metric_rows for k in row.keys()}))
    write_csv(out / "metrics_by_suite_split.csv", suite_metric_rows, sorted({k for row in suite_metric_rows for k in row.keys()}))
    write_csv(out / "split_manifest.csv", split_manifest, ["group_key", "split"])
    report = {
        "gate": GATE,
        "status": PASS,
        "dataset_csv": str(args.dataset_csv),
        "dataset_sha256": dataset_sha,
        "raw_rows": len(raw_rows),
        "trainable_rows": len(rows),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "suite_counts_by_split": {split: dict(Counter(str(r.get("suite", "UNKNOWN")) for r in data)) for split, data in splits.items()},
        "feature_names": list(SC5_V2_FEATURES),
        "phase_classes": list(SC5_V2_PHASES),
        "event_role_classes": list(SC5_V2_EVENT_ROLES),
        "best_val_score": best_val,
        "history_tail": history[-5:],
        "metrics_by_split": metric_rows,
        "metrics_by_suite_split": suite_metric_rows,
        "checkpoint": str(out / "checkpoint.pt"),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "device": "cpu",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_multisuite_sc5_detector_v2_training_report.json", report)
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    sums = out / "SHA256SUMS"
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--split-column", default="split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--phase-weight", type=float, default=1.0)
    p.add_argument("--corridor-weight", type=float, default=1.0)
    p.add_argument("--release-weight", type=float, default=0.5)
    p.add_argument("--event-role-weight", type=float, default=0.5)
    p.add_argument("--suite-balanced-sampler", action="store_true")
    p.add_argument("--tau-corridor", type=float, default=0.3)
    p.add_argument("--tau-release", type=float, default=0.3)
    p.add_argument("--tau-primary", type=float, default=0.5)
    p.add_argument("--guard", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
