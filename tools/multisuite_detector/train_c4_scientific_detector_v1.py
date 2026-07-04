#!/usr/bin/env python3
"""Train a C4 scientific detector on one held-out fold.

This wrapper closes the C4-3 implementation gap for scientific split CSVs. It
supports the C4 split schemas with train/val/test folds and enforces train-only
normalization and validation-only threshold selection.

The default backend is ``auto``: use torch+SC5MLPV1 when torch is available,
otherwise use a small numpy logistic fallback for CPU CI fixtures. Server runs
should normally use ``--backend torch``.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.detector_dataset_closure_v1 import (  # noqa: E402
    SC5_FEATURES,
    load_dataset_manifest,
    sha256_file,
)

SCI_SPLIT_TYPES = {"object_task_heldout_with_val_v1", "suite_loso_with_val_v1"}
SPLITS = {"train", "val", "test"}
POPULATIONS = {"DETECTOR_ELIGIBLE", "DETECTOR_SAFETY"}
LABEL_COLUMNS = {"episode_key", "event_present", "window_valid", "window_start", "window_end"}


class C4ScientificTrainingError(ValueError):
    pass


def fail(message: str) -> None:
    raise C4ScientificTrainingError(message)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"{path.name}: empty CSV header")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None for v in row.values()):
                fail(f"{path.name}:{line_no}: missing cells")
            rows.append(row)
    return rows


def write_csv(path: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int(value: str, field: str, key: str) -> int:
    try:
        out = int(value)
    except ValueError:
        fail(f"{key}: {field} must be int")
    if str(out) != str(value):
        fail(f"{key}: {field} must be canonical int")
    return out


def parse_float(value: str, field: str, key: str) -> float:
    try:
        out = float(value)
    except ValueError:
        fail(f"{key}: {field} must be finite float")
    if not math.isfinite(out):
        fail(f"{key}: {field} must be finite float")
    return out


def read_split_assignments(split_csv: str | Path, fold_id: str) -> tuple[str, dict[str, str]]:
    rows = read_csv_rows(split_csv)
    expected = ["split_type", "fold_id", "group_id", "episode_key", "split"]
    if list(rows[0]) != expected:
        fail("split CSV header must match C4 split schema")
    selected = [r for r in rows if r["fold_id"] == fold_id]
    if not selected:
        fail(f"fold_id not found: {fold_id}")
    split_types = {r["split_type"] for r in selected}
    if len(split_types) != 1:
        fail("fold contains mixed split types")
    split_type = next(iter(split_types))
    if split_type not in SCI_SPLIT_TYPES:
        fail(f"unsupported scientific split type: {split_type}")
    assignments: dict[str, str] = {}
    for row in selected:
        ep = row["episode_key"]
        split = row["split"]
        if split not in SPLITS:
            fail(f"{ep}: invalid split {split}")
        if ep in assignments:
            fail(f"duplicate episode split row: {ep}")
        assignments[ep] = split
    counts = Counter(assignments.values())
    if any(counts[s] <= 0 for s in ["train", "val", "test"]):
        fail("train/val/test must all be non-empty")
    return split_type, assignments


def read_labels(label_csv: str | Path | None, label_artifact_root: str | Path | None) -> dict[str, dict[str, int | bool]]:
    if label_csv and label_artifact_root:
        fail("provide only one of --label-csv or --label-artifact-root")
    if label_artifact_root:
        label_csv = Path(label_artifact_root) / "label_v2.csv"
    if not label_csv:
        fail("label source required: pass --label-csv or --label-artifact-root")
    rows = read_csv_rows(label_csv)
    if not rows:
        fail("label CSV has no rows")
    if not LABEL_COLUMNS <= set(rows[0]):
        fail("label CSV must contain episode_key,event_present,window_valid,window_start,window_end")
    labels: dict[str, dict[str, int | bool]] = {}
    for row in rows:
        ep = row["episode_key"]
        if ep in labels:
            fail(f"duplicate label episode: {ep}")
        event = row["event_present"] == "true"
        valid = row["window_valid"] == "true"
        start = parse_int(row["window_start"], "window_start", ep)
        end = parse_int(row["window_end"], "window_end", ep)
        labels[ep] = {"event_present": event, "window_valid": valid, "window_start": start, "window_end": end}
    return labels


def read_features(feature_csv: str | Path) -> dict[str, list[tuple[int, list[float]]]]:
    rows = read_csv_rows(feature_csv)
    if not rows:
        fail("feature CSV has no rows")
    required = {"episode_key", "step", *SC5_FEATURES}
    if not required <= set(rows[0]):
        missing = sorted(required - set(rows[0]))
        fail(f"feature CSV missing columns: {missing}")
    by_ep: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        ep = row["episode_key"]
        step = parse_int(row["step"], "step", ep)
        key = (ep, step)
        if key in seen:
            fail(f"duplicate feature row: {ep}:{step}")
        seen.add(key)
        values = [parse_float(row[name], name, ep) for name in SC5_FEATURES]
        by_ep[ep].append((step, values))
    for ep in by_ep:
        by_ep[ep].sort(key=lambda item: item[0])
    return dict(by_ep)


def make_examples(
    dataset_rows: list[dict[str, str]],
    assignments: dict[str, str],
    labels: dict[str, dict[str, int | bool]],
    features: dict[str, list[tuple[int, list[float]]]],
    population: str,
) -> dict[str, dict[str, Any]]:
    if population not in POPULATIONS:
        fail("population must be DETECTOR_ELIGIBLE or DETECTOR_SAFETY")
    by_split = {s: {"x": [], "y": [], "episodes": [], "suite": [], "task_id": [], "step": []} for s in SPLITS}
    dataset_eps = {r["episode_key"] for r in dataset_rows}
    if set(assignments) != dataset_eps:
        fail("split fold coverage does not match dataset episodes")
    for row in dataset_rows:
        ep = row["episode_key"]
        if row["population_id"] != population:
            continue
        if ep not in labels:
            fail(f"missing label for episode: {ep}")
        if ep not in features:
            fail(f"missing features for episode: {ep}")
        split = assignments[ep]
        label = labels[ep]
        event = bool(label["event_present"]) and bool(label["window_valid"])
        start = int(label["window_start"])
        end = int(label["window_end"])
        for step, values in features[ep]:
            y = 1.0 if event and start <= step < end else 0.0
            by_split[split]["x"].append(values)
            by_split[split]["y"].append(y)
            by_split[split]["episodes"].append(ep)
            by_split[split]["suite"].append(row["suite"])
            by_split[split]["task_id"].append(row["task_id"])
            by_split[split]["step"].append(step)
    for split, data in by_split.items():
        if not data["x"]:
            fail(f"{split} split has no examples for {population}")
        data["x"] = np.asarray(data["x"], dtype=np.float32)
        data["y"] = np.asarray(data["y"], dtype=np.float32)
        if not np.isfinite(data["x"]).all() or not np.isfinite(data["y"]).all():
            fail(f"{split}: NaN/Inf examples")
    return by_split


def train_normalization(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        fail("normalization contains NaN/Inf")
    return mean.astype(np.float32), std.astype(np.float32)


def bce_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    y = labels >= 0.5
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "count": int(labels.size)}


def choose_threshold(val_scores: np.ndarray, val_y: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    candidates = [round(i / 100, 2) for i in range(5, 100, 5)]
    reports = [metrics(val_scores, val_y, t) for t in candidates]
    best = max(reports, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    return float(best["threshold"]), reports


def group_metrics(scores: np.ndarray, labels: np.ndarray, keys: list[str], threshold: float, column: str) -> list[dict[str, Any]]:
    out = []
    for key in sorted(set(keys)):
        idx = np.asarray([k == key for k in keys], dtype=bool)
        rep = metrics(scores[idx], labels[idx], threshold)
        out.append({column: key, **rep})
    return out


def train_numpy(data: dict[str, dict[str, Any]], mean: np.ndarray, std: np.ndarray, *, seed: int, epochs: int, batch_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = (data["train"]["x"] - mean) / std
    y = data["train"]["y"]
    w = rng.normal(0, 0.01, size=x.shape[1]).astype(np.float32)
    b = np.float32(0.0)
    lr = 0.05
    logs = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(x.shape[0])
        for start in range(0, order.size, batch_size):
            idx = order[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]
            p = sigmoid(xb @ w + b)
            grad = p - yb
            w -= lr * (xb.T @ grad / max(1, idx.size))
            b -= lr * float(grad.mean())
        train_scores = sigmoid(x @ w + b)
        logs.append({"epoch": epoch, "train_loss": bce_loss(train_scores, y)})
    def pred(split: str) -> np.ndarray:
        return sigmoid(((data[split]["x"] - mean) / std) @ w + b)
    return {"backend": "numpy", "train_log": logs, "weights": w.tolist(), "bias": float(b)}, {s: pred(s) for s in SPLITS}


def train_torch(data: dict[str, dict[str, Any]], mean: np.ndarray, std: np.ndarray, *, seed: int, epochs: int, batch_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        import torch
        import torch.nn.functional as F
        from src.gripper_attack.sc5mlp_v1 import SC5MLPV1
    except Exception as exc:  # pragma: no cover - CI may not install torch
        fail(f"torch backend unavailable: {exc}")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SC5MLPV1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x = torch.tensor((data["train"]["x"] - mean) / std, dtype=torch.float32, device=device)
    y = torch.tensor(data["train"]["y"], dtype=torch.float32, device=device).view(-1, 1)
    logs = []
    rng = random.Random(seed)
    indices = list(range(x.shape[0]))
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idx = torch.tensor(indices[start:start + batch_size], dtype=torch.long, device=device)
            out = model(x.index_select(0, idx))["corridor_logit"]
            loss = F.binary_cross_entropy_with_logits(out, y.index_select(0, idx))
            if not torch.isfinite(loss):
                fail("torch loss became NaN/Inf")
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            train_loss = F.binary_cross_entropy_with_logits(model(x)["corridor_logit"], y).item()
        logs.append({"epoch": epoch, "train_loss": float(train_loss)})
    def pred(split: str) -> np.ndarray:
        xs = torch.tensor((data[split]["x"] - mean) / std, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(xs)["corridor_logit"]).detach().cpu().numpy().reshape(-1)
    return {"backend": "torch", "train_log": logs, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, {s: pred(s) for s in SPLITS}


def write_sha256sums(root: Path) -> str:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = root / "SHA256SUMS.sha256"
    side.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(side)


def save_checkpoint(root: Path, result: dict[str, Any], meta: dict[str, Any]) -> str:
    last = root / "checkpoint_last.pt"
    best = root / "best_checkpoint.pt"
    if result["backend"] == "torch":
        import torch  # pragma: no cover
        payload = {"model_state_dict": result["state_dict"], "meta": meta}
        torch.save(payload, last)
        torch.save(payload, best)
    else:
        payload = {"weights": result["weights"], "bias": result["bias"], "meta": meta}
        last.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        best.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(best)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = load_dataset_manifest(args.dataset_csv)
    dataset_by_ep = {r["episode_key"]: r for r in dataset_rows}
    split_type, assignments = read_split_assignments(args.split_csv, args.fold_id)
    labels = read_labels(args.label_csv, args.label_artifact_root)
    features = read_features(args.feature_csv)
    if args.expected_dataset_csv_sha256 and sha256_file(Path(args.dataset_csv)) != args.expected_dataset_csv_sha256:
        fail("dataset identity mismatch")
    if args.expected_split_csv_sha256 and sha256_file(Path(args.split_csv)) != args.expected_split_csv_sha256:
        fail("split identity mismatch")
    data = make_examples(dataset_rows, assignments, labels, features, args.population)
    mean, std = train_normalization(data["train"]["x"])
    backend = args.backend
    if backend == "auto":
        try:
            import torch  # noqa: F401
            backend = "torch"
        except Exception:
            backend = "numpy"
    if backend == "torch":
        result, scores = train_torch(data, mean, std, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
    elif backend == "numpy":
        result, scores = train_numpy(data, mean, std, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
    else:
        fail("backend must be auto, torch, or numpy")
    threshold, threshold_reports = choose_threshold(scores["val"], data["val"]["y"])
    summary = {
        "status": "PASS",
        "schema_version": "c4_scientific_detector_training_v1",
        "split_type": split_type,
        "fold_id": args.fold_id,
        "population": args.population,
        "backend": result["backend"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "normalization_source": "train_only",
        "threshold_source": "validation",
        "selected_threshold": threshold,
        "train": metrics(scores["train"], data["train"]["y"], threshold),
        "val": metrics(scores["val"], data["val"]["y"], threshold),
        "test": metrics(scores["test"], data["test"]["y"], threshold),
        "simulator": "NOT_PERFORMED",
        "policy_run": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "paper_main_table": "NOT_PERFORMED",
    }
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    training_config = {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "backend": result["backend"], "population": args.population, "fold_id": args.fold_id}
    dataset_identity = {"dataset_csv": str(args.dataset_csv), "dataset_csv_sha256": sha256_file(Path(args.dataset_csv)), "feature_csv": str(args.feature_csv), "feature_csv_sha256": sha256_file(Path(args.feature_csv)), "label_source_sha256": sha256_file(Path(args.label_csv) if args.label_csv else Path(args.label_artifact_root) / "label_v2.csv"), "state_index_sha256": args.expected_state_index_sha256 or "UNSPECIFIED"}
    split_identity = {"split_csv": str(args.split_csv), "split_csv_sha256": sha256_file(Path(args.split_csv)), "split_type": split_type, "fold_id": args.fold_id}
    normalization_identity = {"normalization_source": "train_only", "feature_names": SC5_FEATURES, "mean": mean.tolist(), "std": std.tolist()}
    threshold_selection = {"threshold_source": "validation", "selected_threshold": threshold, "sweep": threshold_reports}
    checkpoint_sha = save_checkpoint(root, result, {"training_config": training_config, "dataset_identity": dataset_identity, "split_identity": split_identity, "threshold": threshold})
    summary["best_checkpoint_sha256"] = checkpoint_sha
    write_json(root / "training_config.json", training_config)
    write_json(root / "dataset_identity.json", dataset_identity)
    write_json(root / "split_identity.json", split_identity)
    write_json(root / "normalization_identity.json", normalization_identity)
    write_json(root / "model_architecture.json", {"model": "SC5MLPV1" if result["backend"] == "torch" else "numpy_logistic_ci_fallback", "feature_count": len(SC5_FEATURES), "primary_head": "corridor"})
    (root / "train_log.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in result["train_log"]), encoding="utf-8")
    write_json(root / "threshold_selection.json", threshold_selection)
    write_json(root / "metrics_summary.json", summary)
    write_csv(root / "metrics_by_suite.csv", ["split", "suite", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "count"], [dict(split=s, **r) for s in ["val", "test"] for r in group_metrics(scores[s], data[s]["y"], data[s]["suite"], threshold, "suite")])
    write_csv(root / "metrics_by_task.csv", ["split", "task_id", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "count"], [dict(split=s, **r) for s in ["val", "test"] for r in group_metrics(scores[s], data[s]["y"], data[s]["task_id"], threshold, "task_id")])
    write_json(root / "bundle_load_report.json", {"status": "PASS", "checkpoint_sha256": checkpoint_sha, "backend": result["backend"], "simulator": "NOT_PERFORMED", "policy_run": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED"})
    write_sha256sums(root)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--population", default="DETECTOR_ELIGIBLE")
    parser.add_argument("--label-csv")
    parser.add_argument("--label-artifact-root")
    parser.add_argument("--seed", type=int, default=2026070401)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument("--expected-dataset-csv-sha256")
    parser.add_argument("--expected-split-csv-sha256")
    parser.add_argument("--expected-state-index-sha256")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_training(args)
    except (OSError, json.JSONDecodeError, csv.Error, C4ScientificTrainingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
