#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import (
    SC5MultiSuiteMLP,
    SC5_V2_EVENT_ROLES,
    SC5_V2_FEATURES,
    SC5_V2_PHASES,
    validate_no_forbidden_inputs,
)

GATE = "D5_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_POST_TRAINING_AUDIT"
PASS = "PASS_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_POST_TRAINING_AUDITED"
OUT_FILES = [
    "clean2000_multisuite_sc5_detector_v2_post_training_audit_report.json",
    "d5_metrics_by_subset.csv",
    "d5_corridor_threshold_sweep.csv",
    "d5_event_role_confusion.csv",
    "d5_phase_confusion.csv",
    "d5_prediction_rows.csv",
    "d5_violations.csv",
    "checksum_report.json",
]


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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def stable_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def bool_label(value: Any) -> int:
    s = str(value).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y", "positive", "valid"}:
        return 1
    if s in {"0", "0.0", "false", "no", "n", "negative", "invalid", ""}:
        return 0
    raise ValueError(f"cannot parse binary label from {value!r}")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def imputed(row: Dict[str, Any]) -> bool:
    if truthy(row.get("has_imputation")):
        return True
    try:
        return int(float(row.get("imputed_feature_count", 0) or 0)) > 0
    except Exception:
        return False


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if y_true.size == 0:
        return {"n": 0}
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
    positive_rate = float(y_pred.mean()) if y_pred.size else 0.0
    true_positive_rate = float(y_true.mean()) if y_true.size else 0.0
    return {"n": int(y_true.size), "acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "pred_positive_rate": positive_rate, "true_positive_rate": true_positive_rate}


def categorical_acc(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return math.nan
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


class RowDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.rows[idx]
        x = np.asarray([stable_float(row.get(fn)) for fn in SC5_V2_FEATURES], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-8)
        return torch.tensor(x, dtype=torch.float32)


def load_model(checkpoint_path: Path) -> Tuple[SC5MultiSuiteMLP, np.ndarray, np.ndarray, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feature_names = list(ckpt.get("feature_names", []))
    if feature_names != list(SC5_V2_FEATURES):
        raise ValueError(f"checkpoint feature mismatch: {feature_names} != {list(SC5_V2_FEATURES)}")
    model = SC5MultiSuiteMLP(n_feat=len(SC5_V2_FEATURES), hidden=int(ckpt.get("hidden", 64)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return model, mean, std, ckpt


def predict_rows(model: SC5MultiSuiteMLP, rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray, batch_size: int) -> List[Dict[str, Any]]:
    ds = RowDataset(rows, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for x in loader:
            out = model(x)
            phase_idx = out["phase_logits"].argmax(dim=1).cpu().numpy().tolist()
            role_idx = out["event_role_logits"].argmax(dim=1).cpu().numpy().tolist()
            corr_prob = torch.sigmoid(out["corridor_logit"]).reshape(-1).cpu().numpy().tolist()
            rel_prob = torch.sigmoid(out["release_logit"]).reshape(-1).cpu().numpy().tolist()
            for j in range(len(phase_idx)):
                row = rows[offset + j]
                preds.append({
                    "label_row_id": row.get("label_row_id", f"row_{offset+j:06d}"),
                    "record_id": row.get("record_id", ""),
                    "split": row.get("split", ""),
                    "suite": row.get("suite", ""),
                    "teacher_label_status": row.get("teacher_label_status", row.get("label_status", "")),
                    "event_role_true": row.get("event_role", ""),
                    "event_role_pred": SC5_V2_EVENT_ROLES[int(role_idx[j])],
                    "phase_true": row.get("phase_label", row.get("phase", "")),
                    "phase_pred": SC5_V2_PHASES[int(phase_idx[j])],
                    "corridor_true": bool_label(row.get("corridor_label", row.get("hazard_label", 0))),
                    "corridor_prob": float(corr_prob[j]),
                    "release_true": bool_label(row.get("release_safe_label", 0)),
                    "release_prob": float(rel_prob[j]),
                    "has_imputation": int(imputed(row)),
                    "imputed_feature_count": int(float(row.get("imputed_feature_count", 0) or 0)) if str(row.get("imputed_feature_count", 0) or 0).replace(".", "", 1).isdigit() else 0,
                })
            offset += len(phase_idx)
    return preds


def subset_metrics(name: str, rows: List[Dict[str, Any]], tau_corridor: float, tau_release: float) -> Dict[str, Any]:
    if not rows:
        return {"subset": name, "n": 0}
    phase_acc = categorical_acc([str(r["phase_true"]) for r in rows], [str(r["phase_pred"]) for r in rows])
    role_acc = categorical_acc([str(r["event_role_true"]) for r in rows], [str(r["event_role_pred"]) for r in rows])
    corr = binary_metrics(np.asarray([int(r["corridor_true"]) for r in rows]), np.asarray([float(r["corridor_prob"]) for r in rows]), tau_corridor)
    rel = binary_metrics(np.asarray([int(r["release_true"]) for r in rows]), np.asarray([float(r["release_prob"]) for r in rows]), tau_release)
    return {
        "subset": name,
        "n": len(rows),
        "phase_acc": phase_acc,
        "event_role_acc": role_acc,
        "corridor_acc": corr.get("acc"),
        "corridor_precision": corr.get("precision"),
        "corridor_recall": corr.get("recall"),
        "corridor_f1": corr.get("f1"),
        "corridor_fp": corr.get("fp"),
        "corridor_fn": corr.get("fn"),
        "corridor_pred_positive_rate": corr.get("pred_positive_rate"),
        "corridor_true_positive_rate": corr.get("true_positive_rate"),
        "release_acc": rel.get("acc"),
        "release_precision": rel.get("precision"),
        "release_recall": rel.get("recall"),
        "release_f1": rel.get("f1"),
        "release_fp": rel.get("fp"),
        "release_fn": rel.get("fn"),
        "release_pred_positive_rate": rel.get("pred_positive_rate"),
        "release_true_positive_rate": rel.get("true_positive_rate"),
    }


def confusion_rows(preds: List[Dict[str, Any]], true_key: str, pred_key: str, out_true: str, out_pred: str) -> List[Dict[str, Any]]:
    c = Counter((str(r[true_key]), str(r[pred_key]), str(r.get("split", "")), str(r.get("suite", ""))) for r in preds)
    return [{out_true: t, out_pred: p, "split": split, "suite": suite, "count": n} for (t, p, split, suite), n in sorted(c.items())]


def threshold_sweep(preds: List[Dict[str, Any]], thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    out = []
    for split in ["train", "val", "test"]:
        rows = [r for r in preds if r.get("split") == split]
        y = np.asarray([int(r["corridor_true"]) for r in rows])
        p = np.asarray([float(r["corridor_prob"]) for r in rows])
        for tau in thresholds:
            m = binary_metrics(y, p, tau)
            out.append({"split": split, "threshold": tau, **m})
    return out


def add_violation(rows: List[Dict[str, Any]], code: str, severity: str, detail: str) -> None:
    rows.append({"violation_code": code, "severity": severity, "detail": detail})


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("D5 audit is CPU-only; use --device cpu")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(SC5_V2_FEATURES)
    dataset_path = Path(args.frozen_dataset)
    training_report_path = Path(args.training_report)
    checkpoint_path = Path(args.checkpoint)
    rows = read_csv(dataset_path)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if str(training_report.get("status", "")) != "PASS_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_CPU_TRAINED":
        raise ValueError(f"training report is not D4 PASS: {training_report.get('status')}")
    if len(rows) != args.expected_rows:
        raise ValueError(f"dataset rows={len(rows)} expected={args.expected_rows}")
    model, mean, std, ckpt = load_model(checkpoint_path)
    preds = predict_rows(model, rows, mean, std, args.batch_size)
    metrics: List[Dict[str, Any]] = []
    metrics.append(subset_metrics("all", preds, args.tau_corridor, args.tau_release))
    for split in ["train", "val", "test"]:
        metrics.append(subset_metrics(f"split={split}", [r for r in preds if r.get("split") == split], args.tau_corridor, args.tau_release))
    for split in ["train", "val", "test"]:
        for suite in sorted(set(str(r.get("suite", "")) for r in preds)):
            metrics.append(subset_metrics(f"split={split}|suite={suite}", [r for r in preds if r.get("split") == split and r.get("suite") == suite], args.tau_corridor, args.tau_release))
    for label in sorted(set(str(r.get("teacher_label_status", "")) for r in preds)):
        metrics.append(subset_metrics(f"label={label}", [r for r in preds if r.get("teacher_label_status") == label], args.tau_corridor, args.tau_release))
    for role in sorted(set(str(r.get("event_role_true", "")) for r in preds)):
        metrics.append(subset_metrics(f"role={role}", [r for r in preds if r.get("event_role_true") == role], args.tau_corridor, args.tau_release))
    metrics.append(subset_metrics("imputed=true", [r for r in preds if int(r.get("has_imputation", 0)) == 1], args.tau_corridor, args.tau_release))
    metrics.append(subset_metrics("imputed=false", [r for r in preds if int(r.get("has_imputation", 0)) == 0], args.tau_corridor, args.tau_release))
    metrics.append(subset_metrics("suite=libero_object|label=VALID_PRIMARY|imputed=true", [r for r in preds if r.get("suite") == "libero_object" and r.get("teacher_label_status") == "VALID_PRIMARY" and int(r.get("has_imputation", 0)) == 1], args.tau_corridor, args.tau_release))
    metrics.append(subset_metrics("suite=libero_spatial|split=test", [r for r in preds if r.get("suite") == "libero_spatial" and r.get("split") == "test"], args.tau_corridor, args.tau_release))

    violations: List[Dict[str, Any]] = []
    by_name = {str(m["subset"]): m for m in metrics}
    test = by_name.get("split=test", {})
    spatial_test = by_name.get("suite=libero_spatial|split=test", {})
    imputed = by_name.get("imputed=true", {})
    non_imputed = by_name.get("imputed=false", {})
    if float(test.get("corridor_f1", 0.0) or 0.0) < args.min_test_corridor_f1:
        add_violation(violations, "LOW_TEST_CORRIDOR_F1", "HOLD", f"test_corridor_f1={test.get('corridor_f1')}")
    if float(test.get("event_role_acc", 0.0) or 0.0) < args.min_test_event_role_acc:
        add_violation(violations, "LOW_TEST_EVENT_ROLE_ACC", "WARN", f"test_event_role_acc={test.get('event_role_acc')}")
    if float(spatial_test.get("corridor_acc", 0.0) or 0.0) < args.min_spatial_test_corridor_acc:
        add_violation(violations, "LOW_SPATIAL_TEST_CORRIDOR_ACC", "WARN", f"spatial_test_corridor_acc={spatial_test.get('corridor_acc')}")
    if int(imputed.get("n", 0) or 0) and int(non_imputed.get("n", 0) or 0):
        gap = abs(float(imputed.get("corridor_f1", 0.0) or 0.0) - float(non_imputed.get("corridor_f1", 0.0) or 0.0))
        if gap > args.max_imputed_nonimputed_corridor_f1_gap:
            add_violation(violations, "IMPUTED_NONIMPUTED_CORRIDOR_F1_GAP", "WARN", f"gap={gap}")
    release_positive_count = sum(int(r["release_true"]) for r in preds)
    if release_positive_count == 0:
        add_violation(violations, "RELEASE_HEAD_NO_POSITIVE_LABELS", "WARN", "release_safe head cannot be meaningfully evaluated; no positive labels in frozen dataset")

    hard = [v for v in violations if v.get("severity") == "HOLD"]
    status = PASS if not hard else "HOLD_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_POST_TRAINING_AUDIT"
    reason = "" if not hard else f"hard_violation_count={len(hard)}"
    role_conf = confusion_rows(preds, "event_role_true", "event_role_pred", "event_role_true", "event_role_pred")
    phase_conf = confusion_rows(preds, "phase_true", "phase_pred", "phase_true", "phase_pred")
    sweep = threshold_sweep(preds, [round(x, 2) for x in np.linspace(0.05, 0.95, 19)])
    prediction_fields = ["label_row_id", "record_id", "split", "suite", "teacher_label_status", "event_role_true", "event_role_pred", "phase_true", "phase_pred", "corridor_true", "corridor_prob", "release_true", "release_prob", "has_imputation", "imputed_feature_count"]
    metric_fields = sorted({k for row in metrics for k in row.keys()})
    write_csv(out / "d5_metrics_by_subset.csv", metrics, metric_fields)
    write_csv(out / "d5_corridor_threshold_sweep.csv", sweep, sorted({k for row in sweep for k in row.keys()}))
    write_csv(out / "d5_event_role_confusion.csv", role_conf, ["split", "suite", "event_role_true", "event_role_pred", "count"])
    write_csv(out / "d5_phase_confusion.csv", phase_conf, ["split", "suite", "phase_true", "phase_pred", "count"])
    write_csv(out / "d5_prediction_rows.csv", preds, prediction_fields)
    write_csv(out / "d5_violations.csv", violations, ["violation_code", "severity", "detail"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "frozen_dataset": str(dataset_path),
        "frozen_dataset_sha256": sha256_file(dataset_path),
        "training_report": str(training_report_path),
        "training_report_sha256": sha256_file(training_report_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "expected_rows": args.expected_rows,
        "row_count": len(rows),
        "split_counts": dict(Counter(str(r.get("split", "")) for r in preds)),
        "suite_counts": dict(Counter(str(r.get("suite", "")) for r in preds)),
        "teacher_label_status_counts": dict(Counter(str(r.get("teacher_label_status", "")) for r in preds)),
        "event_role_true_counts": dict(Counter(str(r.get("event_role_true", "")) for r in preds)),
        "event_role_pred_counts": dict(Counter(str(r.get("event_role_pred", "")) for r in preds)),
        "phase_true_counts": dict(Counter(str(r.get("phase_true", "")) for r in preds)),
        "phase_pred_counts": dict(Counter(str(r.get("phase_pred", "")) for r in preds)),
        "release_positive_count": release_positive_count,
        "imputed_row_count": sum(int(r.get("has_imputation", 0)) for r in preds),
        "imputed_feature_count": sum(int(r.get("imputed_feature_count", 0)) for r in preds),
        "test_metrics": test,
        "spatial_test_metrics": spatial_test,
        "imputed_metrics": imputed,
        "non_imputed_metrics": non_imputed,
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "audit_thresholds": {"min_test_corridor_f1": args.min_test_corridor_f1, "min_test_event_role_acc": args.min_test_event_role_acc, "min_spatial_test_corridor_acc": args.min_spatial_test_corridor_acc, "max_imputed_nonimputed_corridor_f1_gap": args.max_imputed_nonimputed_corridor_f1_gap},
        "interpretation": "CPU-only post-training audit. This audit evaluates the trained detector on the frozen CSV and does not run LIBERO/OpenVLA/rollout/intervention/attack.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "device": "cpu", "OpenVLA_model": "NOT_LOADED", "model_inference": "DETECTOR_MLP_ONLY_ON_FROZEN_CSV", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_multisuite_sc5_detector_v2_post_training_audit_report.json", report)
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
    p.add_argument("--training-report", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-rows", type=int, default=3717)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tau-corridor", type=float, default=0.5)
    p.add_argument("--tau-release", type=float, default=0.5)
    p.add_argument("--min-test-corridor-f1", type=float, default=0.80)
    p.add_argument("--min-test-event-role-acc", type=float, default=0.60)
    p.add_argument("--min-spatial-test-corridor-acc", type=float, default=0.65)
    p.add_argument("--max-imputed-nonimputed-corridor-f1-gap", type=float, default=0.20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
