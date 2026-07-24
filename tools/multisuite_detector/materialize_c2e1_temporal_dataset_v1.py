#!/usr/bin/env python3
"""C2e1 temporal dataset materialization for the clean four-suite detector.

CPU-only materialization gate. This script does not train a detector and does
not run OpenVLA, LIBERO, env.reset/env.step, rollout, intervention, or attack.

Inputs:
- D4C1B context/runtime-objective dataset.
- C2e0C endpoint audit CSV, aligned by row_index.
- C2e0F v2 Object repair manifest.

For LIBERO Object rows, the temporal source is the repaired 25D CSV from C2e0F.
For non-Object rows, features are computed from the original temporal artifact
with probe_clean2000_detector_25d_feature_extraction_v3.compute_features.

Outputs per window W:
- NPZ tensors: X_temporal [N,W,25], X_context [N,C], y [N]
- row manifest CSV with split/suite/group/endpoint/source metadata
- exclusion manifest CSV
- train-split-only normalization stats
- global report, schema, checksums

Causality: for endpoint t and window W, materialized rows are [t-W+1, ..., t].
No future rows are used and no endpoint clamping is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))

from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs  # type: ignore
import probe_clean2000_detector_25d_feature_extraction_v3 as v3  # type: ignore

GATE = "C2E1_TEMPORAL_DATASET_MATERIALIZATION"
PASS = "PASS_C2E1_TEMPORAL_DATASET_MATERIALIZED"
HOLD = "HOLD_C2E1_TEMPORAL_DATASET_MATERIALIZATION"
PRIMARY_LABELS = {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE", "primary_attackable"}
REQUIRED_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
DEFAULT_WINDOWS = [8, 16, 32]
CONTEXT_PREFIXES = (
    "suite_onehot_",
    "ctx_suite_",
    "ctx_task_index_hash_",
    "ctx_suite_task_template_hash_",
    "ctx_task_index_onehot_",
    "suite_task_index_onehot_",
    "task_index_onehot_",
)
FORBIDDEN_SUBSTRINGS = (
    "object_pose", "target_pose", "object_target_distance", "object_to_target",
    "privileged", "sim_state", "oracle", "attack_outcome", "rollout_outcome",
    "success", "failure", "teacher_anchor", "teacher_window", "future_",
    "post_intervention", "openvla_hidden",
)
OUT_FILES_BASE = [
    "c2e1_temporal_dataset_report.json",
    "c2e1_temporal_feature_schema.json",
    "c2e1_context_feature_columns.json",
    "c2e1_source_usage_by_artifact.csv",
    "c2e1_window_summary_by_suite.csv",
    "c2e1_window_summary_by_split.csv",
    "c2e1_violations.csv",
    "checksum_report.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_dict(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def finite(value: Any) -> bool:
    return math.isfinite(finite_float(value))


def parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except Exception:
        return None


def parse_windows(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError("window sizes must be positive")
        vals.append(val)
    if not vals:
        raise ValueError("at least one window is required")
    return sorted(set(vals))


def label_status(row: Dict[str, str]) -> str:
    return str(row.get("teacher_label_status") or row.get("event_status") or row.get("event_role") or "")


def event_role(row: Dict[str, str]) -> str:
    return str(row.get("event_role") or row.get("event_role_true") or "")


def target_label(row: Dict[str, str]) -> int:
    for key in ["runtime_objective_label", "label", "y", "target"]:
        val = parse_int(row.get(key))
        if val in (0, 1):
            return int(val)
    status = label_status(row)
    role = event_role(row)
    if status in PRIMARY_LABELS or role in PRIMARY_LABELS:
        return 1
    return 0


def group_class(row: Dict[str, str]) -> str:
    return "has_primary" if target_label(row) == 1 else "no_primary"


def detect_context_columns(header: Sequence[str]) -> List[str]:
    cols = []
    for name in header:
        s = str(name)
        if s in SC5_V2_FEATURES:
            continue
        if any(s.startswith(p) for p in CONTEXT_PREFIXES):
            cols.append(s)
    return cols


def context_vector(row: Dict[str, str], columns: Sequence[str]) -> List[float]:
    vals = []
    for col in columns:
        vals.append(finite_float(row.get(col), 0.0))
    return vals


def load_endpoint_map(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in read_csv_dict(path):
        idx = parse_int(row.get("row_index"))
        if idx is not None:
            out[idx] = row
    return out


def load_repair_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_csv_dict(path):
        if str(row.get("status", "")) != "REPAIRED":
            continue
        src = row.get("original_temporal_path") or row.get("temporal_path") or ""
        dst = row.get("repaired_feature_path") or row.get("output_path") or ""
        if src and dst:
            out[src] = dst
    return out


def required_indices(endpoint: int, max_window: int, n_rows_hint: Optional[int] = None) -> List[int]:
    start = endpoint - max_window + 1
    if start < 0:
        start = 0
    out = []
    for idx in range(start, endpoint + 1):
        if n_rows_hint is None or 0 <= idx < n_rows_hint:
            out.append(idx)
    return out


def repaired_feature_rows(rows: List[Dict[str, str]], needed: Iterable[int]) -> Tuple[Dict[int, List[float]], Dict[int, str], Counter[str]]:
    feats: Dict[int, List[float]] = {}
    failures: Dict[int, str] = {}
    methods: Counter[str] = Counter()
    n = len(rows)
    for idx in needed:
        if idx < 0 or idx >= n:
            failures[idx] = f"index_oob:{idx}_n={n}"
            continue
        row = rows[idx]
        vec: List[float] = []
        missing = ""
        for feat in SC5_V2_FEATURES:
            value = finite_float(row.get(feat))
            if not math.isfinite(value):
                missing = feat
                break
            vec.append(value)
        if missing:
            failures[idx] = f"missing_{missing}"
        else:
            feats[idx] = vec
            methods["object_repaired_25d_csv"] += 1
    return feats, failures, methods


def v3_feature_rows(rows: List[Dict[str, Any]], needed: Iterable[int], event_status_by_index: Dict[int, str]) -> Tuple[Dict[int, List[float]], Dict[int, str], Counter[str]]:
    feats: Dict[int, List[float]] = {}
    failures: Dict[int, str] = {}
    methods: Counter[str] = Counter()
    n = len(rows)
    for idx in needed:
        if idx < 0 or idx >= n:
            failures[idx] = f"index_oob:{idx}_n={n}"
            continue
        event_status = event_status_by_index.get(idx, "NO_EVENT")
        try:
            values, feat_methods, _fields = v3.compute_features(rows, idx, event_status)
        except Exception as exc:
            failures[idx] = f"v3_exception:{str(exc)[:160]}"
            continue
        vec: List[float] = []
        missing = ""
        for feat in SC5_V2_FEATURES:
            value = finite_float(values.get(feat))
            if not math.isfinite(value):
                missing = feat
                break
            vec.append(value)
        if missing:
            failures[idx] = f"v3_missing_{missing}"
        else:
            feats[idx] = vec
            for feat in SC5_V2_FEATURES:
                methods[str(feat_methods.get(feat, "unknown"))] += 1
    return feats, failures, methods


def process_artifact_task(task: Dict[str, Any]) -> Dict[str, Any]:
    used_path = task["used_path"]
    used_repair = bool(task["used_repair"])
    max_window = int(task["max_window"])
    row_items = task["rows"]
    started = time.time()
    method_counts: Counter[str] = Counter()
    exclusions: List[Dict[str, Any]] = []
    output_items: List[Dict[str, Any]] = []

    try:
        if used_repair:
            trows = read_csv_dict(used_path)
        else:
            trows = v3.read_temporal(used_path)
        n_temporal_rows = len(trows)
    except Exception as exc:
        for item in row_items:
            exclusions.append({
                "row_index": item["row_index"], "suite": item["suite"], "split": item["split"],
                "group_key": item["group_key"], "temporal_path": item["temporal_path"],
                "used_path": used_path, "window": "all", "reason": f"read_error:{str(exc)[:160]}",
            })
        return {
            "used_path": used_path, "used_repair": used_repair, "n_temporal_rows": 0,
            "computed_index_count": 0, "items": output_items, "exclusions": exclusions,
            "method_counts": {}, "runtime_seconds": time.time() - started,
        }

    needed = set()
    event_status_by_index: Dict[int, str] = {}
    for item in row_items:
        endpoint = int(item["endpoint"])
        for idx in required_indices(endpoint, max_window, n_temporal_rows):
            needed.add(idx)
            event_status_by_index.setdefault(idx, item.get("label_status") or "NO_EVENT")

    if used_repair:
        feat_cache, failures, methods = repaired_feature_rows(trows, sorted(needed))
    else:
        feat_cache, failures, methods = v3_feature_rows(trows, sorted(needed), event_status_by_index)
    method_counts.update(methods)

    for item in row_items:
        endpoint = int(item["endpoint"])
        item_out = dict(item)
        item_out["used_path"] = used_path
        item_out["used_repair"] = used_repair
        item_out["n_temporal_rows"] = n_temporal_rows
        item_out["windows"] = {}
        for window in task["windows"]:
            start = endpoint - int(window) + 1
            if start < 0:
                item_out["windows"][int(window)] = {"ok": False, "reason": "endpoint_too_early", "features": None}
                continue
            if endpoint >= n_temporal_rows:
                item_out["windows"][int(window)] = {"ok": False, "reason": f"endpoint_oob:{endpoint}_n={n_temporal_rows}", "features": None}
                continue
            window_rows: List[List[float]] = []
            reason = ""
            for idx in range(start, endpoint + 1):
                if idx not in feat_cache:
                    reason = f"idx{idx}:{failures.get(idx, 'not_computed')}"
                    break
                window_rows.append(feat_cache[idx])
            if reason:
                item_out["windows"][int(window)] = {"ok": False, "reason": reason, "features": None}
            else:
                item_out["windows"][int(window)] = {"ok": True, "reason": "", "features": window_rows}
        output_items.append(item_out)

    return {
        "used_path": used_path,
        "used_repair": used_repair,
        "n_temporal_rows": n_temporal_rows,
        "computed_index_count": len(needed),
        "items": output_items,
        "exclusions": exclusions,
        "method_counts": dict(method_counts),
        "runtime_seconds": time.time() - started,
    }


def split_leakage(rows: Iterable[Dict[str, Any]]) -> int:
    group_to_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        g = str(row.get("group_key", ""))
        s = str(row.get("split", ""))
        if g and s:
            group_to_splits[g].add(s)
    return sum(1 for vals in group_to_splits.values() if len(vals) > 1)


def train_stats(x_temporal: np.ndarray, x_context: np.ndarray, split_values: List[str]) -> Dict[str, Any]:
    train_idx = np.asarray([s == "train" for s in split_values], dtype=bool)
    if not train_idx.any():
        raise ValueError("no train split rows available for normalization stats")
    xt = x_temporal[train_idx].reshape((-1, x_temporal.shape[-1])).astype(np.float64)
    xc = x_context[train_idx].astype(np.float64)
    temporal_mean = xt.mean(axis=0)
    temporal_std = xt.std(axis=0)
    temporal_std[temporal_std < 1e-8] = 1.0
    if xc.shape[1] > 0:
        context_mean = xc.mean(axis=0)
        context_std = xc.std(axis=0)
        context_std[context_std < 1e-8] = 1.0
    else:
        context_mean = np.zeros((0,), dtype=np.float64)
        context_std = np.ones((0,), dtype=np.float64)
    return {
        "fit_split": "train",
        "temporal_feature_mean": temporal_mean.tolist(),
        "temporal_feature_std": temporal_std.tolist(),
        "context_feature_mean": context_mean.tolist(),
        "context_feature_std": context_std.tolist(),
    }


def summarize_manifest(rows: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    suites = Counter(str(r.get("suite")) for r in rows)
    splits = Counter(str(r.get("split")) for r in rows)
    labels = Counter(str(r.get("label")) for r in rows)
    by_suite_label: Dict[str, Counter[str]] = defaultdict(Counter)
    by_split_label: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_suite_label[str(row.get("suite"))][str(row.get("label"))] += 1
        by_split_label[str(row.get("split"))][str(row.get("label"))] += 1
    return {
        "window": window,
        "row_count": len(rows),
        "suite_counts": dict(suites),
        "split_counts": dict(splits),
        "label_counts": dict(labels),
        "label_counts_by_suite": {k: dict(v) for k, v in by_suite_label.items()},
        "label_counts_by_split": {k: dict(v) for k, v in by_split_label.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True)
    ap.add_argument("--repair-manifest", required=True)
    ap.add_argument("--c2e0c-endpoint-csv", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--windows", default="8,16,32")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--min-suite-rate", type=float, default=0.95)
    ap.add_argument("--min-overall-rate", type=float, default=0.95)
    args = ap.parse_args()

    started = time.time()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(args.windows)
    max_window = max(windows)

    validate_no_forbidden_inputs([f"lag00_{f}" for f in SC5_V2_FEATURES])

    context_rows = read_csv_dict(args.context_dataset)
    header = list(context_rows[0].keys()) if context_rows else []
    context_cols = detect_context_columns(header)
    for col in context_cols:
        low = col.lower()
        if any(bad in low for bad in FORBIDDEN_SUBSTRINGS):
            raise ValueError(f"forbidden context column detected: {col}")

    endpoint_map = load_endpoint_map(Path(args.c2e0c_endpoint_csv).expanduser())
    repair_map = load_repair_map(Path(args.repair_manifest).expanduser())

    tasks_by_path: Dict[str, Dict[str, Any]] = {}
    pre_exclusions: List[Dict[str, Any]] = []
    repair_usage: List[Dict[str, Any]] = []

    for idx, row in enumerate(context_rows):
        suite = str(row.get("suite", "unknown"))
        split = str(row.get("split", "unknown"))
        group_key = str(row.get("group_key", ""))
        temporal_path = str(row.get("temporal_path", ""))
        endpoint = parse_int(endpoint_map.get(idx, {}).get("endpoint_index"))
        if endpoint is None:
            pre_exclusions.append({"row_index": idx, "suite": suite, "split": split, "group_key": group_key, "window": "all", "reason": "missing_endpoint", "temporal_path": temporal_path})
            continue
        if not temporal_path:
            pre_exclusions.append({"row_index": idx, "suite": suite, "split": split, "group_key": group_key, "window": "all", "reason": "missing_temporal_path", "temporal_path": temporal_path})
            continue
        used_path = repair_map.get(temporal_path, temporal_path)
        used_repair = used_path != temporal_path
        if used_repair:
            repair_usage.append({"row_index": idx, "suite": suite, "group_key": group_key, "original_path": temporal_path, "repaired_path": used_path})
        task = tasks_by_path.setdefault(used_path, {"used_path": used_path, "used_repair": used_repair, "max_window": max_window, "windows": windows, "rows": []})
        task["rows"].append({
            "row_index": idx,
            "suite": suite,
            "split": split,
            "group_key": group_key,
            "label_status": label_status(row),
            "group_class": group_class(row),
            "label": target_label(row),
            "temporal_path": temporal_path,
            "endpoint": endpoint,
            "context": context_vector(row, context_cols),
        })

    tasks = list(tasks_by_path.values())
    print(f"C2e1: context_rows={len(context_rows)} unique_artifacts={len(tasks)} workers={args.workers} windows={windows}")

    artifact_summaries: List[Dict[str, Any]] = []
    all_items: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = list(pre_exclusions)
    method_counts: Counter[str] = Counter()

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(process_artifact_task, task) for task in tasks]
        for n_done, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            all_items.extend(res["items"])
            exclusions.extend(res["exclusions"])
            method_counts.update(res.get("method_counts", {}))
            artifact_summaries.append({
                "used_path": res["used_path"],
                "used_repair": res["used_repair"],
                "n_temporal_rows": res["n_temporal_rows"],
                "row_count": len(res["items"]),
                "computed_index_count": res["computed_index_count"],
                "runtime_seconds": res["runtime_seconds"],
            })
            if n_done % 100 == 0 or n_done == len(futs):
                print(f"  completed_artifacts={n_done}/{len(futs)} rows={len(all_items)} exclusions={len(exclusions)}")

    all_items.sort(key=lambda x: int(x["row_index"]))

    window_reports: List[Dict[str, Any]] = []
    suite_summary_rows: List[Dict[str, Any]] = []
    split_summary_rows: List[Dict[str, Any]] = []
    violations: List[str] = []
    materialized_files: List[str] = []

    for window in windows:
        manifest_rows: List[Dict[str, Any]] = []
        exclusion_rows: List[Dict[str, Any]] = []
        x_temporal_list: List[np.ndarray] = []
        x_context_list: List[List[float]] = []
        y_list: List[int] = []
        row_index_list: List[int] = []
        split_values: List[str] = []
        suite_values: List[str] = []

        for item in all_items:
            wrec = item["windows"].get(window, {"ok": False, "reason": "missing_window_result", "features": None})
            base = {
                "row_index": item["row_index"],
                "suite": item["suite"],
                "split": item["split"],
                "group_key": item["group_key"],
                "label_status": item["label_status"],
                "group_class": item["group_class"],
                "label": item["label"],
                "temporal_path": item["temporal_path"],
                "used_path": item["used_path"],
                "used_repair": item["used_repair"],
                "endpoint": item["endpoint"],
                "window": window,
                "n_temporal_rows": item["n_temporal_rows"],
            }
            if not wrec.get("ok"):
                erow = dict(base)
                erow["reason"] = wrec.get("reason", "unknown")
                exclusion_rows.append(erow)
                continue
            features = np.asarray(wrec["features"], dtype=np.float32)
            if features.shape != (window, len(SC5_V2_FEATURES)) or not np.isfinite(features).all():
                erow = dict(base)
                erow["reason"] = f"bad_feature_shape_or_nonfinite:{features.shape}"
                exclusion_rows.append(erow)
                continue
            manifest_rows.append(base)
            x_temporal_list.append(features)
            x_context_list.append([float(v) for v in item["context"]])
            y_list.append(int(item["label"]))
            row_index_list.append(int(item["row_index"]))
            split_values.append(str(item["split"]))
            suite_values.append(str(item["suite"]))

        n = len(x_temporal_list)
        x_temporal = np.stack(x_temporal_list).astype(np.float32) if n else np.zeros((0, window, len(SC5_V2_FEATURES)), dtype=np.float32)
        x_context = np.asarray(x_context_list, dtype=np.float32) if n else np.zeros((0, len(context_cols)), dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int64)
        row_indices = np.asarray(row_index_list, dtype=np.int64)
        stats = train_stats(x_temporal, x_context, split_values) if n else {}

        prefix = f"c2e1_w{window:02d}"
        npz_name = f"{prefix}_temporal_dataset.npz"
        np.savez_compressed(
            out / npz_name,
            X_temporal=x_temporal,
            X_context=x_context,
            y=y,
            row_index=row_indices,
            temporal_feature_names=np.asarray(SC5_V2_FEATURES, dtype=object),
            context_feature_names=np.asarray(context_cols, dtype=object),
            suite=np.asarray(suite_values, dtype=object),
            split=np.asarray(split_values, dtype=object),
        )
        materialized_files.append(npz_name)
        write_csv(out / f"{prefix}_row_manifest.csv", manifest_rows, [
            "row_index", "suite", "split", "group_key", "label_status", "group_class", "label", "temporal_path", "used_path", "used_repair", "endpoint", "window", "n_temporal_rows",
        ])
        write_csv(out / f"{prefix}_exclusion_manifest.csv", exclusion_rows, [
            "row_index", "suite", "split", "group_key", "label_status", "group_class", "label", "temporal_path", "used_path", "used_repair", "endpoint", "window", "n_temporal_rows", "reason",
        ])
        write_json(out / f"{prefix}_normalization_stats_train_only.json", stats)

        summary = summarize_manifest(manifest_rows, window)
        summary["excluded_count"] = len(exclusion_rows)
        summary["materialized_npz"] = npz_name
        window_reports.append(summary)

        # By-suite/split rate rows include denominator = all_items + pre_exclusions considered for this window.
        by_suite_total = Counter(str(item.get("suite")) for item in all_items)
        by_suite_total.update(str(e.get("suite")) for e in pre_exclusions)
        by_suite_ok = Counter(str(r.get("suite")) for r in manifest_rows)
        for suite in sorted(by_suite_total):
            total = by_suite_total[suite]
            ok = by_suite_ok[suite]
            rate = ok / total if total else 0.0
            suite_summary_rows.append({"window": window, "suite": suite, "total": total, "materialized": ok, "excluded": total - ok, "rate": rate})
            if total > 0 and rate < args.min_suite_rate:
                violations.append(f"W{window}_SUITE_RATE_BELOW_TARGET:{suite}:{rate:.6f}")
        by_split_total = Counter(str(item.get("split")) for item in all_items)
        by_split_total.update(str(e.get("split")) for e in pre_exclusions)
        by_split_ok = Counter(str(r.get("split")) for r in manifest_rows)
        for split in sorted(by_split_total):
            total = by_split_total[split]
            ok = by_split_ok[split]
            rate = ok / total if total else 0.0
            split_summary_rows.append({"window": window, "split": split, "total": total, "materialized": ok, "excluded": total - ok, "rate": rate})
        overall_total = len(context_rows)
        overall_rate = len(manifest_rows) / overall_total if overall_total else 0.0
        if overall_rate < args.min_overall_rate:
            violations.append(f"W{window}_OVERALL_RATE_BELOW_TARGET:{overall_rate:.6f}")
        leakage = split_leakage(manifest_rows)
        if leakage:
            violations.append(f"W{window}_SPLIT_LEAKAGE:{leakage}")

    # Deduplicate violations while preserving order.
    seen_v = set()
    violations = [v for v in violations if not (v in seen_v or seen_v.add(v))]
    status = PASS if not violations else HOLD

    schema = {
        "gate": GATE,
        "temporal_feature_names": list(SC5_V2_FEATURES),
        "context_feature_names": context_cols,
        "windows": windows,
        "label": "runtime_objective_label if present, else primary candidate/event role fallback",
        "causal_window_rule": "for endpoint t and window W, use rows [t-W+1, ..., t] only",
        "normalization": "stats are computed from train split only and written separately; NPZ tensors are raw finite features",
        "object_source": "C2e0F v2 repaired 25D CSV when repair manifest maps original temporal_path",
        "nonobject_source": "probe_clean2000_detector_25d_feature_extraction_v3.compute_features on original temporal artifact",
    }

    report = {
        "gate": GATE,
        "status": status,
        "reason": "hard_violation_count=0" if not violations else f"hard_violation_count={len(violations)}",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "git_commit": args.git_commit,
        "inputs": {
            "context_dataset": str(Path(args.context_dataset).expanduser().resolve()),
            "context_dataset_sha256": sha256_file(Path(args.context_dataset).expanduser().resolve()),
            "repair_manifest": str(Path(args.repair_manifest).expanduser().resolve()),
            "repair_manifest_sha256": sha256_file(Path(args.repair_manifest).expanduser().resolve()),
            "c2e0c_endpoint_csv": str(Path(args.c2e0c_endpoint_csv).expanduser().resolve()),
            "c2e0c_endpoint_csv_sha256": sha256_file(Path(args.c2e0c_endpoint_csv).expanduser().resolve()),
            "workers": args.workers,
            "windows": windows,
        },
        "row_count": len(context_rows),
        "unique_artifacts": len(tasks),
        "computed_index_total": sum(int(r["computed_index_count"]) for r in artifact_summaries),
        "context_feature_count": len(context_cols),
        "window_reports": window_reports,
        "suite_summary_rows": suite_summary_rows,
        "split_summary_rows": split_summary_rows,
        "repair_files_used": len(set(r["repaired_path"] for r in repair_usage)),
        "source_method_counts_top20": dict(method_counts.most_common(20)),
        "violations": violations,
        "recommendation": "proceed_to_C2E2_temporal_detector_training_gate_design" if not violations else "fix_C2e1_materialization_violations_before_training",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_reset": "NOT_PERFORMED",
            "env_set_init_state": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
            "detector_training": "NOT_PERFORMED",
            "D5C": "NOT_RUN",
            "D6C_v3": "NOT_RUN",
        },
    }

    write_json(out / "c2e1_temporal_dataset_report.json", report)
    write_json(out / "c2e1_temporal_feature_schema.json", schema)
    write_json(out / "c2e1_context_feature_columns.json", {"context_feature_columns": context_cols})
    write_csv(out / "c2e1_source_usage_by_artifact.csv", artifact_summaries, ["used_path", "used_repair", "n_temporal_rows", "row_count", "computed_index_count", "runtime_seconds"])
    write_csv(out / "c2e1_window_summary_by_suite.csv", suite_summary_rows, ["window", "suite", "total", "materialized", "excluded", "rate"])
    write_csv(out / "c2e1_window_summary_by_split.csv", split_summary_rows, ["window", "split", "total", "materialized", "excluded", "rate"])
    write_csv(out / "c2e1_violations.csv", [{"violation": v} for v in violations], ["violation"])
    write_csv(out / "c2e1_repair_usage_manifest.csv", repair_usage, ["row_index", "suite", "group_key", "original_path", "repaired_path"])

    checks = []
    dynamic_files = []
    for window in windows:
        prefix = f"c2e1_w{window:02d}"
        dynamic_files.extend([
            f"{prefix}_temporal_dataset.npz",
            f"{prefix}_row_manifest.csv",
            f"{prefix}_exclusion_manifest.csv",
            f"{prefix}_normalization_stats_train_only.json",
        ])
    for name in OUT_FILES_BASE + dynamic_files + ["c2e1_repair_usage_manifest.csv", "SHA256SUMS"]:
        p = out / name
        if p.exists() and name != "SHA256SUMS":
            checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    checks = []
    for name in OUT_FILES_BASE + dynamic_files + ["c2e1_repair_usage_manifest.csv"]:
        p = out / name
        if p.exists():
            checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checks:
            f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "output_root": str(out),
        "runtime_seconds": report["runtime_seconds"],
        "row_count": len(context_rows),
        "context_feature_count": len(context_cols),
        "window_reports": window_reports,
        "violations": violations,
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
