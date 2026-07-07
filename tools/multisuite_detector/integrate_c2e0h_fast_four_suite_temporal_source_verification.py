#!/usr/bin/env python3
"""C2e0H-fast: parallel four-suite temporal source verification.

This is a drop-in acceleration path for C2e0H. It is CPU-only and does not use
GPU, OpenVLA, LIBERO runtime, env.step, rollout, attack, or detector training.

Main speedups over C2e0H:
- groups rows by temporal artifact and reads each artifact once;
- computes v3 features only for the union of row indices needed by W=8/16/32
  causal windows, not every row in every temporal file;
- processes temporal files in a ProcessPool;
- for repaired Object feature CSVs, checks 25D columns directly without v3.

The gate remains conservative: every requested causal window row must have all
25 SC5 features finite, with no endpoint clamping and no future frames.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))

from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs  # type: ignore
import probe_clean2000_detector_25d_feature_extraction_v3 as v3  # type: ignore

WINDOW_SIZES = [8, 16, 32]
PRIMARY_LABELS = {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE", "primary_attackable"}
OUT_FILES = [
    "c2e0h_fast_four_suite_temporal_source_report.json",
    "c2e0h_fast_window_completeness_by_row.csv",
    "c2e0h_fast_window_completeness_by_suite.csv",
    "c2e0h_fast_window_completeness_by_split.csv",
    "c2e0h_fast_exclusion_manifest.csv",
    "c2e0h_fast_repair_usage_manifest.csv",
    "c2e0h_fast_nonobject_v3_method_summary.csv",
    "c2e0h_fast_artifact_runtime_summary.csv",
    "c2e0h_fast_violations.csv",
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


def finite(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except Exception:
        return None


def label_status(row: Dict[str, str]) -> str:
    return str(row.get("teacher_label_status") or row.get("event_status") or row.get("event_role") or "")


def group_class(row: Dict[str, str]) -> str:
    status = label_status(row)
    role = str(row.get("event_role") or row.get("event_role_true") or "")
    return "has_primary" if status in PRIMARY_LABELS or role in PRIMARY_LABELS else "no_primary"


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


def required_indices_for_endpoint(endpoint: int, n_rows_hint: Optional[int] = None) -> List[int]:
    idxs = set()
    for w in WINDOW_SIZES:
        start = endpoint - w + 1
        if start < 0:
            continue
        for i in range(start, endpoint + 1):
            if n_rows_hint is None or 0 <= i < n_rows_hint:
                idxs.add(i)
    return sorted(idxs)


def check_repaired_indices(rows: List[Dict[str, str]], needed: Iterable[int]) -> Tuple[Dict[int, bool], Dict[int, str], Counter[str]]:
    ok: Dict[int, bool] = {}
    detail: Dict[int, str] = {}
    methods: Counter[str] = Counter()
    n = len(rows)
    for idx in needed:
        if idx < 0 or idx >= n:
            ok[idx] = False
            detail[idx] = f"index_oob:{idx}_n={n}"
            continue
        row = rows[idx]
        missing = ""
        for feat in SC5_V2_FEATURES:
            if not finite(row.get(feat)):
                missing = feat
                break
        ok[idx] = not missing
        detail[idx] = "" if not missing else f"missing_{missing}"
        if ok[idx]:
            methods["object_repaired_25d_csv"] += 1
    return ok, detail, methods


def check_v3_indices(rows: List[Dict[str, Any]], needed: Iterable[int], event_status_by_index: Dict[int, str]) -> Tuple[Dict[int, bool], Dict[int, str], Counter[str]]:
    ok: Dict[int, bool] = {}
    detail: Dict[int, str] = {}
    methods: Counter[str] = Counter()
    n = len(rows)
    for idx in needed:
        if idx < 0 or idx >= n:
            ok[idx] = False
            detail[idx] = f"index_oob:{idx}_n={n}"
            continue
        event_status = event_status_by_index.get(idx, "NO_EVENT")
        try:
            values, feat_methods, _fields = v3.compute_features(rows, idx, event_status)
        except Exception as exc:
            ok[idx] = False
            detail[idx] = f"v3_exception:{str(exc)[:160]}"
            continue
        missing = ""
        for feat in SC5_V2_FEATURES:
            if not finite(values.get(feat)):
                missing = feat
                break
        ok[idx] = not missing
        detail[idx] = "" if not missing else f"v3_missing_{missing}"
        if ok[idx]:
            for feat in SC5_V2_FEATURES:
                methods[str(feat_methods.get(feat, "unknown"))] += 1
    return ok, detail, methods


def process_artifact_task(task: Dict[str, Any]) -> Dict[str, Any]:
    used_path = task["used_path"]
    used_repair = bool(task["used_repair"])
    row_items = task["rows"]
    started = time.time()
    out_rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    n_temporal_rows = 0
    computed_index_count = 0

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
                "group_key": item["group_key"], "reason": f"read_error:{str(exc)[:160]}",
                "temporal_path": item["temporal_path"], "used_path": used_path,
            })
        return {
            "used_path": used_path, "used_repair": used_repair, "rows": out_rows, "exclusions": exclusions,
            "method_counts": {}, "n_temporal_rows": n_temporal_rows, "computed_index_count": 0,
            "runtime_seconds": time.time() - started,
        }

    needed = set()
    event_status_by_index: Dict[int, str] = {}
    for item in row_items:
        endpoint = int(item["endpoint"])
        for idx in required_indices_for_endpoint(endpoint, n_temporal_rows):
            needed.add(idx)
            event_status_by_index.setdefault(idx, item.get("label_status") or "NO_EVENT")
    computed_index_count = len(needed)

    if used_repair:
        idx_ok, idx_detail, methods = check_repaired_indices(trows, sorted(needed))
    else:
        idx_ok, idx_detail, methods = check_v3_indices(trows, sorted(needed), event_status_by_index)
    method_counts.update(methods)

    for item in row_items:
        endpoint = int(item["endpoint"])
        row_out: Dict[str, Any] = {
            "row_index": item["row_index"],
            "suite": item["suite"],
            "split": item["split"],
            "group_key": item["group_key"],
            "label_status": item["label_status"],
            "group_class": item["group_class"],
            "temporal_path": item["temporal_path"],
            "used_path": used_path,
            "used_repair": used_repair,
            "endpoint": endpoint,
            "n_temporal_rows": n_temporal_rows,
        }
        all_ok = True
        for w in WINDOW_SIZES:
            start = endpoint - w + 1
            if start < 0:
                row_out[f"W{w}"] = "endpoint_too_early"
                all_ok = False
                continue
            if endpoint >= n_temporal_rows:
                row_out[f"W{w}"] = f"endpoint_oob:{endpoint}_n={n_temporal_rows}"
                all_ok = False
                continue
            bad = ""
            for idx in range(start, endpoint + 1):
                if not idx_ok.get(idx, False):
                    bad = f"idx{idx}:{idx_detail.get(idx, 'not_computed')}"
                    break
            if bad:
                row_out[f"W{w}"] = bad
                all_ok = False
            else:
                row_out[f"W{w}"] = "PASS"
        row_out["all_windows_ok"] = all_ok
        out_rows.append(row_out)

    return {
        "used_path": used_path,
        "used_repair": used_repair,
        "rows": out_rows,
        "exclusions": exclusions,
        "method_counts": dict(method_counts),
        "n_temporal_rows": n_temporal_rows,
        "computed_index_count": computed_index_count,
        "runtime_seconds": time.time() - started,
    }


def build_split_leakage(rows: Iterable[Dict[str, Any]]) -> int:
    group_to_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        g = str(row.get("group_key", ""))
        s = str(row.get("split", ""))
        if g and s:
            group_to_splits[g].add(s)
    return sum(1 for vals in group_to_splits.values() if len(vals) > 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True)
    ap.add_argument("--repair-manifest", required=True)
    ap.add_argument("--c2e0c-endpoint-csv", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--min-suite-w32", type=float, default=0.95)
    ap.add_argument("--min-overall-w32", type=float, default=0.95)
    args = ap.parse_args()

    started = time.time()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs([f"lag00_{f}" for f in SC5_V2_FEATURES])

    context_rows = read_csv_dict(args.context_dataset)
    endpoint_map = load_endpoint_map(Path(args.c2e0c_endpoint_csv).expanduser())
    repair_map = load_repair_map(Path(args.repair_manifest).expanduser())

    tasks_by_path: Dict[str, Dict[str, Any]] = {}
    exclusions: List[Dict[str, Any]] = []
    repair_usage: List[Dict[str, Any]] = []

    for idx, row in enumerate(context_rows):
        suite = str(row.get("suite", "unknown"))
        split = str(row.get("split", "unknown"))
        gkey = str(row.get("group_key", ""))
        tpath = str(row.get("temporal_path", ""))
        endpoint = parse_int(endpoint_map.get(idx, {}).get("endpoint_index"))
        if endpoint is None:
            exclusions.append({"row_index": idx, "suite": suite, "split": split, "group_key": gkey, "reason": "missing_endpoint", "temporal_path": tpath})
            continue
        if not tpath:
            exclusions.append({"row_index": idx, "suite": suite, "split": split, "group_key": gkey, "reason": "missing_temporal_path", "temporal_path": tpath})
            continue
        used_path = repair_map.get(tpath, tpath)
        used_repair = used_path != tpath
        if used_repair:
            repair_usage.append({"row_index": idx, "suite": suite, "group_key": gkey, "original_path": tpath, "repaired_path": used_path})
        task = tasks_by_path.setdefault(used_path, {"used_path": used_path, "used_repair": used_repair, "rows": []})
        task["rows"].append({
            "row_index": idx,
            "suite": suite,
            "split": split,
            "group_key": gkey,
            "label_status": label_status(row),
            "group_class": group_class(row),
            "temporal_path": tpath,
            "endpoint": endpoint,
        })

    result_rows: List[Dict[str, Any]] = []
    artifact_summaries: List[Dict[str, Any]] = []
    method_counts: Counter[str] = Counter()

    tasks = list(tasks_by_path.values())
    print(f"C2e0H-fast: context_rows={len(context_rows)} unique_artifacts={len(tasks)} workers={args.workers}")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(process_artifact_task, task) for task in tasks]
        for n_done, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            result_rows.extend(res["rows"])
            exclusions.extend(res["exclusions"])
            method_counts.update(res.get("method_counts", {}))
            artifact_summaries.append({
                "used_path": res["used_path"],
                "used_repair": res["used_repair"],
                "n_temporal_rows": res["n_temporal_rows"],
                "row_count": len(res["rows"]),
                "computed_index_count": res["computed_index_count"],
                "runtime_seconds": res["runtime_seconds"],
            })
            if n_done % 100 == 0 or n_done == len(futs):
                print(f"  completed_artifacts={n_done}/{len(futs)} rows={len(result_rows)} exclusions={len(exclusions)}")

    result_rows.sort(key=lambda r: int(r["row_index"]))
    suite_stats: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "complete": 0}))
    split_stats: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "complete": 0}))
    for row in result_rows:
        suite = str(row.get("suite"))
        split = str(row.get("split"))
        for w in WINDOW_SIZES:
            suite_stats[suite][w]["total"] += 1
            split_stats[split][w]["total"] += 1
            if row.get(f"W{w}") == "PASS":
                suite_stats[suite][w]["complete"] += 1
                split_stats[split][w]["complete"] += 1

    suite_summary: List[Dict[str, Any]] = []
    for suite in sorted(suite_stats):
        item: Dict[str, Any] = {"suite": suite}
        for w in WINDOW_SIZES:
            total = suite_stats[suite][w]["total"]
            complete = suite_stats[suite][w]["complete"]
            item[f"W{w}_total"] = total
            item[f"W{w}_complete"] = complete
            item[f"W{w}_rate"] = complete / total if total else 0.0
        suite_summary.append(item)

    split_summary: List[Dict[str, Any]] = []
    for split in sorted(split_stats):
        item: Dict[str, Any] = {"split": split}
        for w in WINDOW_SIZES:
            total = split_stats[split][w]["total"]
            complete = split_stats[split][w]["complete"]
            item[f"W{w}_total"] = total
            item[f"W{w}_complete"] = complete
            item[f"W{w}_rate"] = complete / total if total else 0.0
        split_summary.append(item)

    violations: List[str] = []
    for item in suite_summary:
        rate = float(item.get("W32_rate", 0.0))
        if rate < args.min_suite_w32:
            violations.append(f"SUITE_W32_BELOW_TARGET:{item['suite']}:{rate:.6f}")
    total_w32 = sum(suite_stats[s][32]["total"] for s in suite_stats)
    complete_w32 = sum(suite_stats[s][32]["complete"] for s in suite_stats)
    overall_w32 = complete_w32 / total_w32 if total_w32 else 0.0
    if overall_w32 < args.min_overall_w32:
        violations.append(f"OVERALL_W32_BELOW_TARGET:{overall_w32:.6f}")
    leakage = build_split_leakage(result_rows)
    if leakage:
        violations.append(f"SPLIT_LEAKAGE:{leakage}")
    if exclusions:
        violations.append(f"EXCLUDED_ROWS:{len(exclusions)}")

    status = "PASS_C2E0H_FAST_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION" if not violations else "HOLD_C2E0H_FAST_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION"
    report = {
        "gate": "C2E0H_FAST_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION",
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
        },
        "row_count": len(context_rows),
        "validated_rows": len(result_rows),
        "excluded_rows": len(exclusions),
        "unique_artifacts": len(tasks),
        "computed_index_total": sum(int(r["computed_index_count"]) for r in artifact_summaries),
        "overall_w32_rate": overall_w32,
        "suite_summary": suite_summary,
        "split_leakage_count": leakage,
        "repair_files_used": len(set(r["repaired_path"] for r in repair_usage)),
        "nonobject_v3_method_counts_top20": dict(method_counts.most_common(20)),
        "violations": violations,
        "recommendation": "proceed_to_C2E1_temporal_dataset_materialization" if not violations else "fix_C2e0H_fast_violations_before_C2E1",
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

    write_json(out / "c2e0h_fast_four_suite_temporal_source_report.json", report)
    write_csv(out / "c2e0h_fast_window_completeness_by_row.csv", result_rows, [
        "row_index", "suite", "split", "group_key", "label_status", "group_class", "temporal_path", "used_path", "used_repair", "endpoint", "n_temporal_rows", "W8", "W16", "W32", "all_windows_ok",
    ])
    write_csv(out / "c2e0h_fast_window_completeness_by_suite.csv", suite_summary, ["suite", "W8_total", "W8_complete", "W8_rate", "W16_total", "W16_complete", "W16_rate", "W32_total", "W32_complete", "W32_rate"])
    write_csv(out / "c2e0h_fast_window_completeness_by_split.csv", split_summary, ["split", "W8_total", "W8_complete", "W8_rate", "W16_total", "W16_complete", "W16_rate", "W32_total", "W32_complete", "W32_rate"])
    write_csv(out / "c2e0h_fast_exclusion_manifest.csv", exclusions, ["row_index", "suite", "split", "group_key", "reason", "temporal_path", "used_path"])
    write_csv(out / "c2e0h_fast_repair_usage_manifest.csv", repair_usage, ["row_index", "suite", "group_key", "original_path", "repaired_path"])
    write_csv(out / "c2e0h_fast_nonobject_v3_method_summary.csv", [{"method": k, "count": v} for k, v in method_counts.most_common()], ["method", "count"])
    write_csv(out / "c2e0h_fast_artifact_runtime_summary.csv", artifact_summaries, ["used_path", "used_repair", "n_temporal_rows", "row_count", "computed_index_count", "runtime_seconds"])
    write_csv(out / "c2e0h_fast_violations.csv", [{"violation": v} for v in violations], ["violation"])

    checks = []
    for name in OUT_FILES:
        p = out / name
        if p.exists():
            checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checks:
            f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "output_root": str(out),
        "runtime_seconds": report["runtime_seconds"],
        "unique_artifacts": len(tasks),
        "computed_index_total": report["computed_index_total"],
        "overall_w32_rate": overall_w32,
        "validated_rows": len(result_rows),
        "excluded_rows": len(exclusions),
        "violations": violations,
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
