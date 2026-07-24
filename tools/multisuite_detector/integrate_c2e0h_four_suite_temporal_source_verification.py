#!/usr/bin/env python3
"""C2e0H: four-suite temporal source verification with Object repair and non-Object v3 extraction.

CPU-only readiness audit. No GPU, no OpenVLA, no LIBERO runtime, no env.step,
no rollout, no training.

Inputs:
- D4C1B context dataset.
- C2e0C endpoint audit CSV, aligned by row_index.
- C2e0F v2 repair manifest for LIBERO Object repaired feature caches.

For Object rows, this script reads repaired 25D feature CSVs.
For non-Object rows, this script uses probe_clean2000_detector_25d_feature_extraction_v3.compute_features
on the original temporal CSV/JSON artifacts for every row in the causal window.

It validates full causal windows [endpoint-W+1, ..., endpoint] for W=8,16,32.
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
    "c2e0h_four_suite_temporal_source_report.json",
    "c2e0h_window_completeness_by_row.csv",
    "c2e0h_window_completeness_by_suite.csv",
    "c2e0h_window_completeness_by_split.csv",
    "c2e0h_exclusion_manifest.csv",
    "c2e0h_repair_usage_manifest.csv",
    "c2e0h_nonobject_v3_method_summary.csv",
    "c2e0h_violations.csv",
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
    if not path.exists():
        return out
    for row in read_csv_dict(path):
        status = str(row.get("status", ""))
        if status == "REPAIRED":
            src = row.get("original_temporal_path") or row.get("temporal_path") or ""
            dst = row.get("repaired_feature_path") or row.get("output_path") or ""
            if src and dst:
                out[src] = dst
    return out


def check_repaired_window(rows: List[Dict[str, str]], start: int, end: int) -> Tuple[bool, str, Counter[str]]:
    methods: Counter[str] = Counter()
    if start < 0 or end >= len(rows):
        return False, f"window_out_of_bounds:[{start},{end}]_n={len(rows)}", methods
    for i in range(start, end + 1):
        row = rows[i]
        for feat in SC5_V2_FEATURES:
            if not finite(row.get(feat)):
                return False, f"row{i}_missing_{feat}", methods
        methods["object_repaired_25d_csv"] += 1
    return True, "", methods


def check_v3_window(rows: List[Dict[str, Any]], start: int, end: int, event_status: str) -> Tuple[bool, str, Counter[str]]:
    methods: Counter[str] = Counter()
    if start < 0 or end >= len(rows):
        return False, f"window_out_of_bounds:[{start},{end}]_n={len(rows)}", methods
    for i in range(start, end + 1):
        try:
            values, feat_methods, _fields = v3.compute_features(rows, i, event_status)
        except Exception as exc:
            return False, f"v3_exception_row{i}:{str(exc)[:160]}", methods
        for feat in SC5_V2_FEATURES:
            if not finite(values.get(feat)):
                return False, f"row{i}_v3_missing_{feat}", methods
            methods[str(feat_methods.get(feat, "unknown"))] += 1
    return True, "", methods


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

    temporal_cache: Dict[str, List[Dict[str, Any]]] = {}
    result_rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    repair_usage: List[Dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    suite_stats: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "complete": 0}))
    split_stats: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "complete": 0}))

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

        try:
            if used_path not in temporal_cache:
                if used_repair:
                    temporal_cache[used_path] = read_csv_dict(used_path)
                else:
                    temporal_cache[used_path] = v3.read_temporal(used_path)
            trows = temporal_cache[used_path]
        except Exception as exc:
            exclusions.append({"row_index": idx, "suite": suite, "split": split, "group_key": gkey, "reason": f"read_error:{str(exc)[:160]}", "temporal_path": tpath, "used_path": used_path})
            continue

        row_out: Dict[str, Any] = {
            "row_index": idx,
            "suite": suite,
            "split": split,
            "group_key": gkey,
            "label_status": label_status(row),
            "group_class": group_class(row),
            "temporal_path": tpath,
            "used_path": used_path,
            "used_repair": used_repair,
            "endpoint": endpoint,
            "n_temporal_rows": len(trows),
        }
        all_ok = True
        for W in WINDOW_SIZES:
            start = endpoint - W + 1
            if used_repair:
                ok, detail, methods = check_repaired_window(trows, start, endpoint)
            else:
                ok, detail, methods = check_v3_window(trows, start, endpoint, label_status(row) or "NO_EVENT")
            method_counts.update(methods)
            row_out[f"W{W}"] = "PASS" if ok else detail
            suite_stats[suite][W]["total"] += 1
            split_stats[split][W]["total"] += 1
            if ok:
                suite_stats[suite][W]["complete"] += 1
                split_stats[split][W]["complete"] += 1
            else:
                all_ok = False
        row_out["all_windows_ok"] = all_ok
        result_rows.append(row_out)

    suite_summary: List[Dict[str, Any]] = []
    for suite in sorted(suite_stats):
        item: Dict[str, Any] = {"suite": suite}
        for W in WINDOW_SIZES:
            total = suite_stats[suite][W]["total"]
            complete = suite_stats[suite][W]["complete"]
            item[f"W{W}_total"] = total
            item[f"W{W}_complete"] = complete
            item[f"W{W}_rate"] = complete / total if total else 0.0
        suite_summary.append(item)

    split_summary: List[Dict[str, Any]] = []
    for split in sorted(split_stats):
        item = {"split": split}
        for W in WINDOW_SIZES:
            total = split_stats[split][W]["total"]
            complete = split_stats[split][W]["complete"]
            item[f"W{W}_total"] = total
            item[f"W{W}_complete"] = complete
            item[f"W{W}_rate"] = complete / total if total else 0.0
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

    status = "PASS_C2E0H_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION" if not violations else "HOLD_C2E0H_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION"
    report = {
        "gate": "C2E0H_FOUR_SUITE_TEMPORAL_SOURCE_VERIFICATION",
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
        },
        "row_count": len(context_rows),
        "validated_rows": len(result_rows),
        "excluded_rows": len(exclusions),
        "overall_w32_rate": overall_w32,
        "suite_summary": suite_summary,
        "split_leakage_count": leakage,
        "repair_files_used": len(set(r["repaired_path"] for r in repair_usage)),
        "nonobject_v3_method_counts_top20": dict(method_counts.most_common(20)),
        "violations": violations,
        "recommendation": "proceed_to_C2E1_temporal_dataset_materialization" if not violations else "fix_C2e0H_violations_before_C2E1",
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

    write_json(out / "c2e0h_four_suite_temporal_source_report.json", report)
    write_csv(out / "c2e0h_window_completeness_by_row.csv", result_rows, [
        "row_index", "suite", "split", "group_key", "label_status", "group_class", "temporal_path", "used_path", "used_repair", "endpoint", "n_temporal_rows", "W8", "W16", "W32", "all_windows_ok",
    ])
    write_csv(out / "c2e0h_window_completeness_by_suite.csv", suite_summary, ["suite", "W8_total", "W8_complete", "W8_rate", "W16_total", "W16_complete", "W16_rate", "W32_total", "W32_complete", "W32_rate"])
    write_csv(out / "c2e0h_window_completeness_by_split.csv", split_summary, ["split", "W8_total", "W8_complete", "W8_rate", "W16_total", "W16_complete", "W16_rate", "W32_total", "W32_complete", "W32_rate"])
    write_csv(out / "c2e0h_exclusion_manifest.csv", exclusions, ["row_index", "suite", "split", "group_key", "reason", "temporal_path", "used_path"])
    write_csv(out / "c2e0h_repair_usage_manifest.csv", repair_usage, ["row_index", "suite", "group_key", "original_path", "repaired_path"])
    write_csv(out / "c2e0h_nonobject_v3_method_summary.csv", [{"method": k, "count": v} for k, v in method_counts.most_common()], ["method", "count"])
    write_csv(out / "c2e0h_violations.csv", [{"violation": v} for v in violations], ["violation"])

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
        "overall_w32_rate": overall_w32,
        "validated_rows": len(result_rows),
        "excluded_rows": len(exclusions),
        "violations": violations,
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
