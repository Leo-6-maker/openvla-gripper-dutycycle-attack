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
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs
import probe_clean2000_detector_25d_feature_extraction_v3 as v3

GATE = "D2C4_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBE_HISTORY_GAP_IMPUTATION_AUDIT"
PASS = "PASS_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBED_HISTORY_GAP_IMPUTATION_AUDITED"
OUT_FILES = [
    "detector_25d_feature_extraction_probe_v4_report.json",
    "detector_25d_feature_probe_v4_by_row.csv",
    "detector_25d_feature_probe_v4_ready_feature_values.csv",
    "detector_25d_feature_probe_v4_by_feature.csv",
    "detector_25d_feature_probe_v4_failures.csv",
    "detector_25d_feature_probe_v4_exclusions.csv",
    "detector_25d_feature_probe_v4_imputations.csv",
    "checksum_report.json",
]
HISTORY_GAP_FEATURES = {
    "eef_z_delta_since_close",
    "qpos_delta_1",
    "qpos_delta_3",
    "opening_proxy_delta_3",
    "opening_proxy_variance_5",
}


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def get_current_base(values: Dict[str, float], feature: str) -> float:
    val = values.get(feature, math.nan)
    return float(val) if math.isfinite(val) else math.nan


def impute_history_gaps(values: Dict[str, float], methods: Dict[str, str], fields: Dict[str, str], event_status: str, suite: str, label_id: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feat in list(HISTORY_GAP_FEATURES):
        if math.isfinite(values.get(feat, math.nan)):
            continue
        allowed = False
        basis = ""
        val = 0.0
        if feat in {"qpos_delta_1", "qpos_delta_3"}:
            allowed = math.isfinite(get_current_base(values, "gripper_qpos"))
            basis = "current_gripper_qpos_present"
        elif feat in {"opening_proxy_delta_3", "opening_proxy_variance_5"}:
            allowed = math.isfinite(get_current_base(values, "gripper_opening_proxy"))
            basis = "current_gripper_opening_proxy_present"
        elif feat == "eef_z_delta_since_close":
            allowed = math.isfinite(get_current_base(values, "eef_z"))
            basis = "current_eef_z_present_no_close_origin"
        if not allowed:
            continue
        if event_status != "NO_EVENT" and not args.allow_positive_history_gap_imputation:
            continue
        method = "imputed_zero_history_gap_no_raw_history"
        if event_status == "VALID_PRIMARY_CANDIDATE":
            method = "imputed_zero_positive_history_gap_requires_audit"
        values[feat] = val
        methods[feat] = method
        fields[feat] = basis
        rows.append({
            "label_row_id": label_id,
            "suite": suite,
            "event_status": event_status,
            "feature": feat,
            "imputed_value": val,
            "imputation_method": method,
            "imputation_basis": basis,
            "requires_downstream_audit": "1" if event_status == "VALID_PRIMARY_CANDIDATE" else "0",
        })
    return rows


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    bindings = v3.read_csv(Path(args.feature_source_bindings))
    manifest = v3.read_csv(Path(args.complete_manifest))
    by_id, by_key, by_record = v3.build_manifest_indexes(manifest)
    cache: Dict[str, List[Dict[str, Any]]] = {}
    temporal_files = set()
    by_row: List[Dict[str, Any]] = []
    values_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    imputations: List[Dict[str, Any]] = []
    by_feature: Dict[str, Counter] = defaultdict(Counter)
    missing_counts: Counter = Counter()
    method_counts: Counter = Counter()
    ready_checked = 0
    excluded_count = 0
    read_failures = 0
    positive_imputation_row_ids = set()
    positive_imputation_feature_count = 0
    for i, bind in enumerate(bindings):
        label_id = str(bind.get("label_row_id") or f"row_{i:06d}")
        man = v3.match_manifest(bind, by_id, by_key, by_record)
        base = {
            "label_row_id": label_id,
            "record_id": bind.get("record_id", man.get("record_id", "")),
            "suite": bind.get("suite", man.get("suite", "")),
            "event_status": bind.get("event_status", man.get("event_status", "")),
            "event_role": bind.get("event_role", man.get("event_role", "")),
            "feature_source_status": bind.get("feature_source_status", ""),
        }
        if v3.is_excluded(bind):
            excluded_count += 1
            exclusions.append({**base, "exclusion_reason": bind.get("exclusion_reason", "NO_EVENT_WITHOUT_TEMPORAL_ARTIFACT")})
            by_row.append({**base, "probe_status": "EXCLUDED", "missing_features": "", "feature_extraction_policy": "excluded_no_event_without_temporal_artifact", "temporal_path": ""})
            continue
        if not v3.is_ready(bind):
            failures.append({**base, "failure_reason": "NON_READY_FEATURE_SOURCE_STATUS", "detail": bind.get("feature_source_status", "")})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "non_ready_feature_source", "temporal_path": ""})
            continue
        ready_checked += 1
        path = str(bind.get("temporal_path", ""))
        if not path:
            failures.append({**base, "failure_reason": "READY_ROW_MISSING_TEMPORAL_PATH", "detail": ""})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "missing_temporal_path", "temporal_path": ""})
            continue
        try:
            if path not in cache:
                cache[path] = v3.read_temporal(path)
                temporal_files.add(path)
            rows = cache[path]
        except Exception as exc:
            read_failures += 1
            failures.append({**base, "failure_reason": "TEMPORAL_READ_FAILED", "detail": f"{type(exc).__name__}: {exc}", "temporal_path": path})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "temporal_read_failed", "temporal_path": path})
            continue
        if not rows:
            failures.append({**base, "failure_reason": "EMPTY_TEMPORAL_FILE", "detail": path, "temporal_path": path})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "empty_temporal_file", "temporal_path": path})
            continue
        idx, policy, step = v3.choose_index(bind, man, rows, args)
        vals, methods, fields = v3.compute_features(rows, idx, str(base["event_status"]))
        new_imputations = impute_history_gaps(vals, methods, fields, str(base["event_status"]), str(base["suite"]), label_id, args)
        imputations.extend(new_imputations)
        if any(r["event_status"] == "VALID_PRIMARY_CANDIDATE" for r in new_imputations):
            positive_imputation_row_ids.add(label_id)
            positive_imputation_feature_count += sum(1 for r in new_imputations if r["event_status"] == "VALID_PRIMARY_CANDIDATE")
        missing = []
        for feat in SC5_V2_FEATURES:
            val = vals.get(feat, math.nan)
            method = methods.get(feat, "missing") if math.isfinite(val) else "missing"
            by_feature[feat][method] += 1
            method_counts[method] += 1
            if not math.isfinite(val):
                missing.append(feat)
                missing_counts[feat] += 1
        if missing:
            detail = ";".join(missing)
            failures.append({**base, "failure_reason": "MISSING_OR_NONFINITE_FEATURES", "detail": detail, "temporal_path": path, "feature_extraction_policy": policy})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": detail, "feature_extraction_policy": policy, "temporal_path": path, "extraction_index": idx, "extraction_step": step})
        else:
            vrow = {**base, "feature_extraction_policy": policy, "temporal_path": path, "temporal_source_sha256": sha256_file(Path(path)), "extraction_index": idx, "feature_extraction_step": step}
            for feat in SC5_V2_FEATURES:
                vrow[feat] = vals[feat]
                vrow[f"{feat}__method"] = methods.get(feat, "")
                vrow[f"{feat}__field"] = fields.get(feat, "")
            values_rows.append(vrow)
            by_row.append({**base, "probe_status": "PASS", "missing_features": "", "feature_extraction_policy": policy, "temporal_path": path, "extraction_index": idx, "extraction_step": step})
    forbidden_count = 0
    try:
        validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    except Exception:
        forbidden_count = 1
    by_feature_rows = []
    for feat in SC5_V2_FEATURES:
        by_feature_rows.append({
            "feature": feat,
            "pass_count": ready_checked - missing_counts.get(feat, 0),
            "missing_count": missing_counts.get(feat, 0),
            "methods": json.dumps(dict(by_feature[feat]), sort_keys=True),
        })
    status = PASS
    reason = ""
    if len(bindings) != args.expected_label_rows:
        status, reason = "HOLD_FEATURE_BINDING_ROW_COUNT_MISMATCH", f"label_rows={len(bindings)} expected={args.expected_label_rows}"
    elif ready_checked != args.expected_ready_rows:
        status, reason = "HOLD_READY_ROW_COUNT_MISMATCH", f"ready={ready_checked} expected={args.expected_ready_rows}"
    elif excluded_count != args.expected_excluded_rows:
        status, reason = "HOLD_EXCLUDED_ROW_COUNT_MISMATCH", f"excluded={excluded_count} expected={args.expected_excluded_rows}"
    elif forbidden_count:
        status, reason = "HOLD_FORBIDDEN_FEATURE_COLUMNS", "feature_columns contain forbidden hints"
    elif failures:
        status, reason = "HOLD_FEATURE_EXTRACTION_GAPS", f"failure_count={len(failures)}"
    elif positive_imputation_feature_count and args.max_positive_imputed_features >= 0 and positive_imputation_feature_count > args.max_positive_imputed_features:
        status, reason = "HOLD_POSITIVE_HISTORY_GAP_IMPUTATION_EXCEEDS_LIMIT", f"positive_imputed_features={positive_imputation_feature_count} max={args.max_positive_imputed_features}"
    write_csv(out / "detector_25d_feature_probe_v4_by_row.csv", by_row, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "probe_status", "missing_features", "feature_extraction_policy", "temporal_path", "extraction_index", "extraction_step"])
    value_fields = ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_extraction_policy", "temporal_path", "temporal_source_sha256", "extraction_index", "feature_extraction_step"] + list(SC5_V2_FEATURES) + [f"{f}__method" for f in SC5_V2_FEATURES] + [f"{f}__field" for f in SC5_V2_FEATURES]
    write_csv(out / "detector_25d_feature_probe_v4_ready_feature_values.csv", values_rows, value_fields)
    write_csv(out / "detector_25d_feature_probe_v4_by_feature.csv", by_feature_rows, ["feature", "pass_count", "missing_count", "methods"])
    write_csv(out / "detector_25d_feature_probe_v4_failures.csv", failures, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "failure_reason", "detail", "temporal_path", "feature_extraction_policy"])
    write_csv(out / "detector_25d_feature_probe_v4_exclusions.csv", exclusions, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "exclusion_reason"])
    write_csv(out / "detector_25d_feature_probe_v4_imputations.csv", imputations, ["label_row_id", "suite", "event_status", "feature", "imputed_value", "imputation_method", "imputation_basis", "requires_downstream_audit"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "feature_source_bindings": args.feature_source_bindings,
        "feature_source_bindings_sha256": sha256_file(Path(args.feature_source_bindings)),
        "complete_manifest": args.complete_manifest,
        "complete_manifest_sha256": sha256_file(Path(args.complete_manifest)),
        "expected_label_rows": args.expected_label_rows,
        "expected_ready_rows": args.expected_ready_rows,
        "expected_excluded_rows": args.expected_excluded_rows,
        "label_row_count": len(bindings),
        "ready_rows_checked": ready_checked,
        "ready_rows_passed": len(values_rows),
        "excluded_rows": excluded_count,
        "temporal_files_checked": len(temporal_files),
        "file_read_failure_count": read_failures,
        "failure_count": len(failures),
        "failures_by_reason": dict(Counter(f.get("failure_reason", "") for f in failures)),
        "missing_feature_counts": dict(missing_counts),
        "feature_derivation_counts": dict(method_counts),
        "imputation_count": len(imputations),
        "imputation_by_feature": dict(Counter(r["feature"] for r in imputations)),
        "imputation_by_suite": dict(Counter(r["suite"] for r in imputations)),
        "positive_imputation_row_count": len(positive_imputation_row_ids),
        "positive_imputation_feature_count": positive_imputation_feature_count,
        "allow_positive_history_gap_imputation": bool(args.allow_positive_history_gap_imputation),
        "max_positive_imputed_features": args.max_positive_imputed_features,
        "feature_columns": list(SC5_V2_FEATURES),
        "forbidden_feature_count": forbidden_count,
        "history_gap_imputation_policy": "Only the residual history-derived features may be zero-imputed, and only when the current base clean feature is finite. Positive-row imputations are explicit, counted, and require downstream bias audit before dataset freeze.",
        "interpretation": "CPU-only D2C4 probe. This does not build the detector dataset and does not train. It audits whether residual schema/history gaps can be closed with explicit provenance-marked zero history-gap imputations.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_dataset_build": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "detector_25d_feature_extraction_probe_v4_report.json", report)
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
    p.add_argument("--feature-source-bindings", required=True)
    p.add_argument("--complete-manifest", required=True)
    p.add_argument("--expected-label-rows", type=int, default=3806)
    p.add_argument("--expected-ready-rows", type=int, default=3717)
    p.add_argument("--expected-excluded-rows", type=int, default=89)
    p.add_argument("--no-event-step-policy", choices=["first", "middle", "last"], default="middle")
    p.add_argument("--allow-positive-history-gap-imputation", action="store_true")
    p.add_argument("--max-positive-imputed-features", type=int, default=-1, help="-1 disables limit; otherwise HOLD if positive imputed feature count exceeds this value")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
