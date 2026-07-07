#!/usr/bin/env python3
"""C2e0 design audit for a clean causal temporal cross-suite detector.

This script is intentionally an audit/readiness gate, not training and not replay.
It inspects a frozen context detector dataset and the referenced clean temporal
artifacts, then proposes a causal temporal detector schema for C2e without
loading OpenVLA, LIBERO, a simulator, or any attack/intervention code.

Boundary:
- CPU-only file inspection.
- Existing clean temporal artifacts only.
- Causal windows ending at the label row endpoint; no future frames proposed.
- No privileged simulator state, object/target pose, attack outcome, rollout
  outcome, or OpenVLA hidden state as model input.
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

try:
    from gripper_attack.sc5_multisuite_detector_runtime import (  # type: ignore
        SC5_V2_FEATURES,
        validate_no_forbidden_inputs,
    )
except Exception:  # pragma: no cover - keep audit runnable during isolated debugging.
    SC5_V2_FEATURES = [
        "gripper_command", "gripper_qpos", "gripper_opening_proxy",
        "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
        "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
        "close_onset", "time_since_close", "eef_speed",
        "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
        "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
    ]

    def validate_no_forbidden_inputs(feature_names: List[str]) -> None:
        bad: List[str] = []
        forbidden = [
            "normalized_step", "timestep", "task_id", "state_id", "episode_key",
            "run_id", "parent_id", "object_pose", "target_pose", "object_to_target",
            "teacher_window", "teacher_anchor", "attack_outcome", "rand_outcome",
            "manual_anchor", "oracle_window",
        ]
        for name in feature_names:
            low = str(name).lower()
            if any(hint in low for hint in forbidden):
                bad.append(name)
        if bad:
            raise ValueError(f"forbidden model-input feature names present: {bad}")


GATE = "D4C2E0_CLEAN2000_CONTEXT_TEMPORAL_DETECTOR_DESIGN_AUDIT"
PASS = "PASS_D4C2E0_TEMPORAL_DETECTOR_DESIGN_READY_FOR_C2E1"
HOLD = "HOLD_D4C2E0_TEMPORAL_DETECTOR_DESIGN_AUDIT"

OUT_FILES = [
    "d4c2e0_temporal_detector_design_report.json",
    "temporal_window_coverage_by_suite.csv",
    "temporal_label_alignment_audit.csv",
    "temporal_forbidden_leakage_audit.csv",
    "temporal_feature_schema_proposal.json",
    "temporal_artifact_debug_sample.csv",
    "temporal_violations.csv",
    "checksum_report.json",
]

REQUIRED_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
GROUPING_ONLY_FIELDS = {
    "record_id",
    "group_key",
    "episode_key",
    "label_row_id",
    "event_id",
    "segment_id",
    "split",
    "temporal_path",
    "task_id",
    "task_index_from_group_key",
    "suite_task_template_id",
    "teacher_label_status",
    "event_role",
}
FORBIDDEN_SUBSTRINGS = [
    "object_pose",
    "target_pose",
    "object_target_distance",
    "object_to_target",
    "privileged",
    "sim_state",
    "oracle",
    "attack_outcome",
    "rollout_outcome",
    "success",
    "failure",
    "teacher_anchor",
    "teacher_window",
    "future_",
    "post_intervention",
]
STEP_COLUMN_CANDIDATES = [
    "step",
    "frame_idx",
    "frame_index",
    "row_step",
    "local_step",
    "artifact_step",
    "t",
]
TEMPORAL_PATH_COLUMNS = [
    "temporal_path",
    "source_temporal_path",
    "stream_path",
    "artifact_path",
    "trajectory_path",
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


def read_csv(path: Path) -> List[Dict[str, str]]:
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


def stable_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except Exception:
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def get_first(row: Dict[str, str], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if row.get(key) not in (None, ""):
            return str(row.get(key, ""))
    return default


def row_group_key(row: Dict[str, str]) -> str:
    return get_first(row, ["group_key", "episode_key", "record_id", "parent_id", "run_id", "episode_id"])


def row_episode_key(row: Dict[str, str]) -> str:
    return get_first(row, ["episode_key", "group_key", "record_id", "parent_id", "run_id", "episode_id"])


def row_split(row: Dict[str, str]) -> str:
    return str(row.get("split", row.get("dataset_split", "unknown")) or "unknown")


def row_suite(row: Dict[str, str]) -> str:
    return str(row.get("suite", "unknown") or "unknown")


def row_label(row: Dict[str, str]) -> int:
    for key in ["runtime_objective_label", "label", "y", "target"]:
        if row.get(key) not in (None, ""):
            value = stable_int(row.get(key), None)
            if value in (0, 1):
                return int(value)
    status = str(row.get("teacher_label_status", row.get("label_status", "")))
    role = str(row.get("event_role", row.get("event_role_true", "")))
    if status in {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"} or role == "primary_attackable":
        return 1
    if status in {"VALID_AUXILIARY", "VALID_AUXILIARY_CANDIDATE", "NO_EVENT"} or role in {"auxiliary_manipulation", "unsupported_or_abstain", "distractor_or_setup"}:
        return 0
    return -1


def find_step(row: Dict[str, str]) -> Tuple[Optional[int], str]:
    for key in STEP_COLUMN_CANDIDATES:
        value = stable_int(row.get(key), None)
        if value is not None:
            return value, key
    return None, ""


def parse_window_lengths(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError("window lengths must be positive")
        vals.append(v)
    if not vals:
        raise ValueError("at least one window length is required")
    return sorted(set(vals))


def resolve_path(raw: str, dataset_path: Path, extra_roots: Sequence[Path]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((dataset_path.parent / p).resolve())
        candidates.append((Path.cwd() / p).resolve())
        for root in extra_roots:
            candidates.append((root / p).resolve())
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return candidates[0] if candidates else p


def header_and_count_csv(path: Path, max_rows: Optional[int] = None) -> Tuple[List[str], int]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        n = 0
        for _ in reader:
            n += 1
            if max_rows is not None and n >= max_rows:
                # Count is at least max_rows. For readiness, that is often enough.
                return header, n
        return header, n


def percentile(values: List[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return float(vals[0])
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - idx) + vals[hi] * (idx - lo))


def detect_context_model_inputs(header: Sequence[str]) -> List[str]:
    names: List[str] = []
    allowed_prefixes = (
        "suite_onehot_",
        "ctx_suite_",
        "ctx_task_index_hash_",
        "ctx_suite_task_template_hash_",
        "ctx_task_index_onehot_",
        "suite_task_index_onehot_",
    )
    for name in header:
        if name in SC5_V2_FEATURES:
            names.append(name)
            continue
        if str(name).startswith(allowed_prefixes):
            names.append(str(name))
    return names


def proposed_temporal_features(context_names: Sequence[str], window_lengths: Sequence[int]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "feature_family": "C2E_CAUSAL_TEMPORAL_CONTEXT_DETECTOR",
        "base_features": list(SC5_V2_FEATURES),
        "context_features": [n for n in context_names if n not in SC5_V2_FEATURES],
        "window_lengths": list(window_lengths),
        "causal_window_definition": "For endpoint step t, use rows [t-W+1, ..., t]. Never use t+1 or later.",
        "label_alignment": "The endpoint row keeps the runtime_objective_label; previous rows are clean history only.",
        "forbidden_inputs": sorted(FORBIDDEN_SUBSTRINGS),
        "model_variants": [
            {
                "name": "C2e_temporal_pooling_mlp",
                "description": "mean/std/last/delta pooling over causal 25D windows plus clean context",
            },
            {
                "name": "C2e_small_causal_tcn",
                "description": "small causal 1D temporal convolution over 25D windows plus clean context",
            },
        ],
    }
    # Flattened names are only for forbidden-input validation and debugging; C2e1 may
    # store windows as tensors instead of columns.
    flat_names: List[str] = []
    for w in window_lengths:
        for lag in range(w):
            for feature in SC5_V2_FEATURES:
                flat_names.append(f"lag{lag:02d}_w{w}_{feature}")
    flat_names.extend(schema["context_features"])
    schema["debug_flattened_feature_count_max_window"] = len(flat_names)
    schema["debug_flattened_feature_names_sample"] = flat_names[:80]
    return schema


def audit_split_leakage(rows: List[Dict[str, str]]) -> Tuple[int, List[Dict[str, Any]]]:
    key_to_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = row_episode_key(row)
        if key:
            key_to_splits[key].add(row_split(row))
    leak_rows = []
    for key, splits in key_to_splits.items():
        real = sorted(s for s in splits if s and s != "unknown")
        if len(set(real)) > 1:
            leak_rows.append({"episode_or_group_key": key, "splits": ";".join(real), "violation_code": "SPLIT_LEAKAGE"})
    return len(leak_rows), leak_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True, help="D4C1/D4C1B context runtime-objective dataset CSV")
    ap.add_argument("--context-schema", default="", help="Optional context feature schema JSON")
    ap.add_argument("--expected-rows", type=int, default=3717)
    ap.add_argument("--window-lengths", default="8,16,32")
    ap.add_argument("--temporal-root", action="append", default=[], help="Optional root for relative temporal_path values")
    ap.add_argument("--max-artifact-debug", type=int, default=200, help="Max unique temporal artifacts to inspect")
    ap.add_argument("--min-suite-window-coverage", type=float, default=0.95)
    ap.add_argument("--allow-endpoint-unmapped", action="store_true", help="Allow PASS when exact endpoint step is missing but temporal artifacts exist")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", default="")
    args = ap.parse_args()

    started = time.time()
    dataset_path = Path(args.context_dataset).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    window_lengths = parse_window_lengths(args.window_lengths)
    max_window = max(window_lengths)
    extra_roots = [Path(p).expanduser().resolve() for p in args.temporal_root]

    rows = read_csv(dataset_path)
    header = list(rows[0].keys()) if rows else []
    context_input_names = detect_context_model_inputs(header)
    proposed_schema = proposed_temporal_features(context_input_names, window_lengths)

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if args.expected_rows and len(rows) != args.expected_rows:
        violations.append({"code": "UNEXPECTED_ROW_COUNT", "severity": "hard", "expected": args.expected_rows, "actual": len(rows)})

    suite_counts = Counter(row_suite(r) for r in rows)
    missing_suites = [s for s in REQUIRED_SUITES if suite_counts.get(s, 0) <= 0]
    if missing_suites:
        violations.append({"code": "MISSING_SUITE_COVERAGE", "severity": "hard", "missing_suites": missing_suites})

    label_counts = Counter(str(row_label(r)) for r in rows)
    if label_counts.get("1", 0) <= 0 or label_counts.get("0", 0) <= 0:
        violations.append({"code": "MISSING_RUNTIME_OBJECTIVE_CLASS", "severity": "hard", "label_counts": dict(label_counts)})

    split_leakage_count, split_leak_rows = audit_split_leakage(rows)
    if split_leakage_count:
        violations.append({"code": "SPLIT_LEAKAGE", "severity": "hard", "count": split_leakage_count})

    forbidden_input_rows: List[Dict[str, Any]] = []
    try:
        validate_no_forbidden_inputs(list(proposed_schema["debug_flattened_feature_names_sample"]) + proposed_schema["context_features"])
    except Exception as exc:
        violations.append({"code": "FORBIDDEN_MODEL_INPUT_FEATURE", "severity": "hard", "message": str(exc)})
        forbidden_input_rows.append({"feature": "schema_validation", "reason": str(exc)})

    for name in context_input_names:
        low = str(name).lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in low:
                forbidden_input_rows.append({"feature": name, "reason": f"contains forbidden substring {bad}"})
                break
    if forbidden_input_rows:
        violations.append({"code": "FORBIDDEN_CONTEXT_INPUT", "severity": "hard", "count": len(forbidden_input_rows)})

    temporal_path_col = ""
    for col in TEMPORAL_PATH_COLUMNS:
        if col in header:
            temporal_path_col = col
            break
    if not temporal_path_col:
        violations.append({"code": "TEMPORAL_PATH_COLUMN_MISSING", "severity": "hard", "searched_columns": TEMPORAL_PATH_COLUMNS})

    unique_paths: Dict[str, Optional[Path]] = {}
    for row in rows:
        raw = str(row.get(temporal_path_col, "")) if temporal_path_col else ""
        if raw and raw not in unique_paths:
            unique_paths[raw] = resolve_path(raw, dataset_path, extra_roots)

    artifact_info: Dict[str, Dict[str, Any]] = {}
    debug_rows: List[Dict[str, Any]] = []
    inspected = 0
    for raw, path in unique_paths.items():
        info: Dict[str, Any] = {"raw_temporal_path": raw, "resolved_path": str(path) if path else "", "exists": bool(path and path.exists())}
        if path and path.exists() and inspected < args.max_artifact_debug:
            inspected += 1
            try:
                art_header, n_rows = header_and_count_csv(path, max_rows=max_window)
                info.update({
                    "readable": True,
                    "row_count_at_least": n_rows,
                    "has_all_25d_features": all(f in art_header for f in SC5_V2_FEATURES),
                    "missing_25d_features": ";".join([f for f in SC5_V2_FEATURES if f not in art_header]),
                    "step_columns_present": ";".join([c for c in STEP_COLUMN_CANDIDATES if c in art_header]),
                    "header_sample": ";".join(art_header[:40]),
                })
            except Exception as exc:
                info.update({"readable": False, "read_error": str(exc)})
        elif path and path.exists():
            info.update({"readable": "not_inspected_max_debug", "row_count_at_least": "", "has_all_25d_features": "unknown"})
        artifact_info[raw] = info
        if len(debug_rows) < args.max_artifact_debug:
            debug_rows.append(info)

    # Per-row temporal readiness.
    coverage_by_suite: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for suite in sorted(suite_counts):
        for w in window_lengths:
            coverage_by_suite[(suite, w)] = {
                "suite": suite,
                "window_length": w,
                "row_count": 0,
                "positive_rows": 0,
                "negative_rows": 0,
                "temporal_path_present": 0,
                "temporal_file_exists": 0,
                "temporal_file_has_25d": 0,
                "endpoint_step_present": 0,
                "causal_window_available_strict": 0,
                "causal_window_available_relaxed": 0,
            }

    alignment_rows: List[Dict[str, Any]] = []
    for row in rows:
        suite = row_suite(row)
        label = row_label(row)
        raw_path = str(row.get(temporal_path_col, "")) if temporal_path_col else ""
        resolved = unique_paths.get(raw_path)
        info = artifact_info.get(raw_path, {}) if raw_path else {}
        exists = bool(resolved and resolved.exists())
        has_25d = bool(info.get("has_all_25d_features")) if info else False
        step, step_col = find_step(row)
        for w in window_lengths:
            rec = coverage_by_suite[(suite, w)]
            rec["row_count"] += 1
            rec["positive_rows"] += 1 if label == 1 else 0
            rec["negative_rows"] += 1 if label == 0 else 0
            rec["temporal_path_present"] += 1 if raw_path else 0
            rec["temporal_file_exists"] += 1 if exists else 0
            rec["temporal_file_has_25d"] += 1 if has_25d else 0
            rec["endpoint_step_present"] += 1 if step is not None else 0
            strict = bool(exists and has_25d and step is not None and step + 1 >= w)
            relaxed = bool(exists and has_25d)
            rec["causal_window_available_strict"] += 1 if strict else 0
            rec["causal_window_available_relaxed"] += 1 if relaxed else 0
        if len(alignment_rows) < 5000:
            alignment_rows.append({
                "suite": suite,
                "split": row_split(row),
                "label": label,
                "group_key": row_group_key(row),
                "temporal_path_present": bool(raw_path),
                "temporal_file_exists": exists,
                "temporal_file_has_25d": has_25d,
                "endpoint_step": "" if step is None else step,
                "endpoint_step_column": step_col,
                "proposed_alignment": "causal_endpoint_window" if step is not None else "endpoint_unmapped_needs_c2e1_resolver",
            })

    coverage_rows: List[Dict[str, Any]] = []
    min_strict_rate = 1.0
    min_relaxed_rate = 1.0
    for (_, _), rec in sorted(coverage_by_suite.items()):
        n = max(1, int(rec["row_count"]))
        rec["temporal_path_present_rate"] = float(rec["temporal_path_present"]) / n
        rec["temporal_file_exists_rate"] = float(rec["temporal_file_exists"]) / n
        rec["temporal_file_has_25d_rate"] = float(rec["temporal_file_has_25d"]) / n
        rec["endpoint_step_present_rate"] = float(rec["endpoint_step_present"]) / n
        rec["causal_window_available_strict_rate"] = float(rec["causal_window_available_strict"]) / n
        rec["causal_window_available_relaxed_rate"] = float(rec["causal_window_available_relaxed"]) / n
        min_strict_rate = min(min_strict_rate, float(rec["causal_window_available_strict_rate"]))
        min_relaxed_rate = min(min_relaxed_rate, float(rec["causal_window_available_relaxed_rate"]))
        coverage_rows.append(rec)

    if min_relaxed_rate < args.min_suite_window_coverage:
        violations.append({
            "code": "LOW_TEMPORAL_ARTIFACT_COVERAGE",
            "severity": "hard",
            "min_relaxed_coverage": min_relaxed_rate,
            "required": args.min_suite_window_coverage,
        })
    if min_strict_rate < args.min_suite_window_coverage and not args.allow_endpoint_unmapped:
        violations.append({
            "code": "LOW_STRICT_ENDPOINT_WINDOW_COVERAGE",
            "severity": "hard",
            "min_strict_coverage": min_strict_rate,
            "required": args.min_suite_window_coverage,
            "hint": "Pass --allow-endpoint-unmapped only for design-only HOLD/PASS when C2e1 will implement an endpoint resolver.",
        })
    elif min_strict_rate < args.min_suite_window_coverage:
        warnings.append({
            "code": "STRICT_ENDPOINT_WINDOW_COVERAGE_LOW_ALLOWED",
            "severity": "warning",
            "min_strict_coverage": min_strict_rate,
            "required": args.min_suite_window_coverage,
        })

    # Additional no-future-leakage audit is a schema audit: proposed windows end at t.
    leakage_rows = split_leak_rows + forbidden_input_rows
    leakage_rows.append({
        "feature": "temporal_window_definition",
        "reason": "causal endpoint windows only; no future frame feature names are proposed",
        "violation_code": "NO_FUTURE_LEAKAGE_BY_SCHEMA",
    })

    hard_violations = [v for v in violations if v.get("severity") == "hard"]
    status = PASS if not hard_violations else HOLD

    report = {
        "gate": GATE,
        "status": status,
        "reason": "hard_violation_count=0" if status == PASS else f"hard_violation_count={len(hard_violations)}",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "git_commit": args.git_commit,
        "inputs": {
            "context_dataset": str(dataset_path),
            "context_dataset_sha256": sha256_file(dataset_path) if dataset_path.exists() else "",
            "context_schema": str(Path(args.context_schema).expanduser().resolve()) if args.context_schema else "",
            "window_lengths": window_lengths,
            "temporal_path_column": temporal_path_col,
            "temporal_roots": [str(p) for p in extra_roots],
        },
        "row_count": len(rows),
        "expected_rows": args.expected_rows,
        "suite_counts": dict(suite_counts),
        "label_counts": dict(label_counts),
        "context_model_input_count": len(context_input_names),
        "context_model_inputs_sample": context_input_names[:80],
        "unique_temporal_path_count": len(unique_paths),
        "inspected_temporal_artifact_count": inspected,
        "min_strict_window_coverage_rate": min_strict_rate,
        "min_relaxed_window_coverage_rate": min_relaxed_rate,
        "split_leakage_count": split_leakage_count,
        "forbidden_input_violation_count": len(forbidden_input_rows),
        "hard_violation_count": len(hard_violations),
        "warning_count": len(warnings),
        "violations_by_code": dict(Counter(str(v.get("code")) for v in violations)),
        "warnings_by_code": dict(Counter(str(v.get("code")) for v in warnings)),
        "recommendation": "proceed_to_C2E1_temporal_dataset_materialization" if status == PASS else "hold_fix_temporal_artifact_endpoint_or_coverage_before_C2E1",
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

    out_files_rows = [
        ("d4c2e0_temporal_detector_design_report.json", report),
        ("temporal_feature_schema_proposal.json", proposed_schema),
    ]
    for name, obj in out_files_rows:
        write_json(out / name, obj)

    coverage_fields = [
        "suite", "window_length", "row_count", "positive_rows", "negative_rows",
        "temporal_path_present", "temporal_path_present_rate",
        "temporal_file_exists", "temporal_file_exists_rate",
        "temporal_file_has_25d", "temporal_file_has_25d_rate",
        "endpoint_step_present", "endpoint_step_present_rate",
        "causal_window_available_strict", "causal_window_available_strict_rate",
        "causal_window_available_relaxed", "causal_window_available_relaxed_rate",
    ]
    write_csv(out / "temporal_window_coverage_by_suite.csv", coverage_rows, coverage_fields)
    write_csv(out / "temporal_label_alignment_audit.csv", alignment_rows, [
        "suite", "split", "label", "group_key", "temporal_path_present", "temporal_file_exists",
        "temporal_file_has_25d", "endpoint_step", "endpoint_step_column", "proposed_alignment",
    ])
    write_csv(out / "temporal_forbidden_leakage_audit.csv", leakage_rows, ["feature", "reason", "violation_code"])
    write_csv(out / "temporal_artifact_debug_sample.csv", debug_rows, [
        "raw_temporal_path", "resolved_path", "exists", "readable", "row_count_at_least",
        "has_all_25d_features", "missing_25d_features", "step_columns_present", "header_sample", "read_error",
    ])
    write_csv(out / "temporal_violations.csv", violations + warnings, [
        "code", "severity", "expected", "actual", "missing_suites", "count", "min_relaxed_coverage",
        "min_strict_coverage", "required", "hint", "message",
    ])

    checksum_entries = []
    for name in OUT_FILES:
        p = out / name
        if p.exists():
            checksum_entries.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checksum_entries})
    checksum_entries = []
    for name in OUT_FILES:
        p = out / name
        if p.exists():
            checksum_entries.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checksum_entries:
            f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "output_root": str(out),
        "row_count": len(rows),
        "hard_violation_count": len(hard_violations),
        "warning_count": len(warnings),
        "min_strict_window_coverage_rate": min_strict_rate,
        "min_relaxed_window_coverage_rate": min_relaxed_rate,
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
