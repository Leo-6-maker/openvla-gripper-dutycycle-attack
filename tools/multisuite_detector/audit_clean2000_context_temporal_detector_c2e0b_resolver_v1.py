#!/usr/bin/env python3
"""C2e0B resolver audit for clean causal temporal detector materialization.

This script fixes the two C2e0 design blockers without training:

1. LIBERO-10 temporal artifacts may expose SC5 25D features with telemetry
   aliases such as f_gripper_qpos instead of canonical gripper_qpos.
2. D4C1/D4C1B rows may not contain an explicit endpoint step. This audit tries
   to resolve the endpoint by matching the row's canonical 25D feature vector
   against the referenced temporal artifact using the validated alias mapping.

Boundary:
- CPU-only file inspection and endpoint-resolution audit.
- No detector training.
- No OpenVLA/LIBERO/runtime/rollout/intervention/attack.
- Causal windows only: endpoint index k permits rows [k-W+1, ..., k].
"""
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from gripper_attack.sc5_multisuite_detector_runtime import (  # type: ignore
        SC5_V2_FEATURES,
        validate_no_forbidden_inputs,
    )
except Exception:  # pragma: no cover
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
        forbidden = [
            "normalized_step", "timestep", "state_id", "episode_key", "run_id",
            "parent_id", "object_pose", "target_pose", "object_to_target",
            "teacher_window", "teacher_anchor", "attack_outcome", "rand_outcome",
            "manual_anchor", "oracle_window",
        ]
        bad = []
        for name in feature_names:
            low = str(name).lower()
            if any(hint in low for hint in forbidden):
                bad.append(name)
        if bad:
            raise ValueError(f"forbidden model-input feature names present: {bad}")


GATE = "D4C2E0B_CLEAN2000_TEMPORAL_ENDPOINT_ALIAS_RESOLVER_AUDIT"
PASS = "PASS_D4C2E0B_TEMPORAL_ENDPOINT_ALIAS_RESOLVER_READY_FOR_C2E1"
HOLD = "HOLD_D4C2E0B_TEMPORAL_ENDPOINT_ALIAS_RESOLVER_AUDIT"

REQUIRED_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
TEMPORAL_PATH_COLUMNS = ["temporal_path", "source_temporal_path", "stream_path", "artifact_path", "trajectory_path"]
STEP_COLUMN_CANDIDATES = ["step", "frame_idx", "frame_index", "row_step", "local_step", "artifact_step", "t"]
FORBIDDEN_SUBSTRINGS = [
    "object_pose", "target_pose", "object_target_distance", "object_to_target",
    "privileged", "sim_state", "oracle", "attack_outcome", "rollout_outcome",
    "success", "failure", "teacher_anchor", "teacher_window", "future_",
    "post_intervention", "openvla_hidden",
]
OUT_FILES = [
    "d4c2e0b_temporal_endpoint_alias_resolver_report.json",
    "temporal_feature_alias_schema.json",
    "temporal_artifact_alias_coverage.csv",
    "temporal_endpoint_resolver_audit.csv",
    "temporal_window_coverage_by_suite.csv",
    "temporal_forbidden_leakage_audit.csv",
    "temporal_violations.csv",
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


def parse_window_lengths(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("window lengths must be positive")
        vals.append(value)
    if not vals:
        raise ValueError("at least one window length is required")
    return sorted(set(vals))


def first_present(row: Dict[str, str], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if row.get(key) not in (None, ""):
            return str(row.get(key, ""))
    return default


def row_suite(row: Dict[str, str]) -> str:
    return str(row.get("suite", "unknown") or "unknown")


def row_split(row: Dict[str, str]) -> str:
    return str(row.get("split", row.get("dataset_split", "unknown")) or "unknown")


def row_group_key(row: Dict[str, str]) -> str:
    return first_present(row, ["group_key", "episode_key", "record_id", "parent_id", "run_id", "episode_id"])


def row_episode_key(row: Dict[str, str]) -> str:
    return first_present(row, ["episode_key", "group_key", "record_id", "parent_id", "run_id", "episode_id"])


def row_label(row: Dict[str, str]) -> int:
    for key in ["runtime_objective_label", "label", "y", "target"]:
        val = stable_int(row.get(key), None)
        if val in (0, 1):
            return int(val)
    status = str(row.get("teacher_label_status", row.get("label_status", "")))
    role = str(row.get("event_role", row.get("event_role_true", "")))
    if status in {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"} or role == "primary_attackable":
        return 1
    if status in {"VALID_AUXILIARY", "VALID_AUXILIARY_CANDIDATE", "NO_EVENT"} or role in {"auxiliary_manipulation", "unsupported_or_abstain", "distractor_or_setup"}:
        return 0
    return -1


def resolve_path(raw: str, dataset_path: Path, extra_roots: Sequence[Path]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    candidates: List[Path] = []
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


def alias_candidates(feature: str) -> List[str]:
    cands = [feature, f"f_{feature}", f"feature_{feature}", f"obs_{feature}"]
    extras = {
        "gripper_opening_proxy": ["opening_proxy", "f_opening_proxy", "gripper_width", "f_gripper_width"],
        "gripper_command": ["gripper_action", "f_gripper_action", "policy_gripper_command", "f_policy_gripper_command"],
        "eef_x": ["eef_pos_x", "f_eef_pos_x", "robot0_eef_x", "f_robot0_eef_x"],
        "eef_y": ["eef_pos_y", "f_eef_pos_y", "robot0_eef_y", "f_robot0_eef_y"],
        "eef_z": ["eef_pos_z", "f_eef_pos_z", "robot0_eef_z", "f_robot0_eef_z"],
        "eef_vx": ["eef_vel_x", "f_eef_vel_x", "robot0_eef_vx", "f_robot0_eef_vx"],
        "eef_vy": ["eef_vel_y", "f_eef_vel_y", "robot0_eef_vy", "f_robot0_eef_vy"],
        "eef_vz": ["eef_vel_z", "f_eef_vel_z", "robot0_eef_vz", "f_robot0_eef_vz"],
    }
    cands.extend(extras.get(feature, []))
    # preserve order while removing duplicates
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_alias_map(header: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    header_set = set(header)
    resolved: Dict[str, str] = {}
    all_present: Dict[str, List[str]] = {}
    for feature in SC5_V2_FEATURES:
        present = [c for c in alias_candidates(feature) if c in header_set]
        all_present[feature] = present
        if present:
            # prefer exact canonical, then f_ canonical, then first remaining alias
            if feature in present:
                resolved[feature] = feature
            elif f"f_{feature}" in present:
                resolved[feature] = f"f_{feature}"
            else:
                resolved[feature] = present[0]
    return resolved, all_present


def read_artifact(path: Path, alias_map: Dict[str, str]) -> Tuple[List[str], List[Dict[str, str]], np.ndarray]:
    rows = read_csv(path)
    header = list(rows[0].keys()) if rows else []
    arr = np.full((len(rows), len(SC5_V2_FEATURES)), np.nan, dtype=np.float32)
    for j, feature in enumerate(SC5_V2_FEATURES):
        col = alias_map.get(feature, "")
        if not col:
            continue
        for i, row in enumerate(rows):
            arr[i, j] = stable_float(row.get(col))
    return header, rows, arr


def context_vector(row: Dict[str, str]) -> np.ndarray:
    return np.asarray([stable_float(row.get(f)) for f in SC5_V2_FEATURES], dtype=np.float32)


def find_explicit_step(row: Dict[str, str]) -> Tuple[Optional[int], str]:
    for key in STEP_COLUMN_CANDIDATES:
        value = stable_int(row.get(key), None)
        if value is not None:
            return value, key
    return None, ""


def match_endpoint(ctx: np.ndarray, artifact: np.ndarray, min_features: int, max_abs: float, mean_abs: float) -> Tuple[Optional[int], Dict[str, Any]]:
    if artifact.size == 0:
        return None, {"match_status": "empty_artifact", "best_index": "", "best_feature_count": 0, "best_max_abs": "", "best_mean_abs": ""}
    valid_ctx = np.isfinite(ctx)
    best_idx: Optional[int] = None
    best_mean = float("inf")
    best_max = float("inf")
    best_count = 0
    for i in range(artifact.shape[0]):
        valid = valid_ctx & np.isfinite(artifact[i])
        count = int(valid.sum())
        if count < min_features:
            continue
        diff = np.abs(ctx[valid] - artifact[i, valid])
        d_mean = float(diff.mean())
        d_max = float(diff.max())
        if (d_mean, d_max) < (best_mean, best_max):
            best_idx = i
            best_mean = d_mean
            best_max = d_max
            best_count = count
    if best_idx is None:
        return None, {"match_status": "insufficient_overlap", "best_index": "", "best_feature_count": best_count, "best_max_abs": "", "best_mean_abs": ""}
    ok = best_count >= min_features and best_max <= max_abs and best_mean <= mean_abs
    return (best_idx if ok else None), {
        "match_status": "matched" if ok else "nearest_above_tolerance",
        "best_index": best_idx,
        "best_feature_count": best_count,
        "best_max_abs": best_max,
        "best_mean_abs": best_mean,
    }


def audit_split_leakage(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    key_to_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = row_episode_key(row)
        if key:
            key_to_splits[key].add(row_split(row))
    out = []
    for key, splits in key_to_splits.items():
        real = sorted(s for s in splits if s and s != "unknown")
        if len(set(real)) > 1:
            out.append({"feature": key, "reason": ";".join(real), "violation_code": "SPLIT_LEAKAGE"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True)
    ap.add_argument("--context-schema", default="")
    ap.add_argument("--expected-rows", type=int, default=3717)
    ap.add_argument("--window-lengths", default="8,16,32")
    ap.add_argument("--temporal-root", action="append", default=[])
    ap.add_argument("--min-suite-window-coverage", type=float, default=0.95)
    ap.add_argument("--endpoint-match-min-features", type=int, default=8)
    ap.add_argument("--endpoint-match-max-abs", type=float, default=1e-3)
    ap.add_argument("--endpoint-match-mean-abs", type=float, default=1e-4)
    ap.add_argument("--max-endpoint-debug-rows", type=int, default=20000)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", default="")
    args = ap.parse_args()

    started = time.time()
    dataset_path = Path(args.context_dataset).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    window_lengths = parse_window_lengths(args.window_lengths)
    extra_roots = [Path(p).expanduser().resolve() for p in args.temporal_root]

    rows = read_csv(dataset_path)
    header = list(rows[0].keys()) if rows else []
    temporal_col = ""
    for col in TEMPORAL_PATH_COLUMNS:
        if col in header:
            temporal_col = col
            break

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if args.expected_rows and len(rows) != args.expected_rows:
        violations.append({"code": "UNEXPECTED_ROW_COUNT", "severity": "hard", "expected": args.expected_rows, "actual": len(rows)})
    if not temporal_col:
        violations.append({"code": "TEMPORAL_PATH_COLUMN_MISSING", "severity": "hard"})

    suite_counts = Counter(row_suite(r) for r in rows)
    missing_suites = [s for s in REQUIRED_SUITES if suite_counts.get(s, 0) <= 0]
    if missing_suites:
        violations.append({"code": "MISSING_SUITE_COVERAGE", "severity": "hard", "missing_suites": ";".join(missing_suites)})

    labels = Counter(str(row_label(r)) for r in rows)
    if labels.get("0", 0) <= 0 or labels.get("1", 0) <= 0:
        violations.append({"code": "MISSING_RUNTIME_OBJECTIVE_CLASS", "severity": "hard", "label_counts": dict(labels)})

    forbidden_rows: List[Dict[str, Any]] = []
    try:
        validate_no_forbidden_inputs([f"lag00_{f}" for f in SC5_V2_FEATURES])
    except Exception as exc:
        violations.append({"code": "FORBIDDEN_TEMPORAL_FEATURE", "severity": "hard", "message": str(exc)})
        forbidden_rows.append({"feature": "temporal_schema", "reason": str(exc), "violation_code": "FORBIDDEN_TEMPORAL_FEATURE"})
    for name in header:
        low = str(name).lower()
        if any(bad in low for bad in FORBIDDEN_SUBSTRINGS):
            forbidden_rows.append({"feature": name, "reason": "forbidden substring", "violation_code": "FORBIDDEN_CONTEXT_COLUMN_PRESENT"})
    split_leaks = audit_split_leakage(rows)
    if split_leaks:
        violations.append({"code": "SPLIT_LEAKAGE", "severity": "hard", "count": len(split_leaks)})

    # Load artifact cache with alias coverage.
    unique_raw_paths = sorted({str(r.get(temporal_col, "")) for r in rows if temporal_col and r.get(temporal_col)})
    artifact_cache: Dict[str, Dict[str, Any]] = {}
    artifact_alias_rows: List[Dict[str, Any]] = []
    alias_schema: Dict[str, Any] = {
        "feature_family": "C2E_TEMPORAL_25D_ALIAS_SCHEMA_V1",
        "canonical_features": list(SC5_V2_FEATURES),
        "alias_candidates": {f: alias_candidates(f) for f in SC5_V2_FEATURES},
        "endpoint_resolver": {
            "primary": "explicit endpoint step when available",
            "fallback": "feature-match canonical context 25D vector to aliased temporal artifact rows",
            "causal_window": "use rows [endpoint-W+1, ..., endpoint] only",
            "endpoint_match_min_features": args.endpoint_match_min_features,
            "endpoint_match_max_abs": args.endpoint_match_max_abs,
            "endpoint_match_mean_abs": args.endpoint_match_mean_abs,
        },
    }

    for raw in unique_raw_paths:
        path = resolve_path(raw, dataset_path, extra_roots)
        rec: Dict[str, Any] = {"raw_temporal_path": raw, "resolved_path": str(path) if path else "", "exists": bool(path and path.exists())}
        if not path or not path.exists():
            rec.update({"readable": False, "row_count": 0, "resolved_feature_count": 0, "missing_features": ";".join(SC5_V2_FEATURES)})
            artifact_cache[raw] = {"meta": rec, "rows": [], "arr": np.zeros((0, len(SC5_V2_FEATURES)), dtype=np.float32), "alias_map": {}}
            artifact_alias_rows.append(rec)
            continue
        try:
            art_rows = read_csv(path)
            art_header = list(art_rows[0].keys()) if art_rows else []
            alias_map, all_present = resolve_alias_map(art_header)
            _, _, arr = read_artifact(path, alias_map)
            missing = [f for f in SC5_V2_FEATURES if f not in alias_map]
            rec.update({
                "readable": True,
                "row_count": len(art_rows),
                "resolved_feature_count": len(alias_map),
                "has_all_25d_by_alias": len(missing) == 0,
                "missing_features": ";".join(missing),
                "alias_map_json": json.dumps(alias_map, sort_keys=True),
                "header_sample": ";".join(art_header[:60]),
            })
            artifact_cache[raw] = {"meta": rec, "rows": art_rows, "arr": arr, "alias_map": alias_map, "path": path}
            artifact_alias_rows.append(rec)
        except Exception as exc:
            rec.update({"readable": False, "row_count": 0, "resolved_feature_count": 0, "has_all_25d_by_alias": False, "missing_features": ";".join(SC5_V2_FEATURES), "read_error": str(exc)})
            artifact_cache[raw] = {"meta": rec, "rows": [], "arr": np.zeros((0, len(SC5_V2_FEATURES)), dtype=np.float32), "alias_map": {}}
            artifact_alias_rows.append(rec)

    coverage: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for suite in sorted(suite_counts):
        for w in window_lengths:
            coverage[(suite, w)] = {
                "suite": suite,
                "window_length": w,
                "row_count": 0,
                "positive_rows": 0,
                "negative_rows": 0,
                "temporal_file_exists": 0,
                "alias_25d_complete": 0,
                "endpoint_resolved": 0,
                "causal_window_available": 0,
            }

    endpoint_rows: List[Dict[str, Any]] = []
    endpoint_status_counts = Counter()
    for idx, row in enumerate(rows):
        suite = row_suite(row)
        label = row_label(row)
        raw = str(row.get(temporal_col, "")) if temporal_col else ""
        cache = artifact_cache.get(raw, {"meta": {}, "arr": np.zeros((0, len(SC5_V2_FEATURES)), dtype=np.float32)})
        meta = cache.get("meta", {})
        arr = cache.get("arr", np.zeros((0, len(SC5_V2_FEATURES)), dtype=np.float32))
        endpoint, step_col = find_explicit_step(row)
        strategy = "explicit_step" if endpoint is not None else "feature_match"
        match_meta: Dict[str, Any] = {}
        if endpoint is None:
            endpoint, match_meta = match_endpoint(
                context_vector(row),
                arr,
                min_features=args.endpoint_match_min_features,
                max_abs=args.endpoint_match_max_abs,
                mean_abs=args.endpoint_match_mean_abs,
            )
        else:
            match_meta = {"match_status": "explicit_step", "best_index": endpoint, "best_feature_count": "", "best_max_abs": "", "best_mean_abs": ""}
        resolved = endpoint is not None and endpoint >= 0 and endpoint < int(arr.shape[0])
        status = str(match_meta.get("match_status", "unknown")) if not resolved else "resolved"
        endpoint_status_counts[status] += 1
        for w in window_lengths:
            c = coverage[(suite, w)]
            c["row_count"] += 1
            c["positive_rows"] += 1 if label == 1 else 0
            c["negative_rows"] += 1 if label == 0 else 0
            c["temporal_file_exists"] += 1 if bool(meta.get("exists")) else 0
            c["alias_25d_complete"] += 1 if bool(meta.get("has_all_25d_by_alias")) else 0
            c["endpoint_resolved"] += 1 if resolved else 0
            c["causal_window_available"] += 1 if bool(resolved and endpoint is not None and endpoint + 1 >= w) else 0
        if len(endpoint_rows) < args.max_endpoint_debug_rows:
            endpoint_rows.append({
                "row_index": idx,
                "suite": suite,
                "split": row_split(row),
                "label": label,
                "group_key": row_group_key(row),
                "temporal_path": raw,
                "strategy": strategy,
                "endpoint_index": "" if endpoint is None else endpoint,
                "resolved": resolved,
                "match_status": match_meta.get("match_status", ""),
                "best_feature_count": match_meta.get("best_feature_count", ""),
                "best_max_abs": match_meta.get("best_max_abs", ""),
                "best_mean_abs": match_meta.get("best_mean_abs", ""),
                "artifact_row_count": int(arr.shape[0]),
                "artifact_has_all_25d_by_alias": bool(meta.get("has_all_25d_by_alias")),
                "step_column": step_col,
            })

    coverage_rows: List[Dict[str, Any]] = []
    min_window_rate = 1.0
    min_alias_rate = 1.0
    min_endpoint_rate = 1.0
    for key in sorted(coverage):
        c = coverage[key]
        n = max(1, int(c["row_count"]))
        c["temporal_file_exists_rate"] = float(c["temporal_file_exists"]) / n
        c["alias_25d_complete_rate"] = float(c["alias_25d_complete"]) / n
        c["endpoint_resolved_rate"] = float(c["endpoint_resolved"]) / n
        c["causal_window_available_rate"] = float(c["causal_window_available"]) / n
        min_alias_rate = min(min_alias_rate, float(c["alias_25d_complete_rate"]))
        min_endpoint_rate = min(min_endpoint_rate, float(c["endpoint_resolved_rate"]))
        min_window_rate = min(min_window_rate, float(c["causal_window_available_rate"]))
        coverage_rows.append(c)

    if min_alias_rate < args.min_suite_window_coverage:
        violations.append({"code": "LOW_ALIAS_25D_COVERAGE", "severity": "hard", "actual": min_alias_rate, "required": args.min_suite_window_coverage})
    if min_endpoint_rate < args.min_suite_window_coverage:
        violations.append({"code": "LOW_ENDPOINT_RESOLUTION_COVERAGE", "severity": "hard", "actual": min_endpoint_rate, "required": args.min_suite_window_coverage})
    if min_window_rate < args.min_suite_window_coverage:
        violations.append({"code": "LOW_CAUSAL_WINDOW_COVERAGE", "severity": "hard", "actual": min_window_rate, "required": args.min_suite_window_coverage})

    leakage_rows = split_leaks + forbidden_rows + [{"feature": "temporal_window_definition", "reason": "causal endpoint windows only; no future rows proposed", "violation_code": "NO_FUTURE_LEAKAGE_BY_SCHEMA"}]
    hard = [v for v in violations if v.get("severity") == "hard"]
    status = PASS if not hard else HOLD

    report = {
        "gate": GATE,
        "status": status,
        "reason": "hard_violation_count=0" if status == PASS else f"hard_violation_count={len(hard)}",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "git_commit": args.git_commit,
        "inputs": {
            "context_dataset": str(dataset_path),
            "context_dataset_sha256": sha256_file(dataset_path) if dataset_path.exists() else "",
            "context_schema": str(Path(args.context_schema).expanduser().resolve()) if args.context_schema else "",
            "window_lengths": window_lengths,
            "temporal_path_column": temporal_col,
        },
        "row_count": len(rows),
        "expected_rows": args.expected_rows,
        "suite_counts": dict(suite_counts),
        "label_counts": dict(labels),
        "unique_temporal_path_count": len(unique_raw_paths),
        "artifact_count": len(artifact_alias_rows),
        "endpoint_status_counts": dict(endpoint_status_counts),
        "min_alias_25d_coverage_rate": min_alias_rate,
        "min_endpoint_resolution_rate": min_endpoint_rate,
        "min_causal_window_coverage_rate": min_window_rate,
        "split_leakage_count": len(split_leaks),
        "forbidden_input_violation_count": len(forbidden_rows),
        "hard_violation_count": len(hard),
        "warning_count": len(warnings),
        "violations_by_code": dict(Counter(str(v.get("code")) for v in violations)),
        "warnings_by_code": dict(Counter(str(v.get("code")) for v in warnings)),
        "recommendation": "proceed_to_C2E1_temporal_dataset_materialization" if status == PASS else "hold_fix_alias_or_endpoint_resolution_before_C2E1",
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

    write_json(out / "d4c2e0b_temporal_endpoint_alias_resolver_report.json", report)
    write_json(out / "temporal_feature_alias_schema.json", alias_schema)
    write_csv(out / "temporal_artifact_alias_coverage.csv", artifact_alias_rows, [
        "raw_temporal_path", "resolved_path", "exists", "readable", "row_count", "resolved_feature_count",
        "has_all_25d_by_alias", "missing_features", "alias_map_json", "header_sample", "read_error",
    ])
    write_csv(out / "temporal_endpoint_resolver_audit.csv", endpoint_rows, [
        "row_index", "suite", "split", "label", "group_key", "temporal_path", "strategy", "endpoint_index",
        "resolved", "match_status", "best_feature_count", "best_max_abs", "best_mean_abs",
        "artifact_row_count", "artifact_has_all_25d_by_alias", "step_column",
    ])
    write_csv(out / "temporal_window_coverage_by_suite.csv", coverage_rows, [
        "suite", "window_length", "row_count", "positive_rows", "negative_rows", "temporal_file_exists",
        "temporal_file_exists_rate", "alias_25d_complete", "alias_25d_complete_rate", "endpoint_resolved",
        "endpoint_resolved_rate", "causal_window_available", "causal_window_available_rate",
    ])
    write_csv(out / "temporal_forbidden_leakage_audit.csv", leakage_rows, ["feature", "reason", "violation_code"])
    write_csv(out / "temporal_violations.csv", violations + warnings, ["code", "severity", "expected", "actual", "required", "count", "message", "missing_suites"])

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
        "min_alias_25d_coverage_rate": min_alias_rate,
        "min_endpoint_resolution_rate": min_endpoint_rate,
        "min_causal_window_coverage_rate": min_window_rate,
        "hard_violation_count": len(hard),
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
