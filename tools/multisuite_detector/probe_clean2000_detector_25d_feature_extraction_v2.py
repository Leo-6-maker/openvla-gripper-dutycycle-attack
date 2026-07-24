#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs

GATE = "D2C2_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBE_ROBUST_SCHEMA"
PASS = "PASS_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBED_ROBUST_SCHEMA"
OUT_FILES = [
    "detector_25d_feature_extraction_probe_v2_report.json",
    "detector_25d_feature_probe_v2_by_row.csv",
    "detector_25d_feature_probe_v2_ready_feature_values.csv",
    "detector_25d_feature_probe_v2_by_feature.csv",
    "detector_25d_feature_probe_v2_failures.csv",
    "detector_25d_feature_probe_v2_exclusions.csv",
    "checksum_report.json",
]

STEP_ALIASES = ["step", "timestep", "frame", "frame_idx", "step_idx", "index"]
DIRECT_ALIASES: Dict[str, List[str]] = {
    "gripper_command": ["gripper_command", "command_gripper", "gripper_cmd", "action_gripper", "gripper_action", "a_gripper", "action[-1]", "action_6", "action_7"],
    "gripper_qpos": ["gripper_qpos", "robot0_gripper_qpos", "obs_gripper_qpos", "gripper_width", "gripper_opening", "opening", "qpos_gripper"],
    "gripper_opening_proxy": ["gripper_opening_proxy", "opening_proxy", "gripper_width", "gripper_opening", "opening", "gripper_qpos", "robot0_gripper_qpos", "obs_gripper_qpos"],
    "eef_x": ["eef_x", "eef_pos_x", "robot0_eef_x", "robot0_eef_pos_x", "ee_x", "ee_pos_x", "tcp_x"],
    "eef_y": ["eef_y", "eef_pos_y", "robot0_eef_y", "robot0_eef_pos_y", "ee_y", "ee_pos_y", "tcp_y"],
    "eef_z": ["eef_z", "eef_pos_z", "robot0_eef_z", "robot0_eef_pos_z", "ee_z", "ee_pos_z", "tcp_z"],
    "eef_vx": ["eef_vx", "eef_vel_x", "robot0_eef_vx", "robot0_eef_vel_x", "ee_vx", "ee_vel_x", "tcp_vx"],
    "eef_vy": ["eef_vy", "eef_vel_y", "robot0_eef_vy", "robot0_eef_vel_y", "ee_vy", "ee_vel_y", "tcp_vy"],
    "eef_vz": ["eef_vz", "eef_vel_z", "robot0_eef_vz", "robot0_eef_vel_z", "ee_vz", "ee_vel_z", "tcp_vz"],
    "action_dx": ["action_dx", "delta_x", "act_dx", "action_0", "action[0]", "a0"],
    "action_dy": ["action_dy", "delta_y", "act_dy", "action_1", "action[1]", "a1"],
    "action_dz": ["action_dz", "delta_z", "act_dz", "action_2", "action[2]", "a2"],
    "action_gripper": ["action_gripper", "gripper_action", "a_gripper", "gripper_command", "command_gripper", "action_6", "action_7", "action[-1]"],
}
DERIVED_FEATURES = {
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count", "close_onset", "time_since_close",
    "eef_speed", "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
    "opening_proxy_variance_5", "eef_speed_variance_5",
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(flatten_json(obj))
    return out


def flatten_json(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_json(v, key))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out[f"{key}_{i}"] = x
                out[f"{key}[{i}]"] = x
        else:
            out[key] = v
    return out


def read_temporal(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".csv":
        return read_csv(p)
    if p.suffix.lower() in {".jsonl", ".jl"}:
        return read_jsonl(p)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for key in ["steps", "records", "trajectory", "telemetry", "phase_cues"]:
                if isinstance(obj.get(key), list):
                    return [flatten_json(x) for x in obj[key] if isinstance(x, dict)]
        if isinstance(obj, list):
            return [flatten_json(x) for x in obj if isinstance(x, dict)]
    raise ValueError(f"unsupported temporal format: {path}")


def finite_float(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        v = float(value)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("/", "_").replace(".", "_").replace("-", "_")


def row_value(row: Dict[str, Any], names: Iterable[str]) -> Tuple[float, str]:
    if not row:
        return math.nan, ""
    by_norm = {normalize_name(k): k for k in row.keys()}
    for name in names:
        raw = row.get(name, None)
        val = finite_float(raw)
        if math.isfinite(val):
            return val, name
        key = by_norm.get(normalize_name(name))
        if key is not None:
            val = finite_float(row.get(key))
            if math.isfinite(val):
                return val, key
    return math.nan, ""


def series_value(rows: List[Dict[str, Any]], idx: int, names: Iterable[str]) -> Tuple[float, str]:
    if not rows or idx < 0 or idx >= len(rows):
        return math.nan, ""
    return row_value(rows[idx], names)


def feature_precomputed(row: Dict[str, Any], feature: str) -> Tuple[float, str]:
    aliases = [f"f_{feature}", f"feature_{feature}", f"sc5_{feature}"]
    return row_value(row, aliases)


def get_steps(rows: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    best_field = ""
    best_vals: List[float] = []
    best_count = -1
    for field in STEP_ALIASES:
        vals = [finite_float(r.get(field)) for r in rows]
        count = sum(math.isfinite(v) for v in vals)
        if count > best_count:
            best_count = count
            best_field = field
            best_vals = vals
    if best_count >= max(1, int(0.5 * len(rows))):
        return [v if math.isfinite(v) else float(i) for i, v in enumerate(best_vals)], best_field
    return [float(i) for i in range(len(rows))], "row_index"


def nearest_index(steps: List[float], target: float) -> int:
    if not steps:
        return 0
    return min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(target)))


def choose_index(binding: Dict[str, Any], manifest: Dict[str, Any], rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[int, str, float]:
    steps, step_field = get_steps(rows)
    event_status = str(binding.get("event_status") or manifest.get("event_status") or "")
    anchor = finite_float(binding.get("teacher_anchor_step"))
    if not math.isfinite(anchor):
        anchor = finite_float(manifest.get("teacher_anchor_step"))
    if math.isfinite(anchor) and anchor >= 0:
        return nearest_index(steps, anchor), f"nearest_teacher_anchor_step:{step_field}", anchor
    if event_status == "NO_EVENT":
        if args.no_event_step_policy == "middle":
            idx = len(rows) // 2
        elif args.no_event_step_policy == "last":
            idx = max(0, len(rows) - 1)
        else:
            idx = 0
        return idx, f"no_event_{args.no_event_step_policy}:{step_field}", steps[idx] if steps else float(idx)
    idx = max(0, len(rows) - 1)
    return idx, f"fallback_last:{step_field}", steps[idx] if steps else float(idx)


def manifest_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("record_id", "")), str(row.get("event_id", "")), str(row.get("segment_id", "")),
        str(row.get("segment_index", "")), str(row.get("event_status", "")),
    )


def build_manifest_indexes(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, str, str, str], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        label_id = str(row.get("label_row_id") or row.get("__label_row_id") or f"row_{i:06d}")
        new = dict(row)
        new["label_row_id"] = label_id
        by_id[label_id] = new
        by_key[manifest_key(new)] = new
        by_record[str(new.get("record_id", ""))].append(new)
    return by_id, by_key, by_record


def match_manifest(binding: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], by_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]], by_record: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    label_id = str(binding.get("label_row_id", ""))
    if label_id and label_id in by_id:
        return by_id[label_id]
    key = manifest_key(binding)
    if key in by_key:
        return by_key[key]
    rid = str(binding.get("record_id", ""))
    candidates = by_record.get(rid, [])
    if len(candidates) == 1:
        return candidates[0]
    for cand in candidates:
        if str(cand.get("event_id", "")) == str(binding.get("event_id", "")) and str(cand.get("segment_id", "")) == str(binding.get("segment_id", "")):
            return cand
    return {}


def percentile(values: List[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def series(rows: List[Dict[str, Any]], names: Iterable[str]) -> Tuple[List[float], str]:
    best_field = ""
    best_vals: List[float] = []
    best_count = -1
    for name in names:
        vals = []
        for row in rows:
            v, _ = row_value(row, [name])
            vals.append(v)
        count = sum(math.isfinite(v) for v in vals)
        if count > best_count:
            best_count = count
            best_field = name
            best_vals = vals
    return best_vals, best_field if best_count > 0 else ""


def get_base_series(rows: List[Dict[str, Any]], feature: str) -> Tuple[List[float], str, str]:
    vals = []
    count = 0
    for row in rows:
        v, field = feature_precomputed(row, feature)
        vals.append(v)
        count += int(math.isfinite(v))
    if count >= max(1, int(0.5 * len(rows))):
        return vals, f"f_{feature}", "pre_computed_f_prefix"
    vals, field = series(rows, [feature] + DIRECT_ALIASES.get(feature, []))
    if any(math.isfinite(v) for v in vals):
        return vals, field, "direct_raw_column"
    return vals, "", "missing"


def velocity_from_position(rows: List[Dict[str, Any]], axis: str, idx: int) -> Tuple[float, str]:
    pos_feature = f"eef_{axis}"
    vals, field, _ = get_base_series(rows, pos_feature)
    if not vals or idx <= 0 or idx >= len(vals) or not math.isfinite(vals[idx]) or not math.isfinite(vals[idx - 1]):
        return math.nan, ""
    return vals[idx] - vals[idx - 1], f"derived_velocity_from_{field}"


def close_mask_from_signal(rows: List[Dict[str, Any]]) -> Tuple[List[bool], str]:
    vals, field, _ = get_base_series(rows, "gripper_command")
    if not any(math.isfinite(v) for v in vals):
        vals, field, _ = get_base_series(rows, "action_gripper")
    finite = [v for v in vals if math.isfinite(v)]
    if len(set(round(v, 6) for v in finite)) >= 2:
        low = percentile(finite, 0.25)
        return [math.isfinite(v) and v <= low for v in vals], field or "gripper_command_quantile"
    open_vals, open_field, _ = get_base_series(rows, "gripper_opening_proxy")
    finite_open = [v for v in open_vals if math.isfinite(v)]
    if finite_open:
        low = percentile(finite_open, 0.35)
        return [math.isfinite(v) and v <= low for v in open_vals], open_field or "opening_proxy_quantile"
    return [False for _ in rows], ""


def last_close_onset(mask: List[bool], idx: int) -> int | None:
    last = None
    prev = False
    for i, val in enumerate(mask[: idx + 1]):
        if val and not prev:
            last = i
        prev = val
    return last


def compute_history_features(rows: List[Dict[str, Any]], idx: int, values: Dict[str, float], methods: Dict[str, str]) -> None:
    mask, source = close_mask_from_signal(rows)
    if not mask:
        return
    idx = max(0, min(idx, len(mask) - 1))
    close_streak = 0
    j = idx
    while j >= 0 and mask[j]:
        close_streak += 1
        j -= 1
    open_streak = 0
    j = idx
    while j >= 0 and not mask[j]:
        open_streak += 1
        j -= 1
    window = mask[max(0, idx - 4): idx + 1]
    flips = sum(1 for a, b in zip(window, window[1:]) if a != b)
    prev = mask[idx - 1] if idx > 0 else False
    onset_now = bool(mask[idx] and not prev)
    onset_idx = last_close_onset(mask, idx)
    time_since = idx - onset_idx if onset_idx is not None else 0
    for feat, val in [
        ("recent_close_streak", close_streak), ("recent_open_streak", open_streak),
        ("recent_gripper_flip_count", flips), ("close_onset", int(onset_now)),
        ("time_since_close", time_since),
    ]:
        if not math.isfinite(values.get(feat, math.nan)):
            values[feat] = float(val)
            methods[feat] = f"derived_from_close_mask:{source}" if source else "derived_default_no_close_signal"


def variance_last(vals: List[float], idx: int, n: int = 5) -> float:
    xs = [v for v in vals[max(0, idx - n + 1): idx + 1] if math.isfinite(v)]
    if not xs:
        return math.nan
    if len(xs) == 1:
        return 0.0
    return float(statistics.pvariance(xs))


def compute_features(rows: List[Dict[str, Any]], idx: int) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
    values: Dict[str, float] = {}
    methods: Dict[str, str] = {}
    fields: Dict[str, str] = {}
    row = rows[idx]
    for feat in SC5_V2_FEATURES:
        v, field = feature_precomputed(row, feat)
        if math.isfinite(v):
            values[feat] = v
            methods[feat] = "pre_computed_f_prefix"
            fields[feat] = field
            continue
        if feat in DIRECT_ALIASES or feat in ["eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz", "action_dx", "action_dy", "action_dz", "action_gripper"]:
            names = [feat] + DIRECT_ALIASES.get(feat, [])
            v, field = row_value(row, names)
            if math.isfinite(v):
                values[feat] = v
                methods[feat] = "direct_raw_column"
                fields[feat] = field
    for axis in ["x", "y", "z"]:
        feat = f"eef_v{axis}"
        if not math.isfinite(values.get(feat, math.nan)):
            v, field = velocity_from_position(rows, axis, idx)
            if math.isfinite(v):
                values[feat] = v
                methods[feat] = "derived_from_raw"
                fields[feat] = field
    if not math.isfinite(values.get("eef_speed", math.nan)):
        vx, vy, vz = values.get("eef_vx", math.nan), values.get("eef_vy", math.nan), values.get("eef_vz", math.nan)
        if all(math.isfinite(v) for v in [vx, vy, vz]):
            values["eef_speed"] = math.sqrt(vx * vx + vy * vy + vz * vz)
            methods["eef_speed"] = "derived_from_raw"
            fields["eef_speed"] = "eef_vx,eef_vy,eef_vz"
    compute_history_features(rows, idx, values, methods)
    qpos_vals, qpos_field, _ = get_base_series(rows, "gripper_qpos")
    opening_vals, opening_field, _ = get_base_series(rows, "gripper_opening_proxy")
    eefz_vals, eefz_field, _ = get_base_series(rows, "eef_z")
    speed_vals = []
    for i in range(len(rows)):
        vv, mm, _ = compute_minimal_speed(rows, i)
        speed_vals.append(vv)
    if not math.isfinite(values.get("qpos_delta_1", math.nan)) and idx >= 1 and idx < len(qpos_vals) and math.isfinite(qpos_vals[idx]) and math.isfinite(qpos_vals[idx - 1]):
        values["qpos_delta_1"] = qpos_vals[idx] - qpos_vals[idx - 1]
        methods["qpos_delta_1"] = "derived_from_raw"
        fields["qpos_delta_1"] = qpos_field
    if not math.isfinite(values.get("qpos_delta_3", math.nan)) and idx >= 3 and idx < len(qpos_vals) and math.isfinite(qpos_vals[idx]) and math.isfinite(qpos_vals[idx - 3]):
        values["qpos_delta_3"] = qpos_vals[idx] - qpos_vals[idx - 3]
        methods["qpos_delta_3"] = "derived_from_raw"
        fields["qpos_delta_3"] = qpos_field
    if not math.isfinite(values.get("opening_proxy_delta_3", math.nan)) and idx >= 3 and idx < len(opening_vals) and math.isfinite(opening_vals[idx]) and math.isfinite(opening_vals[idx - 3]):
        values["opening_proxy_delta_3"] = opening_vals[idx] - opening_vals[idx - 3]
        methods["opening_proxy_delta_3"] = "derived_from_raw"
        fields["opening_proxy_delta_3"] = opening_field
    if not math.isfinite(values.get("opening_proxy_variance_5", math.nan)):
        v = variance_last(opening_vals, idx, 5)
        if math.isfinite(v):
            values["opening_proxy_variance_5"] = v
            methods["opening_proxy_variance_5"] = "derived_from_raw"
            fields["opening_proxy_variance_5"] = opening_field
    if not math.isfinite(values.get("eef_speed_variance_5", math.nan)):
        v = variance_last(speed_vals, idx, 5)
        if math.isfinite(v):
            values["eef_speed_variance_5"] = v
            methods["eef_speed_variance_5"] = "derived_from_raw"
            fields["eef_speed_variance_5"] = "eef_speed_series"
    if not math.isfinite(values.get("eef_z_delta_since_close", math.nan)):
        close_onset_f, close_field = row_value(row, ["f_close_onset", "close_onset"])
        if math.isfinite(close_onset_f) and close_onset_f >= 0.5 and idx < len(eefz_vals) and math.isfinite(eefz_vals[idx]):
            values["eef_z_delta_since_close"] = 0.0
            methods["eef_z_delta_since_close"] = "derived_zero_at_current_close_onset"
            fields["eef_z_delta_since_close"] = close_field or "close_onset"
        else:
            mask, src = close_mask_from_signal(rows)
            close_idx = last_close_onset(mask, idx) if mask else None
            if close_idx is not None and close_idx < len(eefz_vals) and idx < len(eefz_vals) and math.isfinite(eefz_vals[idx]) and math.isfinite(eefz_vals[close_idx]):
                values["eef_z_delta_since_close"] = eefz_vals[idx] - eefz_vals[close_idx]
                methods["eef_z_delta_since_close"] = f"derived_from_close_onset:{src}"
                fields["eef_z_delta_since_close"] = eefz_field
    return values, methods, fields


def compute_minimal_speed(rows: List[Dict[str, Any]], idx: int) -> Tuple[float, str, str]:
    row = rows[idx]
    v, field = feature_precomputed(row, "eef_speed")
    if math.isfinite(v):
        return v, "pre_computed_f_prefix", field
    vals = []
    for feat in ["eef_vx", "eef_vy", "eef_vz"]:
        vv, ff = feature_precomputed(row, feat)
        if not math.isfinite(vv):
            vv, ff = row_value(row, [feat] + DIRECT_ALIASES.get(feat, []))
        vals.append(vv)
    if all(math.isfinite(x) for x in vals):
        return math.sqrt(sum(x * x for x in vals)), "derived_from_raw", "eef_vx,eef_vy,eef_vz"
    return math.nan, "missing", ""


def status_is_ready(row: Dict[str, Any]) -> bool:
    return str(row.get("feature_source_status", "")) == "READY_TEMPORAL_SOURCE"


def status_is_excluded(row: Dict[str, Any]) -> bool:
    return str(row.get("feature_source_status", "")) == "NO_EVENT_WITHOUT_TEMPORAL_ARTIFACT"


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    bindings = read_csv(Path(args.feature_source_bindings))
    manifest_rows = read_csv(Path(args.complete_manifest))
    by_id, by_key, by_record = build_manifest_indexes(manifest_rows)
    by_row: List[Dict[str, Any]] = []
    ready_values: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    feature_counts: Dict[str, Counter] = defaultdict(Counter)
    missing_feature_counts: Counter = Counter()
    method_counts: Counter = Counter()
    temporal_cache: Dict[str, List[Dict[str, Any]]] = {}
    temporal_files_checked = set()
    file_read_failure_count = 0
    ready_checked = 0
    excluded_count = 0
    for i, bind in enumerate(bindings):
        label_row_id = str(bind.get("label_row_id") or f"row_{i:06d}")
        manifest = match_manifest(bind, by_id, by_key, by_record)
        base = {"label_row_id": label_row_id, "record_id": bind.get("record_id", manifest.get("record_id", "")), "suite": bind.get("suite", manifest.get("suite", "")), "event_status": bind.get("event_status", manifest.get("event_status", "")), "event_role": bind.get("event_role", manifest.get("event_role", "")), "feature_source_status": bind.get("feature_source_status", "")}
        if status_is_excluded(bind):
            excluded_count += 1
            exclusions.append({**base, "exclusion_reason": bind.get("exclusion_reason", "NO_EVENT_WITHOUT_TEMPORAL_ARTIFACT")})
            by_row.append({**base, "probe_status": "EXCLUDED", "missing_features": "", "feature_extraction_policy": "excluded_no_event_without_temporal_artifact"})
            continue
        if not status_is_ready(bind):
            failures.append({**base, "failure_reason": "NON_READY_FEATURE_SOURCE_STATUS", "detail": bind.get("feature_source_status", "")})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "non_ready_feature_source"})
            continue
        ready_checked += 1
        temporal_path = str(bind.get("temporal_path", ""))
        if not temporal_path:
            failures.append({**base, "failure_reason": "READY_ROW_MISSING_TEMPORAL_PATH", "detail": ""})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "missing_temporal_path"})
            continue
        try:
            if temporal_path not in temporal_cache:
                temporal_cache[temporal_path] = read_temporal(temporal_path)
                temporal_files_checked.add(temporal_path)
            rows = temporal_cache[temporal_path]
        except Exception as exc:
            file_read_failure_count += 1
            failures.append({**base, "failure_reason": "TEMPORAL_READ_FAILED", "detail": f"{type(exc).__name__}: {exc}"})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "temporal_read_failed"})
            continue
        if not rows:
            failures.append({**base, "failure_reason": "EMPTY_TEMPORAL_FILE", "detail": temporal_path})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "feature_extraction_policy": "empty_temporal_file"})
            continue
        idx, policy, extraction_step = choose_index(bind, manifest, rows, args)
        values, methods, fields = compute_features(rows, idx)
        missing = []
        for feat in SC5_V2_FEATURES:
            v = values.get(feat, math.nan)
            method = methods.get(feat, "missing") if math.isfinite(v) else "missing"
            feature_counts[feat][method] += 1
            method_counts[method] += 1
            if not math.isfinite(v):
                missing.append(feat)
                missing_feature_counts[feat] += 1
        if missing:
            failures.append({**base, "failure_reason": "MISSING_OR_NONFINITE_FEATURES", "detail": ";".join(missing), "temporal_path": temporal_path, "feature_extraction_policy": policy})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": ";".join(missing), "feature_extraction_policy": policy, "temporal_path": temporal_path, "extraction_index": idx, "extraction_step": extraction_step})
        else:
            val_row = {**base, "feature_extraction_policy": policy, "temporal_path": temporal_path, "temporal_source_sha256": sha256_file(Path(temporal_path)), "extraction_index": idx, "feature_extraction_step": extraction_step}
            for feat in SC5_V2_FEATURES:
                val_row[feat] = values[feat]
                val_row[f"{feat}__method"] = methods.get(feat, "")
                val_row[f"{feat}__field"] = fields.get(feat, "")
            ready_values.append(val_row)
            by_row.append({**base, "probe_status": "PASS", "missing_features": "", "feature_extraction_policy": policy, "temporal_path": temporal_path, "extraction_index": idx, "extraction_step": extraction_step})
    forbidden_feature_count = 0
    try:
        validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    except Exception:
        forbidden_feature_count = 1
    by_feature = []
    for feat in SC5_V2_FEATURES:
        c = feature_counts[feat]
        by_feature.append({"feature": feat, "pass_count": ready_checked - missing_feature_counts.get(feat, 0), "missing_count": missing_feature_counts.get(feat, 0), "methods": json.dumps(dict(c), sort_keys=True)})
    status = PASS
    reason = ""
    if len(bindings) != args.expected_label_rows:
        status = "HOLD_FEATURE_BINDING_ROW_COUNT_MISMATCH"
        reason = f"label_rows={len(bindings)} expected={args.expected_label_rows}"
    elif ready_checked != args.expected_ready_rows:
        status = "HOLD_READY_ROW_COUNT_MISMATCH"
        reason = f"ready={ready_checked} expected={args.expected_ready_rows}"
    elif excluded_count != args.expected_excluded_rows:
        status = "HOLD_EXCLUDED_ROW_COUNT_MISMATCH"
        reason = f"excluded={excluded_count} expected={args.expected_excluded_rows}"
    elif forbidden_feature_count:
        status = "HOLD_FORBIDDEN_FEATURE_COLUMNS"
        reason = "feature_columns contain forbidden hints"
    elif failures:
        status = "HOLD_FEATURE_EXTRACTION_GAPS"
        reason = f"failure_count={len(failures)}"
    write_csv(out / "detector_25d_feature_probe_v2_by_row.csv", by_row, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "probe_status", "missing_features", "feature_extraction_policy", "temporal_path", "extraction_index", "extraction_step"])
    value_fields = ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_extraction_policy", "temporal_path", "temporal_source_sha256", "extraction_index", "feature_extraction_step"] + list(SC5_V2_FEATURES)
    write_csv(out / "detector_25d_feature_probe_v2_ready_feature_values.csv", ready_values, value_fields)
    write_csv(out / "detector_25d_feature_probe_v2_by_feature.csv", by_feature, ["feature", "pass_count", "missing_count", "methods"])
    write_csv(out / "detector_25d_feature_probe_v2_failures.csv", failures, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "failure_reason", "detail", "temporal_path", "feature_extraction_policy"])
    write_csv(out / "detector_25d_feature_probe_v2_exclusions.csv", exclusions, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "exclusion_reason"])
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
        "ready_rows_passed": len(ready_values),
        "excluded_rows": excluded_count,
        "temporal_files_checked": len(temporal_files_checked),
        "file_read_failure_count": file_read_failure_count,
        "failure_count": len(failures),
        "failures_by_reason": dict(Counter(f.get("failure_reason", "") for f in failures)),
        "missing_feature_counts": dict(missing_feature_counts),
        "feature_derivation_counts": dict(method_counts),
        "feature_columns": list(SC5_V2_FEATURES),
        "forbidden_feature_count": forbidden_feature_count,
        "no_event_step_policy": args.no_event_step_policy,
        "interpretation": "CPU-only robust schema D2C probe. It accepts f_ precomputed features, raw-schema aliases, and deterministic derived clean-only features. Step/timestep may be used only for temporal alignment, never as model input features.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_dataset_build": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "detector_25d_feature_extraction_probe_v2_report.json", report)
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
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
