#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs

GATE = "D2C3_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBE_BOUNDARY_AND_PROXY_DERIVATIONS"
PASS = "PASS_CLEAN2000_DETECTOR_25D_FEATURE_EXTRACTION_PROBED_BOUNDARY_AND_PROXY_DERIVATIONS"
OUT_FILES = [
    "detector_25d_feature_extraction_probe_v3_report.json",
    "detector_25d_feature_probe_v3_by_row.csv",
    "detector_25d_feature_probe_v3_ready_feature_values.csv",
    "detector_25d_feature_probe_v3_by_feature.csv",
    "detector_25d_feature_probe_v3_failures.csv",
    "detector_25d_feature_probe_v3_exclusions.csv",
    "checksum_report.json",
]

STEP_ALIASES = ["step", "timestep", "frame", "frame_idx", "step_idx", "index"]
VECTOR_HINTS = ["action", "actions", "raw_action", "policy_action", "eef_pos", "eef_position", "ee_pos", "tcp_pos", "eef_vel", "ee_vel", "tcp_vel", "gripper_qpos"]
DIRECT_ALIASES: Dict[str, List[str]] = {
    "gripper_command": ["gripper_command", "command_gripper", "gripper_cmd", "action_gripper", "gripper_action", "a_gripper", "action_6", "action_7", "action[-1]", "action_6", "raw_action_6", "policy_action_6"],
    "gripper_qpos": ["gripper_qpos", "robot0_gripper_qpos", "obs_gripper_qpos", "gripper_width", "gripper_opening", "opening", "qpos_gripper", "robot0_gripper_qpos_0", "robot0_gripper_qpos_1"],
    "gripper_opening_proxy": ["gripper_opening_proxy", "opening_proxy", "gripper_width", "gripper_opening", "opening", "gripper_qpos", "robot0_gripper_qpos", "obs_gripper_qpos", "robot0_gripper_qpos_0", "robot0_gripper_qpos_1"],
    "eef_x": ["eef_x", "eef_pos_x", "robot0_eef_x", "robot0_eef_pos_x", "ee_x", "ee_pos_x", "tcp_x", "eef_pos_0", "ee_pos_0", "tcp_pos_0"],
    "eef_y": ["eef_y", "eef_pos_y", "robot0_eef_y", "robot0_eef_pos_y", "ee_y", "ee_pos_y", "tcp_y", "eef_pos_1", "ee_pos_1", "tcp_pos_1"],
    "eef_z": ["eef_z", "eef_pos_z", "robot0_eef_z", "robot0_eef_pos_z", "ee_z", "ee_pos_z", "tcp_z", "eef_pos_2", "ee_pos_2", "tcp_pos_2"],
    "eef_vx": ["eef_vx", "eef_vel_x", "robot0_eef_vx", "robot0_eef_vel_x", "ee_vx", "ee_vel_x", "tcp_vx", "eef_vel_0", "ee_vel_0", "tcp_vel_0"],
    "eef_vy": ["eef_vy", "eef_vel_y", "robot0_eef_vy", "robot0_eef_vel_y", "ee_vy", "ee_vel_y", "tcp_vy", "eef_vel_1", "ee_vel_1", "tcp_vel_1"],
    "eef_vz": ["eef_vz", "eef_vel_z", "robot0_eef_vz", "robot0_eef_vel_z", "ee_vz", "ee_vel_z", "tcp_vz", "eef_vel_2", "ee_vel_2", "tcp_vel_2"],
    "action_dx": ["action_dx", "delta_x", "act_dx", "action_0", "action[0]", "raw_action_0", "policy_action_0", "a0"],
    "action_dy": ["action_dy", "delta_y", "act_dy", "action_1", "action[1]", "raw_action_1", "policy_action_1", "a1"],
    "action_dz": ["action_dz", "delta_z", "act_dz", "action_2", "action[2]", "raw_action_2", "policy_action_2", "a2"],
    "action_gripper": ["action_gripper", "gripper_action", "a_gripper", "gripper_command", "command_gripper", "action_6", "action_7", "action[-1]", "raw_action_6", "policy_action_6"],
}
FORBIDDEN_EXPLANATION = "Step/timestep may be used for alignment only; model input feature_columns remain exactly SC5_V2_FEATURES."


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
        return [augment_row(dict(r)) for r in csv.DictReader(f)]


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
    return augment_row(out)


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
    return str(name).strip().lower().replace("/", "_").replace(".", "_").replace("-", "_").replace(" ", "_")


def parse_vector(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        vals = [finite_float(x) for x in value]
        return [v for v in vals if math.isfinite(v)]
    if value in (None, ""):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("array(") and text.endswith(")"):
        text = text[6:-1]
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple)):
            vals = [finite_float(x) for x in obj]
            return [v for v in vals if math.isfinite(v)]
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 2:
        vals = [finite_float(x) for x in nums]
        return [v for v in vals if math.isfinite(v)]
    return []


def augment_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key, value in list(row.items()):
        low = normalize_name(key)
        if not any(h in low for h in VECTOR_HINTS):
            continue
        vals = parse_vector(value)
        if not vals:
            continue
        for i, val in enumerate(vals):
            out.setdefault(f"{low}_{i}", val)
            out.setdefault(f"{low}[{i}]", val)
        if "action" in low:
            for i, val in enumerate(vals):
                out.setdefault(f"action_{i}", val)
                out.setdefault(f"action[{i}]", val)
        if "eef_pos" in low or "ee_pos" in low or "tcp_pos" in low:
            for name, idx in [("eef_x", 0), ("eef_y", 1), ("eef_z", 2)]:
                if idx < len(vals):
                    out.setdefault(name, vals[idx])
        if "eef_vel" in low or "ee_vel" in low or "tcp_vel" in low:
            for name, idx in [("eef_vx", 0), ("eef_vy", 1), ("eef_vz", 2)]:
                if idx < len(vals):
                    out.setdefault(name, vals[idx])
        if "gripper_qpos" in low and vals:
            out.setdefault("gripper_qpos", sum(vals) / len(vals))
            out.setdefault("gripper_opening_proxy", sum(vals) / len(vals))
    return out


def row_value(row: Dict[str, Any], names: Iterable[str]) -> Tuple[float, str]:
    by_norm = {normalize_name(k): k for k in row.keys()}
    for name in names:
        v = finite_float(row.get(name))
        if math.isfinite(v):
            return v, name
        key = by_norm.get(normalize_name(name))
        if key is not None:
            v = finite_float(row.get(key))
            if math.isfinite(v):
                return v, key
    return math.nan, ""


def feature_precomputed(row: Dict[str, Any], feature: str) -> Tuple[float, str]:
    return row_value(row, [f"f_{feature}", f"feature_{feature}", f"sc5_{feature}"])


def get_steps(rows: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    best = ("row_index", [float(i) for i in range(len(rows))], -1)
    for field in STEP_ALIASES:
        vals = [finite_float(r.get(field)) for r in rows]
        count = sum(math.isfinite(v) for v in vals)
        if count > best[2]:
            best = (field, vals, count)
    if best[2] >= max(1, int(0.5 * len(rows))):
        return [v if math.isfinite(v) else float(i) for i, v in enumerate(best[1])], best[0]
    return [float(i) for i in range(len(rows))], "row_index"


def nearest_index(steps: List[float], target: float) -> int:
    return min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(target))) if steps else 0


def manifest_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (str(row.get("record_id", "")), str(row.get("event_id", "")), str(row.get("segment_id", "")), str(row.get("segment_index", "")), str(row.get("event_status", "")))


def build_manifest_indexes(rows: List[Dict[str, Any]]):
    by_id, by_key, by_record = {}, {}, defaultdict(list)
    for i, row in enumerate(rows):
        new = dict(row)
        new["label_row_id"] = str(new.get("label_row_id") or new.get("__label_row_id") or f"row_{i:06d}")
        by_id[new["label_row_id"]] = new
        by_key[manifest_key(new)] = new
        by_record[str(new.get("record_id", ""))].append(new)
    return by_id, by_key, by_record


def match_manifest(binding, by_id, by_key, by_record):
    lid = str(binding.get("label_row_id", ""))
    if lid and lid in by_id:
        return by_id[lid]
    key = manifest_key(binding)
    if key in by_key:
        return by_key[key]
    candidates = by_record.get(str(binding.get("record_id", "")), [])
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if str(c.get("event_id", "")) == str(binding.get("event_id", "")) and str(c.get("segment_id", "")) == str(binding.get("segment_id", "")):
            return c
    return {}


def choose_index(binding, manifest, rows, args):
    steps, field = get_steps(rows)
    anchor = finite_float(binding.get("teacher_anchor_step"))
    if not math.isfinite(anchor):
        anchor = finite_float(manifest.get("teacher_anchor_step"))
    if math.isfinite(anchor) and anchor >= 0:
        return nearest_index(steps, anchor), f"nearest_teacher_anchor_step:{field}", anchor
    if str(binding.get("event_status") or manifest.get("event_status")) == "NO_EVENT":
        idx = len(rows) // 2 if args.no_event_step_policy == "middle" else (len(rows) - 1 if args.no_event_step_policy == "last" else 0)
        idx = max(0, min(idx, len(rows) - 1))
        return idx, f"no_event_{args.no_event_step_policy}:{field}", steps[idx] if steps else float(idx)
    idx = max(0, len(rows) - 1)
    return idx, f"fallback_last:{field}", steps[idx] if steps else float(idx)


def series(rows: List[Dict[str, Any]], names: Iterable[str]) -> Tuple[List[float], str]:
    best_vals, best_field, best_count = [math.nan for _ in rows], "", -1
    for name in names:
        vals = [row_value(r, [name])[0] for r in rows]
        count = sum(math.isfinite(v) for v in vals)
        if count > best_count:
            best_vals, best_field, best_count = vals, name, count
    return best_vals, best_field if best_count > 0 else ""


def get_base_series(rows, feature):
    vals = [feature_precomputed(r, feature)[0] for r in rows]
    if sum(math.isfinite(v) for v in vals) >= max(1, int(0.5 * len(rows))):
        return vals, f"f_{feature}", "pre_computed_f_prefix"
    vals, field = series(rows, [feature] + DIRECT_ALIASES.get(feature, []))
    if any(math.isfinite(v) for v in vals):
        return vals, field, "direct_raw_column"
    return vals, "", "missing"


def percentile(values: List[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def variance_last(vals: List[float], idx: int, n: int = 5) -> float:
    xs = [v for v in vals[max(0, idx - n + 1): idx + 1] if math.isfinite(v)]
    if not xs:
        return math.nan
    return 0.0 if len(xs) == 1 else float(statistics.pvariance(xs))


def boundary_delta(vals: List[float], idx: int, lag: int) -> Tuple[float, str]:
    if idx < len(vals) and idx - lag >= 0 and math.isfinite(vals[idx]) and math.isfinite(vals[idx - lag]):
        return vals[idx] - vals[idx - lag], "derived_from_raw"
    if idx < len(vals) and math.isfinite(vals[idx]):
        return 0.0, f"derived_boundary_zero_delta_lag{lag}"
    return math.nan, ""


def position_delta(rows, axis, idx):
    vals, field, _ = get_base_series(rows, f"eef_{axis}")
    val, method = boundary_delta(vals, idx, 1)
    return val, f"{method}:{field}" if method else ""


def close_mask_from_signal(rows):
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


def last_close_onset(mask, idx):
    last, prev = None, False
    for i, v in enumerate(mask[: idx + 1]):
        if v and not prev:
            last = i
        prev = v
    return last


def derive_command_from_opening(rows, idx, event_status):
    vals, field, _ = get_base_series(rows, "gripper_opening_proxy")
    if idx < len(vals) and idx > 0 and math.isfinite(vals[idx]) and math.isfinite(vals[idx - 1]):
        d = vals[idx] - vals[idx - 1]
        if abs(d) > 1e-9:
            return (-1.0 if d < 0 else 1.0), f"derived_gripper_direction_from_{field}"
        return 0.0, f"derived_static_gripper_from_{field}"
    if event_status == "NO_EVENT":
        return 0.0, "derived_no_event_static_gripper_default"
    return math.nan, ""


def minimal_speed(rows, idx):
    row = rows[idx]
    v, field = feature_precomputed(row, "eef_speed")
    if math.isfinite(v):
        return v, "pre_computed_f_prefix", field
    comps = []
    for feat in ["eef_vx", "eef_vy", "eef_vz"]:
        vv, ff = feature_precomputed(row, feat)
        if not math.isfinite(vv):
            vv, ff = row_value(row, [feat] + DIRECT_ALIASES.get(feat, []))
        comps.append(vv)
    if all(math.isfinite(x) for x in comps):
        return math.sqrt(sum(x * x for x in comps)), "derived_from_raw", "eef_vx,eef_vy,eef_vz"
    return math.nan, "missing", ""


def compute_features(rows, idx, event_status):
    values, methods, fields = {}, {}, {}
    row = rows[idx]
    for feat in SC5_V2_FEATURES:
        v, field = feature_precomputed(row, feat)
        if math.isfinite(v):
            values[feat], methods[feat], fields[feat] = v, "pre_computed_f_prefix", field
            continue
        v, field = row_value(row, [feat] + DIRECT_ALIASES.get(feat, []))
        if math.isfinite(v):
            values[feat], methods[feat], fields[feat] = v, "direct_raw_column", field
    for axis in ["x", "y", "z"]:
        feat = f"eef_v{axis}"
        if not math.isfinite(values.get(feat, math.nan)):
            v, field = position_delta(rows, axis, idx)
            if math.isfinite(v):
                values[feat], methods[feat], fields[feat] = v, "derived_from_raw", field
    if not math.isfinite(values.get("eef_speed", math.nan)):
        comps = [values.get("eef_vx", math.nan), values.get("eef_vy", math.nan), values.get("eef_vz", math.nan)]
        if all(math.isfinite(v) for v in comps):
            values["eef_speed"], methods["eef_speed"], fields["eef_speed"] = math.sqrt(sum(v * v for v in comps)), "derived_from_raw", "eef_vx,eef_vy,eef_vz"
    for feat, axis in [("action_dx", "x"), ("action_dy", "y"), ("action_dz", "z")]:
        if not math.isfinite(values.get(feat, math.nan)):
            v, field = position_delta(rows, axis, idx)
            if math.isfinite(v):
                values[feat], methods[feat], fields[feat] = v, "derived_action_proxy_from_eef_delta", field
    for feat in ["gripper_command", "action_gripper"]:
        if not math.isfinite(values.get(feat, math.nan)):
            v, field = derive_command_from_opening(rows, idx, event_status)
            if math.isfinite(v):
                values[feat], methods[feat], fields[feat] = v, "derived_gripper_proxy_from_opening_delta", field
    mask, mask_src = close_mask_from_signal(rows)
    close_streak = 0
    j = idx
    while j >= 0 and j < len(mask) and mask[j]:
        close_streak += 1
        j -= 1
    open_streak = 0
    j = idx
    while j >= 0 and j < len(mask) and not mask[j]:
        open_streak += 1
        j -= 1
    window = mask[max(0, idx - 4): idx + 1]
    flips = sum(1 for a, b in zip(window, window[1:]) if a != b)
    prev = mask[idx - 1] if idx > 0 and idx - 1 < len(mask) else False
    onset_now = bool(idx < len(mask) and mask[idx] and not prev)
    onset_idx = last_close_onset(mask, idx) if mask else None
    hist_vals = {
        "recent_close_streak": close_streak,
        "recent_open_streak": open_streak,
        "recent_gripper_flip_count": flips,
        "close_onset": int(onset_now),
        "time_since_close": idx - onset_idx if onset_idx is not None else 0,
    }
    for feat, val in hist_vals.items():
        if not math.isfinite(values.get(feat, math.nan)):
            values[feat], methods[feat], fields[feat] = float(val), f"derived_from_close_mask:{mask_src}" if mask_src else "derived_default_no_close_signal", mask_src
    qpos_vals, qpos_field, _ = get_base_series(rows, "gripper_qpos")
    opening_vals, opening_field, _ = get_base_series(rows, "gripper_opening_proxy")
    eefz_vals, eefz_field, _ = get_base_series(rows, "eef_z")
    speed_vals = [minimal_speed(rows, i)[0] for i in range(len(rows))]
    for feat, vals, field, lag in [("qpos_delta_1", qpos_vals, qpos_field, 1), ("qpos_delta_3", qpos_vals, qpos_field, 3), ("opening_proxy_delta_3", opening_vals, opening_field, 3)]:
        if not math.isfinite(values.get(feat, math.nan)):
            v, method = boundary_delta(vals, idx, lag)
            if math.isfinite(v):
                values[feat], methods[feat], fields[feat] = v, method, field
    if not math.isfinite(values.get("opening_proxy_variance_5", math.nan)):
        v = variance_last(opening_vals, idx, 5)
        if math.isfinite(v):
            values["opening_proxy_variance_5"], methods["opening_proxy_variance_5"], fields["opening_proxy_variance_5"] = v, "derived_from_raw", opening_field
    if not math.isfinite(values.get("eef_speed_variance_5", math.nan)):
        v = variance_last(speed_vals, idx, 5)
        if math.isfinite(v):
            values["eef_speed_variance_5"], methods["eef_speed_variance_5"], fields["eef_speed_variance_5"] = v, "derived_from_raw", "eef_speed_series"
    if not math.isfinite(values.get("eef_z_delta_since_close", math.nan)):
        close_f, close_field = row_value(row, ["f_close_onset", "close_onset"])
        if math.isfinite(close_f) and close_f >= 0.5 and idx < len(eefz_vals) and math.isfinite(eefz_vals[idx]):
            values["eef_z_delta_since_close"], methods["eef_z_delta_since_close"], fields["eef_z_delta_since_close"] = 0.0, "derived_zero_at_current_close_onset", close_field
        elif onset_idx is not None and idx < len(eefz_vals) and onset_idx < len(eefz_vals) and math.isfinite(eefz_vals[idx]) and math.isfinite(eefz_vals[onset_idx]):
            values["eef_z_delta_since_close"] = eefz_vals[idx] - eefz_vals[onset_idx]
            methods["eef_z_delta_since_close"], fields["eef_z_delta_since_close"] = f"derived_from_close_onset:{mask_src}", eefz_field
        elif event_status == "NO_EVENT" and idx < len(eefz_vals) and math.isfinite(eefz_vals[idx]):
            values["eef_z_delta_since_close"], methods["eef_z_delta_since_close"], fields["eef_z_delta_since_close"] = 0.0, "derived_no_event_no_close_zero", eefz_field
    return values, methods, fields


def is_ready(row):
    return str(row.get("feature_source_status", "")) == "READY_TEMPORAL_SOURCE"


def is_excluded(row):
    return str(row.get("feature_source_status", "")) == "NO_EVENT_WITHOUT_TEMPORAL_ARTIFACT"


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    bindings = read_csv(Path(args.feature_source_bindings))
    manifest = read_csv(Path(args.complete_manifest))
    by_id, by_key, by_record = build_manifest_indexes(manifest)
    cache: Dict[str, List[Dict[str, Any]]] = {}
    temporal_files = set()
    by_row, values_rows, failures, exclusions = [], [], [], []
    by_feature: Dict[str, Counter] = defaultdict(Counter)
    missing_counts, method_counts = Counter(), Counter()
    ready_checked = excluded_count = read_failures = 0
    for i, bind in enumerate(bindings):
        label_id = str(bind.get("label_row_id") or f"row_{i:06d}")
        man = match_manifest(bind, by_id, by_key, by_record)
        base = {"label_row_id": label_id, "record_id": bind.get("record_id", man.get("record_id", "")), "suite": bind.get("suite", man.get("suite", "")), "event_status": bind.get("event_status", man.get("event_status", "")), "event_role": bind.get("event_role", man.get("event_role", "")), "feature_source_status": bind.get("feature_source_status", "")}
        if is_excluded(bind):
            excluded_count += 1
            exclusions.append({**base, "exclusion_reason": bind.get("exclusion_reason", "NO_EVENT_WITHOUT_TEMPORAL_ARTIFACT")})
            by_row.append({**base, "probe_status": "EXCLUDED", "missing_features": "", "temporal_path": "", "feature_extraction_policy": "excluded_no_event_without_temporal_artifact"})
            continue
        if not is_ready(bind):
            failures.append({**base, "failure_reason": "NON_READY_FEATURE_SOURCE_STATUS", "detail": bind.get("feature_source_status", "")})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "temporal_path": "", "feature_extraction_policy": "non_ready_feature_source"})
            continue
        ready_checked += 1
        path = str(bind.get("temporal_path", ""))
        if not path:
            failures.append({**base, "failure_reason": "READY_ROW_MISSING_TEMPORAL_PATH", "detail": ""})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "temporal_path": "", "feature_extraction_policy": "missing_temporal_path"})
            continue
        try:
            if path not in cache:
                cache[path] = read_temporal(path)
                temporal_files.add(path)
            rows = cache[path]
        except Exception as exc:
            read_failures += 1
            failures.append({**base, "failure_reason": "TEMPORAL_READ_FAILED", "detail": f"{type(exc).__name__}: {exc}", "temporal_path": path})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "temporal_path": path, "feature_extraction_policy": "temporal_read_failed"})
            continue
        if not rows:
            failures.append({**base, "failure_reason": "EMPTY_TEMPORAL_FILE", "detail": path, "temporal_path": path})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": "ALL", "temporal_path": path, "feature_extraction_policy": "empty_temporal_file"})
            continue
        idx, policy, step = choose_index(bind, man, rows, args)
        vals, methods, fields = compute_features(rows, idx, str(base["event_status"]))
        missing = []
        for feat in SC5_V2_FEATURES:
            v = vals.get(feat, math.nan)
            method = methods.get(feat, "missing") if math.isfinite(v) else "missing"
            by_feature[feat][method] += 1
            method_counts[method] += 1
            if not math.isfinite(v):
                missing.append(feat)
                missing_counts[feat] += 1
        if missing:
            detail = ";".join(missing)
            failures.append({**base, "failure_reason": "MISSING_OR_NONFINITE_FEATURES", "detail": detail, "temporal_path": path, "feature_extraction_policy": policy})
            by_row.append({**base, "probe_status": "FAIL", "missing_features": detail, "temporal_path": path, "feature_extraction_policy": policy, "extraction_index": idx, "extraction_step": step})
        else:
            vrow = {**base, "feature_extraction_policy": policy, "temporal_path": path, "temporal_source_sha256": sha256_file(Path(path)), "extraction_index": idx, "feature_extraction_step": step}
            for feat in SC5_V2_FEATURES:
                vrow[feat] = vals[feat]
                vrow[f"{feat}__method"] = methods.get(feat, "")
                vrow[f"{feat}__field"] = fields.get(feat, "")
            values_rows.append(vrow)
            by_row.append({**base, "probe_status": "PASS", "missing_features": "", "temporal_path": path, "feature_extraction_policy": policy, "extraction_index": idx, "extraction_step": step})
    forbidden_count = 0
    try:
        validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    except Exception:
        forbidden_count = 1
    feature_rows = [{"feature": feat, "pass_count": ready_checked - missing_counts.get(feat, 0), "missing_count": missing_counts.get(feat, 0), "methods": json.dumps(dict(by_feature[feat]), sort_keys=True)} for feat in SC5_V2_FEATURES]
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
    write_csv(out / "detector_25d_feature_probe_v3_by_row.csv", by_row, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "probe_status", "missing_features", "feature_extraction_policy", "temporal_path", "extraction_index", "extraction_step"])
    value_fields = ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_extraction_policy", "temporal_path", "temporal_source_sha256", "extraction_index", "feature_extraction_step"] + list(SC5_V2_FEATURES) + [f"{f}__method" for f in SC5_V2_FEATURES]
    write_csv(out / "detector_25d_feature_probe_v3_ready_feature_values.csv", values_rows, value_fields)
    write_csv(out / "detector_25d_feature_probe_v3_by_feature.csv", feature_rows, ["feature", "pass_count", "missing_count", "methods"])
    write_csv(out / "detector_25d_feature_probe_v3_failures.csv", failures, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "failure_reason", "detail", "temporal_path", "feature_extraction_policy"])
    write_csv(out / "detector_25d_feature_probe_v3_exclusions.csv", exclusions, ["label_row_id", "record_id", "suite", "event_status", "event_role", "feature_source_status", "exclusion_reason"])
    report = {"gate": GATE, "status": status, "reason": reason, "feature_source_bindings": args.feature_source_bindings, "feature_source_bindings_sha256": sha256_file(Path(args.feature_source_bindings)), "complete_manifest": args.complete_manifest, "complete_manifest_sha256": sha256_file(Path(args.complete_manifest)), "expected_label_rows": args.expected_label_rows, "expected_ready_rows": args.expected_ready_rows, "expected_excluded_rows": args.expected_excluded_rows, "label_row_count": len(bindings), "ready_rows_checked": ready_checked, "ready_rows_passed": len(values_rows), "excluded_rows": excluded_count, "temporal_files_checked": len(temporal_files), "file_read_failure_count": read_failures, "failure_count": len(failures), "failures_by_reason": dict(Counter(f.get("failure_reason", "") for f in failures)), "missing_feature_counts": dict(missing_counts), "feature_derivation_counts": dict(method_counts), "feature_columns": list(SC5_V2_FEATURES), "forbidden_feature_count": forbidden_count, "no_event_step_policy": args.no_event_step_policy, "derivation_policy_notes": {"action_proxy": "action_dx/dy/dz may be derived as clean eef position deltas only when raw action columns are absent; method is explicitly recorded as derived_action_proxy_from_eef_delta", "gripper_proxy": "gripper_command/action_gripper may be derived from clean opening/qpos delta when raw command/action columns are absent", "boundary_deltas": "qpos/opening lag features use deterministic zero boundary deltas only when the selected extraction index has insufficient history", "no_event_z_delta": "NO_EVENT rows without a close onset use eef_z_delta_since_close=0 with explicit derived_no_event_no_close_zero provenance"}, "forbidden_feature_policy": FORBIDDEN_EXPLANATION, "interpretation": "CPU-only D2C3 probe. It extends D2C2 with vector-string parsing, action/eef proxy derivations, and boundary-safe no-event lag features; output remains a probe, not a detector dataset build.", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_dataset_build": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"}, "git_commit": args.git_commit, "files_changed": args.files_changed, "tests": args.tests}
    write_json(out / "detector_25d_feature_extraction_probe_v3_report.json", report)
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
