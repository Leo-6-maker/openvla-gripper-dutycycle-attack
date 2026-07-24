#!/usr/bin/env python3
"""C2e0D object-only temporal 25D completeness audit.

Offline CPU-only audit. No detector training and no simulator/runtime calls.

C2e0C resolved all endpoints but left LIBERO Object with partial temporal 25D
coverage. This script isolates object temporal artifacts and checks whether the
missing fields can be repaired by canonical causal reconstruction from the base
13 proprio/action stream. It can optionally use the C2e0C endpoint audit to
report row-level materializability for W=8/16/32.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES, validate_no_forbidden_inputs  # type: ignore
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
    def validate_no_forbidden_inputs(names: List[str]) -> None:
        bad = [n for n in names if any(x in n.lower() for x in ["object_pose", "target_pose", "oracle", "attack_outcome", "teacher_window", "teacher_anchor"])]
        if bad:
            raise ValueError(f"forbidden model-input feature names present: {bad}")

GATE = "D4C2E0D_OBJECT_TEMPORAL_25D_COMPLETENESS_AUDIT"
PASS = "PASS_D4C2E0D_OBJECT_TEMPORAL_25D_COMPLETENESS_READY_FOR_C2E1"
HOLD = "HOLD_D4C2E0D_OBJECT_TEMPORAL_25D_COMPLETENESS_AUDIT"
BASE13 = SC5_V2_FEATURES[:13]
TEMPORAL_PATH_COLUMNS = ["temporal_path", "source_temporal_path", "stream_path", "artifact_path", "trajectory_path"]
OUT_FILES = [
    "d4c2e0d_object_temporal_completeness_report.json",
    "object_temporal_completeness_by_artifact.csv",
    "object_missing_feature_histogram.csv",
    "object_materializable_rows.csv",
    "object_exclusion_manifest.csv",
    "object_temporal_derivation_schema.json",
    "object_temporal_violations.csv",
    "checksum_report.json",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_csv(p: Path) -> List[Dict[str, str]]:
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_csv(p: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def fnum(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default

def inum(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except Exception:
        return None

def windows(raw: str) -> List[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})

def suite(r: Dict[str, str]) -> str:
    return str(r.get("suite", "unknown") or "unknown")

def split(r: Dict[str, str]) -> str:
    return str(r.get("split", r.get("dataset_split", "unknown")) or "unknown")

def group_key(r: Dict[str, str]) -> str:
    for k in ["group_key", "episode_key", "record_id", "parent_id", "run_id", "episode_id"]:
        if r.get(k): return str(r[k])
    return ""

def label(r: Dict[str, str]) -> int:
    for k in ["runtime_objective_label", "label", "y", "target"]:
        v = inum(r.get(k))
        if v in (0, 1): return int(v)
    role = str(r.get("event_role", "")); status = str(r.get("teacher_label_status", ""))
    if role == "primary_attackable" or status in {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"}: return 1
    if role or status: return 0
    return -1

def resolve_path(raw: str, dataset: Path, roots: Sequence[Path]) -> Optional[Path]:
    if not raw: return None
    p = Path(raw)
    cands = [p] if p.is_absolute() else [(dataset.parent / p).resolve(), (Path.cwd() / p).resolve()] + [(r / p).resolve() for r in roots]
    for c in cands:
        if c.exists(): return c
    return cands[0] if cands else None

def alias_candidates(name: str) -> List[str]:
    extras = {
        "gripper_opening_proxy": ["opening_proxy", "f_opening_proxy", "gripper_width", "f_gripper_width"],
        "gripper_command": ["raw_gripper", "f_raw_gripper", "gripper_action", "f_gripper_action"],
        "eef_x": ["eef_pos_x", "f_eef_pos_x"], "eef_y": ["eef_pos_y", "f_eef_pos_y"], "eef_z": ["eef_pos_z", "f_eef_pos_z"],
        "eef_vx": ["eef_vel_x", "f_eef_vel_x"], "eef_vy": ["eef_vel_y", "f_eef_vel_y"], "eef_vz": ["eef_vel_z", "f_eef_vel_z"],
    }
    vals = [name, f"f_{name}", f"feature_{name}", f"obs_{name}"] + extras.get(name, [])
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def alias_map(header: Sequence[str]) -> Dict[str, str]:
    hs, ret = set(header), {}
    for f in SC5_V2_FEATURES:
        present = [c for c in alias_candidates(f) if c in hs]
        if present:
            ret[f] = f if f in present else (f"f_{f}" if f"f_{f}" in present else present[0])
    return ret

def matrix_from_alias(rows: List[Dict[str, str]], amap: Dict[str, str]) -> np.ndarray:
    arr = np.full((len(rows), len(SC5_V2_FEATURES)), np.nan, dtype=np.float32)
    for j, f in enumerate(SC5_V2_FEATURES):
        col = amap.get(f, "")
        if col:
            for i, r in enumerate(rows): arr[i, j] = fnum(r.get(col))
    return arr

def recompute_causal_25d(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    out = arr.copy(); idx = {f: i for i, f in enumerate(SC5_V2_FEATURES)}; src: Dict[str, str] = {}
    # Require base13 for full reconstruction.
    if out.shape[0] == 0 or not all(np.isfinite(out[:, idx[f]]).all() for f in BASE13):
        return out, src
    cmd = out[:, idx["gripper_command"]]; qpos = out[:, idx["gripper_qpos"]]; op = out[:, idx["gripper_opening_proxy"]]
    z = out[:, idx["eef_z"]]; vx = out[:, idx["eef_vx"]]; vy = out[:, idx["eef_vy"]]; vz = out[:, idx["eef_vz"]]
    close_streak = 0; open_streak = 0; flip_count = 0; prev_close: Optional[bool] = None
    last_close = -1; onset_seen = False; speeds: List[float] = []
    for i in range(out.shape[0]):
        raw_close = bool(cmd[i] <= 0.5)
        if raw_close:
            close_streak += 1; open_streak = 0
        else:
            open_streak += 1; close_streak = 0
        if prev_close is not None and prev_close != raw_close: flip_count += 1
        prev_close = raw_close
        if raw_close and last_close < 0: last_close = i
        close_onset = 1 if (raw_close and close_streak == 1 and not onset_seen) else 0
        if close_onset:
            onset_seen = True; last_close = i
        speed = float(math.sqrt(float(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2))); speeds.append(speed)
        out[i, idx["recent_close_streak"]] = close_streak
        out[i, idx["recent_open_streak"]] = open_streak
        out[i, idx["recent_gripper_flip_count"]] = flip_count
        out[i, idx["close_onset"]] = close_onset
        out[i, idx["time_since_close"]] = i - last_close if last_close >= 0 else -1
        out[i, idx["eef_speed"]] = speed
        out[i, idx["eef_z_delta_since_close"]] = float(z[i] - z[last_close]) if last_close >= 0 else 0.0
        out[i, idx["qpos_delta_1"]] = float(qpos[i] - qpos[i-1]) if i >= 1 else 0.0
        out[i, idx["qpos_delta_3"]] = float(qpos[i] - qpos[i-3]) if i >= 3 else 0.0
        out[i, idx["opening_proxy_delta_3"]] = float(op[i] - op[i-3]) if i >= 3 else 0.0
        out[i, idx["opening_proxy_variance_5"]] = float(np.var(op[max(0, i-4):i+1])) if i >= 4 else 0.0
        out[i, idx["eef_speed_variance_5"]] = float(np.var(speeds[max(0, i-4):i+1])) if i >= 4 else 0.0
    for f in SC5_V2_FEATURES[13:]: src[f] = "derived:canonical_causal_from_base13"
    return out, src

def complete(arr: np.ndarray) -> Tuple[bool, List[str]]:
    if arr.shape[0] == 0: return False, list(SC5_V2_FEATURES)
    missing = [f for j, f in enumerate(SC5_V2_FEATURES) if not np.isfinite(arr[:, j]).all()]
    return len(missing) == 0, missing

def load_endpoint_map(path: Path) -> Dict[int, Dict[str, str]]:
    if not path or not path.exists(): return {}
    out: Dict[int, Dict[str, str]] = {}
    for r in read_csv(path):
        v = inum(r.get("row_index"))
        if v is not None: out[v] = r
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True); ap.add_argument("--context-schema", default="")
    ap.add_argument("--c2e0c-endpoint-audit", default="")
    ap.add_argument("--expected-rows", type=int, default=3717); ap.add_argument("--window-lengths", default="8,16,32")
    ap.add_argument("--temporal-root", action="append", default=[]); ap.add_argument("--min-object-materializable-rate", type=float, default=0.95)
    ap.add_argument("--output-root", required=True); ap.add_argument("--git-commit", default="")
    args = ap.parse_args(); started = time.time()
    dataset = Path(args.context_dataset).expanduser().resolve(); out = Path(args.output_root).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    roots = [Path(p).expanduser().resolve() for p in args.temporal_root]; ws = windows(args.window_lengths)
    rows = read_csv(dataset); header = list(rows[0].keys()) if rows else []; temporal_col = next((c for c in TEMPORAL_PATH_COLUMNS if c in header), "")
    obj_rows = [(i, r) for i, r in enumerate(rows) if suite(r) == "libero_object"]
    endpoint_map = load_endpoint_map(Path(args.c2e0c_endpoint_audit).expanduser()) if args.c2e0c_endpoint_audit else {}
    violations: List[Dict[str, Any]] = []
    if args.expected_rows and len(rows) != args.expected_rows: violations.append({"code": "UNEXPECTED_ROW_COUNT", "severity": "hard", "actual": len(rows), "required": args.expected_rows})
    if not temporal_col: violations.append({"code": "TEMPORAL_PATH_COLUMN_MISSING", "severity": "hard"})
    try: validate_no_forbidden_inputs([f"lag00_{f}" for f in SC5_V2_FEATURES])
    except Exception as e: violations.append({"code": "FORBIDDEN_TEMPORAL_FEATURE", "severity": "hard", "message": str(e)})

    raw_paths = sorted({str(r.get(temporal_col, "")) for _, r in obj_rows if temporal_col and r.get(temporal_col)})
    artifact_info: Dict[str, Dict[str, Any]] = {}; artifact_rows: List[Dict[str, Any]] = []; miss_hist = Counter()
    for raw in raw_paths:
        p = resolve_path(raw, dataset, roots); info = {"raw_temporal_path": raw, "resolved_path": str(p) if p else "", "exists": bool(p and p.exists())}
        before_missing: List[str] = list(SC5_V2_FEATURES); after_missing: List[str] = list(SC5_V2_FEATURES); n = 0; src = {}
        if p and p.exists():
            try:
                trs = read_csv(p); n = len(trs); arr0 = matrix_from_alias(trs, alias_map(list(trs[0].keys()) if trs else [])); ok0, before_missing = complete(arr0)
                arr1, src = recompute_causal_25d(arr0); ok1, after_missing = complete(arr1)
                info.update({"readable": True, "row_count": n, "complete_before_recompute": ok0, "missing_before": ";".join(before_missing), "complete_after_recompute": ok1, "missing_after": ";".join(after_missing), "derived_fields": ";".join(sorted(src))})
            except Exception as e:
                info.update({"readable": False, "row_count": 0, "complete_before_recompute": False, "missing_before": ";".join(before_missing), "complete_after_recompute": False, "missing_after": ";".join(after_missing), "read_error": str(e)})
        else:
            info.update({"readable": False, "row_count": 0, "complete_before_recompute": False, "missing_before": ";".join(before_missing), "complete_after_recompute": False, "missing_after": ";".join(after_missing)})
        for m in after_missing: miss_hist[m] += 1
        artifact_info[raw] = info; artifact_rows.append(info)

    mat_rows: List[Dict[str, Any]] = []; excl_rows: List[Dict[str, Any]] = []
    mat_counts = {w: 0 for w in ws}; label_counts = Counter(); split_counts = Counter()
    for i, r in obj_rows:
        raw = str(r.get(temporal_col, "")) if temporal_col else ""; art = artifact_info.get(raw, {})
        ep = inum(endpoint_map.get(i, {}).get("endpoint_index")) if endpoint_map else None
        complete_after = bool(art.get("complete_after_recompute")); row_count = inum(art.get("row_count")) or 0
        row = {"row_index": i, "split": split(r), "label": label(r), "group_key": group_key(r), "temporal_path": raw, "endpoint_index": "" if ep is None else ep, "artifact_complete_after_recompute": complete_after, "artifact_row_count": row_count}
        ok_any = True
        reasons = []
        if not complete_after: ok_any = False; reasons.append("artifact_incomplete_after_recompute")
        if ep is None or ep < 0 or ep >= row_count: ok_any = False; reasons.append("endpoint_missing_or_oob")
        for w in ws:
            ok = bool(ok_any and ep is not None and ep + 1 >= w)
            row[f"materializable_w{w}"] = ok
            if ok: mat_counts[w] += 1
        if ok_any:
            label_counts[str(label(r))] += 1; split_counts[split(r)] += 1
            mat_rows.append(row)
        else:
            row["exclusion_reason"] = ";".join(reasons); excl_rows.append(row)
    min_rate = min((mat_counts[w] / max(1, len(obj_rows)) for w in ws), default=0.0)
    if min_rate < args.min_object_materializable_rate:
        violations.append({"code": "LOW_OBJECT_MATERIALIZABLE_RATE", "severity": "hard", "actual": min_rate, "required": args.min_object_materializable_rate})
    status = PASS if not [v for v in violations if v.get("severity") == "hard"] else HOLD
    report = {"gate": GATE, "status": status, "reason": "hard_violation_count=0" if status == PASS else f"hard_violation_count={len([v for v in violations if v.get('severity') == 'hard'])}", "created_at_unix": time.time(), "runtime_seconds": time.time()-started, "git_commit": args.git_commit, "inputs": {"context_dataset": str(dataset), "context_dataset_sha256": sha256_file(dataset), "context_schema": str(Path(args.context_schema).expanduser().resolve()) if args.context_schema else "", "c2e0c_endpoint_audit": args.c2e0c_endpoint_audit, "temporal_path_column": temporal_col, "window_lengths": ws}, "object_row_count": len(obj_rows), "object_unique_temporal_path_count": len(raw_paths), "object_materializable_counts_by_window": {str(k): v for k, v in mat_counts.items()}, "object_min_materializable_rate": min_rate, "materializable_label_counts": dict(label_counts), "materializable_split_counts": dict(split_counts), "missing_after_recompute_histogram": dict(miss_hist), "hard_violation_count": len([v for v in violations if v.get("severity") == "hard"]), "violations_by_code": dict(Counter(v.get("code") for v in violations)), "recommendation": "proceed_to_C2E1_or_global_C2e1_readiness" if status == PASS else "hold_repair_object_temporal_features_or_explicitly_exclude_nonmaterializable_rows", "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED", "D5C": "NOT_RUN", "D6C_v3": "NOT_RUN"}}
    schema = {"feature_family": "C2E_OBJECT_TEMPORAL_CANONICAL_25D_RECONSTRUCTION_V1", "base13": BASE13, "derived_from_base13": SC5_V2_FEATURES[13:], "causal_rule": "each derived feature uses current and previous rows only"}
    write_json(out / "d4c2e0d_object_temporal_completeness_report.json", report); write_json(out / "object_temporal_derivation_schema.json", schema)
    write_csv(out / "object_temporal_completeness_by_artifact.csv", artifact_rows, ["raw_temporal_path", "resolved_path", "exists", "readable", "row_count", "complete_before_recompute", "missing_before", "complete_after_recompute", "missing_after", "derived_fields", "read_error"])
    write_csv(out / "object_missing_feature_histogram.csv", [{"feature": k, "artifact_count": v} for k, v in sorted(miss_hist.items())], ["feature", "artifact_count"])
    fields = ["row_index", "split", "label", "group_key", "temporal_path", "endpoint_index", "artifact_complete_after_recompute", "artifact_row_count"] + [f"materializable_w{w}" for w in ws]
    write_csv(out / "object_materializable_rows.csv", mat_rows, fields); write_csv(out / "object_exclusion_manifest.csv", excl_rows, fields + ["exclusion_reason"])
    write_csv(out / "object_temporal_violations.csv", violations, ["code", "severity", "actual", "required", "message"])
    checks = []
    for name in OUT_FILES:
        p = out / name
        if p.exists(): checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checks: f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps({"status": status, "output_root": str(out), "object_row_count": len(obj_rows), "object_min_materializable_rate": min_rate, "hard_violation_count": report["hard_violation_count"], "recommendation": report["recommendation"]}, indent=2, sort_keys=True))
    return 0 if status == PASS else 2

if __name__ == "__main__":
    raise SystemExit(main())
