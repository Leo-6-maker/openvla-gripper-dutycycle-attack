#!/usr/bin/env python3
"""C2e0C: audit derived 25D temporal features and endpoint matching.

This is an offline CPU-only readiness audit. It does not train a model and does
not run any simulator. It checks whether C2e temporal windows can be built after
causally deriving three missing SC5 fields and after using conservative endpoint
match profiles.
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
        bad = [n for n in names if any(x in n.lower() for x in ["object_pose", "target_pose", "oracle", "teacher_window", "teacher_anchor"])]
        if bad:
            raise ValueError(f"forbidden model-input feature names present: {bad}")

GATE = "D4C2E0C_CLEAN2000_TEMPORAL_DERIVED_ENDPOINT_RESOLVER_AUDIT"
PASS = "PASS_D4C2E0C_TEMPORAL_DERIVED_ENDPOINT_RESOLVER_READY_FOR_C2E1"
HOLD = "HOLD_D4C2E0C_TEMPORAL_DERIVED_ENDPOINT_RESOLVER_AUDIT"
TEMPORAL_PATH_COLUMNS = ["temporal_path", "source_temporal_path", "stream_path", "artifact_path", "trajectory_path"]
STEP_COLUMNS = ["step", "frame_idx", "frame_index", "row_step", "local_step", "artifact_step", "t"]
REQUIRED_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
DERIVED = {"eef_speed", "eef_z_delta_since_close", "eef_speed_variance_5"}
OUT_FILES = [
    "d4c2e0c_temporal_derived_endpoint_resolver_report.json",
    "temporal_feature_derivation_schema.json",
    "temporal_artifact_25d_derivation_coverage.csv",
    "temporal_endpoint_resolution_by_profile.csv",
    "temporal_window_coverage_by_suite.csv",
    "object_endpoint_mismatch_debug.csv",
    "temporal_violations.csv",
    "checksum_report.json",
]
BASE13 = SC5_V2_FEATURES[:13]
AVAILABLE22 = [f for f in SC5_V2_FEATURES if f not in DERIVED]
PROFILES = {"all25": SC5_V2_FEATURES, "available22": AVAILABLE22, "base13": BASE13}


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

def split_list(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})

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
    role = str(r.get("event_role", ""))
    status = str(r.get("teacher_label_status", ""))
    if role == "primary_attackable" or status in {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"}: return 1
    if role or status: return 0
    return -1

def resolve_path(raw: str, dataset: Path, roots: Sequence[Path]) -> Optional[Path]:
    if not raw: return None
    p = Path(raw); cands = [p] if p.is_absolute() else [(dataset.parent / p).resolve(), (Path.cwd() / p).resolve()] + [(r / p).resolve() for r in roots]
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
    out = [name, f"f_{name}", f"feature_{name}", f"obs_{name}"] + extras.get(name, [])
    seen, ret = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); ret.append(x)
    return ret

def alias_map(header: Sequence[str]) -> Dict[str, str]:
    hs, ret = set(header), {}
    for f in SC5_V2_FEATURES:
        present = [c for c in alias_candidates(f) if c in hs]
        if present:
            ret[f] = f if f in present else (f"f_{f}" if f"f_{f}" in present else present[0])
    return ret

def build_matrix(rows: List[Dict[str, str]], amap: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, str]]:
    arr = np.full((len(rows), len(SC5_V2_FEATURES)), np.nan, dtype=np.float32)
    src: Dict[str, str] = {}
    idx = {f: i for i, f in enumerate(SC5_V2_FEATURES)}
    for j, f in enumerate(SC5_V2_FEATURES):
        col = amap.get(f, "")
        if col:
            for i, r in enumerate(rows): arr[i, j] = fnum(r.get(col))
            src[f] = "alias:" + col
    if not np.isfinite(arr[:, idx["eef_speed"]]).all():
        vx, vy, vz = arr[:, idx["eef_vx"]], arr[:, idx["eef_vy"]], arr[:, idx["eef_vz"]]
        ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
        arr[ok, idx["eef_speed"]] = np.sqrt(vx[ok] ** 2 + vy[ok] ** 2 + vz[ok] ** 2)
        src["eef_speed"] = "derived:sqrt_velocity"
    if not np.isfinite(arr[:, idx["eef_speed_variance_5"]]).all():
        s = arr[:, idx["eef_speed"]]
        for i in range(len(rows)):
            w = s[max(0, i-4):i+1]
            if len(w) >= 5 and np.isfinite(w).all(): arr[i, idx["eef_speed_variance_5"]] = float(np.var(w))
            elif np.isfinite(s[i]): arr[i, idx["eef_speed_variance_5"]] = 0.0
        src["eef_speed_variance_5"] = "derived:causal_var5_speed"
    if not np.isfinite(arr[:, idx["eef_z_delta_since_close"]]).all():
        cmd, z = arr[:, idx["gripper_command"]], arr[:, idx["eef_z"]]
        last_close, onset_seen, close_streak = -1, False, 0
        for i in range(len(rows)):
            if not np.isfinite(cmd[i]) or not np.isfinite(z[i]): continue
            close = bool(cmd[i] <= 0.5)
            close_streak = close_streak + 1 if close else 0
            if close and last_close < 0: last_close = i
            if close and close_streak == 1 and not onset_seen:
                onset_seen, last_close = True, i
            arr[i, idx["eef_z_delta_since_close"]] = float(z[i] - z[last_close]) if last_close >= 0 and np.isfinite(z[last_close]) else 0.0
        src["eef_z_delta_since_close"] = "derived:causal_z_delta_since_first_close"
    return arr, src

def vec(r: Dict[str, str]) -> np.ndarray:
    return np.asarray([fnum(r.get(f)) for f in SC5_V2_FEATURES], dtype=np.float32)

def step(r: Dict[str, str]) -> Tuple[Optional[int], str]:
    for k in STEP_COLUMNS:
        v = inum(r.get(k))
        if v is not None: return v, k
    return None, ""

def profile_idx(profile: str) -> List[int]:
    idx = {f: i for i, f in enumerate(SC5_V2_FEATURES)}
    return [idx[f] for f in PROFILES[profile]]

def match_endpoint(ctx: np.ndarray, art: np.ndarray, profiles: Sequence[str], min_feat: int, max_abs: float, mean_abs: float) -> Tuple[Optional[int], Dict[str, Any]]:
    last = {"match_status": "no_candidate", "profile": ""}
    for p in profiles:
        cols = profile_idx(p); valid_ctx = np.isfinite(ctx[cols])
        if int(valid_ctx.sum()) < min_feat: continue
        scores = []
        for i in range(art.shape[0]):
            vals = art[i, cols]; valid = valid_ctx & np.isfinite(vals); n = int(valid.sum())
            if n < min_feat: continue
            d = np.abs(ctx[cols][valid] - vals[valid]); scores.append((float(d.mean()), float(d.max()), n, i))
        if not scores: continue
        scores.sort(key=lambda x: (x[0], x[1], -x[2], x[3]))
        mean, mx, n, i = scores[0]
        last = {"match_status": "nearest_above_tolerance", "profile": p, "best_index": i, "best_feature_count": n, "best_max_abs": mx, "best_mean_abs": mean}
        if n >= min_feat and mx <= max_abs and mean <= mean_abs:
            return i, {"match_status": "resolved", "profile": p, "best_index": i, "best_feature_count": n, "best_max_abs": mx, "best_mean_abs": mean}
    return None, last

def split_leaks(rows: List[Dict[str, str]]) -> int:
    m: Dict[str, set[str]] = defaultdict(set)
    for r in rows:
        k = group_key(r)
        if k: m[k].add(split(r))
    return sum(1 for vals in m.values() if len([v for v in vals if v != "unknown"]) > 1)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-dataset", required=True); ap.add_argument("--context-schema", default="")
    ap.add_argument("--expected-rows", type=int, default=3717); ap.add_argument("--window-lengths", default="8,16,32")
    ap.add_argument("--temporal-root", action="append", default=[]); ap.add_argument("--min-suite-window-coverage", type=float, default=0.95)
    ap.add_argument("--endpoint-match-profiles", default="all25,available22,base13")
    ap.add_argument("--endpoint-match-min-features", type=int, default=8); ap.add_argument("--endpoint-match-max-abs", type=float, default=1e-3)
    ap.add_argument("--endpoint-match-mean-abs", type=float, default=1e-4); ap.add_argument("--output-root", required=True); ap.add_argument("--git-commit", default="")
    args = ap.parse_args(); started = time.time(); out = Path(args.output_root).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.context_dataset).expanduser().resolve(); windows = split_list(args.window_lengths); roots = [Path(p).expanduser().resolve() for p in args.temporal_root]
    profiles = [p.strip() for p in args.endpoint_match_profiles.split(",") if p.strip()]
    rows = read_csv(dataset); header = list(rows[0].keys()) if rows else []
    temporal_col = next((c for c in TEMPORAL_PATH_COLUMNS if c in header), "")
    violations: List[Dict[str, Any]] = []
    if args.expected_rows and len(rows) != args.expected_rows: violations.append({"code": "UNEXPECTED_ROW_COUNT", "severity": "hard", "actual": len(rows), "required": args.expected_rows})
    if not temporal_col: violations.append({"code": "TEMPORAL_PATH_COLUMN_MISSING", "severity": "hard"})
    suites = Counter(suite(r) for r in rows); labels = Counter(str(label(r)) for r in rows)
    if [s for s in REQUIRED_SUITES if suites.get(s,0)<=0]: violations.append({"code": "MISSING_SUITE_COVERAGE", "severity": "hard"})
    if labels.get("0",0)<=0 or labels.get("1",0)<=0: violations.append({"code": "MISSING_RUNTIME_OBJECTIVE_CLASS", "severity": "hard"})
    try: validate_no_forbidden_inputs([f"lag00_{f}" for f in SC5_V2_FEATURES])
    except Exception as e: violations.append({"code": "FORBIDDEN_TEMPORAL_FEATURE", "severity": "hard", "message": str(e)})
    if split_leaks(rows): violations.append({"code": "SPLIT_LEAKAGE", "severity": "hard"})

    raw_paths = sorted({str(r.get(temporal_col,"")) for r in rows if temporal_col and r.get(temporal_col)})
    cache: Dict[str, Dict[str, Any]] = {}; artifact_out = []
    for raw in raw_paths:
        p = resolve_path(raw, dataset, roots); meta = {"raw_temporal_path": raw, "resolved_path": str(p) if p else "", "exists": bool(p and p.exists())}
        art_rows: List[Dict[str,str]] = []; arr = np.zeros((0, len(SC5_V2_FEATURES)), dtype=np.float32); src = {}
        if p and p.exists():
            try:
                art_rows = read_csv(p); amap = alias_map(list(art_rows[0].keys()) if art_rows else []); arr, src = build_matrix(art_rows, amap)
                finite = np.isfinite(arr).sum(axis=0).tolist() if len(art_rows) else [0]*len(SC5_V2_FEATURES)
                missing = [f for f,c in zip(SC5_V2_FEATURES, finite) if c != len(art_rows) or len(art_rows)==0]
                meta.update({"readable": True, "row_count": len(art_rows), "complete_25d_after_derivation": len(missing)==0, "complete_feature_count": len(SC5_V2_FEATURES)-len(missing), "missing_or_partial_features": ";".join(missing), "derived_features": ";".join(sorted([k for k,v in src.items() if v.startswith("derived:")])), "source_map_json": json.dumps(src, sort_keys=True)})
            except Exception as e: meta.update({"readable": False, "row_count": 0, "complete_25d_after_derivation": False, "complete_feature_count": 0, "missing_or_partial_features": ";".join(SC5_V2_FEATURES), "read_error": str(e)})
        else: meta.update({"readable": False, "row_count": 0, "complete_25d_after_derivation": False, "complete_feature_count": 0, "missing_or_partial_features": ";".join(SC5_V2_FEATURES)})
        cache[raw] = {"meta": meta, "arr": arr}; artifact_out.append(meta)

    cov: Dict[Tuple[str,int], Dict[str,Any]] = {}
    for s in sorted(suites):
        for w in windows: cov[(s,w)] = {"suite": s, "window_length": w, "row_count": 0, "positive_rows": 0, "negative_rows": 0, "derived_25d_complete": 0, "endpoint_resolved": 0, "causal_window_available": 0}
    endpoints, obj_debug = [], []; ep_status = Counter(); ep_profile = Counter()
    for i, r in enumerate(rows):
        raw = str(r.get(temporal_col,"")) if temporal_col else ""; c = cache.get(raw, {"meta": {}, "arr": np.zeros((0,len(SC5_V2_FEATURES)), dtype=np.float32)})
        arr = c["arr"]; meta = c["meta"]; ep, step_col = step(r)
        if ep is not None: mm = {"match_status": "explicit_step", "profile": "explicit_step", "best_index": ep}
        else: ep, mm = match_endpoint(vec(r), arr, profiles, args.endpoint_match_min_features, args.endpoint_match_max_abs, args.endpoint_match_mean_abs)
        resolved = ep is not None and ep >= 0 and ep < int(arr.shape[0]); ep_status[str(mm.get("match_status",""))] += 1; ep_profile[str(mm.get("profile",""))] += 1
        for w in windows:
            cc = cov[(suite(r),w)]; cc["row_count"] += 1; cc["positive_rows"] += 1 if label(r)==1 else 0; cc["negative_rows"] += 1 if label(r)==0 else 0
            cc["derived_25d_complete"] += 1 if bool(meta.get("complete_25d_after_derivation")) else 0; cc["endpoint_resolved"] += 1 if resolved else 0
            cc["causal_window_available"] += 1 if bool(resolved and ep is not None and ep + 1 >= w) else 0
        rec = {"row_index": i, "suite": suite(r), "split": split(r), "label": label(r), "group_key": group_key(r), "endpoint_index": "" if ep is None else ep, "resolved": resolved, "match_status": mm.get("match_status",""), "profile": mm.get("profile",""), "best_feature_count": mm.get("best_feature_count",""), "best_max_abs": mm.get("best_max_abs",""), "best_mean_abs": mm.get("best_mean_abs",""), "artifact_row_count": int(arr.shape[0]), "derived_25d_complete": bool(meta.get("complete_25d_after_derivation")), "step_column": step_col}
        if len(endpoints) < 30000: endpoints.append(rec)
        if suite(r)=="libero_object" and (not resolved) and len(obj_debug)<1000: obj_debug.append(rec)
    cov_rows = []; min_25d = min_ep = min_win = 1.0
    for k in sorted(cov):
        cc = cov[k]; n = max(1, int(cc["row_count"]))
        for name in ["derived_25d_complete", "endpoint_resolved", "causal_window_available"]: cc[name+"_rate"] = float(cc[name]) / n
        min_25d = min(min_25d, cc["derived_25d_complete_rate"]); min_ep = min(min_ep, cc["endpoint_resolved_rate"]); min_win = min(min_win, cc["causal_window_available_rate"]); cov_rows.append(cc)
    if min_25d < args.min_suite_window_coverage: violations.append({"code": "LOW_DERIVED_25D_COVERAGE", "severity": "hard", "actual": min_25d, "required": args.min_suite_window_coverage})
    if min_ep < args.min_suite_window_coverage: violations.append({"code": "LOW_ENDPOINT_RESOLUTION_COVERAGE", "severity": "hard", "actual": min_ep, "required": args.min_suite_window_coverage})
    if min_win < args.min_suite_window_coverage: violations.append({"code": "LOW_CAUSAL_WINDOW_COVERAGE", "severity": "hard", "actual": min_win, "required": args.min_suite_window_coverage})
    hard = [v for v in violations if v.get("severity")=="hard"]; status = PASS if not hard else HOLD
    report = {"gate": GATE, "status": status, "reason": "hard_violation_count=0" if status==PASS else f"hard_violation_count={len(hard)}", "created_at_unix": time.time(), "runtime_seconds": time.time()-started, "git_commit": args.git_commit, "inputs": {"context_dataset": str(dataset), "context_dataset_sha256": sha256_file(dataset), "context_schema": str(Path(args.context_schema).expanduser().resolve()) if args.context_schema else "", "window_lengths": windows, "temporal_path_column": temporal_col, "endpoint_match_profiles": profiles}, "row_count": len(rows), "expected_rows": args.expected_rows, "suite_counts": dict(suites), "label_counts": dict(labels), "unique_temporal_path_count": len(raw_paths), "endpoint_status_counts": dict(ep_status), "endpoint_profile_counts": dict(ep_profile), "min_derived_25d_coverage_rate": min_25d, "min_endpoint_resolution_rate": min_ep, "min_causal_window_coverage_rate": min_win, "hard_violation_count": len(hard), "violations_by_code": dict(Counter(str(v.get("code")) for v in violations)), "recommendation": "proceed_to_C2E1_temporal_dataset_materialization" if status==PASS else "hold_fix_remaining_derivation_or_endpoint_resolution", "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED", "D5C": "NOT_RUN", "D6C_v3": "NOT_RUN"}}
    derivation = {"feature_family": "C2E_TEMPORAL_25D_DERIVATION_SCHEMA_V1", "derived_fields": {"eef_speed": "sqrt(eef_vx^2+eef_vy^2+eef_vz^2)", "eef_speed_variance_5": "causal variance over last 5 eef_speed values", "eef_z_delta_since_close": "causal z delta from first close, gripper_command <= 0.5"}, "endpoint_match_profiles": {p: PROFILES[p] for p in profiles}}
    write_json(out / "d4c2e0c_temporal_derived_endpoint_resolver_report.json", report); write_json(out / "temporal_feature_derivation_schema.json", derivation)
    write_csv(out / "temporal_artifact_25d_derivation_coverage.csv", artifact_out, ["raw_temporal_path", "resolved_path", "exists", "readable", "row_count", "complete_25d_after_derivation", "complete_feature_count", "missing_or_partial_features", "derived_features", "source_map_json", "read_error"])
    write_csv(out / "temporal_endpoint_resolution_by_profile.csv", endpoints, ["row_index", "suite", "split", "label", "group_key", "endpoint_index", "resolved", "match_status", "profile", "best_feature_count", "best_max_abs", "best_mean_abs", "artifact_row_count", "derived_25d_complete", "step_column"])
    write_csv(out / "object_endpoint_mismatch_debug.csv", obj_debug, ["row_index", "suite", "split", "label", "group_key", "endpoint_index", "resolved", "match_status", "profile", "best_feature_count", "best_max_abs", "best_mean_abs", "artifact_row_count", "derived_25d_complete", "step_column"])
    write_csv(out / "temporal_window_coverage_by_suite.csv", cov_rows, ["suite", "window_length", "row_count", "positive_rows", "negative_rows", "derived_25d_complete", "derived_25d_complete_rate", "endpoint_resolved", "endpoint_resolved_rate", "causal_window_available", "causal_window_available_rate"])
    write_csv(out / "temporal_violations.csv", violations, ["code", "severity", "actual", "required", "message"])
    checks = []
    for name in OUT_FILES:
        p = out / name
        if p.exists(): checks.append({"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for item in checks: f.write(f"{item['sha256']}  {item['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps({"status": status, "output_root": str(out), "row_count": len(rows), "min_derived_25d_coverage_rate": min_25d, "min_endpoint_resolution_rate": min_ep, "min_causal_window_coverage_rate": min_win, "hard_violation_count": len(hard), "recommendation": report["recommendation"]}, indent=2, sort_keys=True))
    return 0 if status == PASS else 2

if __name__ == "__main__":
    raise SystemExit(main())
