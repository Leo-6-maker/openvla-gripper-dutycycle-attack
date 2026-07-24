#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

from gripper_attack.sc5_multisuite_detector_runtime import (  # noqa: E402
    SC5MultiSuiteDetectorRuntime,
    SC5_V2_FEATURES,
    validate_no_forbidden_inputs,
)
import probe_clean2000_detector_25d_feature_extraction_v3 as v3  # noqa: E402

GATE = "D6C_CLEAN2000_DETECTOR_V2_STRATIFIED_THRESHOLD_GATE_ROBUSTNESS_AUDIT"
PASS = "PASS_CLEAN2000_DETECTOR_V2_STRATIFIED_THRESHOLD_GATE_ROBUSTNESS_AUDITED"
HOLD = "HOLD_CLEAN2000_DETECTOR_V2_STRATIFIED_THRESHOLD_GATE_ROBUSTNESS_AUDIT"
OUT_FILES = [
    "clean2000_detector_v2_d6c_stratified_threshold_gate_report.json",
    "d6c_selected_groups.csv",
    "d6c_config_metrics.csv",
    "d6c_suite_config_metrics.csv",
    "d6c_best_config_by_suite.csv",
    "d6c_group_replay.csv",
    "d6c_row_replay_sample.csv",
    "d6c_feature_extraction_failures.csv",
    "d6c_violations.csv",
    "checksum_report.json",
]
HISTORY_GAP_FEATURES = {
    "eef_z_delta_since_close",
    "qpos_delta_1",
    "qpos_delta_3",
    "opening_proxy_delta_3",
    "opening_proxy_variance_5",
}
DEFAULT_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return vals or list(DEFAULT_SUITES)


def stable_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def group_key(row: Dict[str, Any]) -> str:
    for key in ["group_key", "episode_key", "parent_id", "run_id", "record_id", "episode_id"]:
        if row.get(key):
            return str(row[key])
    return str(row.get("record_id", ""))


def has_imputation(row: Dict[str, Any]) -> bool:
    if truthy(row.get("has_imputation")):
        return True
    try:
        return int(float(row.get("imputed_feature_count", 0) or 0)) > 0
    except Exception:
        return False


def classify_group(rows: List[Dict[str, Any]]) -> str:
    labels = {str(r.get("teacher_label_status", "")) for r in rows}
    roles = {str(r.get("event_role", "")) for r in rows}
    if "VALID_PRIMARY" in labels or "primary_attackable" in roles:
        return "has_primary"
    if "VALID_AUXILIARY" in labels or "auxiliary_manipulation" in roles:
        return "auxiliary_only"
    if labels == {"NO_EVENT"} or "unsupported_or_abstain" in roles:
        return "no_event_or_unsupported"
    return "other_no_primary"


def suite_or_mixed(rows: List[Dict[str, Any]]) -> str:
    vals = sorted({str(r.get("suite", "")) for r in rows})
    return vals[0] if len(vals) == 1 else "MIXED:" + ";".join(vals)


def split_or_mixed(rows: List[Dict[str, Any]]) -> str:
    vals = sorted({str(r.get("split", "")) for r in rows})
    return vals[0] if len(vals) == 1 else "MIXED:" + ";".join(vals)


def mode_nonempty(values: Iterable[str]) -> str:
    c = Counter(v for v in values if v)
    return c.most_common(1)[0][0] if c else ""


def finite_features(vals: Dict[str, float]) -> Tuple[bool, List[str]]:
    bad = [fn for fn in SC5_V2_FEATURES if not math.isfinite(vals.get(fn, math.nan))]
    return not bad, bad


def stream_history_gap_impute(vals: Dict[str, float], methods: Dict[str, str], fields: Dict[str, str]) -> List[str]:
    imputed: List[str] = []
    for feat in HISTORY_GAP_FEATURES:
        if math.isfinite(vals.get(feat, math.nan)):
            continue
        allowed = False
        basis = ""
        if feat in {"qpos_delta_1", "qpos_delta_3"}:
            allowed = math.isfinite(vals.get("gripper_qpos", math.nan))
            basis = "current_gripper_qpos_present"
        elif feat in {"opening_proxy_delta_3", "opening_proxy_variance_5"}:
            allowed = math.isfinite(vals.get("gripper_opening_proxy", math.nan))
            basis = "current_gripper_opening_proxy_present"
        elif feat == "eef_z_delta_since_close":
            allowed = math.isfinite(vals.get("eef_z", math.nan))
            basis = "current_eef_z_present_no_close_origin"
        if allowed:
            vals[feat] = 0.0
            methods[feat] = "stream_zero_history_gap_current_base_present"
            fields[feat] = basis
            imputed.append(feat)
    return imputed


def extract_stream_features(rows: List[Dict[str, Any]], idx: int, allow_stream_imputation: bool) -> Tuple[Dict[str, float], List[str], List[str]]:
    vals, methods, fields = v3.compute_features(rows, idx, "STREAM_UNKNOWN")
    imputed: List[str] = []
    ok, missing = finite_features(vals)
    if not ok and allow_stream_imputation:
        imputed = stream_history_gap_impute(vals, methods, fields)
        ok, missing = finite_features(vals)
    return {fn: float(vals.get(fn, math.nan)) for fn in SC5_V2_FEATURES}, missing, imputed


def add_violation(rows: List[Dict[str, Any]], code: str, severity: str, detail: str) -> None:
    rows.append({"violation_code": code, "severity": severity, "detail": detail})


def metrics_for(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(group_rows)
    primary = [r for r in group_rows if int(r["has_primary_truth"]) == 1]
    no_primary = [r for r in group_rows if int(r["has_primary_truth"]) == 0]
    tp = sum(1 for r in primary if int(r["emitted"]) == 1)
    fn = sum(1 for r in primary if int(r["emitted"]) == 0)
    fp = sum(1 for r in no_primary if int(r["emitted"]) == 1)
    tn = sum(1 for r in no_primary if int(r["emitted"]) == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    no_primary_emit_rate = fp / max(1, fp + tn)
    return {
        "n_groups": n,
        "primary_groups": len(primary),
        "no_primary_groups": len(no_primary),
        "tp_emit": tp,
        "fn_no_emit": fn,
        "fp_emit": fp,
        "tn_no_emit": tn,
        "primary_recall": recall,
        "emit_precision": precision,
        "emit_f1": f1,
        "no_primary_emit_rate": no_primary_emit_rate,
        "group_emit_rate": sum(int(r["emitted"]) for r in group_rows) / max(1, n),
    }


def build_groups(dataset_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        g = group_key(row)
        if not g:
            raise ValueError(f"row missing group key: {row.get('label_row_id')}")
        groups[g].append(row)
    return groups


def stable_shuffle(items: List[Tuple[str, List[Dict[str, Any]]]], seed: int, salt: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    def key(item: Tuple[str, List[Dict[str, Any]]]) -> str:
        return hashlib.sha256(f"{seed}:{salt}:{item[0]}".encode("utf-8")).hexdigest()
    return sorted(items, key=key)


def select_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    suites: List[str],
    max_groups_per_suite: int,
    seed: int,
    shard_index: int,
    shard_count: int,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    selected: List[Tuple[str, List[Dict[str, Any]]]] = []
    selection_rows: List[Dict[str, Any]] = []
    for suite in suites:
        suite_groups = [(g, rows) for g, rows in groups.items() if suite_or_mixed(rows) == suite]
        primary = [(g, r) for g, r in suite_groups if classify_group(r) == "has_primary"]
        no_primary = [(g, r) for g, r in suite_groups if classify_group(r) != "has_primary"]
        primary = stable_shuffle(primary, seed, suite + ":primary")
        no_primary = stable_shuffle(no_primary, seed, suite + ":no_primary")
        target_each = max_groups_per_suite // 2
        chosen = primary[:target_each] + no_primary[:target_each]
        if len(chosen) < max_groups_per_suite:
            used = {g for g, _ in chosen}
            rest = [(g, r) for g, r in primary + no_primary if g not in used]
            chosen.extend(rest[: max_groups_per_suite - len(chosen)])
        chosen = stable_shuffle(chosen, seed, suite + ":final")
        if shard_count > 1:
            chosen = [(g, r) for j, (g, r) in enumerate(chosen) if j % shard_count == shard_index]
        for g, rows in chosen:
            selected.append((g, rows))
            selection_rows.append({
                "group_key": g,
                "suite": suite_or_mixed(rows),
                "split": split_or_mixed(rows),
                "group_class": classify_group(rows),
                "row_count": len(rows),
                "temporal_path": mode_nonempty(str(r.get("temporal_path", "")) for r in rows),
                "imputed_label_row_count": sum(1 for r in rows if has_imputation(r)),
                "primary_label_rows": sum(1 for r in rows if str(r.get("teacher_label_status")) == "VALID_PRIMARY"),
                "auxiliary_label_rows": sum(1 for r in rows if str(r.get("teacher_label_status")) == "VALID_AUXILIARY"),
                "no_event_label_rows": sum(1 for r in rows if str(r.get("teacher_label_status")) == "NO_EVENT"),
            })
    return selected, selection_rows


def load_stream(
    group_key_value: str,
    group_rows: List[Dict[str, Any]],
    temporal_cache: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
    extraction_failures: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, Dict[str, float], List[str]]], Dict[str, Any]]:
    temporal_path = mode_nonempty(str(r.get("temporal_path", "")) for r in group_rows)
    if not temporal_path:
        extraction_failures.append({"group_key": group_key_value, "temporal_path": "", "row_index": "", "failure_reason": "GROUP_MISSING_TEMPORAL_PATH", "missing_features": "ALL"})
        return [], {"temporal_path": "", "read_failed": 0, "attempted": 0, "used": 0, "missing": 0, "imputation_count": 0}
    try:
        if temporal_path not in temporal_cache:
            temporal_cache[temporal_path] = v3.read_temporal(temporal_path)
        trows = temporal_cache[temporal_path]
    except Exception as exc:
        extraction_failures.append({"group_key": group_key_value, "temporal_path": temporal_path, "row_index": "", "failure_reason": "TEMPORAL_READ_FAILED", "missing_features": f"{type(exc).__name__}: {exc}"})
        return [], {"temporal_path": temporal_path, "read_failed": 1, "attempted": 0, "used": 0, "missing": 0, "imputation_count": 0}
    max_n = len(trows)
    if args.max_steps_per_group > 0:
        max_n = min(max_n, args.max_steps_per_group)
    stream: List[Tuple[int, Dict[str, float], List[str]]] = []
    attempted = 0
    missing_n = 0
    imputation_count = 0
    for idx in range(0, max_n, max(1, args.stride)):
        attempted += 1
        vals, missing, imputed_features = extract_stream_features(trows, idx, args.allow_stream_history_gap_imputation)
        if missing:
            missing_n += 1
            if len(extraction_failures) < args.max_failure_rows:
                extraction_failures.append({"group_key": group_key_value, "temporal_path": temporal_path, "row_index": idx, "failure_reason": "MISSING_STREAM_FEATURES", "missing_features": ";".join(missing)})
            continue
        stream.append((idx, vals, imputed_features))
        imputation_count += len(imputed_features)
    return stream, {"temporal_path": temporal_path, "read_failed": 0, "attempted": attempted, "used": len(stream), "missing": missing_n, "imputation_count": imputation_count}


def config_id(tau_c: float, tau_p: float, guard: int, require_primary_role: bool) -> str:
    role_gate = "role_on" if require_primary_role else "role_off"
    return f"tc{tau_c:.2f}_tp{tau_p:.2f}_g{guard}_{role_gate}"


def replay_group_for_config(
    checkpoint: Path,
    stream: List[Tuple[int, Dict[str, float], List[str]]],
    tau_c: float,
    tau_p: float,
    tau_r: float,
    guard: int,
    require_primary_role: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    detector = SC5MultiSuiteDetectorRuntime(
        str(checkpoint),
        tau_corridor=tau_c,
        tau_release=tau_r,
        tau_primary=tau_p,
        guard=guard,
        require_primary_event_role=require_primary_role,
    )
    effective = {"tau_corridor": detector.tau_c, "tau_primary": detector.tau_p, "tau_release": detector.tau_r, "guard": detector.guard}
    emitted = False
    first_emit_step = ""
    first_emit_local_index = ""
    last_state = "IDLE"
    max_corridor_p = -1.0
    max_primary_p = -1.0
    min_release_p = 1e9
    phase_counts = Counter()
    role_counts = Counter()
    row_samples: List[Dict[str, Any]] = []
    for local_i, (step, vals, imputed_features) in enumerate(stream):
        decision = detector.update(vals, step=step)
        corr = decision.get("corridor_p")
        rel = decision.get("release_p")
        primary = decision.get("primary_p")
        phase = str(decision.get("pred_phase", ""))
        role = str(decision.get("pred_event_role", ""))
        if phase:
            phase_counts[phase] += 1
        if role:
            role_counts[role] += 1
        if corr is not None:
            max_corridor_p = max(max_corridor_p, float(corr))
        if primary is not None:
            max_primary_p = max(max_primary_p, float(primary))
        if rel is not None:
            min_release_p = min(min_release_p, float(rel))
        if bool(decision.get("emitted")) and not emitted:
            emitted = True
            first_emit_step = str(decision.get("emit_step", step))
            first_emit_local_index = str(local_i)
        last_state = str(decision.get("state", ""))
        row_samples.append({
            "local_row_index": local_i,
            "row_index": step,
            "runtime_state": decision.get("state", ""),
            "emitted": int(bool(decision.get("emitted"))),
            "arm_step": decision.get("arm_step", ""),
            "emit_step": decision.get("emit_step", ""),
            "corridor_p": "" if corr is None else corr,
            "release_p": "" if rel is None else rel,
            "primary_p": "" if primary is None else primary,
            "pred_phase": phase,
            "pred_event_role": role,
            "stream_imputed_feature_count": len(imputed_features),
            "stream_imputed_features": ";".join(imputed_features),
        })
    summary = {
        "emitted": int(emitted),
        "first_emit_step": first_emit_step,
        "first_emit_local_index": first_emit_local_index,
        "last_state": last_state,
        "max_corridor_p": max_corridor_p if max_corridor_p >= 0 else "",
        "max_primary_p": max_primary_p if max_primary_p >= 0 else "",
        "min_release_p": min_release_p if min_release_p < 1e9 else "",
        "pred_phase_counts": json.dumps(dict(phase_counts), sort_keys=True),
        "pred_event_role_counts": json.dumps(dict(role_counts), sort_keys=True),
    }
    return summary, row_samples, effective


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise ValueError("D6C is CPU-only; use --device cpu")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    dataset_path = Path(args.frozen_dataset)
    checkpoint_path = Path(args.checkpoint)
    rows = read_csv(dataset_path)
    if len(rows) != args.expected_rows:
        raise ValueError(f"dataset rows={len(rows)} expected={args.expected_rows}")
    suites = parse_csv_strings(args.suites)
    tau_corridors = parse_csv_floats(args.tau_corridor_list)
    tau_primaries = parse_csv_floats(args.tau_primary_list)
    guards = parse_csv_ints(args.guard_list)
    role_gate_values = [True]
    if args.include_disable_primary_event_role_ablation:
        role_gate_values.append(False)
    if not (0 <= args.group_shard_index < args.group_shard_count):
        raise ValueError("expected 0 <= group_shard_index < group_shard_count")

    groups = build_groups(rows)
    selected, selection_rows = select_groups(groups, suites, args.max_groups_per_suite, args.seed, args.group_shard_index, args.group_shard_count)
    extraction_failures: List[Dict[str, Any]] = []
    temporal_cache: Dict[str, List[Dict[str, Any]]] = {}
    group_streams: Dict[str, List[Tuple[int, Dict[str, float], List[str]]]] = {}
    group_stream_stats: Dict[str, Dict[str, Any]] = {}
    for g, grows in selected:
        stream, stats = load_stream(g, grows, temporal_cache, args, extraction_failures)
        group_streams[g] = stream
        group_stream_stats[g] = stats

    configs: List[Dict[str, Any]] = []
    for tc in tau_corridors:
        for tp in tau_primaries:
            for guard in guards:
                for role_gate in role_gate_values:
                    configs.append({"config_id": config_id(tc, tp, guard, role_gate), "tau_corridor": tc, "tau_primary": tp, "tau_release": args.tau_release, "guard": guard, "require_primary_event_role": role_gate})

    group_replay: List[Dict[str, Any]] = []
    row_sample: List[Dict[str, Any]] = []
    effective_threshold_mismatch_count = 0
    for cfg in configs:
        for g, grows in selected:
            stream = group_streams[g]
            stats = group_stream_stats[g]
            gclass = classify_group(grows)
            if not stream:
                group_replay.append({
                    "config_id": cfg["config_id"],
                    "group_key": g,
                    "suite": suite_or_mixed(grows),
                    "split": split_or_mixed(grows),
                    "group_class": gclass,
                    "has_primary_truth": int(gclass == "has_primary"),
                    "no_primary_truth": int(gclass != "has_primary"),
                    "emitted": 0,
                    "stream_rows_attempted": stats.get("attempted", 0),
                    "stream_rows_used": 0,
                    "stream_rows_missing": stats.get("missing", 0),
                    "stream_imputation_count": stats.get("imputation_count", 0),
                    "temporal_path": stats.get("temporal_path", ""),
                    "first_emit_step": "",
                    "first_emit_local_index": "",
                    "last_state": "NO_STREAM",
                    "max_corridor_p": "",
                    "max_primary_p": "",
                    "min_release_p": "",
                    "pred_phase_counts": "{}",
                    "pred_event_role_counts": "{}",
                    "tau_corridor": cfg["tau_corridor"],
                    "tau_primary": cfg["tau_primary"],
                    "tau_release": cfg["tau_release"],
                    "guard": cfg["guard"],
                    "require_primary_event_role": int(cfg["require_primary_event_role"]),
                    "effective_tau_corridor": "",
                    "effective_tau_primary": "",
                    "effective_tau_release": "",
                    "effective_guard": "",
                    "imputed_label_row_count": sum(1 for r in grows if has_imputation(r)),
                    "primary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "VALID_PRIMARY"),
                    "auxiliary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "VALID_AUXILIARY"),
                    "no_event_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "NO_EVENT"),
                })
                continue
            summary, samples, effective = replay_group_for_config(
                checkpoint_path,
                stream,
                float(cfg["tau_corridor"]),
                float(cfg["tau_primary"]),
                float(cfg["tau_release"]),
                int(cfg["guard"]),
                bool(cfg["require_primary_event_role"]),
            )
            if abs(float(effective["tau_corridor"]) - float(cfg["tau_corridor"])) > 1e-9 or abs(float(effective["tau_primary"]) - float(cfg["tau_primary"])) > 1e-9 or int(effective["guard"]) != int(cfg["guard"]):
                effective_threshold_mismatch_count += 1
            base = {
                "config_id": cfg["config_id"],
                "group_key": g,
                "suite": suite_or_mixed(grows),
                "split": split_or_mixed(grows),
                "group_class": gclass,
                "has_primary_truth": int(gclass == "has_primary"),
                "no_primary_truth": int(gclass != "has_primary"),
                "stream_rows_attempted": stats.get("attempted", 0),
                "stream_rows_used": stats.get("used", 0),
                "stream_rows_missing": stats.get("missing", 0),
                "stream_imputation_count": stats.get("imputation_count", 0),
                "temporal_path": stats.get("temporal_path", ""),
                "tau_corridor": cfg["tau_corridor"],
                "tau_primary": cfg["tau_primary"],
                "tau_release": cfg["tau_release"],
                "guard": cfg["guard"],
                "require_primary_event_role": int(cfg["require_primary_event_role"]),
                "effective_tau_corridor": effective["tau_corridor"],
                "effective_tau_primary": effective["tau_primary"],
                "effective_tau_release": effective["tau_release"],
                "effective_guard": effective["guard"],
                "imputed_label_row_count": sum(1 for r in grows if has_imputation(r)),
                "primary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "VALID_PRIMARY"),
                "auxiliary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "VALID_AUXILIARY"),
                "no_event_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status")) == "NO_EVENT"),
            }
            group_replay.append({**base, **summary})
            if len(row_sample) < args.max_row_sample:
                for s in samples[: max(0, args.max_row_sample - len(row_sample))]:
                    row_sample.append({**{k: base[k] for k in ["config_id", "group_key", "suite", "split", "group_class", "tau_corridor", "tau_primary", "guard", "require_primary_event_role"]}, **s})

    config_metrics: List[Dict[str, Any]] = []
    suite_metrics: List[Dict[str, Any]] = []
    best_by_suite: List[Dict[str, Any]] = []
    for cfg in configs:
        rows_cfg = [r for r in group_replay if r["config_id"] == cfg["config_id"]]
        m = metrics_for(rows_cfg)
        config_metrics.append({**cfg, **m})
        for suite in suites:
            sm = metrics_for([r for r in rows_cfg if r["suite"] == suite])
            suite_metrics.append({**cfg, "suite": suite, **sm})
    for suite in suites:
        candidates = [r for r in suite_metrics if r["suite"] == suite]
        if candidates:
            best = sorted(candidates, key=lambda r: (float(r.get("emit_f1", 0.0)), float(r.get("primary_recall", 0.0)), -float(r.get("no_primary_emit_rate", 0.0))), reverse=True)[0]
            feasible = [r for r in candidates if float(r.get("primary_recall", 0.0)) >= args.min_primary_recall and float(r.get("no_primary_emit_rate", 0.0)) <= args.max_no_primary_emit_rate and int(r.get("n_groups", 0)) >= args.min_usable_groups_per_suite]
            best_by_suite.append({**best, "suite_has_feasible_config": int(bool(feasible)), "feasible_config_count": len(feasible)})

    violations: List[Dict[str, Any]] = []
    selected_suites = Counter(r["suite"] for r in selection_rows)
    for suite in suites:
        if selected_suites.get(suite, 0) == 0:
            add_violation(violations, "SUITE_NOT_SELECTED", "HOLD", f"suite={suite}")
    stream_attempted = sum(int(v.get("attempted", 0)) for v in group_stream_stats.values())
    stream_used = sum(int(v.get("used", 0)) for v in group_stream_stats.values())
    stream_missing = sum(int(v.get("missing", 0)) for v in group_stream_stats.values())
    stream_missing_rate = stream_missing / max(1, stream_attempted)
    if stream_used == 0:
        add_violation(violations, "NO_STREAM_ROWS_USABLE", "HOLD", "no finite 25D stream rows could be replayed")
    if stream_missing_rate > args.max_stream_missing_rate:
        add_violation(violations, "HIGH_STREAM_FEATURE_MISSING_RATE", "HOLD", f"missing_rate={stream_missing_rate}")
    if effective_threshold_mismatch_count:
        add_violation(violations, "EFFECTIVE_THRESHOLD_MISMATCH", "HOLD", f"count={effective_threshold_mismatch_count}")
    metric_signature = {(r["tp_emit"], r["fn_no_emit"], r["fp_emit"], r["tn_no_emit"]) for r in config_metrics}
    if len(configs) > 1 and len(metric_signature) == 1:
        add_violation(violations, "GRID_METRICS_IDENTICAL", "WARN", "all threshold configs produced identical aggregate tp/fn/fp/tn; thresholds were still verified as propagated")
    for row in best_by_suite:
        suite = str(row["suite"])
        if int(row.get("n_groups", 0)) < args.min_usable_groups_per_suite:
            add_violation(violations, "SUITE_TOO_FEW_USABLE_GROUPS", "HOLD", f"suite={suite} n={row.get('n_groups')}")
        if int(row.get("no_primary_groups", 0)) > 0 and int(row.get("suite_has_feasible_config", 0)) == 0:
            add_violation(violations, "NO_SAFE_THRESHOLD_CONFIG_FOR_SUITE", "HOLD", f"suite={suite} best_recall={row.get('primary_recall')} best_fp_rate={row.get('no_primary_emit_rate')}")
        elif float(row.get("primary_recall", 0.0)) < args.min_primary_recall:
            add_violation(violations, "LOW_BEST_PRIMARY_RECALL_FOR_SUITE", "HOLD", f"suite={suite} recall={row.get('primary_recall')}")
    phase_counter = Counter()
    role_counter = Counter()
    for r in group_replay:
        try:
            phase_counter.update(json.loads(str(r.get("pred_phase_counts", "{}"))))
        except Exception:
            pass
        try:
            role_counter.update(json.loads(str(r.get("pred_event_role_counts", "{}"))))
        except Exception:
            pass
    if sum(1 for _k, v in phase_counter.items() if int(v) > 0) <= 2:
        add_violation(violations, "PHASE_HEAD_LOW_DIVERSITY", "WARN", f"phase_counts={dict(phase_counter)}")

    hard = [v for v in violations if v["severity"] == "HOLD"]
    status = PASS if not hard else HOLD
    reason = "" if not hard else f"hard_violation_count={len(hard)}"

    write_csv(out / "d6c_selected_groups.csv", selection_rows, ["group_key", "suite", "split", "group_class", "row_count", "temporal_path", "imputed_label_row_count", "primary_label_rows", "auxiliary_label_rows", "no_event_label_rows"])
    write_csv(out / "d6c_config_metrics.csv", config_metrics, sorted({k for r in config_metrics for k in r.keys()}))
    write_csv(out / "d6c_suite_config_metrics.csv", suite_metrics, sorted({k for r in suite_metrics for k in r.keys()}))
    write_csv(out / "d6c_best_config_by_suite.csv", best_by_suite, sorted({k for r in best_by_suite for k in r.keys()}))
    write_csv(out / "d6c_group_replay.csv", group_replay, sorted({k for r in group_replay for k in r.keys()}))
    write_csv(out / "d6c_row_replay_sample.csv", row_sample, sorted({k for r in row_sample for k in r.keys()}))
    write_csv(out / "d6c_feature_extraction_failures.csv", extraction_failures, ["group_key", "temporal_path", "row_index", "failure_reason", "missing_features"])
    write_csv(out / "d6c_violations.csv", violations, ["violation_code", "severity", "detail"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "frozen_dataset": str(dataset_path),
        "frozen_dataset_sha256": sha256_file(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "expected_rows": args.expected_rows,
        "label_row_count": len(rows),
        "selected_group_count": len(selection_rows),
        "selected_suite_counts": dict(selected_suites),
        "selected_group_class_counts": dict(Counter(r["group_class"] for r in selection_rows)),
        "temporal_files_read": len(temporal_cache),
        "stream_rows_attempted": stream_attempted,
        "stream_rows_used": stream_used,
        "stream_rows_missing": stream_missing,
        "stream_feature_missing_rate": stream_missing_rate,
        "stream_imputation_count": sum(int(v.get("imputation_count", 0)) for v in group_stream_stats.values()),
        "config_count": len(configs),
        "configs": configs,
        "config_metrics": config_metrics,
        "best_config_by_suite": best_by_suite,
        "phase_counts_agg": dict(phase_counter),
        "event_role_counts_agg": dict(role_counter),
        "effective_threshold_mismatch_count": effective_threshold_mismatch_count,
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "audit_thresholds": {"min_primary_recall": args.min_primary_recall, "max_no_primary_emit_rate": args.max_no_primary_emit_rate, "min_usable_groups_per_suite": args.min_usable_groups_per_suite, "max_stream_missing_rate": args.max_stream_missing_rate},
        "interpretation": "CPU-only D6C stratified dense temporal threshold/gate audit. Thresholds are explicitly passed into a fresh runtime per config/group and effective runtime thresholds are recorded. This is not LIBERO rollout and does not execute OpenVLA, env.step, intervention, attack, or retraining.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "device": "cpu", "OpenVLA_model": "NOT_LOADED", "model_inference": "DETECTOR_RUNTIME_ONLY_ON_CLEAN_TEMPORAL_ARTIFACTS", "LIBERO_runtime": "NOT_PERFORMED", "env_reset": "NOT_PERFORMED", "env_set_init_state": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_detector_v2_d6c_stratified_threshold_gate_report.json", report)
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
    p.add_argument("--frozen-dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-rows", type=int, default=3717)
    p.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    p.add_argument("--max-groups-per-suite", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-shard-index", type=int, default=0)
    p.add_argument("--group-shard-count", type=int, default=1)
    p.add_argument("--tau-corridor-list", default="0.20,0.30,0.40")
    p.add_argument("--tau-primary-list", default="0.40,0.50,0.60")
    p.add_argument("--tau-release", type=float, default=0.30)
    p.add_argument("--guard-list", default="5")
    p.add_argument("--include-disable-primary-event-role-ablation", action="store_true")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-steps-per-group", type=int, default=160)
    p.add_argument("--allow-stream-history-gap-imputation", action="store_true")
    p.add_argument("--max-row-sample", type=int, default=5000)
    p.add_argument("--max-failure-rows", type=int, default=5000)
    p.add_argument("--min-primary-recall", type=float, default=0.50)
    p.add_argument("--max-no-primary-emit-rate", type=float, default=0.35)
    p.add_argument("--min-usable-groups-per-suite", type=int, default=5)
    p.add_argument("--max-stream-missing-rate", type=float, default=0.20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
