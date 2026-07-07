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
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import (
    SC5MultiSuiteDetectorRuntime,
    SC5_V2_FEATURES,
    validate_no_forbidden_inputs,
)

GATE = "D6A_CLEAN2000_DETECTOR_V2_OFFLINE_FSM_ROW_REPLAY_AUDIT"
PASS = "PASS_CLEAN2000_DETECTOR_V2_OFFLINE_FSM_ROW_REPLAY_AUDITED"
OUT_FILES = [
    "clean2000_detector_v2_offline_fsm_replay_audit_report.json",
    "d6_offline_fsm_group_replay.csv",
    "d6_offline_fsm_row_replay.csv",
    "d6_offline_fsm_metrics_by_suite_class.csv",
    "d6_offline_fsm_violations.csv",
    "checksum_report.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def row_order_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
    step = stable_float(row.get("feature_extraction_step"), math.nan)
    if not math.isfinite(step):
        step = stable_float(row.get("extraction_step"), math.nan)
    if not math.isfinite(step):
        step = stable_float(row.get("teacher_anchor_step"), math.nan)
    if not math.isfinite(step):
        step = stable_float(row.get("extraction_index"), math.nan)
    idx = stable_float(row.get("extraction_index"), 0.0)
    if not math.isfinite(step):
        step = idx if math.isfinite(idx) else 0.0
    return (float(step), float(idx if math.isfinite(idx) else 0.0), str(row.get("label_row_id", "")))


def has_imputation(row: Dict[str, Any]) -> bool:
    if truthy(row.get("has_imputation")):
        return True
    try:
        return int(float(row.get("imputed_feature_count", 0) or 0)) > 0
    except Exception:
        return False


def features(row: Dict[str, Any]) -> Dict[str, float]:
    out = {fn: stable_float(row.get(fn)) for fn in SC5_V2_FEATURES}
    if not all(math.isfinite(v) for v in out.values()):
        bad = [k for k, v in out.items() if not math.isfinite(v)]
        raise ValueError(f"nonfinite features: {bad[:5]}")
    return out


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


def split_or_mixed(rows: List[Dict[str, Any]]) -> str:
    vals = sorted({str(r.get("split", "")) for r in rows})
    return vals[0] if len(vals) == 1 else "MIXED:" + ";".join(vals)


def suite_or_mixed(rows: List[Dict[str, Any]]) -> str:
    vals = sorted({str(r.get("suite", "")) for r in rows})
    return vals[0] if len(vals) == 1 else "MIXED:" + ";".join(vals)


def add_violation(rows: List[Dict[str, Any]], code: str, severity: str, detail: str) -> None:
    rows.append({"violation_code": code, "severity": severity, "detail": detail})


def safe_num(value: Any) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    validate_no_forbidden_inputs(list(SC5_V2_FEATURES))
    if args.device != "cpu":
        raise ValueError("D6A offline replay is CPU-only; use --device cpu")
    dataset_path = Path(args.frozen_dataset)
    checkpoint_path = Path(args.checkpoint)
    d5_report_path = Path(args.d5_report)
    d5_report = json.loads(d5_report_path.read_text(encoding="utf-8")) if args.d5_report else {}
    if d5_report and str(d5_report.get("status", "")) != "PASS_CLEAN2000_MULTISUITE_SC5_DETECTOR_V2_POST_TRAINING_AUDITED":
        raise ValueError(f"D5 report is not PASS: {d5_report.get('status')}")
    rows = read_csv(dataset_path)
    if len(rows) != args.expected_rows:
        raise ValueError(f"dataset rows={len(rows)} expected={args.expected_rows}")

    detector = SC5MultiSuiteDetectorRuntime(
        str(checkpoint_path),
        tau_corridor=args.tau_corridor,
        tau_release=args.tau_release,
        tau_primary=args.tau_primary,
        guard=args.guard,
        require_primary_event_role=not args.disable_primary_event_role_gate,
    )
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        g = group_key(row)
        if not g:
            raise ValueError(f"row missing group key: {row.get('label_row_id')}")
        groups[g].append(row)

    row_replay: List[Dict[str, Any]] = []
    group_replay: List[Dict[str, Any]] = []
    for g, grows_unsorted in sorted(groups.items()):
        grows = sorted(grows_unsorted, key=row_order_key)
        detector.reset()
        first_emit_label = ""
        first_emit_step = ""
        emitted = False
        max_corridor_p = -1.0
        max_primary_p = -1.0
        min_release_p = 1e9
        last_state = "IDLE"
        for local_idx, row in enumerate(grows):
            step = int(round(row_order_key(row)[0]))
            decision = detector.update(features(row), step=step)
            corr = decision.get("corridor_p")
            rel = decision.get("release_p")
            primary = decision.get("primary_p")
            if corr is not None:
                max_corridor_p = max(max_corridor_p, float(corr))
            if primary is not None:
                max_primary_p = max(max_primary_p, float(primary))
            if rel is not None:
                min_release_p = min(min_release_p, float(rel))
            if bool(decision.get("emitted")) and not emitted:
                emitted = True
                first_emit_label = str(row.get("label_row_id", ""))
                first_emit_step = str(decision.get("emit_step", step))
            last_state = str(decision.get("state", ""))
            row_replay.append({
                "group_key": g,
                "local_row_index": local_idx,
                "label_row_id": row.get("label_row_id", ""),
                "record_id": row.get("record_id", ""),
                "split": row.get("split", ""),
                "suite": row.get("suite", ""),
                "teacher_label_status": row.get("teacher_label_status", ""),
                "event_role_true": row.get("event_role", ""),
                "feature_extraction_step": step,
                "runtime_state": decision.get("state", ""),
                "emitted": int(bool(decision.get("emitted"))),
                "arm_step": decision.get("arm_step", ""),
                "emit_step": decision.get("emit_step", ""),
                "corridor_p": "" if corr is None else corr,
                "release_p": "" if rel is None else rel,
                "primary_p": "" if primary is None else primary,
                "pred_phase": decision.get("pred_phase", ""),
                "pred_event_role": decision.get("pred_event_role", ""),
                "has_imputation": int(has_imputation(row)),
            })
        gclass = classify_group(grows)
        is_primary = int(gclass == "has_primary")
        is_no_primary = int(gclass != "has_primary")
        group_replay.append({
            "group_key": g,
            "split": split_or_mixed(grows),
            "suite": suite_or_mixed(grows),
            "group_class": gclass,
            "row_count": len(grows),
            "has_primary_truth": is_primary,
            "no_primary_truth": is_no_primary,
            "emitted": int(emitted),
            "first_emit_label_row_id": first_emit_label,
            "first_emit_step": first_emit_step,
            "last_state": last_state,
            "max_corridor_p": max_corridor_p if max_corridor_p >= 0 else "",
            "max_primary_p": max_primary_p if max_primary_p >= 0 else "",
            "min_release_p": min_release_p if min_release_p < 1e9 else "",
            "imputed_row_count": sum(1 for r in grows if has_imputation(r)),
            "primary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status", "")) == "VALID_PRIMARY"),
            "auxiliary_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status", "")) == "VALID_AUXILIARY"),
            "no_event_label_rows": sum(1 for r in grows if str(r.get("teacher_label_status", "")) == "NO_EVENT"),
        })

    def metrics_for(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        primary = [r for r in subset if int(r["has_primary_truth"]) == 1]
        no_primary = [r for r in subset if int(r["has_primary_truth"]) == 0]
        tp = sum(1 for r in primary if int(r["emitted"]) == 1)
        fn = sum(1 for r in primary if int(r["emitted"]) == 0)
        fp = sum(1 for r in no_primary if int(r["emitted"]) == 1)
        tn = sum(1 for r in no_primary if int(r["emitted"]) == 0)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        no_primary_emit_rate = fp / max(1, fp + tn)
        return {"n_groups": n, "primary_groups": len(primary), "no_primary_groups": len(no_primary), "tp_emit": tp, "fn_no_emit": fn, "fp_emit": fp, "tn_no_emit": tn, "primary_recall": recall, "emit_precision": precision, "emit_f1": f1, "no_primary_emit_rate": no_primary_emit_rate, "group_emit_rate": sum(int(r["emitted"]) for r in subset) / max(1, n)}

    metrics_rows: List[Dict[str, Any]] = []
    all_metrics = metrics_for(group_replay)
    metrics_rows.append({"subset": "all", **all_metrics})
    for split in sorted({str(r["split"]) for r in group_replay}):
        metrics_rows.append({"subset": f"split={split}", **metrics_for([r for r in group_replay if r["split"] == split])})
    for suite in sorted({str(r["suite"]) for r in group_replay}):
        metrics_rows.append({"subset": f"suite={suite}", **metrics_for([r for r in group_replay if r["suite"] == suite])})
    for cls in sorted({str(r["group_class"]) for r in group_replay}):
        metrics_rows.append({"subset": f"group_class={cls}", **metrics_for([r for r in group_replay if r["group_class"] == cls])})
    metrics_rows.append({"subset": "imputed_group=true", **metrics_for([r for r in group_replay if int(r["imputed_row_count"]) > 0])})
    metrics_rows.append({"subset": "imputed_group=false", **metrics_for([r for r in group_replay if int(r["imputed_row_count"]) == 0])})

    violations: List[Dict[str, Any]] = []
    if all_metrics["primary_groups"] and all_metrics["primary_recall"] < args.min_primary_recall:
        add_violation(violations, "LOW_PRIMARY_GROUP_RECALL", "HOLD", f"primary_recall={all_metrics['primary_recall']}")
    if all_metrics["no_primary_groups"] and all_metrics["no_primary_emit_rate"] > args.max_no_primary_emit_rate:
        add_violation(violations, "HIGH_NO_PRIMARY_EMIT_RATE", "HOLD", f"no_primary_emit_rate={all_metrics['no_primary_emit_rate']}")
    for row in metrics_rows:
        if row["subset"] == "suite=libero_spatial" and int(row.get("primary_groups", 0)) and safe_num(row.get("primary_recall")) < args.min_spatial_primary_recall:
            add_violation(violations, "LOW_SPATIAL_PRIMARY_RECALL", "WARN", f"spatial_primary_recall={row.get('primary_recall')}")
    if d5_report:
        if int(d5_report.get("release_positive_count", 0) or 0) == 0:
            add_violation(violations, "RELEASE_HEAD_NO_POSITIVE_LABELS_IN_D5", "WARN", "runtime release gate is not validated by positive release labels")
        phase_pred_counts = d5_report.get("phase_pred_counts", {}) or {}
        nonzero_phase_preds = sum(1 for _k, v in phase_pred_counts.items() if int(v) > 0)
        if nonzero_phase_preds <= 2:
            add_violation(violations, "PHASE_HEAD_COLLAPSED_IN_D5", "WARN", f"nonzero_phase_predictions={nonzero_phase_preds}")

    hard = [v for v in violations if v["severity"] == "HOLD"]
    status = PASS if not hard else "HOLD_CLEAN2000_DETECTOR_V2_OFFLINE_FSM_ROW_REPLAY_AUDIT"
    reason = "" if not hard else f"hard_violation_count={len(hard)}"

    write_csv(out / "d6_offline_fsm_group_replay.csv", group_replay, ["group_key", "split", "suite", "group_class", "row_count", "has_primary_truth", "no_primary_truth", "emitted", "first_emit_label_row_id", "first_emit_step", "last_state", "max_corridor_p", "max_primary_p", "min_release_p", "imputed_row_count", "primary_label_rows", "auxiliary_label_rows", "no_event_label_rows"])
    write_csv(out / "d6_offline_fsm_row_replay.csv", row_replay, ["group_key", "local_row_index", "label_row_id", "record_id", "split", "suite", "teacher_label_status", "event_role_true", "feature_extraction_step", "runtime_state", "emitted", "arm_step", "emit_step", "corridor_p", "release_p", "primary_p", "pred_phase", "pred_event_role", "has_imputation"])
    write_csv(out / "d6_offline_fsm_metrics_by_suite_class.csv", metrics_rows, sorted({k for r in metrics_rows for k in r.keys()}))
    write_csv(out / "d6_offline_fsm_violations.csv", violations, ["violation_code", "severity", "detail"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "frozen_dataset": str(dataset_path),
        "frozen_dataset_sha256": sha256_file(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "d5_report": str(d5_report_path) if args.d5_report else "",
        "d5_report_sha256": sha256_file(d5_report_path) if args.d5_report else "",
        "expected_rows": args.expected_rows,
        "row_count": len(rows),
        "group_count": len(group_replay),
        "row_replay_count": len(row_replay),
        "thresholds": {"tau_corridor": args.tau_corridor, "tau_release": args.tau_release, "tau_primary": args.tau_primary, "guard": args.guard, "require_primary_event_role": not args.disable_primary_event_role_gate},
        "overall_replay_metrics": all_metrics,
        "metrics_by_subset": metrics_rows,
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "audit_thresholds": {"min_primary_recall": args.min_primary_recall, "min_spatial_primary_recall": args.min_spatial_primary_recall, "max_no_primary_emit_rate": args.max_no_primary_emit_rate},
        "interpretation": "CPU-only offline row-stream FSM replay over frozen detector feature rows. This is not LIBERO rollout and does not execute OpenVLA, env.step, intervention, or attack.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "device": "cpu", "OpenVLA_model": "NOT_LOADED", "model_inference": "DETECTOR_RUNTIME_ONLY_ON_FROZEN_CSV", "LIBERO_runtime": "NOT_PERFORMED", "env_reset": "NOT_PERFORMED", "env_set_init_state": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_detector_v2_offline_fsm_replay_audit_report.json", report)
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
    p.add_argument("--d5-report", default="")
    p.add_argument("--expected-rows", type=int, default=3717)
    p.add_argument("--tau-corridor", type=float, default=0.3)
    p.add_argument("--tau-release", type=float, default=0.3)
    p.add_argument("--tau-primary", type=float, default=0.5)
    p.add_argument("--guard", type=int, default=5)
    p.add_argument("--disable-primary-event-role-gate", action="store_true")
    p.add_argument("--min-primary-recall", type=float, default=0.50)
    p.add_argument("--min-spatial-primary-recall", type=float, default=0.40)
    p.add_argument("--max-no-primary-emit-rate", type=float, default=0.35)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
