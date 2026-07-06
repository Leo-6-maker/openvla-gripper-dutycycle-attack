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
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.sc5_multisuite_detector_runtime import (
    SC5_V2_EVENT_ROLES,
    SC5_V2_FEATURES,
    SC5_V2_PHASES,
    validate_no_forbidden_inputs,
)

GATE = "D4A_CLEAN2000_DETECTOR_V2_CPU_TRAINING_READINESS_AUDIT"
PASS = "PASS_CLEAN2000_DETECTOR_V2_CPU_TRAINING_READY"
OUT_FILES = [
    "detector_d4_training_readiness_report.json",
    "detector_d4_training_readiness_by_split_suite_label.csv",
    "detector_d4_training_readiness_imputation_distribution.csv",
    "detector_d4_training_readiness_violations.csv",
    "checksum_report.json",
]
ALLOWED_LABEL_STATUS = {"VALID_PRIMARY", "VALID_AUXILIARY", "NO_EVENT"}
EXPECTED_FEATURES = list(SC5_V2_FEATURES)


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def group_key(row: Dict[str, Any]) -> str:
    for key in ["group_key", "episode_key", "parent_id", "run_id", "record_id", "episode_id"]:
        if row.get(key):
            return str(row[key])
    return ""


def add_violation(rows: List[Dict[str, Any]], code: str, severity: str, detail: str, row_id: str = "") -> None:
    rows.append({"violation_code": code, "severity": severity, "detail": detail, "label_row_id": row_id})


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.frozen_dataset)
    rows = read_csv(dataset_path)
    d2f_report = json.loads(Path(args.d2f_report).read_text(encoding="utf-8")) if args.d2f_report else {}
    violations: List[Dict[str, Any]] = []

    try:
        validate_no_forbidden_inputs(EXPECTED_FEATURES)
    except Exception as exc:
        add_violation(violations, "FORBIDDEN_FEATURE_COLUMNS", "HOLD", str(exc))

    if len(rows) != args.expected_rows:
        add_violation(violations, "FROZEN_ROW_COUNT_MISMATCH", "HOLD", f"rows={len(rows)} expected={args.expected_rows}")

    if d2f_report:
        status = str(d2f_report.get("status", ""))
        if status != "PASS_CLEAN2000_DETECTOR_DATASET_V2_FROZEN":
            add_violation(violations, "D2F_REPORT_NOT_PASS", "HOLD", status)

    missing_columns = [c for c in EXPECTED_FEATURES if rows and c not in rows[0]]
    if missing_columns:
        add_violation(violations, "MISSING_FEATURE_COLUMNS", "HOLD", ";".join(missing_columns))

    seen_label_ids = set()
    duplicate_label_ids = 0
    group_to_split: Dict[str, str] = {}
    leakage_count = 0
    split_counts = Counter()
    suite_counts = Counter()
    label_counts = Counter()
    role_counts = Counter()
    imputed_row_ids = set()
    imputed_feature_total = 0
    nonfinite_feature_count = 0
    invalid_label_count = 0
    invalid_phase_count = 0
    invalid_role_count = 0
    empty_group_count = 0

    by_combo = Counter()
    imputation_combo = Counter()
    for i, row in enumerate(rows):
        lid = str(row.get("label_row_id") or f"row_{i:06d}")
        if lid in seen_label_ids:
            duplicate_label_ids += 1
            add_violation(violations, "DUPLICATE_LABEL_ROW_ID", "HOLD", lid, lid)
        seen_label_ids.add(lid)
        split = str(row.get("split", ""))
        suite = str(row.get("suite", ""))
        label = str(row.get("teacher_label_status", row.get("label_status", "")))
        phase = str(row.get("phase_label", row.get("phase", "")))
        role = str(row.get("event_role", ""))
        split_counts[split] += 1
        suite_counts[suite] += 1
        label_counts[label] += 1
        role_counts[role] += 1
        by_combo[(split, suite, label, role)] += 1
        if split not in {"train", "val", "test"}:
            add_violation(violations, "INVALID_SPLIT", "HOLD", split, lid)
        if label not in ALLOWED_LABEL_STATUS:
            invalid_label_count += 1
            add_violation(violations, "INVALID_TEACHER_LABEL_STATUS", "HOLD", label, lid)
        if phase not in SC5_V2_PHASES:
            invalid_phase_count += 1
            add_violation(violations, "INVALID_PHASE_LABEL", "HOLD", phase, lid)
        if role not in SC5_V2_EVENT_ROLES:
            invalid_role_count += 1
            add_violation(violations, "INVALID_EVENT_ROLE", "HOLD", role, lid)
        g = group_key(row)
        if not g:
            empty_group_count += 1
            add_violation(violations, "EMPTY_GROUP_KEY", "HOLD", lid, lid)
        elif g in group_to_split and group_to_split[g] != split:
            leakage_count += 1
            add_violation(violations, "GROUP_SPLIT_LEAKAGE", "HOLD", f"group={g} {group_to_split[g]} vs {split}", lid)
        else:
            group_to_split[g] = split
        for feat in EXPECTED_FEATURES:
            if not finite(row.get(feat)):
                nonfinite_feature_count += 1
                add_violation(violations, "NONFINITE_FEATURE_VALUE", "HOLD", feat, lid)
        has_imp = truthy(row.get("has_imputation")) or int(float(row.get("imputed_feature_count", "0") or 0)) > 0 if str(row.get("imputed_feature_count", "0") or "0").replace(".", "", 1).isdigit() else truthy(row.get("has_imputation"))
        imp_count = 0
        try:
            imp_count = int(float(row.get("imputed_feature_count", 0) or 0))
        except Exception:
            imp_count = 0
        if has_imp or imp_count > 0:
            imputed_row_ids.add(lid)
            imputed_feature_total += imp_count
            imputation_combo[(split, suite, label, role)] += 1

    for split in ["train", "val", "test"]:
        if split_counts.get(split, 0) == 0:
            add_violation(violations, "EMPTY_SPLIT", "HOLD", split)
        for label in ["VALID_PRIMARY", "VALID_AUXILIARY", "NO_EVENT"]:
            if by_combo[(split, "libero_10", label, "primary_attackable")] + sum(v for (s, _suite, lab, _role), v in by_combo.items() if s == split and lab == label) == 0:
                add_violation(violations, "SPLIT_LABEL_CLASS_ABSENT", "WARN", f"split={split} label={label}")
    for split in ["train", "val", "test"]:
        imp_in_split = sum(v for (s, _suite, _label, _role), v in imputation_combo.items() if s == split)
        if len(imputed_row_ids) and imp_in_split == 0:
            add_violation(violations, "IMPUTED_ROWS_ABSENT_FROM_SPLIT", "HOLD", split)

    combo_rows = [
        {"split": s, "suite": suite, "teacher_label_status": lab, "event_role": role, "count": count}
        for (s, suite, lab, role), count in sorted(by_combo.items())
    ]
    imp_rows = [
        {"split": s, "suite": suite, "teacher_label_status": lab, "event_role": role, "imputed_row_count": count}
        for (s, suite, lab, role), count in sorted(imputation_combo.items())
    ]
    hard = [v for v in violations if v.get("severity") == "HOLD"]
    status = PASS if not hard else "HOLD_CLEAN2000_DETECTOR_V2_CPU_TRAINING_NOT_READY"
    reason = "" if not hard else f"hard_violation_count={len(hard)}"

    write_csv(out / "detector_d4_training_readiness_by_split_suite_label.csv", combo_rows, ["split", "suite", "teacher_label_status", "event_role", "count"])
    write_csv(out / "detector_d4_training_readiness_imputation_distribution.csv", imp_rows, ["split", "suite", "teacher_label_status", "event_role", "imputed_row_count"])
    write_csv(out / "detector_d4_training_readiness_violations.csv", violations, ["violation_code", "severity", "detail", "label_row_id"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "frozen_dataset": str(dataset_path),
        "frozen_dataset_sha256": sha256_file(dataset_path),
        "d2f_report": args.d2f_report,
        "d2f_report_sha256": sha256_file(Path(args.d2f_report)) if args.d2f_report else "",
        "expected_rows": args.expected_rows,
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "suite_counts": dict(suite_counts),
        "teacher_label_status_counts": dict(label_counts),
        "event_role_counts": dict(role_counts),
        "feature_columns": EXPECTED_FEATURES,
        "feature_column_count": len(EXPECTED_FEATURES),
        "duplicate_label_row_id_count": duplicate_label_ids,
        "group_leakage_count": leakage_count,
        "empty_group_key_count": empty_group_count,
        "nonfinite_feature_count": nonfinite_feature_count,
        "invalid_label_count": invalid_label_count,
        "invalid_phase_count": invalid_phase_count,
        "invalid_role_count": invalid_role_count,
        "imputed_row_count": len(imputed_row_ids),
        "imputed_feature_count": imputed_feature_total,
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "training_release_scope": "D4 CPU-only detector training readiness. This audit does not train and does not authorize GPU/LIBERO/rollout/intervention/attack.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "detector_d4_training_readiness_report.json", report)
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
    p.add_argument("--d2f-report", default="")
    p.add_argument("--expected-rows", type=int, default=3717)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
