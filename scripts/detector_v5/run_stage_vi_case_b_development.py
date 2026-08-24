#!/usr/bin/env python3
"""Train the single preregistered Case-B Stage VI development candidate."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.detector_v5 import run_stage_vi_root_cause_diagnostic as base  # noqa: E402

COUNTERS = dict(base.COUNTERS)
SPLITS = ("TRAIN", "VAL", "TEST")
DOSES = ("T3", "T5", "T10")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def model() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=0, class_weight=None),
    )


def causal_window(features: dict[tuple[str, int], np.ndarray], identity: str, step: int) -> np.ndarray:
    return np.concatenate([features.get((identity, past), np.zeros(25, dtype=np.float64)) for past in range(step - 15, step + 1)])


def records_metric(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("METRIC_ROW_SCORE_MISMATCH")
    records = [{"y": int(row["y"]), "score": float(scores[index]), "suite": str(row["suite"]), "split": str(row["split"]), "identity": str(row["canonical_parent_key"])} for index, row in enumerate(rows)]
    return {
        "row_count": len(records),
        "parent_count": len({row["identity"] for row in records}),
        "overall": base.metric([row["y"] for row in records], [row["score"] for row in records]),
        "per_suite": base.grouped_metrics(records, "suite"),
        "emission_rate_at_0.5": float(np.mean([row["score"] >= 0.5 for row in records])) if records else None,
    }


def split_metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    result = {}
    for split in SPLITS:
        selected = [index for index, row in enumerate(rows) if str(row["split"]) == split]
        result[split] = records_metric([rows[index] for index in selected], scores[selected])
    return result


def verify_inputs(args: argparse.Namespace, criteria: dict[str, Any], diagnostic: dict[str, Any], source_commit: str, source_tree: str) -> None:
    if criteria.get("status") != "FROZEN_BEFORE_CASE_B_DEVELOPMENT_SELECTION" or criteria.get("diagnostic_case") != "B":
        raise ValueError("CASE_B_CRITERIA_NOT_FROZEN")
    if diagnostic.get("classification", {}).get("case") != "B" or diagnostic.get("fresh_m4_execution") is not False or diagnostic.get("student_training_performed") is not False:
        raise ValueError("CASE_B_DIAGNOSTIC_BOUNDARY_INVALID")
    seal = read_json(args.diagnostic_root.resolve() / "ROOT_SEAL.json")
    if seal.get("status") != "PASS_READ_ONLY_RECONSTRUCTED_M4_DIAGNOSTIC" or seal.get("protected_counters") != COUNTERS or seal.get("eval160_status") != "UNREAD":
        raise ValueError("CASE_B_DIAGNOSTIC_SEAL_INVALID")
    if args.source_commit != source_commit or args.source_tree != source_tree:
        raise ValueError("SOURCE_COMMIT_OR_TREE_MISMATCH")


def evaluate_teacher(m4_labels: list[dict[str, Any]], privileged: dict[tuple[str, int], np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for dose in DOSES:
        rows = [row for row in m4_labels if str(row["dose"]) == dose]
        x = np.asarray([privileged[(str(row["canonical_parent_key"]), int(row["probe_step"]))] for row in rows], dtype=np.float64)
        y = np.asarray([int(row["y"]) for row in rows], dtype=np.int64)
        train = np.asarray([index for index, row in enumerate(rows) if str(row["split"]) == "TRAIN"], dtype=np.int64)
        if len(set(y[train].tolist())) != 2:
            raise ValueError(f"TEACHER_TRAIN_TARGET_DEGENERATE:{dose}")
        fitted = model()
        fitted.fit(x[train], y[train])
        models[dose] = fitted
        results[dose] = {
            "target": f"V_phys@{dose}",
            "input_schema": list(base.PRIVILEGED_FEATURE_NAMES),
            "fit_parent_count": len({str(rows[index]["canonical_parent_key"]) for index in train}),
            "fit_row_count": int(len(train)),
            "metrics": split_metrics(rows, fitted.predict_proba(x)[:, 1]),
            "abstain_rows_excluded_not_zero": True,
        }
    return results, models


def main_run(args: argparse.Namespace) -> dict[str, Any]:
    source_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip()
    source_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=REPO, text=True).strip()
    criteria = read_json(args.criteria_path.resolve())
    diagnostic = read_json(args.diagnostic_root.resolve() / "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC.json")
    verify_inputs(args, criteria, diagnostic, source_commit, source_tree)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    parent_splits = base.load_parent_splits()
    m4_labels = base.load_m4_labels(parent_splits)
    m4_student = base.load_m4_student(parent_splits, m4_labels)
    features = base.load_features()
    privileged, privileged_provenance = base.load_privileged_features(m4_labels)
    t5_rows = [row for row in m4_labels if str(row["dose"]) == "T5"]
    if len(t5_rows) != len(m4_student):
        raise ValueError("T5_LABEL_SCORE_ROW_COUNT_MISMATCH")
    for split in SPLITS:
        identities = {str(row["canonical_parent_key"]) for row in t5_rows if str(row["split"]) == split}
        if not identities:
            raise ValueError(f"EMPTY_PARENT_SPLIT:{split}")
    if len(set(parent_splits.values())) != 3:
        raise ValueError("PARENT_SPLIT_SCHEMA_INVALID")

    teacher_metrics, teacher_models = evaluate_teacher(m4_labels, privileged)
    teacher_t5 = teacher_models["T5"]
    x_student = np.asarray([causal_window(features, str(row["canonical_parent_key"]), int(row["probe_step"])) for row in t5_rows], dtype=np.float64)
    x_privileged = np.asarray([privileged[(str(row["canonical_parent_key"]), int(row["probe_step"]))] for row in t5_rows], dtype=np.float64)
    y = np.asarray([int(row["y"]) for row in t5_rows], dtype=np.int64)
    train_indices = np.asarray([index for index, row in enumerate(t5_rows) if str(row["split"]) == "TRAIN"], dtype=np.int64)
    teacher_pseudo = (teacher_t5.predict_proba(x_privileged[train_indices])[:, 1] >= 0.5).astype(np.int64)
    if len(set(teacher_pseudo.tolist())) != 2:
        raise ValueError("STUDENT_TEACHER_PSEUDO_TARGET_DEGENERATE")
    student = model()
    student.fit(x_student[train_indices], teacher_pseudo)
    student_scores = student.predict_proba(x_student)[:, 1]

    baseline_by_key = {(str(row["canonical_parent_key"]), int(row["probe_step"])): row for row in m4_student}
    baseline_rows = [row for row in t5_rows if (str(row["canonical_parent_key"]), int(row["probe_step"])) in baseline_by_key]
    baseline_scores = np.asarray([float(baseline_by_key[(str(row["canonical_parent_key"]), int(row["probe_step"]))]["scores"]["physical_criticality"]) for row in baseline_rows], dtype=np.float64)
    baseline_metrics = split_metrics(baseline_rows, baseline_scores)
    candidate_metrics = split_metrics(t5_rows, student_scores)
    dev = candidate_metrics["TEST"]
    baseline_dev = baseline_metrics["TEST"]
    criteria_gate = criteria["promotion_gate"]
    candidate_auroc = dev["overall"]["auroc"]
    baseline_auroc = baseline_dev["overall"]["auroc"]
    candidate_auprc = dev["overall"]["auprc"]
    baseline_auprc = baseline_dev["overall"]["auprc"]
    suite_aurocs = {suite: block["auroc"] for suite, block in dev["per_suite"].items()}
    checks = {
        "auroc_gain": {"value": float(candidate_auroc - baseline_auroc), "required": float(criteria_gate["minimum_auroc_absolute_gain"]), "pass": candidate_auroc >= baseline_auroc + float(criteria_gate["minimum_auroc_absolute_gain"])},
        "auprc_gain": {"value": float(candidate_auprc - baseline_auprc), "required": float(criteria_gate["minimum_auprc_absolute_gain"]), "pass": candidate_auprc >= baseline_auprc + float(criteria_gate["minimum_auprc_absolute_gain"])},
        "per_suite_auroc": {"values": suite_aurocs, "required_minimum": float(criteria_gate["minimum_per_suite_auroc"]), "required_suite_count": int(criteria_gate["minimum_suite_count"]), "pass": len(suite_aurocs) >= int(criteria_gate["minimum_suite_count"]) and all(value is not None and value >= float(criteria_gate["minimum_per_suite_auroc"]) for value in suite_aurocs.values())},
        "emission_coverage": {"value": dev["emission_rate_at_0.5"], "minimum": float(criteria_gate["minimum_emission_coverage"]), "maximum": float(criteria_gate["maximum_emission_coverage"]), "pass": float(criteria_gate["minimum_emission_coverage"]) <= dev["emission_rate_at_0.5"] <= float(criteria_gate["maximum_emission_coverage"])},
        "finite_metrics": {"pass": all(value is not None and np.isfinite(value) for value in (candidate_auroc, baseline_auroc, candidate_auprc, baseline_auprc, dev["emission_rate_at_0.5"]))},
    }
    promotion_pass = all(bool(check["pass"]) for check in checks.values())
    result = {
        "schema": "STAGE_VI_CASE_B_DEVELOPMENT_RESULT_V1",
        "status": "PASS_STAGE_VI_DEVELOPMENT_PROMOTION" if promotion_pass else "STAGE_VI_DEVELOPMENT_NO_IMPROVEMENT",
        "candidate": "S6-C_TV1_PRIVILEGED_LOGISTIC_PLUS_CAUSAL_STUDENT",
        "candidate_ordinal": 1,
        "diagnostic_case": "B",
        "teacher": teacher_metrics,
        "student": {
            "input_schema": list(base.FEATURE_NAMES) * 16,
            "context_steps": 16,
            "future_features": False,
            "privileged_features_at_inference": False,
            "teacher_pseudo_target_threshold": 0.5,
            "emission_threshold": 0.5,
            "fit_parent_count": len({str(t5_rows[index]["canonical_parent_key"]) for index in train_indices}),
            "fit_row_count": int(len(train_indices)),
            "teacher_pseudo_positive_rate": float(teacher_pseudo.mean()),
            "metrics": candidate_metrics,
        },
        "baseline_frozen_stage_v_student": baseline_metrics,
        "promotion_gate": {
            "criteria_sha256": sha256_file(args.criteria_path.resolve()),
            "development_check_split": "TEST",
            "checks": checks,
            "pass": promotion_pass,
            "next_authorized_state": "STAGE_VI_PRE_HOLDOUT_LOCK" if promotion_pass else "STAGE_VI_DEVELOPMENT_NO_IMPROVEMENT",
        },
        "split_contract": {"parent_splits": {split: sorted({str(row["canonical_parent_key"]) for row in t5_rows if str(row["split"]) == split}) for split in SPLITS}, "group_key": "canonical_parent_key", "individual_step_leakage": False, "fresh_stage_vi_holdout_used": False},
        "diagnostic_only_inputs": {"stage_v_development_outcomes_reused": True, "abstains_excluded_not_zero": True, "outcome_informed_model_selection": False},
        "protected_counters": COUNTERS,
        "eval160_status": "UNREAD",
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    output.mkdir(parents=True)
    with (output / "T_V_T5_PRIVILEGED_LOGISTIC.pkl").open("wb") as handle:
        pickle.dump(teacher_models["T5"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output / "S6_C_CAUSAL_STUDENT.pkl").open("wb") as handle:
        pickle.dump(student, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "DEVELOPMENT_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {
        "schema": "STAGE_VI_CASE_B_DEVELOPMENT_PROVENANCE_V1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_status": subprocess.check_output(("git", "status", "--porcelain"), cwd=REPO, text=True),
        "code_sha256": sha256_file(Path(__file__)),
        "criteria_path": str(args.criteria_path.resolve()),
        "criteria_sha256": sha256_file(args.criteria_path.resolve()),
        "diagnostic_root": str(args.diagnostic_root.resolve()),
        "diagnostic_root_seal_sha256": read_json(args.diagnostic_root.resolve() / "ROOT_SEAL.json")["sha256sums_sha256"],
        "m4_labels_sha256": sha256_file(base.M4_AGGREGATE / "M4_ALL_LABELS_V1.jsonl"),
        "m4_student_scores_sha256": sha256_file(base.M4_STUDENT_ROOT / "STUDENT_M4_PROBE_SCORES_V1.jsonl"),
        "m4_split_sha256": sha256_file(base.M4_SPLIT),
        "privileged_feature_provenance": privileged_provenance,
        "fresh_stage_vi_holdout_used": False,
        "protected_counters": COUNTERS,
        "eval160_status": "UNREAD",
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json"})
    (output / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    digest = sha256_file(output / "SHA256SUMS")
    (output / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    (output / "ROOT_SEAL.json").write_text(json.dumps({"schema": "STAGE_VI_CASE_B_DEVELOPMENT_ROOT_SEAL_V1", "status": result["status"], "sha256sums_sha256": digest, "protected_counters": COUNTERS, "eval160_status": "UNREAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria-path", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    try:
        result = main_run(args)
        print(json.dumps({"status": result["status"], "candidate": result["candidate"], "promotion": result["promotion_gate"], "output": str(args.output_dir.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
