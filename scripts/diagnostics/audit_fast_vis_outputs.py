#!/usr/bin/env python3
"""Audit Fast VIS cascade output schemas without running rollout/VIS."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


COMMON_REQUIRED = {
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "label",
    "label_source",
    "label_confidence",
    "gpu_pair",
    "runtime_sec",
    "provenance_status",
}

COMMAND_PROXY_REQUIRED = {
    "measurement_version",
    "action_injection_version",
    "gripper_qpos_source",
    "gripper_qpos_mujoco",
    "gripper_qpos_obs",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "forced_open_value_used",
    "post_transform_gripper_action",
    "clean_gripper_action",
    "forced_gripper_action",
}

LOW_BUDGET_REQUIRED = {
    "action_transform_version",
    "phase_alignment_source",
    "qpos_phase_class",
    "mechanism_status",
    "epsilon_calibration",
    "arm_l2_mean",
    "arm_l2_max",
    "token_flip_count",
    "qpos_opening_delta_mujoco",
    "env_action_gripper_open_count",
    "raw_clean_action_gripper",
    "raw_adv_action_gripper",
    "env_clean_action_gripper_after_transform",
    "env_adv_action_gripper_after_transform",
    "post_transform_gripper_action",
    "gripper_qpos_mujoco",
    "gripper_qpos_obs",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "gripper_qpos_warning",
    "previous_phase_e_v0_status",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-only", default="tables/fast_vis_policy_only_audit_v0.csv")
    ap.add_argument("--command-proxy", default="tables/fast_vis_command_proxy_v0.csv")
    ap.add_argument("--low-budget", default="tables/fast_vis_low_budget_sweep_v0.csv")
    ap.add_argument("--output-report", default="reports/FAST_VIS_OUTPUT_SCHEMA_AUDIT.md")
    ap.add_argument("--output-csv", default="tables/fast_vis_output_schema_audit_v0.csv")
    return ap.parse_args()


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [str(field or "").strip().lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({str(k or "").strip().lstrip("\ufeff"): v for k, v in row.items()})
        return fields, rows


def has_value(row, col: str) -> bool:
    return str(row.get(col, "")).strip() != ""


def row_is_infra_failed(row) -> bool:
    text = " ".join(str(row.get(k, "")) for k in row.keys()).lower()
    return any(tok in text for tok in ["infra_failed", "xid", "out of memory", " oom", "cuda illegal", "cublas"])


def low_budget_silver_eligible(row) -> bool:
    return (
        str(row.get("label_confidence", "")).strip().lower() == "silver_candidate"
        and str(row.get("mechanism_status", "")).strip().lower() == "mechanism_clean"
        and not row_is_infra_failed(row)
        and str(row.get("qpos_phase_class", "")).strip().lower() != "natural_open"
    )


def audit_one(name: str, path: str):
    result = {
        "dataset": name,
        "path": path,
        "status": "ok",
        "row_count": 0,
        "issues": [],
    }
    if not os.path.exists(path):
        result["status"] = "BLOCKED_MISSING_FAST_VIS_OUTPUTS"
        result["issues"].append("missing_csv")
        return result

    fieldnames, rows = read_rows(path)
    result["row_count"] = len(rows)
    fields = set(fieldnames)
    required = set(COMMON_REQUIRED)
    required.add("denominator_status")
    if name == "command_proxy":
        required.update(COMMAND_PROXY_REQUIRED)
    if name == "low_budget":
        required.update(LOW_BUDGET_REQUIRED)
    missing = sorted(required - fields)
    if missing:
        result["issues"].append("missing_required_columns:" + ",".join(missing))

    if rows:
        value_cols = ["gpu_pair", "runtime_sec", "provenance_status", "label_source", "label_confidence"]
        if name == "command_proxy":
            value_cols.extend([
                "measurement_version",
                "action_injection_version",
                "gripper_qpos_source",
                "gripper_qpos_used",
                "gripper_qpos_source_priority",
                "forced_open_value_used",
                "post_transform_gripper_action",
            ])
        if name == "low_budget":
            value_cols.extend([
                "action_transform_version",
                "phase_alignment_source",
                "qpos_phase_class",
                "mechanism_status",
                "epsilon_calibration",
                "arm_l2_mean",
                "arm_l2_max",
                "token_flip_count",
                "qpos_opening_delta_mujoco",
                "env_action_gripper_open_count",
                "post_transform_gripper_action",
                "gripper_qpos_used",
                "gripper_qpos_source_priority",
                "previous_phase_e_v0_status",
            ])
        for col in value_cols:
            if col not in fields:
                continue
            missing_values = sum(1 for r in rows if not has_value(r, col))
            if missing_values:
                result["issues"].append(f"missing_values:{col}:{missing_values}")

        if "denominator_status" in fields:
            missing_den = sum(1 for r in rows if not has_value(r, "denominator_status"))
            if missing_den:
                result["issues"].append(f"missing_values:denominator_status:{missing_den}")

        proxy_gold = 0
        for r in rows:
            confidence = str(r.get("label_confidence", "")).lower()
            source = str(r.get("label_source", "")).lower()
            if "proxy" in confidence and "gold" in confidence:
                proxy_gold += 1
            if "proxy" in source and "gold" in confidence:
                proxy_gold += 1
        if proxy_gold:
            result["issues"].append(f"proxy_label_marked_gold:{proxy_gold}")

        infra_as_label = 0
        for r in rows:
            if not row_is_infra_failed(r):
                continue
            confidence = str(r.get("label_confidence", "")).lower()
            source = str(r.get("label_source", "")).lower()
            if confidence and "not_label" not in confidence and "reference_only" not in source:
                infra_as_label += 1
        if infra_as_label:
            result["issues"].append(f"infra_failed_row_treated_as_label:{infra_as_label}")

        measurement_as_label = 0
        for r in rows:
            if "measurement_failed" not in str(r.get("provenance_status", "")).lower():
                continue
            confidence = str(r.get("label_confidence", "")).lower()
            source = str(r.get("label_source", "")).lower()
            if confidence and "not_label" not in confidence and "reference_only" not in source:
                measurement_as_label += 1
        if measurement_as_label:
            result["issues"].append(f"measurement_failed_row_treated_as_label:{measurement_as_label}")

        if name == "low_budget":
            gold_confidence = sum(1 for r in rows if "gold" in str(r.get("label_confidence", "")).lower())
            if gold_confidence:
                result["issues"].append(f"low_budget_label_confidence_gold:{gold_confidence}")

            non_clean_silver = sum(
                1
                for r in rows
                if str(r.get("label_confidence", "")).strip().lower() == "silver_candidate"
                and str(r.get("mechanism_status", "")).strip().lower() != "mechanism_clean"
            )
            if non_clean_silver:
                result["issues"].append(f"silver_candidate_without_mechanism_clean:{non_clean_silver}")

            train_silver = sum(
                1
                for r in rows
                if str(r.get("label_confidence", "")).strip().lower() == "silver_candidate"
                and str(r.get("label_use", "")).strip().lower() in {"train", "training", "1", "true", "yes"}
            )
            if train_silver:
                result["issues"].append(f"silver_candidate_marked_train_label:{train_silver}")

            infra_metric = sum(
                1
                for r in rows
                if row_is_infra_failed(r)
                and str(r.get("count_toward_metrics", "")).strip().lower() in {"1", "true", "yes"}
            )
            if infra_metric:
                result["issues"].append(f"infra_failed_counted_toward_metrics:{infra_metric}")

            phase_misaligned_metric = sum(
                1
                for r in rows
                if str(r.get("mechanism_status", "")).strip().lower() == "phase_misaligned"
                and str(r.get("count_toward_metrics", "")).strip().lower() in {"1", "true", "yes"}
            )
            if phase_misaligned_metric:
                result["issues"].append(f"phase_misaligned_counted_as_low_budget_result:{phase_misaligned_metric}")

            result["silver_candidate_eligible_rows"] = sum(1 for r in rows if low_budget_silver_eligible(r))

    if result["issues"]:
        result["status"] = "FAIL"
    return result


def write_csv(path: str, audits):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["dataset", "path", "status", "row_count", "silver_candidate_eligible_rows", "issues"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in audits:
            row = dict(a)
            row.setdefault("silver_candidate_eligible_rows", "")
            row["issues"] = ";".join(a["issues"])
            w.writerow(row)


def write_report(path: str, audits):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    overall = "PASS"
    if any(a["status"] == "FAIL" for a in audits):
        overall = "FAIL"
    elif any(a["status"] == "BLOCKED_MISSING_FAST_VIS_OUTPUTS" for a in audits):
        overall = "BLOCKED_MISSING_FAST_VIS_OUTPUTS"

    lines = [
        "# Fast VIS Output Schema Audit",
        "",
        f"**Overall status**: {overall}",
        "",
        "This is a CPU-only schema audit. It does not run rollout, VIS, watcher, or detector training.",
        "",
        "## Dataset Status",
        "",
        "| Dataset | Rows | Silver-eligible | Status | Issues |",
        "|---|---:|---:|---|---|",
    ]
    for a in audits:
        issues = "; ".join(a["issues"]) if a["issues"] else "none"
        eligible = a.get("silver_candidate_eligible_rows", "")
        lines.append(f"| {a['dataset']} | {a['row_count']} | {eligible} | {a['status']} | {issues} |")

    lines.extend([
        "",
        "## Checks",
        "",
        "- Required columns: task_key, state_id, window_start, window_end, label, label_source, label_confidence, gpu_pair, runtime_sec, provenance_status.",
        "- Command-proxy additionally requires measurement_version, action_injection_version, gripper_qpos_source, gripper_qpos_mujoco, gripper_qpos_obs, gripper_qpos_used, gripper_qpos_source_priority, forced_open_value_used, post_transform_gripper_action, clean_gripper_action, and forced_gripper_action.",
        "- Low-budget VIS additionally requires action_transform_version, phase_alignment_source, qpos_phase_class, mechanism_status, epsilon_calibration, arm/token/qpos mechanism fields, raw/env gripper action transform fields, MuJoCo-primary qpos audit fields, and previous_phase_e_v0_status.",
        "- denominator_status is required, including explicit not_applicable values for policy-only and command-proxy outputs.",
        "- Proxy labels must not be marked gold.",
        "- Low-budget label_confidence cannot be gold.",
        "- Low-budget silver_candidate rows are not train labels.",
        "- mechanism_status other than mechanism_clean means the row is not usable as silver_candidate.",
        "- INFRA_FAILED rows cannot count toward metrics.",
        "- phase_misaligned rows cannot count as low-budget failure/success.",
        "- Rows with INFRA_FAILED/Xid/OOM/CUDA failures must not be treated as trainable labels.",
        "- Rows with MEASUREMENT_FAILED must not be treated as proxy labels.",
        "",
        "## Claim Boundary",
        "",
        "- Missing output CSVs are reported as BLOCKED_MISSING_FAST_VIS_OUTPUTS, not as failed experiments.",
        "- Policy-only and command-open proxy results are screening/proxy evidence only, not gold VIS labels.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    audits = [
        audit_one("policy_only", args.policy_only),
        audit_one("command_proxy", args.command_proxy),
        audit_one("low_budget", args.low_budget),
    ]
    write_csv(args.output_csv, audits)
    write_report(args.output_report, audits)
    for a in audits:
        print(f"{a['dataset']}: {a['status']} rows={a['row_count']} issues={';'.join(a['issues']) or 'none'}")
    return 1 if any(a["status"] == "FAIL" for a in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
