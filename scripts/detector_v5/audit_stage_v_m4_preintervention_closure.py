#!/usr/bin/env python3
"""Read-only closure audit for a structurally failed formal-M4 parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
ARMS = ("CONTROL", "T3", "T5", "T10")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}:{line_number}")
        rows.append(dict(value))
    return rows


def _list_field(value: Any, *, field: str, branch_id: str, errors: list[str]) -> list[Any]:
    if value is None:
        errors.append(f"{field}_MISSING:{branch_id}")
        return []
    if not isinstance(value, list):
        errors.append(f"{field}_NOT_LIST:{branch_id}")
        return []
    return value


def _branch_summary(row: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    branch = row.get("branch") if isinstance(row.get("branch"), Mapping) else {}
    branch_id = str(row.get("branch_id", ""))
    arm = str(row.get("arm", branch.get("arm", "")))
    if arm not in ARMS:
        errors.append(f"UNKNOWN_ARM:{branch_id}")
    if branch.get("protected_counters") != COUNTERS:
        errors.append(f"BRANCH_PROTECTED_COUNTERS:{branch_id}")
    rows = _list_field(branch.get("rows"), field="ROWS", branch_id=branch_id, errors=errors)
    actions = _list_field(branch.get("actions"), field="ACTIONS", branch_id=branch_id, errors=errors)
    receipts = _list_field(branch.get("treatment_receipts"), field="TREATMENT_RECEIPTS", branch_id=branch_id, errors=errors)
    treatment_steps = 0
    if arm != "CONTROL":
        try:
            dose = int(branch.get("dose_steps", row.get("dose_steps", 0)))
        except (TypeError, ValueError):
            dose = 0
            errors.append(f"DOSE_NOT_INT:{branch_id}")
        for item in rows:
            if isinstance(item, Mapping):
                try:
                    if int(item.get("relative_step", -1)) < dose:
                        treatment_steps += 1
                except (TypeError, ValueError):
                    errors.append(f"ROW_RELATIVE_STEP_INVALID:{branch_id}")
    nonzero_physical_evidence = bool(rows or actions or receipts)
    treatment_compliant = branch.get("treatment_compliant") is True
    error = str(branch.get("error", ""))
    if error == "CausalSnapshotError:EXACT_BINDING_MISMATCH:runtime_state":
        failure_stage = "PRE_PRIMARY_WINDOW_RUNTIME_EXACTNESS"
    elif branch.get("status") == "FAIL" and nonzero_physical_evidence:
        failure_stage = "PRIMARY_WINDOW_OR_POST_FAILURE"
    elif branch.get("status") == "FAIL":
        failure_stage = "PRE_PRIMARY_WINDOW_UNKNOWN"
    else:
        failure_stage = "PRIMARY_WINDOW_EXECUTED_OR_NONFAIL"
    return {
        "branch_id": branch_id,
        "probe_id": str(row.get("probe_id", branch.get("probe_id", ""))),
        "arm": arm,
        "status": branch.get("status"),
        "error": error,
        "failure_stage": failure_stage,
        "rows_count": len(rows),
        "actions_count": len(actions),
        "treatment_receipts_count": len(receipts),
        "treatment_window_rows_count": treatment_steps,
        "treatment_compliant": treatment_compliant,
        "state_restore_exact": branch.get("state_restore_exact"),
        "runtime_state_exact": branch.get("runtime_state_exact"),
        "causal_input_binding_pass": branch.get("causal_input_binding_pass"),
        "physical_step_evidence": nonzero_physical_evidence,
        "physical_intervention_evidence": arm != "CONTROL" and nonzero_physical_evidence,
    }


def audit(parent_root: Path, output_root: Path, parent_key: str, gate_b_runner: Path, expected_branch_count: int = 96, auditor_source_commit: str = "") -> dict[str, Any]:
    parent_root = parent_root.resolve()
    output_root = output_root.resolve()
    if not parent_root.is_dir():
        raise FileNotFoundError(parent_root)
    gate_b_runner = gate_b_runner.resolve()
    if not gate_b_runner.is_file():
        raise FileNotFoundError(f"GATE_B_RUNNER_MISSING:{gate_b_runner}")
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_ROOT_EXISTS:{output_root}")

    branch_path = parent_root / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl"
    label_path = parent_root / "M4_V_PHYS_LABELS_V1.jsonl"
    observation_path = parent_root / "M4_TREATMENT_OBSERVATIONS_V1.jsonl"
    required = (branch_path, label_path, observation_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("MISSING_INPUT:" + ",".join(missing))

    branches = _jsonl(branch_path)
    labels = _jsonl(label_path)
    observations = _jsonl(observation_path)
    errors: list[str] = []
    summaries = [_branch_summary(row, errors) for row in branches]

    identities = {(item["probe_id"], item["arm"]) for item in summaries}
    expected_identities = {(f"Q{i:02d}", arm) for i in range(24) for arm in ARMS}
    if len(branches) != expected_branch_count:
        errors.append(f"BRANCH_COUNT:{len(branches)}")
    if identities != expected_identities:
        errors.append("BRANCH_IDENTITY_COVERAGE")
    if len(labels) != 72:
        errors.append(f"LABEL_COUNT:{len(labels)}")
    if len(observations) != 72:
        errors.append(f"OBSERVATION_COUNT:{len(observations)}")
    if any(item.get("protected_counters") != COUNTERS for item in branches + labels + observations):
        errors.append("PROTECTED_COUNTERS")

    binary_labels = [row for row in labels if row.get("binary_label_consumable") is True]
    treatment = [item for item in summaries if item["arm"] != "CONTROL"]
    control = [item for item in summaries if item["arm"] == "CONTROL"]
    physical_intervention_evidence = any(item["physical_intervention_evidence"] for item in summaries)
    all_failed_before_action = bool(summaries) and all(
        item["arm"] in ARMS
        and item["status"] == "FAIL"
        and item["failure_stage"] == "PRE_PRIMARY_WINDOW_RUNTIME_EXACTNESS"
        and item["rows_count"] == 0
        and item["actions_count"] == 0
        and item["treatment_receipts_count"] == 0
        and item["state_restore_exact"] is False
        and item["runtime_state_exact"] is False
        and item["causal_input_binding_pass"] is False
        for item in summaries
    )
    if physical_intervention_evidence:
        status = "HOLD_PHYSICAL_ACTION_EVIDENCE_NONZERO"
    elif not all_failed_before_action or errors:
        status = "HOLD_CLOSURE_INCOMPLETE"
    else:
        status = "PASS_PREINTERVENTION_STRUCTURAL_INVALIDATION"

    parent_files = {path.name: _sha(path) for path in required}
    for name in ("PARENT_RESULT.json", "M4_INDEPENDENT_AUDIT.json", "SHA256SUMS", "SHA256SUMS.sha256"):
        path = parent_root / name
        if path.is_file():
            parent_files[name] = _sha(path)
    report: dict[str, Any] = {
        "schema": "STAGE_V_M4_PREINTERVENTION_STRUCTURAL_INVALIDATION_AUDIT_V1",
        "status": status,
        "sealed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_key": parent_key,
        "auditor_source_commit": auditor_source_commit,
        "parent_root": str(parent_root),
        "parent_root_inputs_sha256": parent_files,
        "gate_b_runner_path": str(gate_b_runner),
        "gate_b_runner_sha256": _sha(gate_b_runner),
        "branch_records_materialized": True,
        "branch_record_count": len(branches),
        "observation_record_count": len(observations),
        "label_record_count": len(labels),
        "rows_total": sum(item["rows_count"] for item in summaries),
        "actions_total": sum(item["actions_count"] for item in summaries),
        "treatment_receipts_total": sum(item["treatment_receipts_count"] for item in treatment),
        "successful_env_steps_evidenced_total": sum(item["rows_count"] for item in summaries),
        "physical_env_steps_total": sum(item["rows_count"] for item in summaries),
        "treatment_window_env_steps_evidenced_total": sum(item["treatment_window_rows_count"] for item in treatment),
        "delivered_treatment_step_count": sum(item["treatment_window_rows_count"] for item in treatment),
        "treatment_action_rows_total": sum(item["actions_count"] for item in treatment),
        "post_snapshot_primary_window_steps_total": sum(item["rows_count"] for item in summaries),
        "forced_open_steps_total": sum(item["treatment_window_rows_count"] for item in treatment),
        "control_primary_window_steps_total": sum(item["rows_count"] for item in control),
        "treatment_primary_window_steps_total": sum(item["rows_count"] for item in treatment),
        "treatment_compliant_branch_count": sum(item["treatment_compliant"] for item in treatment),
        "binary_label_consumable_count": len(binary_labels),
        "control_rows_total": sum(item["rows_count"] for item in control),
        "treatment_rows_total": sum(item["rows_count"] for item in treatment),
        "physical_intervention_executed": physical_intervention_evidence,
        "v_phys_binary_consumable": bool(binary_labels),
        "outcome_artifacts_materialized": True,
        "selection_outcomes_read": False,
        "protected_counters": dict(COUNTERS),
        "closure_rule": {
            "env_step_evidence": "one recorded branch row is counted as one post-env.step evidence row",
            "treatment_action_evidence": "any non-CONTROL branch row, action, or treatment receipt is conservative physical-intervention evidence",
            "binary_label_rule": "only binary_label_consumable=true is counted as consumable",
        },
        "all_branches_failed_before_action": all_failed_before_action,
        "pass_branch_count": sum(item["status"] == "PASS" for item in summaries),
        "fail_branch_count": sum(item["status"] == "FAIL" for item in summaries),
        "pre_primary_restore_failure_count": sum(item["failure_stage"] == "PRE_PRIMARY_WINDOW_RUNTIME_EXACTNESS" for item in summaries),
        "failure_stage_counts": {
            stage: sum(item["failure_stage"] == stage for item in summaries)
            for stage in sorted({item["failure_stage"] for item in summaries})
        },
        "errors": sorted(set(errors)),
        "branches": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    report_path = output_root / "PREINTERVENTION_STRUCTURAL_INVALIDATION_AUDIT.json"
    report["sealed"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text(f"{_sha(report_path)}  {report_path.name}\n", encoding="utf-8")
    (output_root / "SHA256SUMS.sha256").write_text(f"{_sha(sums_path)}  SHA256SUMS\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--gate-b-runner", type=Path, required=True)
    parser.add_argument("--auditor-source-commit", required=True)
    parser.add_argument("--expected-branch-count", type=int, default=96)
    args = parser.parse_args(argv)
    try:
        report = audit(args.parent_root, args.output_root, args.parent_key, args.gate_b_runner, args.expected_branch_count, args.auditor_source_commit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD_CLOSURE_AUDIT_ERROR", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2
    print(json.dumps({key: report[key] for key in ("status", "rows_total", "actions_total", "treatment_receipts_total", "physical_env_steps_total", "delivered_treatment_step_count", "treatment_action_rows_total", "post_snapshot_primary_window_steps_total", "forced_open_steps_total", "control_primary_window_steps_total", "treatment_primary_window_steps_total", "successful_env_steps_evidenced_total", "treatment_window_env_steps_evidenced_total", "treatment_compliant_branch_count", "binary_label_consumable_count", "physical_intervention_executed", "pre_primary_restore_failure_count", "failure_stage_counts", "errors")}, sort_keys=True))
    return 0 if report["status"] == "PASS_PREINTERVENTION_STRUCTURAL_INVALIDATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
