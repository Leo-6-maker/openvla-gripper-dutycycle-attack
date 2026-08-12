#!/usr/bin/env python3
"""Independent, producer-free audit for one M4 matched-action parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
ARMS = ("CONTROL", "T3", "T5", "T10")
DOSES = {"T3": 3, "T5": 5, "T10": 10}
BINARY = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}")
            rows.append(dict(value))
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth_label(control_valid: Any, treatment_valid: Any, f_control: Any, f_open: Any) -> str:
    if f_control == 1:
        return "CONTROL_CONTAMINATION_ABSTAIN" if f_open == 1 else "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    if control_valid is not True:
        return "CONTROL_INVALID_ABSTAIN"
    if treatment_valid is not True:
        return "TREATMENT_INVALID_ABSTAIN"
    if f_control is None or f_open is None:
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    return "V_PHYS" if f_open == 1 else "NO_PHYSICAL_VULNERABILITY"


def audit(root: Path, parent_key: str, source_commit: str, source_tree: str) -> dict[str, Any]:
    errors: list[str] = []
    result_path = root / "PARENT_RESULT.json"
    result = _load(result_path) if result_path.is_file() else {}
    if result.get("schema") != "STAGE_V_M4_PARENT_RESULT_V1": errors.append("RESULT_SCHEMA")
    if result.get("status") != "PASS": errors.append("RESULT_STATUS")
    if result.get("canonical_parent_key") != parent_key: errors.append("PARENT_IDENTITY")
    if result.get("source_commit") != source_commit: errors.append("SOURCE_COMMIT")
    if result.get("source_tree") != source_tree: errors.append("SOURCE_TREE")
    if result.get("probe_count") != 24 or result.get("branch_count") != 96 or result.get("treatment_label_count") != 72: errors.append("ACCOUNTING")
    if result.get("selection_outcomes_read") is not False: errors.append("SELECTION_OUTCOME_LEAKAGE")
    if result.get("fresh_render_primary_consumption") is not False or result.get("native_policy_calls_in_primary_window") != 0: errors.append("PRIMARY_RENDER_OR_POLICY_VIOLATION")
    if result.get("protected_counters") != COUNTERS: errors.append("RESULT_COUNTERS")
    clean = _load(root / "CLEAN_TRAJECTORY_V1_4.json") if (root / "CLEAN_TRAJECTORY_V1_4.json").is_file() else {}
    plan = _load(root / "PROBE_PLAN_V1_4.json") if (root / "PROBE_PLAN_V1_4.json").is_file() else {}
    canary = _load(root / "M4_CAUSAL_SNAPSHOT_CANARY.json") if (root / "M4_CAUSAL_SNAPSHOT_CANARY.json").is_file() else {}
    if clean.get("outcomes_read") is not False: errors.append("CLEAN_OUTCOME_FLAG")
    if plan.get("outcomes_read") is not False: errors.append("PLAN_OUTCOME_FLAG")
    if canary.get("status") != "PASS" or canary.get("fresh_render_equality_gate_used") is not False or len(canary.get("canaries", [])) != 24: errors.append("CANARY")
    branch_path = root / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl"
    label_path = root / "M4_V_PHYS_LABELS_V1.jsonl"
    observation_path = root / "M4_TREATMENT_OBSERVATIONS_V1.jsonl"
    branches = _jsonl(branch_path) if branch_path.is_file() else []
    labels = _jsonl(label_path) if label_path.is_file() else []
    observations = _jsonl(observation_path) if observation_path.is_file() else []
    if len(branches) != 96: errors.append("BRANCH_COUNT")
    if len(labels) != 72: errors.append("LABEL_COUNT")
    if len(observations) != 72: errors.append("OBSERVATION_COUNT")
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in branches:
        probe = str(row.get("probe_id")); arm = str(row.get("arm")); branch = row.get("branch") if isinstance(row.get("branch"), Mapping) else {}
        key = (probe, arm)
        if row.get("schema") != "STAGE_V_M4_PHYSICAL_EXECUTION_V1" or key in by_identity: errors.append("BRANCH_SCHEMA_OR_DUPLICATE")
        by_identity[key] = row
        if arm not in ARMS or not probe: errors.append("BRANCH_IDENTITY")
        if row.get("protected_counters") != COUNTERS or branch.get("status") != "PASS": errors.append("BRANCH_STATUS_OR_COUNTERS")
        if branch.get("state_restore_exact") is not True or branch.get("runtime_state_exact") is not True or branch.get("causal_input_binding_pass") is not True: errors.append("BRANCH_EXACT_BINDING")
        if branch.get("primary_input_authority") != "loaded_frozen_canonical_bytes" or branch.get("fresh_render_equality_gate_used") is not False or branch.get("fresh_render_primary_consumption") is not False or branch.get("native_policy_calls_in_primary_window") != 0: errors.append("BRANCH_PRIMARY_AUTHORITY")
        if branch.get("reference_action_exact") is not True or branch.get("control_action_reference_exact") is not True: errors.append("REFERENCE_ACTION_BINDING")
        if arm != "CONTROL":
            dose = DOSES.get(arm)
            compliance = branch.get("treatment_compliance") if isinstance(branch.get("treatment_compliance"), Mapping) else {}
            receipts = branch.get("treatment_receipts") if isinstance(branch.get("treatment_receipts"), list) else []
            if branch.get("treatment_compliant") is not True or compliance.get("treatment_compliant") is not True or compliance.get("delivered_open_steps") != dose or len(receipts) != dose: errors.append("TREATMENT_COMPLIANCE")
            if any(float(item.get("arm_delta_linf", 1.0)) != 0.0 for item in receipts): errors.append("ARM_ISOLATION")
    probes = {f"Q{i:02d}" for i in range(24)}
    if {(probe, arm) for probe in probes for arm in ARMS} != set(by_identity): errors.append("BRANCH_COVERAGE")
    label_ids: set[tuple[str, str]] = set()
    for row in labels:
        probe, dose = str(row.get("probe_id")), str(row.get("dose")); key = (probe, dose)
        if key in label_ids or probe not in probes or dose not in DOSES: errors.append("LABEL_IDENTITY")
        label_ids.add(key)
        expected = _truth_label(row.get("control_valid"), row.get("treatment_valid"), row.get("f_control"), row.get("f_open"))
        if row.get("label_class") != expected: errors.append("LABEL_TRUTH_TABLE")
        if row.get("binary_label_consumable") is not (expected in BINARY): errors.append("LABEL_BINARY_FLAG")
        if row.get("repeatability_status") != "NOT_APPLICABLE_SINGLE_EXECUTION": errors.append("LABEL_REPEATABILITY_MODE")
        control = by_identity.get((probe, "CONTROL")); treatment = by_identity.get((probe, dose))
        if control is None or treatment is None: errors.append("LABEL_BRANCH_LINEAGE")
        else:
            if row.get("control_branch_id") != control.get("branch_id") or row.get("treatment_branch_id") != treatment.get("branch_id"): errors.append("LABEL_BRANCH_REFERENCE")
            if row.get("control_result_sha256") != control.get("branch_result_sha256") or row.get("treatment_result_sha256") != treatment.get("branch_result_sha256"): errors.append("LABEL_RESULT_REFERENCE")
        if row.get("protected_counters") != COUNTERS: errors.append("LABEL_COUNTERS")
    if label_ids != {(probe, dose) for probe in probes for dose in DOSES}: errors.append("LABEL_COVERAGE")
    status = "PASS_M4_PARENT_INDEPENDENT" if not errors else "FAIL_M4_PARENT_INDEPENDENT"
    audit_result = {"schema": "STAGE_V_M4_PARENT_INDEPENDENT_AUDIT_V1", "status": status, "verdict": "PASS" if not errors else "FAIL", "canonical_parent_key": parent_key, "source_commit": source_commit, "source_tree": source_tree, "branch_count": len(branches), "label_count": len(labels), "observation_count": len(observations), "errors": sorted(set(errors)), "protected_counters": dict(COUNTERS), "parent_result_sha256": _sha(result_path) if result_path.is_file() else None}
    (root / "M4_INDEPENDENT_AUDIT.json").write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    try:
        value = audit(args.root.resolve(), args.parent_key, args.source_commit, args.source_tree)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL_M4_PARENT_INDEPENDENT", "errors": [f"{type(exc).__name__}:{exc}"]}, sort_keys=True))
        return 2
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] == "PASS_M4_PARENT_INDEPENDENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
