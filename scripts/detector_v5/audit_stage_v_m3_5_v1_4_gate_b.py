#!/usr/bin/env python3
"""Independent Gate-B audit for one V1.4 parent root."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_canonical_execution_core import canonical_value
from gripper_attack.stage_v_causal_observation_snapshot import (
    CausalSnapshotError,
    assert_primary_observation_exact,
    load_snapshot,
)


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
DOSES = {"T3": 3, "T5": 5, "T10": 10}
H_PHYS = 10
FAILURES = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
BINARY = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}:{line_number}")
        rows.append(dict(value))
    return rows


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _verify_seal(root: Path) -> list[str]:
    errors: list[str] = []
    sums = root / "SHA256SUMS"
    header = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not header.is_file():
        return ["SEAL_FILES_MISSING"]
    header_parts = header.read_text(encoding="utf-8").split()
    if len(header_parts) != 2 or header_parts[1] != "SHA256SUMS" or header_parts[0] != _sha_file(sums):
        errors.append("SEAL_HEADER_INVALID")
    seen: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in seen or len(digest) != 64:
            errors.append(f"SEAL_ROW_INVALID:{relative}")
            continue
        seen.add(relative)
        try:
            target = _inside(root, relative)
        except (OSError, ValueError):
            errors.append(f"SEAL_PATH_INVALID:{relative}")
            continue
        if not target.is_file() or _sha_file(target) != digest:
            errors.append(f"SEAL_SHA_MISMATCH:{relative}")
    return errors


def _position(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("post_object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _contact_loss(rows: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]] | None = None) -> int | None:
    count = 0
    for index, row in enumerate(rows):
        lost = row.get("post_object_gripper_contact") is not True if control is None else control[index].get("post_object_gripper_contact") is True and row.get("post_object_gripper_contact") is not True
        count = count + 1 if lost else 0
        if count >= 2:
            return index - 1
    return None


def _physical(branch: Mapping[str, Any], required: int, reference: Mapping[str, Any] | None = None) -> str:
    if branch.get("status") != "PASS" or branch.get("state_restore_exact") is not True or branch.get("causal_input_binding_pass") is not True or int(branch.get("available_horizon_steps", -1)) < int(required):
        return "HORIZON_CENSORED_ABSTAIN"
    rows = list(branch.get("rows", []))[:required]
    if len(rows) != required or any(row.get("post_contact_telemetry_valid") is not True for row in rows):
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    if reference is None:
        if branch.get("control_action_reference_exact") is not True:
            return "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"
        return "GRIPPER_CONTACT_LOSS" if _contact_loss(rows) is not None else "NO_PHYSICAL_FAILURE"
    control_rows = list(reference.get("rows", []))[:required]
    if len(control_rows) != required or any(row.get("post_contact_telemetry_valid") is not True for row in control_rows):
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    for left, right in zip(control_rows, rows):
        a, b = _position(left), _position(right)
        if a is None or b is None:
            return "PHYSICAL_AMBIGUITY_ABSTAIN"
        displacement = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        if a[2] - b[2] >= 0.02 and right.get("post_object_gripper_contact") is not True and right.get("post_object_support_contact") is not True:
            return "OBJECT_DROP"
        if displacement >= 0.01 and left.get("post_object_gripper_contact") is True and right.get("post_object_gripper_contact") is not True:
            return "PREMATURE_OBJECT_RELEASE"
    return "GRIPPER_CONTACT_LOSS" if _contact_loss(rows, control_rows) is not None else "NO_PHYSICAL_FAILURE"


def _compliant(branch: Mapping[str, Any], dose: int) -> bool:
    if dose == 0:
        return branch.get("treatment_compliant") is True and branch.get("treatment_compliance", {}).get("delivered_open_steps") == 0
    receipts = branch.get("treatment_receipts")
    if not isinstance(receipts, list) or len(receipts) != dose:
        return False
    for row in receipts:
        if not isinstance(row, Mapping):
            return False
        try:
            if not isinstance(row.get("raw_policy_action"), list) or abs(float(row["raw_policy_action"][-1]) - 1.0) > 1e-7:
                return False
            if not isinstance(row.get("normalized_action"), list) or abs(float(row["normalized_action"][-1]) - 1.0) > 1e-7:
                return False
            if not isinstance(row.get("env_action"), list) or abs(float(row["env_action"][-1]) + 1.0) > 1e-7:
                return False
            if float(row.get("arm_delta_linf", math.inf)) != 0.0:
                return False
        except (TypeError, ValueError, IndexError):
            return False
    return branch.get("treatment_compliant") is True and branch.get("treatment_compliance", {}).get("delivered_open_steps") == dose


def _label(control: Mapping[str, Any], treatment: Mapping[str, Any], dose: int) -> str:
    required = dose + H_PHYS
    control_class = _physical(control, required)
    treatment_class = _physical(treatment, required, control)
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    treatment_valid = _compliant(treatment, dose) and treatment_class in FAILURES | {"NO_PHYSICAL_FAILURE"}
    f_control = 1 if control_class in FAILURES else (0 if control_valid else None)
    f_open = 1 if treatment_class in FAILURES else (0 if treatment_valid else None)
    if f_control == 1:
        return "CONTROL_CONTAMINATION_ABSTAIN" if f_open == 1 else "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    if not control_valid:
        return "CONTROL_INVALID_ABSTAIN"
    if not treatment_valid:
        return "TREATMENT_INVALID_ABSTAIN"
    if f_control is None or f_open is None:
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    return "V_PHYS" if f_open == 1 else "NO_PHYSICAL_VULNERABILITY"


def audit(root: Path, *, parent_key: str, source_commit: str, source_tree: str) -> dict[str, Any]:
    errors = _verify_seal(root)
    receipt_path = root / "M3_5_V1_4_GATE_B_RECEIPT.json"
    try:
        receipt = _load(receipt_path)
        branches = _jsonl(root / "COUNTERFACTUAL_BRANCHES_V1_4.jsonl")
        observations = _jsonl(root / "TREATMENT_REPETITION_OBSERVATIONS_V1_4.jsonl")
        collapsed = _jsonl(root / "COLLAPSED_PROBE_DOSE_LABELS_V1_4.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        receipt, branches, observations, collapsed = {}, [], [], []
        errors.append(f"ARTIFACT_READ_FAIL:{type(exc).__name__}:{exc}")
    if receipt.get("schema") != "STAGE_V_M3_5_V1_4_GATE_B_RECEIPT_V1":
        errors.append("RECEIPT_SCHEMA_INVALID")
    if receipt.get("canonical_parent_key") != parent_key or receipt.get("source_commit") != source_commit or receipt.get("source_tree") != source_tree:
        errors.append("PROVENANCE_BINDING_INVALID")
    if receipt.get("fresh_render_equality_gate_used") is not False or receipt.get("fresh_render_primary_consumption") is not False or receipt.get("native_policy_calls_in_primary_window") != 0:
        errors.append("PRIMARY_RENDER_OR_POLICY_BOUNDARY_INVALID")
    if receipt.get("protected_counters") != COUNTERS:
        errors.append("PROTECTED_COUNTERS_NONZERO")
    protocol_path = root / "M3_5_V1_4_GATE_B_PROTOCOL.json"
    try:
        protocol = _load(protocol_path)
        binding = protocol.get("requires", {}).get("gate_a_bindings", {}).get(parent_key)
        if protocol.get("requires", {}).get("gate_a_binding_mode") != "PER_PARENT_EXACT_SHA256" or not isinstance(binding, Mapping):
            errors.append("GATE_A_BINDING_CONTRACT_INVALID")
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        binding = None
        errors.append("GATE_B_PROTOCOL_BINDING_READ_FAIL")
    if len(branches) != 288 or len(observations) != 216 or len(collapsed) != 72:
        errors.append(f"ACCOUNTING_INVALID:{len(branches)}/{len(observations)}/{len(collapsed)}")
    expected_observations: dict[str, tuple[str, dict[str, str | None]]] = {}
    gate_a_root = Path(str(receipt.get("gate_a_root", ""))).resolve()
    try:
        gate_a = _load(gate_a_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
        gate_a_receipt_path = gate_a_root / "M3_5_V1_4_GATE_A_RECEIPT.json"
        gate_a_audit_path = gate_a_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
        if not isinstance(binding, Mapping) or binding.get("gate_a_receipt_sha256") != _sha_file(gate_a_receipt_path) or binding.get("gate_a_independent_audit_sha256") != _sha_file(gate_a_audit_path):
            errors.append("GATE_A_PROVENANCE_BINDING_INVALID")
        if receipt.get("gate_a_receipt_sha256") != _sha_file(gate_a_receipt_path) or receipt.get("gate_a_independent_audit_sha256") != _sha_file(gate_a_audit_path):
            errors.append("RECEIPT_GATE_A_PROVENANCE_INVALID")
        for snapshot in gate_a.get("snapshots", []):
            probe_id = str(snapshot["probe_id"])
            snapshot_root = gate_a_root / str(snapshot["path"])
            loaded = load_snapshot(snapshot_root, materialize_torch=True)
            expected_observations[probe_id] = (
                _sha_file(snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json"),
                assert_primary_observation_exact(loaded["payload"]),
            )
    except (OSError, KeyError, TypeError, ValueError, CausalSnapshotError) as exc:
        errors.append(f"GATE_A_OBSERVATION_BINDING_READ_FAIL:{type(exc).__name__}:{exc}")
    by_identity: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    controls: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in branches:
        branch = row.get("branch") if isinstance(row.get("branch"), Mapping) else {}
        key = (str(row.get("probe_id")), int(row.get("repetition", -1)), str(row.get("arm")))
        if key in by_identity:
            errors.append(f"DUPLICATE_BRANCH:{key}")
        by_identity[key] = branch
        if str(row.get("canonical_parent_key")) != parent_key or branch.get("status") != "PASS" or branch.get("causal_input_binding_pass") is not True or branch.get("primary_input_authority") != "loaded_frozen_canonical_bytes" or branch.get("fresh_render_primary_consumption") is not False or branch.get("native_policy_calls_in_primary_window") != 0:
            errors.append(f"BRANCH_CAUSAL_GATE_INVALID:{key}")
        expected_observation = expected_observations.get(str(row.get("probe_id")))
        if expected_observation is None or branch.get("snapshot_manifest_sha256") != expected_observation[0] or branch.get("primary_observation_hashes") != expected_observation[1]:
            errors.append(f"BRANCH_OBSERVATION_BINDING_INVALID:{key}")
        expected_branch_sha = _sha_json(canonical_value(branch))
        if row.get("branch_result_sha256") != expected_branch_sha:
            errors.append(f"BRANCH_SHA_INVALID:{key}")
        for action in branch.get("rows", []):
            if not isinstance(action, Mapping) or float(action.get("arm_delta_linf", math.inf)) != 0.0:
                errors.append(f"ARM_ISOLATION_INVALID:{key}")
        if str(row.get("arm")) == "CONTROL":
            controls[(str(row.get("probe_id")), int(row.get("repetition", -1)))] = branch
    expected_keys = {(f"Q{i:02d}", rep, arm) for i in range(24) for rep in range(3) for arm in ("CONTROL", *DOSES)}
    if set(by_identity) != expected_keys:
        errors.append("BRANCH_IDENTITY_COVERAGE_INVALID")
    recomputed_observations = []
    for probe in range(24):
        for repetition in range(3):
            control = controls.get((f"Q{probe:02d}", repetition))
            for arm, dose in DOSES.items():
                treatment = by_identity.get((f"Q{probe:02d}", repetition, arm))
                if control is None or treatment is None:
                    errors.append(f"PAIR_MISSING:Q{probe:02d}:R{repetition}:{arm}")
                    continue
                recomputed_observations.append((f"Q{probe:02d}", repetition, arm, _label(control, treatment, dose)))
    observed_map = {(str(row.get("probe_id")), int(row.get("repetition", -1)), str(row.get("dose"))): str(row.get("label_class")) for row in observations}
    for probe, repetition, arm, label in recomputed_observations:
        if observed_map.get((probe, repetition, arm)) != label:
            errors.append(f"OBSERVATION_RECOMPUTE_MISMATCH:{probe}:R{repetition}:{arm}")
    repeatability_pass = True
    for probe in range(24):
        for arm in DOSES:
            labels = [label for p, _rep, a, label in recomputed_observations if p == f"Q{probe:02d}" and a == arm]
            if len(labels) != 3 or len(set(labels)) != 1 or labels[0] not in BINARY:
                repeatability_pass = False
    if not repeatability_pass:
        errors.append("REPEATABILITY_3_OF_3_INVALID")
    for row in collapsed:
        key = (str(row.get("probe_id")), str(row.get("dose")))
        expected = [label for probe, _rep, arm, label in recomputed_observations if (probe, arm) == key]
        if not expected or row.get("label_class") != (expected[0] if len(set(expected)) == 1 else "HOLD_STOCHASTIC_INTERVENTION_OUTCOME") or row.get("binary_label_consumable") is not (len(expected) == 3 and len(set(expected)) == 1 and expected[0] in BINARY):
            errors.append(f"COLLAPSED_LABEL_INVALID:{key}")
    result = {
        "schema": "STAGE_V_M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT_V1",
        "status": "PASS_PARENT_INDEPENDENT" if not errors else "FAIL_SEALED",
        "auditor_role": "independent_gate_b_recompute_no_producer_decision_helper",
        "canonical_parent_key": parent_key,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "recomputed_branch_count": len(branches),
        "recomputed_observation_count": len(recomputed_observations),
        "repeatability_pass": repeatability_pass,
        "errors": sorted(set(errors)),
        "protected_counters": dict(COUNTERS),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root.resolve(), parent_key=args.parent_key, source_commit=args.source_commit, source_tree=args.source_tree)
    output = args.output.resolve() if args.output else args.root.resolve() / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json"
    output.write_bytes((json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return 0 if result["status"] == "PASS_PARENT_INDEPENDENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
