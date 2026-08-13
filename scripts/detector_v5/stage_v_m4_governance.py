"""Shared fail-closed governance for formal M4 authorization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SPLITS = ("TRAIN", "VAL", "TEST")
REQUIRED_SUITE_COUNTS = {suite: 10 for suite in SUITES}
REQUIRED_SPLIT_COUNTS = {"TRAIN": 24, "VAL": 8, "TEST": 8}
REQUIRED_PER_SUITE_SPLIT_COUNTS = {suite: {"TRAIN": 6, "VAL": 2, "TEST": 2} for suite in SUITES}


class M4GovernanceError(ValueError):
    """A formal M4 artifact is not consumable under current governance."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M4GovernanceError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _bound_path(value: Any, protocol_path: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (protocol_path.parents[1] / path).resolve()


def protocol_declares_corridor_gate(protocol: Mapping[str, Any]) -> bool:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    required = (
        "supersession_hold_path",
        "supersession_hold_sha256",
        "formal_parent_manifest_path",
        "formal_parent_manifest_sha256",
        "corridor_pass_receipt_path",
        "corridor_pass_receipt_sha256",
        "corridor_qualification_protocol_sha256",
        "corridor_qualification_authorization_sha256",
        "corridor_reconciliation_sha256",
    )
    return all(isinstance(inputs.get(key), str) and inputs[key] for key in required)


def _parent_keys(value: Mapping[str, Any], field: str) -> list[str]:
    rows = value.get(field)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise M4GovernanceError(f"M4_{field.upper()}_INVALID")
    keys = [str(row.get("canonical_parent_key", "")) for row in rows]
    if len(keys) != len(set(keys)) or any(not key for key in keys):
        raise M4GovernanceError(f"M4_{field.upper()}_DUPLICATE_OR_EMPTY")
    return keys


def _counts(rows: list[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    suites = {suite: 0 for suite in SUITES}
    splits = {split: 0 for split in SPLITS}
    by_suite = {suite: {split: 0 for split in SPLITS} for suite in SUITES}
    for row in rows:
        suite, split = str(row.get("suite", "")), str(row.get("split", ""))
        if suite not in suites or split not in splits:
            raise M4GovernanceError("M4_PARENT_SUITE_OR_SPLIT_INVALID")
        suites[suite] += 1
        splits[split] += 1
        by_suite[suite][split] += 1
    return suites, splits, by_suite


ACTIVE_STUDENT_HEADS = (
    "physical_criticality",
    "k10_feasibility",
    "instability",
    "gripper_closing_state",
)
M4_V2_MATRIX = {
    "parents": 40,
    "probes_per_parent": 24,
    "repetitions": 1,
    "conditions": ["CONTROL", "T3", "T5", "T10"],
    "physical_executions_per_parent": 96,
    "treatment_labels_per_parent": 72,
    "physical_executions_total": 3840,
    "treatment_labels_total": 2880,
    "primary_dose": "T5",
    "secondary_doses": ["T3", "T10"],
    "dose_steps": {"T3": 3, "T5": 5, "T10": 10},
    "h_phys": 10,
}


def _bound_file(inputs: Mapping[str, Any], protocol_path: Path, name: str) -> tuple[Path, str]:
    path_key, sha_key = f"{name}_path", f"{name}_sha256"
    path = _bound_path(inputs.get(path_key), protocol_path)
    expected = inputs.get(sha_key)
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise M4GovernanceError(f"M4_V2_{name.upper()}_BINDING_INVALID")
    return path, expected


def _bound_root(inputs: Mapping[str, Any], protocol_path: Path, name: str, *, exact_plan: bool = False) -> Path:
    root = _bound_path(inputs.get(f"{name}_root"), protocol_path)
    expected = inputs.get(f"{name}_root_seal_sha256")
    sums = root / "SHA256SUMS"
    seal = root / ("ROOT_SEAL.sha256" if exact_plan else "SHA256SUMS.sha256")
    if not root.is_dir() or not sums.is_file() or not seal.is_file() or not isinstance(expected, str):
        raise M4GovernanceError(f"M4_V2_{name.upper()}_ROOT_INCOMPLETE")
    token = seal.read_text(encoding="utf-8").split()
    if not token or token[0] != sha256(sums) or token[0] != expected:
        raise M4GovernanceError(f"M4_V2_{name.upper()}_ROOT_SEAL_INVALID")
    return root


def _path_matches(value: Any, expected: Path) -> bool:
    try:
        return Path(str(value)).resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _require_current_boundary(value: Mapping[str, Any], *, name: str) -> None:
    if value.get("protected_counters") != COUNTERS:
        raise M4GovernanceError(f"M4_V2_{name}_PROTECTED_BOUNDARY_INVALID")
    if value.get("formal_m4_authorized") is not False:
        raise M4GovernanceError(f"M4_V2_{name}_FORMAL_AUTHORIZATION_INVALID")
    if value.get("m4_outcomes_read", value.get("outcomes_read")) is not False:
        raise M4GovernanceError(f"M4_V2_{name}_OUTCOME_BOUNDARY_INVALID")


def validate_formal_m4_v2_authority(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    split_path: Path,
    source_commit: str,
    source_tree: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the current post-freeze authority contract, without reading outcomes."""
    if protocol.get("schema") == "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1":
        raise M4GovernanceError("M4_PROTOCOL_V1_SUPERSEDED_CURRENT_MAINLINE")
    if protocol.get("schema") != "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2":
        raise M4GovernanceError("M4_PROTOCOL_V2_REQUIRED")
    if protocol.get("status") != "FROZEN_PROSPECTIVE_NOT_AUTHORIZED" or protocol.get("runtime_authorized") is not False:
        raise M4GovernanceError("M4_V2_PROTOCOL_MUST_REMAIN_PROSPECTIVE")
    if protocol.get("requires_explicit_owner_authorization") is not True:
        raise M4GovernanceError("M4_V2_EXPLICIT_OWNER_AUTHORIZATION_REQUIRED")
    source = protocol.get("source_binding")
    if not isinstance(source, Mapping) or source.get("runtime_commit") != source_commit or source.get("runtime_tree") != source_tree:
        raise M4GovernanceError("M4_V2_SOURCE_BINDING_MISMATCH")
    if protocol.get("matrix") != M4_V2_MATRIX:
        raise M4GovernanceError("M4_V2_MATRIX_INVALID")
    operation = protocol.get("operation")
    if not isinstance(operation, Mapping) or operation.get("matched_clean_action_replay") is not True or operation.get("clean_reference_action_lineage") is not True or operation.get("native_policy_calls_in_primary_window") != 0 or operation.get("fresh_render_primary_consumption") is not False or operation.get("fresh_render_equality_gate_used") is not False or operation.get("arm_delta_linf_exact_zero") is not True or operation.get("treatment_only_difference") != "GRIPPER_OPEN":
        raise M4GovernanceError("M4_V2_PRIMARY_CONTRACT_INVALID")
    if protocol.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_V2_PROTOCOL_PROTECTED_BOUNDARY_INVALID")

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise M4GovernanceError("M4_V2_INPUT_BINDINGS_MISSING")
    required_files = (
        "v1_supersession_receipt",
        "formal_parent_manifest",
        "formal_parent_split",
        "exact_plan_manifest",
        "exact_plan_audit",
        "exact_plan_result",
        "primary_firewall_report",
        "teacher_student_freeze_report",
        "pre_m4_lock_report",
        "student_checkpoint",
        "student_thresholds",
        "feature_schema",
        "architecture_addendum",
    )
    files = {name: _bound_file(inputs, protocol_path, name) for name in required_files}
    if not _path_matches(inputs.get("formal_parent_split_path"), split_path):
        raise M4GovernanceError("M4_V2_FORMAL_SPLIT_PATH_MISMATCH")

    supersession = _load(files["v1_supersession_receipt"][0])
    if supersession.get("schema") != "STAGE_V_M4_PROTOCOL_V1_SUPERSESSION_RECEIPT_V1" or supersession.get("status") != "HISTORICAL_NONCONSUMABLE_FOR_CURRENT_MAINLINE" or supersession.get("old_artifacts_modified") is not False:
        raise M4GovernanceError("M4_V2_V1_SUPERSESSION_INVALID")
    if supersession.get("formal_m4_authorized") is not False or supersession.get("m4_outcomes_read") is not False or supersession.get("v_phys_generated") is not False or supersession.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_V2_V1_SUPERSESSION_BOUNDARY_INVALID")

    manifest_path, manifest_sha = files["formal_parent_manifest"]
    manifest = _load(manifest_path)
    if manifest.get("schema") != "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" or manifest.get("status") != "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" or manifest.get("parent_count") != 40:
        raise M4GovernanceError("M4_V2_FINAL40_MANIFEST_INVALID")
    _require_current_boundary(manifest, name="FINAL40")
    manifest_keys = _parent_keys(manifest, "parents")
    manifest_suites, manifest_splits, _ = _counts(manifest["parents"])
    if manifest_suites != REQUIRED_SUITE_COUNTS or manifest_splits != REQUIRED_SPLIT_COUNTS or manifest.get("per_suite_split_counts") != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_V2_FINAL40_POPULATION_INVALID")

    split_bound_path, split_sha = files["formal_parent_split"]
    split = _load(split_bound_path)
    if split.get("schema") != "STAGE_V_M4_FINAL_PARENT_SPLIT_V2" or split.get("status") != "FROZEN":
        raise M4GovernanceError("M4_V2_FINAL_SPLIT_INVALID")
    _require_current_boundary(split, name="FINAL_SPLIT")
    split_keys = _parent_keys(split, "parents")
    split_suites, split_counts, per_suite = _counts(split["parents"])
    if split_keys != manifest_keys or split_suites != REQUIRED_SUITE_COUNTS or split_counts != REQUIRED_SPLIT_COUNTS or per_suite != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_V2_FINAL_SPLIT_POPULATION_INVALID")
    if split.get("final_manifest_sha256") != manifest_sha or not _path_matches(split.get("final_manifest_path"), manifest_path):
        raise M4GovernanceError("M4_V2_FINAL_SPLIT_MANIFEST_BINDING_INVALID")

    exact_root = _bound_root(inputs, protocol_path, "exact_plan", exact_plan=True)
    exact_manifest_path, exact_manifest_sha = files["exact_plan_manifest"]
    exact_audit_path, exact_audit_sha = files["exact_plan_audit"]
    exact_result_path, exact_result_sha = files["exact_plan_result"]
    if exact_manifest_path.parent != exact_root or exact_audit_path.parent != exact_root or exact_result_path.parent != exact_root:
        raise M4GovernanceError("M4_V2_EXACT_PLAN_ROOT_BINDING_INVALID")
    exact_manifest = _load(exact_manifest_path)
    exact_audit = _load(exact_audit_path)
    exact_result = _load(exact_result_path)
    if exact_manifest.get("schema") != "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1" or exact_manifest.get("status") != "PASS_EXACT_40X24_PLAN_ONLY" or exact_manifest.get("parent_count") != 40 or exact_manifest.get("probe_count_per_parent") != 24 or exact_manifest.get("probe_count_total") != 960 or exact_manifest.get("planned_branch_authority_count") != 3840:
        raise M4GovernanceError("M4_V2_EXACT_PLAN_MANIFEST_INVALID")
    if exact_manifest.get("independent_audit_sha256") != exact_audit_sha or exact_audit.get("status") != "PASS" or exact_result.get("status") != "PASS" or exact_result.get("manifest_status") != "PASS_EXACT_40X24_PLAN_ONLY" or exact_result.get("audit_sha256") != exact_audit_sha or exact_manifest.get("final40_manifest_sha256") != manifest_sha or exact_manifest.get("final_split_sha256") != split_sha:
        raise M4GovernanceError("M4_V2_EXACT_PLAN_UPSTREAM_BINDING_INVALID")
    if exact_manifest.get("selection_outcomes_read") is not False or exact_manifest.get("intervention_executed") is not False or exact_manifest.get("v_phys_generated") is not False or exact_manifest.get("teacher_predictions_read") is not False or exact_manifest.get("student_predictions_read") is not False or exact_manifest.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_V2_EXACT_PLAN_BOUNDARY_INVALID")

    firewall_root = _bound_root(inputs, protocol_path, "primary_firewall")
    firewall_path, firewall_sha = files["primary_firewall_report"]
    firewall = _load(firewall_path)
    if firewall_path.parent != firewall_root or firewall.get("schema") != "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3" or firewall.get("status") != "PASS_PRIMARY_DATA_FIREWALL_EXACT55" or firewall.get("source_artifacts_modified") is not False:
        raise M4GovernanceError("M4_V2_PRIMARY_FIREWALL_INVALID")
    _require_current_boundary(firewall, name="PRIMARY_FIREWALL")
    if firewall.get("exact_plan", {}).get("manifest_sha256") != exact_manifest_sha or firewall.get("final40", {}).get("sha256") != manifest_sha or firewall.get("final_split", {}).get("sha256") != split_sha or firewall.get("primary_identity_firewall", {}).get("attempted_overlap_count") != 0 or firewall.get("primary_identity_firewall", {}).get("final40_overlap_count") != 0:
        raise M4GovernanceError("M4_V2_PRIMARY_FIREWALL_BINDING_INVALID")

    freeze_root = _bound_root(inputs, protocol_path, "teacher_student_freeze")
    freeze_path, freeze_sha = files["teacher_student_freeze_report"]
    freeze = _load(freeze_path)
    if freeze_path.parent != freeze_root or freeze.get("schema") != "STAGE_V_PRIMARY_TEACHER_STUDENT_FREEZE_V1" or freeze.get("status") != "PASS_PRIMARY_TEACHER_STUDENT_FREEZE":
        raise M4GovernanceError("M4_V2_TEACHER_STUDENT_FREEZE_INVALID")
    _require_current_boundary(freeze, name="TEACHER_STUDENT_FREEZE")
    if freeze.get("v_phys_generated") is not False or freeze.get("architecture_order") != ["CLEAN_ROLLOUT", "PRIVILEGED_CLEAN_TEACHER_C_t", "CLEAN_TEACHER_SUPERVISED_CAUSAL_STUDENT_C_HAT_t", "HELD_OUT_MATCHED_COUNTERFACTUAL_VALIDATION_V_t_d"]:
        raise M4GovernanceError("M4_V2_TEACHER_STUDENT_ORDER_INVALID")
    if freeze.get("final40", {}).get("sha256") != manifest_sha or freeze.get("final40", {}).get("split_sha256") != split_sha or freeze.get("exact_plan", {}).get("manifest_sha256") != exact_manifest_sha or freeze.get("primary_data_firewall", {}).get("report_sha256") != firewall_sha:
        raise M4GovernanceError("M4_V2_TEACHER_STUDENT_UPSTREAM_BINDING_INVALID")
    coverage = freeze.get("coverage", {})
    if coverage.get("eligible_heads") != list(ACTIVE_STUDENT_HEADS) or coverage.get("held_heads") != ["safe_release"] or freeze.get("student", {}).get("active_heads") != list(ACTIVE_STUDENT_HEADS) or freeze.get("g7", {}).get("test_read_count") != 1:
        raise M4GovernanceError("M4_V2_TEACHER_STUDENT_HEAD_OR_TEST_INVALID")

    lock_root = _bound_root(inputs, protocol_path, "pre_m4_lock")
    lock_path, lock_sha = files["pre_m4_lock_report"]
    lock = _load(lock_path)
    if lock_path.parent != lock_root or lock.get("schema") != "STAGE_V_PRE_M4_LOCK_V1" or lock.get("status") != "PASS_PRE_M4_LOCK" or lock.get("parent_count") != 40 or lock.get("probe_count") != 960 or lock.get("planned_branch_count") != 3840:
        raise M4GovernanceError("M4_V2_PRE_M4_LOCK_INVALID")
    _require_current_boundary(lock, name="PRE_M4_LOCK")
    if lock.get("intervention_executed") is not False or lock.get("teacher_predictions_read") is not False or lock.get("student_predictions_read") is not False or lock.get("freeze", {}).get("report_sha256") != freeze_sha or lock.get("primary_data_firewall", {}).get("report_sha256") != firewall_sha or lock.get("exact_plan", {}).get("manifest_sha256") != exact_manifest_sha:
        raise M4GovernanceError("M4_V2_PRE_M4_LOCK_BINDING_INVALID")

    checkpoint_path, checkpoint_sha = files["student_checkpoint"]
    thresholds_path, thresholds_sha = files["student_thresholds"]
    feature_path, feature_sha = files["feature_schema"]
    thresholds = _load(thresholds_path)
    if set(thresholds) != set(ACTIVE_STUDENT_HEADS):
        raise M4GovernanceError("M4_V2_STUDENT_THRESHOLD_HEADS_INVALID")
    feature = _load(feature_path)
    if feature.get("schema") != "V5_R3_SC5_FEATURE_BINDING_V1" or feature.get("status") != "FROZEN_ENGINEERING_BINDING" or len(feature.get("feature_order", [])) != 25 or feature.get("future_fields_used") is not False or feature.get("teacher_fields_used") is not False or feature.get("outcome_fields_used") is not False or feature.get("attack_enabled") is not False or feature.get("feature_order_sha256") != inputs.get("feature_order_sha256"):
        raise M4GovernanceError("M4_V2_FEATURE_SCHEMA_INVALID")
    if freeze.get("student", {}).get("checkpoint_sha256") != checkpoint_sha or freeze.get("student", {}).get("thresholds_sha256") != thresholds_sha or freeze.get("feature_schema_sha256") != feature_sha or freeze.get("feature_order_sha256") != inputs.get("feature_order_sha256"):
        raise M4GovernanceError("M4_V2_STUDENT_BINDING_INVALID")

    architecture = _load(files["architecture_addendum"][0])
    required_order = ["V2_TERMINAL_HOLD", "POST_HOLD_CORRIDOR_REPLENISHMENT", "COMPOSITE_CORRIDOR_RECONCILIATION", "FINAL40_FREEZE", "SPLIT_FREEZE", "EXACT_40X24_PLAN_AND_SNAPSHOT_ONLY", "PRIMARY_DATA_FIREWALL", "CLEAN_ONLY_TEACHER_FREEZE", "CAUSAL_STUDENT_FREEZE", "FORMAL_M4_INTERVENTION_AUTHORIZATION", "M4_OUTCOME_READ"]
    if architecture.get("schema") != "STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM" or architecture.get("status") != "ACTIVE_STATUS_ADDENDUM" or architecture.get("architecture_semantics_changed") is not False or architecture.get("mainline_order_lock") != required_order or not set(architecture.get("m4_outcome_read_prerequisites", [])).issuperset({"FINAL40_AND_SPLIT_SEALED", "EXACT_40X24_PLAN_AND_SNAPSHOT_MANIFEST_AUDITED", "PRIMARY_DATA_FIREWALL_SEALED", "PRIMARY_TEACHER_FREEZE_SHA_BOUND", "PRIMARY_STUDENT_FREEZE_SHA_BOUND", "STUDENT_FEATURE_SCHEMA_SHA_BOUND", "STUDENT_THRESHOLD_SHA_BOUND"}):
        raise M4GovernanceError("M4_V2_ARCHITECTURE_ADDENDUM_INVALID")

    if authorization is not None:
        if authorization.get("schema") != "STAGE_V_M4_RUNTIME_AUTHORIZATION_V2" or authorization.get("status") != "PASS" or authorization.get("authorization_kind") != "FORMAL_M4_V2" or authorization.get("runtime_authorized") is not True or authorization.get("protocol_sha256") != sha256(protocol_path) or authorization.get("source_commit") != source_commit or authorization.get("source_tree") != source_tree or authorization.get("protected_counters") != COUNTERS or authorization.get("intervention_executed") is not False or authorization.get("outcomes_read") is not False or authorization.get("v_phys_generated") is not False:
            raise M4GovernanceError("M4_V2_RUNTIME_AUTHORIZATION_INVALID")
        bindings = authorization.get("authority_bindings")
        if not isinstance(bindings, Mapping) or bindings.get("formal_parent_manifest_sha256") != manifest_sha or bindings.get("formal_parent_split_sha256") != split_sha or bindings.get("exact_plan_manifest_sha256") != exact_manifest_sha or bindings.get("primary_firewall_report_sha256") != firewall_sha or bindings.get("teacher_student_freeze_report_sha256") != freeze_sha or bindings.get("pre_m4_lock_report_sha256") != lock_sha or bindings.get("student_checkpoint_sha256") != checkpoint_sha or bindings.get("student_thresholds_sha256") != thresholds_sha or bindings.get("feature_schema_sha256") != feature_sha or bindings.get("feature_order_sha256") != inputs.get("feature_order_sha256"):
            raise M4GovernanceError("M4_V2_RUNTIME_AUTHORITY_BINDING_INVALID")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "split_path": str(split_bound_path),
        "split_sha256": split_sha,
        "parent_keys": sorted(manifest_keys),
        "suite_counts": dict(REQUIRED_SUITE_COUNTS),
        "split_counts": dict(REQUIRED_SPLIT_COUNTS),
        "exact_plan_root": str(exact_root),
        "exact_plan_manifest_sha256": exact_manifest_sha,
        "primary_firewall_report_sha256": firewall_sha,
        "teacher_student_freeze_report_sha256": freeze_sha,
        "pre_m4_lock_report_sha256": lock_sha,
        "student_checkpoint_sha256": checkpoint_sha,
        "student_thresholds_sha256": thresholds_sha,
        "feature_schema_sha256": feature_sha,
    }


def validate_formal_m4_corridor_gate(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    split_path: Path,
    source_commit: str,
    source_tree: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a new corridor PASS receipt before formal M4 is consumable."""
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs.get("supersession_hold_path"):
        raise M4GovernanceError("M4_FORMAL_AUTHORIZATION_SUPERSEDED_BY_CORRIDOR_HOLD")
    if not protocol_declares_corridor_gate(protocol):
        raise M4GovernanceError("M4_CORRIDOR_GATE_BINDING_INCOMPLETE")

    hold_path = _bound_path(inputs["supersession_hold_path"], protocol_path)
    if not hold_path.is_file() or sha256(hold_path) != inputs["supersession_hold_sha256"]:
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_BINDING_INVALID")
    hold = _load(hold_path)
    if hold.get("schema") != "STAGE_V_M4_FORMAL_AUTHORIZATION_SUPERSESSION_HOLD_V1" or hold.get("status") != "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT":
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_STATUS_INVALID")
    if hold.get("stable_parent_count") != 29 or hold.get("required_parent_count") != 40 or hold.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_BINDING_INVALID")
    stable_keys = hold.get("exact_stable_parent_keys")
    if not isinstance(stable_keys, list) or len(stable_keys) != 29 or len(set(stable_keys)) != 29:
        raise M4GovernanceError("M4_SUPERSESSION_STABLE_KEYS_INVALID")
    if hold.get("required_suite_counts") != REQUIRED_SUITE_COUNTS or hold.get("required_split_counts") != REQUIRED_SPLIT_COUNTS or hold.get("required_per_suite_split_counts") != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_SUPERSESSION_POPULATION_CONTRACT_INVALID")

    manifest_path = _bound_path(inputs["formal_parent_manifest_path"], protocol_path)
    if not manifest_path.is_file() or sha256(manifest_path) != inputs["formal_parent_manifest_sha256"]:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_BINDING_INVALID")
    manifest = _load(manifest_path)
    manifest_keys = _parent_keys(manifest, "parents")
    if len(manifest_keys) != 40:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_COUNT_INVALID")
    manifest_suites, manifest_splits, _ = _counts(manifest.get("parents", []))
    if manifest_suites != REQUIRED_SUITE_COUNTS or manifest_splits != REQUIRED_SPLIT_COUNTS:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_SPLIT_INVALID")

    if not split_path.is_file() or sha256(split_path) != inputs.get("formal_parent_split_sha256"):
        raise M4GovernanceError("M4_FORMAL_SPLIT_BINDING_INVALID")
    split = _load(split_path)
    split_keys = _parent_keys(split, "parents")
    if split_keys != manifest_keys:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_SPLIT_KEY_MISMATCH")
    split_suites, split_counts, per_suite = _counts(split.get("parents", []))
    if split_suites != REQUIRED_SUITE_COUNTS or split_counts != REQUIRED_SPLIT_COUNTS or per_suite != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_FORMAL_SPLIT_POPULATION_INVALID")

    receipt_path = _bound_path(inputs["corridor_pass_receipt_path"], protocol_path)
    if not receipt_path.is_file() or sha256(receipt_path) != inputs["corridor_pass_receipt_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_RECEIPT_BINDING_INVALID")
    receipt = _load(receipt_path)
    if receipt.get("schema") != "STAGE_V_M4_CORRIDOR_PASS_RECEIPT_V1" or receipt.get("status") != "PASS_FORMAL_M4_CORRIDOR":
        raise M4GovernanceError("M4_CORRIDOR_PASS_RECEIPT_NOT_PASS")
    if receipt.get("parent_count") != 40 or receipt.get("parent_keys") != sorted(manifest_keys):
        raise M4GovernanceError("M4_CORRIDOR_PASS_PARENT_SET_INVALID")
    if receipt.get("formal_parent_manifest_sha256") != inputs["formal_parent_manifest_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_MANIFEST_BINDING_INVALID")
    if receipt.get("formal_split_sha256") != inputs["formal_parent_split_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_SPLIT_BINDING_INVALID")
    if receipt.get("suite_counts") != REQUIRED_SUITE_COUNTS or receipt.get("split_counts") != REQUIRED_SPLIT_COUNTS or receipt.get("per_suite_split_counts") != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_CORRIDOR_PASS_POPULATION_INVALID")
    if receipt.get("protocol_sha256") != inputs["corridor_qualification_protocol_sha256"] or receipt.get("authorization_sha256") != inputs["corridor_qualification_authorization_sha256"] or receipt.get("reconciliation_sha256") != inputs["corridor_reconciliation_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_UPSTREAM_BINDING_INVALID")
    if receipt.get("source_commit") != source_commit or receipt.get("source_tree") != source_tree or receipt.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_CORRIDOR_PASS_SOURCE_OR_BOUNDARY_INVALID")
    if authorization is not None:
        if authorization.get("corridor_pass_receipt_sha256") != inputs["corridor_pass_receipt_sha256"]:
            raise M4GovernanceError("M4_AUTHORIZATION_CORRIDOR_PASS_BINDING_INVALID")
        if authorization.get("corridor_pass_receipt") != str(receipt_path):
            raise M4GovernanceError("M4_AUTHORIZATION_CORRIDOR_PASS_PATH_INVALID")

    return {
        "hold_path": str(hold_path),
        "hold_sha256": sha256(hold_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "parent_keys": sorted(manifest_keys),
        "suite_counts": dict(REQUIRED_SUITE_COUNTS),
        "split_counts": dict(REQUIRED_SPLIT_COUNTS),
    }
