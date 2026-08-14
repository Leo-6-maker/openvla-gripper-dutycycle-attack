"""Fail-closed authority checks for the one-probe zero-treatment Q00 canary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
Q00_PARENT_KEY = "libero_10/task_01/state_42"
Q00_PROBE_ID = "Q00"
Q00_ARMS = ("CONTROL", "T3", "T5", "T10")
SCHEMA = "STAGE_V_M4_Q00_ZERO_TREATMENT_AUTHORITY_V1"
PASS_STATUS = "PASS_Q00_ZERO_TREATMENT_AUTHORITY"
DESIGN_STATUS = "FROZEN_PROSPECTIVE_NOT_AUTHORIZED"
BOUND_FILES = (
    "m4_v2_protocol",
    "exact_plan_manifest",
    "snapshot_manifest",
    "runtime_provenance_receipt",
    "zero_treatment_auditor",
    "runtime_diff",
)


class Q00AuthorityError(ValueError):
    """The prospective Q00 canary authority is incomplete or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authority_sha256(authority: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(authority), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Q00AuthorityError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Q00AuthorityError(f"{name}_MISSING")
    return value


def _require(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise Q00AuthorityError(f"{name}_INVALID")


def _require_hex(value: Any, length: int, name: str) -> None:
    text = str(value or "").lower()
    if len(text) != length or any(char not in "0123456789abcdef" for char in text):
        raise Q00AuthorityError(f"{name}_INVALID")


def _bound_file(bindings: Mapping[str, Any], name: str) -> tuple[Path, str]:
    row = _mapping(bindings.get(name), f"BINDING_{name}")
    path = Path(str(row.get("path", ""))).resolve()
    expected = str(row.get("sha256", "")).lower()
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise Q00AuthorityError(f"BINDING_{name}_MISMATCH")
    return path, expected


def _check_zero_boundary(authority: Mapping[str, Any]) -> None:
    zero = _mapping(authority.get("zero_treatment"), "ZERO_TREATMENT")
    expected = {
        "post_snapshot_primary_window_steps": 0,
        "treatment_steps": 0,
        "forced_open_steps": 0,
        "label_records": 0,
        "v_phys_generated": False,
        "intervention_executed": False,
        "outcomes_read": False,
        "protected_counters": COUNTERS,
        "primary_input_authority": "loaded_frozen_canonical_bytes",
        "fresh_render_primary_consumption": False,
    }
    for name, value in expected.items():
        _require(zero.get(name), value, f"ZERO_TREATMENT_{name.upper()}")


def _check_resource_contract(authority: Mapping[str, Any]) -> None:
    resource = _mapping(authority.get("resource_contract"), "RESOURCE_CONTRACT")
    _require(resource.get("minimum_free_memory_mib"), 20_480, "RESOURCE_MINIMUM_FREE_MEMORY")
    _require(resource.get("strict_comparison"), "free_memory_mib > minimum_free_memory_mib", "RESOURCE_COMPARISON")
    _require(resource.get("maximum_project_workers_per_gpu"), 1, "RESOURCE_MAX_WORKERS_PER_GPU")
    _require(resource.get("foreign_workload_allowed"), True, "RESOURCE_FOREIGN_WORKLOAD")
    _require(resource.get("foreign_process_interference"), False, "RESOURCE_FOREIGN_INTERFERENCE")
    _require(resource.get("partial_fleet_allowed"), True, "RESOURCE_PARTIAL_FLEET")


def _check_protocol(path: Path) -> None:
    protocol = _load(path)
    _require(protocol.get("schema"), "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2", "M4_V2_PROTOCOL_SCHEMA")
    _require(protocol.get("status"), "FROZEN_PROSPECTIVE_NOT_AUTHORIZED", "M4_V2_PROTOCOL_STATUS")
    _require(protocol.get("runtime_authorized"), False, "M4_V2_PROTOCOL_RUNTIME_AUTHORIZED")
    _require(protocol.get("formal_m4_authorized", False), False, "M4_V2_PROTOCOL_FORMAL_AUTHORIZED")
    _require(protocol.get("requires_explicit_owner_authorization"), True, "M4_V2_PROTOCOL_OWNER_GATE")
    _require(protocol.get("protected_counters"), COUNTERS, "M4_V2_PROTOCOL_PROTECTED_COUNTERS")


def _check_exact_plan(path: Path, q00: Mapping[str, Any]) -> None:
    plan = _load(path)
    _require(plan.get("schema"), "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1", "EXACT_PLAN_SCHEMA")
    _require(plan.get("status"), "PASS_EXACT_40X24_PLAN_ONLY", "EXACT_PLAN_STATUS")
    for name, value in (
        ("selection_outcomes_read", False),
        ("intervention_executed", False),
        ("v_phys_generated", False),
        ("teacher_predictions_read", False),
        ("student_predictions_read", False),
        ("protected_counters", COUNTERS),
    ):
        _require(plan.get(name), value, f"EXACT_PLAN_{name.upper()}")
    rows = [
        row for row in plan.get("probe_authorities", [])
        if isinstance(row, Mapping)
        and row.get("canonical_parent_key") == q00.get("parent_key")
        and row.get("probe_id") == q00.get("probe_id")
    ]
    if {row.get("arm") for row in rows} != set(Q00_ARMS) or len(rows) != len(Q00_ARMS):
        raise Q00AuthorityError("Q00_EXACT_PLAN_ARM_CLOSURE_INVALID")
    if any(row.get("probe_step") != q00.get("probe_step") for row in rows):
        raise Q00AuthorityError("Q00_EXACT_PLAN_STEP_MISMATCH")
    if any(row.get("snapshot_manifest_sha256") != q00.get("snapshot_manifest_sha256") for row in rows):
        raise Q00AuthorityError("Q00_EXACT_PLAN_SNAPSHOT_MISMATCH")


def _check_snapshot(path: Path, q00: Mapping[str, Any]) -> None:
    snapshot = _load(path)
    _require(snapshot.get("schema"), "STAGE_V_CAUSAL_PROBE_SNAPSHOT_V2", "Q00_SNAPSHOT_SCHEMA")
    _require(snapshot.get("status"), "SEALED_PROSPECTIVE_SNAPSHOT", "Q00_SNAPSHOT_STATUS")
    binding = _mapping(snapshot.get("binding"), "Q00_SNAPSHOT_BINDING")
    expected_binding = {
        "parent_key": q00.get("parent_key"),
        "probe_id": q00.get("probe_id"),
        "source_commit": q00.get("snapshot_source_commit"),
        "source_tree": q00.get("snapshot_source_tree"),
        "step": q00.get("probe_step"),
    }
    for name, expected in expected_binding.items():
        _require(binding.get(name), expected, f"Q00_SNAPSHOT_{name.upper()}")
    _require(snapshot.get("primary_input_authority"), "loaded_frozen_canonical_bytes", "Q00_SNAPSHOT_PRIMARY_AUTHORITY")
    _require(snapshot.get("fresh_render_equality_gate_used"), False, "Q00_SNAPSHOT_FRESH_RENDER_GATE")
    payload = _mapping(snapshot.get("payload"), "Q00_SNAPSHOT_PAYLOAD")
    required = {"episode_start_rng_state", "required_rng_state", "full_simulator_state", "controller_and_wrapper_runtime_state", "clean_reference_action_window"}
    if not required.issubset(payload):
        raise Q00AuthorityError("Q00_SNAPSHOT_RUNTIME_FIELDS_INCOMPLETE")


def _check_provenance(path: Path, source: Mapping[str, Any]) -> None:
    receipt = _load(path)
    _require(receipt.get("schema"), "STAGE_V_EXTERNAL_RUNTIME_PROVENANCE_V1", "PROVENANCE_SCHEMA")
    _require(receipt.get("status"), "PASS_RUNTIME_PROVENANCE_CAPTURED", "PROVENANCE_STATUS")
    _require(receipt.get("runtime_authorized"), False, "PROVENANCE_RUNTIME_AUTHORIZED")
    _require(receipt.get("outcomes_read"), False, "PROVENANCE_OUTCOMES_READ")
    _require(receipt.get("intervention_executed"), False, "PROVENANCE_INTERVENTION")
    _require(receipt.get("protected_counters"), COUNTERS, "PROVENANCE_PROTECTED_COUNTERS")
    worktree = _mapping(receipt.get("source_worktree"), "PROVENANCE_SOURCE_WORKTREE")
    _require(worktree.get("commit"), source.get("runtime_commit"), "PROVENANCE_RUNTIME_COMMIT")
    _require(worktree.get("tree"), source.get("runtime_tree"), "PROVENANCE_RUNTIME_TREE")
    _require(worktree.get("status_porcelain"), "", "PROVENANCE_WORKTREE_DIRTY")


def validate_q00_authority(authority: Mapping[str, Any], *, require_launch: bool = True) -> dict[str, Any]:
    """Validate the Q00 boundary and every bound file before any env is made."""
    _require(authority.get("schema"), SCHEMA, "Q00_AUTHORITY_SCHEMA")
    expected_status = PASS_STATUS if require_launch else DESIGN_STATUS
    _require(authority.get("status"), expected_status, "Q00_AUTHORITY_STATUS")
    _require(authority.get("scope"), "ZERO_TREATMENT_Q00_ONLY", "Q00_AUTHORITY_SCOPE")
    _require(authority.get("authorization_kind"), "Q00_ZERO_TREATMENT_CANARY", "Q00_AUTHORIZATION_KIND")
    _require(authority.get("requires_explicit_owner_authorization"), True, "Q00_OWNER_GATE")
    if require_launch:
        _require(authority.get("canary_authorized"), True, "Q00_CANARY_AUTHORIZED")
        _require(authority.get("runtime_authorized"), True, "Q00_RUNTIME_AUTHORIZED")
        _require(authority.get("owner_authorized"), True, "Q00_OWNER_AUTHORIZED")
        if not str(authority.get("owner_authorization_basis", "")).strip():
            raise Q00AuthorityError("Q00_OWNER_AUTHORIZATION_BASIS_MISSING")
    else:
        _require(authority.get("canary_authorized"), False, "Q00_DESIGN_CANARY_AUTHORIZED")
        _require(authority.get("runtime_authorized"), False, "Q00_DESIGN_RUNTIME_AUTHORIZED")
    _require(authority.get("formal_m4_authorized"), False, "Q00_FORMAL_M4_MUST_REMAIN_FALSE")
    _check_zero_boundary(authority)
    _check_resource_contract(authority)

    q00 = _mapping(authority.get("q00"), "Q00")
    for name, value in (("parent_key", Q00_PARENT_KEY), ("probe_id", Q00_PROBE_ID)):
        _require(q00.get(name), value, f"Q00_{name.upper()}")
    if not isinstance(q00.get("probe_step"), int) or q00["probe_step"] < 0:
        raise Q00AuthorityError("Q00_PROBE_STEP_INVALID")
    _require(q00.get("clean_prefix_replay_allowed"), True, "Q00_CLEAN_PREFIX_REPLAY")
    _require(q00.get("post_snapshot_primary_window_steps"), 0, "Q00_PRIMARY_WINDOW")
    for name in ("snapshot_manifest_sha256", "snapshot_source_commit", "snapshot_source_tree", "exact_plan_manifest_sha256"):
        if not isinstance(q00.get(name), str) or not q00[name]:
            raise Q00AuthorityError(f"Q00_{name.upper()}_MISSING")
    _require_hex(q00.get("snapshot_manifest_sha256"), 64, "Q00_SNAPSHOT_MANIFEST_SHA256")
    _require_hex(q00.get("exact_plan_manifest_sha256"), 64, "Q00_EXACT_PLAN_MANIFEST_SHA256")

    source = _mapping(authority.get("source_binding"), "SOURCE_BINDING")
    for name in ("runtime_commit", "runtime_tree", "snapshot_source_commit", "snapshot_source_tree"):
        if not isinstance(source.get(name), str) or not source[name]:
            raise Q00AuthorityError(f"SOURCE_{name.upper()}_MISSING")
    _require_hex(source.get("runtime_commit"), 40, "SOURCE_RUNTIME_COMMIT")
    _require_hex(source.get("runtime_tree"), 40, "SOURCE_RUNTIME_TREE")
    _require_hex(source.get("snapshot_source_commit"), 40, "SOURCE_SNAPSHOT_COMMIT")
    _require_hex(source.get("snapshot_source_tree"), 40, "SOURCE_SNAPSHOT_TREE")
    _require(source.get("snapshot_source_commit"), q00.get("snapshot_source_commit"), "SOURCE_SNAPSHOT_COMMIT")
    _require(source.get("snapshot_source_tree"), q00.get("snapshot_source_tree"), "SOURCE_SNAPSHOT_TREE")

    bindings = _mapping(authority.get("bindings"), "BINDINGS")
    bound = {name: _bound_file(bindings, name) for name in BOUND_FILES}
    _check_protocol(bound["m4_v2_protocol"][0])
    _check_exact_plan(bound["exact_plan_manifest"][0], q00)
    _require(bound["exact_plan_manifest"][1], q00["exact_plan_manifest_sha256"].lower(), "Q00_EXACT_PLAN_SHA")
    _check_snapshot(bound["snapshot_manifest"][0], q00)
    _require(bound["snapshot_manifest"][1], q00["snapshot_manifest_sha256"].lower(), "Q00_SNAPSHOT_SHA")
    _check_provenance(bound["runtime_provenance_receipt"][0], source)

    return {
        "schema": SCHEMA,
        "status": PASS_STATUS if require_launch else DESIGN_STATUS,
        "authority_sha256": authority_sha256(authority),
        "scope": "ZERO_TREATMENT_Q00_ONLY",
        "parent_key": q00["parent_key"],
        "probe_id": q00["probe_id"],
        "probe_step": q00["probe_step"],
        "snapshot_manifest_sha256": bound["snapshot_manifest"][1],
        "exact_plan_manifest_sha256": bound["exact_plan_manifest"][1],
        "runtime_provenance_receipt_sha256": bound["runtime_provenance_receipt"][1],
        "runtime_commit": source["runtime_commit"],
        "runtime_tree": source["runtime_tree"],
        "protected_counters": dict(COUNTERS),
    }
