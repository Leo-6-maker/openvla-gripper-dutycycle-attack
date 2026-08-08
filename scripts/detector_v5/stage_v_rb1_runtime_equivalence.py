"""Fail-closed RB1 runtime-equivalence receipt and trace checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_runner_binding_protocol import CONTRACT_FIELDS, canonical_sha256
except ImportError:  # direct execution on the server
    from stage_v_runner_binding_protocol import CONTRACT_FIELDS, canonical_sha256


SCHEMA = "STAGE_V_RB1_RUNTIME_EQUIVALENCE_PROTOCOL_V1"
RECEIPT_SCHEMA = "STAGE_V_RB1_RUNTIME_RECEIPT_V1"
IDENTITY_FIELDS = ("canonical_parent_key", "suite", "task_index", "state_index")
COMMON_TRACE_HASH_FIELDS = (
    "policy_token_trace_sha256",
    "postprocessed_action_trace_sha256",
    "observation_trace_sha256",
    "physical_state_trace_sha256",
)
NOOP_TRACE_HASH_FIELDS = ("snapshot_restore_trace_sha256", "noop_action_trace_sha256")
BOUNDARY_FIELDS = (
    "eval160_reads",
    "protected_eval_reads",
    "vis_pgd_attack_rollouts",
    "attack_rollouts",
    "intervention_applied_steps",
    "counterfactual_open_steps",
)
HEX64 = set("0123456789abcdef")


class RuntimeEquivalenceError(ValueError):
    """Raised when an RB1 receipt or pair is incomplete or non-equivalent."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeEquivalenceError(f"{label} must be a JSON object")
    return dict(value)


def _hex64(value: Any, label: str) -> None:
    text = str(value)
    if len(text) != 64 or set(text.lower()) - HEX64:
        raise RuntimeEquivalenceError(f"{label} must be a lowercase SHA256")


def _required_list(protocol: Mapping[str, Any], key: str, expected: list[str]) -> None:
    if protocol.get(key) != expected:
        raise RuntimeEquivalenceError(f"RB1_PROTOCOL_{key.upper()}_MISMATCH")


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _object(protocol, "protocol")
    if protocol.get("schema") != SCHEMA:
        raise RuntimeEquivalenceError("RB1_PROTOCOL_SCHEMA_MISMATCH")
    if not str(protocol.get("status", "")).startswith("FROZEN_RUNTIME_GATE"):
        raise RuntimeEquivalenceError("RB1_PROTOCOL_NOT_FROZEN")
    if protocol.get("equality_mode") != "EXACT_CANONICAL_JSON_AND_TRACE_SHA256":
        raise RuntimeEquivalenceError("RB1_PROTOCOL_EQUALITY_MODE_NOT_EXACT")
    _required_list(protocol, "contract_fields", list(CONTRACT_FIELDS))
    _required_list(protocol, "identity_fields", list(IDENTITY_FIELDS))
    _required_list(protocol, "common_trace_hash_fields", list(COMMON_TRACE_HASH_FIELDS))
    _required_list(protocol, "noop_trace_hash_fields", list(NOOP_TRACE_HASH_FIELDS))
    _required_list(protocol, "protected_boundaries", list(BOUNDARY_FIELDS))
    if protocol.get("initial_state_exact_required") is not True:
        raise RuntimeEquivalenceError("RB1_INITIAL_STATE_EXACT_GATE_MISSING")
    if protocol.get("action_trace_exact_required") is not True:
        raise RuntimeEquivalenceError("RB1_ACTION_TRACE_EXACT_GATE_MISSING")
    if protocol.get("token_trace_exact_required") is not True:
        raise RuntimeEquivalenceError("RB1_TOKEN_TRACE_EXACT_GATE_MISSING")
    if protocol.get("tolerance_allowed") is not False:
        raise RuntimeEquivalenceError("RB1_TOLERANCE_MUST_BE_DISABLED")
    if protocol.get("independent_recompute_required") is not True:
        raise RuntimeEquivalenceError("RB1_INDEPENDENT_RECOMPUTE_GATE_MISSING")
    return protocol


def _validate_contract(receipt: Mapping[str, Any]) -> dict[str, Any]:
    contract = _object(receipt.get("execution_contract"), "execution_contract")
    missing = [field for field in CONTRACT_FIELDS if field not in contract]
    extra = sorted(set(contract) - set(CONTRACT_FIELDS))
    if missing:
        raise RuntimeEquivalenceError("RB1_CONTRACT_FIELDS_MISSING:" + ",".join(missing))
    if extra:
        raise RuntimeEquivalenceError("RB1_CONTRACT_FIELDS_UNEXPECTED:" + ",".join(extra))
    declared = receipt.get("execution_contract_sha256")
    if declared != canonical_sha256(contract):
        raise RuntimeEquivalenceError("RB1_EXECUTION_CONTRACT_DIGEST_MISMATCH")
    if receipt.get("clean_core_sha256") != contract["clean_core_sha256"]:
        raise RuntimeEquivalenceError("RB1_CLEAN_CORE_DIGEST_MISMATCH")
    return contract


def _validate_recompute(receipt: Mapping[str, Any]) -> None:
    audit = _object(receipt.get("independent_recompute"), "independent_recompute")
    if audit.get("status") != "PASS" or audit.get("recomputed") is not True:
        raise RuntimeEquivalenceError("RB1_INDEPENDENT_RECOMPUTE_MISSING")
    for field in ("auditor_source_commit", "auditor_source_tree"):
        if not audit.get(field):
            raise RuntimeEquivalenceError("RB1_INDEPENDENT_AUDITOR_BINDING_MISSING:" + field)
    for field in ("auditor_sha256", "protocol_sha256"):
        _hex64(audit.get(field), "independent_recompute." + field)


def _validate_trace_hashes(receipt: Mapping[str, Any], scope: str) -> dict[str, Any]:
    hashes = _object(receipt.get("trace_hashes"), "trace_hashes")
    required = list(COMMON_TRACE_HASH_FIELDS)
    if scope == "NOOP_CONTINUATION":
        required += list(NOOP_TRACE_HASH_FIELDS)
    if set(hashes) != set(required):
        raise RuntimeEquivalenceError("RB1_TRACE_HASH_FIELD_SET_MISMATCH")
    for field in required:
        _hex64(hashes.get(field), "trace_hashes." + field)
    return hashes


def _validate_artifact_manifest(receipt: Mapping[str, Any], scope: str) -> dict[str, Any]:
    artifacts = _object(receipt.get("trace_artifacts"), "trace_artifacts")
    required = ["initial_state", "policy_token_trace", "postprocessed_action_trace", "observation_trace", "physical_state_trace"]
    if scope == "NOOP_CONTINUATION":
        required += ["snapshot_restore_trace", "noop_action_trace"]
    if set(artifacts) != set(required):
        raise RuntimeEquivalenceError("RB1_TRACE_ARTIFACT_FIELD_SET_MISMATCH")
    for name in required:
        item = _object(artifacts.get(name), "trace_artifacts." + name)
        path = Path(str(item.get("path", "")))
        if not item.get("path") or path.is_absolute() or ".." in path.parts:
            raise RuntimeEquivalenceError("RB1_TRACE_ARTIFACT_PATH_NOT_RELATIVE:" + name)
        _hex64(item.get("sha256"), "trace_artifacts." + name + ".sha256")
    return artifacts


def validate_receipt(
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    require_independent_recompute: bool = True,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol)
    receipt = _object(receipt, "receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeEquivalenceError("RB1_RECEIPT_SCHEMA_MISMATCH")
    scope = receipt.get("comparison_scope")
    if scope not in protocol.get("comparison_scopes", []):
        raise RuntimeEquivalenceError("RB1_COMPARISON_SCOPE_INVALID")
    if receipt.get("mode") not in protocol.get("modes", []):
        raise RuntimeEquivalenceError("RB1_RECEIPT_MODE_INVALID")
    for field in IDENTITY_FIELDS:
        if field not in receipt or receipt[field] in (None, ""):
            raise RuntimeEquivalenceError("RB1_IDENTITY_MISSING:" + field)
    _validate_contract(receipt)
    _hex64(receipt.get("initial_state_sha256"), "initial_state_sha256")
    step_count = receipt.get("trace_step_count")
    termination_step = receipt.get("termination_step")
    if not isinstance(step_count, int) or step_count <= 0:
        raise RuntimeEquivalenceError("RB1_TRACE_STEP_COUNT_INVALID")
    if not isinstance(termination_step, int) or termination_step < 0 or termination_step >= step_count:
        raise RuntimeEquivalenceError("RB1_TERMINATION_STEP_INVALID")
    if not receipt.get("terminal_outcome"):
        raise RuntimeEquivalenceError("RB1_TERMINAL_OUTCOME_MISSING")
    _validate_trace_hashes(receipt, str(scope))
    _validate_artifact_manifest(receipt, str(scope))
    if require_independent_recompute:
        _validate_recompute(receipt)
    for field in BOUNDARY_FIELDS:
        if receipt.get(field, 0) != 0:
            raise RuntimeEquivalenceError("RB1_PROTECTED_BOUNDARY:" + field)
    if receipt.get("clean_prefix_shared") is not True:
        raise RuntimeEquivalenceError("RB1_CLEAN_PREFIX_NOT_SHARED")
    if scope == "NOOP_CONTINUATION":
        if not isinstance(receipt.get("probe_step"), int) or receipt["probe_step"] < 0:
            raise RuntimeEquivalenceError("RB1_PROBE_STEP_INVALID")
        _hex64(receipt.get("probe_state_sha256"), "probe_state_sha256")
    return receipt


def verify_artifact_files(
    receipt: Mapping[str, Any],
    artifact_root: Path,
    protocol: Mapping[str, Any],
    *,
    require_independent_recompute: bool = True,
) -> None:
    receipt = validate_receipt(receipt, protocol, require_independent_recompute=require_independent_recompute)
    root = artifact_root.resolve()
    trace_to_artifact = {
        "policy_token_trace_sha256": "policy_token_trace",
        "postprocessed_action_trace_sha256": "postprocessed_action_trace",
        "observation_trace_sha256": "observation_trace",
        "physical_state_trace_sha256": "physical_state_trace",
        "snapshot_restore_trace_sha256": "snapshot_restore_trace",
        "noop_action_trace_sha256": "noop_action_trace",
    }
    for trace_field, artifact_name in trace_to_artifact.items():
        if trace_field in receipt["trace_hashes"] and receipt["trace_hashes"][trace_field] != receipt["trace_artifacts"][artifact_name]["sha256"]:
            raise RuntimeEquivalenceError("RB1_TRACE_HASH_MANIFEST_MISMATCH:" + trace_field)
    for name, item in receipt["trace_artifacts"].items():
        path = (root / str(item["path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeEquivalenceError("RB1_TRACE_ARTIFACT_MISSING_OR_OUTSIDE_ROOT:" + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeEquivalenceError("RB1_TRACE_ARTIFACT_SHA256_MISMATCH:" + name)
        if name == "initial_state":
            try:
                initial = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeEquivalenceError("RB1_INITIAL_STATE_ARTIFACT_INVALID") from exc
            if initial.get("initial_state_sha256") != receipt.get("initial_state_sha256"):
                raise RuntimeEquivalenceError("RB1_INITIAL_STATE_ARTIFACT_IDENTITY_MISMATCH")
            initial_identity = initial.get("identity")
            if not isinstance(initial_identity, Mapping):
                raise RuntimeEquivalenceError("RB1_INITIAL_STATE_ARTIFACT_IDENTITY_MISSING")
            if any(initial_identity.get(field) != receipt.get(field) for field in IDENTITY_FIELDS):
                raise RuntimeEquivalenceError("RB1_INITIAL_STATE_ARTIFACT_PARENT_IDENTITY_MISMATCH")


def validate_pair(left: Mapping[str, Any], right: Mapping[str, Any], protocol: Mapping[str, Any], pair_kind: str) -> dict[str, Any]:
    protocol = validate_protocol(protocol)
    pair = _object(protocol.get("pair_kinds"), "pair_kinds").get(pair_kind)
    pair = _object(pair, "pair_kind")
    left = validate_receipt(left, protocol)
    right = validate_receipt(right, protocol)
    if left.get("mode") != pair.get("left_mode") or right.get("mode") != pair.get("right_mode"):
        raise RuntimeEquivalenceError("RB1_PAIR_MODE_MISMATCH")
    if left.get("comparison_scope") != pair.get("scope") or right.get("comparison_scope") != pair.get("scope"):
        raise RuntimeEquivalenceError("RB1_PAIR_SCOPE_MISMATCH")
    compare_fields = list(IDENTITY_FIELDS) + [
        "clean_core_sha256", "execution_contract_sha256", "initial_state_sha256",
        "trace_step_count", "termination_step", "terminal_outcome",
    ]
    mismatches = [field for field in compare_fields if left.get(field) != right.get(field)]
    if left["trace_hashes"] != right["trace_hashes"]:
        mismatches.append("trace_hashes")
    if pair_kind == "RB1B_NOOP_CONTINUATION":
        for field in ("probe_step", "probe_state_sha256"):
            if left.get(field) != right.get(field):
                mismatches.append(field)
    if mismatches:
        raise RuntimeEquivalenceError("RB1_RUNTIME_TRACE_MISMATCH:" + ",".join(mismatches))
    return {
        "schema": "STAGE_V_RB1_RUNTIME_PAIR_AUDIT_V1",
        "verdict": "PASS",
        "pair_kind": pair_kind,
        "contract_identity": "PASS",
        "initial_state_identity": "PASS",
        "runtime_trace_equivalence": "PASS",
        "noop_restore_equivalence": "PASS" if pair_kind == "RB1B_NOOP_CONTINUATION" else "NOT_APPLICABLE",
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--left-receipt", type=Path, required=True)
    parser.add_argument("--right-receipt", type=Path, required=True)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--pair-kind", choices=("RB1A_CLEAN_PATH", "RB1B_NOOP_CONTINUATION"), required=True)
    args = parser.parse_args()
    try:
        protocol = _load(args.protocol)
        left = _load(args.left_receipt)
        right = _load(args.right_receipt)
        verify_artifact_files(left, args.left_root, protocol)
        verify_artifact_files(right, args.right_root, protocol)
        result = validate_pair(left, right, protocol, args.pair_kind)
    except (OSError, json.JSONDecodeError, RuntimeEquivalenceError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
