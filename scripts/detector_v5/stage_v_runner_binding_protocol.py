"""Fail-closed binding checks for Stage V clean qualification and science runs.

The protocol freezes the *identity* of the clean execution core.  It does not
implement a second evaluator: both callers must emit the same receipt fields
and the same canonical digests before a new qualification or formal map may
start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "STAGE_V_RUNNER_BINDING_PROTOCOL_V1"
RECEIPT_SCHEMA = "STAGE_V_RUNNER_BINDING_RECEIPT_V1"

CONTRACT_FIELDS = (
    "clean_core_sha256",
    "source_commit",
    "source_tree",
    "runner_sha256",
    "model_tree_sha256",
    "processor_sha256",
    "tokenizer_sha256",
    "prompt_template",
    "unnorm_key",
    "seed",
    "num_steps_wait",
    "suite_horizon",
    "termination_predicate",
    "success_predicate",
    "reset_restore_contract",
    "action_decode_contract",
    "action_postprocess_contract",
    "gripper_semantics",
    "initial_state_hash_algorithm",
    "initial_state_identity_schema",
)

BOUNDARY_FIELDS = (
    "eval160_reads",
    "protected_eval_reads",
    "vis_pgd_attack_rollouts",
)


class RunnerBindingError(ValueError):
    """Raised when a runner binding is missing, inconsistent, or unverifiable."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerBindingError(f"{label} must be a JSON object")
    return dict(value)


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _object(protocol, "protocol")
    if protocol.get("schema") != SCHEMA:
        raise RunnerBindingError("RUNNER_BINDING_PROTOCOL_SCHEMA_MISMATCH")
    if protocol.get("equality_mode") != "EXACT_CANONICAL_JSON":
        raise RunnerBindingError("RUNNER_BINDING_EQUALITY_MODE_NOT_EXACT")
    fields = protocol.get("contract_fields")
    if fields != list(CONTRACT_FIELDS):
        raise RunnerBindingError("RUNNER_BINDING_CONTRACT_FIELD_ORDER_OR_SET_MISMATCH")
    if protocol.get("shared_clean_core_required") is not True:
        raise RunnerBindingError("SHARED_CLEAN_CORE_REQUIREMENT_MISSING")
    if protocol.get("protected_boundaries") != list(BOUNDARY_FIELDS):
        raise RunnerBindingError("RUNNER_BINDING_BOUNDARY_SCHEMA_MISMATCH")
    return protocol


def _contract(receipt: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _object(receipt, "receipt")
    value = receipt.get("execution_contract")
    contract = _object(value, "execution_contract")
    missing = [field for field in CONTRACT_FIELDS if field not in contract]
    if missing:
        raise RunnerBindingError(f"RUNNER_BINDING_CONTRACT_FIELDS_MISSING:{','.join(missing)}")
    extra = sorted(set(contract) - set(CONTRACT_FIELDS))
    if extra:
        raise RunnerBindingError(f"RUNNER_BINDING_CONTRACT_FIELDS_UNEXPECTED:{','.join(extra)}")
    declared = receipt.get("execution_contract_sha256")
    actual = canonical_sha256(contract)
    if declared != actual:
        raise RunnerBindingError("RUNNER_BINDING_EXECUTION_CONTRACT_DIGEST_MISMATCH")
    if receipt.get("clean_core_sha256") != contract["clean_core_sha256"]:
        raise RunnerBindingError("RUNNER_BINDING_CLEAN_CORE_DIGEST_MISMATCH")
    return contract


def validate_receipt(receipt: Mapping[str, Any], *, expected_mode: str) -> dict[str, Any]:
    receipt = _object(receipt, "receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RunnerBindingError("RUNNER_BINDING_RECEIPT_SCHEMA_MISMATCH")
    if receipt.get("mode") != expected_mode:
        raise RunnerBindingError(f"RUNNER_BINDING_MODE_MISMATCH:{expected_mode}")
    _contract(receipt)
    for field in BOUNDARY_FIELDS:
        if receipt.get(field, 0) != 0:
            raise RunnerBindingError(f"RUNNER_BINDING_PROTECTED_BOUNDARY:{field}")
    if receipt.get("clean_prefix_shared") is not True:
        raise RunnerBindingError("RUNNER_BINDING_CLEAN_PREFIX_NOT_SHARED")
    return receipt


def validate_pair(
    qualification_receipt: Mapping[str, Any],
    science_receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_protocol(protocol)
    qualification = validate_receipt(qualification_receipt, expected_mode="CLEAN_QUALIFICATION")
    science = validate_receipt(science_receipt, expected_mode="COUNTERFACTUAL")
    q_contract = qualification["execution_contract"]
    s_contract = science["execution_contract"]
    mismatches = [field for field in CONTRACT_FIELDS if q_contract[field] != s_contract[field]]
    if mismatches:
        raise RunnerBindingError("RUNNER_BINDING_MISMATCH:" + ",".join(mismatches))
    if qualification["clean_core_sha256"] != science["clean_core_sha256"]:
        raise RunnerBindingError("RUNNER_BINDING_CLEAN_CORE_MISMATCH")
    if qualification["execution_contract_sha256"] != science["execution_contract_sha256"]:
        raise RunnerBindingError("RUNNER_BINDING_EXECUTION_CONTRACT_MISMATCH")
    return {
        "schema": "STAGE_V_RUNNER_BINDING_PAIR_AUDIT_V1",
        "verdict": "PASS",
        "mismatches": [],
        "clean_core_sha256": qualification["clean_core_sha256"],
        "execution_contract_sha256": qualification["execution_contract_sha256"],
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--science-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_pair(_load(args.qualification_receipt), _load(args.science_receipt), _load(args.protocol))
    except (OSError, json.JSONDecodeError, RunnerBindingError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
