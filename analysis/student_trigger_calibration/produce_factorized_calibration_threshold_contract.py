#!/usr/bin/env python3
"""Produce a schema-exact Factorized V2 calibration-and-threshold V3 contract.

This is a provenance-binding step, not a data-fitting step. It accepts only:
- an independently fitted, sealed calibration contract;
- a completed independent policy-selection threshold contract;
- exact split/checkpoint/source/feature/scheduler/config cross-bindings.

The output is validated against the canonical Codex JSON Schema and consumed by
``FactorizedV2SchedulerAdapter``. Any blocked condition emits a sealed blocker
receipt and exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT / "schemas/factorized_v2_calibration_and_threshold_contract_v3.schema.json"
HEADS = ("grasp", "manipulation", "release")
SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"JSON_MISSING:{path}")
    duplicates: list[str] = []

    def hook(pairs):
        seen = set()
        result = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(str(key))
            seen.add(key)
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON_PARSE_ERROR:{path}:{exc}") from exc
    if duplicates:
        raise SystemExit(f"DUPLICATE_JSON_KEY:{path}:{sorted(set(duplicates))}")
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise SystemExit(f"{label}_SHA_INVALID")
    return value.lower()


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise SystemExit(f"{label}_COMMIT_INVALID")
    return value.lower()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(f"{label}_NUMBER_INVALID")
    return float(value)


def _probability(value: Any, label: str) -> float:
    number = _number(value, label)
    if not 0.0 <= number <= 1.0:
        raise SystemExit(f"{label}_OUT_OF_RANGE")
    return number


def validate_against_schema(contract: dict[str, Any]) -> None:
    schema = _strict_json(SCHEMA_PATH)
    try:
        import jsonschema
    except ImportError:
        _manual_validate(contract, schema)
        return
    try:
        jsonschema.validate(contract, schema)
    except Exception as exc:
        raise SystemExit(f"SCHEMA_VALIDATION_FAILED:{exc}") from exc


def _manual_validate(contract: dict[str, Any], schema: dict[str, Any]) -> None:
    required = set(schema.get("required", []))
    properties = set(schema.get("properties", {}))
    if set(contract) != required or set(contract) - properties:
        raise SystemExit(
            f"TOP_LEVEL_FIELD_SET_INVALID:"
            f"missing={sorted(required - set(contract))}:extra={sorted(set(contract) - properties)}"
        )
    if contract["schema"] != "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3":
        raise SystemExit("BAD_SCHEMA")
    if contract["status"] not in {"DIAGNOSTIC", "AUTHORITATIVE"}:
        raise SystemExit("BAD_STATUS")
    if not re.fullmatch(r"o[0-3]_i[0-2]", contract["split"]):
        raise SystemExit("BAD_SPLIT")
    for field in (
        "checkpoint_sha256",
        "scheduler_source_sha256",
        "structural_config_sha256",
        "feature_order_sha256",
    ):
        _sha(contract[field], field.upper())
    _commit(contract["student_source_commit"], "STUDENT_SOURCE")
    for field in ("training_authorized", "full_fit_authorized", "attack_authorized"):
        if contract[field] is not False:
            raise SystemExit(f"{field}_MUST_BE_FALSE")
    for head in HEADS:
        value = contract[head]
        required_head = set(schema["$defs"]["head"]["required"])
        allowed_head = set(schema["$defs"]["head"]["properties"])
        if set(value) != required_head or set(value) - allowed_head:
            raise SystemExit(f"HEAD_FIELD_SET_INVALID:{head}")
        if value["method"] not in {"RAW", "INTERCEPT_ONLY", "PLATT"}:
            raise SystemExit(f"HEAD_METHOD_INVALID:{head}")
        _number(value["a"], f"{head}.a")
        _number(value["b"], f"{head}.b")
        _probability(value["threshold"], f"{head}.threshold")
        if value["transform"] != "probability=sigmoid(a*raw_logit+b)":
            raise SystemExit(f"HEAD_TRANSFORM_INVALID:{head}")
        if value["method_valid"] is not True or value["transform_valid"] is not True:
            raise SystemExit(f"HEAD_METHOD_INVALID:{head}")
        if not isinstance(value["fit_data_valid"], bool):
            raise SystemExit(f"HEAD_FIT_DATA_FLAG_INVALID:{head}")
        _sha(value["fit_manifest_sha256"], f"{head}.fit_manifest")
        _sha(value["policy_selection_manifest_sha256"], f"{head}.policy_selection_manifest")


def _seal_single_json(output_root: Path, filename: str, value: dict[str, Any]) -> None:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    target = staging / filename
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = staging / "SHA256SUMS"
    sums.write_text(f"{sha256_file(target)}  {filename}\n", encoding="utf-8")
    (staging / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(sums)}  SHA256SUMS\n",
        encoding="utf-8",
    )
    os.replace(staging, output_root)


def blocker(output_root: Path, reason: str, details: dict[str, Any] | None = None) -> int:
    receipt = {
        "schema": "FACTORIZED_V2_CALIBRATION_THRESHOLD_BLOCKER_RECEIPT_V1",
        "status": reason,
        "details": details or {},
        "authoritative_l3": False,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _seal_single_json(output_root, "BLOCKER_RECEIPT.json", receipt)
    print(f"BLOCKER:{reason}:{output_root}")
    return 2


def _calibrator_map(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = calibration.get("calibrators")
    if not isinstance(values, list):
        raise SystemExit("CALIBRATORS_LIST_MISSING")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or value.get("head") not in HEADS:
            raise SystemExit("CALIBRATOR_HEAD_INVALID")
        head = value["head"]
        if head in result:
            raise SystemExit(f"CALIBRATOR_HEAD_DUPLICATE:{head}")
        result[head] = value
    if set(result) != set(HEADS):
        raise SystemExit(f"CALIBRATOR_HEAD_CLOSURE_FAIL:{sorted(set(HEADS) - set(result))}")
    return result


def _threshold_bindings(
    threshold_contract: dict[str, Any],
    split: str,
) -> tuple[dict[str, float], str]:
    if threshold_contract.get("schema") != "FACTORIZED_V2_THRESHOLD_SELECTION_CONTRACT_V2":
        raise SystemExit("THRESHOLD_CONTRACT_SCHEMA_INVALID")
    if threshold_contract.get("status") != "COMPLETE":
        raise SystemExit("THRESHOLD_CONTRACT_NOT_COMPLETE")
    if threshold_contract.get("provenance") != "INDEPENDENT_POLICY_SELECTION":
        raise SystemExit("THRESHOLD_PROVENANCE_INVALID")
    if threshold_contract.get("formal_selection_eligible") is not True:
        raise SystemExit("THRESHOLD_FORMAL_SELECTION_NOT_ELIGIBLE")

    selected = threshold_contract.get("selected_thresholds")
    if not isinstance(selected, dict) or set(selected) != set(HEADS):
        raise SystemExit("THRESHOLD_FIELD_SET_INVALID")
    thresholds = {
        head: _probability(selected[head], f"threshold.{head}")
        for head in HEADS
    }

    checkpoint_by_split = threshold_contract.get("checkpoint_sha256_by_split")
    source_by_split = threshold_contract.get("student_source_commit_by_split")
    feature_by_split = threshold_contract.get("feature_order_sha256_by_split")
    calibration_by_split = threshold_contract.get("calibration_contract_sha256_by_split")
    for label, value in (
        ("checkpoint_sha256_by_split", checkpoint_by_split),
        ("student_source_commit_by_split", source_by_split),
        ("feature_order_sha256_by_split", feature_by_split),
        ("calibration_contract_sha256_by_split", calibration_by_split),
    ):
        if not isinstance(value, dict) or split not in value:
            raise SystemExit(f"THRESHOLD_{label.upper()}_MISSING:{split}")

    _sha(checkpoint_by_split[split], f"threshold.checkpoint.{split}")
    _commit(source_by_split[split], f"threshold.source.{split}")
    _sha(feature_by_split[split], f"threshold.feature.{split}")
    _sha(calibration_by_split[split], f"threshold.calibration.{split}")
    policy_manifest_sha = _sha(
        threshold_contract.get("policy_selection_manifest_sha256"),
        "POLICY_SELECTION_MANIFEST",
    )
    return thresholds, policy_manifest_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-fit-contract", type=Path, required=True)
    parser.add_argument("--threshold-selection-contract", type=Path, required=True)
    parser.add_argument("--scheduler-source-sha256", required=True)
    parser.add_argument("--structural-config-sha256", required=True)
    parser.add_argument("--feature-order-sha256", required=True)
    parser.add_argument("--student-source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS:{output_root}")
    if not re.fullmatch(r"o[0-3]_i[0-2]", args.split):
        return blocker(output_root, "BLOCKED_SPLIT_INVALID", {"split": args.split})

    calibration_path = args.calibration_fit_contract.resolve()
    threshold_path = args.threshold_selection_contract.resolve()
    calibration = _strict_json(calibration_path)
    threshold_contract = _strict_json(threshold_path)

    if calibration.get("schema") != "FACTORIZED_V2_CALIBRATION_CONTRACT_V2":
        return blocker(output_root, "BLOCKED_CALIBRATION_SCHEMA_INVALID")
    if calibration.get("split") != args.split:
        return blocker(
            output_root,
            "BLOCKED_CALIBRATION_SPLIT_MISMATCH",
            {"contract_split": calibration.get("split"), "requested_split": args.split},
        )
    if calibration.get("provenance") != "INDEPENDENT_CALIBRATION":
        return blocker(output_root, "BLOCKED_NOT_INDEPENDENT_CALIBRATION")
    if calibration.get("authoritative") is not True or calibration.get("all_heads_valid") is not True:
        return blocker(output_root, "BLOCKED_CALIBRATION_NOT_AUTHORITATIVE")

    calibrators = _calibrator_map(calibration)
    if not all(value.get("method_valid") is True for value in calibrators.values()):
        return blocker(output_root, "BLOCKED_CALIBRATOR_NOT_VALID")

    try:
        thresholds, policy_manifest_sha = _threshold_bindings(threshold_contract, args.split)
    except SystemExit as exc:
        return blocker(output_root, "BLOCKED_THRESHOLD_CONTRACT_INVALID", {"reason": str(exc)})

    actual_calibration_sha = sha256_file(calibration_path)
    expected_calibration_sha = threshold_contract["calibration_contract_sha256_by_split"][args.split]
    if actual_calibration_sha != expected_calibration_sha.lower():
        return blocker(output_root, "BLOCKED_CALIBRATION_THRESHOLD_BINDING_MISMATCH")

    checkpoint_sha = _sha(calibration.get("checkpoint_sha256"), "CALIBRATION_CHECKPOINT")
    student_commit = _commit(args.student_source_commit, "STUDENT_SOURCE")
    scheduler_sha = _sha(args.scheduler_source_sha256, "SCHEDULER_SOURCE")
    structural_sha = _sha(args.structural_config_sha256, "STRUCTURAL_CONFIG")
    feature_sha = _sha(args.feature_order_sha256, "FEATURE_ORDER")

    for label, observed, expected in (
        (
            "checkpoint",
            checkpoint_sha,
            threshold_contract["checkpoint_sha256_by_split"][args.split].lower(),
        ),
        (
            "student_source_commit",
            student_commit,
            threshold_contract["student_source_commit_by_split"][args.split].lower(),
        ),
        (
            "feature_order",
            feature_sha,
            threshold_contract["feature_order_sha256_by_split"][args.split].lower(),
        ),
        (
            "scheduler_source",
            scheduler_sha,
            str(threshold_contract.get("scheduler_source_sha256", "")).lower(),
        ),
        (
            "structural_config",
            structural_sha,
            str(threshold_contract.get("structural_config_sha256", "")).lower(),
        ),
    ):
        if observed != expected:
            return blocker(
                output_root,
                f"BLOCKED_{label.upper()}_BINDING_MISMATCH",
                {"observed": observed, "expected": expected},
            )

    fit_manifest_sha = _sha(calibration.get("fit_manifest_sha256"), "FIT_MANIFEST")
    heads: dict[str, dict[str, Any]] = {}
    for head in HEADS:
        calibrator = calibrators[head]
        method = calibrator.get("method")
        if method not in {"RAW", "INTERCEPT_ONLY", "PLATT"}:
            return blocker(output_root, f"BLOCKED_METHOD_INVALID_{head.upper()}")
        a = _number(calibrator.get("a"), f"{head}.a")
        b = _number(calibrator.get("b"), f"{head}.b")
        fit_data_valid = (
            calibrator.get("method_valid") is True
            and isinstance(calibrator.get("n_fit_pos"), int)
            and isinstance(calibrator.get("n_fit_neg"), int)
            and calibrator["n_fit_pos"] > 0
            and calibrator["n_fit_neg"] > 0
        )
        if not fit_data_valid:
            return blocker(output_root, f"BLOCKED_FIT_DATA_INVALID_{head.upper()}")
        heads[head] = {
            "method": method,
            "a": a,
            "b": b,
            "threshold": thresholds[head],
            "transform": "probability=sigmoid(a*raw_logit+b)",
            "method_valid": True,
            "transform_valid": True,
            "fit_data_valid": True,
            "provenance_class": "INDEPENDENT_CALIBRATION",
            "fit_manifest_sha256": fit_manifest_sha,
            "policy_selection_manifest_sha256": policy_manifest_sha,
        }

    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
        "status": "AUTHORITATIVE",
        "split": args.split,
        "checkpoint_sha256": checkpoint_sha,
        "scheduler_source_sha256": scheduler_sha,
        "structural_config_sha256": structural_sha,
        "student_source_commit": student_commit,
        "feature_order_sha256": feature_sha,
        "calibration_fit_authoritative": True,
        "threshold_selection_authoritative": True,
        "l3_evaluation_eligible": True,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "grasp": heads["grasp"],
        "manipulation": heads["manipulation"],
        "release": heads["release"],
    }
    validate_against_schema(contract)
    _seal_single_json(
        output_root,
        "calibration_and_threshold_contract.json",
        contract,
    )
    print(f"Contract V3:{output_root}:status=AUTHORITATIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
