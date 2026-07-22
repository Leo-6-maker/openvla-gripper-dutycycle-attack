"""Preparation-only contracts for leakage-free inner-train calibration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

INNER_TRAIN_CALIBRATION_SCHEMA = "FACTORIZED_V2_INNER_TRAIN_CALIBRATION_PREDICTION_V1"
AUTHORIZATION_SCHEMA = "OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V1"
PLAN_SCHEMA_V2 = "OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_PLAN_V2"
AUTHORIZATION_SCHEMA_V2 = "OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V2"
EXECUTION_RECEIPT_SCHEMA = "FACTORIZED_V2_INFERENCE_EXECUTION_RECEIPT_V1"
EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
ALLOWED_PREDICTOR_MODULES = ("scripts.detector_v5.predict_factorized_v2_inner_cv",)
PROTECTED_MARKERS = ("fit-dev", "fit_dev", "cal", "check", "cs200", "attack")


class CalibrationPlanError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise CalibrationPlanError(code)
    return value.lower()


def _identities(value: Any, code: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise CalibrationPlanError(code)
    if len(set(value)) != len(value):
        raise CalibrationPlanError(f"{code}_DUPLICATE")
    return value


def validate_inner_train_plan(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    forbidden_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate the 12-run input closure without loading a model or data."""

    if len(checkpoints) != 12:
        raise CalibrationPlanError("CHECKPOINT_COUNT_MUST_BE_12")
    seen: set[tuple[Any, Any, Any]] = set()
    all_train: set[str] = set()
    for item in checkpoints:
        coord = (item.get("outer_fold"), item.get("inner_fold"), item.get("seed"))
        if coord in seen:
            raise CalibrationPlanError("DUPLICATE_CHECKPOINT_COORDINATE")
        seen.add(coord)
        _sha(item.get("checkpoint_sha256"), "CHECKPOINT_SHA_INVALID")
        _sha(item.get("identity_manifest_sha256"), "IDENTITY_MANIFEST_SHA_INVALID")
        train = set(_identities(item.get("inner_train_identities"), "INNER_TRAIN_IDENTITIES_INVALID"))
        heldout = set(_identities(item.get("heldout_identities"), "HELDOUT_IDENTITIES_INVALID"))
        if train & heldout:
            raise CalibrationPlanError("INNER_TRAIN_HELDOUT_OVERLAP")
        all_train.update(train)
        path_values = [str(item.get(name, "")) for name in ("checkpoint_root", "identity_manifest_root", "feature_root")]
        for root in [*path_values, *map(str, forbidden_roots)]:
            lowered = root.replace("\\", "/").lower()
            if any(marker in lowered.split("/") for marker in PROTECTED_MARKERS):
                raise CalibrationPlanError("PROTECTED_SPLIT_PATH")
    return {
        "schema": INNER_TRAIN_CALIBRATION_SCHEMA,
        "checkpoint_count": len(checkpoints),
        "coordinate_count": len(seen),
        "inner_train_identity_union_count": len(all_train),
        "validation_or_cal_read": False,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    }


def validate_authorization_template(value: Mapping[str, Any], *, allow_execution: bool = False) -> None:
    required = {
        "schema", "status", "execution_authorized", "formal_selection_eligible",
        "training_authorized", "full_fit_authorized", "attack_authorized",
        "cal_check_authorized", "checkpoint_bindings",
        "checkpoint_binding_count_required",
        "feature_input_seals", "predictor_source", "allowed_output_roots",
        "forbidden_roots",
    }
    if set(value) != required or value.get("schema") != AUTHORIZATION_SCHEMA:
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_SCHEMA")
    if value.get("execution_authorized") is not (True if allow_execution else False) or value.get("formal_selection_eligible") is not False:
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_MUST_BE_DISABLED")
    if any(value.get(name) is not False for name in ("training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized")):
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_ATTACK_OR_TRAINING")
    if value.get("checkpoint_binding_count_required") != 12 or not isinstance(value.get("checkpoint_bindings"), list) or len(value["checkpoint_bindings"]) != 12:
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_CHECKPOINT_COUNT")
    if not isinstance(value.get("forbidden_roots"), list):
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_FORBIDDEN_ROOTS")


def _required_sha(value: Any, code: str) -> str:
    return _sha(value, code)


def _required_commit(value: Any, code: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise CalibrationPlanError(code)
    return value.lower()


def _protected_path(path: Any) -> bool:
    lowered = str(path or "").replace("\\", "/").lower()
    return any(marker in lowered.split("/") for marker in PROTECTED_MARKERS)


def validate_structured_inner_plan(value: Mapping[str, Any], *, execute: bool = False) -> dict[str, Any]:
    """Validate a closed plan; execution never accepts caller-provided commands."""
    required = {"schema", "status", "jobs", "forbidden_roots", "formal_selection_eligible", "training_authorized", "attack_authorized"}
    if set(value) != required or value.get("schema") != PLAN_SCHEMA_V2 or value.get("status") not in {"PLAN_ONLY", "SEALED_EXECUTION_PLAN"}:
        raise CalibrationPlanError("STRUCTURED_PLAN_SCHEMA")
    if any(value.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "attack_authorized")):
        raise CalibrationPlanError("STRUCTURED_PLAN_AUTHORIZATION")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or tuple(sorted(job.get("split") for job in jobs if isinstance(job, Mapping))) != EXPECTED_SPLITS:
        raise CalibrationPlanError("STRUCTURED_PLAN_EXACT_SPLITS")
    job_fields = {
        "split", "outer_fold", "inner_fold", "seed", "predictor_module", "predictor_script",
        "checkpoint_path", "checkpoint_sha256", "identity_manifest_path", "identity_manifest_sha256",
        "inner_train_identities", "heldout_identities", "feature_root", "feature_seal_sha256",
        "output_root", "expected_output_schema",
    }
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, Mapping) or set(job) != job_fields:
            raise CalibrationPlanError("STRUCTURED_JOB_SCHEMA")
        split = str(job["split"])
        if split in seen or split not in EXPECTED_SPLITS or job["predictor_module"] not in ALLOWED_PREDICTOR_MODULES:
            raise CalibrationPlanError("STRUCTURED_JOB_COORDINATE_OR_MODULE")
        seen.add(split)
        if job["expected_output_schema"] != INNER_TRAIN_CALIBRATION_SCHEMA:
            raise CalibrationPlanError("STRUCTURED_OUTPUT_SCHEMA")
        train = set(_identities(job["inner_train_identities"], "INNER_TRAIN_IDENTITIES_INVALID"))
        heldout = set(_identities(job["heldout_identities"], "HELDOUT_IDENTITIES_INVALID"))
        if train & heldout:
            raise CalibrationPlanError("INNER_TRAIN_HELDOUT_OVERLAP")
        if any(_protected_path(job.get(name)) for name in ("checkpoint_path", "identity_manifest_path", "feature_root", "output_root")):
            raise CalibrationPlanError("PROTECTED_SPLIT_PATH")
        if execute:
            checkpoint = Path(str(job["checkpoint_path"])).resolve()
            manifest = Path(str(job["identity_manifest_path"])).resolve()
            feature_root = Path(str(job["feature_root"])).resolve()
            output_root = Path(str(job["output_root"])).resolve()
            if not checkpoint.is_file() or sha256_file(checkpoint) != str(job["checkpoint_sha256"]).lower():
                raise CalibrationPlanError("CHECKPOINT_BINDING_INVALID")
            if not manifest.is_file() or sha256_file(manifest) != str(job["identity_manifest_sha256"]).lower():
                raise CalibrationPlanError("IDENTITY_MANIFEST_BINDING_INVALID")
            if not feature_root.is_dir():
                raise CalibrationPlanError("FEATURE_ROOT_MISSING")
            if output_root.exists() or not str(output_root):
                raise CalibrationPlanError("OUTPUT_ROOT_INVALID")
            if str(job["predictor_script"]) and not Path(str(job["predictor_script"])).is_file():
                raise CalibrationPlanError("PREDICTOR_SCRIPT_MISSING")
            _required_sha(job["feature_seal_sha256"], "FEATURE_SEAL_INVALID")
    return {"schema": PLAN_SCHEMA_V2, "split_count": len(jobs), "split_names": list(EXPECTED_SPLITS), "formal_selection_eligible": False, "training_authorized": False, "attack_authorized": False}


def validate_execution_authorization_v2(value: Mapping[str, Any], plan: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    required = {"schema", "status", "execution_authorized", "formal_selection_eligible", "training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized", "bindings", "allowed_output_roots", "forbidden_roots"}
    if set(value) != required or value.get("schema") != AUTHORIZATION_SCHEMA_V2 or value.get("status") != "SEALED_EXECUTION_AUTHORIZATION":
        raise CalibrationPlanError("EXECUTION_AUTHORIZATION_SCHEMA")
    if value.get("execution_authorized") is not True or any(value.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized")):
        raise CalibrationPlanError("EXECUTION_AUTHORIZATION_FLAGS")
    bindings = value.get("bindings")
    jobs = plan.get("jobs") if isinstance(plan, Mapping) else None
    if not isinstance(bindings, list) or not isinstance(jobs, list) or len(bindings) != 12 or len(jobs) != 12:
        raise CalibrationPlanError("EXECUTION_BINDING_COUNT")
    allowed = {str(Path(item).resolve()) for item in value.get("allowed_output_roots", [])}
    if str(output_root.resolve()) not in allowed:
        raise CalibrationPlanError("OUTPUT_ROOT_NOT_ALLOWLISTED")
    if not isinstance(value.get("forbidden_roots"), list):
        raise CalibrationPlanError("FORBIDDEN_ROOTS_INVALID")
    expected_by_split = {job["split"]: job for job in jobs}
    seen: set[str] = set()
    for binding in bindings:
        fields = {"split", "outer_fold", "inner_fold", "seed", "checkpoint_path", "checkpoint_sha256", "identity_manifest_path", "identity_manifest_sha256", "feature_root", "feature_seal_sha256", "predictor_module", "predictor_source_sha256", "output_root"}
        if not isinstance(binding, Mapping) or set(binding) != fields:
            raise CalibrationPlanError("EXECUTION_BINDING_SCHEMA")
        split = binding["split"]
        if split in seen or split not in expected_by_split:
            raise CalibrationPlanError("EXECUTION_BINDING_SPLIT")
        seen.add(split)
        job = expected_by_split[split]
        for field in fields - {"split"}:
            if binding.get(field) in (None, ""):
                raise CalibrationPlanError("EXECUTION_BINDING_NULL")
        for field in ("checkpoint_sha256", "identity_manifest_sha256", "feature_seal_sha256", "predictor_source_sha256"):
            _required_sha(binding[field], f"{field.upper()}_INVALID")
        if binding["output_root"] != job["output_root"] or binding["checkpoint_path"] != job["checkpoint_path"] or binding["predictor_module"] != job["predictor_module"]:
            raise CalibrationPlanError("EXECUTION_BINDING_PLAN_MISMATCH")
    if seen != set(EXPECTED_SPLITS):
        raise CalibrationPlanError("EXECUTION_BINDING_EXACT_SPLITS")
    return {"status": "PASS", "execution_authorized": True, "split_count": len(bindings), "formal_selection_eligible": False}


def validate_execution_authorization_template_v2(value: Mapping[str, Any]) -> None:
    required = {"schema", "status", "execution_authorized", "formal_selection_eligible", "training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized", "bindings", "allowed_output_roots", "forbidden_roots"}
    if set(value) != required or value.get("schema") != AUTHORIZATION_SCHEMA_V2 or value.get("status") != "TEMPLATE_ONLY":
        raise CalibrationPlanError("EXECUTION_TEMPLATE_SCHEMA")
    if value.get("execution_authorized") is not False or any(value.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized")):
        raise CalibrationPlanError("EXECUTION_TEMPLATE_FLAGS")
    if not isinstance(value.get("bindings"), list) or len(value["bindings"]) != 12 or not isinstance(value.get("allowed_output_roots"), list) or not isinstance(value.get("forbidden_roots"), list):
        raise CalibrationPlanError("EXECUTION_TEMPLATE_SHAPE")


__all__ = [
    "ALLOWED_PREDICTOR_MODULES", "AUTHORIZATION_SCHEMA", "AUTHORIZATION_SCHEMA_V2",
    "CalibrationPlanError", "EXPECTED_SPLITS", "EXECUTION_RECEIPT_SCHEMA",
    "INNER_TRAIN_CALIBRATION_SCHEMA", "PLAN_SCHEMA_V2", "sha256_file",
    "validate_authorization_template", "validate_execution_authorization_v2",
    "validate_execution_authorization_template_v2", "validate_inner_train_plan", "validate_structured_inner_plan",
]
