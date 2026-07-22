"""Preparation-only contracts for leakage-free inner-train calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

INNER_TRAIN_CALIBRATION_SCHEMA = "FACTORIZED_V2_INNER_TRAIN_CALIBRATION_PREDICTION_V1"
AUTHORIZATION_SCHEMA = "OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V1"
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


def validate_authorization_template(value: Mapping[str, Any]) -> None:
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
    if value.get("execution_authorized") is not False or value.get("formal_selection_eligible") is not False:
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_MUST_BE_DISABLED")
    if any(value.get(name) is not False for name in ("training_authorized", "full_fit_authorized", "attack_authorized", "cal_check_authorized")):
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_ATTACK_OR_TRAINING")
    if value.get("checkpoint_binding_count_required") != 12 or not isinstance(value.get("checkpoint_bindings"), list) or len(value["checkpoint_bindings"]) != 12:
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_CHECKPOINT_COUNT")
    if not isinstance(value.get("forbidden_roots"), list):
        raise CalibrationPlanError("AUTHORIZATION_TEMPLATE_FORBIDDEN_ROOTS")


__all__ = [
    "AUTHORIZATION_SCHEMA", "CalibrationPlanError", "INNER_TRAIN_CALIBRATION_SCHEMA",
    "sha256_file", "validate_authorization_template", "validate_inner_train_plan",
]
