"""Prospective physical telemetry and blinded manual-audit schemas."""

from __future__ import annotations

from typing import Any, Mapping

from .contract import DOSES, StageZHold
from .matrix import StageZArm


TELEMETRY_SCHEMA = "STAGE_Z_PHYSICAL_TELEMETRY_V1"
MANUAL_AUDIT_SCHEMA = "STAGE_Z_BLINDED_MANUAL_AUDIT_V1"
MANUAL_LABELS = (
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
)


def make_telemetry_record(
    *,
    model_id: str,
    parent_key: str,
    arm: StageZArm,
    requested_open_duration: int,
    evidence_status: str = "NOT_EXECUTED",
) -> dict[str, Any]:
    if requested_open_duration not in (0, *DOSES):
        raise StageZHold("TELEMETRY_DOSE_NOT_FROZEN")
    if not model_id or not parent_key:
        raise StageZHold("TELEMETRY_AUTHORITY_BINDING_MISSING")
    return {
        "schema": TELEMETRY_SCHEMA,
        "evidence_status": evidence_status,
        "model_id": model_id,
        "parent_key": parent_key,
        "arm": arm.value,
        "requested_open_duration": requested_open_duration,
        "commanded_open_fraction": None,
        "executed_open_fraction": None,
        "gripper_qpos_or_width": None,
        "aperture_excess_vs_clean": None,
        "contact_loss": None,
        "contact_loss_step": None,
        "object_displacement": None,
        "v_phys": None,
        "official_task_success_secondary": None,
        "branch_valid": None,
        "manual_audit_id": None,
    }


def make_manual_audit_record(*, audit_id: str, blinded_video_id: str, label: str = "NOT_IDENTIFIABLE") -> dict[str, Any]:
    if not audit_id or not blinded_video_id:
        raise StageZHold("MANUAL_AUDIT_BINDING_MISSING")
    if label not in MANUAL_LABELS:
        raise StageZHold("UNKNOWN_MANUAL_AUDIT_LABEL")
    return {
        "schema": MANUAL_AUDIT_SCHEMA,
        "evidence_status": "NOT_EXECUTED",
        "audit_id": audit_id,
        "blinded_video_id": blinded_video_id,
        "stable_grasp_maintained": None,
        "premature_aperture": None,
        "slip_or_contact_loss": None,
        "premature_release_or_drop": None,
        "object_displacement_consistent_with_loss": None,
        "label": label,
    }


def validate_synthetic_row(row: Mapping[str, Any]) -> None:
    if row.get("evidence_status") != "TEST_ONLY_NON_SCIENTIFIC":
        raise StageZHold("SYNTHETIC_ROW_MUST_BE_EXPLICITLY_NON_SCIENTIFIC")
    if row.get("model_inference", 0) or row.get("env_step", 0) or row.get("protected_reads", 0):
        raise StageZHold("SYNTHETIC_ROW_PROTECTED_COUNTER_NONZERO")


__all__ = ["MANUAL_AUDIT_LABELS", "MANUAL_AUDIT_SCHEMA", "TELEMETRY_SCHEMA", "make_manual_audit_record", "make_telemetry_record", "validate_synthetic_row"]
