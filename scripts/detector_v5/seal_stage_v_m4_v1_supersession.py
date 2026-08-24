"""Seal the append-only supersession of the historical M4 V1 authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_r3_contact_input import sha256_file, verify_seal


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
REQUIRED_ORDER = [
    "CLEAN_ROLLOUT",
    "PRIVILEGED_CLEAN_TEACHER_C_t",
    "CLEAN_TEACHER_SUPERVISED_CAUSAL_STUDENT_C_HAT_t",
    "HELD_OUT_MATCHED_COUNTERFACTUAL_VALIDATION_V_t_d",
    "TIMING_VIS_DEFENSE_LATER",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _file(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"missing regular file: {value}")
    return value


def _root(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_dir() or value.is_symlink():
        raise ValueError(f"invalid evidence root: {value}")
    return value


def _sealed(root: Path, name: str) -> tuple[dict[str, Any], str]:
    root = _root(root)
    seal = verify_seal(root)["sha256sums_sha256"]
    path = _file(root / name)
    return _json(path), seal


def _seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def seal(*, old_protocol: Path, old_authorization: Path, architecture_addendum: Path, freeze_root: Path, pre_m4_lock_root: Path, output_root: Path) -> dict[str, Any]:
    old_protocol = _file(old_protocol)
    old_authorization = _file(old_authorization)
    architecture_addendum = _file(architecture_addendum)
    protocol = _json(old_protocol)
    authorization = _json(old_authorization)
    addendum = _json(architecture_addendum)
    freeze, freeze_seal = _sealed(freeze_root, "PRIMARY_TEACHER_STUDENT_FREEZE.json")
    lock, lock_seal = _sealed(pre_m4_lock_root, "PRE_M4_LOCK.json")

    if protocol.get("schema") != "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise ValueError("historical V1 protocol does not match the preserved artifact")
    if authorization.get("schema") != "STAGE_V_M4_RUNTIME_AUTHORIZATION_V1" or authorization.get("status") != "PASS":
        raise ValueError("historical V1 authorization does not match the preserved artifact")
    if authorization.get("protocol_sha256") != sha256_file(old_protocol):
        raise ValueError("historical protocol/authorization hash mismatch")
    if addendum.get("schema") != "STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM" or addendum.get("status") != "ACTIVE_STATUS_ADDENDUM" or addendum.get("architecture_semantics_changed") is not False:
        raise ValueError("current architecture addendum is not active and append-only")
    if addendum.get("architecture_freeze", {}).get("required_order") != REQUIRED_ORDER or addendum.get("mainline_order_lock", [])[0:2] != ["V2_TERMINAL_HOLD", "POST_HOLD_CORRIDOR_REPLENISHMENT"]:
        raise ValueError("current architecture order is not the frozen order")
    if freeze.get("status") != "PASS_PRIMARY_TEACHER_STUDENT_FREEZE" or freeze.get("formal_m4_authorized") is not False or freeze.get("m4_outcomes_read") is not False or freeze.get("protected_counters") != COUNTERS:
        raise ValueError("Teacher/Student freeze is not closed")
    if lock.get("status") != "PASS_PRE_M4_LOCK" or lock.get("formal_m4_authorized") is not False or lock.get("m4_outcomes_read") is not False or lock.get("protected_counters") != COUNTERS:
        raise ValueError("pre-M4 lock is not closed")

    report = {
        "schema": "STAGE_V_M4_PROTOCOL_V1_SUPERSESSION_RECEIPT_V1",
        "status": "HISTORICAL_NONCONSUMABLE_FOR_CURRENT_MAINLINE",
        "append_only": True,
        "purpose": "Preserve the historical V1 authority while preventing its use for the current Teacher-Student-held-out-M4 mainline.",
        "historical_protocol": {
            "path": str(old_protocol),
            "sha256": sha256_file(old_protocol),
            "schema": protocol["schema"],
            "status": protocol["status"],
            "runtime_authorized": protocol["runtime_authorized"],
            "source_commit": protocol.get("source_binding", {}).get("runtime_commit"),
            "source_tree": protocol.get("source_binding", {}).get("runtime_tree"),
        },
        "historical_authorization": {
            "path": str(old_authorization),
            "sha256": sha256_file(old_authorization),
            "schema": authorization["schema"],
            "status": authorization["status"],
            "protocol_sha256": authorization["protocol_sha256"],
            "source_commit": authorization.get("source_commit"),
            "source_tree": authorization.get("source_tree"),
        },
        "superseded_by": {
            "architecture_addendum_path": str(architecture_addendum),
            "architecture_addendum_sha256": sha256_file(architecture_addendum),
            "freeze_root": str(_root(freeze_root)),
            "freeze_seal_sha256sums_sha256": freeze_seal,
            "freeze_report_sha256": sha256_file(_root(freeze_root) / "PRIMARY_TEACHER_STUDENT_FREEZE.json"),
            "pre_m4_lock_root": str(_root(pre_m4_lock_root)),
            "pre_m4_lock_seal_sha256sums_sha256": lock_seal,
            "pre_m4_lock_report_sha256": sha256_file(_root(pre_m4_lock_root) / "PRE_M4_LOCK.json"),
            "required_current_bindings": [
                "FINAL40_AND_SPLIT_SEALED",
                "EXACT_40X24_PLAN_AND_SNAPSHOT_MANIFEST_AUDITED",
                "PRIMARY_DATA_FIREWALL_SEALED",
                "PRIMARY_TEACHER_STUDENT_FREEZE_SHA_BOUND",
                "PRE_M4_LOCK_SHA_BOUND",
                "STUDENT_FEATURE_SCHEMA_SHA_BOUND",
                "STUDENT_THRESHOLD_SHA_BOUND",
            ],
        },
        "old_artifacts_modified": False,
        "formal_m4_authorized": False,
        "m4_outcomes_read": False,
        "v_phys_generated": False,
        "protected_counters": dict(COUNTERS),
        "failure_action": "HOLD_UNTIL_CURRENT_V2_RUNTIME_AUTHORIZATION_PASS",
    }
    output_root = output_root.resolve()
    if output_root.parent != _root(pre_m4_lock_root).parent or output_root.exists():
        raise ValueError("supersession output must be a new sibling of pre-M4 lock")
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "M4_V1_SUPERSESSION_RECEIPT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _seal(staging)
    staging.rename(output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("old-protocol", "old-authorization", "architecture-addendum", "freeze-root", "pre-m4-lock-root", "output-root"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(old_protocol=args.old_protocol, old_authorization=args.old_authorization, architecture_addendum=args.architecture_addendum, freeze_root=args.freeze_root, pre_m4_lock_root=args.pre_m4_lock_root, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
