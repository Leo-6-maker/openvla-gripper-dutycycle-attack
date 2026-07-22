#!/usr/bin/env python3
"""Build V3.2 only after a real, sealed production-input chain passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

try:
    from scripts.detector_v5.audit_factorized_v2_production_inputs import verify_sealed_root
except ModuleNotFoundError:  # pragma: no cover - direct CLI execution
    from audit_factorized_v2_production_inputs import verify_sealed_root

EXPECTED_SPLITS = [f"o{outer}_i{inner}" for outer in range(4) for inner in range(3)]


class HandoffBuildError(ValueError):
    pass


def _strict(path: Path) -> Any:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HandoffBuildError(f"DUPLICATE_JSON_KEY:{key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)


def _canonical(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("handoff_blob_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def build(audit_path: Path, identity_audit_path: Path, bundle_root: Path, *, branch: str, base_commit: str, code_snapshot_commit: str, output: Path, receipt: Path) -> dict[str, Any]:
    audit = _strict(audit_path)
    identity = _strict(identity_audit_path)
    if audit.get("production_chain_ready") is not True:
        raise HandoffBuildError("PRODUCTION_INPUT_AUDIT_NOT_PASS")
    if identity.get("verdict") != "GROUP_CROSS_FITTED_OOF_FEASIBLE":
        raise HandoffBuildError("IDENTITY_AUDIT_NOT_FEASIBLE")
    bundle_seal = verify_sealed_root(bundle_root)
    if not bundle_seal["pass"]:
        raise HandoffBuildError("BUNDLE_ROOT_NOT_SEALED")
    manifest = _strict(bundle_root / "manifest.json")
    if manifest.get("split_keys") != EXPECTED_SPLITS or manifest.get("formal_selection_eligible") is not False:
        raise HandoffBuildError("BUNDLE_SPLIT_OR_AUTHORIZATION_INVALID")
    value = {
        "schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_2",
        "interface_revision": "V3.2",
        "status": "READY_FOR_DEEPSEEK_STATIC_INTEGRATION",
        "branch": branch,
        "base_commit": base_commit,
        "code_snapshot_commit": code_snapshot_commit,
        "metadata_commit_consumption_rule": "DeepSeek checks out code_snapshot_commit for code and reads this handoff plus its receipt from the later metadata commit.",
        "expected_split_keys": EXPECTED_SPLITS,
        "production_input_audit": {"path": str(audit_path), "sha256": _sha(audit_path), "verdict": audit["production_chain_ready"]},
        "identity_audit": {"path": str(identity_audit_path), "sha256": _sha(identity_audit_path), "verdict": identity["verdict"]},
        "production_bundle": {"path": str(bundle_root), "sha256s_sha256": _sha(bundle_root / "SHA256SUMS"), "manifest_sha256": _sha(bundle_root / "manifest.json")},
        "execution_boundary": {"model_inference": False, "training": False, "full_fit": False, "cal_check": False, "rollout": False, "shadow": False, "attack": False},
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    }
    value["handoff_blob_sha256"] = _canonical(value)
    receipt_value = {
        "schema": "FACTORIZED_V3_2_HANDOFF_RECEIPT_V1",
        "status": "STATIC_INTEGRATION_PASS",
        "branch": branch,
        "base_commit": base_commit,
        "code_snapshot_commit": code_snapshot_commit,
        "handoff_path": str(output),
        "handoff_blob_sha256": value["handoff_blob_sha256"],
        "production_execution": False,
        "model_inference": False,
        "training": False,
        "attack": False,
    }
    _write(output, value)
    _write(receipt, receipt_value)
    return {"status": "PASS", "handoff_blob_sha256": value["handoff_blob_sha256"], "receipt_sha256": _sha(receipt)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--code-snapshot-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.audit, args.identity_audit, args.bundle_root, branch=args.branch, base_commit=args.base_commit, code_snapshot_commit=args.code_snapshot_commit, output=args.output, receipt=args.receipt), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
