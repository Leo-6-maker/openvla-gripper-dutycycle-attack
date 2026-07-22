#!/usr/bin/env python3
"""Build a V3.2 handoff as a strict superset of the V3.1 interface.

This builder is a metadata-only gate.  It never runs inference, training,
calibration, threshold selection, rollout, or attack.  It refuses to emit a
V3.2 handoff unless the production-input audit, identity audit, and sealed
bundle are already present and passing.
"""

from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SPLITS = [f"o{outer}_i{inner}" for outer in range(4) for inner in range(3)]
TEXT_SUFFIXES = {".json", ".csv", ".py", ".yml", ".yaml", ".md", ".schema", ".sha256"}


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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _repo_relative(path: Path, *, allow_directory: bool = False) -> str:
    path = path.resolve()
    try:
        relative = path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HandoffBuildError(f"REFERENCE_OUTSIDE_REPO:{path}") from exc
    if allow_directory:
        if not path.is_dir():
            raise HandoffBuildError(f"REFERENCE_DIRECTORY_MISSING:{path}")
    elif not path.is_file():
        raise HandoffBuildError(f"REFERENCE_FILE_MISSING:{path}")
    return relative.as_posix()


def _file_ref(path: Path) -> dict[str, str]:
    return {"path": _repo_relative(path), "sha256": _sha(path)}


def _receipt_binding_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("handoff_blob_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def _refresh_v31_file_refs(value: Any) -> None:
    """Refresh inherited repo-file refs to the code snapshot being built.

    Receipt refs use the documented canonical receipt binding rather than the
    raw receipt bytes and are therefore intentionally left unchanged here.
    """
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            path = ROOT / value["path"]
            if path.is_file() and not path.name.endswith("HANDOFF_RECEIPT.json"):
                value["sha256"] = _sha(path)
        for child in value.values():
            _refresh_v31_file_refs(child)
    elif isinstance(value, list):
        for child in value:
            _refresh_v31_file_refs(child)


def build(
    audit_path: Path,
    identity_audit_path: Path,
    bundle_root: Path,
    *,
    v31_handoff: Path,
    branch: str,
    base_commit: str,
    code_snapshot_commit: str,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    base = _strict(v31_handoff)
    if base.get("schema") != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1":
        raise HandoffBuildError("V3_1_BASE_HANDOFF_REQUIRED")
    if base.get("interface_revision") != "V3.1":
        raise HandoffBuildError("V3_1_BASE_REVISION_REQUIRED")
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

    value = copy.deepcopy(base)
    _refresh_v31_file_refs(value)
    value.update({
        "schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_2",
        "interface_revision": "V3.2",
        "status": "READY_FOR_DEEPSEEK_STATIC_INTEGRATION",
        "branch": branch,
        "base_commit": base_commit,
        "code_snapshot_commit": code_snapshot_commit,
        "handoff_metadata_parent_commit": code_snapshot_commit,
        "metadata_commit_consumption_rule": "DeepSeek checks out code_snapshot_commit for implementation and reads this V3.2 handoff plus its receipt from the later metadata commit containing them.",
        "expected_split_keys": EXPECTED_SPLITS,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    })
    value["production_input_audit"] = {
        "summary": _file_ref(audit_path),
        "verdict": audit["production_chain_ready"],
    }
    value["identity_audit"] = {
        "summary": _file_ref(identity_audit_path),
        "verdict": identity["verdict"],
    }
    value["production_bundle"] = {
        "root_path": _repo_relative(bundle_root, allow_directory=True),
        "manifest": _file_ref(bundle_root / "manifest.json"),
        "seal": _file_ref(bundle_root / "SHA256SUMS"),
        "split_keys": EXPECTED_SPLITS,
    }
    value.setdefault("execution_boundary", {}).update({
        "static_interface": True,
        "sealed_artifact_audit": True,
        "runtime_rematerialization": False,
        "offline_evaluation_bundle": False,
        "model_inference": False,
        "training": False,
        "full_fit": False,
        "cal_check": False,
        "rollout": False,
        "shadow": False,
        "attack": False,
    })

    receipt_path = _repo_relative(receipt)
    value.setdefault("production_receipt_requirements", {})["handoff_receipt"] = {
        "path": receipt_path,
        "sha256": "0" * 64,
    }
    receipt_value = {
        "schema": "FACTORIZED_V3_2_HANDOFF_RECEIPT_V1",
        "status": "STATIC_INTEGRATION_PASS",
        "branch": branch,
        "base_commit": base_commit,
        "code_snapshot_commit": code_snapshot_commit,
        "handoff_metadata_parent_commit": code_snapshot_commit,
        "handoff_path": _repo_relative(output),
        "handoff_blob_sha256": "0" * 64,
        "handoff_binding_definition": "SHA256(canonical handoff JSON with handoff_blob_sha256 omitted)",
        "production_execution": False,
        "model_inference": False,
        "training": False,
        "full_fit": False,
        "cal_check": False,
        "rollout": False,
        "shadow": False,
        "attack": False,
    }
    value["production_receipt_requirements"]["handoff_receipt"]["sha256"] = _receipt_binding_sha(receipt_value)
    value["handoff_blob_sha256"] = _canonical(value)
    receipt_value["handoff_blob_sha256"] = value["handoff_blob_sha256"]

    _write(output, value)
    _write(receipt, receipt_value)
    return {
        "status": "PASS",
        "handoff_blob_sha256": value["handoff_blob_sha256"],
        "receipt_sha256": _sha(receipt),
        "receipt_binding_sha256": value["production_receipt_requirements"]["handoff_receipt"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-handoff", type=Path, required=True)
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
        print(json.dumps(build(
            args.audit.resolve(), args.identity_audit.resolve(), args.bundle_root.resolve(),
            v31_handoff=args.v31_handoff.resolve(), branch=args.branch,
            base_commit=args.base_commit, code_snapshot_commit=args.code_snapshot_commit,
            output=args.output.resolve(), receipt=args.receipt.resolve(),
        ), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
