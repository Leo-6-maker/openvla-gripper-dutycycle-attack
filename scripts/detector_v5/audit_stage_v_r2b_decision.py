"""Independently audit the read-only R2B support decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now
    from scripts.monitoring.audit_stage_v_closure import verify_sha_manifest
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now
    from scripts.monitoring.audit_stage_v_closure import verify_sha_manifest


BOUNDARY_FIELDS = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")
VALID_STATUSES = {"R2B_REQUIRED", "R2B_NOT_REQUIRED"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    root: Path,
    *,
    r2a_root: Path,
    r2a_manifest: Path,
    candidate_manifest: Path,
    expected_source_commit: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    errors: list[str] = []
    decision_path = root / "STAGE_V_R2B_DECISION.json"
    try:
        decision = _read(decision_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        decision = {}
        errors.append("DECISION_MISSING_OR_INVALID")
    if not isinstance(decision, Mapping):
        decision = {}
        errors.append("DECISION_NOT_OBJECT")

    closure_path = r2a_root / "STAGE_V_CLOSURE_RECEIPT.json"
    r2a_audit_path = r2a_root / "STAGE_V_COUNTERFACTUAL_AUDIT.json"
    try:
        closure = _read(closure_path)
        r2a_audit = _read(r2a_audit_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        closure, r2a_audit = {}, {}
        errors.append("R2A_CLOSURE_INPUT_INVALID")
    if not isinstance(closure, Mapping) or closure.get("status") != "STAGE_V_FORMAL_MAP_CLOSED":
        errors.append("R2A_CLOSURE_NOT_PASS")
    if not isinstance(r2a_audit, Mapping) or r2a_audit.get("verdict") != "PASS":
        errors.append("R2A_AUDIT_NOT_PASS")
    if int(closure.get("accepted_parents", -1)) != 40 or int(closure.get("completed_branches", -1)) != 2880:
        errors.append("R2A_CLOSURE_COUNTS_INVALID")
    seal_ok, seal_errors, _ = verify_sha_manifest(r2a_root)
    if not seal_ok:
        errors.extend(f"R2A_SEAL:{item}" for item in seal_errors)

    if decision.get("schema") != "STAGE_V_R2B_PRE_REGISTERED_DECISION_V1":
        errors.append("DECISION_SCHEMA_INVALID")
    if decision.get("status") not in VALID_STATUSES:
        errors.append("DECISION_STATUS_INVALID")
    if decision.get("r2a_root") != str(r2a_root.resolve()):
        errors.append("R2A_ROOT_BINDING_MISMATCH")
    if decision.get("r2a_manifest_sha256") != sha256_file(r2a_manifest):
        errors.append("R2A_MANIFEST_BINDING_MISMATCH")
    if decision.get("candidate_manifest_sha256") != sha256_file(candidate_manifest):
        errors.append("CANDIDATE_MANIFEST_BINDING_MISMATCH")
    if decision.get("source_commit") not in (None, expected_source_commit):
        errors.append("SOURCE_COMMIT_MISMATCH")
    if decision.get("source_tree") not in (None, expected_source_tree):
        errors.append("SOURCE_TREE_MISMATCH")
    for field in BOUNDARY_FIELDS:
        if decision.get(field, 0) != 0:
            errors.append(f"BOUNDARY_NONZERO:{field}")
    if decision.get("errors"):
        errors.append("PRODUCER_REPORTED_ERRORS")

    status = str(decision.get("status", ""))
    selected = decision.get("selected_parents")
    if not isinstance(selected, list):
        selected = []
        errors.append("SELECTED_PARENTS_INVALID")
    keys = [str(row.get("canonical_parent_key", "")) for row in selected if isinstance(row, Mapping)]
    if len(keys) != len(selected) or not all(keys) or len(keys) != len(set(keys)):
        errors.append("SELECTED_PARENT_IDENTITIES_INVALID")
    if status == "R2B_NOT_REQUIRED":
        if int(decision.get("selected_count", -1)) != 0 or selected:
            errors.append("NOT_REQUIRED_SELECTION_NONEMPTY")
        if (root / "STAGE_V_R2B_PARENT_MANIFEST.json").exists():
            errors.append("NOT_REQUIRED_MANIFEST_PRESENT")
    elif status == "R2B_REQUIRED":
        if int(decision.get("selected_count", -1)) != 40 or len(selected) != 40:
            errors.append("REQUIRED_SELECTION_COUNT_INVALID")
        manifest_path = root / "STAGE_V_R2B_PARENT_MANIFEST.json"
        try:
            manifest = _read(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            manifest = {}
            errors.append("R2B_MANIFEST_MISSING_OR_INVALID")
        if not isinstance(manifest, Mapping) or manifest.get("status") != "FROZEN_PRELAUNCH":
            errors.append("R2B_MANIFEST_NOT_FROZEN")
        if isinstance(manifest, Mapping) and manifest.get("selected_count") != 40:
            errors.append("R2B_MANIFEST_COUNT_INVALID")
        if isinstance(manifest, Mapping) and manifest.get("selected_parents") != selected:
            errors.append("R2B_MANIFEST_SELECTION_MISMATCH")
        if manifest_path.is_file():
            sidecar = root / "STAGE_V_R2B_PARENT_MANIFEST.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != f"{sha256_file(manifest_path)}  {manifest_path.name}\n":
                errors.append("R2B_MANIFEST_SHA256_SIDECAR_INVALID")

    report = {
        "schema": "STAGE_V_R2B_DECISION_AUDIT_V1",
        "verdict": "PASS" if not errors else "FAIL",
        "root": str(root.resolve()),
        "r2a_root": str(r2a_root.resolve()),
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "decision_sha256": sha256_file(decision_path) if decision_path.is_file() else None,
        "errors": sorted(set(errors)),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "audited_utc": utc_now(),
    }
    atomic_write_json(root / "STAGE_V_R2B_DECISION_AUDIT.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r2a-root", type=Path, required=True)
    parser.add_argument("--r2a-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    args = parser.parse_args(argv)
    report = audit(
        args.root.resolve(),
        r2a_root=args.r2a_root.resolve(),
        r2a_manifest=args.r2a_manifest.resolve(),
        candidate_manifest=args.candidate_manifest.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
    )
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
