#!/usr/bin/env python3
"""Issue the current M4 authority receipt without executing the experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.detector_v5.stage_v_m4_governance import (  # noqa: E402
    M4GovernanceError,
    validate_formal_m4_v2_authority,
)


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
RUNTIME_FILES = (
    "scripts/detector_v5/run_stage_v_m4_matched_parent.py",
    "scripts/detector_v5/audit_stage_v_m4_matched_parent.py",
    "scripts/detector_v5/stage_v_m4_governance.py",
    "scripts/detector_v5/audit_stage_v_m4_static.py",
    "scripts/detector_v5/issue_stage_v_m4_runtime_authorization_v2.py",
    "scripts/detector_v5/run_stage_v_m4_formal_scheduler.py",
    "scripts/detector_v5/run_stage_v_m4_formal_parent_with_resource_gate.py",
    "scripts/detector_v5/stage_v_gpu_resource_contract.py",
    "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py",
    "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py",
    "src/gripper_attack/stage_v_causal_observation_snapshot.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _source_file_unchanged(repo: Path, commit: str, relative: str) -> bool:
    try:
        expected = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
    except subprocess.CalledProcessError:
        return False
    return expected == (repo / relative).read_bytes()


def _authority_bindings(protocol: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    inputs = protocol.get("inputs", {})
    names = (
        "v1_supersession_receipt_sha256",
        "formal_parent_manifest_sha256",
        "formal_parent_split_sha256",
        "exact_plan_manifest_sha256",
        "exact_plan_audit_sha256",
        "exact_plan_result_sha256",
        "primary_firewall_report_sha256",
        "teacher_student_freeze_report_sha256",
        "pre_m4_lock_report_sha256",
        "student_checkpoint_sha256",
        "student_thresholds_sha256",
        "feature_schema_sha256",
        "architecture_addendum_sha256",
        "feature_order_sha256",
    )
    result = {name: inputs[name] for name in names}
    result.update(
        {
            "manifest_sha256": authority["manifest_sha256"],
            "split_sha256": authority["split_sha256"],
            "formal_parent_manifest_sha256": authority["manifest_sha256"],
            "formal_parent_split_sha256": authority["split_sha256"],
            "exact_plan_manifest_sha256": authority["exact_plan_manifest_sha256"],
            "primary_firewall_report_sha256": authority["primary_firewall_report_sha256"],
            "teacher_student_freeze_report_sha256": authority["teacher_student_freeze_report_sha256"],
            "pre_m4_lock_report_sha256": authority["pre_m4_lock_report_sha256"],
            "student_checkpoint_sha256": authority["student_checkpoint_sha256"],
            "student_thresholds_sha256": authority["student_thresholds_sha256"],
            "feature_schema_sha256": authority["feature_schema_sha256"],
        }
    )
    if protocol.get("successor_protocol") is True:
        inputs = protocol["inputs"]
        result.update(
            {
                "successor_protocol": True,
                "snapshot_rebind_receipt_sha256": inputs["snapshot_rebind_receipt_sha256"],
                "compatibility_audit_root_seal_sha256": inputs["compatibility_audit_root_seal_sha256"],
                "compatibility_q00_result_sha256": inputs["compatibility_q00_result_sha256"],
                "compatibility_q00_audit_sha256": inputs["compatibility_q00_audit_sha256"],
                "compatibility_fleet_preflight_sha256": inputs["compatibility_fleet_preflight_sha256"],
                "compatibility_fleet_authority_sha256": inputs["compatibility_fleet_authority_sha256"],
                "compatibility_fleet_result_sha256": inputs["compatibility_fleet_result_sha256"],
                "compatibility_runtime_provenance_sha256": inputs["compatibility_runtime_provenance_sha256"],
                "successor_runtime_provenance_sha256": inputs["successor_runtime_provenance_sha256"],
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    audit_path = args.static_audit.resolve()
    output_path = args.output.resolve()
    protocol = _load(protocol_path)
    audit = _load(audit_path)
    if protocol.get("schema") != "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2" or protocol.get("status") != "FROZEN_PROSPECTIVE_NOT_AUTHORIZED" or protocol.get("runtime_authorized") is not False:
        raise SystemExit("M4_V2_PROTOCOL_NOT_PROSPECTIVE")
    if not str(args.owner_basis).strip() or protocol.get("requires_explicit_owner_authorization") is not True:
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if _git(repo, "status", "--porcelain"):
        raise SystemExit("RUNTIME_SOURCE_WORKTREE_NOT_CLEAN")
    if _git(repo, "rev-parse", f"{args.source_commit}^{{tree}}") != args.source_tree:
        raise SystemExit("RUNTIME_SOURCE_TREE_MISMATCH")
    if subprocess.call(["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"], cwd=repo) != 0:
        raise SystemExit("RUNTIME_SOURCE_NOT_ANCESTOR")
    if not all(_source_file_unchanged(repo, args.source_commit, relative) for relative in RUNTIME_FILES):
        raise SystemExit("RUNTIME_SOURCE_FILE_DRIFT")
    source = protocol.get("source_binding", {})
    if source.get("runtime_commit") != args.source_commit or source.get("runtime_tree") != args.source_tree:
        raise SystemExit("PROTOCOL_SOURCE_BINDING_MISMATCH")
    expected_source_hashes = source.get("runtime_file_sha256", {})
    if not isinstance(expected_source_hashes, Mapping) or any(expected_source_hashes.get(relative) != _sha(repo / relative) for relative in RUNTIME_FILES):
        raise SystemExit("PROTOCOL_RUNTIME_FILE_HASH_MISMATCH")
    if audit.get("schema") != "STAGE_V_M4_STATIC_AUDIT_V2" or audit.get("status") != "PASS_STATIC_DESIGN_ONLY" or audit.get("protocol_sha256") != _sha(protocol_path) or audit.get("source_commit") != args.source_commit or audit.get("source_tree") != args.source_tree:
        raise SystemExit("M4_V2_STATIC_AUDIT_NOT_PASS_OR_BOUND")
    try:
        authority = validate_formal_m4_v2_authority(
            protocol,
            protocol_path=protocol_path,
            split_path=Path(str(protocol["inputs"]["formal_parent_split_path"])).resolve(),
            source_commit=args.source_commit,
            source_tree=args.source_tree,
        )
    except M4GovernanceError as exc:
        raise SystemExit(str(exc)) from exc
    if output_path.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{output_path}")
    receipt = {
        "schema": "STAGE_V_M4_RUNTIME_AUTHORIZATION_V2",
        "status": "PASS",
        "authorization_kind": "FORMAL_M4_V2",
        "runtime_authorized": True,
        "formal_m4_authorized": True,
        "owner_authorized": True,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "static_audit": str(audit_path),
        "static_audit_sha256": _sha(audit_path),
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "repository_head": actual_commit,
        "repository_tree": actual_tree,
        "runtime_file_sha256": {relative: _sha(repo / relative) for relative in RUNTIME_FILES},
        "runtime_provenance_sha256": protocol["inputs"]["successor_runtime_provenance_sha256"],
        "authority_bindings": _authority_bindings(protocol, authority),
        "formal_parent_count": 40,
        "matrix": protocol.get("matrix"),
        "owner_authorization_basis": str(args.owner_basis),
        "foreign_workload_allowed": True,
        "minimum_free_memory_mib": 20480,
        "intervention_executed": False,
        "outcomes_read": False,
        "v_phys_generated": False,
        "protected_counters": dict(COUNTERS),
        "eval160_reads": 0,
        "attack_rollouts": 0,
        "outcome_informed_revision": False,
        "successor_protocol": protocol.get("successor_protocol") is True,
        "compatibility_only_evidence_bound": protocol.get("successor_protocol") is True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path.with_name(output_path.name + ".sha256").write_text(f"{_sha(output_path)}  {output_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_path), "protocol_sha256": receipt["protocol_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
