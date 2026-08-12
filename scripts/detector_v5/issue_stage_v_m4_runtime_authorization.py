#!/usr/bin/env python3
"""Issue M4 runtime authorization only after the frozen upstream gates bind."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.detector_v5.stage_v_m4_governance import (
    M4GovernanceError,
    validate_formal_m4_corridor_gate,
)


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _source_file_unchanged(repo: Path, commit: str, relative: str) -> bool:
    try:
        expected = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
    except subprocess.CalledProcessError:
        return False
    return expected == (repo / relative).read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--v7-receipt", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    audit_path = args.static_audit.resolve()
    v7_path = args.v7_receipt.resolve()
    split_audit_path = args.split_audit.resolve()
    split_path = args.split.resolve()
    protocol = _load(protocol_path)
    audit = _load(audit_path)
    v7 = _load(v7_path)
    split_audit = _load(split_audit_path)
    split = _load(split_path)
    if protocol.get("schema") != "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise SystemExit("M4_PROTOCOL_NOT_FROZEN_OR_AUTHORIZED")
    if not str(args.owner_basis).strip() or protocol.get("requires_explicit_owner_authorization") is not True:
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    actual_status = _git(repo, "status", "--porcelain")
    source_tree = _git(repo, "rev-parse", f"{args.source_commit}^{{tree}}")
    if source_tree != args.source_tree or actual_status:
        raise SystemExit("SOURCE_COMMIT_OR_TREE_MISMATCH")
    if subprocess.call(["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"], cwd=repo) != 0:
        raise SystemExit("RUNTIME_SOURCE_NOT_ANCESTOR")
    runtime_files = (
        "scripts/detector_v5/run_stage_v_m4_matched_parent.py",
        "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py",
        "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py",
        "src/gripper_attack/stage_v_causal_observation_snapshot.py",
    )
    if not all(_source_file_unchanged(repo, args.source_commit, relative) for relative in runtime_files):
        raise SystemExit("RUNTIME_SOURCE_FILE_DRIFT")
    source = protocol.get("source_binding", {})
    if source.get("runtime_commit") != args.source_commit or source.get("runtime_tree") != args.source_tree:
        raise SystemExit("PROTOCOL_SOURCE_BINDING_MISMATCH")
    if audit.get("schema") != "STAGE_V_M4_STATIC_AUDIT_V1" or audit.get("status") != "PASS_STATIC_DESIGN_ONLY":
        raise SystemExit("M4_STATIC_AUDIT_NOT_PASS")
    if audit.get("protocol_sha256") != _sha(protocol_path) or audit.get("source_commit") != args.source_commit or audit.get("source_tree") != args.source_tree:
        raise SystemExit("M4_STATIC_AUDIT_BINDING_MISMATCH")
    inputs = protocol.get("inputs", {})
    if str(inputs.get("formal_parent_split_path")) != str(split_path) or inputs.get("formal_parent_split_sha256") != _sha(split_path):
        raise SystemExit("M4_SPLIT_BINDING_MISMATCH")
    if split.get("schema") != "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1" or split.get("status") != "FROZEN" or len(split.get("parents", [])) != 40:
        raise SystemExit("M4_SPLIT_NOT_FROZEN_OR_COUNT_INVALID")
    if split_audit.get("schema") != "STAGE_V_FORMAL_PARENT_SPLIT_INDEPENDENT_AUDIT_V1" or split_audit.get("verdict") != "PASS":
        raise SystemExit("M4_SPLIT_AUDIT_NOT_PASS")
    if inputs.get("formal_parent_split_audit_sha256") != _sha(split_audit_path):
        raise SystemExit("M4_SPLIT_AUDIT_HASH_MISMATCH")
    if v7.get("schema") != "STAGE_V_V7_FORMAL_QUALIFICATION_PASS_RECEIPT_V1" or v7.get("status") != "PASS_FORMAL_PARENT_QUALIFICATION" or v7.get("M4_authorized") is not True or v7.get("formal_parent_count") != 40:
        raise SystemExit("V7_FORMAL_QUALIFICATION_NOT_AUTHORIZED")
    if inputs.get("v7_formal_receipt_sha256") != _sha(v7_path):
        raise SystemExit("V7_FORMAL_RECEIPT_HASH_MISMATCH")
    if protocol.get("protected_counters") != COUNTERS:
        raise SystemExit("M4_PROTECTED_BOUNDARY_INVALID")
    try:
        corridor_gate = validate_formal_m4_corridor_gate(
            protocol,
            protocol_path=protocol_path,
            split_path=split_path,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
        )
    except M4GovernanceError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    receipt = {
        "schema": "STAGE_V_M4_RUNTIME_AUTHORIZATION_V1",
        "status": "PASS",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "static_audit": str(audit_path),
        "static_audit_sha256": _sha(audit_path),
        "v7_formal_receipt": str(v7_path),
        "v7_formal_receipt_sha256": _sha(v7_path),
        "formal_parent_split": str(split_path),
        "formal_parent_split_sha256": _sha(split_path),
        "formal_parent_split_audit": str(split_audit_path),
        "formal_parent_split_audit_sha256": _sha(split_audit_path),
        "supersession_hold": corridor_gate["hold_path"],
        "supersession_hold_sha256": corridor_gate["hold_sha256"],
        "corridor_pass_receipt": corridor_gate["receipt_path"],
        "corridor_pass_receipt_sha256": corridor_gate["receipt_sha256"],
        "corridor_qualification_protocol_sha256": inputs["corridor_qualification_protocol_sha256"],
        "corridor_qualification_authorization_sha256": inputs["corridor_qualification_authorization_sha256"],
        "corridor_reconciliation_sha256": inputs["corridor_reconciliation_sha256"],
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "repository_head": actual_commit,
        "repository_tree": actual_tree,
        "source_status": actual_status,
        "formal_parent_count": 40,
        "matrix": protocol.get("matrix"),
        "owner_authorization_basis": str(args.owner_basis),
        "foreign_workload_allowed": True,
        "minimum_free_memory_mib": 20480,
        "protected_counters": dict(COUNTERS),
        "eval160_reads": 0,
        "attack_rollouts": 0,
        "outcome_informed_revision": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = _sha(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "protocol_sha256": receipt["protocol_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
