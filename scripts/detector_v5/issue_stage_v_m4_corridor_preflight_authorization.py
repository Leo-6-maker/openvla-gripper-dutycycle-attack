#!/usr/bin/env python3
"""Authorize only the clean-only M4 corridor preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
            rows.append(value)
    return rows


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--v7-receipt", type=Path, required=True)
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
    split_path = args.split.resolve()
    protocol = _load(protocol_path)
    audit = _load(audit_path)
    v7 = _load(v7_path)
    split = _load(split_path)
    if protocol.get("schema") != "STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise SystemExit("M4_CORRIDOR_PROTOCOL_NOT_AUTHORIZED")
    if not str(args.owner_basis).strip() or protocol.get("requires_explicit_owner_authorization") is not True:
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    if _git(repo, "status", "--porcelain"):
        raise SystemExit("SOURCE_WORKTREE_DIRTY")
    if _git(repo, "rev-parse", f"{args.source_commit}^{{tree}}") != args.source_tree:
        raise SystemExit("SOURCE_TREE_MISMATCH")
    if subprocess.call(["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"], cwd=repo) != 0:
        raise SystemExit("SOURCE_NOT_ANCESTOR")
    if audit.get("schema") != "STAGE_V_M4_CORRIDOR_STATIC_AUDIT_V1" or audit.get("status") != "PASS_STATIC_DESIGN_ONLY" or audit.get("protocol_sha256") != _sha(protocol_path):
        raise SystemExit("M4_CORRIDOR_STATIC_AUDIT_NOT_BOUND")
    inputs = protocol.get("inputs", {})
    reserve_manifest = bool(inputs.get("candidate_parent_manifest_path"))
    manifest_key = "candidate_parent_manifest_path" if reserve_manifest else "formal_parent_split_path"
    manifest_sha_key = "candidate_parent_manifest_sha256" if reserve_manifest else "formal_parent_split_sha256"
    candidate_count = int(protocol.get("qualification", {}).get("candidate_parent_count", 0) or 0)
    if str(inputs.get(manifest_key)) != str(split_path) or inputs.get(manifest_sha_key) != _sha(split_path):
        raise SystemExit("M4_CORRIDOR_SPLIT_BINDING_MISMATCH")
    allowed_schemas = {"STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1", "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1"}
    if split.get("schema") not in allowed_schemas or split.get("status") != "FROZEN" or len(split.get("parents", [])) != candidate_count:
        raise SystemExit("M4_CORRIDOR_SPLIT_INVALID")
    if not reserve_manifest and candidate_count != 40:
        raise SystemExit("M4_CORRIDOR_FORMAL_SPLIT_INVALID")
    if reserve_manifest:
        rows_path = Path(str(inputs.get("v7_control_qualification_rows_path", ""))).resolve()
        if not rows_path.is_file() or inputs.get("v7_control_qualification_rows_sha256") != _sha(rows_path):
            raise SystemExit("M4_CORRIDOR_V7_ROWS_BINDING_INVALID")
        by_key = {str(row.get("canonical_parent_key")): row for row in _load_jsonl(rows_path)}
        for parent in split["parents"]:
            key = str(parent.get("canonical_parent_key", ""))
            row = by_key.get(key)
            if row is None or row.get("qualified") is not True or row.get("errors") != []:
                raise SystemExit(f"M4_CORRIDOR_RESERVE_NOT_V7_QUALIFIED:{key}")
            replicates = row.get("replicates", {})
            if any(not isinstance(replicates.get(rep), dict) or replicates[rep].get("status") != "PASS" for rep in ("A", "B")):
                raise SystemExit(f"M4_CORRIDOR_RESERVE_REPLICATES_INVALID:{key}")
            if parent.get("v7_candidate_sha256") != row.get("candidate_sha256"):
                raise SystemExit(f"M4_CORRIDOR_RESERVE_CANDIDATE_BINDING_INVALID:{key}")
    if v7.get("schema") != "STAGE_V_V7_FORMAL_QUALIFICATION_PASS_RECEIPT_V1" or v7.get("status") != "PASS_FORMAL_PARENT_QUALIFICATION" or v7.get("formal_parent_count") != 40:
        raise SystemExit("V7_RECEIPT_NOT_PASS")
    if inputs.get("v7_formal_receipt_sha256") != _sha(v7_path) or protocol.get("protected_counters") != COUNTERS:
        raise SystemExit("M4_CORRIDOR_INPUT_OR_BOUNDARY_INVALID")
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    receipt = {
        "schema": "STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION_V1",
        "status": "PASS",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "static_audit": str(audit_path),
        "static_audit_sha256": _sha(audit_path),
        "v7_formal_receipt": str(v7_path),
        "v7_formal_receipt_sha256": _sha(v7_path),
        "authorization_kind": "RESERVE_CANDIDATE" if reserve_manifest else "FORMAL",
        "candidate_parent_manifest": str(split_path),
        "candidate_parent_manifest_sha256": _sha(split_path),
        "candidate_parent_count": candidate_count,
        "formal_parent_split": None if reserve_manifest else str(split_path),
        "formal_parent_split_sha256": None if reserve_manifest else _sha(split_path),
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "repository_head": _git(repo, "rev-parse", "HEAD"),
        "repository_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "formal_parent_count": None if reserve_manifest else 40,
        "clean_replicates": ["A", "B"],
        "minimum_corridor_candidates": 24,
        "owner_authorization_basis": str(args.owner_basis),
        "foreign_workload_allowed": True,
        "minimum_free_memory_mib": 20480,
        "protected_counters": dict(COUNTERS),
        "outcomes_read": False,
        "intervention_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(f"{_sha(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "protocol_sha256": receipt["protocol_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
