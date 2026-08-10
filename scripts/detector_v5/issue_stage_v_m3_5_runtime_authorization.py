#!/usr/bin/env python3
"""Issue a launch receipt only after the protocol-bound static audit passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    audit_path = args.audit_report.resolve()
    protocol = _load(protocol_path)
    audit = _load(audit_path)
    if not str(args.owner_basis).strip() or protocol.get("requires_explicit_owner_authorization") is not True:
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    actual_status = _git(repo, "status", "--porcelain")
    audit_binding = protocol.get("static_audit_binding", {})
    if audit.get("schema") != audit_binding.get("receipt_schema") or audit.get("status") != "PASS":
        raise SystemExit("STATIC_AUDIT_NOT_PASS")
    if audit.get("protocol_sha256") != _sha256(protocol_path):
        raise SystemExit("AUDIT_PROTOCOL_SHA_MISMATCH")
    if actual_commit != str(args.source_commit) or actual_tree != str(args.source_tree) or actual_status:
        raise SystemExit("SOURCE_COMMIT_OR_TREE_MISMATCH")
    if audit.get("actual_source_commit") != actual_commit or audit.get("actual_source_tree") != actual_tree or audit.get("actual_source_status") not in ("", None):
        raise SystemExit("STATIC_AUDIT_SOURCE_BINDING_MISMATCH")
    if protocol.get("runtime_authorized") is not True or protocol.get("protected_eval160", {}).get("reads_allowed") is not False:
        raise SystemExit("PROTOCOL_NOT_RUNTIME_AUTHORIZED_OR_EVAL160_NOT_PROTECTED")
    selection_binding = protocol.get("diagnostic_parent_selection", {})
    selection_path = Path(str(selection_binding.get("path", ""))).resolve()
    if not selection_path.is_file() or _sha256(selection_path) != str(selection_binding.get("sha256", "")):
        raise SystemExit("DIAGNOSTIC_SELECTION_BINDING_MISMATCH")
    receipt_schema = str(audit_binding.get("authorization_receipt_schema", ""))
    if not receipt_schema:
        raise SystemExit("AUTHORIZATION_RECEIPT_SCHEMA_MISSING")
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    receipt = {
        "schema": receipt_schema,
        "status": "PASS",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "static_audit_report": str(audit_path),
        "static_audit_sha256": _sha256(audit_path),
        "source_commit": actual_commit,
        "source_tree": actual_tree,
        "source_status": actual_status,
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "runtime_python": protocol.get("source_binding", {}).get("runtime_python"),
        "owner_authorization_basis": str(args.owner_basis),
        "protected_counters": dict(COUNTERS),
        "eval160_reads": 0,
        "attack_rollouts": 0,
        "outcome_informed_revision": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = _sha256(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "protocol_sha256": receipt["protocol_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
