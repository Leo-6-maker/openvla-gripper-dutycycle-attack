#!/usr/bin/env python3
"""Issue V1.4 authorization only after protocol, source, regression and audit bind."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    audit_path = args.audit_report.resolve()
    regression_path = args.regression_receipt.resolve()
    protocol = _load(protocol_path)
    audit = _load(audit_path)
    regression = _load(regression_path)
    if protocol.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise SystemExit("V1_4_PROTOCOL_NOT_FROZEN_OR_AUTHORIZED")
    if not str(args.owner_basis).strip() or protocol.get("requires_explicit_owner_authorization") is not True:
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    actual_status = _git(repo, "status", "--porcelain")
    if actual_commit != args.source_commit or actual_tree != args.source_tree or actual_status:
        raise SystemExit("SOURCE_COMMIT_OR_TREE_MISMATCH")
    binding = protocol.get("source_binding", {})
    if binding.get("runtime_commit") != actual_commit or binding.get("runtime_tree") != actual_tree:
        raise SystemExit("PROTOCOL_SOURCE_BINDING_MISMATCH")
    static_binding = protocol.get("static_audit_binding", {})
    if audit.get("schema") != static_binding.get("receipt_schema") or audit.get("status") != "PASS_STATIC_DESIGN_ONLY":
        raise SystemExit("V1_4_STATIC_AUDIT_NOT_PASS")
    if audit.get("protocol_sha256") != _sha(protocol_path) or audit.get("source_commit") != actual_commit or audit.get("source_tree") != actual_tree:
        raise SystemExit("V1_4_STATIC_AUDIT_BINDING_MISMATCH")
    regression_binding = protocol.get("exact_a800_regression", {})
    if regression.get("schema") != "STAGE_V_M3_5_EXACT_A800_REGRESSION_RECEIPT_V1" or regression.get("status") != "PASS":
        raise SystemExit("V1_4_EXACT_REGRESSION_NOT_PASS")
    if regression.get("source_commit") != actual_commit or regression.get("source_tree") != actual_tree or regression.get("source_status_porcelain") != "" or regression.get("cuda_visible_devices") != "":
        raise SystemExit("V1_4_EXACT_REGRESSION_SOURCE_BINDING_MISMATCH")
    if regression.get("collected") != regression_binding.get("expected_collected") or regression.get("passed") + regression.get("skipped") != regression.get("collected") or regression.get("failed") != 0 or regression.get("errors") != 0 or regression.get("protected_counters") != COUNTERS:
        raise SystemExit("V1_4_EXACT_REGRESSION_COUNTS_INVALID")
    if str(regression_binding.get("path")) != str(regression_path) or regression_binding.get("sha256") != _sha(regression_path):
        raise SystemExit("V1_4_EXACT_REGRESSION_RECEIPT_BINDING_MISMATCH")
    selection = protocol.get("diagnostic_parent_selection", {})
    selection_path = Path(str(selection.get("path", ""))).resolve()
    if not selection_path.is_file() or _sha(selection_path) != selection.get("sha256") or selection.get("outcomes_read") is not False:
        raise SystemExit("V1_4_SELECTION_BINDING_INVALID")
    if protocol.get("protected_counters") != COUNTERS or protocol.get("protected_eval160", {}).get("reads_allowed") is not False:
        raise SystemExit("V1_4_PROTECTED_BOUNDARY_INVALID")
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    receipt = {
        "schema": "STAGE_V_M3_5_V1_4_RUNTIME_AUTHORIZATION_V1",
        "status": "PASS",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "static_audit_report": str(audit_path),
        "static_audit_sha256": _sha(audit_path),
        "exact_regression_receipt": str(regression_path),
        "exact_regression_sha256": _sha(regression_path),
        "source_commit": actual_commit,
        "source_tree": actual_tree,
        "source_status": actual_status,
        "selection": str(selection_path),
        "selection_sha256": _sha(selection_path),
        "runtime_python": binding.get("runtime_python"),
        "owner_authorization_basis": str(args.owner_basis),
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
