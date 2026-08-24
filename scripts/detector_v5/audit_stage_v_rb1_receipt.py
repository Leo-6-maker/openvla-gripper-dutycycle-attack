#!/usr/bin/env python3
"""Independently recompute an RB1 receipt without running the environment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_rb1_runtime_equivalence import (
        RuntimeEquivalenceError,
        canonical_sha256,
        validate_protocol,
        validate_receipt,
        verify_artifact_files,
    )
    from gripper_attack.stage_v_canonical_execution_core import sha256_file
except ImportError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from stage_v_rb1_runtime_equivalence import RuntimeEquivalenceError, canonical_sha256, validate_protocol, validate_receipt, verify_artifact_files
    from gripper_attack.stage_v_canonical_execution_core import sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeEquivalenceError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(*, protocol_path: Path, receipt_path: Path, artifact_root: Path, core_path: Path, output_path: Path, repo: Path) -> dict[str, Any]:
    protocol = validate_protocol(_load(protocol_path))
    receipt = _load(receipt_path)
    validate_receipt(receipt, protocol, require_independent_recompute=False)
    verify_artifact_files(receipt, artifact_root, protocol, require_independent_recompute=False)

    core_sha = sha256_file(core_path)
    if receipt.get("clean_core_sha256") != core_sha:
        raise RuntimeEquivalenceError("RB1_INDEPENDENT_CORE_SHA256_MISMATCH")
    contract = receipt["execution_contract"]
    if contract.get("clean_core_sha256") != core_sha:
        raise RuntimeEquivalenceError("RB1_INDEPENDENT_CONTRACT_CORE_SHA256_MISMATCH")
    if receipt.get("execution_contract_sha256") != canonical_sha256(contract):
        raise RuntimeEquivalenceError("RB1_INDEPENDENT_CONTRACT_DIGEST_MISMATCH")

    auditor_repo = repo.resolve()
    if _git(auditor_repo, "status", "--porcelain"):
        raise RuntimeEquivalenceError("RB1_AUDITOR_WORKTREE_DIRTY")
    audited = dict(receipt)
    audited["independent_recompute"] = {
        "status": "PASS",
        "recomputed": True,
        "auditor_source_commit": _git(auditor_repo, "rev-parse", "HEAD"),
        "auditor_source_tree": _git(auditor_repo, "rev-parse", "HEAD^{tree}"),
        "auditor_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol_path),
        "core_sha256_recomputed": core_sha,
        "execution_contract_sha256_recomputed": canonical_sha256(contract),
    }
    _write_json(output_path, audited)
    validate_receipt(_load(output_path), protocol)
    return {
        "schema": "STAGE_V_RB1_INDEPENDENT_RECEIPT_AUDIT_V1",
        "verdict": "PASS",
        "receipt": str(output_path.resolve()),
        "core_sha256": core_sha,
        "protocol_sha256": sha256_file(protocol_path),
        "artifact_verification": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.receipt.resolve():
            raise RuntimeEquivalenceError("RB1_AUDIT_OUTPUT_MUST_NOT_OVERWRITE_PRODUCER_RECEIPT")
        result = audit(
            protocol_path=args.protocol.resolve(), receipt_path=args.receipt.resolve(),
            artifact_root=args.artifact_root.resolve(), core_path=args.core.resolve(),
            output_path=args.output.resolve(), repo=args.repo.resolve(),
        )
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeEquivalenceError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
