#!/usr/bin/env python3
"""Issue an append-only binding for the current formal-M4 outer gate."""
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

from scripts.detector_v5.run_stage_v_m4_formal_parent_with_resource_gate import (  # noqa: E402
    LAUNCH_GATE_FILES,
)
from scripts.detector_v5.stage_v_m4_governance import COUNTERS  # noqa: E402
from scripts.detector_v5.stage_v_gpu_resource_contract import MIN_FREE_MEMORY_MIB  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"REFUSE_OVERWRITE:{path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=20_480)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    protocol = args.protocol.resolve()
    authorization_path = args.authorization.resolve()
    output = args.output.resolve()
    auth = _load(authorization_path)
    if auth.get("schema") != "STAGE_V_M4_RUNTIME_AUTHORIZATION_V2" or auth.get("status") != "PASS" or auth.get("authorization_kind") != "FORMAL_M4_V2":
        raise SystemExit("M4_V2_AUTHORIZATION_NOT_PASS")
    if auth.get("runtime_authorized") is not True or auth.get("formal_m4_authorized") is not True or auth.get("intervention_executed") is not False or auth.get("outcomes_read") is not False or auth.get("v_phys_generated") is not False:
        raise SystemExit("M4_V2_AUTHORIZATION_BOUNDARY_INVALID")
    if auth.get("protected_counters") != COUNTERS:
        raise SystemExit("M4_V2_AUTHORIZATION_PROTECTED_BOUNDARY_INVALID")
    if auth.get("protocol_sha256") != _sha(protocol):
        raise SystemExit("M4_V2_PROTOCOL_SHA_MISMATCH")
    if not str(args.owner_basis).strip():
        raise SystemExit("EXPLICIT_OWNER_AUTHORIZATION_BASIS_REQUIRED")
    protocol_contract = _load(protocol).get("resource_contract", {})
    if (args.minimum_free_mib != MIN_FREE_MEMORY_MIB
            or int(protocol_contract.get("minimum_free_memory_mib", -1)) != MIN_FREE_MEMORY_MIB):
        raise SystemExit("RESOURCE_THRESHOLD_MUST_EQUAL_SUCCESSOR_CONTRACT")
    if _git(repo, "status", "--porcelain"):
        raise SystemExit("RUNTIME_SOURCE_WORKTREE_NOT_CLEAN")

    actual_head = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    authorized_head = str(auth.get("repository_head", ""))
    if not authorized_head or subprocess.call(["git", "merge-base", "--is-ancestor", authorized_head, actual_head], cwd=repo) != 0:
        raise SystemExit("CURRENT_RUNTIME_NOT_DESCENDANT_OF_V2_AUTHORIZATION")
    runtime_files: dict[str, str] = {}
    for relative in LAUNCH_GATE_FILES:
        path = repo / relative
        if not path.is_file():
            raise SystemExit(f"RUNTIME_FILE_MISSING:{relative}")
        runtime_files[relative] = _sha(path)

    receipt = {
        "schema": "STAGE_V_M4_FORMAL_LAUNCH_GATE_BINDING_V1",
        "status": "PASS",
        "binding_kind": "FORMAL_M4_OUTER_GATE_AND_RESOURCE_CONTRACT",
        "launch_gate_authorized": True,
        "authorization": str(authorization_path),
        "authorization_sha256": _sha(authorization_path),
        "protocol": str(protocol),
        "protocol_sha256": _sha(protocol),
        "authorization_repository_head": auth.get("repository_head"),
        "authorization_repository_tree": auth.get("repository_tree"),
        "source_commit": auth.get("source_commit"),
        "source_tree": auth.get("source_tree"),
        "repository_head": actual_head,
        "repository_tree": actual_tree,
        "runtime_file_sha256": runtime_files,
        "owner_authorization_basis": str(args.owner_basis),
        "minimum_free_memory_mib": args.minimum_free_mib,
        "resource_threshold_override": {
            "default_minimum_free_memory_mib": 20_480,
            "authorized_minimum_free_memory_mib": args.minimum_free_mib,
            "owner_basis": str(args.owner_basis),
        },
        "partial_fleet_allowed": True,
        "foreign_workload_allowed": True,
        "formal_m4_authorized": True,
        "runtime_authorized": True,
        "intervention_executed": False,
        "outcomes_read": False,
        "v_phys_generated": False,
        "protected_counters": dict(COUNTERS),
    }
    _write_new(output, receipt)
    sidecar = output.with_name(output.name + ".sha256")
    try:
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(f"{_sha(output)}  {output.name}\n")
    except FileExistsError as exc:
        raise SystemExit(f"REFUSE_OVERWRITE:{sidecar}") from exc
    print(json.dumps({"status": "PASS", "output": str(output), "repository_head": actual_head}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
