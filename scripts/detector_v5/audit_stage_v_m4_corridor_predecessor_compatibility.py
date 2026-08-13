#!/usr/bin/env python3
"""Audit that the immutable 32-pair predecessor and V1.1 use one science plane."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _option(command: list[Any], name: str) -> str:
    index = command.index(name)
    return str(command[index + 1])


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")


def audit(outer_path: Path, new_protocol_path: Path, new_audit_path: Path, new_auth_path: Path) -> dict[str, Any]:
    outer = _load(outer_path)
    predecessor = outer.get("predecessor", {})
    source = outer.get("source_binding", {})
    runtime = outer.get("runtime_binding", {})
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    check("outer_frozen_authorized", outer.get("status") == "FROZEN_RUNTIME_AUTHORIZED" and outer.get("runtime_authorized") is True)
    check("outer_boundary", outer.get("protected_counters") == COUNTERS and outer.get("operation", {}).get("outcomes_read") is False)
    terminal_path = Path(str(predecessor.get("terminal_report", {}).get("path", "")))
    inventory_path = Path(str(predecessor.get("pass_pass_inventory", {}).get("path", "")))
    old_protocol_path = Path(str(predecessor.get("historical_protocol", {}).get("path", "")))
    old_auth_path = Path(str(predecessor.get("historical_authorization", {}).get("path", "")))
    old_launch_path = Path(str(predecessor.get("historical_launch_manifest", {}).get("path", "")))
    for name, path, binding in (
        ("terminal_report", terminal_path, predecessor.get("terminal_report", {})),
        ("predecessor_inventory", inventory_path, predecessor.get("pass_pass_inventory", {})),
        ("old_protocol", old_protocol_path, predecessor.get("historical_protocol", {})),
        ("old_authorization", old_auth_path, predecessor.get("historical_authorization", {})),
        ("old_launch", old_launch_path, predecessor.get("historical_launch_manifest", {})),
    ):
        check(f"{name}_hash", path.is_file() and _sha(path) == binding.get("sha256"))

    terminal, inventory = _load(terminal_path), _load(inventory_path)
    old_protocol, old_auth, old_launch = _load(old_protocol_path), _load(old_auth_path), _load(old_launch_path)
    new_protocol, new_static, new_auth = _load(new_protocol_path), _load(new_audit_path), _load(new_auth_path)
    check("terminal_hold_immutable", terminal.get("status") == "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT" and terminal.get("sealed") is True and terminal.get("immutable") is True)
    check("predecessor_exact_32_pairs", inventory.get("status") == "PASS_EXACT_32_IMMUTABLE_CURRENT_SOURCE_PAIRS" and inventory.get("parent_count") == 32 and inventory.get("receipt_count") == 64 and len(inventory.get("parents", [])) == 32)
    check("predecessor_receipts_pass_pass", all(row.get("status_pair") == "PASS/PASS" for row in inventory.get("parents", [])))
    check("new_protocol_hash", _sha(new_protocol_path) == new_auth.get("protocol_sha256"))
    check("new_static_hash", _sha(new_audit_path) == new_auth.get("static_audit_sha256"))
    check("new_static_pass", new_static.get("status") == "PASS_STATIC_DESIGN_ONLY" and new_static.get("runner_sha256") == source.get("corridor_runner_sha256"))
    check("new_authorization_pass", new_auth.get("status") == "PASS" and new_auth.get("authorization_kind") == "RESERVE_CANDIDATE")
    check("new_authorization_boundary", new_auth.get("protected_counters") == COUNTERS and new_auth.get("outcomes_read") is False and new_auth.get("intervention_executed") is False)
    check("old_authorization_binding", old_auth.get("status") == "PASS" and old_auth.get("protocol_sha256") == _sha(old_protocol_path))
    check("old_authorization_boundary", old_auth.get("protected_counters") == COUNTERS and old_auth.get("outcomes_read") is False and old_auth.get("intervention_executed") is False)

    science_root = Path(str(source.get("science_worktree", ""))).resolve()
    science_commit, science_tree = str(source.get("science_commit", "")), str(source.get("science_tree", ""))
    check("science_commit", _git(science_root, "rev-parse", "HEAD") == science_commit)
    check("science_tree", _git(science_root, "rev-parse", "HEAD^{tree}") == science_tree)
    check("science_worktree_clean", _git(science_root, "status", "--porcelain") == "")
    runner = science_root / str(source.get("corridor_runner_path", ""))
    check("runner_hash", runner.is_file() and _sha(runner) == source.get("corridor_runner_sha256"))
    science_files = source.get("science_files", {})
    check("complete_imported_science_implementation", isinstance(science_files, Mapping) and len(science_files) == 9 and all((science_root / str(path)).is_file() and _sha(science_root / str(path)) == expected for path, expected in science_files.items()))
    check("old_new_source_equal", old_protocol.get("source_binding", {}).get("runtime_commit") == new_protocol.get("source_binding", {}).get("runtime_commit") == science_commit and old_protocol.get("source_binding", {}).get("runtime_tree") == new_protocol.get("source_binding", {}).get("runtime_tree") == science_tree)

    old_q, new_q = old_protocol.get("qualification", {}), new_protocol.get("qualification", {})
    semantic_fields = ("probe_count", "selection_version", "h_phys", "minimum_remaining_horizon", "minimum_corridor_candidates")
    check("qualification_semantics_equal", all(old_q.get(field) == new_q.get(field) for field in semantic_fields))
    check("replicate_semantics_equal", old_q.get("clean_replicates") == new_q.get("clean_replicates") == ["A", "B"])
    check("clean_only_equal", old_protocol.get("operation", {}).get("clean_only") is True and new_protocol.get("operation", {}).get("clean_only") is True)
    check("outcome_blind_equal", old_protocol.get("operation", {}).get("outcomes_read") is False and new_protocol.get("operation", {}).get("outcomes_read") is False)
    check("clean_success_pass_rule", new_q.get("clean_success_required") is True and new_q.get("pass_pair_rule") == "PASS/PASS")
    check("horizon_semantics", outer.get("qualification", {}).get("h_phys") == 10 and outer.get("qualification", {}).get("minimum_remaining_horizon") == 20 and outer.get("qualification", {}).get("suite_horizons") == {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220})

    tasks = old_launch.get("tasks", [])
    expected_commands = {
        "python": str(runtime.get("python_path", "")),
        "runner": str(runner),
        "snapshot": str(runtime.get("official_snapshot", {}).get("path", "")),
        "upstream": str(runtime.get("upstream", {}).get("path", "")),
        "model_root": str(runtime.get("model_root", "")),
    }
    commands_valid = len(tasks) == 80
    for task in tasks:
        command = task.get("command", [])
        try:
            commands_valid &= (
                command[0] == expected_commands["python"]
                and command[1] == expected_commands["runner"]
                and _option(command, "--official-snapshot-root") == expected_commands["snapshot"]
                and _option(command, "--upstream-root") == expected_commands["upstream"]
                and _option(command, "--model-root") == expected_commands["model_root"]
                and _option(command, "--source-commit") == science_commit
                and _option(command, "--source-tree") == science_tree
                and _option(command, "--replicate") in {"A", "B"}
            )
        except (IndexError, ValueError):
            commands_valid = False
    check("old_80_commands_bind_same_runtime", commands_valid and old_launch.get("status") == "COMPLETED_NO_M4_OUTCOMES")

    python_path = Path(str(runtime.get("python_path", "")))
    check("official_python_path", python_path.is_file() and str(python_path.resolve()) == runtime.get("python_resolved_path"))
    check("official_python_hash", python_path.is_file() and _sha(python_path) == runtime.get("python_executable_sha256"))
    check("official_python_version", platform.python_version() == runtime.get("python_version"))
    packages = {name: metadata.version(name) for name in runtime.get("required_packages", {})}
    check("runtime_packages", packages == runtime.get("required_packages"))
    observed["runtime_packages"] = packages

    for name in ("official_snapshot", "upstream"):
        binding = runtime.get(name, {})
        path = Path(str(binding.get("path", ""))).resolve()
        check(f"{name}_commit", _git(path, "rev-parse", "HEAD") == binding.get("git_commit"))
        check(f"{name}_tree", _git(path, "rev-parse", "HEAD^{tree}") == binding.get("git_tree"))
        check(f"{name}_clean", _git(path, "status", "--porcelain") == "")
    adapter = Path(str(runtime["official_snapshot"]["path"])) / str(runtime["official_snapshot"]["adapter_path"])
    check("official_adapter_hash", _sha(adapter) == runtime["official_snapshot"]["adapter_sha256"])

    sys.path.insert(0, str(science_root))
    from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import _directory_tree_binding  # noqa: E402
    model_observed: dict[str, Any] = {}
    for suite, binding in runtime.get("models", {}).items():
        actual = _directory_tree_binding(Path(str(binding["path"])))
        model_observed[str(suite)] = actual
        check(f"model_binding:{suite}", actual.get("algorithm") == runtime.get("model_tree_algorithm") and all(actual.get(field) == binding.get(field) for field in ("tree_sha256", "file_count", "total_bytes")))
    observed["models"] = model_observed

    check("old_new_protocols_distinct_and_bound", _sha(old_protocol_path) != _sha(new_protocol_path) and old_auth.get("protocol_sha256") == _sha(old_protocol_path) and new_auth.get("protocol_sha256") == _sha(new_protocol_path))
    check("protected_counters_zero_everywhere", terminal.get("independent_audit", {}).get("protected_counters") == inventory.get("protected_counters") == outer.get("protected_counters") == new_protocol.get("protected_counters") == COUNTERS)
    status = "PASS_PREDECESSOR_32_COMPATIBLE_WITH_POST_HOLD_V1_1" if all(checks.values()) else "HOLD_SOURCE_OR_PROTOCOL_COMPATIBILITY_NOT_ESTABLISHED"
    return {
        "schema": "STAGE_V_M4_CORRIDOR_PREDECESSOR_COMPATIBILITY_AUDIT_V1",
        "status": status,
        "runtime_authorized": False,
        "auditor_path": str(Path(__file__).resolve()),
        "auditor_sha256": _sha(Path(__file__).resolve()),
        "outer_protocol": str(outer_path),
        "outer_protocol_sha256": _sha(outer_path),
        "historical_protocol": str(old_protocol_path),
        "historical_protocol_sha256": _sha(old_protocol_path),
        "historical_authorization": str(old_auth_path),
        "historical_authorization_sha256": _sha(old_auth_path),
        "new_protocol": str(new_protocol_path),
        "new_protocol_sha256": _sha(new_protocol_path),
        "new_static_audit": str(new_audit_path),
        "new_static_audit_sha256": _sha(new_audit_path),
        "new_authorization": str(new_auth_path),
        "new_authorization_sha256": _sha(new_auth_path),
        "checks": checks,
        "check_count": len(checks),
        "failure_count": sum(not value for value in checks.values()),
        "observed": observed,
        "claim": "The 32 predecessor PASS/PASS pairs and new V1.1 receipts have equivalent per-parent clean-only corridor semantics under one immutable science implementation; their population protocols and authorizations remain distinct.",
        "outcomes_read": False,
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
        "next_action": "RUN_POST_HOLD_V1_1_CLEAN_ONLY" if status.startswith("PASS_") else "SEAL_HOLD_AND_REQUEST_OWNER_DIRECTION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--new-protocol", type=Path, required=True)
    parser.add_argument("--new-static-audit", type=Path, required=True)
    parser.add_argument("--new-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    result = audit(*(path.resolve() for path in (args.outer_protocol, args.new_protocol, args.new_static_audit, args.new_authorization)))
    _write(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "check_count": result["check_count"], "failure_count": result["failure_count"]}, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
