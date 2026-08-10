#!/usr/bin/env python3
"""Independent static audit for the prospective M3.5 V1.2 runtime contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "STAGE_V_M3_5_STATIC_INDEPENDENT_AUDIT_V1_2"


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


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def _git(repo: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def audit(repo_root: Path, protocol_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    protocol_path = protocol_path.resolve()
    protocol = _load(protocol_path)
    checks: list[dict[str, Any]] = []
    _check(checks, "protocol_schema", protocol.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_2", str(protocol.get("schema")))
    _check(checks, "protocol_status", protocol.get("status") == "FROZEN_PROSPECTIVE_RUNTIME_READY_PENDING_INDEPENDENT_AUDIT", str(protocol.get("status")))
    _check(checks, "runtime_authorization", protocol.get("runtime_authorized") is True and protocol.get("launch_policy", {}).get("runtime_authorized") is True, "V1.2 runtime authorization is explicit")
    _check(checks, "eval160_hard_stop", protocol.get("protected_eval160") == {"reads_allowed": False, "rollouts_allowed": False, "hard_stop": True}, str(protocol.get("protected_eval160")))
    _check(checks, "exact_python", protocol.get("source_binding", {}).get("runtime_python") == "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python", str(protocol.get("source_binding", {}).get("runtime_python")))

    label_binding = protocol.get("contract_bindings", {}).get("label_contract", {})
    label_path = repo / str(label_binding.get("path", ""))
    label = _load(label_path) if label_path.is_file() else {}
    _check(checks, "label_exists", label_path.is_file(), str(label_path))
    _check(checks, "label_sha_bound", label_path.is_file() and _sha256(label_path) == str(label_binding.get("sha256", "")), str(label_binding.get("sha256", "")))
    _check(checks, "label_schema", label.get("schema") == "STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_2", str(label.get("schema")))
    _check(checks, "label_truth_table", label.get("truth_table", {}).get("primary_estimand") == "V_phys" and "V_PHYS" in label.get("truth_table", {}).get("classes", []), "V_phys truth table bound")

    for binding_name, expected_schema in (("phase_classifier", "STAGE_V_M3_5_PHASE_CLASSIFIER_V1"), ("probe_plan_builder", "STAGE_V_M3_5_PROBE_PLAN_V1"), ("physical_taxonomy", "STAGE_V_M3_5_PHYSICAL_TAXONOMY_V1")):
        binding = protocol.get("contract_bindings", {}).get(binding_name, {})
        path = repo / str(binding.get("path", ""))
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        _check(checks, f"{binding_name}_exists", path.is_file(), str(path))
        _check(checks, f"{binding_name}_sha", path.is_file() and _sha256(path) == str(binding.get("sha256", "")), str(binding.get("sha256", "")))
        _check(checks, f"{binding_name}_schema_token", expected_schema in text, expected_schema)
        if binding_name == "phase_classifier":
            tokens = ("def classify_phase", "def classify_trajectory", "UNKNOWN", "outcome_blind")
        elif binding_name == "probe_plan_builder":
            tokens = ("def select_probe_steps", "PROBE_PLAN_INSUFFICIENT", "outcomes_read", "backfill_allowed")
        else:
            tokens = ("def bind_object_taxonomy", "def telemetry_from_env", "def evaluate_treatment_compliance", "def v_phys_label", "object_gripper_contact", "object_support_contact")
        _check(checks, f"{binding_name}_executable_tokens", all(token in text for token in tokens), ", ".join(tokens))

    runner = protocol.get("contract_bindings", {}).get("runner", {})
    runner_path = repo / str(runner.get("path", ""))
    runner_text = runner_path.read_text(encoding="utf-8") if runner_path.is_file() else ""
    _check(checks, "runner_exists", runner_path.is_file(), str(runner_path))
    _check(checks, "runner_sha", runner_path.is_file() and _sha256(runner_path) == str(runner.get("sha256", "")), str(runner.get("sha256", "")))
    runner_tokens = ("def run_parent", "def _run_branch", "build_forced_open_action", "repeatability_receipt", "parent_atomic", "actual_physical_branches", "protected_counters")
    _check(checks, "runner_parent_atomic_contract", all(token in runner_text for token in runner_tokens), ", ".join(runner_tokens))
    forbidden_runtime_reads = ("open_eval160", "read_eval160", "eval160_path", "protected_eval160_root")
    _check(checks, "runner_no_protected_eval160_reader", not any(token in runner_text.lower() for token in forbidden_runtime_reads), "no protected Eval160 reader token")

    selection_binding = protocol.get("freshness_bindings", {}).get("diagnostic_selection", {})
    selection_path = Path(str(selection_binding.get("path", "")))
    selection = _load(selection_path) if selection_path.is_file() else {}
    _check(checks, "selection_exists", selection_path.is_file(), str(selection_path))
    _check(checks, "selection_sha", selection_path.is_file() and _sha256(selection_path) == str(selection_binding.get("sha256", "")), str(selection_binding.get("sha256", "")))
    _check(checks, "selection_count", selection.get("selected_count") == 8 and selection.get("selected_counts_by_suite") == {"libero_10": 2, "libero_goal": 2, "libero_object": 2, "libero_spatial": 2}, str(selection.get("selected_counts_by_suite")))
    _check(checks, "selection_outcome_blind", selection.get("selection_reads", {}).get("outcomes_read") is False and selection.get("selection_reads", {}).get("branch_results_read") is False, str(selection.get("selection_reads")))

    audit_binding = protocol.get("static_audit_binding", {})
    for key in ("auditor_script_path", "authorization_issuer_script_path"):
        path = repo / str(audit_binding.get(key, ""))
        hash_key = key.replace("_path", "_sha256")
        _check(checks, f"{key}_exists", path.is_file(), str(path))
        _check(checks, f"{key}_sha", path.is_file() and _sha256(path) == str(audit_binding.get(hash_key, "")), str(audit_binding.get(hash_key, "")))

    for path in (repo / str(protocol.get("contract_bindings", {}).get("runner", {}).get("path", "")), repo / str(protocol.get("contract_bindings", {}).get("physical_taxonomy", {}).get("path", ""))):
        result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=repo, capture_output=True, text=True)
        _check(checks, f"py_compile:{path.name}", result.returncode == 0, result.stderr.strip() or "compiled")
    rc, diff = _git(repo, "diff", "--check")
    _check(checks, "git_diff_check", rc == 0, diff or "clean")

    failures = [check for check in checks if check["status"] != "PASS"]
    return {
        "schema": SCHEMA,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "failure_count": len(failures),
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.repo_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = _sha256(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "check_count": report["check_count"], "failure_count": report["failure_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
