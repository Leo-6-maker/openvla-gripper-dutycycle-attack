#!/usr/bin/env python3
"""Create the static C6_1E legacy-runner reset binding artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

GATE = "C6_1E_LEGACY_RUNNER_RESET_BINDING_PATCH"
PASS = "PASS_STATIC_SHIM_ARG_BINDING"
RESET_ARGS = ["--initial-state-hash", "--initial_state_hash", "--initial-state", "--initial_state", "--reset-state-hash", "--reset_state_hash"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runner_args(path: str | Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"add_argument\(\s*[\"'](--[^\"']+)", text))


def first_parent(audit: dict[str, Any]) -> dict[str, Any]:
    parents = audit.get("parents")
    if not isinstance(parents, list) or not parents or not isinstance(parents[0], dict):
        raise ValueError("audit missing parents[0]")
    return parents[0]


def status_for(parent: dict[str, Any], accepted: set[str]) -> str:
    counts = [parent.get("dataset_match_count"), parent.get("split_match_count"), parent.get("label_match_count")]
    if counts != [1, 1, 1]:
        return "HOLD_MATCH_COUNT_NOT_UNIQUE"
    reset = parent.get("reset_candidate_fields")
    if not isinstance(reset, dict) or not reset.get("initial_state_hash"):
        return "HOLD_RESET_FIELD_MISSING"
    if not any(arg in accepted for arg in RESET_ARGS):
        return "HOLD_RESET_ARG_NOT_ACCEPTED_BY_RUNNER"
    return PASS


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(args.input_audit_json)
    parent = first_parent(audit)
    accepted = runner_args(args.legacy_runner)
    status = status_for(parent, accepted)
    reset_value = ""
    if isinstance(parent.get("reset_candidate_fields"), dict):
        reset_value = str(parent["reset_candidate_fields"].get("initial_state_hash", ""))
    reset_arg = next((arg for arg in RESET_ARGS if arg in accepted), None)
    argv = []
    if status == PASS:
        argv = [
            "python3", str(args.legacy_runner),
            "--parent-id", str(parent["parent_id"]),
            "--episode-key", str(parent["episode_key"]),
            "--suite", str(parent["suite"]),
            "--task-id", str(parent["task_id"]),
            "--condition", "CLEAN",
            "--output-json", "{legacy_result_json}",
            "--work-dir", "{work_dir}",
            str(reset_arg), reset_value,
            "--dry-run",
        ]
    return {
        "gate": GATE,
        "status": status,
        "input_audit_json": str(args.input_audit_json),
        "input_audit_json_sha256": sha256_file(args.input_audit_json),
        "selected_parent": {
            "parent_id": parent.get("parent_id", ""),
            "episode_key": parent.get("episode_key", ""),
            "suite": parent.get("suite", ""),
            "task_id": parent.get("task_id", ""),
        },
        "match_counts": {
            "dataset_match_count": parent.get("dataset_match_count"),
            "split_match_count": parent.get("split_match_count"),
            "label_match_count": parent.get("label_match_count"),
        },
        "reset_binding": {
            "field": "initial_state_hash",
            "value": reset_value,
            "source": "C6_1D_PARENT_RESET_BINDING_AUDIT",
        },
        "legacy_runner": {
            "path": str(args.legacy_runner),
            "accepts_parent_or_state_args": "--parent-id" in accepted and "--episode-key" in accepted,
            "accepts_initial_state_hash_or_equivalent": reset_arg is not None,
            "accepted_reset_arg": reset_arg,
            "mentions_exact_prefix_or_restore": bool(audit.get("legacy_runner", {}).get("mentions_exact_prefix_or_restore")),
            "mentions_json_output": bool(audit.get("legacy_runner", {}).get("mentions_json_output")),
        },
        "constructed_invocation": {"mode": "STATIC_DRY_RUN_ONLY", "argv": argv, "kwargs": {}},
        "boundaries": {
            "OpenVLA": "NOT_PERFORMED",
            "LIBERO": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
            "artifact_mutation": "NOT_PERFORMED",
        },
        "files_changed": ["scripts/c6_run_one_condition_openvla_libero.py", "tools/multisuite_detector/bind_c6_legacy_runner_reset_v1.py"],
        "git_commit": args.git_commit,
        "tests": args.tests,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-audit-json", required=True)
    p.add_argument("--legacy-runner", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--tests", action="append", default=[])
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    write_json(out / "legacy_runner_reset_binding_patch.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
