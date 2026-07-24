from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.multisuite_detector.bind_c6_legacy_runner_reset_v1 import PASS, build_report


FULL_RUNNER_ARGS = "--parent-id --episode-key --suite --task-id --condition --output-json --work-dir --dry-run --initial-state-hash"


def audit(tmp_path: Path, **overrides):
    parent = {
        "parent_id": "libero_goal/task_01/state_000",
        "episode_key": "libero_goal/task_01/state_000/clean/attempt_01",
        "suite": "libero_goal",
        "task_id": "1",
        "dataset_match_count": 1,
        "split_match_count": 1,
        "label_match_count": 1,
        "reset_candidate_fields": {"initial_state_hash": "abc123"},
    }
    parent.update(overrides)
    p = tmp_path / "audit.json"
    p.write_text(json.dumps({"parents": [parent], "legacy_runner": {"mentions_exact_prefix_or_restore": True, "mentions_json_output": True}}))
    return p


def runner(tmp_path: Path, args: str):
    p = tmp_path / "runner.py"
    p.write_text("import argparse\np=argparse.ArgumentParser()\n" + "\n".join(f"p.add_argument('{x}')" for x in args.split()))
    return p


def call(tmp_path: Path, audit_path: Path, runner_path: Path):
    return build_report(argparse.Namespace(input_audit_json=str(audit_path), legacy_runner=str(runner_path), output_root=str(tmp_path), git_commit="test", tests=[]))


def test_happy_path_constructs_static_invocation(tmp_path):
    r = call(tmp_path, audit(tmp_path), runner(tmp_path, FULL_RUNNER_ARGS))
    assert r["status"] == PASS
    assert "--initial-state-hash" in r["constructed_invocation"]["argv"]
    assert "abc123" in r["constructed_invocation"]["argv"]


def test_fail_count(tmp_path):
    r = call(tmp_path, audit(tmp_path, dataset_match_count=2), runner(tmp_path, FULL_RUNNER_ARGS))
    assert r["status"] == "HOLD_MATCH_COUNT_NOT_UNIQUE"
    assert r["constructed_invocation"]["argv"] == []


def test_missing_reset_field(tmp_path):
    r = call(tmp_path, audit(tmp_path, reset_candidate_fields={}), runner(tmp_path, FULL_RUNNER_ARGS))
    assert r["status"] == "HOLD_RESET_FIELD_MISSING"


def test_unsupported_runner_arg(tmp_path):
    r = call(tmp_path, audit(tmp_path), runner(tmp_path, "--parent-id --episode-key --suite --task-id --condition --output-json --work-dir --dry-run"))
    assert r["status"] == "HOLD_RESET_ARG_NOT_ACCEPTED_BY_RUNNER"


def test_bind_requires_parent_episode_args(tmp_path):
    r = call(tmp_path, audit(tmp_path), runner(tmp_path, "--initial-state-hash --suite --task-id --condition --output-json --work-dir --dry-run"))
    assert r["status"] == "HOLD_RUNNER_REQUIRED_ARGS_NOT_ACCEPTED"
    assert r["constructed_invocation"]["argv"] == []
