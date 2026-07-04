import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from tools.multisuite_detector.prepare_c4_codex_execution_plan_v1 import DEFAULTS, build_plan, write_bash


def ns(**overrides):
    values = {
        "dataset_root": DEFAULTS["dataset_root"],
        "feature_csv": DEFAULTS["feature_csv"],
        "c4_1_root": DEFAULTS["c4_1_root"],
        "c4_2_root": DEFAULTS["c4_2_root"],
        "c4_3_split_root": DEFAULTS["c4_3_split_root"],
        "c4_freeze_root": DEFAULTS["c4_freeze_root"],
        "dataset_csv_sha256": DEFAULTS["dataset_csv_sha256"],
        "split_csv_sha256": DEFAULTS["split_csv_sha256"],
        "state_index_sha256": DEFAULTS["state_index_sha256"],
        "checkpoint_sha256": DEFAULTS["checkpoint_sha256"],
        "threshold": DEFAULTS["threshold"],
        "seed": DEFAULTS["seed"],
        "val_ratio": DEFAULTS["val_ratio"],
    }
    values.update(overrides)
    return Namespace(**values)


def test_c4_codex_plan_contains_direct_commands():
    plan = build_plan(ns())
    assert plan["schema_version"] == "c4_codex_execution_plan_v1"
    ids = [step["id"] for step in plan["steps"]]
    assert ids == [
        "C4_2_BUNDLE_AUDIT_VALIDATE",
        "C4_3A_OBJECT_TASK_HELDOUT_SPLIT_BUILD",
        "C4_3B_SUITE_LOSO_SPLIT_BUILD",
        "C4_3C_DETECTOR_FREEZE_VALIDATE",
    ]
    commands = "\n".join(cmd for step in plan["steps"] for cmd in step["commands"])
    assert "validate_c4_bundle_audit_v1.py" in commands
    assert "build-object-task-heldout" in commands
    assert "build-suite-loso" in commands
    assert "validate_c4_detector_freeze_v1.py" in commands
    assert "<SCIENTIFIC_SPLIT_SHA256_FROM_FREEZE_BUNDLE>" in commands
    assert plan["global_non_actions"]["OpenVLA"] == "NOT_PERFORMED"


def test_c4_codex_plan_custom_paths_and_bash(tmp_path):
    plan = build_plan(ns(dataset_root="/tmp/dataset", c4_3_split_root="/tmp/splits"))
    assert plan["identities"]["dataset_csv"] == "/tmp/dataset/detector_dataset_manifest_v1.csv"
    assert any("/tmp/splits/object_task_heldout_with_val_v1.csv" in cmd for step in plan["steps"] for cmd in step["commands"])
    bash = tmp_path / "run.sh"
    write_bash(plan, bash)
    text = bash.read_text()
    assert "set -euo pipefail" in text
    assert "C4_3B_SUITE_LOSO_SPLIT_BUILD" in text


def test_c4_codex_plan_cli_writes_json_and_bash(tmp_path):
    out_json = tmp_path / "plan.json"
    out_bash = tmp_path / "plan.sh"
    subprocess.run([
        sys.executable,
        "tools/multisuite_detector/prepare_c4_codex_execution_plan_v1.py",
        "--output-json",
        str(out_json),
        "--output-bash",
        str(out_bash),
    ], check=True)
    obj = json.loads(out_json.read_text())
    assert obj["status"] == "PLAN_ONLY"
    assert out_bash.is_file()
    assert len(obj["steps"]) == 4
