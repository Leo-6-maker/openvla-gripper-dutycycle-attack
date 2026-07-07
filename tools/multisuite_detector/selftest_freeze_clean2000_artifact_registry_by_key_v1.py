#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))

import freeze_clean2000_artifact_registry_by_key_v1 as freeze_tool


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_positive(tmp: Path) -> None:
    source = tmp / "source.csv"
    inventory = tmp / "inventory.csv"
    artifact = tmp / "sc5_cross_suite_clean1500_v1" / "libero_10" / "task_00" / "state_006"
    artifact.mkdir(parents=True)
    (artifact / "step_telemetry.csv").write_text("step,gripper_command\n0,0\n1,-1\n", encoding="utf-8")
    write_csv(source, [
        {"parent_id": "clean2000/libero_10/task_00/state_006", "suite": "libero_10", "task_id": "0", "state_id": "6", "clean_success": "true"},
        {"parent_id": "clean2000/libero_object/task_00/state_006", "suite": "libero_object", "task_id": "0", "state_id": "6", "clean_success": "true"},
    ], ["parent_id", "suite", "task_id", "state_id", "clean_success"])
    write_csv(inventory, [
        {"artifact_dir": str(artifact), "suite_hint": "libero_10", "present_temporal_sentinels": "step_telemetry.csv", "path_step_records.jsonl": "", "path_step_telemetry.csv": str(artifact / "step_telemetry.csv"), "path_phase_cues.csv": "", "path_episode_manifest.json": ""},
    ], ["artifact_dir", "suite_hint", "present_temporal_sentinels", "path_step_records.jsonl", "path_step_telemetry.csv", "path_phase_cues.csv", "path_episode_manifest.json"])
    out = tmp / "out_positive"
    rc = freeze_tool.run(Namespace(
        clean2000_records=str(source), artifact_inventory=str(inventory), target_suite="libero_10",
        expected_total=2, expected_target=1, allowed_suite_hint=["libero_10"], prefer_artifact_substring=[],
        require_files_exist=True, require_unique_artifact_dir=True, allow_partial_debug=False,
        max_ambiguity_rows_per_record=5, output_root=str(out), git_commit="SELFTEST",
        files_changed=[], tests=["selftest_positive"],
    ))
    assert rc == 0, rc
    report = json.loads((out / "clean2000_artifact_registry_by_key_report.json").read_text())
    assert report["status"] == freeze_tool.PASS, report
    assert report["accepted_count"] == 1, report


def run_ambiguous_negative(tmp: Path) -> None:
    source = tmp / "source2.csv"
    inventory = tmp / "inventory2.csv"
    a1 = tmp / "root_a" / "libero_10" / "task_00" / "state_006"
    a2 = tmp / "root_b" / "libero_10" / "task_00" / "state_006"
    for a in [a1, a2]:
        a.mkdir(parents=True)
        (a / "step_telemetry.csv").write_text("step,gripper_command\n0,0\n1,-1\n", encoding="utf-8")
    write_csv(source, [
        {"parent_id": "clean2000/libero_10/task_00/state_006", "suite": "libero_10", "task_id": "0", "state_id": "6"},
    ], ["parent_id", "suite", "task_id", "state_id"])
    write_csv(inventory, [
        {"artifact_dir": str(a1), "suite_hint": "libero_10", "present_temporal_sentinels": "step_telemetry.csv", "path_step_records.jsonl": "", "path_step_telemetry.csv": str(a1 / "step_telemetry.csv"), "path_phase_cues.csv": "", "path_episode_manifest.json": ""},
        {"artifact_dir": str(a2), "suite_hint": "libero_10", "present_temporal_sentinels": "step_telemetry.csv", "path_step_records.jsonl": "", "path_step_telemetry.csv": str(a2 / "step_telemetry.csv"), "path_phase_cues.csv": "", "path_episode_manifest.json": ""},
    ], ["artifact_dir", "suite_hint", "present_temporal_sentinels", "path_step_records.jsonl", "path_step_telemetry.csv", "path_phase_cues.csv", "path_episode_manifest.json"])
    out = tmp / "out_negative"
    rc = freeze_tool.run(Namespace(
        clean2000_records=str(source), artifact_inventory=str(inventory), target_suite="libero_10",
        expected_total=1, expected_target=1, allowed_suite_hint=["libero_10"], prefer_artifact_substring=[],
        require_files_exist=True, require_unique_artifact_dir=True, allow_partial_debug=False,
        max_ambiguity_rows_per_record=5, output_root=str(out), git_commit="SELFTEST",
        files_changed=[], tests=["selftest_ambiguous_negative"],
    ))
    assert rc == 2, rc
    report = json.loads((out / "clean2000_artifact_registry_by_key_report.json").read_text())
    assert report["status"] == "HOLD_STRUCTURED_KEY_REGISTRY_FREEZE_INCOMPLETE", report
    assert report["rejections_by_reason"].get("AMBIGUOUS_EXACT_TASK_STATE_CANDIDATES") == 1, report


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run_positive(tmp)
        run_ambiguous_negative(tmp)
    print("SELFTEST_STRUCTURED_KEY_REGISTRY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
