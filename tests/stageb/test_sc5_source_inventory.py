from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.inventory_all_sc5_sources_v2 import (  # noqa: E402
    normalize_task,
    scan_roots,
    summarize,
)


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _base_step(**extra):
    row = {
        "step_idx": 0,
        "gripper_command": 0.9,
        "gripper_qpos": 0.04,
        "gripper_width": 0.04,
        "eef_x": 0.1,
        "eef_y": 0.2,
        "eef_z": 0.3,
        "eef_vx": 0.01,
        "eef_vy": 0.02,
        "eef_vz": 0.03,
        "action_dx": 0.0,
        "action_dy": 0.1,
        "action_dz": 0.2,
        "action_gripper": 0.9,
        "success_check": 1,
        "attack_active": False,
        "attack_method": "none",
        "task_name": "butter",
        "suite": "libero_object",
    }
    row.update(extra)
    return row


def _config(root: Path):
    return {
        "roots": [{"path": str(root), "role": "test", "recursive": True}],
        "exclude_dir_name_substrings": [".git"],
        "expected_historical_counts": {
            "directories_scanned": 0,
            "step_records": 0,
            "manifests": 0,
            "known_clean_success": 0,
            "clean_fail": 0,
            "initially_unknown_task_names": 0,
        },
    }


def _aliases():
    return yaml.safe_load((REPO_ROOT / "configs" / "v2_sc5_schema_aliases.yaml").read_text(encoding="utf-8"))


def test_source_inventory_classifies_clean_object_candidate(tmp_path):
    run = tmp_path / "butter_s1_clean"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"task_name": "butter", "state_id": 1, "success": True, "run_id": "butter_clean"}),
        encoding="utf-8",
    )
    _write_jsonl(run / "step_records.jsonl", [_base_step()])

    roots, episodes = scan_roots(_config(tmp_path), _aliases())
    assert roots[0]["step_records_found"] == 1
    assert len(episodes) == 1
    row = episodes[0]
    assert row["clean_status"] == "CLEAN"
    assert row["success"] == "1"
    assert row["schema_status"] == "PASS"
    assert row["tier"] == "PRIMARY_SC5_POSITIVE_CANDIDATE"


def test_source_inventory_excludes_attack_or_intervention(tmp_path):
    run = tmp_path / "butter_s1_vis"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"task_name": "butter", "state_id": 1, "success": True, "run_id": "butter_vis"}),
        encoding="utf-8",
    )
    _write_jsonl(run / "step_records.jsonl", [_base_step(attack_active=True, attack_method="pgd")])

    _, episodes = scan_roots(_config(tmp_path), _aliases())
    assert episodes[0]["clean_status"] == "ATTACK_OR_INTERVENTION"
    assert episodes[0]["tier"] == "EXCLUDED_AUDIT_ONLY"


def test_source_inventory_fails_schema_when_required_field_missing(tmp_path):
    run = tmp_path / "butter_s1_clean"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"task_name": "butter", "state_id": 1, "success": True, "run_id": "butter_clean"}),
        encoding="utf-8",
    )
    row = _base_step()
    row.pop("action_dy")
    _write_jsonl(run / "step_records.jsonl", [row])

    _, episodes = scan_roots(_config(tmp_path), _aliases())
    assert episodes[0]["schema_status"] == "MISSING_FIELDS"
    assert "action_dy" in episodes[0]["schema_note"]
    assert episodes[0]["tier"] == "EXCLUDED_AUDIT_ONLY"


def test_source_inventory_summary_reports_current_scan_drift(tmp_path):
    run = tmp_path / "butter_s1_clean"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"task_name": "butter", "state_id": 1, "success": True, "run_id": "butter_clean"}),
        encoding="utf-8",
    )
    _write_jsonl(run / "step_records.jsonl", [_base_step()])
    roots, episodes = scan_roots(_config(tmp_path), _aliases())
    summary = summarize(_config(tmp_path), roots, episodes)
    assert summary["status"] == "SC5_SOURCE_CENSUS_FROZEN_WITH_CURRENT_SCAN_DRIFT"
    assert summary["counts"]["step_records"] == 1
    assert summary["tier_counts"]["PRIMARY_SC5_POSITIVE_CANDIDATE"] == 1
    assert summary["repo_head"]
    assert summary["repo_branch"]
    assert summary["repo_dirty"]
    assert summary["repo_dirty"] == "CLEAN" or summary["repo_dirty"].startswith("DIRTY:")
    assert summary["repo_provenance"] == "PASS"


def test_suite_inference_does_not_promote_goal_task_to_object_primary(tmp_path):
    run = tmp_path / "runs" / "libero_goal" / "put_the_cream_cheese_in_the_bowl_s0"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"task_name": "cream_cheese", "state_id": 0, "success": True, "run_id": "goal_clean"}),
        encoding="utf-8",
    )
    _write_jsonl(run / "step_records.jsonl", [_base_step(suite="libero_goal", task_name="cream_cheese")])

    _, episodes = scan_roots(_config(tmp_path), _aliases())
    assert episodes[0]["suite"] == "libero_goal"
    assert episodes[0]["tier"] == "CONDITIONAL_PLACE_CANDIDATE"


def test_normalize_task_uses_first_object_mention_deterministically():
    text = "living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
    assert normalize_task(text) == "cream_cheese"
