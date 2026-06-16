from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.watch_m3_gpu15_stage2 import (
    STAGE2_PHASES,
    STAGE2_SEEDS,
    Stage2PlanError,
    build_stage2_plan,
    evaluate_s5_gate,
    flatten_command_rows,
    load_selected_lambda,
    run_prepare,
    s6_enabled_from_s5_gate,
    select_multi_parent_rows,
)
from gripper_attack.m3_telemetry_schema import TelemetrySchemaError, read_required_int


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _gate_row(**overrides):
    row = {
        "lambda": 0.25,
        "passed": True,
        "output_dir": "/out/canary",
        "result_class": "FULL_SELECTIVE_V4_SEED_PASS",
        "true_margin": 10.0,
        "rand_margin": 1.0,
        "shuffled_margin": 0.5,
        "true_arm": 6,
        "true_token": 31744,
    }
    row.update(overrides)
    return row


def _write_gate_pair(tmp_path: Path, selected: dict | None = None, *, status: str = "PASS") -> tuple[Path, Path]:
    selected = selected or _gate_row()
    gate = tmp_path / "gate_result.json"
    csv_path = tmp_path / "m3_v3_tomato_results.csv"
    _write_json(gate, {"status": status, "selected": selected})
    _write_csv(csv_path, [selected])
    return gate, csv_path


def _config(tmp_path: Path, gate: Path, results: Path, handoff: Path) -> Path:
    cfg = {
        "python": "/env/bin/python",
        "stage2_runner": "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py",
        "stage2_config": "configs/m3_step78_true_pgd_31744_logratio_arm_v4.yaml",
        "stage2_mode": "canary_v4",
        "stage2_output_root": "/stage2/out",
        "prepare_output_dir": str(tmp_path / "prepare"),
        "s3_gate_result_path": str(gate),
        "s3_results_csv": str(results),
        "handoff_csv": str(handoff),
        "max_parents": 3,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _handoff(tmp_path: Path) -> Path:
    path = tmp_path / "handoff.csv"
    _write_csv(
        path,
        [
            {"task": "butter", "state_id": 2, "selected_step": 77, "status": "PASS"},
            {"task": "cream_cheese", "state_id": 1, "selected_step": 91, "status": "PASS"},
            {"task": "ketchup", "state_id": 0, "selected_step": 83, "status": "PASS"},
            {"task": "tomato_sauce", "state_id": 0, "selected_step": 78, "status": "PASS"},
        ],
    )
    return path


def test_s3_fail_gate_rejects_stage2(tmp_path):
    gate, results = _write_gate_pair(tmp_path, status="FAIL")
    with pytest.raises(Stage2PlanError, match="not PASS"):
        load_selected_lambda(gate, results)


def test_selected_lambda_must_match_results_csv(tmp_path):
    selected = _gate_row(true_margin=10.0)
    gate, results = _write_gate_pair(tmp_path, selected)
    rows = [dict(selected)]
    rows[0]["true_margin"] = 9.0
    _write_csv(results, rows)
    with pytest.raises(Stage2PlanError, match="mismatch"):
        load_selected_lambda(gate, results)


def test_selected_lambda_requires_token_arm_and_control_margins(tmp_path):
    for override in ({"true_token": 31872}, {"true_arm": 4}, {"rand_margin": 20.0}, {"shuffled_margin": 20.0}):
        gate, results = _write_gate_pair(tmp_path / str(len(str(override))), _gate_row(**override))
        with pytest.raises(Stage2PlanError):
            load_selected_lambda(gate, results)


def test_select_multi_parent_rows_prefers_distinct_tasks(tmp_path):
    handoff = _handoff(tmp_path)
    rows = select_multi_parent_rows(handoff, max_parents=3)
    assert [r["task"] for r in rows] == ["butter", "cream_cheese", "ketchup"]


def test_select_multi_parent_rows_skips_invalid_rows(tmp_path):
    path = tmp_path / "handoff.csv"
    _write_csv(
        path,
        [
            {"task": "bad", "state_id": 0, "selected_step": 1, "status": "INFRA_INVALID"},
            {"task": "a", "state_id": 0, "selected_step": 1, "status": "PASS"},
            {"task": "b", "state_id": 0, "selected_step": 2, "status": "PASS"},
            {"task": "c", "state_id": 0, "selected_step": 3, "status": "PASS"},
        ],
    )
    rows = select_multi_parent_rows(path, max_parents=3)
    assert [r["task"] for r in rows] == ["a", "b", "c"]


def test_stage2_plan_uses_selected_lambda_seeds_conditions_and_gpu_mapping(tmp_path):
    gate, results = _write_gate_pair(tmp_path)
    selected = load_selected_lambda(gate, results)
    parents = select_multi_parent_rows(_handoff(tmp_path), max_parents=3)
    template = yaml.safe_load(Path("configs/m3_step78_true_pgd_31744_logratio_arm_v4.yaml").read_text(encoding="utf-8"))
    cfg = {
        "python": "/env/bin/python",
        "stage2_runner": "runner.py",
        "stage2_config": "config.yaml",
        "stage2_output_root": "/stage2",
    }
    plan = build_stage2_plan(cfg, selected, parents, config_dir=tmp_path / "configs", template_config=template)
    rows = flatten_command_rows(plan)
    assert len(rows) == 3 * len(STAGE2_PHASES)
    assert {r["selected_lambda"] for r in rows} == {0.25}
    assert {r["cuda_visible_devices"] for r in rows} == {"1,5"}
    assert {r["phase"] for r in rows} == set(STAGE2_PHASES)
    assert sum(1 for r in rows if r["phase"] == "capture_input") == 3
    assert sum(1 for r in rows if r["phase"] == "preflight_zero_step") == 3
    assert sum(1 for r in rows if r["phase"] == "canary_v4_seed81") == 3
    assert sum(1 for r in rows if r["phase"] == "canary_v4_seed82") == 3
    assert all("--condition" not in r["command"] for r in rows)
    assert all("--selected_lambda" not in r["command"] for r in rows)
    assert all("--task" not in r["command"] for r in rows)
    assert all("--render_gpu_device_id 1" in r["command"] for r in rows)
    assert all(Path(r["config_path"]).exists() for r in rows)
    parent_cfg = yaml.safe_load(Path(rows[0]["config_path"]).read_text(encoding="utf-8"))
    assert parent_cfg["attack_optimizer"]["arm_preserve_weight"] == 0.25
    assert parent_cfg["attack_optimizer"]["target_token_id"] == 31744


def test_s5_gate_requires_two_seeds_per_parent_and_two_parents_total():
    rows = []
    for parent in ("p1", "p2", "p3"):
        for seed in STAGE2_SEEDS:
            rows.append({"parent_id": parent, "seed": seed, "frame_status": "FRAME_FULL_SELECTIVE_PASS"})
    assert evaluate_s5_gate(rows)["status"] == "S5_MULTI_PARENT_PASS"
    rows[-1]["frame_status"] = "FAIL"
    rows[-2]["frame_status"] = "FAIL"
    gate = evaluate_s5_gate(rows)
    assert gate["status"] == "S5_MULTI_PARENT_PASS"
    rows[1]["frame_status"] = "FAIL"
    assert evaluate_s5_gate(rows)["status"] == "S5_MULTI_PARENT_FAIL"


def test_s6_enabled_only_after_s5_pass():
    assert s6_enabled_from_s5_gate({"status": "S5_MULTI_PARENT_PASS", "parent_pass_count": 2})
    assert not s6_enabled_from_s5_gate({"status": "S5_MULTI_PARENT_PASS", "parent_pass_count": 1})
    assert not s6_enabled_from_s5_gate({"status": "S5_MULTI_PARENT_FAIL", "parent_pass_count": 2})


def test_prepare_entry_writes_plan_without_gpu_execution(tmp_path):
    gate, results = _write_gate_pair(tmp_path)
    handoff = _handoff(tmp_path)
    cfg = _config(tmp_path, gate, results, handoff)

    class Args:
        config = str(cfg)
        s3_gate = ""
        s3_results_csv = ""
        handoff_csv = ""
        output_dir = ""

    run_prepare(Args())
    plan = json.loads((tmp_path / "prepare" / "m3_gpu15_stage2_plan.json").read_text(encoding="utf-8"))
    assert plan["no_gpu_execution"] is True
    assert plan["tomato_selected_lambda"] == 0.25
    ledger = (tmp_path / "prepare" / "m3_gpu15_stage2_command_ledger.csv").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" not in ledger
    assert "--mode canary_v4" in ledger
    assert "--condition" not in ledger
    assert "--selected_lambda" not in ledger


def test_arm_accessor_prefers_canonical_and_accepts_legacy_alias():
    assert read_required_int({"arm_prefix_match_count": "6", "arm_match_count": "2"}, canonical="arm_prefix_match_count", legacy_aliases=["arm_match_count"]) == 6
    assert read_required_int({"official_arm_match_count": "5"}, canonical="arm_prefix_match_count", legacy_aliases=["official_arm_match_count", "arm_match_count"]) == 5


def test_arm_accessor_missing_or_invalid_is_infra_invalid():
    with pytest.raises(TelemetrySchemaError, match="missing required telemetry field"):
        read_required_int({}, canonical="arm_prefix_match_count", legacy_aliases=["arm_match_count"])
    with pytest.raises(TelemetrySchemaError, match="not an integer"):
        read_required_int({"arm_prefix_match_count": "nan"}, canonical="arm_prefix_match_count")
