from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R_T1D1_SCREENING_CLEAN_PROTOCOL_V1.json"
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_t1d1_screening_clean.py"


def test_d1_protocol_is_frozen_and_closed_for_attack():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_FOR_SCREENING_CLEAN_EXECUTION"
    assert protocol["parent_population"]["count"] == 39
    assert protocol["parent_population"]["replacement"] is False
    assert protocol["parent_population"]["canary_ordinals"] == [1, 11, 20, 30]
    assert protocol["authorization"]["screening_clean_authorized"] is True
    assert protocol["authorization"]["pgd_authorized"] is False
    assert protocol["authorization"]["physical_intervention_authorized"] is False
    assert protocol["authorization"]["next_gate"] == "PROTOCOL_ATTACK_ELIGIBLE_PRE_MANUAL_REVIEW"
    assert protocol["protected_boundary"]["eval160"] == "UNREAD"
    assert protocol["protected_boundary"]["protected_evaluation"] == "UNREAD"


def test_d1_runner_has_no_top_level_attack_or_simulator_import():
    module = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported = []
    for node in module.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module.split(".")[0] if node.module else "")
    assert not set(imported) & {"torch", "transformers", "libero", "mujoco", "robosuite", "gym"}


def test_d1_parent_seed_report_has_exact_39_no_replacement_rows():
    report = json.loads((ROOT / "reports/STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1.json").read_text(encoding="utf-8"))
    rows = report["rows"]
    assert report["status"] == "PASS_D0R1_INVARIANTS"
    assert len(rows) == 39
    assert len({row["canonical_parent_key"] for row in rows}) == 39
    assert report["population"]["replacement"] is False
    assert report["population"]["missing_cell"] == ["libero_goal/task_01"]
