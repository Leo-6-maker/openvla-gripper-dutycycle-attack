from __future__ import annotations

import json
from pathlib import Path


def test_q3r3_c_protocol_is_engineering_only_and_strictly_bound():
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root / "configs/STAGE_X_X1R2_Q3R3_BRANCH_REPLAY_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_ENGINEERING_BRANCH_REPLAY_ONLY"
    assert protocol["scientific_authority"] is False
    assert protocol["reference_clean"]["t5"] == 5
    assert protocol["reference_clean"]["h_phys"] == 10
    assert protocol["branch_replay"]["repeat_count"] == 2
    assert protocol["branch_replay"]["prebranch_openvla_calls"] == 0
    assert protocol["branch_replay"]["prebranch_student_calls"] == 0
    assert protocol["state_contract"]["float_atol"] == 1e-12
    assert protocol["state_contract"]["float_rtol"] == 0.0
    assert protocol["resource"]["free_memory_mib_strictly_greater_than"] == 20480
    assert protocol["resource"]["max_project_workers"] == 8


def test_q3r3_c_runner_checks_committed_source_before_model_load():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/stage_x/run_stage_x1r2_q3r3_branch_replay.py").read_text(encoding="utf-8")
    assert 'source = source_receipt(args.source_commit, args.source_tree)' in source
    assert "verify_student_source_binding(authority)" in source
    assert "seed_all(seed)" in source
    assert "clean.load_openvla" in source
    assert "physical_interventions" in source


def test_q3r3_c_aggregator_has_no_runtime_or_attack_dependency():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/stage_x/audit_stage_x1r2_q3r3_branch_replay.py").read_text(encoding="utf-8").lower()
    for forbidden in ("import torch", "transformers", "mujoco", "from libero", "env.step("):
        assert forbidden not in source
