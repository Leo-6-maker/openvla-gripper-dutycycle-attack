import json
from pathlib import Path

from scripts.stage_x.audit_stage_x1r_t1d0_static_authority import audit


ROOT = Path(__file__).resolve().parents[2]


def test_t1d0_static_hold_is_fail_closed():
    protocol = json.loads((ROOT / "configs/STAGE_X_X1R_T1D0_ATTACK_PARENT_AUTHORITY_V1.json").read_text())
    stage_ix = json.loads((ROOT / "configs/STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json").read_text())
    receipt = audit(protocol, stage_ix)
    assert receipt["status"] == "STAGE_X_X1R_T1D0_HOLD_TIMING_ANCHOR_AUTHORITY"
    assert receipt["selected_static_candidate_count"] == 39
    assert receipt["missing_task_slots"] == ["libero_goal/task_01"]
    assert receipt["x1r_pgd_executed"] is False
    assert receipt["env_step_executed"] is False
    assert not receipt["errors"]


def test_static_audit_has_no_model_or_runtime_execution_imports():
    source = (ROOT / "scripts/stage_x/audit_stage_x1r_t1d0_static_authority.py").read_text().lower()
    assert "import torch" not in source
    assert "transformers" not in source
    assert "env.step(" not in source
