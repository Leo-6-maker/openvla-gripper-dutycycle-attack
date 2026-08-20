import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_q3r3_d_protocol_is_engineering_only_and_resource_bound():
    protocol = json.loads((ROOT / "configs/STAGE_X_X1R2_Q3R3_ENGINEERING_MATRIX_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_ENGINEERING_ONLY_PRE_GPU"
    assert protocol["scientific_authority"] is False
    assert len(protocol["fixtures"]) == 4
    assert len(protocol["arms"]) == 5
    resource = protocol["resource"]
    assert resource["free_memory_mib_strictly_greater_than"] == 20480
    assert resource["one_project_worker_per_physical_gpu"] is True
    assert resource["max_project_workers"] == 8
    assert resource["foreign_processes_untouched"] is True
    assert protocol["protected_boundary"]["physical_interventions"] == 0
    assert protocol["protected_boundary"]["vphys_reads"] == 0
    assert protocol["protected_boundary"]["eval160"] == "UNREAD"


def test_q3r3_d_runner_has_fail_closed_branch_gates():
    source = (ROOT / "scripts/stage_x/run_stage_x1r2_q3r3_engineering_matrix.py").read_text(encoding="utf-8")
    for marker in (
        "capture_branch_state",
        "compare_branch_state",
        "STRICT_CANDIDATE_AUDIT_V1",
        "D_ARM_ISOLATION_INVALID",
        "D_BRANCH_TERMINATED_BEFORE_15_ACTIONS",
        "free_memory_mib",
        "physical_interventions",
        "vphys_reads",
        "eval160",
    ):
        assert marker in source
    assert "read_eval160" not in source
    assert "read_vphys" not in source
    assert "physical_intervention(" not in source
