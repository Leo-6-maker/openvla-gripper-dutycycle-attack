from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_requires_pre_m4_boundary() -> None:
    source = (ROOT / "scripts/detector_v5/seal_stage_v_primary_teacher_student_freeze.py").read_text(encoding="utf-8")
    assert "PASS_PRIMARY_TEACHER_STUDENT_FREEZE" in source
    assert '"formal_m4_authorized": False' in source
    assert '"m4_outcomes_read": False' in source
    assert '"normalization_drift", {}).get("status") != "PASS_RECOMPUTED_CONSISTENT"' in source


def test_pre_m4_lock_binds_freeze_and_exact_plan() -> None:
    source = (ROOT / "scripts/detector_v5/seal_stage_v_pre_m4_lock.py").read_text(encoding="utf-8")
    assert "PASS_PRE_M4_LOCK" in source
    assert '"planned_branch_count": 3840' in source
    assert '"teacher_predictions_read": False' in source
    assert '"student_predictions_read": False' in source


def test_freeze_accepts_exact_plan_root_seal_format() -> None:
    source = (ROOT / "scripts/detector_v5/seal_stage_v_primary_teacher_student_freeze.py").read_text(encoding="utf-8")
    assert "ROOT_SEAL.sha256" in source
    assert "_sealed_exact_plan" in source
