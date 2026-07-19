"""Unit tests for R7 K10 Opportunity Labeler V1.2.1."""
import json, pytest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "detector_v4"))
from label_k10_v121 import (
    compute_critical, compute_burst, K, STABLE_GRASP_MIN,
    LIFT_MIN, TARGET_PROGRESS_MIN, RELEASE_RISK_MAX, REGRASP_RISK_MAX,
)


def _r(**kw):
    """Build one synthetic Physics V2.1 record with safe defaults."""
    defaults = {
        "known_mask": True, "student_valid": True, "candidate_close": True,
        "stable_grasp_score": 0.8, "lift_score": 0.5, "support_removed": 0.0,
        "target_progress": 0.0, "target_progress_known": False,
        "release_risk": 0.0, "regrasp_or_instability_risk": 0.0,
        "task_grasp_necessity": 1.0,
        "component_valid_mask": {
            "lift_score": True, "support_removed": False,
            "target_progress": False, "relative_pose_stability": True,
            "release_risk": True, "regrasp_or_instability_risk": True,
            "object_eef_comotion_score": True,
        },
        "phase_name": "VALID_RETENTION", "window_id": "candidate:0",
        "step": 0, "physics_protocol_schema": "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21",
    }
    defaults.update(kw)
    if "step" not in kw:
        defaults["step"] = 0
    return defaults


def make_records(n, **overrides):
    recs = []
    for i in range(n):
        r = _r(step=i, **overrides)
        recs.append(r)
    return recs


class TestCritical:
    def test_all_conditions_met(self):
        recs = make_records(30)
        critical, reasons, bm = compute_critical(recs)
        assert all(critical[:20])

    def test_unknown_mask_blocks(self):
        recs = make_records(5, known_mask=False)
        critical, reasons, bm = compute_critical(recs)
        assert not any(critical)
        assert reasons[0] == "unknown_mask"

    def test_student_invalid_blocks(self):
        recs = make_records(5, student_valid=False)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "student_invalid"

    def test_not_candidate_close_blocks(self):
        recs = make_records(5, candidate_close=False)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "not_candidate_close"

    def test_task_role_blocks(self):
        recs = make_records(5, task_grasp_necessity=0.0)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "task_role_not_applicable"

    def test_stable_grasp_blocks(self):
        recs = make_records(5, stable_grasp_score=0.1)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "not_stable_grasp"

    def test_stable_grasp_unknown_blocks(self):
        recs = make_records(5, stable_grasp_score=0.1,
                           component_valid_mask={**_r()["component_valid_mask"],
                                                 "relative_pose_stability": False})
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "stable_grasp_unknown"

    def test_no_manipulation_blocks(self):
        recs = make_records(5, lift_score=0.0, support_removed=0.0, target_progress=0.0,
                           component_valid_mask={**_r()["component_valid_mask"],
                                                 "lift_score": True, "support_removed": True,
                                                 "target_progress": True})
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "not_manipulation_active"

    def test_release_unknown_blocks(self):
        recs = make_records(5, component_valid_mask={**_r()["component_valid_mask"],
                                                     "release_risk": False})
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "release_risk_unknown"

    def test_regrasp_unknown_blocks(self):
        recs = make_records(5, component_valid_mask={**_r()["component_valid_mask"],
                                                     "regrasp_or_instability_risk": False})
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "regrasp_risk_unknown"

    def test_release_risk_veto(self):
        recs = make_records(5, release_risk=0.8)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "release_risk"

    def test_regrasp_risk_veto(self):
        recs = make_records(5, regrasp_or_instability_risk=0.8)
        critical, reasons, bm = compute_critical(recs)
        assert reasons[0] == "regrasp_or_instability"

    def test_component_bitmask(self):
        recs = make_records(5, lift_score=0.5, support_removed=0.5, target_progress=0.1,
                           target_progress_known=True,
                           component_valid_mask={**_r()["component_valid_mask"],
                                                 "lift_score": True, "support_removed": True,
                                                 "target_progress": True})
        critical, reasons, bm = compute_critical(recs)
        assert critical[0]
        assert bm[0] == 7

    def test_lift_only_bitmask(self):
        recs = make_records(5, lift_score=0.5, support_removed=0.0, target_progress=0.0,
                           component_valid_mask={**_r()["component_valid_mask"],
                                                 "lift_score": True, "support_removed": False,
                                                 "target_progress": False})
        critical, reasons, bm = compute_critical(recs)
        assert critical[0]
        assert bm[0] == 1


class TestBurst:
    def test_exact_length_one_start(self):
        recs = make_records(30)
        critical, reasons, bm = compute_critical(recs)
        burst, is_start = compute_burst(critical, 30, recs)
        starts = [i for i, s in enumerate(is_start) if s]
        assert len(starts) == 21

    def test_window_id_segment_crossing_blocked(self):
        """Only starts 6–14 cross the boundary between steps 14 and 15."""
        recs = make_records(30)
        for i in range(15, 30):
            recs[i]["window_id"] = "candidate:1"
        critical, reasons, bm = compute_critical(recs)
        burst, is_start = compute_burst(critical, 30, recs)
        for t in range(6, 15):
            assert not burst[t], f"Window at {t} crosses segment boundary"
        assert burst[5]
        assert burst[15]

    def test_unknown_gap_blocks_critical_not_burst(self):
        recs = make_records(30)
        recs[15]["known_mask"] = False
        critical, reasons, bm = compute_critical(recs)
        burst, is_start = compute_burst(critical, 30, recs)
        for t in range(6, 16):
            assert not burst[t]
