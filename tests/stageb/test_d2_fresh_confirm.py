"""D2.1a regression tests: fresh confirmation candidate grouping.
Tests import select_eligible_multi_traces() directly from the production script.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))

from run_d2_fresh_confirm import select_eligible_multi_traces


def test_multi_candidate_trace_is_identified():
    """A trace with 3 candidates, 1 positive, and ELIGIBLE_MULTI category → selected."""
    candidates = [
        {"trace_id": "t1", "task_key": "a", "state_id": "0",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t1", "task_key": "a", "state_id": "0",
         "is_teacher_p": "0", "candidate_step": "20", "total_score": "2.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t1", "task_key": "a", "state_id": "0",
         "is_teacher_p": "0", "candidate_step": "30", "total_score": "1.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    status = [{"trace_id": "t1", "category": "ELIGIBLE_MULTI_CANDIDATE"}]
    result = select_eligible_multi_traces(candidates, status)
    assert len(result) == 1
    assert len(result["t1"]) == 3


def test_single_candidate_trace_is_excluded():
    """Trace with 1 candidate → excluded from multi."""
    candidates = [
        {"trace_id": "t2", "task_key": "b", "state_id": "1",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    status = [{"trace_id": "t2", "category": "ELIGIBLE_MULTI_CANDIDATE"}]
    result = select_eligible_multi_traces(candidates, status)
    assert len(result) == 0


def test_ambiguous_trace_is_excluded_by_category():
    """3 candidates, 2 positives, TEACER_P_AMBIGUOUS category → excluded."""
    candidates = [
        {"trace_id": "t3", "task_key": "c", "state_id": "2",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t3", "task_key": "c", "state_id": "2",
         "is_teacher_p": "1", "candidate_step": "20", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t3", "task_key": "c", "state_id": "2",
         "is_teacher_p": "0", "candidate_step": "30", "total_score": "1.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    status = [{"trace_id": "t3", "category": "TEACHER_P_AMBIGUOUS"}]
    result = select_eligible_multi_traces(candidates, status)
    assert len(result) == 0


def test_eligible_single_category_is_excluded():
    """Trace with ELIGIBLE_SINGLE category but 3 candidates → still excluded."""
    candidates = [
        {"trace_id": "t4", "task_key": "d", "state_id": "3",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t4", "task_key": "d", "state_id": "3",
         "is_teacher_p": "0", "candidate_step": "20", "total_score": "2.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    status = [{"trace_id": "t4", "category": "ELIGIBLE_SINGLE_CANDIDATE"}]
    result = select_eligible_multi_traces(candidates, status)
    assert len(result) == 0


def test_multiple_traces_mixed_categories():
    """Two traces: one eligible multi, one unavailable → only multi selected."""
    candidates = [
        {"trace_id": "t5", "task_key": "e", "state_id": "4",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t5", "task_key": "e", "state_id": "4",
         "is_teacher_p": "0", "candidate_step": "20", "total_score": "2.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t6", "task_key": "f", "state_id": "5",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
        {"trace_id": "t6", "task_key": "f", "state_id": "5",
         "is_teacher_p": "0", "candidate_step": "20", "total_score": "2.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    status = [
        {"trace_id": "t5", "category": "ELIGIBLE_MULTI_CANDIDATE"},
        {"trace_id": "t6", "category": "TEACHER_P_UNAVAILABLE"},
    ]
    result = select_eligible_multi_traces(candidates, status)
    assert len(result) == 1
    assert "t5" in result
    assert "t6" not in result
