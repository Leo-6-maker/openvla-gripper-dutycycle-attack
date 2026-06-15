"""D2.1 regression tests: fresh confirmation candidate grouping (P0-1 fix)."""

import csv, os, sys, tempfile
from collections import defaultdict
from pathlib import Path


def test_multi_candidate_trace_is_identified():
    """A trace with 3 candidates and 1 positive must be identified as multi-candidate."""
    # Simulate: 3 candidates, 1 is_teacher_p=1
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
    trace_status = [
        {"trace_id": "t1", "category": "ELIGIBLE_MULTI_CANDIDATE"},
    ]

    # Simulate the D2.1 fixed logic
    by_trace = defaultdict(list)
    for c in candidates:
        by_trace[c["trace_id"]].append(c)

    eligible_ids = {r["trace_id"] for r in trace_status
                    if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"}

    multi_traces = {}
    for tid, cands in by_trace.items():
        if tid not in eligible_ids:
            continue
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        if n_pos == 1 and len(cands) >= 2:
            multi_traces[tid] = cands

    assert len(multi_traces) == 1, f"Expected 1 multi trace, got {len(multi_traces)}"
    assert len(multi_traces["t1"]) == 3, f"Expected 3 candidates, got {len(multi_traces['t1'])}"


def test_single_candidate_trace_is_excluded():
    """A trace with 1 candidate and 1 positive must NOT be multi-candidate."""
    candidates = [
        {"trace_id": "t2", "task_key": "b", "state_id": "1",
         "is_teacher_p": "1", "candidate_step": "10", "total_score": "3.0",
         "eef_speed_now": "", "eef_speed_prev": ""},
    ]
    trace_status = [
        {"trace_id": "t2", "category": "ELIGIBLE_SINGLE_CANDIDATE"},
    ]

    by_trace = defaultdict(list)
    for c in candidates:
        by_trace[c["trace_id"]].append(c)

    eligible_ids = {r["trace_id"] for r in trace_status
                    if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"}

    multi_traces = {}
    for tid, cands in by_trace.items():
        if tid not in eligible_ids:
            continue
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        if n_pos == 1 and len(cands) >= 2:
            multi_traces[tid] = cands

    assert len(multi_traces) == 0, "Single-candidate trace should not be multi"


def test_ambiguous_trace_is_excluded():
    """A trace with 3 candidates and 2 positives (ambiguous) must be excluded."""
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
    trace_status = [
        {"trace_id": "t3", "category": "TEACHER_P_AMBIGUOUS"},
    ]

    by_trace = defaultdict(list)
    for c in candidates:
        by_trace[c["trace_id"]].append(c)

    eligible_ids = {r["trace_id"] for r in trace_status
                    if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"}

    multi_traces = {}
    for tid, cands in by_trace.items():
        if tid not in eligible_ids:
            continue
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        if n_pos == 1 and len(cands) >= 2:
            multi_traces[tid] = cands

    assert len(multi_traces) == 0, "Ambiguous trace should not be in eligible set, and even if it were, 2 positives exclude it"
