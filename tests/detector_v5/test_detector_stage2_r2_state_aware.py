import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from run_detector_stage2_r2 import detailed_candidate_metrics, scheduler_trace  # noqa: E402


def _rows(scores):
    return [
        {"episode_id": "suite/task/state", "step": index, "score": score, "target": 0.0}
        for index, score in enumerate(scores)
    ]


def test_anticipatory_activation_is_event_overlap_not_false_onset():
    trace = scheduler_trace(_rows([0.0, 1.0, 1.0, 1.0, 0.0]), 0.5, 1, 0.25, 0)
    assert trace[1]["emission"]
    assert all(trace[index]["latched_active"] for index in (2, 3))
    metrics, _ = detailed_candidate_metrics(
        _rows([0.0, 1.0, 1.0, 1.0, 0.0]),
        {"suite/task/state": [{"fragment_ranges": [(2, 3)], "fragment_count": 1}]},
        {"threshold": 0.5, "persistence": 1, "hysteresis": 0.25, "cooldown": 0},
    )
    assert metrics["active_overlap_event_recall"] == 1.0
    assert metrics["anticipatory_event_recall_at_2"] == 1.0
    assert metrics["false_onset_count"] == 0


def test_event_internal_activation_reports_delay_two():
    trace = scheduler_trace(_rows([0.0, 0.0, 1.0, 1.0, 0.0]), 0.5, 1, 0.25, 0)
    assert trace[2]["emission"]
    assert trace[2]["latched_active"]


def test_activation_after_event_is_not_event_hit():
    trace = scheduler_trace(_rows([0.0, 0.0, 0.0, 0.0, 1.0]), 0.5, 1, 0.25, 0)
    assert not any(row["latched_active"] for row in trace[:2])
    assert trace[4]["emission"]


def test_latch_can_cover_multiple_formal_fragments():
    trace = scheduler_trace(
        [{"episode_id": "suite/task/state", "step": step, "score": 1.0, "target": 1.0} for step in (1, 2, 3)],
        0.5,
        1,
        0.0,
        0,
    )
    assert any(row["latched_active"] for row in trace if row["step"] in (1, 3))
