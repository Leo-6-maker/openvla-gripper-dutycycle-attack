import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.select_provisional_layer3_two_suite_parents import select_parents, selection_key  # noqa: E402


def row(suite: str, key: str, status: str = "ELIGIBLE_EVENT", stratum: str = "DEV_CANARY", extra: dict | None = None):
    out = {
        "review_round_id": "round",
        "review_stratum": stratum,
        "review_id": f"review_{key}",
        "episode_key": key,
        "suite": suite,
        "task_idx": "1",
        "state_id": "2",
        "teacher_status": status,
        "event_id": f"{key}|event0",
        "proposed_anchor": "10",
        "proposed_window_start": "8",
        "proposed_window_end": "20",
        "raw_video_path": "/tmp/raw.mp4",
        "teacher_only_timeline_path": "/tmp/timeline.csv",
        "teacher_only_overlay_path": "/tmp/overlay.mp4",
    }
    if extra:
        out.update(extra)
    return out


def test_selects_first_two_per_suite_by_frozen_hash():
    rows = [
        row("libero_spatial", "sp_a"),
        row("libero_spatial", "sp_b"),
        row("libero_spatial", "sp_c"),
        row("libero_goal", "go_a"),
        row("libero_goal", "go_b"),
        row("libero_goal", "go_c"),
    ]
    selected, audit = select_parents(rows, per_suite=2)
    assert audit["status"] == "PASS"
    assert len(selected) == 4
    for suite in ["libero_spatial", "libero_goal"]:
        expected = sorted([r for r in rows if r["suite"] == suite], key=selection_key)[:2]
        got = [r for r in selected if r["suite"] == suite]
        assert [r["canonical_episode_key"] for r in got] == [r["episode_key"] for r in expected]


def test_ignores_noneligible_and_libero10_rows():
    rows = [
        row("libero_spatial", "sp_ok"),
        row("libero_spatial", "sp_bad", status="NO_RELEVANT_GRASP_EVENT"),
        row("libero_goal", "go_ok"),
        row("libero_10", "l10_ok"),
    ]
    selected, audit = select_parents(rows, per_suite=2)
    assert audit["status"] == "REDUCED_DENOMINATOR"
    assert {r["canonical_episode_key"] for r in selected} == {"sp_ok", "go_ok"}


def test_selection_does_not_change_when_forbidden_fields_change():
    base = [
        row("libero_spatial", "sp_a", extra={"student_emit": "999", "VIS_result": "bad"}),
        row("libero_spatial", "sp_b", extra={"student_emit": "1", "VIS_result": "good"}),
        row("libero_goal", "go_a", extra={"task_success": "true", "reviewer_id": "A"}),
        row("libero_goal", "go_b", extra={"task_success": "false", "reviewer_id": "B"}),
    ]
    changed = [dict(r) for r in base]
    for r in changed:
        r["student_emit"] = "0"
        r["VIS_result"] = "flipped"
        r["task_success"] = "false"
        r["reviewer_id"] = "changed"
    selected_a, _ = select_parents(base, per_suite=1)
    selected_b, _ = select_parents(changed, per_suite=1)
    assert [r["canonical_episode_key"] for r in selected_a] == [r["canonical_episode_key"] for r in selected_b]

