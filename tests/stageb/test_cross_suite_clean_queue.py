from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.run_sc5_cross_suite_clean_queue import (  # noqa: E402
    build_jobs,
    command_for_job,
)


class Args:
    python = "/env/bin/python"
    detector_path = "/detector.pt"
    source_commit = "abc123"
    render_gpu = 6


def test_clean_queue_builds_balanced_300_episode_matrix():
    jobs = build_jobs(eval_seed=0)
    assert len(jobs) == 300
    assert sum(1 for j in jobs if j.wave == "A") == 150
    assert sum(1 for j in jobs if j.wave == "B") == 150
    assert {j.suite for j in jobs[:150]} == {"libero_spatial", "libero_goal", "libero_10"}
    for suite in {"libero_spatial", "libero_goal", "libero_10"}:
        suite_jobs = [j for j in jobs if j.suite == suite]
        assert len(suite_jobs) == 100
        assert {j.task_idx for j in suite_jobs} == set(range(10))
        assert {j.state_id for j in suite_jobs} == set(range(10))


def test_clean_queue_commands_are_clean_collector_only(tmp_path):
    job = build_jobs(eval_seed=0)[0]
    cmd = command_for_job(Args(), job, tmp_path / "out")
    joined = " ".join(cmd)
    assert "run_sc5_cross_suite_clean.py" in joined
    assert "--save_video" in cmd
    assert "VIS" not in joined
    assert "RAND" not in joined
    assert "attack" not in joined.lower()
