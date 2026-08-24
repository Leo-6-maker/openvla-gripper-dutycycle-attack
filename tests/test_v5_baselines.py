from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.detector_v5.run_v5_window_baselines import choose_window


def test_longest_window_baseline_is_deterministic():
    windows = [
        {"window_id": "a", "start_step": 2, "step_count": 4},
        {"window_id": "b", "start_step": 8, "step_count": 7},
    ]
    assert choose_window(windows, "B3_LONGEST") == 1
    assert choose_window(windows, "B0_RANDOM", seed=20260717) == choose_window(windows, "B0_RANDOM", seed=20260717)
