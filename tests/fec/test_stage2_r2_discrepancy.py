from __future__ import annotations

import numpy as np
import pytest

from fec.audit_stage2_r2_discrepancy import (
    correlation,
    false_onset_detail,
    rank_values,
)


def test_rank_values_uses_average_ties():
    assert rank_values(np.asarray([2.0, 1.0, 1.0, 3.0])).tolist() == [3.0, 1.5, 1.5, 4.0]


def test_correlation_and_false_onset_detail():
    left = np.asarray([1.0, 2.0, 3.0])
    assert correlation(left, left * 2.0) > 0.99
    trace = [
        {"step": 0, "score": 0.0, "emission": False, "latched_active": False},
        {"step": 1, "score": 0.6, "emission": True, "latched_active": True},
    ]
    detail = false_onset_detail(trace, [], 0.5)
    assert detail["false_onset"] is True
    assert detail["first_false_onset_step"] == 1
    assert detail["first_false_onset_margin"] == pytest.approx(0.1)
