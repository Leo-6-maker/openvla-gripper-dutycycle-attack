from __future__ import annotations

from gripper_attack.v5_scheduler import V5OneShotScheduler


def test_scheduler_replay_is_causal_min_dwell_and_one_shot():
    scheduler = V5OneShotScheduler()
    emitted = []
    for step in range(20):
        result = scheduler.update(
            step=step, candidate_close=True, valid=True,
            utility_probability=0.9, release_probability=0.0,
            regrasp_probability=0.0, uncertainty_probability=0.99,
        )
        emitted.append(result["emit"])
        assert result["teacher_inputs_consumed"] is False
    assert sum(emitted) == 1
    assert emitted[0] is False
    assert scheduler.emit_step >= 9
