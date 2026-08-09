from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def _replay(config: V5SchedulerConfig) -> dict:
    scheduler = V5OneShotScheduler(config)
    result = {}
    for step in range(10):
        result = scheduler.update(
            step=step,
            candidate_close=True,
            valid=True,
            utility_probability=0.9,
            release_probability=0.9,
            regrasp_probability=0.1,
            uncertainty_probability=0.0,
        )
    return result


def test_veto_ablation_switches_are_explicit():
    blocked = _replay(V5SchedulerConfig())
    open_release = _replay(V5SchedulerConfig(release_veto_enabled=False))
    assert blocked["emit"] is False
    assert open_release["emit"] is True
