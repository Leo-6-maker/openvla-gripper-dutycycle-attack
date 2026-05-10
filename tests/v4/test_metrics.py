from gripper_attack.metrics import aggregate_episode_from_steps, aggregate_run, normalized_action_discrepancy_cleanref

def test_attacked_ratio_uses_attack_active():
    steps = [{"attack_active": False, "trigger_active_raw": True, "budget_blocked": True}, {"attack_active": True, "trigger_active_raw": True, "budget_blocked": False}]
    e = aggregate_episode_from_steps(steps, False, False, False)
    assert e["attacked_step_ratio"] == 0.5 and e["raw_trigger_rate"] == 1.0

def test_matched_clean_fr_drop():
    r = aggregate_run([{"success": False}, {"success": True}], [{"success": True}, {"success": True}])
    assert r["SR_clean_matched"] == 1.0 and r["FR_drop"] == 0.5


def test_cleanref_nad_normalizes_by_bounds():
    clean = [0.0, 0.5]
    executed = [0.5, 1.0]
    low = [-1.0, -1.0]
    high = [1.0, 1.0]
    # dim0: 0.5 / 1.0; dim1: 0.5 / 1.5 => mean 0.416666...
    val = normalized_action_discrepancy_cleanref(clean, executed, low, high, dims=[0, 1])
    assert abs(val - ((0.5 / 1.0 + 0.5 / 1.5) / 2.0)) < 1e-6

def test_aggregate_exposes_cleanref_and_l2_names():
    steps = [{"attack_active": True, "trigger_active_raw": True, "budget_blocked": False, "delta_l2": 2.0, "delta_linf": 1.0, "nad_cleanref_step": 0.25}]
    e = aggregate_episode_from_steps(steps, False, False, False)
    assert e["action_delta_l2_all"] == 2.0
    assert e["nad_cleanref_all"] == 0.25
    r = aggregate_run([e])
    assert r["action_delta_l2_mean"] == 2.0
    assert r["NAD_cleanref_mean"] == 0.25
