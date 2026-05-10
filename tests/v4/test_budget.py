from gripper_attack.budget import OnlineBudgetController

def test_zero_budget():
    b = OnlineBudgetController(0, 30)
    assert b.budget_max_steps == 0
    assert not b.decide(True).attack_active

def test_floor_budget():
    assert OnlineBudgetController(0.1, 30).budget_max_steps == 3

def test_blocked_after_budget():
    b = OnlineBudgetController(0.1, 10)
    assert b.decide(True).attack_active
    d = b.decide(True)
    assert not d.attack_active and d.budget_blocked
