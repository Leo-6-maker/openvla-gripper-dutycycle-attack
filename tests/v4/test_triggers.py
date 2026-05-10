import numpy as np
from gripper_attack.types import TriggerContext
from gripper_attack.triggers import RandomBernoulliBudgetedTrigger, PeriodicBudgetedTrigger, EntropyThresholdTrigger, MarginThresholdTrigger

def ctx(step, rho=0.1, logits=None):
    return TriggerContext("task", 1, step, rho, prefix_logits=logits)

def test_random_reproducible_independent():
    a = RandomBernoulliBudgetedTrigger(seed=7); b = RandomBernoulliBudgetedTrigger(seed=7)
    assert [a.evaluate(ctx(i)).raw_active for i in [5,1,3]] == [b.evaluate(ctx(i)).raw_active for i in [5,1,3]]

def test_periodic():
    p = PeriodicBudgetedTrigger()
    assert p.evaluate(ctx(0, 0.2)).raw_active
    assert not p.evaluate(ctx(1, 0.2)).raw_active
    assert p.evaluate(ctx(5, 0.2)).raw_active

def test_entropy_threshold():
    th = {"tasks": {"task": {"rho_0.10": {"entropy": 0.1, "margin": 10}}}}
    assert EntropyThresholdTrigger(th).evaluate(ctx(0, 0.1, np.zeros((2,3)))).raw_active

def test_margin_threshold():
    th = {"tasks": {"task": {"rho_0.10": {"entropy": 0.1, "margin": 10}}}}
    assert MarginThresholdTrigger(th).evaluate(ctx(0, 0.1, np.zeros((2,3)))).raw_active
