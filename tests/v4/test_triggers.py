import numpy as np
from gripper_attack.types import TriggerContext
from gripper_attack.triggers import (
    RandomBernoulliBudgetedTrigger,
    PeriodicBudgetedTrigger,
    EntropyThresholdTrigger,
    MarginThresholdTrigger,
    MokaSecondPotRelativeWindowBudgetedTrigger,
)

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


def test_moka_relative_trigger_respects_stage_and_anchor(monkeypatch):
    monkeypatch.setenv("V4_MOKA_SECOND_WINDOW_START", "2")
    monkeypatch.setenv("V4_MOKA_SECOND_WINDOW_END", "5")
    trig = MokaSecondPotRelativeWindowBudgetedTrigger()
    # first phase blocked
    d0 = trig.evaluate(TriggerContext("task", 0, 10, 1.0, metadata={"moka_stage_id": "first_pot_phase", "moka_stage_anchor_step": 3}))
    assert not d0.raw_active
    # second phase but outside relative window
    d1 = trig.evaluate(TriggerContext("task", 0, 4, 1.0, metadata={"moka_stage_id": "second_pot_phase", "moka_stage_anchor_step": 3}))
    assert not d1.raw_active
    # in second phase and inside relative window
    d2 = trig.evaluate(TriggerContext("task", 0, 6, 1.0, metadata={"moka_stage_id": "second_pot_phase", "moka_stage_anchor_step": 3}))
    assert d2.raw_active


def test_moka_relative_trigger_honors_phase_disable_flag(monkeypatch):
    monkeypatch.setenv("V4_MOKA_SECOND_WINDOW_START", "0")
    monkeypatch.setenv("V4_MOKA_SECOND_WINDOW_END", "99")
    trig = MokaSecondPotRelativeWindowBudgetedTrigger()
    dec = trig.evaluate(TriggerContext("task", 0, 10, 1.0, metadata={"moka_stage_id": "second_pot_phase", "moka_stage_anchor_step": 1}, phase_attack_enabled=False))
    assert not dec.raw_active
