"""Test R9P FSM state machine: persistence, release veto, one-shot, reset."""
import unittest

from src.gripper_attack.c2g_gripper_critical_window_detector import (
    FixedBurstTriggerScheduler,
    SchedulerState,
)


class FSMTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = FixedBurstTriggerScheduler(
            burst_length=10,
            tau_critical=0.5,
            tau_release=0.5,
            tau_ground=0.5,
            persistence_window=3,
            persistence_required=2,
        )

    def test_initial_state_idle(self):
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)

    def test_single_gate_no_trigger(self):
        d = self.scheduler.update(
            critical_probability=0.9,
            release_safe_probability=0.1,
            grounding_confidence_probability=0.9,
            valid=True,
        )
        self.assertTrue(d.gate_now)
        self.assertFalse(d.trigger_started)
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)

    def test_persistence_2_of_3_triggers(self):
        # Two gates within 3 steps should trigger
        for _ in range(2):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=True)
        # Not yet — need 3rd step to have window of 3
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)
        d = self.scheduler.update(critical_probability=0.4, release_safe_probability=0.8,
                                  grounding_confidence_probability=0.2, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.BURST)
        self.assertTrue(d.trigger_started)

    def test_release_veto_prevents_trigger(self):
        for _ in range(3):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.9,
                                  grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)

    def test_grounding_gate_prevents_trigger(self):
        for _ in range(3):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.1, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)

    def test_one_shot_no_second_trigger(self):
        # Trigger first attack (consumes frame 0)
        for _ in range(3):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.BURST)
        # Complete burst: 9 remaining frames (indices 1-9)
        for _ in range(9):
            d = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                      grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.DONE)
        # Next update should stay DONE
        d = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(d.state, SchedulerState.DONE)
        self.assertFalse(d.trigger_started)

    def test_burst_length_exact(self):
        # 3rd update triggers and consumes frame 0 immediately
        d_trigger = None
        for _ in range(3):
            d_trigger = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                              grounding_confidence_probability=0.9, valid=True)
        self.assertTrue(d_trigger.trigger_started)
        self.assertTrue(d_trigger.attack_active)
        self.assertEqual(d_trigger.attack_index, 0)
        # Remaining B-1 frames
        emitted = 1  # frame 0 consumed by trigger
        for _ in range(9):
            d = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                      grounding_confidence_probability=0.9, valid=True)
            if d.attack_active:
                emitted += 1
        self.assertEqual(emitted, 10)
        # Next update should be DONE
        d = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.DONE)
        self.assertFalse(d.attack_active)

    def test_reset(self):
        for _ in range(3):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(self.scheduler.state, SchedulerState.BURST)
        self.scheduler.reset()
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)
        self.assertEqual(self.scheduler._emitted, 0)
        self.assertEqual(self.scheduler._gate_history, [])

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            self.scheduler.update(critical_probability=1.5, release_safe_probability=0.5,
                                  grounding_confidence_probability=0.5, valid=True)
        with self.assertRaises(ValueError):
            self.scheduler.update(critical_probability=0.5, release_safe_probability=-0.1,
                                  grounding_confidence_probability=0.5, valid=True)

    def test_valid_false_no_trigger(self):
        for _ in range(5):
            self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                  grounding_confidence_probability=0.9, valid=False)
        self.assertEqual(self.scheduler.state, SchedulerState.IDLE)

    def test_different_persistence_3_of_5(self):
        s = FixedBurstTriggerScheduler(
            burst_length=10, tau_critical=0.5, tau_release=0.5, tau_ground=0.5,
            persistence_window=5, persistence_required=3,
        )
        # 3 gates in 5 steps
        for i in range(5):
            crit = 0.9 if i < 3 else 0.1
            s.update(critical_probability=crit, release_safe_probability=0.1,
                     grounding_confidence_probability=0.9, valid=True)
        self.assertEqual(s.state, SchedulerState.BURST)

    def test_attack_index_increments(self):
        d_trigger = None
        for _ in range(3):
            d_trigger = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                              grounding_confidence_probability=0.9, valid=True)
        # Trigger frame consumes index 0
        self.assertEqual(d_trigger.attack_index, 0)
        # Subsequent frames get 1, 2, 3
        idx = []
        for _ in range(3):
            d = self.scheduler.update(critical_probability=0.9, release_safe_probability=0.1,
                                      grounding_confidence_probability=0.9, valid=True)
            idx.append(d.attack_index)
        self.assertEqual(idx, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
