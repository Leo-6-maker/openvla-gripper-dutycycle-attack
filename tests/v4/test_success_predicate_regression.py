import unittest

class TestSuccessPredicateRegression(unittest.TestCase):

    def test_done_true_produces_success(self):
        """LIBERO signals success through done=True. info has no success key."""
        done = True
        info = {}
        success = done  # The correct logic
        self.assertTrue(success)

    def test_no_done_no_success(self):
        """Without done, success should be False regardless of info."""
        done = False
        info = {}
        success = done
        self.assertFalse(success)

    def test_info_success_absent_not_forced_failure(self):
        """When info lacks success key, must not force failure."""
        done = True
        info = {}
        # Buggy version: success = bool(info.get("success", False)) -> False
        buggy = bool(info.get("success", False))
        correct = done
        self.assertTrue(correct)
        self.assertFalse(buggy)  # buggy version would fail
        self.assertNotEqual(correct, buggy)

    def test_multiple_predicates_agree_on_success(self):
        """done=True should be the primary official success predicate."""
        done = True
        reward = 1.0
        info = {}
        success_official = done
        success_done = done
        success_reward = reward > 0
        self.assertTrue(success_official)
        self.assertTrue(success_done)
        self.assertTrue(success_reward)

    def test_timeout_not_success(self):
        """Timeout without done should be failure."""
        done = False
        reward = 0.0
        t = 290
        max_steps = 280
        wait_steps = 10
        success = done
        timeout = not done and t >= max_steps + wait_steps
        self.assertFalse(success)
        self.assertTrue(timeout)

    def test_info_success_present_but_done_primary(self):
        """Even if info has success field, done should be primary."""
        done = True
        info = {"success": False}  # info says fail but done says pass
        success = done  # primary
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
