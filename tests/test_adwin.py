import random
import unittest

from tests import _path  # noqa: F401

from RetrainingPolicy import ADWINErrorRetrainingPolicy


def run_stream(policy: ADWINErrorRetrainingPolicy, p_before: float, p_after: float, seed: int = 0):
    rng = random.Random(seed)
    updates = []
    for i in range(2000):
        p = p_before if i < 1000 else p_after
        y_true = 0
        y_pred = 1 if rng.random() < p else 0
        updates.append(policy.update(y_true, y_pred))
    return updates


class OneSidedADWINTest(unittest.TestCase):

    def test_error_increase_is_degradation(self):
        updates = run_stream(ADWINErrorRetrainingPolicy(delta=0.1, one_sided=True), 0.2, 0.5)
        accepted = [u for u in updates if u.drift_detected]
        self.assertTrue(accepted)
        self.assertTrue(all(u.direction == 'increase' for u in accepted))

    def test_error_decrease_is_ignored_when_one_sided(self):
        policy = ADWINErrorRetrainingPolicy(delta=0.1, one_sided=True)
        updates = run_stream(policy, 0.5, 0.2)
        self.assertTrue(any(u.raw_drift_detected for u in updates))
        self.assertFalse(any(u.drift_detected for u in updates))
        self.assertGreater(policy.ignored_improvements, 0)

    def test_two_sided_keeps_old_behaviour(self):
        updates = run_stream(ADWINErrorRetrainingPolicy(delta=0.1, one_sided=False), 0.5, 0.2)
        self.assertTrue(any(u.drift_detected for u in updates))


if __name__ == '__main__':
    unittest.main()
