import unittest

from tests import _path  # noqa: F401

from RetrainingPolicy import ADWINErrorDecentralizedRetrainingPolicy


class QuorumTest(unittest.TestCase):

    def test_sticky_flags_accumulate_until_retrain(self):
        policy = ADWINErrorDecentralizedRetrainingPolicy(
            drifted_dt_fraction_threshold=0.4, min_comparable_dts=5, sticky_dt_flags=True,
        )
        evaluated = {f'dt{i}' for i in range(5)}
        first = policy.decide({'dt0'}, evaluated)
        self.assertFalse(first.retrain)
        second = policy.decide({'dt1'}, evaluated)
        self.assertTrue(second.retrain)  # 2 of 5 DTs flagged since retrain
        policy.on_retrain()
        third = policy.decide(set(), evaluated)
        self.assertFalse(third.retrain)
        self.assertEqual(third.drifted_dt_ids, [])

    def test_non_sticky_flags_only_count_current_batch(self):
        policy = ADWINErrorDecentralizedRetrainingPolicy(
            drifted_dt_fraction_threshold=0.4, min_comparable_dts=5, sticky_dt_flags=False,
        )
        evaluated = {f'dt{i}' for i in range(5)}
        policy.decide({'dt0'}, evaluated)
        self.assertFalse(policy.decide({'dt1'}, evaluated).retrain)

    def test_small_institutions_can_signal(self):
        policy = ADWINErrorDecentralizedRetrainingPolicy(
            drifted_dt_fraction_threshold=0.2, min_comparable_dts=5,
        )
        decision = policy.decide({'dt0'}, {'dt0', 'dt1', 'dt2'}, population_size=3)
        self.assertTrue(decision.eligible)
        self.assertTrue(decision.retrain)
        self.assertEqual(decision.min_comparable_dts, 3)


if __name__ == '__main__':
    unittest.main()
