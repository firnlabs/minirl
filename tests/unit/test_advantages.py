"""Tests for advantages.py."""

from minirl import advantages


def test_compute_discounted_returns():
    gamma = 0.9
    rewards = [1.0, 1.0, 1.0, 1.0]
    num_rewards = len(rewards)
    expected_returns = [0.0 for _ in range(num_rewards + 1)]
    for i in reversed(range(num_rewards)):
        expected_returns[i] = rewards[i] + expected_returns[i + 1] * gamma

    returns = advantages.compute_discounted_returns(rewards, gamma)

    assert returns == expected_returns[:-1]
