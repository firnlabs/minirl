"""Tests for advantages.py."""

import torch

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


def test_compute_gae():
    timesteps = 5
    rewards = torch.rand(timesteps)
    values = torch.rand(timesteps)
    last_value = torch.rand(1).squeeze()
    dones = torch.randint(0, 2, (timesteps,)).bool()

    advantages_, returns = advantages.compute_gae(
        rewards=rewards, values=values, last_value=last_value, dones=dones
    )

    assert len(advantages_) == timesteps
    assert len(returns) == timesteps
