"""Tests for reinforce.py."""

import gymnasium as gym
import pytest
from shape_extensions import Int
import torch

from minirl import reinforce


class TestPolicyNetwork:
    """Tests for PolicyNetwork class."""

    def test_forward(self):
        obs_dim: Int[4] = 4
        action_dim: Int[2] = 2
        policy = reinforce.PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        obs = torch.rand((2, obs_dim))

        probs = policy.forward(obs)

        assert probs.shape == (2, 2)
        assert torch.all(probs.sum(dim=-1) == 1.0)


def test_collect_trajectory_data():
    env = gym.make("CartPole-v1", max_episode_steps=2)

    with torch.no_grad():
        policy = reinforce.PolicyNetwork(obs_dim=4, action_dim=2)

    log_probs, rewards = reinforce.collect_trajectory_data(env, policy)

    assert len(log_probs) >= 1
    assert len(rewards) >= 1


def test_compute_discounted_returns():
    rewards = [1.0, 1.0, 1.0, 1.0]
    gamma = 0.9

    num_rewards = len(rewards)
    expected_returns = [0.0 for _ in range(num_rewards + 1)]
    for i in reversed(range(num_rewards)):
        expected_returns[i] = rewards[i] + expected_returns[i + 1] * gamma

    returns = reinforce.compute_discounted_returns(rewards, gamma)

    assert returns == expected_returns[:-1]


def test_compute_loss():
    returns = [1.9, 1]
    with torch.no_grad():
        log_probs = [torch.tensor(0.1), torch.tensor(0.2)]

        loss = reinforce.compute_loss(log_probs, returns)

    assert loss.item() == pytest.approx(-0.39)
