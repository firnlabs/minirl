"""Tests for reinforce.py."""

import gymnasium as gym
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
    policy = reinforce.PolicyNetwork(obs_dim=4, action_dim=2)

    log_probs, rewards = reinforce.collect_trajectory_data(env, policy)

    assert len(log_probs) >= 1
    assert len(rewards) >= 1
