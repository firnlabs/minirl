"""Tests for reinforce.py."""

import torch

from minirl import reinforce


class TestPolicyNetwork:
    """Tests for PolicyNetwork class."""

    def test_forward(self):
        obs_dim = 4
        action_dim = 2
        policy = reinforce.PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        obs = torch.rand(obs_dim)

        probs = policy.forward(obs)

        assert probs.shape == (2,)
        assert probs.sum().item() == 1.0
