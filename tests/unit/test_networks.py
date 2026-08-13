"""Tests for networks.py."""

import torch

from minirl import networks


class TestPolicyNetwork:
    """Tests for PolicyNetwork class."""

    def test_forward(self):
        obs_dim = 4
        action_dim = 2
        hidden_dim = 64
        batch_size = 2
        policy_net = networks.PolicyNetwork(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        obs = torch.rand((batch_size, obs_dim))

        logits = policy_net.forward(obs)

        assert logits.shape == (batch_size, action_dim)
