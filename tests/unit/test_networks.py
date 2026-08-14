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


class TestDiscreteActor:
    """Tests for DiscreteActor class."""

    def test_forward(self):
        obs_dim = 4
        action_dim = 2
        hidden_dim = 32
        batch_size = 4

        actor = networks.DiscreteActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )

        obs = torch.rand((batch_size, obs_dim))
        logits = actor(obs)

        assert logits.shape == (batch_size, action_dim)


class TestCritic:
    """Tests for Critic class."""

    def test_forward(self):
        obs_dim = 4
        hidden_dim = 32
        batch_size = 4

        critic = networks.Critic(obs_dim=obs_dim, hidden_dim=hidden_dim)

        obs = torch.rand((batch_size, obs_dim))
        value = critic(obs)

        assert value.shape == (batch_size, 1)
