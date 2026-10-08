import torch

from minirl import networks


class TestDiscreteActor:
    def test_forward(self):
        obs_dim = 4
        action_dim = 2
        hidden_dim = 32
        batch_size = 4

        actor = networks.DiscreteActor(
            obs_dim=obs_dim, num_actions=action_dim, hidden_dim=hidden_dim
        )

        obs = torch.rand((batch_size, obs_dim))
        logits = actor(obs)

        assert logits.shape == (batch_size, action_dim)


class TestCritic:
    def test_forward(self):
        obs_dim = 4
        hidden_dim = 32
        batch_size = 4

        critic = networks.Critic(obs_dim=obs_dim, hidden_dim=hidden_dim)

        obs = torch.rand((batch_size, obs_dim))
        value = critic(obs)

        assert value.shape == (batch_size, 1)
