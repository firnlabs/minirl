"""Integration test for PPO training."""

import pytest

from minirl import ppo


@pytest.fixture
def ppo_trainer() -> ppo.PPOTrainer:
    return ppo.PPOTrainer(
        hidden_dim=16,
        num_steps=16,
        num_epochs=2,
        batch_size=8,
    )


class TestPPOTrainer:
    """Integration test for PPOTrainer."""

    def test_train(self, ppo_trainer):
        total_timesteps = 32

        mean_returns = ppo_trainer.train(total_timesteps)

        exp_iterations = total_timesteps // ppo_trainer._num_steps
        assert len(mean_returns) == exp_iterations

    def test_train_error(self, ppo_trainer):
        total_timesteps = 1

        with pytest.raises(ValueError, match="total_timesteps"):
            ppo_trainer.train(total_timesteps)
