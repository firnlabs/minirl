"""Tests for buffers.py."""

import pytest
import torch

from minirl import buffers


@pytest.fixture
def capacity() -> int:
    return 4


@pytest.fixture
def rollout_buffer(obs_dim: int, capacity: int) -> buffers.RolloutBuffer:
    return buffers.RolloutBuffer(capacity=capacity, obs_dim=obs_dim)


@pytest.fixture
def step(obs_dim) -> buffers.Step:
    eps = 1e-4
    return buffers.Step(
        obs=torch.rand(obs_dim) + eps,
        action=torch.randint(low=1, high=2, size=(1,)),  # non-zero discrete action
        reward=1.0,
        value=torch.rand(1) + eps,
        log_prob=torch.rand(1) + eps,
        done=True,
    )


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_add(self, rollout_buffer, step):
        buffer = rollout_buffer

        buffer.add(step)

        assert buffer._ptr == 1
        assert torch.all(buffer.obs[0] != 0)
        assert buffer.actions[0].item() > 0
        assert buffer.rewards[0].item() > 0
        assert buffer.values[0].item() > 0
        assert buffer.log_probs[0].item() > 0
        assert buffer.dones[0].item() is True

    def test_get_batches(self, rollout_buffer, capacity, obs_dim):
        buffer = rollout_buffer
        batch_size = 2
        num_batches = capacity / batch_size
        # simulate post compute_advantages_and_returns() behavior
        buffer._advantages = torch.rand(capacity)
        buffer._returns = torch.rand(capacity)

        batches = list(buffer.get_batches(batch_size=batch_size))

        assert len(batches) == num_batches
        batch = batches[0]
        assert batch.obs.shape == (batch_size, obs_dim)
        assert batch.actions.shape == (batch_size,)
        assert batch.rewards.shape == (batch_size,)
        assert batch.values.shape == (batch_size,)
        assert batch.old_log_probs.shape == (batch_size,)
        assert batch.dones.shape == (batch_size,)
        assert batch.advantages.shape == (batch_size,)
        assert batch.returns.shape == (batch_size,)

    def test_get_batches_error(self, rollout_buffer):
        buffer = rollout_buffer
        batch_size = 2
        assert buffer.advantages is None
        assert buffer.returns is None

        with pytest.raises(RuntimeError, match="Advantages and returns"):
            next(buffer.get_batches(batch_size=batch_size))

    def test_is_full(self, rollout_buffer, capacity, step):
        buffer = rollout_buffer
        assert not buffer.is_full()
        for _ in range(capacity):
            buffer.add(step)

        assert buffer.is_full()

    def test_reset(self, rollout_buffer, capacity, step):
        buffer = rollout_buffer
        for _ in range(capacity):
            buffer.add(step)

        assert buffer.is_full()

        buffer.reset()

        assert buffer._ptr == 0
        assert not buffer.is_full()

    @pytest.mark.parametrize("last_done", [True, False])
    def test_compute_advantages_and_returns(
        self,
        last_done,
        rollout_buffer,
        step,
        capacity,
        mocker,
    ):
        advantages = torch.rand(capacity)
        returns = torch.rand(capacity)
        gamma = 0.99
        lam = 0.95
        mock_compute_gae = mocker.patch(
            "minirl.advantages.compute_gae", return_value=(advantages, returns)
        )
        last_value = torch.tensor(1.0)
        buffer = rollout_buffer
        for _ in range(capacity):
            buffer.add(step)

        buffer.compute_advantages_and_returns(
            last_value=last_value, last_done=last_done
        )

        mock_compute_gae.assert_called_once_with(
            rewards=buffer.rewards,
            values=buffer.values,
            last_value=torch.tensor(0.0) if last_done else last_value,
            dones=buffer.dones,
            gamma=gamma,
            lam=lam,
        )
        assert buffer.advantages is not None
        assert buffer.returns is not None
