"""Tests for buffers.py."""

import pytest
import torch

from minirl import buffers


@pytest.fixture
def rollout_buffer(obs_dim: int) -> buffers.RolloutBuffer:
    return buffers.RolloutBuffer(capacity=2, obs_dim=obs_dim)


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
