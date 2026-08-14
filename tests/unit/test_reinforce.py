"""Tests for reinforce.py."""

import pytest
import torch
from torch import optim

from minirl import networks
from minirl import reinforce


@pytest.fixture
def gamma() -> float:
    return 0.9


@pytest.fixture
def optimizer(policy: networks.DiscreteActor) -> optim.Optimizer:
    return optim.Adam(policy.parameters(), lr=0.01)


def test_collect_trajectory_data(env):
    with torch.no_grad():
        policy = networks.DiscreteActor(obs_dim=4, action_dim=2)

    log_probs, rewards = reinforce.collect_trajectory_data(env, policy)

    assert len(log_probs) >= 1
    assert len(rewards) >= 1


def test_train_episode(env, policy, optimizer, gamma):
    total_reward = reinforce.train_episode(env, policy, optimizer, gamma)

    assert total_reward > 0
