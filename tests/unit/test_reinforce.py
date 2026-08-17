"""Tests for reinforce.py."""

import pytest
import torch
from torch import optim

from minirl import networks
from minirl import reinforce
import minirl.reinforce


@pytest.fixture
def gamma() -> float:
    return 0.9


@pytest.fixture
def optimizer(policy: networks.DiscreteActor) -> optim.Optimizer:
    return optim.Adam(policy.parameters(), lr=0.01)


def test_collect_trajectory_data(env):
    with torch.no_grad():
        policy = networks.DiscreteActor(obs_dim=4, num_actions=2)

    log_probs, rewards = reinforce.collect_trajectory_data(env, policy)

    assert len(log_probs) >= 1
    assert len(rewards) >= 1


def test_train_episode(env, policy, optimizer, gamma):
    total_reward = reinforce.train_episode(env, policy, optimizer, gamma)

    assert total_reward > 0


def test_compute_policy_gradient_loss():
    returns = [1.9, 1]
    with torch.no_grad():
        log_probs = [torch.tensor(0.1), torch.tensor(0.2)]

        loss = minirl.reinforce.compute_policy_gradient_loss(log_probs, returns)

    assert loss.item() == pytest.approx(-0.39)
