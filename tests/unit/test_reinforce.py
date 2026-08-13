"""Tests for reinforce.py."""

import freezegun
import gymnasium as gym
import pytest
import torch
from torch import optim

from minirl import reinforce
import minirl.networks


@pytest.fixture
def max_episode_steps() -> int:
    return 2


@pytest.fixture
def gamma() -> float:
    return 0.9


@pytest.fixture
def env(max_episode_steps: int) -> gym.Env:
    return gym.make("CartPole-v1", max_episode_steps=max_episode_steps)


@pytest.fixture
def obs_dim(env: gym.Env) -> int:
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape is not None
    return env.observation_space.shape[0]


@pytest.fixture
def action_dim(env: gym.Env) -> int:
    assert isinstance(env.action_space, gym.spaces.Discrete)
    return int(env.action_space.n)


@pytest.fixture
def policy(obs_dim: int, action_dim: int) -> minirl.networks.PolicyNetwork:
    return minirl.networks.PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)


@pytest.fixture
def optimizer(policy: minirl.networks.PolicyNetwork) -> optim.Optimizer:
    return optim.Adam(policy.parameters(), lr=0.01)


def test_collect_trajectory_data(env):
    with torch.no_grad():
        policy = minirl.networks.PolicyNetwork(obs_dim=4, action_dim=2)

    log_probs, rewards = reinforce.collect_trajectory_data(env, policy)

    assert len(log_probs) >= 1
    assert len(rewards) >= 1


def test_compute_discounted_returns(gamma):
    rewards = [1.0, 1.0, 1.0, 1.0]
    num_rewards = len(rewards)
    expected_returns = [0.0 for _ in range(num_rewards + 1)]
    for i in reversed(range(num_rewards)):
        expected_returns[i] = rewards[i] + expected_returns[i + 1] * gamma

    returns = reinforce.compute_discounted_returns(rewards, gamma)

    assert returns == expected_returns[:-1]


def test_compute_loss():
    returns = [1.9, 1]
    with torch.no_grad():
        log_probs = [torch.tensor(0.1), torch.tensor(0.2)]

        loss = reinforce.compute_loss(log_probs, returns)

    assert loss.item() == pytest.approx(-0.39)


def test_train_episode(env, policy, optimizer, gamma):
    total_reward = reinforce.train_episode(env, policy, optimizer, gamma)

    assert total_reward > 0


@freezegun.freeze_time("2026-01-01 00:00:00")
def test_make_run_dir(tmp_path):
    run_dir = reinforce._make_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.stem == "reinforce-run-20260101-000000"


def test_save_reward_plot(tmp_path):
    rewards = [10.0, 20.0, 30.0]
    plot_file = tmp_path / "rewards.png"
    reinforce._save_reward_plot(rewards, plot_file)
    assert plot_file.exists()


def test_record_simulation_video(tmp_path, policy):
    video_dir = tmp_path / "video"
    reinforce._record_simulation_video("CartPole-v1", policy, video_dir, max_steps=5)
    assert video_dir.exists()
    assert any(video_dir.glob("*.mp4"))
