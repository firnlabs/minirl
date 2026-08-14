"""Common unit test fixtures."""

import gymnasium as gym
import pytest

from minirl import networks


@pytest.fixture
def env() -> gym.Env:
    return gym.make("CartPole-v1", max_episode_steps=2)


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
def policy(obs_dim: int, action_dim: int) -> networks.DiscreteActor:
    return networks.DiscreteActor(obs_dim=obs_dim, action_dim=action_dim)
