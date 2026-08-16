"""Unit tests for ppo.py."""

import gymnasium as gym
import pytest

from minirl.ppo import _parse_action_space


@pytest.mark.parametrize(
    (
        "action_space",
        "exp_is_discrete",
        "exp_num_actions",
        "exp_action_dim",
    ),
    [
        (gym.spaces.Discrete(4), True, 4, 1),
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)), False, 3, 3),
    ],
)
def test_parse_action_space(
    action_space,
    exp_is_discrete,
    exp_num_actions,
    exp_action_dim,
):
    is_discrete, num_actions, action_dim = _parse_action_space(action_space)
    assert is_discrete == exp_is_discrete
    assert num_actions == exp_num_actions
    assert action_dim == exp_action_dim


def test_parse_action_space_unsupported():
    with pytest.raises(NotImplementedError):
        _parse_action_space(gym.spaces.MultiDiscrete([2, 2]))
