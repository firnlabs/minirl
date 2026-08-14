"""Neural networks for reinforcement learning."""

import numpy as np
from shape_extensions import Int
from shape_extensions import IntVar
from torch import nn
from torch import Tensor


def _layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """Initializes neural net layer using orthogonal initialization.

    References:
        Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients:
        A Case Study on PPO and TRPO. https://arxiv.org/abs/2005.12729
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork[
    InDim: IntVar,
    OutDim: IntVar,
    HiddenDim: IntVar,
](nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(
        self,
        *,
        obs_dim: Int[InDim],
        action_dim: Int[OutDim],
        hidden_dim: Int[HiddenDim] = 128,
    ) -> None:
        """Initializes the policy network.

        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim

        self._fc1 = nn.Linear(obs_dim, hidden_dim)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(hidden_dim, action_dim)
        self._softmax = nn.Softmax(dim=-1)

    def forward[B: IntVar](self, obs: Tensor[[B, InDim]]) -> Tensor[[B, OutDim]]:
        """Runs the forward pass and returns action logits."""
        x: Tensor[[B, HiddenDim]] = self._relu(self._fc1(obs))
        logits: Tensor[[B, OutDim]] = self._fc2(x)
        return logits


class DiscreteActor[
    ObsDim: IntVar,
    ActionDim: IntVar,
    HiddenDim: IntVar,
](nn.Module):
    """Actor network for discrete actions."""

    def __init__(
        self,
        *,
        obs_dim: Int[ObsDim],
        action_dim: Int[ActionDim],
        hidden_dim: Int[HiddenDim] = 64,
    ):
        super().__init__()
        self._fc1 = _layer_init(nn.Linear(obs_dim, hidden_dim))
        self._tanh1 = nn.Tanh()
        self._fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim))
        self._tanh2 = nn.Tanh()
        self._fc3 = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward[BatchSize: IntVar](
        self, obs: Tensor[[BatchSize, ObsDim]]
    ) -> Tensor[[BatchSize, ActionDim]]:
        h1: Tensor[[BatchSize, HiddenDim]] = self._tanh1(self._fc1(obs))
        h2: Tensor[[BatchSize, HiddenDim]] = self._tanh2(self._fc2(h1))
        return self._fc3(h2)


class Critic[ObsDim: IntVar, HiddenDim: IntVar](nn.Module):
    """Critic network."""

    def __init__(
        self,
        *,
        obs_dim: Int[ObsDim],
        hidden_dim: Int[HiddenDim] = 64,
    ):
        super().__init__()
        self._fc1 = _layer_init(nn.Linear(obs_dim, hidden_dim))
        self._tanh1 = nn.Tanh()
        self._fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim))
        self._tanh2 = nn.Tanh()
        self._fc3 = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward[BatchSize: IntVar](
        self, obs: Tensor[[BatchSize, ObsDim]]
    ) -> Tensor[[BatchSize, 1]]:
        h1: Tensor[[BatchSize, HiddenDim]] = self._tanh1(self._fc1(obs))
        h2: Tensor[[BatchSize, HiddenDim]] = self._tanh2(self._fc2(h1))
        return self._fc3(h2)
