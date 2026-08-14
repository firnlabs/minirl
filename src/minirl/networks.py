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


class DiscreteActor[
    ObsDim: IntVar,
    NumActions: IntVar,
    HiddenDim: IntVar,
](nn.Module):
    """Actor network for discrete actions."""

    def __init__(
        self,
        *,
        obs_dim: Int[ObsDim],
        num_actions: Int[NumActions],
        hidden_dim: Int[HiddenDim] = 64,
    ):
        super().__init__()
        self._fc1 = _layer_init(nn.Linear(obs_dim, hidden_dim))
        self._tanh1 = nn.Tanh()
        self._fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim))
        self._tanh2 = nn.Tanh()
        self._fc3 = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

    def forward[BatchSize: IntVar](
        self, obs: Tensor[[BatchSize, ObsDim]]
    ) -> Tensor[[BatchSize, NumActions]]:
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
