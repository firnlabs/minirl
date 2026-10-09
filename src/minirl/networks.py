"""Neural networks for reinforcement learning."""

from jaxtyping import Float
import numpy as np
from torch import nn
from torch import Tensor


def _layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0
) -> nn.Linear:
    """Initializes neural net layer using orthogonal initialization.

    References:
        Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients:
        A Case Study on PPO and TRPO. https://arxiv.org/abs/2005.12729
    """
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActor(nn.Module):
    """Actor network for discrete actions."""

    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self._fc1 = _layer_init(nn.Linear(obs_dim, hidden_dim))
        self._tanh1 = nn.Tanh()
        self._fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim))
        self._tanh2 = nn.Tanh()
        self._fc3 = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

    def forward(
        self, obs: Float[Tensor, "batch_size obs_dim"]
    ) -> Float[Tensor, "batch_size num_actions"]:
        """Computes unnormalized action logits for given observations.

        Args:
            obs: Observation tensor.

        Returns:
            Unnormalized action logits.
        """
        h1: Float[Tensor, "batch_size hidden_dim"] = self._tanh1(self._fc1(obs))
        h2: Float[Tensor, "batch_size hidden_dim"] = self._tanh2(self._fc2(h1))
        return self._fc3(h2)


class Critic(nn.Module):
    """Critic network."""

    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self._fc1 = _layer_init(nn.Linear(obs_dim, hidden_dim))
        self._tanh1 = nn.Tanh()
        self._fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim))
        self._tanh2 = nn.Tanh()
        self._fc3 = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(
        self, obs: Float[Tensor, "batch_size obs_dim"]
    ) -> Float[Tensor, "batch_size 1"]:
        """Computes scalar state value estimates for given observations.

        Args:
            obs: Observation tensor.

        Returns:
            Estimated state values.
        """
        h1: Float[Tensor, "batch_size hidden_dim"] = self._tanh1(self._fc1(obs))
        h2: Float[Tensor, "batch_size hidden_dim"] = self._tanh2(self._fc2(h1))
        return self._fc3(h2)
