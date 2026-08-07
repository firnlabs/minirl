"""Implements the REINFORCE policy gradient algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from shape_extensions import Int
    from shape_extensions import IntVar
    from torch import Tensor


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
        """Runs the forward pass and returns action probabilities."""
        h: Tensor[[B, HiddenDim]] = self._relu(self._fc1(obs))
        probs: Tensor[[B, OutDim]] = self._softmax(self._fc2(h))
        return probs
