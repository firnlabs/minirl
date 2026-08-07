"""Implements the REINFORCE policy gradient algorithm."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, *, obs_dim: int, action_dim: int) -> None:
        """Initializes the policy network.

        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
        """
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_dim = 128

        self._fc = nn.Sequential(
            nn.Linear(self._obs_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass and returns action probabilities."""
        return self._fc(obs)
