"""Implements the REINFORCE policy gradient algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from torch import distributions
from torch import nn
from torch import optim

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


def collect_trajectory_data(
    env: gym.Env,
    policy: PolicyNetwork,
) -> tuple[list[Tensor], list[float]]:
    log_probs = []
    rewards = []
    done = False

    obs, _ = env.reset()

    while not done:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action_probs = policy(obs_tensor)

        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        obs, reward, terminated, truncated, _ = env.step(action.item())

        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))

        done = terminated or truncated

    return log_probs, rewards


def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []

    # Compute returns from the end of the trajectory going back
    current_return = 0.0
    for r in reversed(rewards):
        current_return = r + current_return * gamma
        returns.append(current_return)

    return returns[::-1]


def compute_loss(log_probs: list[Tensor], returns: list[float]) -> Tensor:
    losses = []

    for log_prob, ret in zip(log_probs, returns, strict=True):
        losses.append(-log_prob * ret)

    loss = torch.stack(losses).sum()
    return loss


def train_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    optimizer: optim.Optimizer,
    gamma: float,
) -> float:
    log_probs, rewards = collect_trajectory_data(env, policy)
    returns = compute_discounted_returns(rewards, gamma)
    loss = compute_loss(log_probs, returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(sum(rewards))
