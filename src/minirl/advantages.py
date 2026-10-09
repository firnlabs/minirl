"""Routines for computing advantages."""

from shape_extensions import IntVar
import torch
from torch import Tensor


def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Computes discounted cumulative returns backwards along a trajectory.

    Args:
        rewards: Immediate rewards collected along trajectory.
        gamma: Reward discount factor.

    Returns:
        List of discounted cumulative return per timestep.
    """
    returns = []

    current_return = 0.0
    for r in reversed(rewards):
        current_return = r + current_return * gamma
        returns.append(current_return)

    return returns[::-1]


def compute_gae[Timesteps: IntVar](
    *,
    rewards: Tensor[[Timesteps]],
    values: Tensor[[Timesteps]],
    last_value: Tensor[[]],
    dones: Tensor[[Timesteps]],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor[[Timesteps]], Tensor[[Timesteps]]]:
    """Computes generalized advantage estimation (GAE).

    Args:
        rewards: Rewards at each timestep.
        values: Critic values at each timestep.
        last_value: Value after the last timestep (i.e. t + 1 = T).
        dones: Whether the episode completed at each timestep.
        gamma: Discount factor.
        lam: GAE decay factor to control bias-variance trade-off (lambda).

    Returns:
        tuple of:
        - advantage at every timestep.
        - return at every timestep (for critic training).

    Raises:
        RuntimeError: If the devices of the input tensors do not match.
    """
    if not (rewards.device == values.device == last_value.device == dones.device):
        raise RuntimeError(
            f"Device mismatch: rewards={rewards.device}, values={values.device},"
            f" last_value={last_value.device}, dones={dones.device}"
        )

    timesteps = rewards.shape[-1]
    advantages = torch.zeros_like(rewards)
    last_gae = 0  # recursive estimate
    dones_ = dones.to(torch.int32)

    for t in reversed(range(timesteps)):
        if t == timesteps - 1:
            next_value = last_value
        else:
            next_value = values[..., t + 1]
        delta = (
            rewards[..., t]
            + gamma * next_value * (1.0 - dones_[..., t])
            - values[..., t]
        )

        last_gae = delta + gamma * lam * (1.0 - dones_[..., t]) * last_gae
        advantages[..., t] = last_gae

    returns = advantages + values

    return advantages, returns
