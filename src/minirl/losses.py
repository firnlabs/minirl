"""Loss functions for reinforcement learning."""

import torch
from torch import Tensor


def compute_policy_gradient_loss(
    log_probs: list[Tensor], returns: list[float]
) -> Tensor:
    losses = []

    for log_prob, ret in zip(log_probs, returns, strict=True):
        losses.append(-log_prob * ret)

    loss = torch.stack(losses).sum()
    return loss
