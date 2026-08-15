"""Loss functions for reinforcement learning."""

from shape_extensions import IntVar
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


def compute_ppo_loss[BatchSize: IntVar](
    *,
    old_log_probs: Tensor[[BatchSize]],
    new_log_probs: Tensor[[BatchSize]],
    advantages: Tensor[[BatchSize]],
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    entropy: Tensor[[BatchSize]] | None = None,
) -> tuple[Tensor[[]], Tensor[[]], Tensor[[]]]:
    prob_ratio = torch.exp(new_log_probs - old_log_probs)

    surr_obj = prob_ratio * advantages
    surr_obj_clipped = (
        torch.clamp(prob_ratio, min=1 - clip_eps, max=1 + clip_eps) * advantages
    )

    policy_surr_loss = torch.mean(torch.minimum(surr_obj, surr_obj_clipped))

    if entropy is not None:
        mean_entropy = torch.mean(entropy)
    else:
        mean_entropy = torch.tensor(0.0, device=old_log_probs.device)

    # Add minus sign since PyTorch minimizes loss by default
    policy_loss = -(policy_surr_loss + entropy_coef * mean_entropy)

    # diagnostics
    with torch.no_grad():
        clip_fraction = (torch.abs(prob_ratio - 1.0) > clip_eps).float().mean()

    return policy_loss, mean_entropy, clip_fraction
