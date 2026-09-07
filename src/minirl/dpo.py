"""Direct Preference Optimization (DPO) implementation.

References:
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
"""

from dataclasses import dataclass

from shape_extensions import IntVar
import torch
from torch import Tensor
from torch.nn import functional as F


def compute_sequence_log_probs[B: IntVar, T: IntVar, V: IntVar](
    logits: Tensor[[B, T, V]],
    labels: Tensor[[B, T]],
    attention_mask: Tensor[[B, T]],
    prompt_mask: Tensor[[B, T]] | None = None,
) -> Tensor[[B]]:
    """Compute the log probabilities of sequences given logits and labels.

    Args:
        logits: Logits from the model.
        labels: Ground truth labels.
        attention_mask: Mask indicating valid tokens (1: valid, 0: not valid).
        prompt_mask: Optional mask indicating prompt tokens (1: prompt, 0: not prompt).

    Returns:
        Log probabilities of the sequences.
    """
    shift_logits: Tensor[[B, T - 1, V]] = logits[..., :-1, :]
    shift_labels: Tensor[[B, T - 1]] = labels[..., 1:]

    # Safely handle potential -100 ignore_index in labels
    clamped_labels = torch.clamp(shift_labels, min=0)

    log_probs: Tensor[[B, T - 1, V]] = F.log_softmax(shift_logits, dim=-1)
    select_log_probs: Tensor[[B, T - 1]] = torch.gather(
        log_probs, dim=-1, index=clamped_labels[..., None]
    ).squeeze(-1)

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(attention_mask)

    mask: Tensor[[B, T - 1]] = (attention_mask[..., 1:] == 1) * (
        prompt_mask[..., 1:] == 0
    )

    masked_log_probs: Tensor[[B, T - 1]] = select_log_probs * mask.to(
        select_log_probs.dtype
    )

    return masked_log_probs.sum(dim=-1)


def _compute_preference_log_probs[B: IntVar](
    preference_logits: Tensor[[B]], label_smoothing: float = 0.0
) -> Tensor[[B]]:
    """Compute the log probabilities of preferences given preference logits.

    Args:
        preference_logits: Logits representing the preference between two options.
        label_smoothing: Optional label smoothing factor.

    Returns:
        Log probabilities of the preferences.
    """
    assert label_smoothing >= 0.0
    assert label_smoothing <= 0.5

    chosen_log_probs = -(1 - label_smoothing) * F.logsigmoid(preference_logits)
    rejected_log_probs = label_smoothing * F.logsigmoid(-preference_logits)
    return chosen_log_probs - rejected_log_probs


def compute_dpo_loss[B: IntVar](
    policy_chosen_logps: Tensor[[B]],
    policy_rejected_logps: Tensor[[B]],
    ref_chosen_logps: Tensor[[B]],
    ref_rejected_logps: Tensor[[B]],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tensor[[]]:
    """Computes DPO loss and metrics.

    Args:
        policy_chosen_logps: Log probabilities of the chosen sequences from the
            policy model.
        policy_rejected_logps: Log probabilities of the rejected sequences from the
            policy model.
        ref_chosen_logps: Log probabilities of the chosen sequences from the reference
            model.
        ref_rejected_logps: Log probabilities of the rejected sequences from the
            reference model.
        beta: Scaling factor for the DPO loss.
        label_smoothing: Optional label smoothing factor.

    Returns:
        tuple containing loss, chosen_rewards, and rejected_rewards.
    """
    policy_log_ratio: Tensor[[B]] = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio: Tensor[[B]] = ref_chosen_logps - ref_rejected_logps
    preference_logits: Tensor[[B]] = beta * (policy_log_ratio - ref_log_ratio)
    preference_log_probs: Tensor[[B]] = _compute_preference_log_probs(
        preference_logits, label_smoothing=label_smoothing
    )

    return -preference_log_probs.mean()


@dataclass(frozen=True, slots=True)
class DPORewardMetrics[B: IntVar]:
    """Dataclass to hold DPO reward metrics."""

    chosen_rewards: Tensor[[B]]
    rejected_rewards: Tensor[[B]]
    reward_margin: Tensor[[B]]
    reward_accuracy: float


def compute_reward_metrics[B: IntVar](
    policy_chosen_logps: Tensor[[B]],
    policy_rejected_logps: Tensor[[B]],
    ref_chosen_logps: Tensor[[B]],
    ref_rejected_logps: Tensor[[B]],
    beta: float = 0.1,
) -> DPORewardMetrics:
    """Computes DPO reward metrics.

    Args:
        policy_chosen_logps: Log probabilities of the chosen sequences from the
            policy model.
        policy_rejected_logps: Log probabilities of the rejected sequences from the
            policy model.
        ref_chosen_logps: Log probabilities of the chosen sequences from the reference
            model.
        ref_rejected_logps: Log probabilities of the rejected sequences from the
            reference model.
        beta: Scaling factor for the DPO loss.

    Returns:
        DPORewardMetrics: reward metrics.
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = chosen_rewards - rejected_rewards
    reward_accuracy = (reward_margin > 0).float().mean().item()
    return DPORewardMetrics(
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margin=reward_margin,
        reward_accuracy=reward_accuracy,
    )
