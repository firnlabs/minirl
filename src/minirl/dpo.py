"""Direct Preference Optimization (DPO) implementation.

References:
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
"""

from dataclasses import dataclass
from typing import Protocol
from typing import TypedDict

from shape_extensions import IntVar
import torch
from torch import nn
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


class Tokenizer(Protocol):
    """Protocol for a tokenizer."""

    def __call__(
        self, text: str, max_length: int | None = None
    ) -> dict[str, list[int]]: ...


class PreferenceInstance(TypedDict):
    prompt: str
    chosen: str
    rejected: str


def _tokenize_pair(
    prompt: str,
    response: str,
    tokenizer: Tokenizer,
    max_length: int = 512,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize a prompt-response pair.

    Returns:
        Tuple of (token_ids, attention_mask, prompt_mask) for the
        concatenated prompt + response sequence.
    """
    p = tokenizer(prompt)["input_ids"]

    # ensure at least one prompt token
    max_prompt_length = min(max_prompt_length or max_length // 2, max_length - 1)
    if len(p) > max_prompt_length:
        p = p[-max_prompt_length:]

    r = tokenizer(response)["input_ids"]
    max_response_length = max_length - len(p)
    if len(r) > max_response_length:
        r = r[:max_response_length]

    # check that the response IDs are not empty
    if not r:
        raise ValueError("Response is empty after truncation.")

    token_ids = p + r
    attention_mask = [1] * len(token_ids)
    prompt_mask = [1] * len(p) + [0] * len(r)

    return token_ids, attention_mask, prompt_mask


def collate_batch(
    instances: list[PreferenceInstance],
    tokenizer: Tokenizer,
    max_length: int = 512,
    max_prompt_length: int | None = None,
) -> dict[str, Tensor]:
    """Collates a batch of data into a single dictionary of tensors.

    The output batch is ordered as:
    [chosen_1, ..., chosen_B, rejected_1, ..., rejected_B]
    so that the first half contains all chosen sequences and the second half
    contains all rejected sequences.

    Args:
        instances: A list of dictionaries, each containing a
            prompt, chosen response and rejected response.
        tokenizer: A tokenizer that converts text into tokens.
        max_length: The maximum length of the tokenized sequences.
        max_prompt_length: The maximum length of the tokenized prompt sequence.

    Returns:
        Dictionary of tensors containing input_ids, attention_mask and prompt_mask.

    Raises:
        ValueError: when `instances` list is empty.
    """
    if not instances:
        raise ValueError("`instances` is empty.")

    chosen: list[tuple[list[int], list[int], list[int]]] = []
    rejected: list[tuple[list[int], list[int], list[int]]] = []

    for inst in instances:
        chosen.append(
            _tokenize_pair(
                inst["prompt"],
                inst["chosen"],
                tokenizer,
                max_length,
                max_prompt_length,
            )
        )
        rejected.append(
            _tokenize_pair(
                inst["prompt"],
                inst["rejected"],
                tokenizer,
                max_length,
                max_prompt_length,
            )
        )

    # order: all chosen first, then all rejected
    ordered = chosen + rejected

    # pad to the longest sequence in the batch
    max_seq_len = max(len(ids) for ids, _, _ in ordered)

    # get pad token ID
    # add "or 0" guard in case `tokenizer.pad_token_id` exists but is `None`
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def pad(seq: list[int], value: int = 0) -> list[int]:
        return seq + [value] * (max_seq_len - len(seq))

    return {
        "input_ids": torch.tensor(
            [pad(ids, pad_token_id) for ids, _, _ in ordered], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [pad(attn) for _, attn, _ in ordered], dtype=torch.long
        ),
        "prompt_mask": torch.tensor(
            [pad(pmask) for _, _, pmask in ordered], dtype=torch.long
        ),
    }


def concatenated_forward[B: IntVar](
    model: nn.Module,
    batch: dict[str, Tensor],
) -> tuple[Tensor[[B]], Tensor[[B]]]:
    """Generates chosen and rejected log probabilities from a preference batch.

    Args:
        model: Causal LM model generating output.logits on __call__.
        batch: Dict of collated batch sequences (see `collate_batch`).

    Returns:
        tuple of chosen_log_probs, rejected_log_probs.

    Raises:
        ValueError: if batch[`input_ids`] size is not a non-zero even number.
    """
    input_ids = batch["input_ids"]
    batch_size = input_ids.shape[0]
    if batch_size == 0 or batch_size % 2 != 0:
        raise ValueError("batch `input_ids` should be a non-zero even number.")

    attention_mask = batch["attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    log_probs = compute_sequence_log_probs(
        logits, input_ids, attention_mask, batch.get("prompt_mask")
    )

    half_batch = batch_size // 2
    chosen_log_probs = log_probs[:half_batch]
    rejected_log_probs = log_probs[half_batch:]

    return chosen_log_probs, rejected_log_probs


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainStepMetrics:
    """Output metrics of DPOTrainer."""

    loss: float
    mean_chosen_reward: float
    mean_rejected_reward: float
    margin: float
    accuracy: float


class DPOTrainer:
    """DPO trainer class."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        max_grad_norm: float | None = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        self._policy_model = policy_model
        self._ref_model = ref_model
        self._optimizer = optimizer
        self._beta = beta
        self._max_grad_norm = max_grad_norm
        self._label_smoothing = label_smoothing

        # ensure reference model parameters are frozen
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False

    def train_step(self, batch: dict[str, Tensor]) -> TrainStepMetrics:
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = concatenated_forward(
                self._ref_model, batch
            )

        policy_chosen_logps, policy_rejected_logps = concatenated_forward(
            self._policy_model, batch
        )

        policy_loss = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self._beta,
            self._label_smoothing,
        )

        self._optimizer.zero_grad()
        policy_loss.backward()

        if self._max_grad_norm is not None and self._max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._policy_model.parameters(),
                max_norm=self._max_grad_norm,
            )

        self._optimizer.step()

        reward_metrics = compute_reward_metrics(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self._beta,
        )

        return TrainStepMetrics(
            loss=policy_loss.item(),
            mean_chosen_reward=reward_metrics.chosen_rewards.mean().item(),
            mean_rejected_reward=reward_metrics.rejected_rewards.mean().item(),
            margin=reward_metrics.reward_margin.mean().item(),
            accuracy=reward_metrics.reward_accuracy,
        )
