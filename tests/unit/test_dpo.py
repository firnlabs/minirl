"""Tests for dpo module."""

import copy
from dataclasses import dataclass

import pytest
from shape_extensions import IntVar
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F

from minirl import dpo

torch.manual_seed(42)


@pytest.mark.parametrize("use_prompt_mask", [True, False])
def test_compute_sequence_log_probs(use_prompt_mask: bool):
    B, T, V = 2, 5, 4
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    attention_mask = torch.ones((B, T), dtype=torch.long)
    attention_mask[1, 4] = 0  # Second sequence has padding at the end

    # Prompt is the first 2 tokens (indices 0 and 1)
    if use_prompt_mask:
        prompt_mask = torch.zeros((B, T), dtype=torch.long)
        prompt_mask[:, :2] = 1
    else:
        prompt_mask = None

    output = dpo.compute_sequence_log_probs(
        logits=logits,
        labels=labels,
        attention_mask=attention_mask,
        prompt_mask=prompt_mask,
    )

    assert output.shape == (B,)

    exp_logps_row0 = torch.log_softmax(logits[0, :-1], dim=-1)
    shifted_labels = labels[0, 1:]
    if prompt_mask is None:
        loss_mask = attention_mask[0, 1:] == 1
    else:
        loss_mask = (attention_mask[0, 1:] == 1) * (prompt_mask[0, 1:] == 0)
    expected_row0 = exp_logps_row0[loss_mask, shifted_labels[loss_mask]].sum()

    torch.testing.assert_close(output[0], expected_row0, rtol=1e-5, atol=1e-5)


def test_compute_dpo_loss():
    policy_chosen_logps = torch.tensor([-2.0, -5.0])
    policy_rejected_logps = torch.tensor([-3.0, -4.0])
    ref_chosen_logps = torch.tensor([-2.0, -5.0])
    ref_rejected_logps = torch.tensor([-3.0, -4.0])
    beta = 0.5
    label_smoothing = 0.5

    loss = dpo.compute_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        label_smoothing,
    )

    assert loss.shape == ()
    torch.testing.assert_close(loss, F.logsigmoid(torch.tensor([0.0, 0.0])).mean())


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.5])
def test_compute_preference_log_probs(label_smoothing):
    preference_logits = torch.tensor([-1.0, -2.0])

    preference_log_probs = dpo._compute_preference_log_probs(
        preference_logits, label_smoothing=label_smoothing
    )

    assert preference_log_probs.shape == preference_logits.shape
    if label_smoothing == 0.0:
        expected = -F.logsigmoid(preference_logits)
    elif label_smoothing > 0.0:
        expected = -(1 - label_smoothing) * F.logsigmoid(
            preference_logits
        ) - label_smoothing * F.logsigmoid(-preference_logits)
    else:
        raise ValueError("Invalid label_smoothing value")
    torch.testing.assert_close(preference_log_probs, expected, rtol=1e-5, atol=1e-5)


def test_compute_reward_metrics():
    policy_chosen_logps = torch.tensor([-2.0, -5.0])
    policy_rejected_logps = torch.tensor([-3.0, -4.0])
    ref_chosen_logps = torch.tensor([-3.0, -6.0])
    ref_rejected_logps = torch.tensor([-3.0, -4.0])
    beta = 0.5

    metrics = dpo.compute_reward_metrics(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )

    assert isinstance(metrics, dpo.DPORewardMetrics)
    torch.testing.assert_close(
        metrics.chosen_rewards, torch.tensor([0.5, 0.5]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        metrics.rejected_rewards, torch.tensor([0.0, 0.0]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        metrics.reward_margin, torch.tensor([0.5, 0.5]), rtol=1e-5, atol=1e-5
    )
    assert metrics.reward_accuracy == pytest.approx(1.0, rel=1e-5)


class DummyTokenizer:
    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1

    def __call__(
        self,
        text: str,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        token_ids = [hash(w) % 1000 + 2 for w in text.split()]
        if max_length is not None and max_length > 0:
            token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

    @property
    def pad_token_id(self) -> int:
        return 99


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_collate_batch(tokenizer: DummyTokenizer):
    instances: list[dpo.PreferenceInstance] = [
        {"prompt": "Hello world", "chosen": "Good", "rejected": "Bad answer"},
        {"prompt": "Hi there", "chosen": "Great job", "rejected": "Wrong"},
    ]

    out = dpo.collate_batch(instances, tokenizer)

    batch_size = len(instances)

    # batch dimension: chosen first, then rejected = 2 * batch_size
    assert out["input_ids"].shape[0] == 2 * batch_size
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["prompt_mask"].shape == out["input_ids"].shape

    # verify chosen/rejected ordering and token ID correctness
    pad_token_id = tokenizer.pad_token_id
    for i, inst in enumerate(instances):
        prompt_ids = tokenizer(inst["prompt"])["input_ids"]
        chosen_ids = tokenizer(inst["chosen"])["input_ids"]
        rejected_ids = tokenizer(inst["rejected"])["input_ids"]
        prompt_len = len(prompt_ids)
        chosen_len = len(prompt_ids) + len(chosen_ids)
        rejected_len = len(prompt_ids) + len(rejected_ids)

        # chosen rows are in the first half (with zero-padding at end)
        row = out["input_ids"][i].tolist()
        assert row[:chosen_len] == prompt_ids + chosen_ids
        assert row[chosen_len:] == [pad_token_id] * (len(row) - chosen_len)

        # rejected rows are in the second half
        row = out["input_ids"][batch_size + i].tolist()
        assert row[:rejected_len] == prompt_ids + rejected_ids
        assert row[rejected_len:] == [pad_token_id] * (len(row) - rejected_len)

        # prompt mask: 1 for prompt tokens, 0 for response + padding
        chosen_pmask = out["prompt_mask"][i].tolist()
        assert chosen_pmask[:prompt_len] == [1] * prompt_len
        assert chosen_pmask[prompt_len:] == [0] * (len(chosen_pmask) - prompt_len)

        # attention mask: 1 for real tokens, 0 for padding
        chosen_attn = out["attention_mask"][i].tolist()
        assert chosen_attn[:chosen_len] == [1] * chosen_len
        assert chosen_attn[chosen_len:] == [0] * (len(chosen_attn) - chosen_len)


@pytest.mark.parametrize("max_length", [2, 100])
def test_collate_batch_max_length(max_length: int, tokenizer: DummyTokenizer):
    instances: list[dpo.PreferenceInstance] = [
        {"prompt": "a b c d e", "chosen": "f g h", "rejected": "i j"},
    ]

    out = dpo.collate_batch(instances, tokenizer, max_length=max_length)

    prompt_ids = tokenizer("a b c d e")["input_ids"]
    chosen_ids = tokenizer("f g h")["input_ids"]
    max_seq_len = min(len(prompt_ids) + len(chosen_ids), max_length)

    assert out["input_ids"].shape == (2, max_seq_len)


@pytest.mark.parametrize("max_prompt_length", [None, 2, 100])
def test_collate_batch_max_prompt_length(
    max_prompt_length: int | None, tokenizer: DummyTokenizer
):
    instances: list[dpo.PreferenceInstance] = [
        {"prompt": "a b c d e", "chosen": "f g h", "rejected": "i j"},
    ]

    out = dpo.collate_batch(instances, tokenizer, max_prompt_length=max_prompt_length)

    prompt_ids = tokenizer("a b c d e")["input_ids"]
    chosen_ids = tokenizer("f g h")["input_ids"]
    prompt_length = len(prompt_ids)
    if max_prompt_length is not None:
        prompt_length = min(prompt_length, max_prompt_length)
    max_seq_len = prompt_length + len(chosen_ids)

    assert out["input_ids"].shape == (2, max_seq_len)
    assert out["input_ids"][0][:prompt_length].tolist() == prompt_ids[-prompt_length:]


def test_collate_batch_error_instances_empty(tokenizer):
    instances = []

    with pytest.raises(ValueError, match="`instances` is empty"):
        dpo.collate_batch(instances, tokenizer)


@dataclass(frozen=True)
class DummyModelOutput:
    logits: Tensor


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_dim: int = 16):
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._embedding = nn.Embedding(vocab_size, hidden_dim)
        self._linear = nn.Linear(hidden_dim, vocab_size)

    def forward[B: IntVar, T: IntVar, V: IntVar](
        self,
        input_ids: Tensor[[B, T]],
        attention_mask: Tensor[[B, T]] | None = None,
    ) -> DummyModelOutput:
        _ = attention_mask  # unused
        emb = self._embedding(input_ids)
        logits: Tensor[[B, T, V]] = self._linear(emb)
        return DummyModelOutput(logits=logits)


@pytest.fixture
def model() -> DummyModel:
    return DummyModel()


def test_concatenated_forward(model: DummyModel):
    vocab_size = 8
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, vocab_size)).view(
        batch_size, -1
    )
    prompt_mask = torch.zeros_like(input_ids)
    prompt_mask[:, torch.arange(2)] = 1
    attention_mask = torch.ones_like(input_ids)

    batch = {
        "input_ids": input_ids,
        "prompt_mask": prompt_mask,
        "attention_mask": attention_mask,
    }

    chosen_log_probs, rejected_log_probs = dpo.concatenated_forward(model, batch)

    half_batch = batch_size // 2
    assert chosen_log_probs.shape == (half_batch,)
    assert rejected_log_probs.shape == (half_batch,)


@pytest.mark.parametrize("input_ids", [[], [[1, 2, 3]]])
def test_concatenated_forward_invalid_batch_size(input_ids, model: DummyModel):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    prompt_mask = torch.zeros_like(input_ids)
    if len(prompt_mask.shape) == 2:
        prompt_mask[:, 0] = 1

    batch = {
        "input_ids": input_ids,
        "prompt_mask": prompt_mask,
        "attention_mask": attention_mask,
    }

    with pytest.raises(
        ValueError, match="batch `input_ids` should be a non-zero even number."
    ):
        dpo.concatenated_forward(model, batch)


class TestDPOTrainer:
    """Tests for DPOTrainer class."""

    def test_train_step(self, model):
        policy_model = model
        ref_model = copy.deepcopy(policy_model)
        optimizer = optim.Adam(policy_model.parameters())
        trainer = dpo.DPOTrainer(policy_model, ref_model, optimizer)
        vocab_size = 8
        batch_size = 2
        input_ids = torch.randint(0, vocab_size, (batch_size, vocab_size)).view(
            batch_size, -1
        )
        prompt_mask = torch.zeros_like(input_ids)
        prompt_mask[:, torch.arange(2)] = 1
        attention_mask = torch.ones_like(input_ids)

        batch = {
            "input_ids": input_ids,
            "prompt_mask": prompt_mask,
            "attention_mask": attention_mask,
        }

        metrics = trainer.train_step(batch)

        assert isinstance(metrics, dpo.TrainStepMetrics)
