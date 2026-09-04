"""Tests for dpo module."""

import pytest
import torch

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

    loss = dpo.compute_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )

    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.6931, rel=1e-4)
