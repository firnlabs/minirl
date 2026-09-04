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

    exp_logps_row0 = torch.log_softmax(logits[0], dim=-1)  # (T, V)
    expected_row0 = (
        # First two tokens (indices 0 and 1) are prompts so we skip them using
        # the prompt_mask
        # Note that the mask is shifted by 1 to the right
        exp_logps_row0[1, labels[0, 2]]
        + exp_logps_row0[2, labels[0, 3]]
        + exp_logps_row0[3, labels[0, 4]]
    )
    if prompt_mask is None:
        # Include the first token if no prompt mask
        expected_row0 += exp_logps_row0[0, labels[0, 1]]

    torch.testing.assert_close(output[0], expected_row0, rtol=1e-5, atol=1e-5)
