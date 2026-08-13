"""Tests for losses.py."""

import pytest
import torch

import minirl.losses


def test_compute_policy_gradient_loss():
    returns = [1.9, 1]
    with torch.no_grad():
        log_probs = [torch.tensor(0.1), torch.tensor(0.2)]

        loss = minirl.losses.compute_policy_gradient_loss(log_probs, returns)

    assert loss.item() == pytest.approx(-0.39)
