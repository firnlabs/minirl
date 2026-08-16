"""Tests for losses.py."""

import pytest
import torch

from minirl import losses


def test_compute_policy_gradient_loss():
    returns = [1.9, 1]
    with torch.no_grad():
        log_probs = [torch.tensor(0.1), torch.tensor(0.2)]

        loss = losses.compute_policy_gradient_loss(log_probs, returns)

    assert loss.item() == pytest.approx(-0.39)


def test_compute_ppo_loss():
    old_log_probs = torch.tensor([0.0, 0.0])
    new_log_probs = torch.tensor([0.1, -0.4])
    advantages = torch.tensor([2.0, -1.0])
    clip_eps = 0.2
    entropy_coef = 0.01
    entropy = torch.tensor(0.5)

    loss, mean_entropy, clip_fraction = losses.compute_ppo_loss(
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        advantages=advantages,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        entropy=entropy,
    )

    assert loss.item() == pytest.approx(-0.7102, abs=1e-4)
    assert mean_entropy.item() == pytest.approx(0.5)
    assert clip_fraction.item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "old_values_list, clip_eps",
    [
        (None, None),
        ([0.0, 1.0], None),
        (None, 0.1),
        ([0.0, 1.0], 0.1),
    ],
)
def test_compute_ppo_value_loss(old_values_list, clip_eps):
    values = torch.tensor([1.0, 2.0])
    target_values = torch.tensor([2.0, 3.0])
    old_values = torch.tensor(old_values_list) if old_values_list is not None else None

    loss = losses.compute_ppo_value_loss(
        values=values,
        target_values=target_values,
        old_values=old_values,
        clip_eps=clip_eps,
    )

    if old_values_list is None or clip_eps is None:
        exp_loss = pytest.approx(0.5)
    else:
        exp_loss = pytest.approx(1.8050)
    assert loss.item() == exp_loss
