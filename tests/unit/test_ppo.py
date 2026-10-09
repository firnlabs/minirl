import gymnasium as gym
import pytest
import torch

import minirl.ppo
from minirl.ppo import _parse_action_space


@pytest.mark.parametrize(
    (
        "action_space",
        "exp_is_discrete",
        "exp_num_actions",
        "exp_action_dim",
    ),
    [
        (gym.spaces.Discrete(4), True, 4, 1),
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)), False, 3, 3),
    ],
)
def test_parse_action_space(
    action_space,
    exp_is_discrete,
    exp_num_actions,
    exp_action_dim,
):
    is_discrete, num_actions, action_dim = _parse_action_space(action_space)
    assert is_discrete == exp_is_discrete
    assert num_actions == exp_num_actions
    assert action_dim == exp_action_dim


def test_parse_action_space_unsupported():
    with pytest.raises(NotImplementedError):
        _parse_action_space(gym.spaces.MultiDiscrete([2, 2]))


def test_compute_ppo_loss():
    old_log_probs = torch.tensor([0.0, 0.0])
    new_log_probs = torch.tensor([0.1, -0.4])
    advantages = torch.tensor([2.0, -1.0])
    clip_eps = 0.2
    entropy_coef = 0.01
    entropy = torch.tensor([0.5, 0.5])

    loss, mean_entropy, clip_fraction = minirl.ppo.compute_ppo_loss(
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

    loss = minirl.ppo.compute_ppo_value_loss(
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


@pytest.mark.parametrize("normalize_advantages", [True, False])
def test_ppo_trainer_advantage_normalization(normalize_advantages):
    trainer = minirl.ppo.PPOTrainer(
        env_name="CartPole-v1",
        num_steps=64,
        num_epochs=1,
        batch_size=32,
        normalize_advantages=normalize_advantages,
    )
    # Collect transitions
    trainer.collect_rollouts()
    # Compute advantages & returns
    trainer._buffer.compute_advantages_and_returns(
        last_value=torch.tensor(0.0),
        gamma=0.99,
        lam=0.95,
        last_done=True,
    )
    metrics = trainer.train_epoch(trainer._buffer)
    assert "approx_kl" in metrics
    assert "early_stopped" in metrics
    assert not torch.isnan(torch.tensor(metrics["approx_kl"]))


@pytest.mark.parametrize(
    "advantages",
    [
        [1.0, 5.0, 10.0, 20.0],
        [1.0, 0.0],
    ],
)
def test_normalize_advantages(advantages):
    raw_adv = torch.tensor(advantages, dtype=torch.float32)
    norm_adv = minirl.ppo.normalize_advantages(raw_adv)
    torch.testing.assert_close(norm_adv.mean(), torch.tensor(0.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(norm_adv.std(), torch.tensor(1.0), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("advantages", [[5.0], []])
def test_normalize_advantages_error_invalid_batch_size(advantages):
    raw_adv = torch.tensor(advantages, dtype=torch.float32)
    with pytest.raises(
        ValueError, match="Cannot normalize advantages with batch size <= 1"
    ):
        minirl.ppo.normalize_advantages(raw_adv)
