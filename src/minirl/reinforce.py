"""Implements the REINFORCE policy gradient algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from torch import distributions
from torch import optim
from tqdm import tqdm
import typer

from minirl import advantages
from minirl import evaluation
from minirl import experiments
from minirl import losses
from minirl import networks

if TYPE_CHECKING:
    from torch import Tensor


_ENV_ID = "CartPole-v1"


def collect_trajectory_data(
    env: gym.Env,
    policy: networks.DiscreteActor,
) -> tuple[list[Tensor], list[float]]:
    log_probs = []
    rewards = []
    done = False

    obs, _ = env.reset()

    while not done:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action_logits = policy(obs_tensor)

        dist = distributions.Categorical(logits=action_logits)
        action = dist.sample()

        obs, reward, terminated, truncated, _ = env.step(action.item())

        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))

        done = terminated or truncated

    return log_probs, rewards


def train_episode(
    env: gym.Env,
    policy: networks.DiscreteActor,
    optimizer: optim.Optimizer,
    gamma: float,
) -> float:
    log_probs, rewards = collect_trajectory_data(env, policy)
    returns = advantages.compute_discounted_returns(rewards, gamma)
    loss = losses.compute_policy_gradient_loss(log_probs, returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(sum(rewards))


def train(
    learning_rate: float = typer.Option(1e-3, help="Learning rate."),
    num_episodes: int = typer.Option(
        100, min=1, help="Number of episodes to train on."
    ),
    gamma: float = typer.Option(0.99, min=0, max=1, help="Reward discount factor."),
    max_episode_steps: int = typer.Option(
        100, min=1, help="Maximum number of steps per episode."
    ),
    output_dir: Path = typer.Option(
        Path("runs/latest"),
        help=(
            "Directory path to save run outputs."
            " Files will be stored in a timestamped subdirectory."
        ),
    ),
) -> None:
    env = gym.make(_ENV_ID, max_episode_steps=max_episode_steps)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]

    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = int(env.action_space.n)

    policy = networks.DiscreteActor(obs_dim=obs_dim, action_dim=action_dim)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_rewards = []
    pbar = tqdm(range(1, num_episodes + 1), desc="Training episodes")
    for _ in pbar:
        total_reward = train_episode(env, policy, optimizer, gamma)
        episode_rewards.append(total_reward)
        pbar.set_postfix({"total_reward": total_reward})

    run_dir = experiments.make_run_dir(output_dir, name_prefix="reinforce")

    # Save model policy state dict
    model_path = experiments.save_model(policy, run_dir)
    typer.echo(f"Saved policy model to {model_path}")

    # Save plot of total rewards over episodes
    plot_path = experiments.save_reward_plot(episode_rewards, run_dir)
    typer.echo(f"Saved reward plot to {plot_path}")

    # Record video simulation
    video_dir = run_dir / "video"
    evaluation.record_simulation_video(
        _ENV_ID, policy, video_dir, max_steps=max_episode_steps
    )
    typer.echo(f"Saved simulation video to {video_dir}")


if __name__ == "__main__":
    typer.run(train)
