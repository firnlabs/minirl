"""Implements the REINFORCE policy gradient algorithm."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
from matplotlib import pyplot as plt
import torch
from torch import distributions
from torch import nn
from torch import optim
import typer

if TYPE_CHECKING:
    from shape_extensions import Int
    from shape_extensions import IntVar
    from torch import Tensor


_ENV_ID = "CartPole-v1"


class PolicyNetwork[
    InDim: IntVar,
    OutDim: IntVar,
    HiddenDim: IntVar,
](nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(
        self,
        *,
        obs_dim: Int[InDim],
        action_dim: Int[OutDim],
        hidden_dim: Int[HiddenDim] = 128,
    ) -> None:
        """Initializes the policy network.

        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim

        self._fc1 = nn.Linear(obs_dim, hidden_dim)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(hidden_dim, action_dim)
        self._softmax = nn.Softmax(dim=-1)

    def forward[B: IntVar](self, obs: Tensor[[B, InDim]]) -> Tensor[[B, OutDim]]:
        """Runs the forward pass and returns action probabilities."""
        h: Tensor[[B, HiddenDim]] = self._relu(self._fc1(obs))
        probs: Tensor[[B, OutDim]] = self._softmax(self._fc2(h))
        return probs


def collect_trajectory_data(
    env: gym.Env,
    policy: PolicyNetwork,
) -> tuple[list[Tensor], list[float]]:
    log_probs = []
    rewards = []
    done = False

    obs, _ = env.reset()

    while not done:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action_probs = policy(obs_tensor)

        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        obs, reward, terminated, truncated, _ = env.step(action.item())

        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))

        done = terminated or truncated

    return log_probs, rewards


def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []

    # Compute returns from the end of the trajectory going back
    current_return = 0.0
    for r in reversed(rewards):
        current_return = r + current_return * gamma
        returns.append(current_return)

    return returns[::-1]


def compute_loss(log_probs: list[Tensor], returns: list[float]) -> Tensor:
    losses = []

    for log_prob, ret in zip(log_probs, returns, strict=True):
        losses.append(-log_prob * ret)

    loss = torch.stack(losses).sum()
    return loss


def train_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    optimizer: optim.Optimizer,
    gamma: float,
) -> float:
    log_probs, rewards = collect_trajectory_data(env, policy)
    returns = compute_discounted_returns(rewards, gamma)
    loss = compute_loss(log_probs, returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(sum(rewards))


def _make_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"reinforce-run-{timestamp}"
    run_dir.mkdir(parents=True)
    return run_dir


def _save_reward_plot(episode_rewards: list[float], output_path: Path) -> None:
    """Saves a plot of total rewards over training episodes."""
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Training Rewards")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def _record_simulation_video(
    env_id: str,
    policy: PolicyNetwork,
    video_folder: Path,
    max_steps: int = 100,
) -> None:
    """Runs a single simulation episode using the trained policy and records video."""
    eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_steps)

    # NOTE: RecordVideo will assume `video_folder` does not exist and will create it
    # If the folder does exist, it will output a potential overwrite warning.
    eval_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=str(video_folder),
        name_prefix="simulation",
        disable_logger=True,
    )

    obs, _ = eval_env.reset()
    done = False
    with torch.no_grad():
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_tensor.unsqueeze(0))
            action = int(torch.argmax(probs.squeeze(0)).item())
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
    eval_env.close()


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

    policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_rewards = []
    for i in range(num_episodes):
        total_reward = train_episode(env, policy, optimizer, gamma)
        episode_rewards.append(total_reward)
        typer.echo(f"Episode {i + 1}: total reward = {total_reward}")

    run_dir = _make_run_dir(output_dir)

    # Save model policy state dict
    model_path = run_dir / "policy.pt"
    torch.save(policy.state_dict(), model_path)
    typer.echo(f"Saved policy model to {model_path}")

    # Save plot of total rewards over episodes
    plot_path = run_dir / "rewards.png"
    _save_reward_plot(episode_rewards, plot_path)
    typer.echo(f"Saved reward plot to {plot_path}")

    # Record video simulation
    video_dir = run_dir / "video"
    _record_simulation_video(_ENV_ID, policy, video_dir, max_steps=max_episode_steps)
    typer.echo(f"Saved simulation video to {video_dir}")


if __name__ == "__main__":
    typer.run(train)
