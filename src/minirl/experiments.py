"""Helper routines for running experiments."""

from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch import nn


def make_run_dir(output_dir: Path, name_prefix: str = "run") -> Path:
    """Creates timestamped directory to store experiment files."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{name_prefix}-{timestamp}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_reward_plot(
    episode_rewards: list[float], output_dir: Path, name: str = "rewards_plot"
) -> Path:
    """Saves a plot of total rewards over training episodes.

    Args:
        episode_rewards: List of reward per episode.
        output_dir: Directory to store saved plot as an image file.
        name: Name of the file containing the plot (no extension).
            Extension (.png) will be added.

    Returns:
        File path of the saved plot.
    """
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.grid(True)

    output_path = output_dir / f"{name}.png"
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_model(model: nn.Module, output_dir: Path, name: str = "policy") -> Path:
    """Saves policy model to a file.

    Args:
        model: PyTorch model to save.
        output_dir: Output directory to store model file.
        name: Model file name without suffix (".pl" will be added to it)

    Returns:
        The path where the model was saved.
    """
    model_path = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), model_path)
    return model_path
