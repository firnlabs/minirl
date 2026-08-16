"""Model evaluation routines."""

from pathlib import Path

import gymnasium as gym
import torch
from torch import nn


def record_simulation_video(
    env_id: str,
    policy: nn.Module,
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
    device = next(policy.parameters()).device
    with torch.no_grad():
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action_logits = policy(obs_tensor.unsqueeze(0))
            probs = torch.softmax(action_logits, dim=-1)
            action = int(torch.argmax(probs.squeeze(0)).item())
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
    eval_env.close()
