"""PPO training algorithm."""

import logging
from pathlib import Path

import gymnasium as gym
import torch
from torch import distributions
from torch import nn
from tqdm import tqdm
import typer

from minirl import buffers
from minirl import evaluation
from minirl import experiments
from minirl import losses
from minirl import networks


def _parse_action_space(action_space: gym.Space) -> tuple[bool, int, int]:
    if isinstance(action_space, gym.spaces.Discrete):
        is_discrete = True
        num_actions = int(action_space.n)
        action_dim = 1
    elif isinstance(action_space, gym.spaces.Box):
        is_discrete = False
        if action_space.shape is None:
            raise ValueError("Box action space must have a shape")
        num_actions = int(action_space.shape[0])
        action_dim = int(action_space.shape[0])
    else:
        raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

    return is_discrete, num_actions, action_dim


class PPOTrainer:
    def __init__(
        self,
        *,
        env_name: str = "CartPole-v1",
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.999,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        target_kl: float | None = None,
        max_grad_norm: float = 0.5,
        num_steps: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self._env_name = env_name
        self._env = gym.make(self._env_name)
        self._num_steps = num_steps
        self._device = device
        self._gamma = gamma
        self._lam = lam
        self._clip_eps = clip_eps
        self._entropy_coef = entropy_coef
        self._vf_coef = vf_coef
        self._max_grad_norm = max_grad_norm
        self._lr_actor = lr_actor
        self._lr_critic = lr_critic

        if self._env.observation_space.shape is None:
            raise ValueError("Observation space must have a shape")
        self._obs_dim = int(self._env.observation_space.shape[0])
        self._is_discrete, self._num_actions, self._action_dim = _parse_action_space(
            self._env.action_space
        )

        if self._is_discrete:
            self._actor = networks.DiscreteActor(
                obs_dim=self._obs_dim,
                num_actions=self._num_actions,
                hidden_dim=hidden_dim,
            ).to(self._device)
        else:
            # TODO: implement continuous action space support
            raise NotImplementedError("Continuous action spaces not supported yet")

        self._critic = networks.Critic(
            obs_dim=self._obs_dim,
            hidden_dim=hidden_dim,
        ).to(self._device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=lr_actor)
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), lr=lr_critic
        )

        self._buffer = buffers.RolloutBuffer(
            capacity=self._num_steps,
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            is_discrete=self._is_discrete,
            device=self._device,
        )
        self._batch_size = batch_size
        self._target_kl = target_kl
        self._num_epochs = num_epochs
        self._last_done = False

        self._last_obs, _ = self._env.reset()

    @property
    def actor(self) -> nn.Module:
        return self._actor

    @property
    def critic(self) -> nn.Module:
        return self._critic

    def collect_rollouts(self) -> float:
        """Run policy for num_steps, computing rewards and value estimates."""
        self._buffer.reset()

        completed_episode_returns = []
        current_ep_reward = 0.0

        for _ in range(self._num_steps):
            obs_tensor = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            with torch.no_grad():
                action_logits = self._actor(obs_tensor)
                dist = distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self._critic(obs_tensor).squeeze(-1)

            next_obs, reward, terminated, truncated, _ = self._env.step(action.item())
            done = terminated or truncated

            reward = float(reward)
            current_ep_reward += reward

            self._buffer.add(
                buffers.Step(
                    obs=obs_tensor.squeeze(0),
                    action=action.squeeze(0),
                    reward=reward,
                    value=value.squeeze(0),
                    log_prob=log_prob.squeeze(0),
                    done=done,
                )
            )

            self._last_obs = next_obs
            self._last_done = done
            if done:
                completed_episode_returns.append(current_ep_reward)
                current_ep_reward = 0.0
                self._last_obs, _ = self._env.reset()

        mean_return = (
            sum(completed_episode_returns) / len(completed_episode_returns)
            if completed_episode_returns
            else 0.0
        )
        return mean_return

    def train_epoch(self, buffer: buffers.RolloutBuffer) -> dict:
        """Executes one epoch of SGD optimization over mini-batches."""
        epoch_kls = []
        early_stopped = False

        for i, minibatch in enumerate(buffer.get_batches(self._batch_size), start=1):
            # compute new log probs & values for minibatch
            action_logits = self._actor(minibatch.obs)

            if self._is_discrete:
                dist = distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(minibatch.actions)
                entropy = dist.entropy()
            else:
                raise NotImplementedError("Continuous action spaces not supported yet")

            values = self._critic(minibatch.obs).squeeze(-1)

            # compute losses
            assert minibatch.returns is not None
            value_loss = losses.compute_ppo_value_loss(
                values=values,
                target_values=minibatch.returns,
                old_values=minibatch.values,
                clip_eps=self._clip_eps,
            )

            assert minibatch.advantages is not None
            policy_loss, mean_entropy, clip_fraction = losses.compute_ppo_loss(
                old_log_probs=minibatch.old_log_probs,
                new_log_probs=new_log_probs,
                advantages=minibatch.advantages,
                clip_eps=self._clip_eps,
                entropy_coef=self._entropy_coef,
                entropy=entropy,
            )
            logging.debug(
                "Minibatch %d: mean_entropy=%.4f, clip_fraction=%.4f",
                i,
                mean_entropy,
                clip_fraction,
            )

            # optimize actor and critic separately
            self._actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), max_norm=self._max_grad_norm
            )
            self._actor_optimizer.step()

            self._critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._critic.parameters(), max_norm=self._max_grad_norm
            )
            self._critic_optimizer.step()

            # monitor KL divergence for early stopping
            with torch.no_grad():
                log_ratio = new_log_probs - minibatch.old_log_probs
                # using Schulman's unbiased low-variance estimator
                # guaranteed to be non-negative (>= 0)
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                epoch_kls.append(approx_kl)

            if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                logging.info(
                    "Early stopping at minibatch due to high KL divergence:"
                    "approx_kl (%.4f) > 1.5 * self._target_kl (%.4f)",
                    approx_kl,
                    self._target_kl,
                )
                early_stopped = True
                break

        mean_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
        return {"approx_kl": mean_kl, "early_stopped": early_stopped}

    def train_iteration(self) -> float:
        """Runs one complete PPO iteration with rollout + optimization."""
        # collect rollouts
        mean_return = self.collect_rollouts()

        # compute value of last observation
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            last_value = self._critic(last_obs_tensor)[0, 0]

        # compute advantages and returns
        self._buffer.compute_advantages_and_returns(
            last_value=last_value,
            last_done=self._last_done,
            gamma=self._gamma,
            lam=self._lam,
        )

        # run K epochs of SGD over the buffer
        for epoch in tqdm(range(1, self._num_epochs + 1)):
            metrics = self.train_epoch(self._buffer)
            if metrics["early_stopped"]:
                logging.info("Early stopped remaining PPO epochs at epoch %d", epoch)
                break

        return mean_return

    def train(self, total_timesteps: int) -> list:
        """Main training loop until total_timesteps is reached."""
        if total_timesteps < self._num_steps:
            raise ValueError(
                f"total_timesteps ({total_timesteps}) <"
                f" num_steps per iteration ({self._num_steps})"
            )
        num_iterations = total_timesteps // self._num_steps
        mean_returns = []
        for iteration in range(1, num_iterations + 1):
            mean_return = self.train_iteration()
            logging.info("Iteration %d: mean_return=%.4f", iteration, mean_return)
            mean_returns.append(mean_return)
        return mean_returns


def train(
    env_name: str = typer.Option("CartPole-v1", help="Gymnasium environment ID."),
    total_timesteps: int = typer.Option(
        150_000, min=1, help="Total timesteps to train for."
    ),
    hidden_dim: int = typer.Option(64, min=1, help="Hidden dimension for networks."),
    lr_actor: float = typer.Option(3e-4, help="Learning rate for actor."),
    lr_critic: float = typer.Option(3e-4, help="Learning rate for critic."),
    gamma: float = typer.Option(0.999, min=0.0, max=1.0, help="Discount factor."),
    lam: float = typer.Option(0.95, min=0.0, max=1.0, help="GAE lambda parameter."),
    clip_eps: float = typer.Option(
        0.2, min=0.0, help="PPO policy ratio clipping parameter."
    ),
    entropy_coef: float = typer.Option(
        0.01, min=0.0, help="Entropy coefficient for loss."
    ),
    vf_coef: float = typer.Option(
        0.5, min=0.0, help="Value function loss coefficient."
    ),
    target_kl: float | None = typer.Option(
        None, help="Target KL divergence for early stopping."
    ),
    max_grad_norm: float = typer.Option(
        0.5, min=0.0, help="Max gradient norm for clipping."
    ),
    num_steps: int = typer.Option(
        2048, min=1, help="Number of steps per rollout iteration."
    ),
    num_epochs: int = typer.Option(
        10, min=1, help="Number of optimization epochs per iteration."
    ),
    batch_size: int = typer.Option(
        64, min=1, help="Mini-batch size for SGD optimization."
    ),
    device: str = typer.Option(
        "cpu", help="Device to run training on (e.g. cpu, cuda)."
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        help=(
            "Directory path to save run outputs."
            " Files will be stored in a timestamped subdirectory."
        ),
    ),
) -> None:
    trainer = PPOTrainer(
        env_name=env_name,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        lam=lam,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        vf_coef=vf_coef,
        target_kl=target_kl,
        max_grad_norm=max_grad_norm,
        num_steps=num_steps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
    )

    mean_returns = trainer.train(total_timesteps=total_timesteps)

    run_dir = experiments.make_run_dir(output_dir, name_prefix="ppo")

    actor_path = experiments.save_model(trainer.actor, run_dir, name="actor")
    typer.echo(f"Saved actor model to {actor_path}")

    critic_path = experiments.save_model(trainer.critic, run_dir, name="critic")
    typer.echo(f"Saved critic model to {critic_path}")

    plot_path = experiments.save_reward_plot(mean_returns, run_dir, name="rewards_plot")
    typer.echo(f"Saved reward plot to {plot_path}")

    video_dir = run_dir / "video"
    evaluation.record_simulation_video(env_name, trainer.actor, video_dir)
    typer.echo(f"Saved simulation video to {video_dir}")


if __name__ == "__main__":
    typer.run(train)
