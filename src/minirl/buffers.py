"""Buffers for storing simulation rollouts."""

from collections.abc import Iterator
import dataclasses

from jaxtyping import Bool
from jaxtyping import Float
import torch
from torch import Tensor

from minirl import advantages


@dataclasses.dataclass(slots=True, frozen=True)
class Step:
    """Single transition step within a rollout trajectory."""

    obs: Float[Tensor, "obs_dim"]
    action: Tensor
    reward: float
    value: Float[Tensor, "1"]
    log_prob: Float[Tensor, "1"]
    done: bool


@dataclasses.dataclass(slots=True, frozen=True)
class Minibatch:
    """Minibatch of rollout data."""

    obs: Float[Tensor, "batch_size obs_dim"]
    actions: Tensor
    rewards: Float[Tensor, "batch_size"]
    values: Float[Tensor, "batch_size"]
    old_log_probs: Float[Tensor, "batch_size"]
    dones: Bool[Tensor, "batch_size"]
    advantages: Float[Tensor, "batch_size"] | None
    returns: Float[Tensor, "batch_size"] | None


class RolloutBuffer:
    """Buffer for storing simulation rollouts."""

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        action_dim: int | None = None,
        is_discrete: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize."""
        self._capacity = capacity
        self._ptr = 0

        # pre-allocated tensor storage
        self._obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)

        if is_discrete:
            action_dtype = torch.long
            action_space = (capacity,)
        else:
            if action_dim is None:
                raise ValueError(
                    "action_dim must be provided when is_discrete is False"
                )
            action_dtype = torch.float32
            action_space = (capacity, action_dim)

        self._actions = torch.zeros(action_space, dtype=action_dtype, device=device)
        self._rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self._values = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self._log_probs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self._dones = torch.zeros((capacity,), dtype=torch.bool, device=device)

        # calculated values
        self._advantages = None
        self._returns = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def obs(self) -> Float[Tensor, "capacity obs_dim"]:
        return self._obs

    @property
    def actions(self) -> Tensor:
        return self._actions

    @property
    def rewards(self) -> Float[Tensor, "capacity"]:
        return self._rewards

    @property
    def values(self) -> Float[Tensor, "capacity"]:
        return self._values

    @property
    def log_probs(self) -> Float[Tensor, "capacity"]:
        return self._log_probs

    @property
    def dones(self) -> Bool[Tensor, "capacity"]:
        return self._dones

    @property
    def advantages(self) -> Float[Tensor, "capacity"] | None:
        return self._advantages

    @property
    def returns(self) -> Float[Tensor, "capacity"] | None:
        return self._returns

    @property
    def device(self) -> torch.device:
        return self._obs.device

    def add(self, step: Step) -> None:
        """Appends a transition step to the buffer.

        Args:
            step: Single environment transition step to record.
        """
        self._obs[self._ptr] = step.obs.detach()
        self._actions[self._ptr] = step.action.detach()
        self._rewards[self._ptr] = step.reward
        self._values[self._ptr] = step.value.detach()
        self._log_probs[self._ptr] = step.log_prob.detach()
        self._dones[self._ptr] = step.done

        self._ptr += 1

    def get_batches(self, batch_size: int) -> Iterator[Minibatch]:
        """Generates minibatches of rollout data.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            Minibatch of rollout data taken from a random sample.

        Raises:
            RuntimeError: If advantages and returns have not been computed
                using compute_advantages_and_returns().
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Advantages and returns must be computed before sampling batches."
                " Call compute_advantages_and_returns() first."
            )

        num_samples = self.capacity
        indices = torch.randperm(num_samples, device="cpu")

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield Minibatch(
                obs=self.obs[batch_indices],
                actions=self.actions[batch_indices],
                rewards=self.rewards[batch_indices],
                values=self.values[batch_indices],
                old_log_probs=self.log_probs[batch_indices],
                dones=self.dones[batch_indices],
                advantages=(
                    self.advantages[batch_indices]
                    if self.advantages is not None
                    else None
                ),
                returns=(
                    self.returns[batch_indices] if self.returns is not None else None
                ),
            )

    def is_full(self) -> bool:
        """Returns True if the buffer has reached maximum capacity."""
        return self._ptr >= self._capacity

    def reset(self) -> None:
        """Resets the buffer insertion pointer to zero."""
        self._ptr = 0

    def compute_advantages_and_returns(
        self,
        last_value: Float[Tensor, ""],
        last_done: bool,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        """Computes GAE advantages and target returns across stored rollouts.

        Args:
            last_value: Critic value estimate for the observation at step T.
            last_done: Terminal flag of the final step.
            gamma: Discount factor.
            lam: Generalized Advantage Estimation decay factor (lambda).
        """
        last_value = (
            torch.tensor(0.0, device=self.device, dtype=self._values.dtype)
            if last_done
            else last_value
        )

        advantages_, returns = advantages.compute_gae(
            rewards=self._rewards,
            values=self._values,
            last_value=last_value,
            dones=self._dones,
            gamma=gamma,
            lam=lam,
        )

        self._advantages = advantages_
        self._returns = returns
