"""Buffers for storing simulation rollouts."""

import dataclasses

from shape_extensions import Int
from shape_extensions import IntVar
import torch
from torch import Tensor


@dataclasses.dataclass(slots=True, frozen=True)
class Step[ObsDim: IntVar, ActionDim: IntVar]:
    """Single transition step within a rollout trajectory."""

    obs: Tensor[[ObsDim]]
    action: Tensor[[ActionDim]]
    reward: float
    value: Tensor[[1]]
    log_prob: Tensor[[1]]
    done: bool


class RolloutBuffer[Capacity: IntVar, ObsDim: IntVar, ActionDim: IntVar]:
    def __init__(
        self,
        *,
        capacity: Int[Capacity],
        obs_dim: Int[ObsDim],
        action_dim: Int[ActionDim] | None = None,
        is_discrete: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize."""
        self._capacity = capacity
        self._device = device
        self._ptr = 0

        # pre-allocated tensor storage
        self._obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)

        if is_discrete:
            action_dtype = torch.long
            action_space = (capacity,)
        else:
            action_dtype = torch.float32
            action_space = (capacity, action_dim)

        self._actions = torch.zeros(action_space, dtype=action_dtype)
        self._rewards = torch.zeros((capacity,), dtype=torch.float32)
        self._values = torch.zeros((capacity,), dtype=torch.float32)
        self._log_probs = torch.zeros((capacity,), dtype=torch.float32)
        self._dones = torch.zeros((capacity,), dtype=torch.bool)

        # calculated values
        self._advantages = None
        self._returns = None

    def add(self, step: Step) -> None:
        self._obs[self._ptr] = step.obs.detach()
        self._actions[self._ptr] = step.action.detach()
        self._rewards[self._ptr] = step.reward
        self._values[self._ptr] = step.value.detach()
        self._log_probs[self._ptr] = step.log_prob.detach()
        self._dones[self._ptr] = step.done

        self._ptr += 1

    @property
    def obs(self) -> Tensor[[Capacity, ObsDim]]:
        return self._obs

    @property
    def actions(self) -> Tensor[[Capacity]] | Tensor[[Capacity, ActionDim]]:
        return self._actions

    @property
    def rewards(self) -> Tensor[[Capacity]]:
        return self._rewards

    @property
    def values(self) -> Tensor[[Capacity]]:
        return self._values

    @property
    def log_probs(self) -> Tensor[[Capacity]]:
        return self._log_probs

    @property
    def dones(self) -> Tensor[[Capacity]]:
        return self._dones

    @property
    def advantages(self) -> Tensor[[Capacity]] | None:
        return self._advantages

    @property
    def returns(self) -> Tensor[[Capacity]] | None:
        return self._returns
