"""Routines for computing advantages."""


def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []

    # Compute returns from the end of the trajectory going back
    current_return = 0.0
    for r in reversed(rewards):
        current_return = r + current_return * gamma
        returns.append(current_return)

    return returns[::-1]
