# miniRL: Minimal reinforcement learning library

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-96.8%25-brightgreen.svg)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A modular reinforcement learning library implemented in PyTorch for learning purposes.

## Getting started

Prerequisites: Python 3.12+ and [uv](https://github.com/astral-sh/uv).

Install dependencies:

```bash
uv sync
```

For running examples, install the following dependencies:

```bash
uv sync --group examples
```

### Algorithms

#### REINFORCE

Train a policy on CartPole with baseline subtraction:

```bash
uv run python src/minirl/reinforce.py --num-episodes 500
```

#### PPO

Train an actor-critic agent with Generalized Advantage Estimation (GAE) and clipped surrogate objectives:

```bash
uv run python src/minirl/ppo.py --env-name CartPole-v1 --total-timesteps 50000
```

#### DPO (preference alignment example)

Align a language model on paired preference data using Direct Preference Optimization:

```bash
uv run --group examples python examples/train_dpo_alignment.py \
  --model-name distilbert/distilgpt2 \
  --num-steps 50
```

---

## Directory structure

```text
.
├── src/minirl/
│   ├── advantages.py           # Generalized Advantage Estimation (GAE) & returns
│   ├── buffers.py              # Rollout buffer and experience collation
│   ├── dpo.py                  # Direct Preference Optimization loss & trainer
│   ├── evaluation.py           # Policy evaluation and video simulation rendering
│   ├── experiments.py          # Metric logging, checkpointing, & plot generation
│   ├── networks.py             # Actor and Critic neural network architectures
│   ├── ppo.py                  # Proximal Policy Optimization (PPO) algorithm
│   └── reinforce.py            # REINFORCE policy gradient algorithm with baseline
├── examples/
│   └── train_dpo_alignment.py  # Language model preference alignment with DPO
├── tests/
│   ├── unit/                   # Algorithmic losses, buffers, & advantage unit tests
│   └── integration/            # Convergence tests for REINFORCE and PPO
├── pyproject.toml              # uv-based Python project configuration
└── uv.lock                     # Hermetic dependency lockfile
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

Check code formatting and linting:

```bash
uv run ruff check && uv run ruff format --check
```

Validate static tensor shape annotations:

```bash
uv run pyrefly check
```

---

## Contributing

Contributions, feedback, and bug reports are welcome. To contribute:

1. Fork the repository and create a feature branch (e.g., `git checkout -b feat/my-feature`).
2. Verify all tests pass, linting succeeds, and types check (`uv run pytest && uv run ruff check && uv run pyrefly check`).
3. Submit a pull request describing the rationale, changes made, and invariants preserved.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
