import pytest
import typer
from typer.testing import CliRunner

from minirl import ppo


@pytest.fixture
def cli_app():
    app = typer.Typer()
    app.command()(ppo.train)
    return app


@pytest.fixture
def ppo_trainer() -> ppo.PPOTrainer:
    return ppo.PPOTrainer(
        hidden_dim=16,
        num_steps=16,
        num_epochs=2,
        batch_size=8,
    )


class TestPPOTrainer:
    """Integration test for PPOTrainer."""

    def test_train(self, ppo_trainer):
        total_timesteps = 32

        mean_returns = ppo_trainer.train(total_timesteps)

        exp_iterations = total_timesteps // ppo_trainer._num_steps
        assert len(mean_returns) == exp_iterations

    def test_train_error(self, ppo_trainer):
        total_timesteps = 1

        with pytest.raises(ValueError, match="total_timesteps"):
            ppo_trainer.train(total_timesteps)


def test_train_cli(cli_app, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli_app,
        [
            "--total-timesteps",
            "16",
            "--num-steps",
            "16",
            "--num-epochs",
            "1",
            "--batch-size",
            "8",
            "--output-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, (
        f"Failed with exception: {repr(result.exception)}\nCLI Output:\n{result.output}"
    )
    assert "Saved actor model to" in result.stdout
    assert "Saved critic model to" in result.stdout
    assert "Saved reward plot to" in result.stdout
    assert "Saved simulation video to" in result.stdout
