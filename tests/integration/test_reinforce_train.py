"""Integration test for reinforce.py train routine."""

import pytest
import typer
from typer.testing import CliRunner

from minirl import reinforce


@pytest.fixture
def cli_app():
    app = typer.Typer()
    app.command()(reinforce.train)
    return app


def test_train_cli(cli_app, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli_app,
        [
            "--learning-rate",
            "1e-3",
            "--num-episodes",
            "2",
            "--gamma",
            "0.99",
            "--max-episode-steps",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, (
        f"Failed with exception: {repr(result.exception)}\nCLI Output:\n{result.output}"
    )
    assert "Saved policy model to" in result.stdout
