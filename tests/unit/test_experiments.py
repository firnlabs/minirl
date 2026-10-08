import freezegun
import pytest

from minirl import experiments


@pytest.mark.parametrize("name_prefix", ["test", None])
@freezegun.freeze_time("2026-01-01 00:00:00")
def test_make_run_dir(name_prefix, tmp_path):
    if name_prefix is not None:
        run_dir = experiments.make_run_dir(tmp_path, name_prefix)
    else:
        run_dir = experiments.make_run_dir(tmp_path)
        name_prefix = "run"  # default

    assert run_dir.exists()
    assert run_dir.stem == f"{name_prefix}-20260101-000000"


@pytest.mark.parametrize("name", ["myplot", None])
def test_save_reward_plot(name, tmp_path):
    rewards = [10.0, 20.0, 30.0]

    out_path = experiments.save_reward_plot(rewards, tmp_path, name)

    exp_path = tmp_path / f"{name}.png"
    assert out_path == exp_path
    assert out_path.exists()


@pytest.mark.parametrize("name", ["mymodel", None])
def test_save_model(name, policy, tmp_path):
    if name is not None:
        model_path = experiments.save_model(policy, tmp_path, name=name)
    else:
        model_path = experiments.save_model(policy, tmp_path)
        name = "policy"  # default

    exp_model_path = tmp_path / f"{name}.pt"

    assert model_path == exp_model_path
    assert model_path.exists()
