"""Tests for evaluation.py."""

from minirl import evaluation
from minirl.networks import DiscreteActor


def test_record_simulation_video(tmp_path, policy: DiscreteActor):
    video_dir = tmp_path / "video"
    evaluation.record_simulation_video("CartPole-v1", policy, video_dir, max_steps=5)
    assert video_dir.exists()
    assert any(video_dir.glob("*.mp4"))
