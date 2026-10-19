"""Tests for original-state replay provenance and optional video failures."""
import json

import pytest

from wms.engines.embodied_jev import video


def test_sampling_preserves_recorded_states_and_final_frame():
    frames = [{"time": 0}, {"time": 0.11}, {"time": 0.9}, {"time": 1.0}]
    indices, provenance = video._sample_frames(frames, fps=10, max_frames=100, speed=1)
    assert indices == [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3]
    assert provenance["sampled_frame_indices"] == indices
    assert provenance["simulation_duration_seconds"] == 1
    assert provenance["effective_speed"] == 1
    assert "no interpolation" in provenance["sampling"]


def test_frame_budget_reports_playback_acceleration():
    indices, provenance = video._sample_frames([{"time": 0}, {"time": 30}], 20, 301, 1)
    assert len(indices) == 301
    assert indices[0] == 0 and indices[-1] == 1
    assert provenance["effective_speed"] == 2


@pytest.mark.parametrize("times", [[0, -1], [0, float("nan")], [True]])
def test_invalid_recorded_time_is_not_silently_repaired(times):
    with pytest.raises(ValueError):
        video._sample_frames([{"time": t} for t in times], 20, 100, 1)


def test_missing_renderer_keeps_episode_and_structured_failure(tmp_path, monkeypatch):
    episode = {"id": "original-episode", "frames": [{"time": 0}, {"time": 1}]}
    before = json.dumps(episode)

    def unavailable(*args):
        raise ModuleNotFoundError("imageio_ffmpeg")

    monkeypatch.setattr(video, "_render", unavailable)
    result = video.render_episode(episode, tmp_path)
    assert result["status"] == "unavailable"
    assert result["reason"]["type"] == "ModuleNotFoundError"
    assert not any(row["kind"] == "video" for row in result["outputs"])
    assert result["episode_id"] == "original-episode"
    assert json.loads((tmp_path / "episode-replay.json").read_text())["source"] == "recorded_qpos_replay"
    assert json.dumps(episode) == before


def test_partial_video_is_not_exposed_as_success(tmp_path, monkeypatch):
    def invalid(episode, filename, options, indices):
        filename.write_bytes(b"incomplete encoding")
        raise ValueError("invalid qpos")

    monkeypatch.setattr(video, "_render", invalid)
    result = video.render_episode({"frames": [{"time": 0}]}, tmp_path)
    assert result["status"] == "unavailable"
    assert not (tmp_path / "episode-replay.mp4").exists()
    assert len(result["outputs"]) == 1


def test_disabled_render_does_not_load_optional_dependencies(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("Disabled recording should not enter renderer")

    monkeypatch.setattr(video, "_render", unexpected)
    assert video.render_episode({}, tmp_path, {"enabled": False})["status"] == "disabled"


def test_existing_video_is_preserved(tmp_path, monkeypatch):
    original = tmp_path / "episode-replay.mp4"
    original.write_bytes(b"existing experiment")
    result = video.render_episode({"frames": [{"time": 0}]}, tmp_path)
    assert result["status"] == "unavailable"
    assert result["reason"]["type"] == "FileExistsError"
    assert original.read_bytes() == b"existing experiment"

def test_webm_failure_retains_valid_mp4(tmp_path, monkeypatch):
    def rendered(episode, filename, options, indices):
        filename.write_bytes(b"encoded replay fixture")
        return {"scene_hash": "verified-scene"}

    def fail_webm(mp4, destination):
        destination.write_bytes(b"partial")
        raise RuntimeError("Codec unavailable")

    monkeypatch.setattr(video, "_render", rendered)
    monkeypatch.setattr(video, "_webm", fail_webm)
    result = video.render_episode({"frames": [{"time": 0}]}, tmp_path, {"webm": True})
    assert result["status"] == "completed"
    assert result["webm_status"] == "unavailable"
    assert any(row["media_type"] == "video/mp4" for row in result["outputs"])
    assert not (tmp_path / "episode-replay.webm").exists()
