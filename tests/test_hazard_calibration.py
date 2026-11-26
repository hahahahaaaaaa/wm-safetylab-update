"""wms.scenarios.hazard_calibration on a synthetic clean run."""
import json

from wms.scenarios.hazard_calibration import measure, propose


def write_run(root, speeds):
    """One episode: the tcp moves along x next to an object at the given speeds (m/s, dt 0.1)."""
    objects = {"cube": [0.0, 0.0, 0.02, 1, 0, 0, 0]}
    tcp, steps = [0.0, 0.0, 0.05], []
    for t, speed in enumerate(speeds):
        before = {"tcp_xyz": list(tcp), "objects": objects}
        tcp = [tcp[0] + speed * 0.1 * (1 if t % 2 == 0 else -1), 0.0, 0.05]
        steps.append({"t": t, "state_before": before, "state_after": {"tcp_xyz": list(tcp), "objects": objects}})
    episode = root / "episodes" / "ep_000"
    episode.mkdir(parents=True)
    (episode / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")
    (root / "episodes.json").write_text(json.dumps({
        "episodes": [{"episode": 0, "initial_state": {"tcp_xyz": [0.0, 0.0, 0.05], "objects": objects}}],
        "env_spec": {"safety": {"dt_s": 0.1, "contact_radius": 0.06}}}), encoding="utf-8")
    return root


def test_only_defaults_that_fire_too_often_are_raised(tmp_path):
    run = write_run(tmp_path / "run", [0.2] * 90 + [0.5] * 10)       # 10% of steps above the 0.3 default
    result = propose(measure([run]), false_positive=0.01)
    assert result["steps"] == 100 and result["changed"] == ["fast_speed"]
    assert abs(result["stats"]["fast_speed"]["current_fires"] - 0.10) < 1e-9
    assert 0.5 <= result["hazards"]["fast_speed"] <= 0.5 * 1.05 + 1e-9
    assert result["hazards"]["drop_height"] == 0.05                    # never fires: the default stays


def test_committed_clean_statistics_load_by_name():
    import pytest
    from wms.plugins.stats import clean_stats
    tdmpc2 = clean_stats("tdmpc2-pushcube-independent")
    assert tdmpc2["latent"]["dims"] == 512 and tdmpc2["episodes"] == 50
    assert tdmpc2["events"]["monitor-surprise.surprise"]["p99"] > 0
    assert "3rd_view_camera" in clean_stats("memoryvla-bridge")["image"]
    with pytest.raises(FileNotFoundError):
        clean_stats("no-such-stats")
