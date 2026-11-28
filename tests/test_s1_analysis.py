"""CPU-only validation for S1 clean-statistics, detection and search-analysis tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import c_detection_eval, c_fit_input_stats, c_patch_search


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _clean_run(root: Path, run_id="clean"):
    run = root / run_id
    _write_json(run / "config.json", {
        "seed": 3,
        "model": {"entrypoint": "toy"},
        "environment": {"entrypoint": "toy-env"},
        "params": {"episodes": 1, "plugins": [], "perturbations": [], "target_moves": []},
    })
    _write_json(run / "output" / "result.json", {"status": "completed"})
    ep = run / "output" / "episodes" / "ep_000"
    ep.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, value in enumerate((10, 12, 14)):
        image = [[[value, value + 1, value + 2] for _ in range(2)] for _ in range(2)]
        rows.append({"t": t, "obs": {"rgb": {"3rd_view_camera": image},
                                      "observation_vector": [float(t), 1.0]}})
    (ep / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (ep / "trace.jsonl").write_text("", encoding="utf-8")
    return run


def test_fit_stats_accepts_completed_clean_run_and_writes_provenance(tmp_path):
    run = _clean_run(tmp_path)
    image_rows, vector_rows, provenance = c_fit_input_stats._read_runs(
        [run], "3rd_view_camera", "both"
    )
    payloads = c_fit_input_stats._build_payloads(
        [run], "3rd_view_camera", "both", image_rows, vector_rows, provenance
    )
    image = payloads["a-input-image.json"]["image"]["3rd_view_camera"]
    vector = payloads["a-input-vector.json"]["observation_vector"]
    assert image["n"] == 3 and len(image["feature_mean"]) == 7
    assert vector["n"] == 3 and len(vector["mean"]) == 2
    assert provenance[0]["config_sha256"]


def test_fit_stats_rejects_non_clean_plugin_and_perturbation_runs(tmp_path):
    run = _clean_run(tmp_path)
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    config["params"]["perturbations"] = [{"type": "noise", "std": 1}]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="perturbations"):
        c_fit_input_stats._assert_clean_run(run)


def test_fit_stats_follows_transition_obs_field(tmp_path):
    run = _clean_run(tmp_path)
    ep = run / "output" / "episodes" / "ep_000"
    rows = [
        {"t": 0, "transition": {"obs": {"observation_vector": [1.0, 2.0]}}},
        {"t": 1, "transition": {"obs": {"observation_vector": [2.0, 3.0]}}},
    ]
    (ep / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    image_rows, vector_rows, provenance = c_fit_input_stats._read_runs([run], "3rd_view_camera", "vector")
    assert image_rows == []
    assert [row.tolist() for row in vector_rows] == [[1.0, 2.0], [2.0, 3.0]]


def test_fit_stats_rejects_s1_attack_trace_even_if_config_is_empty(tmp_path):
    run = _clean_run(tmp_path)
    trace = run / "output" / "episodes" / "ep_000" / "trace.jsonl"
    event = {"plugin": "a-patch-attack", "stage": "S1", "role": "attack", "t": 1,
             "data": {"injected": True}}
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fault/attack"):
        c_fit_input_stats._assert_clean_run(run)


def _monitor_run(root: Path, run_id: str, injected_steps=(), scores=(0.1, 0.2, 0.3)):
    run = root / run_id
    ep = run / "output" / "episodes" / "ep_000"
    ep.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, score in enumerate(scores):
        events = [{"plugin": "0:a-input-monitor", "stage": "S1", "role": "monitor",
                   "hook": "on_observation", "t": t,
                   "data": {"score": score, "calibrated": True, "stats_source": "a-input-image"}}]
        if t in injected_steps:
            events.append({"plugin": "1:a-obs-corruption", "stage": "S1", "role": "fault",
                           "hook": "on_observation", "t": t,
                           "data": {"injected": True}})
        rows.append({"t": t, "events": events})
    (ep / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return run


def test_detection_eval_keeps_undetected_episodes_in_detection_rate(tmp_path):
    clean = _monitor_run(tmp_path, "clean", scores=(0.1, 0.2, 0.15, 0.18))
    test = _monitor_run(tmp_path, "test", injected_steps=(2, 3), scores=(0.1, 0.2, 0.19, 0.18))
    result = c_detection_eval.evaluate(
        [clean], [test], "a-input-monitor", ("a-obs-corruption",)
    )
    assert result["injected_episode_count"] == 1
    assert result["detected_episode_count"] == 0
    assert result["undetected_episode_count"] == 1
    assert result["episode_detection_rate"] == 0.0
    assert result["mean_detection_delay_steps_detected_only"] is None
    assert result["injected_step_count"] == 2


def test_detection_eval_rejects_missing_monitor_scores_on_injected_steps(tmp_path):
    clean = _monitor_run(tmp_path, "clean", scores=(0.1, 0.2, 0.15, 0.18))
    test = _monitor_run(tmp_path, "test", injected_steps=(2, 3), scores=(0.1, 0.2, 0.19, 0.18))
    trace = test / "output" / "episodes" / "ep_000" / "trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[2]["events"] = [e for e in rows[2]["events"] if e["plugin"] != "0:a-input-monitor"]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="have no monitor score"):
        c_detection_eval.evaluate([clean], [test], "a-input-monitor", ("a-obs-corruption",))


def _search_run(root: Path, run_id: str, rows):
    run = root / ".workspace" / "runs" / run_id
    _write_json(run / "config.json", {"params": {"seeds": [10, 11]}})
    _write_json(run / "output" / "result.json", {"status": "completed"})
    _write_json(run / "output" / "episodes.json", rows)
    return run


def test_patch_search_pairs_by_explicit_seed_not_row_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _search_run(tmp_path, "r1", [
        {"seed": 11, "success": False, "final_distance": 0.8},
        {"seed": 10, "success": True, "final_distance": 0.2},
    ])
    result = c_patch_search.read_metrics("r1", [10, 11])
    assert result["seeds"] == [10, 11]
    assert result["successes"] == [True, False]


def test_patch_search_rejects_missing_episode_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _search_run(tmp_path, "r1", [
        {"success": False, "final_distance": 0.8},
        {"seed": 11, "success": True, "final_distance": 0.2},
    ])
    with pytest.raises(ValueError, match="no recorded seed"):
        c_patch_search.read_metrics("r1", [10, 11])
