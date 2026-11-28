"""CPU integration tests for the S1 plugins running together on both toy modalities."""
from __future__ import annotations

import json

from wms.runners.unified import UnifiedWorker


TOY_IMAGE_ENV = {
    "entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
    "config": {"image_size": 64, "max_steps": 4},
}
TOY_VECTOR_ENV = {
    "entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
    "config": {"image_size": 64, "max_steps": 4, "observation": "vector"},
}
TOY_CENTROID = {
    "entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy",
    "config": {"memory": True},
}
TOY_PLANNER = {
    "entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy",
    "config": {"samples": 16, "horizon": 3},
}


def _run(output, model, environment, plugins):
    config = {
        "task": "native",
        "entrypoint": "wms.episodes.kernel:run",
        "seed": 17,
        "model": model,
        "environment": environment,
        "params": {
            "episodes": 1,
            "video": False,
            "save_raw": False,
            "plugin_strict": True,
            "plugins": plugins,
        },
    }
    code = UnifiedWorker(config, output).run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert code == 0, result.get("error")
    trace_path = output / "episodes" / "ep_000" / "trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def _seen_plugin_ids(rows):
    found = set()
    for row in rows:
        for ids in (row.get("plugins") or {}).values():
            found.update(ids)
    return found


def _has_suffix(ids, registered_id):
    return any(value.rsplit(":", 1)[-1] == registered_id for value in ids)


def test_image_s1_fault_attack_defense_monitor_compose(tmp_path):
    plugins = [
        {"plugin": "a-obs-corruption", "config": {
            "kind": "gaussian_noise", "severity": 1, "start": 0, "end": 4, "camera": "*",
        }},
        {"plugin": "a-camera-jitter", "config": {
            "shift": 0.02, "rotate": 2.0, "scale": 0.05, "mode": "persistent", "camera": "*",
        }},
        {"plugin": "a-patch-attack", "config": {
            "area": 0.04, "box": [0.2, 0.2, 0.4, 0.4], "camera": "*", "value": 0,
        }},
        {"plugin": "a-frame-guard", "config": {
            "tau": 1.0, "camera": "3rd_view_camera", "max_replacements": 5,
        }},
        {"plugin": "a-input-monitor", "config": {
            "stats": "a-input-image", "fallback_stats": "memoryvla-bridge", "camera": "3rd_view_camera",
        }},
    ]
    rows = _run(tmp_path / "image-composition", TOY_CENTROID, TOY_IMAGE_ENV, plugins)
    ids = _seen_plugin_ids(rows)
    for registered_id in (
        "a-obs-corruption", "a-camera-jitter", "a-patch-attack", "a-frame-guard", "a-input-monitor",
    ):
        assert _has_suffix(ids, registered_id), f"plugin was not called: {registered_id}"
    assert any(
        event.get("data", {}).get("feature_kind") == "image"
        for row in rows for event in row.get("events", [])
    ), "image monitor did not log a feature score"


def test_vector_s1_noise_and_monitor_compose(tmp_path):
    plugins = [
        {"plugin": "a-obs-noise-vector", "config": {
            "c": 0.1,
            "stats": "tdmpc2-pushcube-independent",
            "toy_std": [1.0] * 7,
        }},
        {"plugin": "a-input-monitor-vector", "config": {
            "stats": "a-input-vector",
            "fallback_stats": "tdmpc2-pushcube-independent",
            "toy_stats": {"mean": [0.0] * 7, "std": [1.0] * 7},
        }},
    ]
    rows = _run(tmp_path / "vector-composition", TOY_PLANNER, TOY_VECTOR_ENV, plugins)
    ids = _seen_plugin_ids(rows)
    assert _has_suffix(ids, "a-obs-noise-vector")
    assert _has_suffix(ids, "a-input-monitor-vector")
    assert any(
        event.get("data", {}).get("feature_kind") == "vector"
        for row in rows for event in row.get("events", [])
    ), "vector monitor did not log a feature score"
