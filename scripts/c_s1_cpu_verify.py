#!/usr/bin/env python3
"""One-shot CPU-only active-configuration smoke test for S1 compositions.

Runs the same small active image/vector compositions as the S1 integration
pytest, plus matched no-plugin baselines. It uses only ToyPickEnv and never
loads MemoryVLA, TD-MPC2, CUDA or simulator assets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

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
    "probes": ["state"],
    "config": {"memory": True},
}
TOY_PLANNER = {
    "entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy",
    "probes": ["state", "predict", "plan", "embed"],
    "config": {"samples": 16, "horizon": 3},
}

IMAGE_PLUGINS = [
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
VECTOR_PLUGINS = [
    {"plugin": "a-obs-noise-vector", "config": {
        "c": 0.1, "stats": "tdmpc2-pushcube-independent", "toy_std": [1.0] * 7,
    }},
    {"plugin": "a-input-monitor-vector", "config": {
        "stats": "a-input-vector", "fallback_stats": "tdmpc2-pushcube-independent",
        "toy_stats": {"mean": [0.0] * 7, "std": [1.0] * 7},
    }},
]


def _run(label, model, environment, plugins, output):
    from wms.runners.unified import UnifiedWorker

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
    output.mkdir(parents=True, exist_ok=False)
    code = UnifiedWorker(config, output).run()
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    if code != 0 or result.get("status") != "completed":
        raise RuntimeError(f"{label} failed: exit={code}, result={result}")
    return output


def _plugin_ids(trace_path):
    found = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for ids in (row.get("plugins") or {}).values():
            found.update(ids)
    return found


def _actions(steps_path):
    rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row.get("action") for row in rows]


def _suffix_present(ids, registered_id):
    return any(value.rsplit(":", 1)[-1] == registered_id for value in ids)


def main():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = ROOT / ".workspace" / "results" / f"s1-active-toy-{stamp}-{uuid4().hex[:8]}"
    output_root.mkdir(parents=True, exist_ok=False)

    report = {"mode": "CPU toy only; no GPU or real model", "seed": 17, "checks": []}

    image_active = _run(
        "image-active", TOY_CENTROID, TOY_IMAGE_ENV, IMAGE_PLUGINS,
        output_root / "image-active",
    )
    image_baseline = _run(
        "image-baseline", TOY_CENTROID, TOY_IMAGE_ENV, [],
        output_root / "image-baseline",
    )
    image_ids = _plugin_ids(image_active / "episodes" / "ep_000" / "trace.jsonl")
    required_image_ids = [
        "a-obs-corruption", "a-camera-jitter", "a-patch-attack", "a-frame-guard", "a-input-monitor",
    ]
    missing = [plugin_id for plugin_id in required_image_ids if not _suffix_present(image_ids, plugin_id)]
    if missing:
        raise RuntimeError(f"Image composition plugins not called: {missing}")
    image_actions = _actions(image_active / "episodes" / "ep_000" / "steps.jsonl")
    image_base_actions = _actions(image_baseline / "episodes" / "ep_000" / "steps.jsonl")
    image_diff = sum(a != b for a, b in zip(image_actions, image_base_actions)) + abs(
        len(image_actions) - len(image_base_actions)
    )
    report["checks"].append({
        "name": "image_composition",
        "status": "PASS",
        "plugins_called": required_image_ids,
        "action_difference_steps_vs_baseline": image_diff,
        "active_run": str(image_active.relative_to(ROOT)),
        "baseline_run": str(image_baseline.relative_to(ROOT)),
    })

    vector_active = _run(
        "vector-active", TOY_PLANNER, TOY_VECTOR_ENV, VECTOR_PLUGINS,
        output_root / "vector-active",
    )
    vector_baseline = _run(
        "vector-baseline", TOY_PLANNER, TOY_VECTOR_ENV, [],
        output_root / "vector-baseline",
    )
    vector_ids = _plugin_ids(vector_active / "episodes" / "ep_000" / "trace.jsonl")
    required_vector_ids = ["a-obs-noise-vector", "a-input-monitor-vector"]
    missing = [plugin_id for plugin_id in required_vector_ids if not _suffix_present(vector_ids, plugin_id)]
    if missing:
        raise RuntimeError(f"Vector composition plugins not called: {missing}")
    vector_actions = _actions(vector_active / "episodes" / "ep_000" / "steps.jsonl")
    vector_base_actions = _actions(vector_baseline / "episodes" / "ep_000" / "steps.jsonl")
    vector_diff = sum(a != b for a, b in zip(vector_actions, vector_base_actions)) + abs(
        len(vector_actions) - len(vector_base_actions)
    )
    report["checks"].append({
        "name": "vector_composition",
        "status": "PASS",
        "plugins_called": required_vector_ids,
        "action_difference_steps_vs_baseline": vector_diff,
        "active_run": str(vector_active.relative_to(ROOT)),
        "baseline_run": str(vector_baseline.relative_to(ROOT)),
    })

    report["note"] = (
        "PASS means the non-neutral plugin compositions executed on the CPU toy and produced trace evidence. "
        "Action differences are reported, not assumed; this is not a MemoryVLA/TD-MPC2 efficacy experiment."
    )
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
