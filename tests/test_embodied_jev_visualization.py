"""Geometry replay, immutable historical cache, and lightweight HTTP contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from wms.app.embodied_visual import EPISODE_ENTRY, create_embodied_visual_router, visual_status
from wms.core.storage import Store
from wms.engines.embodied_jev import visualization as visual


class FakeRunner:
    def __init__(self, store):
        self.store = store
        self.submissions = []

    def submit(self, config):
        self.submissions.append(copy.deepcopy(config))
        return self.store.create(config)


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / "store")
    runner = FakeRunner(store)
    app = FastAPI()
    app.include_router(create_embodied_visual_router({"store": store, "runner": runner}))
    with TestClient(app) as client:
        yield store, runner, client
    store.close()


def archived_run(store, *, state="completed"):
    run = store.create({"task": "native", "entrypoint": EPISODE_ENTRY,
                        "runtime": {"python": sys.executable, "config_file": "/never-read-profile.json",
                                    "env": {"API_KEY": "fake-not-inherited"}}})
    store.update(run["id"], state=state)
    output = store.output_dir(run["id"])
    visual.atomic_json(output / "episode.raw.json", {"format": "embodied-jev-episode-v1", "scene_hash": "test-scene"})
    visual.atomic_json(output / "run_metadata.json", {"source_dir": "/test-upstream", "source_hashes": {"physics.py": "test-hash"}})
    return run["id"], output


def fake_geometry(episode, **kwargs):
    return ({"geometries": [], "meshes": {}, "scene_hash": "test-scene"},
            {"frames": [{"index": 0, "time": 0, "positions": [], "rotations": []}],
             "duration_s": 0, "time_basis": "simulation_seconds", "reconstruction": "recorded_qpos_mj_forward_only"})


def test_prepare_is_idempotent_and_does_not_modify_source_or_read_profile(service):
    store, runner, client = service
    run_id, output = archived_run(store)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    record = store.get(run_id)
    first = client.post(f"/api/v1/runs/{run_id}/embodied-jev/prepare")
    second = client.post(f"/api/v1/runs/{run_id}/embodied-jev/prepare")
    assert first.status_code == second.status_code == 200
    assert first.json()["prepare_run_id"] == second.json()["prepare_run_id"]
    assert first.json()["state"] == "preparing"
    assert len(runner.submissions) == 1
    prepared = runner.submissions[0]
    assert set(prepared["runtime"]) == {"python"}
    assert "fake-not-inherited" not in json.dumps(prepared)
    assert prepared["params"]["cache_dir"] == str(store.root / "visual-cache" / run_id)
    assert store.get(run_id) == record
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_prepare_rejects_non_episode_active_and_missing_raw(service):
    store, runner, client = service
    other = store.create({"task": "native", "entrypoint": "other:run"})
    assert client.post(f"/api/v1/runs/{other['id']}/embodied-jev/prepare").status_code == 404
    run_id, output = archived_run(store, state="running")
    assert client.post(f"/api/v1/runs/{run_id}/embodied-jev/prepare").status_code == 409
    store.update(run_id, state="completed")
    (output / "episode.raw.json").unlink()
    assert client.post(f"/api/v1/runs/{run_id}/embodied-jev/prepare").status_code == 409
    assert runner.submissions == []


def test_historical_cache_is_hash_checked_and_asset_whitelisted(service, monkeypatch):
    store, runner, client = service
    run_id, output = archived_run(store)
    monkeypatch.setattr(visual, "reconstruct_geometry", fake_geometry)
    cache = store.root / "visual-cache" / run_id
    raw = output / "episode.raw.json"
    visual.write_visualization(json.loads(raw.read_text()), cache, run_id=run_id,
                               raw_path=raw, source_hashes={"physics.py": "test-hash"})
    status = client.get(f"/api/v1/runs/{run_id}/embodied-jev").json()
    assert status["ready"] and status["source"] == "cache" and status["state"] == "ready"
    assert client.get(status["frames_url"]).json()["frames"][0]["time"] == 0
    assert client.get(f"/api/v1/runs/{run_id}/embodied-jev/assets/run_metadata.json").status_code == 404
    assert client.post(status["prepare_url"]).json()["ready"] is True
    assert runner.submissions == []
    raw.write_text(json.dumps({"format": "embodied-jev-episode-v1", "scene_hash": "changed"}))
    changed = client.get(f"/api/v1/runs/{run_id}/embodied-jev").json()
    assert changed["state"] == "stale" and not changed["ready"]
    assert changed["frames_url"] is None


def test_cache_geometry_tampering_is_detected(service, monkeypatch):
    store, _, client = service
    run_id, output = archived_run(store)
    monkeypatch.setattr(visual, "reconstruct_geometry", fake_geometry)
    cache = store.root / "visual-cache" / run_id
    raw = output / "episode.raw.json"
    visual.write_visualization(json.loads(raw.read_text()), cache, run_id=run_id,
                               raw_path=raw, source_hashes={"physics.py": "test-hash"})
    (cache / "visual.frames.json").write_text("{}")
    assert client.get(f"/api/v1/runs/{run_id}/embodied-jev").json()["state"] == "stale"


def test_live_snapshot_updates_geometry_and_decisions_atomically(service):
    store, _, client = service
    run_id, output = archived_run(store, state="running")
    frame = {"time": 1.5, "positions": [[1, 2, 3]], "rotations": [[1, 0, 0, 0, 1, 0, 0, 0, 1]], "observation": {"held": False}}
    session = SimpleNamespace(lock=threading.RLock(), frames=[{"time": 1.0}], last_frame=frame,
        world=SimpleNamespace(scene=lambda: {"geometries": [], "meshes": {}}), cycles=3, phase="approach",
        status="running", stage="deciding", history=[], last_decision={"choice": "move"}, last_intent=None,
        current_candidates=[], events=[], policy=SimpleNamespace(calls=1), started=None, finished=None, message=None)
    recorder = visual.VisualRecorder(output, session, run_id=run_id)
    first_scene = (output / "visual.scene.json").read_bytes()
    status = client.get(f"/api/v1/runs/{run_id}/embodied-jev").json()
    assert status["ready"] and status["state"] == "live"
    first = client.get(status["live_url"]).json()
    assert first["frame"]["time"] == .5 and first["cycle"] == 3
    session.cycles = 4
    session.last_frame = {**frame, "time": 1.7}
    recorder.snapshot(force=True)
    assert client.get(status["live_url"]).json()["cycle"] == 4
    assert (output / "visual.scene.json").read_bytes() == first_scene
    assert not list(output.glob("*.tmp"))


def test_api_import_does_not_load_physics():
    code = "import sys; import wms.app.embodied_visual; assert 'mujoco' not in sys.modules; assert 'embodied_jev.physics' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def saved_episode():
    mujoco = pytest.importorskip("mujoco")
    physics = pytest.importorskip("embodied_jev.physics")
    xml, target, _ = physics.build_scene("transfer", 0)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    scene_hash = hashlib.sha256(xml.replace(str(physics.ASSETS), "ASSETS").encode()).hexdigest()
    # qpos0 is a valid recorded shape. No initialization dynamics are needed.
    frames = [{"time": 1 + i * .04, "qpos": data.qpos.tolist(), "cycle": i, "phase": "test",
               "evaluation_target": (target + [i * .01, 0, 0]).tolist(), "observation": {"sample": i}} for i in range(2)]
    return {"format": "embodied-jev-episode-v1", "task": "transfer", "seed": 0,
            "scene_hash": scene_hash, "scene_config": {}, "frames": frames}


def test_reconstruct_uses_forward_only_and_preserves_recorded_observations(saved_episode, monkeypatch):
    import mujoco
    from embodied_jev.physics import RobotWorld
    def forbidden(*args, **kwargs):
        pytest.fail("Replay must never construct a live world or step physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(RobotWorld, "__init__", forbidden)
    before = copy.deepcopy(saved_episode)
    scene, frames = visual.reconstruct_geometry(saved_episode)
    assert scene["scene_hash"] == saved_episode["scene_hash"]
    assert frames["frames"][0]["time"] == 0
    assert frames["duration_s"] == .04
    assert frames["frames"][1]["observation"] == {"sample": 1}
    support = next(row["id"] for row in scene["geometries"] if row["name"] == "support")
    assert frames["frames"][1]["positions"][support][0] - frames["frames"][0]["positions"][support][0] == pytest.approx(.01)
    assert saved_episode == before


def test_reconstruct_rejects_scene_mismatch(saved_episode):
    saved_episode["scene_hash"] = "incorrect"
    with pytest.raises(ValueError, match="Scene hash mismatch"):
        visual.reconstruct_geometry(saved_episode)


@pytest.mark.parametrize("speed", [-1, 11, float("nan"), True])
def test_native_speed_is_validated(tmp_path, speed):
    from wms.engines.embodied_jev.native import _settings
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"source_dir": "/test"}))
    with pytest.raises(ValueError, match="speed"):
        _settings({"runtime": {"config_file": str(runtime)}, "params": {"speed": speed}})



def test_prepare_worker_keeps_historical_files_and_hashes_unchanged(saved_episode, tmp_path, monkeypatch):
    import mujoco
    import embodied_jev.physics as physics
    def forbidden(*args, **kwargs):
        pytest.fail("Historical preparation must not execute physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(physics.RobotWorld, "__init__", forbidden)
    original = tmp_path / "original-output"
    original.mkdir()
    raw = original / "episode.raw.json"
    visual.atomic_json(raw, saved_episode)
    visual.atomic_json(original / "run_metadata.json", {"original": True})
    before = {path.name: path.read_bytes() for path in original.iterdir()}
    source = Path(physics.__file__).resolve().parents[2]
    cache = tmp_path / "store" / "visual-cache" / "run_recorded"
    config = {"params": {"source_output_dir": str(original), "cache_dir": str(cache),
                         "source_run_id": "run_recorded", "raw_sha256": visual.sha256_file(raw),
                         "source_dir": str(source), "source_hashes": {
                             "physics.py": visual.sha256_file(physics.__file__)}}}
    context = SimpleNamespace(check_cancelled=lambda: None, event=lambda *args: None)
    result = visual.prepare_cache(config, tmp_path / "job-output", context)
    assert result["status"] == "completed"
    assert result["data"]["model_calls"] == 0 and result["data"]["physics_steps"] == 0
    manifest = json.loads((cache / "visual.manifest.json").read_text())
    assert manifest["raw_sha256"] == config["params"]["raw_sha256"]
    assert manifest["run_id"] == "run_recorded"
    assert {path.name: path.read_bytes() for path in original.iterdir()} == before


def test_native_accepts_zero_and_live_speed(tmp_path):
    from wms.engines.embodied_jev.native import _settings
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"source_dir": "/test"}))
    for speed in (0, 1.5, 10):
        _, settings, _ = _settings({"runtime": {"config_file": str(runtime)}, "params": {"speed": speed}})
        assert settings["speed"] == speed
