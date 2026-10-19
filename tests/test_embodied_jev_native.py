"""Adapter lifecycle and registry contracts; no model service is called."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from wms.engines.embodied_jev.native import _drive, _settings
from wms.runners.unified import Cancelled


class FakeSession:
    def __init__(self, status="running"):
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.wake = threading.Event()
        self.status = status
        self.cycles = 0
        self.stage = "waiting"
        self.worker = None

    def start(self):
        self.worker = threading.Thread(target=lambda: self.cancel.wait(1), daemon=True)
        self.worker.start()


class Context:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled
        self.events = []

    def check_cancelled(self):
        if self.cancelled:
            raise Cancelled("test cancel")

    def event(self, kind, payload):
        self.events.append((kind, payload))


def test_timeout_stops_inflight_session_and_preserves_outcome():
    session = FakeSession()
    assert _drive(session, Context(), .01)
    assert session.cancel.is_set() and session.wake.is_set()
    assert session.status == "timeout"
    assert not session.worker.is_alive()


def test_user_cancel_never_resumes_or_falls_back():
    session = FakeSession()
    assert _drive(session, Context(cancelled=True), 1)
    assert session.status == "cancelled" and session.cancel.is_set()
    assert session.cycles == 0 and session.worker is None


def test_uncertainty_is_preserved_without_automatic_retry():
    session = FakeSession("uncertain")
    assert _drive(session, Context(), 1)
    assert session.status == "uncertain"
    assert session.cancel.is_set() and session.cycles == 0


def test_runtime_profile_is_not_registry_profile_alias(tmp_path):
    from wms.core.registry import Registry
    machine = tmp_path / "runtime.json"
    machine.write_text(json.dumps({"python": "/worker/python", "source_dir": "/upstream",
                                   "profiles": {"chat": {"model": "mock", "api_key_env": "TEST_KEY"}}}))
    registry = Registry(native_configs={"embodied_jev": str(machine)})
    config = registry.resolve({"recipe": "embodied-jev-chat", "params": {"connection_profile": "chat"}})
    assert config["runtime"]["config_file"] == str(machine)
    assert config["runtime"]["python"] == "/worker/python"
    assert "profiles" not in config["runtime"]
    assert config["params"]["connection_profile"] == "chat"


@pytest.mark.parametrize("params", [
    {"preview": "false"}, {"max_cycles": 0}, {"max_api_calls": -1},
    {"threshold": float("nan")}, {"observation_mode": "vision", "provider": "jev"},
    {"provider": "unknown"}, {"episode_timeout_s": 0},
])
def test_invalid_experiment_is_rejected_before_loading_credentials_or_simulator(tmp_path, params):
    machine = tmp_path / "runtime.json"
    machine.write_text(json.dumps({"source_dir": "/upstream"}))
    with pytest.raises(ValueError):
        _settings({"runtime": {"config_file": str(machine)}, "params": params})

def test_core_evidence_exists_before_video_postprocessing(tmp_path, monkeypatch):
    from wms.engines.embodied_jev import native, connections, video
    machine = tmp_path / "runtime.json"
    machine.write_text(json.dumps({"source_dir": "/not-imported"}))
    raw = {"format": "embodied-jev-episode-v1", "id": "test", "task": "transfer",
           "provider": "baseline", "status": "completed", "success": True,
           "model_calls": 0, "wall_seconds": 1, "model_latency_ms": [],
           "history": [], "frames": [], "events": []}

    class Session:
        def __init__(self, **kwargs):
            self.image_frames = {}
            self.worker = None

        def export(self):
            return raw

        def _close_resources(self):
            pass

    controller = SimpleNamespace(snapshot=lambda: {"actual_http_requests": 0, "api_calls": []}, close=lambda: None)
    monkeypatch.setattr(native, "_prepare_source", lambda runtime: SimpleNamespace(Session=Session))
    monkeypatch.setattr(native, "_provenance", lambda source: {})
    monkeypatch.setattr(native, "_drive", lambda *args: True)
    monkeypatch.setattr(connections, "install_policy", lambda *args: controller)

    def interrupted_render(episode, output, options):
        assert json.loads((output / "episode.raw.json").read_text()) == raw
        assert (output / "trajectory.json").is_file()
        assert json.loads((output / "run_metadata.json").read_text())["video"]["status"] == "pending"
        raise Cancelled("interrupted during rendering")

    monkeypatch.setattr(video, "render_episode", interrupted_render)
    with pytest.raises(Cancelled):
        native.run_episode({"runtime": {"config_file": str(machine)}}, tmp_path / "output", Context())
    assert (tmp_path / "output/episode.raw.json").is_file()