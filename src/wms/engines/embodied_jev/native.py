"""Run one upstream EmbodiedJev episode in a platform-managed worker."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from wms.engines.runtime import load_runtime
from wms.runners.unified import Cancelled

VERSION = "embodied-jev/1"
_SESSION_KEYS = {
    "task", "preview", "threshold", "max_cycles", "scene_config", "user_context",
    "observation_mode", "control_mode", "intervention", "shuffle_candidates", "camera_views",
}
_DEFAULTS = {
    "task": "transfer", "provider": "baseline", "preview": True, "threshold": .55,
    "max_cycles": 30, "observation_mode": "privileged", "control_mode": "skills",
    "shuffle_candidates": False, "max_api_calls": 20, "timeout_seconds": 40,
    "episode_timeout_s": 300, "max_429_retries": 1, "retry_backoff_seconds": 1,
    "record_video": True, "speed": 0,
}


def _positive(value, name, *, integer=False, maximum=None):
    expected = (int,) if integer else (int, float)
    if type(value) not in expected or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds supported limit {maximum}")
    return value


def _settings(config):
    ref = config.get("runtime", {}).get("config_file")
    if not ref:
        raise ValueError("EmbodiedJev requires runtime.config_file")
    runtime = load_runtime(ref)
    base = Path(ref).resolve().parent if isinstance(ref, str) else Path.cwd()
    params = {**_DEFAULTS, **config.get("params", {})}
    if params["provider"] not in {"baseline", "chat", "jev"}:
        raise ValueError("Supported providers: baseline, chat, jev")
    if params["task"] not in {"transfer", "stack", "barrier"}:
        raise ValueError("Supported tasks: transfer, stack, barrier")
    for name in ("preview", "shuffle_candidates", "record_video"):
        if type(params[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    speed = params["speed"]
    if type(speed) not in (int, float) or not math.isfinite(speed) or not 0 <= speed <= 10:
        raise ValueError("speed must be in [0, 10]")
    _positive(params["max_cycles"], "max_cycles", integer=True, maximum=200)
    _positive(params["timeout_seconds"], "timeout_seconds", maximum=120)
    _positive(params["episode_timeout_s"], "episode_timeout_s", maximum=3600)
    threshold = params["threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if type(params["max_api_calls"]) is not int or not 0 <= params["max_api_calls"] <= 500:
        raise ValueError("max_api_calls must be an integer in [0, 500]")
    if params["observation_mode"] not in {"privileged", "rgbd", "vision"}:
        raise ValueError("Unknown observation_mode")
    if params["control_mode"] not in {"skills", "incremental", "hierarchical"}:
        raise ValueError("Unknown control_mode")
    if params["observation_mode"] == "vision" and (params["provider"] != "chat" or params["control_mode"] != "incremental"):
        raise ValueError("Raw vision requires chat + incremental")
    seed = config.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    source = Path(runtime["source_dir"])
    source = source if source.is_absolute() else base / source
    runtime = {**runtime, "source_dir": str(source.resolve())}
    return runtime, params, base


def _prepare_source(runtime):
    source = Path(runtime["source_dir"])
    package = source / "src" / "embodied_jev" / "runtime.py"
    if not package.is_file():
        raise FileNotFoundError("EmbodiedJev source_dir does not contain src/embodied_jev/runtime.py")
    sys.path.insert(0, str(source / "src"))
    import embodied_jev.runtime as upstream
    actual = Path(upstream.__file__).resolve()
    if actual != package.resolve():
        raise RuntimeError("A different EmbodiedJev checkout is already imported; use a fresh worker")
    return upstream


def _provenance(source):
    source = Path(source)
    try:
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    hashes = {name: hashlib.sha256((source / "src" / "embodied_jev" / name).read_bytes()).hexdigest()
              for name in ("runtime.py", "policies.py", "physics.py", "planning.py")}
    versions = {}
    for name in ("mujoco", "numpy", "httpx", "imageio-ffmpeg"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_dir": str(source), "source_commit": commit, "source_hashes": hashes,
            "python": sys.version.split()[0], "packages": versions, "adapter_version": VERSION}


def _halt(session, status, message):
    # Preserve the evaluation outcome instead of stop() replacing it with stopped.
    with session.lock:
        session.cancel.set()
        session.wake.set()
        session.status, session.message = status, message
        session.finished = time.perf_counter()


def _drive(session, context, timeout, visual=None):
    try:
        context.check_cancelled()
    except Cancelled:
        _halt(session, 'cancelled', 'Cancelled before starting model or action')
        return True
    session.start()
    deadline = time.monotonic() + timeout
    last_progress = None
    while session.worker.is_alive():
        if visual is not None:
            visual.snapshot()
        try:
            context.check_cancelled()
        except Cancelled:
            _halt(session, "cancelled", "Cancelled by platform")
            break
        with session.lock:
            status = session.status
            progress = (session.cycles, session.stage, status)
        if progress != last_progress:
            context.event("progress", {"phase": "embodied_jev", "cycle": progress[0],
                                      "stage": progress[1], "episode_status": status})
            last_progress = progress
        if status == "uncertain":
            _halt(session, "uncertain", "Candidate probability below threshold; no automatic fallback")
            break
        if time.monotonic() >= deadline:
            _halt(session, "timeout", f"Episode exceeded {timeout} seconds")
            break
        session.worker.join(.05)
    session.worker.join(.25)
    if visual is not None:
        visual.snapshot(force=True)
    return not session.worker.is_alive()


def run_episode(config, output_dir, context):
    """Native entry point. Logical task outcomes stay separate from HTTP failures."""
    from .connections import build_connection, install_policy
    from .artifacts import write_artifacts

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime, params, base = _settings(config)
    connection = build_connection(runtime, params["provider"],
                                  params.get("connection_profile"), base_dir=base)
    upstream = _prepare_source(runtime)
    metadata = {**_provenance(runtime["source_dir"]),
                "connection": connection.public_config(),
                "experiment": {k: v for k, v in params.items() if k in _SESSION_KEYS or k in {
                    "provider", "max_api_calls", "timeout_seconds", "episode_timeout_s", "speed"}},
                "safety_scope": "Configured upstream end-effector/finger table/obstacle contact checks only",
                "prediction_semantics": "Candidate selection; simulator preview is not a learned world model"}
    options = {k: v for k, v in params.items() if k in _SESSION_KEYS}
    # Construction observes but does not infer. Pass an empty key until our policy is installed.
    public = connection.public_config()
    placeholder = {"key": "", "model": public.get("model", params["provider"]),
                   "url": public.get("url", ""), "json_mode": True}
    session = upstream.Session(seed=config.get("seed", 0), provider=params["provider"],
                               speed=params["speed"], connection=placeholder, **options)
    controller = None
    finished = False
    try:
        controller = install_policy(session, connection, params)
        session.profile_id = params.get("connection_profile", params["provider"])
        context.event("progress", {"phase": "embodied_jev_ready", "provider": params["provider"],
                                  "task": params["task"], "seed": config.get("seed", 0)})
        from .visualization import VisualRecorder, write_visualization
        visual = None
        try:
            visual = VisualRecorder(output, session)
            metadata["visualization"] = {"status": "live", "schema_version": "embodied-jev-visual-v1"}
        except Exception as exc:
            metadata["visualization"] = {"status": "failed", "error_type": type(exc).__name__}
        finished = _drive(session, context, params["episode_timeout_s"], visual)
        episode = session.export()
        metadata["transport"] = controller.snapshot()
        metadata["api_calls"] = metadata["transport"].get("api_calls", [])
        metadata["worker_joined"] = finished
        outputs = []
        if session.image_frames:
            (output / "cameras.zip").write_bytes(session.camera_archive())
            outputs.append({"kind": "camera_archive", "path": "cameras.zip"})
        # Persist the core evidence before slow rendering. Runner cancellation can
        # terminate this process during encoding, but must not lose the episode.
        metadata["video"] = {"status": "pending" if params["record_video"] else "disabled"}
        write_artifacts(output, episode, metadata)
        try:
            run_id = visual.run_id if visual is not None else output.parent.name
            outputs.extend(write_visualization(episode, output, run_id=run_id,
                                               source_hashes=metadata["source_hashes"]))
            if (output / "visual.live.json").is_file():
                outputs.append({"kind": "visualization", "path": "visual.live.json", "media_type": "application/json"})
            metadata["visualization"] = {"status": "ready", "schema_version": "embodied-jev-visual-v1"}
        except Exception as exc:
            metadata["visualization"] = {"status": "failed", "error_type": type(exc).__name__}
        cancelled = episode["status"] == "cancelled"
        try:
            context.check_cancelled()
        except Cancelled:
            cancelled = True
        if params["record_video"] and finished and not cancelled:
            from .video import render_episode
            context.event("progress", {"phase": "embodied_jev_video"})
            video = render_episode(episode, output, {"webm": True, **params.get("video", {})})
            outputs.extend(video.get("outputs", []))
            metadata["video"] = {k: v for k, v in video.items() if k != "outputs"}
        else:
            metadata["video"] = {"status": "not_recorded", "reason": "disabled_or_cancelled_or_worker_not_joined"}
        try:
            context.check_cancelled()
        except Cancelled:
            cancelled = True
        outputs.extend(write_artifacts(output, episode, metadata))
        status = episode["status"]
        platform_status = "cancelled" if cancelled else "failed" if status in {"error", "timeout"} else "completed"
        return {"status": platform_status, "outputs": outputs,
                "error": episode.get("message") if platform_status == "failed" else None,
                "data": {"engine": "embodied_jev", "task": params["task"], "provider": params["provider"],
                         "episode_status": status, "task_success": episode["success"],
                         "model": episode.get("model"), "model_calls": episode.get("model_calls"),
                         "http_requests": metadata["transport"].get("actual_http_requests"),
                         "video_status": metadata["video"].get("status"), "safety_scope": metadata["safety_scope"]}}
    finally:
        if session.worker and session.worker.is_alive():
            session.cancel.set()
            session.wake.set()
            session.worker.join(.25)
        if not session.worker or not session.worker.is_alive():
            session._close_resources()
            if controller is not None:
                controller.close()

