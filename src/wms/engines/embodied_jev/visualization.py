"""Geometry replay and live JSON, with optional physics imports confined to workers."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

SCHEMA = "embodied-jev-visual-v1"
ASSETS = {"scene": "visual.scene.json", "frames": "visual.frames.json", "live": "visual.live.json",
          "episode": "episode.raw.json", "decisions": "decisions.json", "summary": "summary.json",
          "events": "events.json"}


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Frame time must be finite")
    return float(value)


def frame_document(frame: dict, index: int, time_origin: float) -> dict:
    absolute = _time(frame["time"])
    return {"index": index, "time": round(absolute - time_origin, 8), "sim_time": absolute,
            "cycle": frame.get("cycle", 0), "phase": frame.get("phase"),
            "positions": copy.deepcopy(frame["positions"]), "rotations": copy.deepcopy(frame["rotations"]),
            "observation": copy.deepcopy(frame.get("observation", {}))}


class VisualRecorder:
    """Write one scene and rate-limited snapshots while the native episode runs."""

    def __init__(self, output, session, run_id=None, interval=.5):
        self.output = Path(output)
        self.session = session
        self.run_id = run_id or os.environ.get("WMS_RUN_ID") or self.output.parent.name
        self.interval = interval
        self.last_write = -math.inf
        with session.lock:
            self.time_origin = _time(session.frames[0]["time"])
            scene = session.world.scene()
        atomic_json(self.output / ASSETS["scene"], {"schema_version": SCHEMA, "run_id": self.run_id, **scene})
        self.snapshot(force=True)

    def snapshot(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < self.interval:
            return
        session = self.session
        with session.lock:
            frame = {**session.last_frame, "cycle": session.cycles, "phase": session.phase}
            document = {"schema_version": SCHEMA, "run_id": self.run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": session.status, "stage": session.stage, "cycle": session.cycles,
                "frame": frame_document(frame, max(0, len(session.frames) - 1), self.time_origin),
                "history": copy.deepcopy(session.history), "last_decision": copy.deepcopy(session.last_decision),
                "last_intent": copy.deepcopy(session.last_intent), "candidates": copy.deepcopy(session.current_candidates),
                "events": copy.deepcopy(list(session.events)), "model_calls": session.policy.calls,
                "wall_seconds": round((session.finished or time.perf_counter()) - session.started, 3) if session.started else 0,
                "message": session.message, "time_basis": "simulation_seconds"}
        atomic_json(self.output / ASSETS["live"], document)
        self.last_write = now


def reconstruct_geometry(episode: dict, *, context=None) -> tuple[dict, dict]:
    """Restore saved qpos with mj_forward only: no RobotWorld constructor or mj_step."""
    import mujoco
    import numpy as np
    from embodied_jev.physics import ASSETS as PANDA_ASSETS, RobotWorld, build_scene

    recorded = episode.get("frames")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("Episode has no recorded frames")
    xml, target, _ = build_scene(episode["task"], episode["seed"], episode.get("scene_config"))
    scene_hash = hashlib.sha256(xml.replace(str(PANDA_ASSETS), "ASSETS").encode()).hexdigest()
    if scene_hash != episode.get("scene_hash"):
        raise ValueError("Scene hash mismatch; replay requires the original upstream source")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # scene() only serializes the static model and these explicit fields. Bypass
    # RobotWorld.__init__: it advances physics during initialization.
    world = object.__new__(RobotWorld)
    world.model, world.target = model, np.asarray(target, dtype=float)
    world.task, world.seed, world.scene_hash = episode["task"], episode["seed"], scene_hash
    world.scene_config = copy.deepcopy(episode.get("scene_config", {}))
    world.scene_name = world.scene_config.get("name", episode["task"])
    scene = world.scene()
    target_ids = [i for i in range(model.ngeom) if model.geom(i).name == "support"
                  or (model.geom(i).name or "").startswith("tray_wall_")]
    original_target = np.asarray(target, dtype=float)
    original_positions = model.geom_pos[target_ids].copy()
    first_time = _time(recorded[0]["time"])
    previous_time = first_time
    frames = []
    for index, saved in enumerate(recorded):
        if context is not None and index % 100 == 0:
            context.check_cancelled()
        stamp = _time(saved["time"])
        if stamp < previous_time:
            raise ValueError("Recorded simulation times must be nondecreasing")
        previous_time = stamp
        qpos = np.asarray(saved.get("qpos"), dtype=float)
        evaluation_target = np.asarray(saved.get("evaluation_target", original_target), dtype=float)
        if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("Recorded qpos has an invalid shape or non-finite value")
        if evaluation_target.shape != (3,) or not np.isfinite(evaluation_target).all():
            raise ValueError("Recorded target has an invalid shape or non-finite value")
        data.qpos[:] = qpos
        data.time = stamp
        model.geom_pos[target_ids] = original_positions + evaluation_target - original_target
        mujoco.mj_forward(model, data)
        frame = {**saved, "positions": data.geom_xpos.round(5).tolist(),
                 "rotations": data.geom_xmat.round(6).tolist()}
        frames.append(frame_document(frame, index, first_time))
    return scene, {"time_basis": "simulation_seconds", "time_origin": first_time,
                   "duration_s": round(previous_time - first_time, 8), "frames": frames,
                   "reconstruction": "recorded_qpos_mj_forward_only"}


def write_visualization(episode, output, *, run_id, raw_path=None, source_hashes=None, context=None):
    """Write derived scene/poses; manifest is committed last after both assets."""
    output = Path(output)
    raw_path = Path(raw_path) if raw_path is not None else output / ASSETS["episode"]
    raw_hash = sha256_file(raw_path)
    scene, frames = reconstruct_geometry(episode, context=context)
    scene_document = {"schema_version": SCHEMA, "run_id": run_id, **scene}
    scene_path = output / ASSETS["scene"]
    try:
        existing_scene = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        existing_scene = None
    # A new live episode already wrote this static scene; do not rewrite it.
    if existing_scene != scene_document:
        atomic_json(scene_path, scene_document)
    atomic_json(output / ASSETS["frames"], {"schema_version": SCHEMA, "run_id": run_id, **frames})
    if sha256_file(raw_path) != raw_hash:
        raise ValueError("Raw episode changed during geometry reconstruction")
    manifest = {"schema_version": SCHEMA, "status": "ready", "run_id": run_id,
                "raw_sha256": raw_hash, "source_hashes": dict(source_hashes or {}),
                "scene_hash": episode["scene_hash"], "frame_count": len(frames["frames"]),
                "duration_s": frames["duration_s"], "reconstruction": frames["reconstruction"],
                "assets": {key: {"file": ASSETS[key], "sha256": sha256_file(output / ASSETS[key])}
                           for key in ("scene", "frames")}}
    atomic_json(output / "visual.manifest.json", manifest)
    return [{"kind": "visualization", "path": name, "media_type": "application/json"}
            for name in (ASSETS["scene"], ASSETS["frames"], "visual.manifest.json")]


def prepare_cache(config, output_dir, context):
    """Queued native worker: derive a cache without modifying original evidence."""
    params = config["params"]
    source_output = Path(params["source_output_dir"]).resolve()
    cache = Path(params["cache_dir"]).resolve()
    run_id = params["source_run_id"]
    if cache.name != run_id or cache.parent.name != "visual-cache" or cache == source_output:
        raise ValueError("Invalid visual cache directory")
    raw = source_output / ASSETS["episode"]
    if sha256_file(raw) != params["raw_sha256"]:
        raise ValueError("Source episode changed after preparation was queued")
    source = Path(params["source_dir"]).resolve()
    source_hashes = params.get("source_hashes", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Original source hashes are required")
    for name, expected in source_hashes.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Invalid source hash entry")
        if sha256_file(source / "src" / "embodied_jev" / name) != expected:
            raise ValueError("Upstream source changed; original source is required for replay")
    sys.path.insert(0, str(source / "src"))
    import embodied_jev.physics as physics
    if Path(physics.__file__).resolve() != source / "src" / "embodied_jev" / "physics.py":
        raise ValueError("A different upstream physics module is already loaded")
    context.check_cancelled()
    episode = json.loads(raw.read_text(encoding="utf-8-sig"))
    if episode.get("format") != "embodied-jev-episode-v1":
        raise ValueError("Unsupported raw episode format")
    context.event("progress", {"phase": "embodied_jev_visual_cache", "source_run": run_id})
    write_visualization(episode, cache, run_id=run_id, raw_path=raw,
                        source_hashes=source_hashes, context=context)
    if sha256_file(raw) != params["raw_sha256"]:
        raise ValueError("Source episode changed while preparing replay")
    receipt = {"schema_version": SCHEMA, "source_run_id": run_id, "raw_sha256": params["raw_sha256"],
               "cache_dir": str(cache), "model_calls": 0, "physics_steps": 0}
    atomic_json(Path(output_dir) / "visual-cache-receipt.json", receipt)
    return {"status": "completed", "outputs": [{"kind": "evidence", "path": "visual-cache-receipt.json"}],
            "data": {"engine": "embodied_jev_visual_cache", **receipt}}


