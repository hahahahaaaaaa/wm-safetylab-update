"""Read-only EmbodiedJev geometry assets plus queued, credential-free preparation."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from wms.engines.embodied_jev.visualization import ASSETS, SCHEMA, sha256_file

EPISODE_ENTRY = "wms.engines.embodied_jev.native:run_episode"
PREPARE_ENTRY = "wms.engines.embodied_jev.visualization:prepare_cache"
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


@lru_cache(maxsize=256)
def _digest(path, size, mtime_ns):
    return sha256_file(path)


def _file_hash(path):
    stat = path.stat()
    return _digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _contained_file(directory, name):
    directory = Path(directory).resolve()
    target = (directory / name).resolve()
    return target if target.is_relative_to(directory) and target.is_file() else None


def _episode_run(store, run_id):
    run = store.get(run_id)
    if run.get("config", {}).get("entrypoint") != EPISODE_ENTRY:
        raise HTTPException(404, "This run is not an EmbodiedJev episode")
    return run


def _pending(store, run_id, raw_hash=None):
    for row in store.list_runs():
        config = row.get("config", {})
        params = config.get("params", {})
        if (config.get("entrypoint") == PREPARE_ENTRY and params.get("source_run_id") == run_id
                and row["state"] not in TERMINAL
                and (raw_hash is None or params.get("raw_sha256") == raw_hash)):
            return row
    return None


def visual_status(store, run_id):
    run = _episode_run(store, run_id)
    output = store.output_dir(run_id).resolve()
    cache = (store.root / "visual-cache" / run_id).resolve()
    if not cache.is_relative_to((store.root / "visual-cache").resolve()):
        raise HTTPException(404, "Invalid visual cache path")
    root_url = f"/api/v1/runs/{run_id}/embodied-jev"
    result = {"schema_version": SCHEMA, "run_id": run_id, "ready": False, "state": "missing",
              "source": None, "prepare_url": root_url + "/prepare", "prepare_run_id": None,
              "reason": None, "platform_state": run["state"],
              **{name + "_url": None for name in ASSETS}}
    locations = {}
    raw = _contained_file(output, ASSETS["episode"])
    metadata = _json(output / "run_metadata.json")
    for name in ("episode", "decisions", "summary", "events"):
        target = _contained_file(output, ASSETS[name])
        if target is not None:
            locations[name] = target
    raw_hash = _file_hash(raw) if raw else None
    stale = False
    for origin, folder in (("output", output), ("cache", cache)):
        manifest = _json(folder / "visual.manifest.json")
        if not manifest:
            continue
        valid = (manifest.get("schema_version") == SCHEMA and manifest.get("status") == "ready"
                 and manifest.get("run_id") == run_id and raw_hash is not None
                 and manifest.get("raw_sha256") == raw_hash
                 and manifest.get("source_hashes", {}) == metadata.get("source_hashes", {}))
        files = {name: _contained_file(folder, ASSETS[name]) for name in ("scene", "frames")}
        for name, target in files.items():
            asset = manifest.get("assets", {}).get(name, {})
            valid = valid and target is not None and asset.get("file") == ASSETS[name]
            if valid:
                valid = asset.get("sha256") == _file_hash(target)
        if valid:
            locations.update(files)
            result.update(ready=True, state="ready", source=origin,
                          frame_count=manifest.get("frame_count"), duration_s=manifest.get("duration_s"),
                          raw_sha256=raw_hash)
            break
        stale = True
    live = _contained_file(output, ASSETS["live"])
    if live is not None:
        locations["live"] = live
    if not result["ready"] and run["state"] not in TERMINAL:
        scene = _contained_file(output, ASSETS["scene"])
        if live is not None and scene is not None:
            locations["scene"] = scene
            result.update(ready=True, state="live", source="output")
        else:
            result.update(state="unavailable", reason="episode_starting")
    elif not result["ready"]:
        pending = _pending(store, run_id, raw_hash)
        if pending:
            result.update(state="preparing", prepare_run_id=pending["id"])
        elif raw is None:
            result.update(state="unavailable", reason="raw_episode_missing")
        else:
            result.update(state="stale" if stale else "missing", reason="prepare_recorded_geometry")
    for name in locations:
        result[name + "_url"] = root_url + "/assets/" + name
    return result, locations


def create_embodied_visual_router(services):
    router = APIRouter(prefix="/api/v1/runs")
    preparation_lock = threading.Lock()

    @router.get("/{run_id}/embodied-jev")
    def status(run_id: str):
        return visual_status(services["store"], run_id)[0]

    @router.get("/{run_id}/embodied-jev/assets/{asset}")
    def asset(run_id: str, asset: str):
        if asset not in ASSETS:
            raise HTTPException(404, "Unknown visualization asset")
        _, locations = visual_status(services["store"], run_id)
        target = locations.get(asset)
        if target is None:
            raise HTTPException(404, "Visualization asset is not ready")
        return FileResponse(target, media_type="application/json", headers={"Cache-Control": "no-store"})

    @router.post("/{run_id}/embodied-jev/prepare")
    def prepare(run_id: str):
        store = services["store"]
        with preparation_lock:
            run = _episode_run(store, run_id)
            if run["state"] not in TERMINAL:
                raise HTTPException(409, "Wait for this episode to finish before preparing historical replay")
            status, locations = visual_status(store, run_id)
            if status["ready"] or status["state"] == "preparing":
                return status
            raw = locations.get("episode")
            if raw is None:
                raise HTTPException(409, "This run has no raw episode to replay")
            output = store.output_dir(run_id).resolve()
            metadata = _json(output / "run_metadata.json")
            if not metadata.get("source_dir") or not metadata.get("source_hashes"):
                raise HTTPException(409, "Original source provenance is missing")
            # Reuse only the worker interpreter and search paths. Never read a
            # runtime profile, inherit API env vars, or resolve credentials.
            runtime = {key: value for key, value in run["config"].get("runtime", {}).items()
                       if key in {"python", "python_paths", "cwd"}}
            if not runtime.get("python"):
                raise HTTPException(409, "Original worker Python is missing")
            raw_hash = _file_hash(raw)
            job = services["runner"].submit({
                "name": f"Prepare EmbodiedJev replay · {run_id}", "task": "native",
                "direction": "embodied-replay", "entrypoint": PREPARE_ENTRY,
                "runtime": runtime, "timeout_s": 300, "metrics": [],
                "params": {"source_run_id": run_id, "source_output_dir": str(output),
                           "cache_dir": str(store.root / "visual-cache" / run_id),
                           "raw_sha256": raw_hash, "source_dir": metadata["source_dir"],
                           "source_hashes": metadata["source_hashes"]}})
            return {**status, "state": "preparing", "prepare_run_id": job["id"], "reason": None}

    return router
