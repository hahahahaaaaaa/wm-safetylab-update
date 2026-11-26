"""Re-render saved MuJoCo states; never run a policy or integrate physics.

The MP4 is a replay of recorded qpos, not a recording of wall-clock execution
or the camera pixels supplied to a model. Scene-hash checks prevent rendering
an episode with a different scene definition.
"""
from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE = "recorded_qpos_replay"


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _sample_frames(frames: list[dict], fps: float, max_frames: int, speed: float) -> tuple[list[int], dict]:
    """Uniform simulation-time sampling using existing states, without interpolation."""
    if not frames:
        raise ValueError("Episode has no recorded frames")
    times = [_number(frame.get("time"), "frame time") for frame in frames]
    if any(after < before for before, after in zip(times, times[1:])):
        raise ValueError("Recorded simulation times must be nondecreasing")
    duration = times[-1] - times[0]
    if duration == 0:
        indices = [len(frames) - 1]
        effective_speed = speed
    else:
        count = min(max_frames, max(2, math.ceil(duration * fps / speed) + 1))
        # Pick the preceding recorded state. Repetition is explicit; there is no
        # interpolation or new simulation between saved states.
        sample_times = [times[0] + duration * i / (count - 1) for i in range(count)]
        indices = [max(0, bisect.bisect_right(times, value) - 1) for value in sample_times]
        indices[0], indices[-1] = 0, len(frames) - 1
        effective_speed = duration * fps / (count - 1)
    return indices, {
        "recorded_frames": len(frames),
        "recorded_time_start": times[0],
        "recorded_time_end": times[-1],
        "simulation_duration_seconds": duration,
        "requested_speed": speed,
        "effective_speed": effective_speed,
        "sampling": "uniform simulation time; preceding recorded qpos; no interpolation",
        "sampled_frame_indices": indices,
        "frames": len(indices),
        "fps": fps,
        "playback_duration_seconds": len(indices) / fps,
        "max_frames": max_frames,
    }


def _render(episode: dict, filename: Path, options: dict, indices: list[int]) -> dict:
    # Keep optional physics/video libraries out of the platform service process.
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from embodied_jev.physics import ASSETS, build_scene

    xml, initial_target, _ = build_scene(episode["task"], episode["seed"], episode.get("scene_config"))
    scene_hash = hashlib.sha256(xml.replace(str(ASSETS), "ASSETS").encode()).hexdigest()
    if not episode.get("scene_hash") or scene_hash != episode["scene_hash"]:
        raise ValueError("Scene hash mismatch; use the episode's original upstream source and scene")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    width, height = options["width"], options["height"]
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    target_ids = [i for i in range(model.ngeom) if model.geom(i).name == "support"
                  or (model.geom(i).name or "").startswith("tray_wall_")]
    target_positions = model.geom_pos[target_ids].copy()
    initial_target = np.asarray(initial_target, dtype=float)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.43, 0, .035]
    camera.distance, camera.azimuth, camera.elevation = 1.0, 180.0, -65.0
    renderer = None
    encoder = None
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
        encoder = imageio_ffmpeg.write_frames(
            str(filename), (width, height), fps=options["fps"],
            codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1, output_params=["-movflags", "+faststart"],
        )
        encoder.send(None)
        for index in indices:
            frame = episode["frames"][index]
            qpos = np.asarray(frame.get("qpos"), dtype=float)
            if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
                raise ValueError(f"Recorded frame {index} has invalid qpos")
            target = np.asarray(frame.get("evaluation_target", initial_target), dtype=float)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError(f"Recorded frame {index} has invalid evaluation target")
            data.qpos[:] = qpos
            data.time = frame["time"]
            model.geom_pos[target_ids] = target_positions + target - initial_target
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            pixels = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(pixels)
            draw.rectangle((0, 0, width, 35), fill="#102c37")
            label = (f"RECORDED STATE REPLAY | {episode['task']} | seed {episode['seed']} | "
                     f"cycle {frame.get('cycle', '?')} | sim {frame['time']:.2f}s")
            draw.text((9, 10), label, fill="white")
            encoder.send(pixels.tobytes())
        encoder.close()
        encoder = None
    finally:
        if encoder is not None:
            with contextlib.suppress(Exception):
                encoder.close()
        if renderer is not None:
            with contextlib.suppress(Exception):
                renderer.close()
    if not filename.is_file() or filename.stat().st_size == 0:
        raise RuntimeError("Video encoder produced no output")
    return {"scene_hash": scene_hash, "width": width, "height": height,
            "mujoco_version": mujoco.__version__, "overlay": "RECORDED STATE REPLAY"}


def _webm(mp4: Path, destination: Path) -> None:
    import subprocess
    import imageio_ffmpeg

    # Transcode the same replay; this does not invoke MuJoCo or a model again.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-i", str(mp4),
         "-an", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34", "-deadline", "realtime",
         "-cpu-used", "8", "-threads", "2", str(destination)],
        capture_output=True, timeout=300, creationflags=flags,
    )
    if completed.returncode or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"WebM encoder failed with status {completed.returncode}")


def render_episode(episode: dict, output: Path, options: dict | None = None) -> dict:
    """Create an MP4 plus provenance, preserving an episode if rendering fails.

    ``output`` is the run's output directory (or an episode subdirectory).
    Options: enabled (default True), fps (20), width (768), height (576),
    max_frames (900), speed (1), webm (False). Returns output descriptors and a
    structured ``status``. Dependencies and renderer failures never invalidate
    the already recorded experiment. All frame indices refer to ``episode``.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = dict(options or {})
    filename = output / "episode-replay.mp4"
    manifest = output / "episode-replay.json"
    result: dict[str, Any] = {
        "source": SOURCE, "episode_id": episode.get("id"), "status": "pending",
        "description": "Recorded qpos rendered again; no policy calls, physics steps, or interpolation. API waiting time is omitted.",
        "model_input_images": False,
        "outputs": [],
    }
    try:
        if not options.get("enabled", True):
            result.update(status="disabled", frames=0)
        else:
            if filename.exists():
                raise FileExistsError("Replay video already exists; use a fresh episode output directory")
            fps = _number(options.get("fps", 20), "fps")
            speed = _number(options.get("speed", 1), "speed")
            width, height, max_frames = (options.get("width", 768), options.get("height", 576),
                                         options.get("max_frames", 900))
            if not 1 <= fps <= 120 or not 0 < speed <= 100:
                raise ValueError("fps must be 1..120 and speed must be >0..100")
            if any(type(value) is not int or value % 2 or not 64 <= value <= 2048 for value in (width, height)):
                raise ValueError("width and height must be even integers in 64..2048")
            if type(max_frames) is not int or not 2 <= max_frames <= 10000:
                raise ValueError("max_frames must be an integer in 2..10000")
            options.update(fps=fps, speed=speed, width=width, height=height, max_frames=max_frames)
            indices, sampling = _sample_frames(episode.get("frames", []), fps, max_frames, speed)
            result.update(sampling)
            details = _render(episode, filename, options, indices)
            result.update(details, status="completed")
            result["outputs"].append({"kind": "video", "path": str(filename), "media_type": "video/mp4"})
            if options.get("webm", False):
                webm = output / "episode-replay.webm"
                try:
                    if webm.exists():
                        raise FileExistsError("WebM replay already exists")
                    _webm(filename, webm)
                    result["outputs"].insert(0, {"kind": "video", "path": str(webm), "media_type": "video/webm"})
                    result["webm_status"] = "completed"
                except Exception as webm_error:
                    result["webm_status"] = "unavailable"
                    result["webm_reason"] = {"type": type(webm_error).__name__, "message": str(webm_error)}
                    if webm.exists() and not isinstance(webm_error, FileExistsError):
                        with contextlib.suppress(OSError):
                            webm.unlink()
    except Exception as exc:
        # Error text comes only from local scene validation/rendering. No
        # connection configuration or remote provider response enters this API.
        result.update(status="unavailable", reason={"type": type(exc).__name__, "message": str(exc)})
        result["outputs"] = []
        # A partial file is deliberately not registered as an experiment video.
        if filename.exists() and not isinstance(exc, FileExistsError):
            with contextlib.suppress(OSError):
                filename.unlink()
    result["outputs"].append({"kind": "json", "path": str(manifest), "media_type": "application/json"})
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result