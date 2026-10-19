"""Aligned per-step logs: what the policy saw, what it did, and where things really were.

Ported from the sub-platform's sublab_ext.steplog; the layout is the same.

Layout under the run's output directory:

  episodes.json                       one summary row per episode
  episodes/ep_000/steps.jsonl         one line per control step t
  episodes/ep_000/trace.jsonl         one line per control step t: which plugins ran per stage, their
                                      events, the proposed and executed action, takeover, hazards
  episodes/ep_000/frames_<cam>.npz    seen[k]: frame the policy received at step k (k = 0..T-1),
                                      seen[T]: final frame after the last action
  episodes/ep_000/raw_<cam>.npz       same frames before perturbation (save_raw=True)
  episodes/ep_000/depth_<cam>.npz     rgbd only: depth seen by the policy ("seen") and before
                                      perturbation ("raw", save_raw=True), same indexing
  episodes/ep_000/<cam>.mp4           video of the seen frames (when imageio is available)

Line t of steps.jsonl belongs to frame t: state_before is the true state when
frame t was observed, action is what the policy returned for it, state_after is
the true state after that action. Perturbations and target moves applied at
step t are listed on the same line.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    return value


def strip_images(obs):
    """Observation without image arrays (they are stored in the npz files)."""
    if not isinstance(obs, dict):
        return jsonable(obs)
    return jsonable({k: v for k, v in obs.items() if k not in ("rgb", "depth")})


class StepLogger:
    def __init__(self, output_dir, save_raw=False, video=True, fps=10):
        self.output = Path(output_dir)
        self.save_raw, self.video, self.fps = save_raw, video, fps
        self.episodes = []
        self.files = []
        self.count = 0

    def begin_episode(self, index, seed, state=None, extra=None):
        self.index, self.seed = index, seed
        self.dir = self.output / "episodes" / f"ep_{index:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = (self.dir / "steps.jsonl").open("w", encoding="utf-8")
        self.trace_lines = (self.dir / "trace.jsonl").open("w", encoding="utf-8")
        self.seen, self.raw = {}, {}
        self.depth_seen, self.depth_raw = {}, {}
        self.initial_state = jsonable(state)
        self.extra = jsonable(extra or {})
        self.count = 0

    def frame(self, seen_obs, raw_obs=None):
        """Store one observation's images (and depth, when present); call once per observation, in order."""
        for camera, image in (seen_obs.get("rgb") or {}).items() if isinstance(seen_obs, dict) else []:
            self.seen.setdefault(camera, []).append(np.asarray(image, dtype=np.uint8))
        for camera, depth in (seen_obs.get("depth") or {}).items() if isinstance(seen_obs, dict) else []:
            self.depth_seen.setdefault(camera, []).append(np.asarray(depth))
        if self.save_raw and isinstance(raw_obs, dict):
            for camera, image in (raw_obs.get("rgb") or {}).items():
                self.raw.setdefault(camera, []).append(np.asarray(image, dtype=np.uint8))
            for camera, depth in (raw_obs.get("depth") or {}).items():
                self.depth_raw.setdefault(camera, []).append(np.asarray(depth))

    def step(self, t, *, action, transition, state_before, state_after, perturbations=(), moves=(),
             policy_info=None, seen_obs=None):
        row = {"t": t, "action": jsonable(action),
               "reward": jsonable(transition.get("reward")), "terminated": bool(transition.get("terminated")),
               "truncated": bool(transition.get("truncated")), "events": jsonable(transition.get("events", [])),
               "state_before": jsonable(state_before), "state_after": jsonable(state_after),
               "perturbations": jsonable(list(perturbations)), "target_moves": jsonable(list(moves)),
               "policy_info": jsonable(policy_info), "obs_seen": strip_images(seen_obs)}
        self.lines.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.count += 1

    def trace(self, t, **fields):
        """Line t of trace.jsonl: what the plugins did at step t (see wms.plugins), hazards, takeover."""
        row = {"t": t, **jsonable(fields)}
        self.trace_lines.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def end_episode(self, summary):
        self.lines.close()
        self.trace_lines.close()
        written = ["steps.jsonl", "trace.jsonl"]
        for camera, frames in self.seen.items():
            name = f"frames_{camera}.npz"
            np.savez_compressed(self.dir / name, seen=np.stack(frames))
            written.append(name)
            if self.video:
                written += self._video(camera, frames)
        for camera, frames in self.raw.items():
            name = f"raw_{camera}.npz"
            np.savez_compressed(self.dir / name, raw=np.stack(frames))
            written.append(name)
        for camera, frames in self.depth_seen.items():
            name = f"depth_{camera}.npz"
            arrays = {"seen": np.stack(frames)}
            if camera in self.depth_raw:
                arrays["raw"] = np.stack(self.depth_raw[camera])
            np.savez_compressed(self.dir / name, **arrays)
            written.append(name)
        row = {"episode": self.index, "seed": self.seed, "steps": self.count, "initial_state": self.initial_state,
               "frames": {camera: len(frames) for camera, frames in self.seen.items()},
               **self.extra, **jsonable(summary)}
        (self.dir / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append("summary.json")
        self.files += [(self.dir / name) for name in written]
        self.episodes.append(row)
        return row

    def _video(self, camera, frames):
        try:
            import imageio.v2 as imageio
        except ImportError:
            return []
        name = f"{camera}.mp4"
        try:
            with imageio.get_writer(self.dir / name, fps=self.fps, codec="libx264", quality=8,
                                    macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception:  # noqa: BLE001 - video is a convenience; npz frames are the evidence
            (self.dir / name).unlink(missing_ok=True)
            return []
        return [name]

    def close(self, extra=None):
        summary = {"episodes": self.episodes, **jsonable(extra or {})}
        (self.output / "episodes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs = [{"path": "episodes.json", "kind": "report", "media_type": "application/json"}]
        for path in self.files:
            relative = path.relative_to(self.output).as_posix()
            kind = "video" if path.suffix == ".mp4" else "frames" if path.suffix == ".npz" else "evidence"
            outputs.append({"path": relative, "kind": kind})
        return outputs
