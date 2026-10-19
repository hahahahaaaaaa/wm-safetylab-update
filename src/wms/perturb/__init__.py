"""Perturb what the policy sees; the simulator state is never changed here.

Used by the episode kernel (wms.episodes.kernel); copied from the sub-platform's
sublab_ext.perturb, unchanged in behaviour.

A schedule is a list of items, each active for steps start <= t < end:

  {"type": "occlude", "start": 20, "end": 26, "camera": "base_camera",
   "box": [0.3, 0.3, 0.7, 0.7], "value": 0}        box as fractions (all <= 1) or pixels
  {"type": "occlude", "start": 4, "end": 12, "box": "object:cube", "margin": 0.04}
                                                    box that follows an object every step
                                                    (needs an environment with image_box())
  {"type": "blur",    "start": 30, "end": 40, "sigma": 2.0}
  {"type": "noise",   "start": 0,  "end": 999, "std": 8.0}     pixel units, clipped to 0..255
  {"type": "drop",    "start": 10, "end": 12}                   whole frame black (sensor dropout)
  {"type": "freeze",  "start": 15, "end": 20}                   repeat the last frame seen before start

With rgbd observations, occlude / drop / freeze also change that camera's depth
(occluded or dropped pixels read "depth_value", default 0); blur and noise do not.
"camera" may be omitted or "*" for every camera. "end" may be replaced by
"steps" (start + steps). Items apply in list order. Every applied item is
returned as a record so logs show exactly what each frame went through.
"""
from __future__ import annotations

import copy
import math

import numpy as np

TYPES = {"occlude", "blur", "noise", "drop", "freeze"}


def _box(shape, box):
    height, width = shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1, y0, y1 = x0 * width, x1 * width, y0 * height, y1 * height
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def occlude(image, box, value=0):
    out = image.copy()
    x0, y0, x1, y1 = _box(image.shape, box)
    if value == "mean":
        out[y0:y1, x0:x1] = image.reshape(-1, image.shape[-1]).mean(axis=0).astype(image.dtype)
    else:
        out[y0:y1, x0:x1] = value
    return out


def _kernel(sigma):
    radius = max(1, int(math.ceil(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum(), radius


def blur(image, sigma):
    if sigma <= 0:
        return image.copy()
    kernel, radius = _kernel(float(sigma))
    data = image.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * data.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(data, pad, mode="edge")
        acc = np.zeros_like(data)
        for offset, weight in enumerate(kernel):
            acc += weight * np.take(padded, range(offset, offset + data.shape[axis]), axis=axis)
        data = acc
    return np.clip(np.rint(data), 0, 255).astype(image.dtype)


def noise(image, std, rng):
    out = image.astype(np.float64) + rng.normal(0.0, float(std), size=image.shape)
    return np.clip(np.rint(out), 0, 255).astype(image.dtype)


class Perturbations:
    def __init__(self, schedule=None, seed=0):
        self.items = []
        for index, item in enumerate(schedule or []):
            item = dict(item)
            if item.get("type") not in TYPES:
                raise ValueError(f"perturbation {index}: type must be one of {sorted(TYPES)}")
            start = int(item.get("start", 0))
            end = int(item["end"]) if "end" in item else start + int(item.get("steps", 1))
            if end <= start:
                raise ValueError(f"perturbation {index}: end must be after start")
            if item["type"] == "occlude" and "box" not in item:
                raise ValueError(f"perturbation {index}: occlude needs box")
            if item["type"] == "occlude" and isinstance(item["box"], str) and not item["box"].startswith(("object:", "goal")):
                raise ValueError(f"perturbation {index}: box name must be 'object:<name>' or 'goal'")
            if item["type"] == "blur" and float(item.get("sigma", 0)) <= 0:
                raise ValueError(f"perturbation {index}: blur needs sigma > 0")
            if item["type"] == "noise" and float(item.get("std", 0)) <= 0:
                raise ValueError(f"perturbation {index}: noise needs std > 0")
            self.items.append({**item, "index": index, "start": start, "end": end,
                               "camera": item.get("camera", "*")})
        self.rng = np.random.default_rng(seed)
        self.last_delivered = {}

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last_delivered = {}

    def active(self, t):
        return [item for item in self.items if item["start"] <= t < item["end"]]

    def needs_locate(self):
        return any(item["type"] == "occlude" and isinstance(item["box"], str) for item in self.items)

    def apply(self, obs, t, locate=None):
        """Return (obs_as_seen_by_policy, records). obs['rgb'] maps camera -> HxWx3 uint8;
        obs['depth'], when present, maps camera -> HxW(x1) depth.

        occlude, drop and freeze also apply to that camera's depth (an occluded or dropped
        pixel reads depth_value, default 0 = no reading), so a policy cannot see through them
        in depth. blur and noise model the colour image only and leave depth unchanged.

        locate(name, camera) -> [x0, y0, x1, y1] fractions, used for boxes named
        'object:<name>' or 'goal' so the occluder follows the object."""
        images = obs.get("rgb") if isinstance(obs, dict) else None
        if not images:
            return obs, []
        depths = obs.get("depth") or {}
        seen = copy.copy(obs)
        seen["rgb"] = dict(images)
        if depths:
            seen["depth"] = dict(depths)
        records = []
        for camera, image in images.items():
            frame = np.asarray(image)
            depth = np.asarray(depths[camera]) if camera in depths else None
            for item in self.active(t):
                if item["camera"] not in ("*", camera):
                    continue
                kind = item["type"]
                extra = {}
                if kind == "occlude":
                    box = item["box"]
                    if isinstance(box, str):
                        if locate is None:
                            raise ValueError(f"box {box!r} needs an environment with image_box()")
                        x0, y0, x1, y1 = locate(box, camera)
                        margin = float(item.get("margin", 0.0))
                        box = [x0 - margin, y0 - margin, x1 + margin, y1 + margin]
                        box = [min(1.0, max(0.0, v)) for v in box]
                        extra["resolved_box"] = box
                    frame = occlude(frame, box, item.get("value", 0))
                    if depth is not None:
                        depth = occlude(depth, box, item.get("depth_value", 0))
                elif kind == "blur":
                    frame = blur(frame, float(item["sigma"]))
                elif kind == "noise":
                    frame = noise(frame, float(item["std"]), self.rng)
                elif kind == "drop":
                    frame = np.zeros_like(frame)
                    if depth is not None:
                        depth = np.full_like(depth, item.get("depth_value", 0))
                elif kind == "freeze":
                    held = self.last_delivered.get((camera, item["index"]))
                    if held is not None:
                        frame = held.copy()
                    held_depth = self.last_delivered.get((camera, item["index"], "depth"))
                    if depth is not None and held_depth is not None:
                        depth = held_depth.copy()
                if depth is not None and kind in ("occlude", "drop", "freeze"):
                    extra["depth"] = True
                records.append({"index": item["index"], "type": kind, "camera": camera,
                                **{k: v for k, v in item.items()
                                   if k in ("box", "value", "depth_value", "sigma", "std", "margin")},
                                **extra})
            seen["rgb"][camera] = frame
            if depth is not None:
                seen["depth"][camera] = depth
            for item in self.items:
                # remember the frame delivered just before each freeze window starts
                if item["type"] == "freeze" and t < item["start"] and item["camera"] in ("*", camera):
                    self.last_delivered[(camera, item["index"])] = frame.copy()
                    if depth is not None:
                        self.last_delivered[(camera, item["index"], "depth")] = depth.copy()
        return seen, records


def apply_target_moves(env, moves, t):
    """Move objects or the goal at step t. Each move: {"step", "object"|"goal", "xyz"|"delta"}."""
    done = []
    for index, move in enumerate(moves or []):
        if int(move.get("step", -1)) != t:
            continue
        before = env.get_state() if hasattr(env, "get_state") else None
        if isinstance(move.get("goal"), dict) or move.get("goal") is True or move.get("target") == "goal":
            spec = move["goal"] if isinstance(move.get("goal"), dict) else move
            env.set_goal(xyz=spec.get("xyz"), delta=spec.get("delta"))
            what = "goal"
        else:
            spec = move
            env.set_object_pose(move["object"], xyz=move.get("xyz"), delta=move.get("delta"))
            what = move["object"]
        after = env.get_state() if hasattr(env, "get_state") else None
        done.append({"index": index, "step": t, "target": what, "xyz": spec.get("xyz"),
                     "delta": spec.get("delta"), "state_before": before, "state_after": after})
    return done
