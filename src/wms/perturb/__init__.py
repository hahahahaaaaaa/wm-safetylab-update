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

def affine(image, dx, dy, angle, scale):
    """NumPy-only inverse affine warp with bilinear interpolation and edge extension."""
    src = np.asarray(image)
    dx, dy, angle, scale = map(float, (dx, dy, angle, scale))
    if src.ndim != 3 or not src.shape[0] or not src.shape[1] or not src.shape[2]:
        raise ValueError("affine expects a non-empty HxWxC image")
    if not all(math.isfinite(value) for value in (dx, dy, angle, scale)) or scale <= 0:
        raise ValueError("affine parameters must be finite and scale must be positive")
    if dx == 0 and dy == 0 and angle == 0 and scale == 1:
        return image

    height, width, _ = src.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    center_x, center_y = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    x = xx - center_x - dx
    y = yy - center_y - dy

    source_x = np.clip((cosine * x + sine * y) / scale + center_x, 0, width - 1)
    source_y = np.clip((-sine * x + cosine * y) / scale + center_y, 0, height - 1)
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (source_x - x0)[..., None]
    wy = (source_y - y0)[..., None]

    data = src.astype(np.float64)
    top = data[y0, x0] * (1 - wx) + data[y0, x1] * wx
    bottom = data[y1, x0] * (1 - wx) + data[y1, x1] * wx
    warped = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(warped), 0, 255).astype(src.dtype)


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

# C/S1 RGB corruption primitives
def _check_rgb_uint8(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    return arr


def _check_severity(severity):
    if isinstance(severity, (bool, np.bool_)) or not isinstance(
        severity, (int, np.integer)
    ):
        raise ValueError("severity must be an integer from 0 to 5")
    severity = int(severity)
    if not 0 <= severity <= 5:
        raise ValueError("severity must be between 0 and 5")
    return severity


def _to_uint8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def brightness(image, severity):
    """ImageNet-C style brightness corruption in HSV space."""
    image = _check_rgb_uint8(image)
    severity = _check_severity(severity)

    if severity == 0:
        return image

    levels = (0.1, 0.2, 0.3, 0.4, 0.5)
    rgb = image.astype(np.float64) / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum

    safe_delta = np.where(delta > 0, delta, 1.0)

    red_max = (maximum == r) & (delta > 0)
    green_max = (maximum == g) & (delta > 0)
    blue_max = (maximum == b) & (delta > 0)

    h = np.select(
        [red_max, green_max, blue_max],
        [
            ((g - b) / safe_delta) % 6.0,
            ((b - r) / safe_delta) + 2.0,
            ((r - g) / safe_delta) + 4.0,
        ],
        default=0.0,
    ) / 6.0

    s = np.divide(
        delta, maximum,
        out=np.zeros_like(delta),
        where=maximum > 0,
    )
    v = np.clip(maximum + levels[severity - 1], 0.0, 1.0)

    sector_float = h * 6.0
    sector = np.floor(sector_float).astype(np.int32) % 6
    f = sector_float - np.floor(sector_float)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    out = np.empty_like(rgb)

    choices = (
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    )

    for index, channels in enumerate(choices):
        mask = sector == index
        candidate = np.stack(channels, axis=-1)
        out[mask] = candidate[mask]

    return _to_uint8(out * 255.0)


def contrast(image, severity):
    """ImageNet-C style contrast reduction."""
    image = _check_rgb_uint8(image)
    severity = _check_severity(severity)

    if severity == 0:
        return image

    levels = (0.4, 0.3, 0.2, 0.1, 0.05)

    data = image.astype(np.float64) / 255.0
    means = data.mean(axis=(0, 1), keepdims=True)

    out = (data - means) * levels[severity - 1] + means
    return _to_uint8(out * 255.0)


def color_bias(image, severity):
    """Custom red-blue channel bias; not an ImageNet-C corruption."""
    image = _check_rgb_uint8(image)
    severity = _check_severity(severity)

    if severity == 0:
        return image

    # Custom normalized channel offsets, increasing with severity.
    levels = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    amount = levels[severity] * 255.0

    bias = np.array([amount, 0.0, -amount], dtype=np.float64)
    out = image.astype(np.float64) + bias.reshape(1, 1, 3)

    return _to_uint8(out)
