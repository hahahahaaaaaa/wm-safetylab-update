"""S1 #2: persistent or per-step camera pose jitter."""
from __future__ import annotations

import numpy as np

from wms.plugins.base import Plugin
from wms.perturb import affine

from .c_s1_utils import camera_items, replace_rgb


class CameraJitter(Plugin):
    stage, role = "S1", "fault"
    requires = ("rgb",)
    neutral = {"shift": 0.0, "rotate": 0.0, "scale": 0.0}

    def __init__(self, config):
        super().__init__(config)
        self.shift = float(self.config.get("shift", 0.0))
        self.rotate = float(self.config.get("rotate", 0.0))
        self.scale = float(self.config.get("scale", 0.0))
        self.mode = str(self.config.get("mode", "persistent"))
        self.camera = self.config.get("camera", "*")
        self._persistent = None
        limits = (self.shift, self.rotate, self.scale)
        if not all(np.isfinite(value) and value >= 0 for value in limits):
            raise ValueError("shift, rotate and scale limits must be finite and non-negative")
        if self.scale >= 1.0:
            raise ValueError("scale limit must be < 1 so the sampled scale stays positive")
        if self.mode not in {"persistent", "per_step"}:
            raise ValueError("mode must be 'persistent' or 'per_step'")

    def _draw(self, ctx):
        return (
            float(ctx.rng.uniform(-self.shift, self.shift)),
            float(ctx.rng.uniform(-self.shift, self.shift)),
            float(ctx.rng.uniform(-self.rotate, self.rotate)),
            float(1.0 + ctx.rng.uniform(-self.scale, self.scale)),
        )

    @staticmethod
    def _shape_from_spec(spec_shape):
        if isinstance(spec_shape, dict):
            height = spec_shape.get("height", spec_shape.get("h"))
            width = spec_shape.get("width", spec_shape.get("w"))
            channels = spec_shape.get("channels", spec_shape.get("c", 3))
            if height is None or width is None:
                return None
            return int(height), int(width), int(channels)
        if isinstance(spec_shape, (list, tuple)) and len(spec_shape) >= 2:
            return tuple(int(value) for value in spec_shape[:3])
        return None

    def reset(self, episode, ctx):
        neutral = self.shift == 0 and self.rotate == 0 and self.scale == 0
        self._persistent = (
            self._draw(ctx) if self.mode == "persistent" and not neutral else None
        )

    def on_observation(self, obs, ctx):
        if self.shift == 0 and self.rotate == 0 and self.scale == 0:
            return obs
        selected = camera_items(obs, self.camera)
        if not selected:
            return obs
        transform = self._persistent if self.mode == "persistent" else self._draw(ctx)
        if transform is None:
            transform = self._draw(ctx)
            self._persistent = transform
        shift_x, shift_y, angle, scale = transform
        updates = {}
        spec_cameras = (getattr(ctx, "spec", {}) or {}).get("cameras", {})
        for camera, image in selected:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {camera!r} must provide an HxWx3 uint8 RGB image")
            height, width = image.shape[:2]
            spec_shape = self._shape_from_spec(spec_cameras.get(camera)) if isinstance(spec_cameras, dict) else None
            if spec_shape is not None and tuple(spec_shape[:2]) != (height, width):
                raise ValueError(
                    f"camera shape disagrees with ctx.spec for {camera}: image={(height, width)}, spec={spec_shape}"
                )
            dx, dy = shift_x * width, shift_y * height
            updates[camera] = affine(image, dx=dx, dy=dy, angle=angle, scale=scale)
            ctx.log(dx=dx, dy=dy, angle=angle, scale=scale, mode=self.mode, camera=camera, injected=True)
        return replace_rgb(obs, updates)
