"""S1 #3: fixed-location RGB patch attack; search is performed externally."""
from __future__ import annotations

import math

import numpy as np

from wms.plugins.base import Plugin
from wms.perturb import occlude

from .c_s1_utils import camera_items, replace_rgb


class FixedPatch(Plugin):
    stage, role = "S1", "attack"
    requires = ("rgb",)
    neutral = {"area": 0.0}
    threat = {
        "knowledge": "black_box",
        "access": "S1 image only",
        "budget": {"area": 0.04, "queries": 40},
        "goal": "lower success",
        "adaptive": False,
    }

    def __init__(self, config):
        super().__init__(config)
        self.area = float(self.config.get("area", 0.0))
        self.box = self.config.get("box")
        self.value = self.config.get("value", 0)
        self.camera = self.config.get("camera", "*")
        max_area = float(self.threat["budget"]["area"])
        if not math.isfinite(self.area) or not 0.0 <= self.area <= max_area:
            raise ValueError(f"area must be finite and within the declared threat budget [0, {max_area}]")
        if self.area > 0:
            if not isinstance(self.box, (list, tuple)) or len(self.box) != 4:
                raise ValueError("non-neutral patch requires box=[x0,y0,x1,y1] in image fractions")
            x0, y0, x1, y1 = map(float, self.box)
            if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                raise ValueError("patch coordinates must be finite")
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError("patch box must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1")
            box_area = (x1 - x0) * (y1 - y0)
            if not math.isclose(box_area, self.area, rel_tol=1e-3, abs_tol=1e-6):
                raise ValueError(f"box area {box_area:.6f} does not match configured area {self.area:.6f}")

    def on_observation(self, obs, ctx):
        if self.area == 0:
            return obs
        selected = camera_items(obs, self.camera)
        if not selected:
            return obs
        updates = {}
        for camera, image in selected:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {camera!r} must provide an HxWx3 uint8 RGB image")
            updates[camera] = occlude(image, self.box, value=self.value)
            ctx.log(box=list(self.box), area=self.area, camera=camera, injected=True)
        return replace_rgb(obs, updates)
