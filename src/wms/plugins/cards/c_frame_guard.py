"""S1 #4: abnormal-frame defense using a clean frame-difference p99 threshold."""
from __future__ import annotations

import numpy as np

from wms.plugins.base import Plugin
from wms.plugins.stats import clean_stats

from .c_s1_utils import camera_items, replace_rgb


def frame_score(image, previous):
    """Mean absolute pixel difference in raw 0..255 units."""
    current = np.asarray(image, dtype=np.float64)
    old = np.asarray(previous, dtype=np.float64)
    if current.shape != old.shape:
        return float("inf")
    return float(np.mean(np.abs(current - old)))


class FrameGuard(Plugin):
    stage, role = "S1", "defense"
    requires = ("rgb",)
    neutral = {"tau": None}

    def __init__(self, config):
        super().__init__(config)
        self.camera = self.config.get("camera", "3rd_view_camera")
        if self.camera in (None, "*"):
            raise ValueError("FrameGuard requires one explicit camera name; camera='*' is unsupported")
        self.stats_name = self.config.get("stats", "memoryvla-bridge")
        self._auto_tau = "tau" not in self.config
        self.tau = self.config.get("tau")
        self.threshold_source = self.stats_name if self._auto_tau else "config"
        self.max_replacements = int(self.config.get("max_replacements", 5))
        self._previous = None
        self._streak = 0
        self._threshold_logged = False
        if self.max_replacements < 0:
            raise ValueError("max_replacements must be non-negative")
        if self.tau is not None and (not np.isfinite(float(self.tau)) or float(self.tau) < 0):
            raise ValueError("tau must be finite and non-negative, or None for neutral")

    def reset(self, episode, ctx):
        self._previous = None
        self._streak = 0
        self._threshold_logged = False
        if self._auto_tau:
            self.tau = float(clean_stats(self.stats_name)["image"][self.camera]["frame_diff"]["p99"])
            self.threshold_source = self.stats_name
            if not np.isfinite(self.tau) or self.tau < 0:
                raise ValueError("clean frame-difference p99 threshold must be finite and non-negative")

    def on_observation(self, obs, ctx):
        if self.tau is None:
            return obs
        selected = camera_items(obs, self.camera)
        if not selected:
            return obs
        camera, image = selected[0]
        if not self._threshold_logged:
            ctx.log(kind="frame_guard_threshold", tau=float(self.tau), threshold_source=self.threshold_source)
            self._threshold_logged = True
        current = np.asarray(image)
        if current.dtype != np.uint8 or current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError(f"camera {camera!r} must provide an HxWx3 uint8 RGB image")
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.copy()
            self._streak = 0
            return obs

        score = frame_score(current, self._previous)
        if score > float(self.tau) and self._streak < self.max_replacements:
            self._streak += 1
            ctx.log(kind="frame_guard", score=score, tau=float(self.tau), threshold_source=self.threshold_source,
                    replaced=True, streak=self._streak)
            ctx.alarm("frame_replaced", score=score)
            return replace_rgb(obs, {camera: self._previous})

        # After five consecutive replacements, accept the sixth frame as a new reference.
        self._previous = current.copy()
        self._streak = 0
        return obs
