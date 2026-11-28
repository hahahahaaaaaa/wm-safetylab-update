"""S1 #1: RGB corruption and vector Gaussian observation noise."""
from __future__ import annotations

import numpy as np

from wms.plugins.base import Plugin
from wms.plugins.stats import clean_stats
from wms.perturb import blur, brightness, color_bias, contrast, noise

from .c_s1_utils import camera_items, in_window, replace_rgb


_GAUSSIAN_STD = (0.08, 0.12, 0.18, 0.26, 0.38)
_BLUR_SIGMA = (0.5, 1.0, 1.5, 2.0, 3.0)


def _severity(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("severity must be an integer from 0 to 5")
    value = int(value)
    if not 0 <= value <= 5:
        raise ValueError("severity must be between 0 and 5")
    return value


class ObsCorruption(Plugin):
    """S1 image corruption; severity 0 is an exact identity."""

    stage, role = "S1", "fault"
    requires = ("rgb",)
    neutral = {"severity": 0}

    def __init__(self, config):
        super().__init__(config)
        self.kind = str(self.config.get("kind", "gaussian_noise"))
        if self.kind not in {"gaussian_noise", "brightness", "contrast", "color_bias", "blur"}:
            raise ValueError(f"unsupported corruption kind: {self.kind}")
        self.severity = _severity(self.config.get("severity", 0))
        self.start = int(self.config.get("start", 0))
        self.end = self.config.get("end")
        self.end = None if self.end is None else int(self.end)
        if self.start < 0 or (self.end is not None and self.end <= self.start):
            raise ValueError("corruption window must satisfy 0 <= start < end")
        self.camera = self.config.get("camera", "*")

    def _apply(self, image, rng):
        if self.severity == 0:
            return image
        if self.kind == "gaussian_noise":
            return noise(image, _GAUSSIAN_STD[self.severity - 1] * 255.0, rng)
        if self.kind == "brightness":
            return brightness(image, self.severity)
        if self.kind == "contrast":
            return contrast(image, self.severity)
        if self.kind == "color_bias":
            # The existing workspace primitive is deterministic and takes two arguments.
            return color_bias(image, self.severity)
        return blur(image, _BLUR_SIGMA[self.severity - 1])

    def on_observation(self, obs, ctx):
        if self.severity == 0 or not in_window(ctx.t, self.start, self.end):
            return obs
        selected = camera_items(obs, self.camera)
        if not selected:
            return obs
        updates = {}
        for camera, image in selected:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {camera!r} must provide an HxWx3 uint8 RGB image")
            updates[camera] = self._apply(image, ctx.rng)
            ctx.log(kind=self.kind, severity=self.severity, camera=camera, injected=True)
        return replace_rgb(obs, updates)


class VectorObsNoise(Plugin):
    """Vector Gaussian noise with sigma_i = c * clean_std_i."""

    stage, role = "S1", "fault"
    requires = ("vector",)
    neutral = {"c": 0.0}

    def __init__(self, config):
        super().__init__(config)
        self.c = float(self.config.get("c", 0.0))
        if not np.isfinite(self.c) or self.c < 0:
            raise ValueError("c must be finite and non-negative")
        self.stats_name = self.config.get("stats", "tdmpc2-pushcube-independent")
        self._std = (
            np.asarray(self.config["std"], dtype=float).reshape(-1)
            if self.config.get("std") is not None else None
        )
        self.std_source = "config" if self._std is not None else "not_loaded"
        self.toy_std = self.config.get("toy_std")
        self.start = int(self.config.get("start", 0))
        self.end = self.config.get("end")
        self.end = None if self.end is None else int(self.end)
        if self.start < 0 or (self.end is not None and self.end <= self.start):
            raise ValueError("noise window must satisfy 0 <= start < end")

    def reset(self, episode, ctx):
        if self.c > 0 and self._std is None:
            stats = clean_stats(self.stats_name)
            self._std = np.asarray(stats["observation_vector"]["std"], dtype=float).reshape(-1)
            self.std_source = self.stats_name
        if self._std is not None and (not np.isfinite(self._std).all() or np.any(self._std < 0)):
            raise ValueError("clean vector std must be finite and non-negative")

    def on_observation(self, obs, ctx):
        if self.c == 0 or not in_window(ctx.t, self.start, self.end):
            return obs
        values = np.asarray(obs, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("vector observation must contain only finite values")
        std = self._std
        if std is None:
            raise RuntimeError("VectorObsNoise.reset() must run before a non-neutral call")
        if values.size == 7 and std.size == 35:
            if self.toy_std is None:
                raise ValueError("7-D Toy Vector needs explicit toy_std; refusing to truncate 35-D TD-MPC2 std")
            std = np.asarray(self.toy_std, dtype=float).reshape(-1)
        if std.size != values.size:
            raise ValueError(f"vector/std dimension mismatch: observation={values.size}, std={std.size}")
        if not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("std must be finite and non-negative")
        out = values + ctx.rng.normal(0.0, self.c * std, size=values.shape)
        if not np.isfinite(out).all():
            raise ValueError("noisy vector contains non-finite values")
        ctx.log(kind="vector_gaussian_noise", c=self.c, std_source=self.std_source, std_dim=int(std.size), injected=True)
        if isinstance(obs, list):
            return out.tolist()
        if isinstance(obs, tuple):
            return tuple(out.tolist())
        return out.reshape(np.asarray(obs).shape).astype(np.asarray(obs).dtype, copy=False)
