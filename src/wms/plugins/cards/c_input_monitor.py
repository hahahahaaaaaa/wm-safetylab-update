"""S1 #5: read-only image/vector input anomaly monitors."""
from __future__ import annotations

import numpy as np

from wms.plugins.base import Plugin
from wms.plugins.stats import clean_stats

from .c_frame_guard import frame_score
from .c_s1_utils import camera_items


def _triplet(value, default):
    array = np.asarray(value if value is not None else default, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.size < 3:
        array = np.asarray(default, dtype=float).reshape(-1)
    return array[:3]


class InputMahalanobis(Plugin):
    """Image monitor: frame difference plus RGB channel means and standard deviations."""

    stage, role = "S1", "monitor"
    requires = ("rgb",)

    def __init__(self, config):
        super().__init__(config)
        self.camera = self.config.get("camera", "3rd_view_camera")
        if self.camera in (None, "*"):
            raise ValueError("InputMahalanobis requires one explicit camera name; camera='*' is unsupported")
        self.stats_name = self.config.get("stats", "a-input-image")
        self.fallback_stats = self.config.get("fallback_stats", "memoryvla-bridge")
        self._previous = None
        self._mean = None
        self._inverse_covariance = None
        self.calibrated = False
        self.stats_source = None

    def reset(self, episode, ctx):
        self._previous = None
        try:
            stats = clean_stats(self.stats_name)
            node = stats["image"][self.camera]
            self.calibrated = True
            self.stats_source = self.stats_name
        except (FileNotFoundError, KeyError):
            stats = clean_stats(self.fallback_stats)
            node = stats["image"][self.camera]
            self.calibrated = False
            self.stats_source = self.fallback_stats

        if "feature_mean" in node and "feature_covariance" in node:
            self._mean = np.asarray(node["feature_mean"], dtype=float).reshape(-1)
            covariance = np.asarray(node["feature_covariance"], dtype=float)
        else:
            # Compatibility fallback for CPU conformance only; not final calibration.
            channel_mean = _triplet(node.get("mean"), [0.0, 0.0, 0.0])
            channel_std = np.maximum(_triplet(node.get("std"), [1.0, 1.0, 1.0]), 1.0)
            frame_p99 = float(node["frame_diff"]["p99"])
            frame_scale = max(frame_p99 / 2.326, 1.0)
            self._mean = np.concatenate(([0.0], channel_mean, channel_std))
            scales = np.concatenate(([frame_scale], channel_std, channel_std))
            covariance = np.diag(scales ** 2)
            self.calibrated = False

        ridge = float(self.config.get("ridge", 1e-6))
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if covariance.shape != (self._mean.size, self._mean.size):
            raise ValueError("image feature covariance shape does not match feature_mean")
        if not np.isfinite(self._mean).all() or not np.isfinite(covariance).all():
            raise ValueError("image feature statistics must be finite")
        covariance = covariance + np.eye(covariance.shape[0]) * ridge
        self._inverse_covariance = np.linalg.pinv(covariance)

    def on_observation(self, obs, ctx):
        selected = camera_items(obs, self.camera)
        if not selected:
            return obs
        camera, image = selected[0]
        current = np.asarray(image, dtype=np.float64)
        if current.ndim != 3 or current.shape[-1] != 3 or not np.isfinite(current).all():
            raise ValueError(f"camera {camera!r} must provide a finite HxWx3 RGB image")
        difference = 0.0 if self._previous is None else frame_score(current, self._previous)
        self._previous = current.copy()
        features = np.concatenate((
            [difference], current.mean(axis=(0, 1)), current.std(axis=(0, 1))
        ))
        delta = features - self._mean
        distance_squared = float(delta @ self._inverse_covariance @ delta)
        score = float(np.sqrt(max(distance_squared, 0.0)))
        if not np.isfinite(score):
            raise ValueError("image anomaly score is non-finite")
        ctx.log(
            score=score,
            features=features.tolist(),
            camera=camera,
            feature_kind="image",
            calibrated=self.calibrated,
            stats_source=self.stats_source,
        )
        return obs


class InputMahalanobisVector(Plugin):
    """Vector monitor using diagonal Mahalanobis distance (z-score norm)."""

    stage, role = "S1", "monitor"
    requires = ("vector",)

    def __init__(self, config):
        super().__init__(config)
        self.stats_name = self.config.get("stats", "a-input-vector")
        self.fallback_stats = self.config.get("fallback_stats", "tdmpc2-pushcube-independent")
        self.toy_stats = self.config.get("toy_stats", {"mean": [0.0] * 7, "std": [1.0] * 7})
        self._mean = None
        self._std = None
        self.calibrated = False
        self.stats_source = None

    def reset(self, episode, ctx):
        try:
            stats = clean_stats(self.stats_name)
            self.calibrated = True
            self.stats_source = self.stats_name
        except (FileNotFoundError, KeyError):
            stats = clean_stats(self.fallback_stats)
            self.calibrated = False
            self.stats_source = self.fallback_stats
        node = stats["observation_vector"]
        self._mean = np.asarray(node["mean"], dtype=float).reshape(-1)
        self._std = np.asarray(node["std"], dtype=float).reshape(-1)
        if self._mean.size != self._std.size:
            raise ValueError("vector mean/std dimensions differ")
        if not np.isfinite(self._mean).all() or not np.isfinite(self._std).all() or np.any(self._std < 0):
            raise ValueError("vector mean/std must be finite and std non-negative")

    def on_observation(self, obs, ctx):
        values = np.asarray(obs, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("vector observation must contain only finite values")
        mean, std = self._mean, self._std
        calibrated = self.calibrated
        score_stats_source = self.stats_source
        if mean is None or std is None:
            raise RuntimeError("InputMahalanobisVector.reset() must run before use")
        if values.size == 7 and mean.size == 35:
            mean = np.asarray(self.toy_stats["mean"], dtype=float).reshape(-1)
            std = np.asarray(self.toy_stats["std"], dtype=float).reshape(-1)
            calibrated = False
            score_stats_source = "toy_stats"
        if mean.size != values.size or std.size != values.size:
            raise ValueError(
                f"vector dimension mismatch: observation={values.size}, mean={mean.size}, std={std.size}; "
                "Toy 7-D uses toy_stats and TD-MPC2 uses 35-D clean statistics"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("selected vector statistics must be finite and std non-negative")
        epsilon = float(self.config.get("epsilon", 1e-8))
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        z_score = (values - mean) / np.maximum(std, epsilon)
        score = float(np.linalg.norm(z_score))
        if not np.isfinite(score):
            raise ValueError("vector anomaly score is non-finite")
        ctx.log(score=score, features=values.tolist(), feature_kind="vector", calibrated=calibrated,
                stats_source=score_stats_source)
        return obs
