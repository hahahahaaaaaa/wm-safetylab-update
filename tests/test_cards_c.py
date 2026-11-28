"""Focused unit tests for the C-workspace S1 overlay.

Run with:
  ./dev.sh python -m pytest tests/test_cards_c.py -q

The authoritative framework integration check remains:
  ./dev.sh plugin-test <registered-plugin-id>
"""
from __future__ import annotations

import json
import random

import numpy as np

from wms.perturb import affine, brightness, color_bias, contrast
from wms.plugins.cards.c_obs_corruption import ObsCorruption, VectorObsNoise
from wms.plugins.cards.c_camera_jitter import CameraJitter
from wms.plugins.cards.c_patch_attack import FixedPatch
from wms.plugins.cards.c_frame_guard import FrameGuard, frame_score
from wms.plugins.cards.c_input_monitor import InputMahalanobis, InputMahalanobisVector


class FakeContext:
    def __init__(self, seed=7, t=0, cameras=None):
        self.t = t
        self.rng = np.random.default_rng(seed)
        self.spec = {"cameras": cameras or {"cam": [64, 64, 3]}}
        self.logs = []
        self.alarms = []

    def log(self, data=None, **fields):
        self.logs.append(dict(data or {}, **fields))

    def alarm(self, reason, score=None, takeover=None, steps=1, action=None):
        self.alarms.append({"reason": reason, "score": score, "takeover": takeover})


def test_photometric_functions_zero_severity_identity_and_uint8():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert brightness(image, 0) is image
    assert contrast(image, 0) is image
    assert color_bias(image, 0) is image
    for output in (brightness(image, 3), contrast(image, 3), color_bias(image, 3)):
        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_photometric_severity_increases_effect_on_non_saturated_image():
    gradient = np.linspace(30, 220, 64, dtype=np.uint8)
    image = np.tile(gradient[None, :, None], (32, 1, 3))
    baseline = image.astype(float)
    for transform in (brightness, contrast, color_bias):
        low = np.mean(np.abs(transform(image, 1).astype(float) - baseline))
        high = np.mean(np.abs(transform(image, 5).astype(float) - baseline))
        assert high > low


def test_affine_rejects_invalid_parameters():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        try:
            affine(image, 0, 0, 0, scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid affine scale accepted: {scale!r}")


def test_affine_identity_and_inverse_translation_center_region():
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    assert affine(image, 0, 0, 0, 1) is image
    moved = affine(image, 3, 0, 0, 1)
    restored = affine(moved, -3, 0, 0, 1)
    error = np.mean(np.abs(restored[8:-8, 8:-8].astype(float) - image[8:-8, 8:-8]))
    assert error < 1.0


def test_obs_corruption_neutral_identity_and_seeded_noise():
    original = np.full((16, 16, 3), 100, dtype=np.uint8)
    obs = {"rgb": {"cam": original}}
    neutral = ObsCorruption({"kind": "brightness", "severity": 0})
    assert neutral.on_observation(obs, FakeContext()) is obs

    first = ObsCorruption({"kind": "gaussian_noise", "severity": 3})
    second = ObsCorruption({"kind": "gaussian_noise", "severity": 3})
    out_a = first.on_observation(obs, FakeContext(seed=11))
    out_b = second.on_observation(obs, FakeContext(seed=11))
    assert out_a is not obs
    assert np.array_equal(out_a["rgb"]["cam"], out_b["rgb"]["cam"])
    assert np.array_equal(obs["rgb"]["cam"], original)


def test_image_corruption_and_jitter_use_context_rng_not_global_rng():
    gradient = np.arange(64, dtype=np.uint8)[None, :, None]
    image = np.tile(gradient, (64, 1, 3))
    obs = {"rgb": {"cam": image}}
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    corruption = ObsCorruption({"kind": "gaussian_noise", "severity": 2})
    corruption.on_observation(obs, FakeContext(seed=31))
    jitter = CameraJitter({"shift": 0.03, "rotate": 2, "scale": 0.03})
    jitter_ctx = FakeContext(seed=37)
    jitter.reset({}, jitter_ctx)
    jitter.on_observation(obs, jitter_ctx)

    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(numpy_state, numpy_after)
    )


def test_vector_noise_neutral_reproducible_and_does_not_touch_global_rng():
    observation = [0.0] * 7
    neutral = VectorObsNoise({"c": 0.0})
    assert neutral.on_observation(observation, FakeContext()) is observation

    config = {"c": 0.1, "std": [1.0] * 35, "toy_std": [1.0] * 7}
    state_py = random.getstate()
    state_np = np.random.get_state()
    first = VectorObsNoise(config)
    second = VectorObsNoise(config)
    ctx_a, ctx_b = FakeContext(seed=19), FakeContext(seed=19)
    first.reset({}, ctx_a)
    second.reset({}, ctx_b)
    out_a = first.on_observation(observation, ctx_a)
    out_b = second.on_observation(observation, ctx_b)
    assert isinstance(out_a, list) and len(out_a) == len(observation)
    assert out_a == out_b and out_a != observation
    assert observation == [0.0] * 7
    assert random.getstate() == state_py
    after_np = np.random.get_state()
    assert all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(state_np, after_np)
    )


def test_camera_jitter_neutral_does_not_consume_context_rng():
    plugin = CameraJitter({"shift": 0, "rotate": 0, "scale": 0, "mode": "persistent"})
    ctx = FakeContext(seed=101)
    before = dict(ctx.rng.bit_generator.state)
    plugin.reset({}, ctx)
    assert ctx.rng.bit_generator.state == before


def test_camera_jitter_neutral_and_persistent_transform():
    gradient = np.arange(64, dtype=np.uint8)[None, :, None]
    image = np.tile(gradient, (64, 1, 3))
    obs = {"rgb": {"cam": image}}
    neutral = CameraJitter({"shift": 0, "rotate": 0, "scale": 0})
    neutral_ctx = FakeContext()
    neutral.reset({}, neutral_ctx)
    assert neutral.on_observation(obs, neutral_ctx) is obs

    jitter = CameraJitter({"shift": 0.05, "rotate": 2, "scale": 0.05, "mode": "persistent"})
    ctx = FakeContext(seed=23)
    jitter.reset({}, ctx)
    first = jitter.on_observation(obs, ctx)
    second = jitter.on_observation(obs, ctx)
    assert first is not obs and second is not obs
    assert np.array_equal(first["rgb"]["cam"], second["rgb"]["cam"])
    assert not np.array_equal(first["rgb"]["cam"], image)
    assert first["rgb"]["cam"].shape == image.shape


def test_fixed_patch_neutral_and_non_mutating():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    obs = {"rgb": {"cam": image}}
    assert FixedPatch({"area": 0}).on_observation(obs, FakeContext()) is obs
    changed = FixedPatch({"area": 0.04, "box": [0.1, 0.1, 0.3, 0.3]}).on_observation(obs, FakeContext())
    assert changed is not obs
    assert np.any(changed["rgb"]["cam"] != image)
    assert np.all(image == 255)
    try:
        FixedPatch({"area": 0.05, "box": [0.1, 0.1, 0.3236068, 0.3236068]})
    except ValueError as exc:
        assert "threat budget" in str(exc)
    else:
        raise AssertionError("patch larger than the declared 0.04 area budget was accepted")


def test_invalid_non_integer_severity_and_non_finite_noise_are_rejected():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for severity in (True, 1.5, -1, 6):
        try:
            ObsCorruption({"kind": "brightness", "severity": severity})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid severity accepted: {severity!r}")
    try:
        VectorObsNoise({"c": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite vector noise strength was accepted")
    try:
        CameraJitter({"shift": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite camera jitter was accepted")
    try:
        CameraJitter({"scale": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("camera scale range allowing zero/negative scale was accepted")


def test_frame_guard_replaces_only_a_single_spike_and_alarms_once():
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    anomaly = np.full_like(base, 255)
    plugin = FrameGuard({"tau": 1.0, "camera": "cam"})
    ctx = FakeContext()
    plugin.reset({}, ctx)

    first = {"rgb": {"cam": base}}
    normal = {"rgb": {"cam": base.copy()}}
    spike = {"rgb": {"cam": anomaly}}
    assert plugin.on_observation(first, ctx) is first
    assert plugin.on_observation(normal, ctx) is normal
    replaced = plugin.on_observation(spike, ctx)
    assert np.array_equal(replaced["rgb"]["cam"], base)
    assert len(ctx.alarms) == 1
    assert ctx.alarms[0]["reason"] == "frame_replaced"
    assert frame_score(anomaly, base) == 255.0
    assert plugin.on_observation(normal, ctx) is normal
    assert len(ctx.alarms) == 1


def test_frame_guard_releases_after_five_consecutive_replacements():
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    anomaly = np.full_like(base, 255)
    plugin = FrameGuard({"tau": 1.0, "camera": "cam", "max_replacements": 5})
    ctx = FakeContext()
    plugin.reset({}, ctx)
    plugin.on_observation({"rgb": {"cam": base}}, ctx)
    for _ in range(5):
        output = plugin.on_observation({"rgb": {"cam": anomaly}}, ctx)
        assert np.array_equal(output["rgb"]["cam"], base)
    sixth = plugin.on_observation({"rgb": {"cam": anomaly}}, ctx)
    assert np.array_equal(sixth["rgb"]["cam"], anomaly)
    assert len(ctx.alarms) == 5



def test_repository_clean_stats_fallbacks_load_for_image_and_vector_toys():
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image_obs = {"rgb": {"top": image}}

    guard = FrameGuard({})
    guard_ctx = FakeContext()
    guard.reset({}, guard_ctx)
    assert np.isfinite(guard.tau) and guard.tau >= 0
    assert guard.on_observation(image_obs, guard_ctx) is image_obs

    image_monitor = InputMahalanobis({})
    image_ctx = FakeContext()
    image_monitor.reset({}, image_ctx)
    assert image_monitor.calibrated is False
    assert image_monitor.stats_source == "memoryvla-bridge"
    assert image_monitor.on_observation(image_obs, image_ctx) is image_obs
    assert np.isfinite(image_ctx.logs[-1]["score"])

    vector_noise = VectorObsNoise({"c": 0.1, "toy_std": [1.0] * 7})
    vector_monitor = InputMahalanobisVector({})
    vector_ctx = FakeContext()
    vector_noise.reset({}, vector_ctx)
    vector_monitor.reset({}, vector_ctx)
    vector = [0.0] * 7
    noisy = vector_noise.on_observation(vector, vector_ctx)
    assert len(noisy) == 7 and noisy != vector
    assert vector_monitor.on_observation(noisy, vector_ctx) is noisy
    assert np.isfinite(vector_ctx.logs[-1]["score"])


def test_input_monitors_are_read_only_and_log_finite_scores(tmp_path):
    image_stats = tmp_path / "image_stats.json"
    image_stats.write_text(json.dumps({
        "image": {"cam": {
            "feature_mean": [0, 100, 100, 100, 10, 10, 10],
            "feature_covariance": np.eye(7).tolist(),
        }}
    }), encoding="utf-8")
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    obs = {"rgb": {"cam": image}}
    monitor = InputMahalanobis({"stats": str(image_stats), "camera": "cam"})
    ctx = FakeContext()
    monitor.reset({}, ctx)
    assert monitor.on_observation(obs, ctx) is obs
    assert len(ctx.logs) == 1 and np.isfinite(ctx.logs[0]["score"])
    assert np.all(obs["rgb"]["cam"] == 100)

    vector_stats = tmp_path / "vector_stats.json"
    vector_stats.write_text(json.dumps({
        "observation_vector": {"mean": [0.0] * 7, "std": [1.0] * 7}
    }), encoding="utf-8")
    vector = [0.1] * 7
    vector_monitor = InputMahalanobisVector({"stats": str(vector_stats)})
    vector_ctx = FakeContext()
    vector_monitor.reset({}, vector_ctx)
    assert vector_monitor.on_observation(vector, vector_ctx) is vector
    assert len(vector_ctx.logs) == 1 and np.isfinite(vector_ctx.logs[0]["score"])
    assert vector == [0.1] * 7


def test_frame_guard_and_image_monitor_reject_ambiguous_wildcard_camera():
    for factory in (FrameGuard, InputMahalanobis):
        try:
            factory({"camera": "*"})
        except ValueError as exc:
            assert "explicit camera" in str(exc)
        else:
            raise AssertionError(f"{factory.__name__} unexpectedly accepted camera='*'")
