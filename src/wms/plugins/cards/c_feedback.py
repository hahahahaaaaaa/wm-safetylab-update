"""S6 calibrated CUSUM and toy-only online-feedback poisoning."""
from __future__ import annotations

import copy
from wms.plugins.base import Plugin
from wms.plugins.cards.c_action import _integer, _positive
from wms.plugins.residuals import RESIDUAL_VERSION, planned_next_residual


class CusumState:
    """One-sided standardized CUSUM; only eligible samples update the state."""
    def __init__(self, center, spread, drift, threshold, cooldown):
        self.center = _positive(center, "center", allow_zero=True)
        self.spread = _positive(spread, "spread")
        self.drift = _positive(drift, "drift", allow_zero=True)
        self.threshold = _positive(threshold, "threshold")
        self.cooldown = _integer(cooldown, "cooldown")
        self.score, self.until = 0.0, -1

    def update(self, t, residual):
        if residual is None or t < self.until:
            self.score = 0.0
            return {"score": 0.0, "z": None, "detected": False, "cooldown": t < self.until}
        z = (residual - self.center) / self.spread
        self.score = max(0.0, self.score + z - self.drift)
        score = self.score
        detected = score >= self.threshold
        if detected:
            self.score = 0.0
            self.until = t + self.cooldown + 1
        return {"score": score, "z": z, "detected": detected, "cooldown": False}


class _Residual(Plugin):
    stage, side, requires = "S6", "model", ("predict", "embed")

    def __init__(self, config):
        super().__init__(config)
        self.scale = self.config.get("scale", 1.0)
        self.include_changed = self.config.get("include_changed", False)
        if not isinstance(self.include_changed, bool):
            raise ValueError("include_changed must be a boolean")
        values = self.scale if isinstance(self.scale, list) else [self.scale]
        if not values:
            raise ValueError("scale cannot be empty")
        for value in values:
            _positive(value, "scale")

    def on_feedback(self, transition, ctx):
        return transition

    def residual(self, step):
        return planned_next_residual(step, self.scale, self.include_changed)


class ResidualMonitor(_Residual):
    """Clean-run calibration monitor. Returned feedback and actions are untouched."""
    role = "monitor"

    def reset(self, episode, ctx):
        self.values, self.skipped = [], 0

    def after_step(self, step, ctx):
        sample = self.residual(step)
        if sample["valid"]:
            self.values.append(sample["residual"])
        else:
            self.skipped += 1
        ctx.log(residual_sample=sample, residual_version=RESIDUAL_VERSION)

    def end_episode(self, ctx):
        return {"valid_samples": len(self.values), "skipped_samples": self.skipped,
                "mean_residual": sum(self.values) / len(self.values) if self.values else None}


class CusumDefense(_Residual):
    role = "defense"

    def __init__(self, config):
        super().__init__(config)
        self.takeover = self.config.get("takeover", "none")
        if self.takeover not in ("none", "stop", "fallback"):
            raise ValueError("takeover must be none, stop or fallback")
        self.steps = _integer(self.config.get("steps", 2), "steps", 1)
        self.cooldown = _integer(self.config.get("cooldown", 5), "cooldown")
        if self.takeover == "fallback" and self.cooldown < self.steps:
            raise ValueError("fallback cooldown must be >= takeover steps")
        self._state()  # reject invalid parameters before running an episode

    def _state(self):
        return CusumState(self.config.get("center", 0.0), self.config.get("spread", .01),
                          self.config.get("drift", .5), self.config.get("threshold", 8.0), self.cooldown)

    def reset(self, episode, ctx):
        import numpy as np
        self.detector = self._state()
        self.alarms = self.samples = self.skipped = 0
        self.fallback = self.config.get("fallback_action")
        if self.takeover == "fallback":
            if self.fallback is None:
                self.fallback = ctx.spec.get("safe_action")
            if self.fallback is None:
                raise ValueError("fallback requires an explicit controller-specific fallback_action or spec.safe_action")
            action = np.asarray(self.fallback, dtype=float)
            low, high = np.asarray(ctx.spec["action_low"]), np.asarray(ctx.spec["action_high"])
            if action.shape != low.shape or action.shape != high.shape or not np.isfinite(action).all() or (action < low).any() or (action > high).any():
                raise ValueError("fallback_action must match controller dimensions and bounds")
            self.fallback = action.tolist()

    def after_step(self, step, ctx):
        sample = self.residual(step)
        self.samples += bool(sample["valid"])
        self.skipped += not sample["valid"]
        result = self.detector.update(ctx.t, sample["residual"] if sample["valid"] else None)
        ctx.log(residual_sample=sample, residual_version=RESIDUAL_VERSION, **result,
                center=self.detector.center, spread=self.detector.spread,
                drift=self.detector.drift, threshold=self.detector.threshold, cooldown_steps=self.cooldown)
        if result["detected"]:
            self.alarms += 1
            ctx.alarm("S6 CUSUM closed-loop deviation", score=result["score"],
                      takeover=None if self.takeover == "none" else self.takeover,
                      steps=self.steps, action=self.fallback)

    def end_episode(self, ctx):
        return {"alarms": self.alarms, "valid_samples": self.samples, "skipped_samples": self.skipped}


class FeedbackPoison(Plugin):
    """Shrink reported gripper displacement for ToyOnlinePlannerPolicy only.

    Never changes simulator state, raw observations in the trace, or real weights.
    """
    stage, role, side, requires = "S6", "attack", "model", ("learn",)
    neutral = {"factor": 1.0}
    threat = {"knowledge": "gray_box", "access": "toy online-learning next_obs position feedback",
              "budget": "scale the first two reported position increments by factor in [0,1]",
              "goal": "bias the toy learner's estimated action-to-motion gain", "adaptive": False}

    def __init__(self, config):
        super().__init__(config)
        self.factor = _positive(self.config.get("factor", .3), "factor", allow_zero=True)
        if self.factor > 1:
            raise ValueError("factor must be in [0,1]")
        self.start = _integer(self.config.get("start", 0), "start")
        self.end = self.config.get("end")
        if self.end is not None:
            _integer(self.end, "end", self.start + 1)

    def reset(self, episode, ctx):
        self.changed = 0
        # The current learn probe belongs to the toy online model. Restrict the
        # environment as well, so future online robot models cannot use this card accidentally.
        if ctx.spec.get("action_dim") != 3 or ctx.spec.get("action_meaning") != ["dx", "dy", "grip (>0 closes)"]:
            raise ValueError("FeedbackPoison is restricted to ToyPick online-learning experiments")

    def on_feedback(self, transition, ctx):
        if self.factor == 1 or ctx.t < self.start or (self.end is not None and ctx.t >= self.end):
            return transition
        import numpy as np
        before, after = transition["obs"], transition["next_obs"]
        field = "proprio" if isinstance(before, dict) else None
        a, b = (before[field], after[field]) if field else (before, after)
        clean = np.asarray(b[:2], dtype=float)
        poisoned = np.asarray(a[:2], dtype=float) + self.factor * (clean - np.asarray(a[:2], dtype=float))
        out = copy.deepcopy(transition)
        destination = out["next_obs"][field] if field else out["next_obs"]
        destination[:2] = poisoned.tolist()
        self.changed += 1
        ctx.log(factor=self.factor, clean_position=clean.tolist(), poisoned_position=poisoned.tolist(),
                reported_displacement=(poisoned - np.asarray(a[:2], dtype=float)).tolist(),
                scope="learning_feedback_only")
        return out

    def end_episode(self, ctx):
        return {"poisoned_steps": self.changed}
