"""S5 command faults and a command-space barrier; no simulator truth is used."""
from __future__ import annotations

import math
from collections import deque

from wms.plugins.base import Plugin


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name, allow_zero=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def _restore(values, original):
    import numpy as np
    if isinstance(original, np.ndarray):
        return np.asarray(values, dtype=original.dtype)
    return tuple(values) if isinstance(original, tuple) else list(values)


class _Window(Plugin):
    def __init__(self, config):
        super().__init__(config)
        self.start = _integer(self.config.get("start", 0), "start")
        self.end = self.config.get("end")
        if self.end is not None:
            _integer(self.end, "end", self.start + 1)
        self.dims = self.config.get("dims", 2)
        if self.dims is not None:
            _integer(self.dims, "dims", 1)

    def active(self, ctx):
        return ctx.t >= self.start and (self.end is None or ctx.t < self.end)

    def count(self, action):
        if self.dims is not None and self.dims > len(action):
            raise ValueError(f"dims={self.dims} exceeds action dimension {len(action)}")
        return len(action) if self.dims is None else self.dims


class ActionNoise(_Window):
    """Gaussian noise in controller command units; clipping is a separate defense."""
    stage, role = "S5", "fault"
    neutral = {"std": 0.0}

    def __init__(self, config):
        super().__init__(config)
        self.std = _positive(self.config.get("std", 0.1), "std", allow_zero=True)
        self.clip_to_bounds = self.config.get("clip_to_bounds", False)
        if not isinstance(self.clip_to_bounds, bool):
            raise ValueError("clip_to_bounds must be a boolean")

    def on_action(self, action, ctx):
        if not self.active(ctx) or self.std == 0:
            return action
        n = self.count(action)
        values = list(map(float, action))
        noise = ctx.rng.normal(0, self.std, n)
        out = [v + float(noise[i]) if i < n else v for i, v in enumerate(values)]
        raw = list(out)
        if self.clip_to_bounds:
            low, high = ctx.spec.get("action_low"), ctx.spec.get("action_high")
            if low is None or high is None or len(low) != len(out) or len(high) != len(out):
                raise ValueError("noise clipping requires bounds matching the action dimension")
            out[:n] = [max(float(low[i]), min(float(high[i]), out[i])) for i in range(n)]
        ctx.log(input_action=values, output_action=out, raw_noisy_action=raw,
                clipped_dimensions=sum(a != b for a, b in zip(raw, out)),
                clip_to_bounds=self.clip_to_bounds, std=self.std, dims=n)
        return _restore(out, action)


class ActionDelay(_Window):
    """Delay selected dimensions; warmup repeats the first active command."""
    stage, role = "S5", "fault"
    neutral = {"steps": 0}

    def __init__(self, config):
        super().__init__(config)
        self.steps = _integer(self.config.get("steps", 1), "steps")

    def reset(self, episode, ctx):
        self.queue = deque()
        self.first = None

    def on_action(self, action, ctx):
        if not self.active(ctx):
            self.queue.clear()
            self.first = None
            return action
        if self.steps == 0:
            return action
        n = self.count(action)
        values = list(map(float, action))
        if self.first is None:
            self.first = (ctx.t, values[:n])
        self.queue.append((ctx.t, values[:n]))
        warmup = len(self.queue) <= self.steps
        source_t, delayed = self.first if warmup else self.queue.popleft()
        out = [*delayed, *values[n:]]
        ctx.log(input_action=values, output_action=out, delay_steps=self.steps,
                source_t=source_t, warmup=warmup, dims=n)
        return _restore(out, action)


class ActionBarrier(_Window):
    """Limit magnitude and inter-step delta on selected dimensions, not collision risk.

    The first executed command has no slew reference. After-step updates use the
    actual executed command, including any other plugin's takeover.
    """
    stage, role = "S5", "defense"

    def __init__(self, config):
        super().__init__(config)
        self.limit = _positive(self.config.get("limit", 1.0), "limit")
        delta = self.config.get("max_delta", 0.25)
        self.delta = None if delta is None else _positive(delta, "max_delta")

    def reset(self, episode, ctx):
        self.previous = None
        self.changed_steps = 0

    def on_action(self, action, ctx):
        import numpy as np
        if not self.active(ctx):
            return action
        n = self.count(action)
        values = np.asarray(action, dtype=float)
        low = np.asarray(ctx.spec["action_low"], dtype=float)
        high = np.asarray(ctx.spec["action_high"], dtype=float)
        if low.shape != values.shape or high.shape != values.shape:
            raise ValueError("ActionBarrier requires bounds matching the action shape")
        lo, hi = np.maximum(low[:n], -self.limit), np.minimum(high[:n], self.limit)
        if np.any(lo > hi):
            raise ValueError("barrier limit does not intersect controller bounds")
        bounded = np.clip(values[:n], lo, hi)
        magnitude_limited = bool(np.any(bounded != values[:n]))
        before_slew = bounded.copy()
        out = values.copy()
        if self.previous is not None and self.delta is not None:
            # Magnitude takes priority if the previous action is outside the new limits.
            bounded = np.clip(np.clip(bounded, self.previous[:n] - self.delta,
                                     self.previous[:n] + self.delta), lo, hi)
        out[:n] = bounded
        changed = bool(np.any(out != values))
        reasons = (["magnitude_or_controller_bounds"] if magnitude_limited else [])
        if np.any(bounded != before_slew):
            reasons.append("slew_limit")
        ctx.log(input_action=values.tolist(), output_action=out.tolist(), modified=changed,
                dims=n, limit=self.limit, max_delta=self.delta,
                first_action=self.previous is None, reason_codes=reasons,
                previous_executed_action=None if self.previous is None else self.previous.tolist())
        if changed:
            self.changed_steps += 1
            ctx.intervene("fallback", action=out.tolist(), reason="S5 command barrier: " + ", ".join(reasons))
        return _restore(out.tolist(), action)

    def after_step(self, step, ctx):
        import numpy as np
        if step.get("action") is not None:
            self.previous = np.asarray(step["action"], dtype=float).copy()

    def end_episode(self, ctx):
        return {"modified_steps": self.changed_steps, "scope": "command_limits_only"}
